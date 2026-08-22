from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from dndllm26.api.deps import get_resources
from dndllm26.api.schemas import PlayerAction, validate_idempotency_key
from dndllm26.api.sse import SseEmitter, stream_headers
from dndllm26.core.errors import AppError, ModelResponseError
from dndllm26.core.resources import AppResources
from dndllm26.db.models import DiceRoll
from dndllm26.game.combat import (
    apply_combat_roll,
    assess_combat,
    active_encounter,
    encounter_detail,
    normalise_npc_identity,
    start_encounter,
)
from dndllm26.game.campaigns import add_event
from dndllm26.game.play import (
    create_pending_roll,
    mark_request_failed,
    persist_narrative,
    prepare_action,
    prepare_roll_resolution,
)
from dndllm26.game.narration import ensure_second_person
from dndllm26.game.prompts import (
    build_dm_prompt,
    build_combat_transition_retry_messages,
    build_lore_query,
    build_roll_resolution_prompt,
    combat_transition_violations,
    combat_narration_is_grounded,
    safe_combat_transition,
    safe_second_person_narration,
)
from dndllm26.game.rules import decide_roll
from dndllm26.game.text import MAX_DM_CHARS, trim_dm_text
from dndllm26.game.world import analyse_world_update

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/campaigns/{campaign_id}", tags=["play"])


def _idempotency_key(value: str | None) -> str:
    if value is None:
        return uuid4().hex
    try:
        return validate_idempotency_key(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _error_payload(exc: BaseException) -> tuple[str, str, str | None, bool]:
    if isinstance(exc, AppError):
        return exc.code, exc.message, exc.detail, exc.retryable
    return (
        "unexpected_error",
        "The local operation failed unexpectedly. It can be retried without duplicating the action.",
        None,
        True,
    )


def _mark_failure(
    resources: AppResources,
    request_id: int,
    exc: BaseException,
    *,
    interrupted: bool = False,
) -> tuple[str, str, str | None, bool]:
    code, message, detail, retryable = _error_payload(exc)
    stored_message = f"{message} {detail}"[:500] if detail else message
    with Session(resources.engine) as session:
        mark_request_failed(
            session,
            request_id=request_id,
            code=code,
            message=stored_message,
            retryable=retryable or interrupted,
            interrupted=interrupted,
        )
    return code, message, detail, retryable or interrupted


@router.post("/actions/stream")
async def stream_action(
    campaign_id: int,
    payload: PlayerAction,
    request: Request,
    idempotency_header: str | None = Header(default=None, alias="Idempotency-Key"),
    resources: AppResources = Depends(get_resources),
) -> StreamingResponse:
    key = _idempotency_key(idempotency_header)
    if len(payload.content) > resources.settings.request_max_chars:
        raise HTTPException(
            status_code=422,
            detail=f"Action exceeds the configured {resources.settings.request_max_chars} character limit.",
        )
    with Session(resources.engine) as session:
        preparation = prepare_action(
            session,
            campaign_id=campaign_id,
            idempotency_key=key,
            action=payload.content,
        )

    correlation_id = request.state.request_id
    emitter = SseEmitter(correlation_id, campaign_id)

    async def events():
        yield emitter.emit(
            "stream_started",
            {
                "operation": "action",
                "operation_id": preparation.request_id,
                "idempotency_key": key,
                "replayed": preparation.replay_status is not None,
            },
        )
        if preparation.replay_status == "complete":
            yield emitter.emit("replay", {"status": "complete"})
            yield emitter.emit("done", {"status": "complete", "authoritative_refresh": True})
            return
        if preparation.replay_status == "roll_required" and preparation.pending_roll:
            yield emitter.emit("roll_required", asdict(preparation.pending_roll))
            yield emitter.emit("done", {"status": "roll_required", "authoritative_refresh": True})
            return

        try:
            yield emitter.emit("phase", {"status": "checking_action"})
            decision = await decide_roll(
                resources.ollama,
                preparation.context,
                preparation.action_text,
            )
            combat_context = "\n".join(
                (
                    preparation.context.world.scene_summary,
                    *(turn.content for turn in preparation.context.turns[-2:]),
                )
            )
            combat = assess_combat(
                preparation.action_text,
                model_enemy_keys=decision.encounter_enemies,
                model_names=decision.encounter_names,
                known_npcs=preparation.context.world.npcs,
                context_text=combat_context,
            )
            hero_name = next(
                (
                    character.name
                    for character in preparation.context.characters
                    if character.role == "protagonist"
                ),
                "The hero",
            )

            async def validate_transition(source: str) -> str:
                violations = combat_transition_violations(source)
                if not violations:
                    return source
                retry_system, retry_prompt = build_combat_transition_retry_messages(
                    source,
                    violations,
                    hero_name=hero_name,
                    opponent_names=combat.display_names,
                )
                retry_parts: list[str] = []
                retry_count = 0
                try:
                    async for retry_token in resources.ollama.stream_dm(
                        retry_system,
                        retry_prompt,
                        narration_style="focused",
                    ):
                        if retry_count >= MAX_DM_CHARS:
                            break
                        fragment = retry_token[: MAX_DM_CHARS - retry_count]
                        retry_parts.append(fragment)
                        retry_count += len(fragment)
                except AppError:
                    logger.warning(
                        "Combat transition rewrite failed; applying deterministic sanitizer",
                        exc_info=True,
                        extra={"campaign_id": campaign_id},
                    )
                retry_text = trim_dm_text("".join(retry_parts))
                if retry_text and not combat_transition_violations(retry_text):
                    return retry_text
                return safe_combat_transition(
                    retry_text or source,
                    hero_name=hero_name,
                    opponent_names=combat.display_names,
                )

            if decision.requires_roll and not combat.starts:
                with Session(resources.engine) as session:
                    pending = create_pending_roll(
                        session,
                        request_id=preparation.request_id,
                        decision=decision,
                    )
                if pending.narration:
                    yield emitter.emit("narration", {"content": pending.narration})
                yield emitter.emit("roll_required", asdict(pending))
                yield emitter.emit(
                    "done", {"status": "roll_required", "authoritative_refresh": True}
                )
                return

            lore = []
            if preparation.context.lore_documents:
                try:
                    lore = await resources.rag.search(
                        build_lore_query(
                            preparation.context,
                            preparation.action_text,
                            decision.lore_query,
                        ),
                        limit=4,
                        documents=preparation.context.lore_documents,
                    )
                except AppError:
                    logger.warning(
                        "Lore retrieval failed; continuing without RAG",
                        exc_info=True,
                        extra={
                            "campaign_id": campaign_id,
                            "action_request_id": preparation.request_id,
                        },
                    )
            system, prompt = build_dm_prompt(
                preparation.context,
                preparation.action_text,
                lore,
                resources.narration_style,
                combat_transition=combat.starts,
                combatants=(hero_name, *combat.display_names),
            )
            yield emitter.emit("phase", {"status": "dm_streaming"})
            parts: list[str] = []
            character_count = 0
            async for token in resources.ollama.stream_dm(
                system, prompt, narration_style=resources.narration_style
            ):
                if character_count >= MAX_DM_CHARS:
                    break
                fragment = token[: MAX_DM_CHARS - character_count]
                if fragment:
                    parts.append(fragment)
                    character_count += len(fragment)
                    if not combat.starts:
                        yield emitter.emit("narration_delta", {"content": fragment})
            dm_text = trim_dm_text("".join(parts))
            if not dm_text:
                raise ModelResponseError("The narrator returned no playable text.")
            if combat.starts:
                dm_text = await validate_transition(dm_text)
                dm_text = await ensure_second_person(resources.ollama, dm_text, hero_name)
                yield emitter.emit("narration", {"content": dm_text})
            if not combat.starts:
                combat = assess_combat(
                    preparation.action_text,
                    dm_text,
                    model_enemy_keys=decision.encounter_enemies,
                    model_names=decision.encounter_names,
                    known_npcs=preparation.context.world.npcs,
                    context_text=combat_context,
                )
                corrected = await validate_transition(dm_text) if combat.starts else dm_text
                corrected = await ensure_second_person(resources.ollama, corrected, hero_name)
                if corrected != dm_text:
                    dm_text = corrected
                    yield emitter.emit("narration_replace", {"content": dm_text})

            yield emitter.emit("phase", {"status": "utility_analyzing"})
            world_update = await analyse_world_update(
                resources.ollama,
                dm_text,
                preparation.context.world,
                action=preparation.action_text,
                protagonist=hero_name,
                party_names=tuple(character.name for character in preparation.context.characters),
            )
            if combat.starts and combat.display_names:
                participants = list(world_update.npcs)
                known_keys = {normalise_npc_identity(value).casefold() for value in participants}
                for name in combat.display_names:
                    identity_key = normalise_npc_identity(name).casefold()
                    if identity_key and identity_key not in known_keys and name != "Opponent":
                        participants.append(name)
                        known_keys.add(identity_key)
                world_update = replace(world_update, npcs=tuple(participants[:20]))
            opening_events: list[str] = []
            with Session(resources.engine) as session:
                persist_narrative(
                    session,
                    request_id=preparation.request_id,
                    dm_text=dm_text,
                    world_update=world_update,
                )
                if combat.starts and not active_encounter(session, campaign_id):
                    start_encounter(
                        session,
                        campaign_id,
                        combat.enemy_keys,
                        combat.display_names,
                        opening_events,
                    )
                    add_event(
                        session,
                        campaign_id,
                        "combat_classified",
                        {
                            "trigger": combat.trigger,
                            "evidence": combat.evidence,
                            "enemy_templates": list(combat.enemy_keys),
                            "display_names": list(combat.display_names),
                            "duel": combat.duel,
                        },
                        action_request_id=preparation.request_id,
                    )
                session.commit()
                logger.info(
                    "Action finalization committed",
                    extra={
                        "campaign_id": campaign_id,
                        "action_request_id": preparation.request_id,
                        "operation_stage": "finalized",
                    },
                )
                started_encounter = (
                    encounter_detail(session, campaign_id) if combat.starts else None
                )
            yield emitter.emit(
                "choices_updated",
                {
                    "choices": list(world_update.choices),
                    "location": world_update.location,
                    "objective": world_update.objective,
                    "summary": world_update.summary,
                },
            )
            if started_encounter:
                yield emitter.emit("encounter_started", started_encounter)
            if opening_events:
                yield emitter.emit("combat_log", {"events": opening_events})
            yield emitter.emit("done", {"status": "complete", "authoritative_refresh": True})
        except asyncio.CancelledError as exc:
            _mark_failure(resources, preparation.request_id, exc, interrupted=True)
            raise
        except Exception as exc:
            logger.exception(
                "Action stream failed",
                extra={
                    "campaign_id": campaign_id,
                    "action_request_id": preparation.request_id,
                },
            )
            code, message, detail, retryable = _mark_failure(resources, preparation.request_id, exc)
            yield emitter.emit(
                "error",
                {
                    "code": code,
                    "message": message,
                    "detail": detail,
                    "retryable": retryable,
                },
            )
            yield emitter.emit("done", {"status": "failed", "authoritative_refresh": True})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers=stream_headers(correlation_id),
    )


@router.post("/rolls/{pending_roll_id}/resolve/stream")
async def resolve_roll_stream(
    campaign_id: int,
    pending_roll_id: int,
    request: Request,
    idempotency_header: str | None = Header(default=None, alias="Idempotency-Key"),
    resources: AppResources = Depends(get_resources),
) -> StreamingResponse:
    key = _idempotency_key(idempotency_header)
    with Session(resources.engine) as session:
        preparation = prepare_roll_resolution(
            session,
            campaign_id=campaign_id,
            pending_roll_id=pending_roll_id,
            idempotency_key=key,
        )

    correlation_id = request.state.request_id
    emitter = SseEmitter(correlation_id, campaign_id)

    async def events():
        yield emitter.emit(
            "stream_started",
            {
                "operation": "roll_resolution",
                "operation_id": preparation.request_id,
                "idempotency_key": key,
                "replayed": preparation.replay_status is not None,
            },
        )
        yield emitter.emit("roll_result", asdict(preparation.dice_roll))
        if preparation.replay_status == "complete":
            yield emitter.emit("replay", {"status": "complete"})
            yield emitter.emit("done", {"status": "complete", "authoritative_refresh": True})
            return

        try:
            combat_events: list[str] = []
            with Session(resources.engine) as session:
                stored_roll = session.get(DiceRoll, preparation.dice_roll.id)
                if stored_roll:
                    combat_events = apply_combat_roll(session, pending_roll_id, stored_roll)
                updated_encounter = (
                    encounter_detail(session, campaign_id) if combat_events else None
                )
            lore = []
            if preparation.context.lore_documents:
                try:
                    lore = await resources.rag.search(
                        build_lore_query(
                            preparation.context,
                            preparation.pending_roll.action_text,
                        ),
                        limit=4,
                        documents=preparation.context.lore_documents,
                    )
                except AppError:
                    logger.warning(
                        "Lore retrieval failed during roll resolution; continuing without RAG",
                        exc_info=True,
                        extra={
                            "campaign_id": campaign_id,
                            "action_request_id": preparation.request_id,
                        },
                    )
            hero_name = next(
                (
                    character.name
                    for character in preparation.context.characters
                    if character.role == "protagonist"
                ),
                "The hero",
            )
            narrative_combat_events = tuple(
                safe_second_person_narration(event, hero_name) for event in combat_events
            )
            system, prompt = build_roll_resolution_prompt(
                preparation.context,
                preparation.pending_roll,
                preparation.dice_roll,
                lore,
                narrative_combat_events,
                resources.narration_style,
            )
            yield emitter.emit("phase", {"status": "dm_streaming"})
            parts: list[str] = []
            character_count = 0
            async for token in resources.ollama.stream_dm(
                system, prompt, narration_style=resources.narration_style
            ):
                if character_count >= MAX_DM_CHARS:
                    break
                fragment = token[: MAX_DM_CHARS - character_count]
                if fragment:
                    parts.append(fragment)
                    character_count += len(fragment)
                    if not combat_events:
                        yield emitter.emit("narration_delta", {"content": fragment})
            dm_text = trim_dm_text("".join(parts))
            if not dm_text:
                raise ModelResponseError("The narrator returned no playable text.")
            corrected = await ensure_second_person(resources.ollama, dm_text, hero_name)
            identity_changed = corrected != dm_text
            dm_text = corrected
            if combat_events and not combat_narration_is_grounded(
                preparation.context,
                dm_text,
                narrative_combat_events,
            ):
                raise ModelResponseError(
                    "The narrator did not preserve the authoritative combat outcome.",
                    detail="The saved dice result and combat effects are unchanged; retry narration safely.",
                )
            elif combat_events:
                yield emitter.emit("narration", {"content": dm_text})
            elif identity_changed:
                yield emitter.emit("narration_replace", {"content": dm_text})

            yield emitter.emit("phase", {"status": "utility_analyzing"})
            world_update = await analyse_world_update(
                resources.ollama,
                dm_text,
                preparation.context.world,
                action=preparation.pending_roll.action_text,
                authoritative_events=narrative_combat_events,
                protagonist=hero_name,
                party_names=tuple(character.name for character in preparation.context.characters),
            )
            with Session(resources.engine) as session:
                persist_narrative(
                    session,
                    request_id=preparation.request_id,
                    dm_text=dm_text,
                    world_update=world_update,
                )
                session.commit()
                logger.info(
                    "Roll narration finalization committed",
                    extra={
                        "campaign_id": campaign_id,
                        "action_request_id": preparation.request_id,
                        "operation_stage": "finalized",
                    },
                )
            yield emitter.emit(
                "choices_updated",
                {
                    "choices": list(world_update.choices),
                    "location": world_update.location,
                    "objective": world_update.objective,
                    "summary": world_update.summary,
                },
            )
            if combat_events:
                yield emitter.emit("combat_log", {"events": combat_events})
            if updated_encounter:
                yield emitter.emit("encounter_updated", updated_encounter)
                if updated_encounter.get("status") == "defeat":
                    yield emitter.emit(
                        "party_recovered",
                        {"message": "The party survived with a lasting setback."},
                    )
            yield emitter.emit("done", {"status": "complete", "authoritative_refresh": True})
        except asyncio.CancelledError as exc:
            _mark_failure(resources, preparation.request_id, exc, interrupted=True)
            raise
        except Exception as exc:
            logger.exception(
                "Roll resolution stream failed",
                extra={
                    "campaign_id": campaign_id,
                    "action_request_id": preparation.request_id,
                },
            )
            code, message, detail, retryable = _mark_failure(resources, preparation.request_id, exc)
            yield emitter.emit(
                "error",
                {
                    "code": code,
                    "message": message,
                    "detail": detail,
                    "retryable": retryable,
                    "dice_result_saved": True,
                },
            )
            yield emitter.emit("done", {"status": "failed", "authoritative_refresh": True})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers=stream_headers(correlation_id),
    )
