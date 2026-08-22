from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from dndllm26.api.deps import get_resources
from dndllm26.api.schemas import validate_idempotency_key
from dndllm26.api.sse import SseEmitter, stream_headers
from dndllm26.core.errors import AppError, ModelResponseError
from dndllm26.core.resources import AppResources
from dndllm26.game.openings import complete_opening, fail_opening, prepare_opening
from dndllm26.game.narration import collect_dm_narration, ensure_second_person
from dndllm26.game.prompts import build_campaign_intro_messages, protagonist_name
from dndllm26.game.world import analyse_world_update

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/campaigns/{campaign_id}/opening", tags=["campaigns"])


def _idempotency_key(value: str | None) -> str:
    if value is None:
        return uuid4().hex
    try:
        return validate_idempotency_key(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _error(exc: BaseException) -> tuple[str, str, str | None, bool]:
    if isinstance(exc, AppError):
        return exc.code, exc.message, exc.detail, exc.retryable
    return (
        "unexpected_error",
        "Opening generation failed unexpectedly and can be retried safely.",
        None,
        True,
    )


@router.post("/stream")
async def stream_opening(
    campaign_id: int,
    request: Request,
    idempotency_header: str | None = Header(default=None, alias="Idempotency-Key"),
    resources: AppResources = Depends(get_resources),
) -> StreamingResponse:
    key = _idempotency_key(idempotency_header)
    with Session(resources.engine) as session:
        preparation = prepare_opening(
            session,
            campaign_id=campaign_id,
            idempotency_key=key,
        )
    emitter = SseEmitter(request.state.request_id, campaign_id)

    async def events():
        yield emitter.emit(
            "stream_started",
            {
                "operation": "campaign_opening",
                "operation_id": preparation.opening_id,
                "idempotency_key": preparation.idempotency_key,
                "replayed": preparation.replay_status is not None,
            },
        )
        if preparation.replay_status == "complete":
            yield emitter.emit("replay", {"status": "complete"})
            yield emitter.emit("done", {"status": "complete", "authoritative_refresh": True})
            return
        try:
            lore = []
            if preparation.context.lore_documents:
                try:
                    lore = await resources.rag.search(
                        f"{preparation.context.title}\n{preparation.context.setting}",
                        limit=4,
                        documents=preparation.context.lore_documents,
                    )
                except AppError:
                    logger.warning("Opening lore retrieval failed; continuing without RAG")
            system, prompt = build_campaign_intro_messages(preparation.context, lore)
            yield emitter.emit("phase", {"status": "dm_streaming"})
            narration = await collect_dm_narration(
                resources.ollama,
                system,
                prompt,
                narration_style=resources.narration_style,
            )
            if not narration:
                raise ModelResponseError("The narrator returned no playable opening text.")
            hero_name = protagonist_name(preparation.context)
            narration = await ensure_second_person(
                resources.ollama,
                narration,
                hero_name,
            )
            yield emitter.emit("narration", {"content": narration})
            yield emitter.emit("phase", {"status": "utility_analyzing"})
            world_update = await analyse_world_update(
                resources.ollama,
                narration,
                preparation.context.world,
                action="Begin the campaign",
                bootstrap=True,
                protagonist=hero_name,
                party_names=tuple(character.name for character in preparation.context.characters),
            )
            with Session(resources.engine) as session:
                complete_opening(
                    session,
                    opening_id=preparation.opening_id,
                    narration=narration,
                    world_update=world_update,
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
            yield emitter.emit("done", {"status": "complete", "authoritative_refresh": True})
        except asyncio.CancelledError:
            with Session(resources.engine) as session:
                fail_opening(
                    session,
                    opening_id=preparation.opening_id,
                    code="interrupted",
                    message="Opening generation was interrupted and can be retried safely.",
                )
            raise
        except Exception as exc:
            logger.exception("Campaign opening stream failed", extra={"campaign_id": campaign_id})
            code, message, detail, retryable = _error(exc)
            stored = f"{message} {detail}" if detail else message
            with Session(resources.engine) as session:
                fail_opening(
                    session,
                    opening_id=preparation.opening_id,
                    code=code,
                    message=stored,
                )
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
        headers=stream_headers(request.state.request_id),
    )
