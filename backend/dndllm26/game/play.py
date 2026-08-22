from __future__ import annotations

from datetime import timezone
import json
import logging
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from dndllm26.core.errors import ConflictError, NotFoundError, OperationInProgressError
from dndllm26.db.models import (
    ActionRequest,
    Campaign,
    CampaignOpening,
    Character,
    DiceRoll,
    PendingRoll,
    Turn,
    WorldState,
    now_utc,
)
from dndllm26.game.campaigns import (
    add_event,
    build_campaign_context,
    choices_from_state,
    get_or_create_world_state,
    get_pending_roll,
    pending_roll_snapshot,
    touch_campaign,
)
from dndllm26.game.dice import RandomSource, outcome_for, roll_formula
from dndllm26.game.catalog import SKILL_ABILITIES, ability_modifier, proficiency_bonus
from dndllm26.game.schemas import RollDecisionOutput
from dndllm26.game.types import (
    ActionPreparation,
    DiceRollSnapshot,
    RollPreparation,
    WorldSnapshot,
    WorldUpdate,
    PendingRollSnapshot,
)

logger = logging.getLogger(__name__)


def _request_by_key(
    session: Session,
    campaign_id: int,
    idempotency_key: str,
    operation: str,
) -> ActionRequest | None:
    return session.exec(
        select(ActionRequest)
        .where(ActionRequest.campaign_id == campaign_id)
        .where(ActionRequest.idempotency_key == idempotency_key)
        .where(ActionRequest.operation == operation)
    ).first()


def _request_age_seconds(request: ActionRequest) -> float:
    updated = request.updated_at
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    return max(0.0, (now_utc() - updated).total_seconds())


def dice_roll_snapshot(roll: DiceRoll) -> DiceRollSnapshot:
    if roll.id is None:
        raise ValueError("Dice roll has not been persisted")
    try:
        raw_rolls = json.loads(roll.rolls_json)
    except json.JSONDecodeError:
        raw_rolls = []
    rolls = tuple(int(value) for value in raw_rolls if isinstance(value, int))
    return DiceRollSnapshot(
        id=roll.id,
        pending_roll_id=roll.pending_roll_id,
        formula=roll.formula,
        rolls=rolls,
        modifier=roll.modifier,
        total=roll.total,
        dc=roll.dc,
        outcome=roll.outcome,
        reason=roll.reason,
    )


def recover_stale_operations(engine: Engine) -> int:
    """Mark every operation left by a previous process as retryable.

    Startup owns the database exclusively, so no processing row can still have a
    live producer regardless of its age.
    """

    with Session(engine) as session:
        rows = session.exec(select(ActionRequest).where(ActionRequest.status == "processing")).all()
        for request in rows:
            request.status = "interrupted"
            request.error_code = "process_restarted"
            request.error_message = "The operation was interrupted and can be retried safely."
            request.retryable = True
            request.updated_at = now_utc()
            session.add(request)
        if rows:
            session.commit()
        return len(rows)


def prepare_action(
    session: Session,
    *,
    campaign_id: int,
    idempotency_key: str,
    action: str,
    prompt_history_limit: int = 14,
    stale_seconds: int = 900,
) -> ActionPreparation:
    campaign = session.get(Campaign, campaign_id)
    if not campaign:
        raise NotFoundError("Campaign not found.")
    opening = session.exec(
        select(CampaignOpening).where(CampaignOpening.campaign_id == campaign_id)
    ).first()
    if opening and opening.status != "complete":
        raise ConflictError("Generate the campaign opening before submitting an action.")

    existing = _request_by_key(session, campaign_id, idempotency_key, "action")
    if existing:
        if existing.action_text != action:
            raise ConflictError("This idempotency key was already used for a different action.")
        if existing.id is None:
            raise RuntimeError("Action request is missing its id")
        pending = (
            session.get(PendingRoll, existing.pending_roll_id) if existing.pending_roll_id else None
        )
        if existing.status in {"complete", "roll_required"}:
            context = build_campaign_context(
                session,
                campaign_id,
                turn_limit=prompt_history_limit,
                exclude_turn_id=existing.player_turn_id,
            )
            return ActionPreparation(
                request_id=existing.id,
                campaign_id=campaign_id,
                idempotency_key=idempotency_key,
                action_text=action,
                context=context,
                replay_status=existing.status,
                pending_roll=pending_roll_snapshot(pending) if pending else None,
            )
        if existing.status == "processing":
            if _request_age_seconds(existing) < stale_seconds:
                raise OperationInProgressError("This action is already being processed.")
            # A stale in-process request can be resumed safely because the player turn
            # is linked to this request and will not be inserted again.
        elif existing.status not in {"failed", "interrupted"} or not existing.retryable:
            raise ConflictError("This action cannot be retried with the same idempotency key.")
        existing.status = "processing"
        existing.error_code = None
        existing.error_message = None
        existing.retryable = False
        existing.updated_at = now_utc()
        session.add(existing)
        session.commit()
        context = build_campaign_context(
            session,
            campaign_id,
            turn_limit=prompt_history_limit,
            exclude_turn_id=existing.player_turn_id,
        )
        return ActionPreparation(
            request_id=existing.id,
            campaign_id=campaign_id,
            idempotency_key=idempotency_key,
            action_text=action,
            context=context,
        )

    if get_pending_roll(session, campaign_id):
        raise ConflictError("Resolve the pending dice roll before submitting another action.")
    recoverable = session.exec(
        select(ActionRequest)
        .where(ActionRequest.campaign_id == campaign_id)
        .where(ActionRequest.status.in_(["failed", "interrupted"]))
        .where(ActionRequest.retryable.is_(True))
        .order_by(ActionRequest.updated_at.desc(), ActionRequest.id.desc())
    ).first()
    if recoverable:
        raise ConflictError(
            "Retry the interrupted campaign operation before submitting another action."
        )

    context = build_campaign_context(
        session,
        campaign_id,
        turn_limit=prompt_history_limit,
    )
    request = ActionRequest(
        campaign_id=campaign_id,
        idempotency_key=idempotency_key,
        operation="action",
        status="processing",
        action_text=action,
        request_payload_json=json.dumps({"action": action}, sort_keys=True, separators=(",", ":")),
    )
    try:
        session.add(request)
        session.flush()
        if request.id is None:
            raise RuntimeError("Action request id was not generated")
        player_turn = Turn(
            campaign_id=campaign_id,
            action_request_id=request.id,
            speaker="Player",
            content=action,
        )
        session.add(player_turn)
        session.flush()
        request.player_turn_id = player_turn.id
        request.updated_at = now_utc()
        session.add(request)
        add_event(
            session,
            campaign_id,
            "player_action",
            {"content": action},
            action_request_id=request.id,
        )
        touch_campaign(session, campaign_id)
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise OperationInProgressError(
            "Another campaign operation is already in progress."
        ) from exc
    return ActionPreparation(
        request_id=request.id,
        campaign_id=campaign_id,
        idempotency_key=idempotency_key,
        action_text=action,
        context=context,
    )


def create_pending_roll(
    session: Session,
    *,
    request_id: int,
    decision: RollDecisionOutput,
) -> PendingRollSnapshot:
    request = session.get(ActionRequest, request_id)
    if not request:
        raise NotFoundError("Action request not found.")
    if request.pending_roll_id:
        existing = session.get(PendingRoll, request.pending_roll_id)
        if existing:
            return pending_roll_snapshot(existing)
    if request.status != "processing":
        raise ConflictError("Action request is no longer awaiting a roll decision.")
    active = get_pending_roll(session, request.campaign_id)
    if active:
        raise ConflictError("A dice roll is already pending for this campaign.")

    protagonist = session.exec(
        select(Character)
        .where(Character.campaign_id == request.campaign_id)
        .where(Character.role == "protagonist")
    ).first()
    ability_name = decision.ability.casefold().split()[0]
    skill_name = decision.skill.title() if decision.skill else None
    if skill_name in SKILL_ABILITIES:
        ability_name = SKILL_ABILITIES[skill_name]
    modifier = 0
    if protagonist and hasattr(protagonist, ability_name):
        modifier = ability_modifier(int(getattr(protagonist, ability_name)))
        if skill_name and skill_name in json.loads(protagonist.skills_json):
            modifier += proficiency_bonus(protagonist.level)
    formula = f"1d20{modifier:+d}" if modifier else "1d20"
    pending = PendingRoll(
        campaign_id=request.campaign_id,
        action_request_id=request.id,
        action_text=request.action_text,
        formula=formula,
        ability=decision.ability,
        skill=decision.skill,
        dc=decision.dc,
        reason=decision.reason or "Resolve the uncertain outcome.",
        narration=decision.narration,
        status="pending",
    )
    try:
        session.add(pending)
        session.flush()
        request.pending_roll_id = pending.id
        request.status = "roll_required"
        request.updated_at = now_utc()
        session.add(request)
        add_event(
            session,
            request.campaign_id,
            "roll_required",
            {
                "pending_roll_id": pending.id,
                "formula": pending.formula,
                "ability": pending.ability,
                "skill": pending.skill,
                "dc": pending.dc,
                "reason": pending.reason,
            },
            action_request_id=request.id,
        )
        session.commit()
        session.refresh(pending)
    except IntegrityError as exc:
        session.rollback()
        raise ConflictError("A dice roll is already pending for this campaign.") from exc
    return pending_roll_snapshot(pending)


def mark_request_failed(
    session: Session,
    *,
    request_id: int,
    code: str,
    message: str,
    retryable: bool,
    interrupted: bool = False,
) -> None:
    request = session.get(ActionRequest, request_id)
    if not request or request.status == "complete":
        return
    request.status = "interrupted" if interrupted else "failed"
    request.error_code = code[:80]
    request.error_message = message[:500]
    request.retryable = retryable
    request.updated_at = now_utc()
    session.add(request)
    add_event(
        session,
        request.campaign_id,
        "operation_interrupted" if interrupted else "operation_failed",
        {"code": code, "retryable": retryable},
        action_request_id=request.id,
    )
    session.commit()


def _world_snapshot(state: WorldState) -> WorldSnapshot:
    return WorldSnapshot(
        current_location=state.current_location,
        active_objective=state.active_objective,
        scene_summary=state.scene_summary,
        choices=tuple(choices_from_state(state)),
        facts=tuple(json.loads(state.facts_json)),
        npcs=tuple(json.loads(state.npcs_json)),
    )


def current_world_snapshot(session: Session, campaign_id: int) -> WorldSnapshot:
    return _world_snapshot(get_or_create_world_state(session, campaign_id))


def persist_narrative(
    session: Session,
    *,
    request_id: int,
    dm_text: str,
    world_update: WorldUpdate,
) -> WorldUpdate:
    request = session.get(ActionRequest, request_id)
    if not request:
        raise NotFoundError("Action request not found.")
    if request.status == "complete":
        state = get_or_create_world_state(session, request.campaign_id)
        return WorldUpdate(
            location=state.current_location,
            objective=state.active_objective,
            summary=state.scene_summary,
            choices=tuple(choices_from_state(state)),
            facts=tuple(json.loads(state.facts_json)),
            npcs=tuple(json.loads(state.npcs_json)),
        )
    if request.status != "processing":
        raise ConflictError("Operation is no longer in a state that can be completed.")

    dm_turn = Turn(
        campaign_id=request.campaign_id,
        action_request_id=request.id,
        speaker="DM",
        content=dm_text,
    )
    session.add(dm_turn)
    session.flush()
    state = get_or_create_world_state(session, request.campaign_id)
    state.current_location = world_update.location
    state.active_objective = world_update.objective
    state.scene_summary = world_update.summary
    state.choices_json = json.dumps(list(world_update.choices), ensure_ascii=False)
    state.facts_json = json.dumps(list(world_update.facts), ensure_ascii=False)
    state.npcs_json = json.dumps(list(world_update.npcs), ensure_ascii=False)
    state.updated_at = now_utc()
    session.add(state)

    request.dm_turn_id = dm_turn.id
    request.status = "complete"
    request.error_code = None
    request.error_message = None
    request.retryable = False
    request.updated_at = now_utc()
    request.completed_at = now_utc()
    session.add(request)

    if request.operation == "roll_resolution" and request.pending_roll_id:
        pending = session.get(PendingRoll, request.pending_roll_id)
        if pending and pending.action_request_id:
            original = session.get(ActionRequest, pending.action_request_id)
            if original:
                original.status = "complete"
                original.dm_turn_id = dm_turn.id
                original.completed_at = now_utc()
                original.updated_at = now_utc()
                session.add(original)

    add_event(
        session,
        request.campaign_id,
        "dm_response",
        {"content_length": len(dm_text)},
        action_request_id=request.id,
    )
    add_event(
        session,
        request.campaign_id,
        "choices_updated",
        {
            "choices": list(world_update.choices),
            "location": world_update.location,
            "objective": world_update.objective,
            "summary": world_update.summary,
        },
        action_request_id=request.id,
    )
    touch_campaign(session, request.campaign_id)
    return world_update


def prepare_roll_resolution(
    session: Session,
    *,
    campaign_id: int,
    pending_roll_id: int,
    idempotency_key: str,
    prompt_history_limit: int = 14,
    stale_seconds: int = 900,
    random_source: RandomSource | None = None,
) -> RollPreparation:
    campaign = session.get(Campaign, campaign_id)
    if not campaign:
        raise NotFoundError("Campaign not found.")
    pending = session.get(PendingRoll, pending_roll_id)
    if not pending or pending.campaign_id != campaign_id:
        raise NotFoundError("Pending roll not found.")

    existing = _request_by_key(session, campaign_id, idempotency_key, "roll_resolution")
    if existing:
        if existing.pending_roll_id != pending_roll_id:
            raise ConflictError("This idempotency key was already used for a different roll.")
        if existing.id is None:
            raise RuntimeError("Roll request is missing its id")
        roll = session.get(DiceRoll, existing.dice_roll_id) if existing.dice_roll_id else None
        if not roll:
            roll = session.exec(
                select(DiceRoll).where(DiceRoll.pending_roll_id == pending_roll_id)
            ).first()
        if existing.status == "complete" and roll:
            context = build_campaign_context(
                session,
                campaign_id,
                turn_limit=prompt_history_limit,
            )
            return RollPreparation(
                request_id=existing.id,
                campaign_id=campaign_id,
                idempotency_key=idempotency_key,
                pending_roll=pending_roll_snapshot(pending),
                dice_roll=dice_roll_snapshot(roll),
                context=context,
                replay_status="complete",
            )
        if existing.status == "processing":
            if _request_age_seconds(existing) < stale_seconds:
                raise OperationInProgressError("This dice roll is already being resolved.")
            # Resume stale processing with the stored dice result; never reroll.
        elif existing.status not in {"failed", "interrupted"} or not existing.retryable:
            raise ConflictError("This dice resolution cannot be retried with the same key.")
        if not roll:
            raise ConflictError("The stored dice result is unavailable; no reroll was performed.")
        existing.status = "processing"
        existing.error_code = None
        existing.error_message = None
        existing.retryable = False
        existing.updated_at = now_utc()
        session.add(existing)
        session.commit()
        context = build_campaign_context(
            session,
            campaign_id,
            turn_limit=prompt_history_limit,
        )
        return RollPreparation(
            request_id=existing.id,
            campaign_id=campaign_id,
            idempotency_key=idempotency_key,
            pending_roll=pending_roll_snapshot(pending),
            dice_roll=dice_roll_snapshot(roll),
            context=context,
        )

    prior_resolution = session.exec(
        select(ActionRequest)
        .where(ActionRequest.campaign_id == campaign_id)
        .where(ActionRequest.operation == "roll_resolution")
        .where(ActionRequest.pending_roll_id == pending_roll_id)
        .order_by(ActionRequest.created_at.desc())
    ).first()
    if prior_resolution:
        if prior_resolution.status == "processing":
            raise OperationInProgressError("This dice roll is already being resolved.")
        prior_roll = (
            session.get(DiceRoll, prior_resolution.dice_roll_id)
            if prior_resolution.dice_roll_id
            else session.exec(
                select(DiceRoll).where(DiceRoll.pending_roll_id == pending_roll_id)
            ).first()
        )
        if prior_resolution.status == "complete" and prior_roll and prior_resolution.id is not None:
            context = build_campaign_context(
                session,
                campaign_id,
                turn_limit=prompt_history_limit,
            )
            return RollPreparation(
                request_id=prior_resolution.id,
                campaign_id=campaign_id,
                idempotency_key=idempotency_key,
                pending_roll=pending_roll_snapshot(pending),
                dice_roll=dice_roll_snapshot(prior_roll),
                context=context,
                replay_status="complete",
            )

    request = ActionRequest(
        campaign_id=campaign_id,
        idempotency_key=idempotency_key,
        operation="roll_resolution",
        status="processing",
        action_text=pending.action_text,
        request_payload_json=json.dumps(
            {"pending_roll_id": pending_roll_id}, sort_keys=True, separators=(",", ":")
        ),
        pending_roll_id=pending.id,
    )
    try:
        session.add(request)
        session.flush()
        if request.id is None:
            raise RuntimeError("Roll request id was not generated")

        roll = session.exec(
            select(DiceRoll).where(DiceRoll.pending_roll_id == pending_roll_id)
        ).first()
        if roll is None:
            if pending.status != "pending":
                raise ConflictError("This roll has already been claimed or resolved.")
            result = session.exec(
                update(PendingRoll)
                .where(PendingRoll.id == pending_roll_id)
                .where(PendingRoll.status == "pending")
                .values(status="resolving")
            )
            if result.rowcount != 1:
                raise ConflictError("This roll has already been claimed by another request.")
            dice = roll_formula(pending.formula, random_source=random_source)
            outcome = outcome_for(dice.total, pending.dc)
            if dice.formula.startswith("1d20") and dice.rolls:
                if dice.rolls[0] == 20:
                    outcome = "success"
                elif dice.rolls[0] == 1:
                    outcome = "failure"
            roll = DiceRoll(
                campaign_id=campaign_id,
                pending_roll_id=pending_roll_id,
                formula=dice.formula,
                rolls_json=json.dumps(dice.rolls),
                modifier=dice.modifier,
                total=dice.total,
                dc=pending.dc,
                outcome=outcome,
                reason=pending.reason,
            )
            session.add(roll)
            session.flush()
            pending.status = "resolved"
            pending.resolved_at = now_utc()
            session.add(pending)
            modifier = f" {dice.modifier:+d}" if dice.modifier else ""
            roll_turn = Turn(
                campaign_id=campaign_id,
                action_request_id=request.id,
                speaker="Roll",
                content=(
                    f"{pending.ability}{f' ({pending.skill})' if pending.skill else ''}: "
                    f"{dice.formula} -> {dice.rolls}{modifier} = {dice.total} "
                    f"vs DC {pending.dc}: {outcome}"
                ),
            )
            session.add(roll_turn)
            add_event(
                session,
                campaign_id,
                "dice_rolled",
                {
                    "pending_roll_id": pending.id,
                    "formula": dice.formula,
                    "rolls": dice.rolls,
                    "modifier": dice.modifier,
                    "total": dice.total,
                    "dc": pending.dc,
                    "outcome": outcome,
                },
                action_request_id=request.id,
            )
        request.dice_roll_id = roll.id
        request.updated_at = now_utc()
        session.add(request)
        touch_campaign(session, campaign_id)
        session.commit()
        session.refresh(pending)
        session.refresh(roll)
    except IntegrityError as exc:
        session.rollback()
        raise OperationInProgressError(
            "Another campaign operation is already in progress."
        ) from exc
    except Exception:
        session.rollback()
        raise

    context = build_campaign_context(
        session,
        campaign_id,
        turn_limit=prompt_history_limit,
    )
    return RollPreparation(
        request_id=request.id,
        campaign_id=campaign_id,
        idempotency_key=idempotency_key,
        pending_roll=pending_roll_snapshot(pending),
        dice_roll=dice_roll_snapshot(roll),
        context=context,
    )
