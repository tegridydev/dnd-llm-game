from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
import json

from sqlmodel import Session, select

from dndllm26.core.errors import ConflictError, NotFoundError, OperationInProgressError
from dndllm26.db.models import Campaign, CampaignOpening, Turn, now_utc
from dndllm26.game.campaigns import (
    add_event,
    build_campaign_context,
    get_or_create_world_state,
    touch_campaign,
)
from dndllm26.game.types import CampaignContext, WorldUpdate


@dataclass(frozen=True, slots=True)
class OpeningPreparation:
    opening_id: int
    campaign_id: int
    idempotency_key: str
    context: CampaignContext
    replay_status: str | None = None


def _age_seconds(opening: CampaignOpening) -> float:
    updated = opening.updated_at
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    return max(0.0, (now_utc() - updated).total_seconds())


def prepare_opening(
    session: Session,
    *,
    campaign_id: int,
    idempotency_key: str,
    prompt_history_limit: int = 14,
    stale_seconds: int = 900,
) -> OpeningPreparation:
    if not session.get(Campaign, campaign_id):
        raise NotFoundError("Campaign not found.")
    opening = session.exec(
        select(CampaignOpening).where(CampaignOpening.campaign_id == campaign_id)
    ).first()
    if not opening:
        has_dm = session.exec(
            select(Turn.id).where(Turn.campaign_id == campaign_id).where(Turn.speaker == "DM")
        ).first()
        opening = CampaignOpening(
            campaign_id=campaign_id,
            status="complete" if has_dm is not None else "needed",
        )
        session.add(opening)
        session.commit()
        session.refresh(opening)
    if opening.id is None:
        raise RuntimeError("Campaign opening id was not generated")
    if opening.status == "complete":
        return OpeningPreparation(
            opening.id,
            campaign_id,
            opening.idempotency_key or idempotency_key,
            build_campaign_context(session, campaign_id, turn_limit=prompt_history_limit),
            "complete",
        )
    if opening.status == "processing" and _age_seconds(opening) < stale_seconds:
        raise OperationInProgressError("The campaign opening is already being generated.")
    if opening.idempotency_key and opening.idempotency_key != idempotency_key:
        raise ConflictError("Retry the campaign opening with its original idempotency key.")
    opening.idempotency_key = idempotency_key
    opening.status = "processing"
    opening.error_code = None
    opening.error_message = None
    opening.updated_at = now_utc()
    session.add(opening)
    session.commit()
    return OpeningPreparation(
        opening.id,
        campaign_id,
        idempotency_key,
        build_campaign_context(session, campaign_id, turn_limit=prompt_history_limit),
    )


def fail_opening(
    session: Session,
    *,
    opening_id: int,
    code: str,
    message: str,
) -> None:
    opening = session.get(CampaignOpening, opening_id)
    if not opening or opening.status == "complete":
        return
    opening.status = "failed"
    opening.error_code = code[:80]
    opening.error_message = message[:500]
    opening.updated_at = now_utc()
    session.add(opening)
    session.commit()


def complete_opening(
    session: Session,
    *,
    opening_id: int,
    narration: str,
    world_update: WorldUpdate,
) -> None:
    opening = session.get(CampaignOpening, opening_id)
    if not opening:
        raise NotFoundError("Campaign opening not found.")
    if opening.status == "complete":
        return
    session.add(Turn(campaign_id=opening.campaign_id, speaker="DM", content=narration))
    state = get_or_create_world_state(session, opening.campaign_id)
    state.current_location = world_update.location
    state.active_objective = world_update.objective
    state.scene_summary = world_update.summary
    state.choices_json = json.dumps(list(world_update.choices), ensure_ascii=False)
    state.facts_json = json.dumps(list(world_update.facts), ensure_ascii=False)
    state.npcs_json = json.dumps(list(world_update.npcs), ensure_ascii=False)
    session.add(state)
    opening.status = "complete"
    opening.error_code = None
    opening.error_message = None
    opening.updated_at = now_utc()
    session.add(opening)
    add_event(
        session,
        opening.campaign_id,
        "campaign_intro_generated",
        {"location": world_update.location, "objective": world_update.objective},
    )
    touch_campaign(session, opening.campaign_id)
    session.commit()


def recover_openings(engine) -> int:
    with Session(engine) as session:
        rows = session.exec(
            select(CampaignOpening).where(CampaignOpening.status == "processing")
        ).all()
        for opening in rows:
            opening.status = "failed"
            opening.error_code = "process_restarted"
            opening.error_message = "Opening generation was interrupted and can be retried safely."
            opening.updated_at = now_utc()
            session.add(opening)
        if rows:
            session.commit()
        return len(rows)
