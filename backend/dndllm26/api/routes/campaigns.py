from __future__ import annotations

from dataclasses import asdict
from fastapi import APIRouter, Depends, Query, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

from dndllm26.api.deps import get_resources, get_session
from dndllm26.api.schemas import (
    CampaignCreate,
    CampaignDetailOut,
    CampaignOut,
    CampaignSummaryOut,
    CampaignUpdate,
    RestRequest,
)
from dndllm26.core.resources import AppResources
from dndllm26.db.models import (
    Campaign,
    CampaignLore,
    CampaignOpening,
    Character,
    Combatant,
    Encounter,
    LoreDocument,
    Quest,
    Turn,
    WorldState,
    now_utc,
)
from dndllm26.core.errors import NotFoundError
from dndllm26.game.campaigns import (
    campaign_detail,
    create_campaign,
    list_campaigns,
)
from dndllm26.game.progression import complete_quest, rest_party

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


@router.get("", response_model=list[CampaignSummaryOut])
def campaigns(
    include_archived: bool = False,
    session: Session = Depends(get_session),
) -> list[dict[str, object]]:
    rows = list_campaigns(session, include_archived=include_archived)
    ids = [campaign.id for campaign in rows if campaign.id is not None]
    states = (
        session.exec(select(WorldState).where(WorldState.campaign_id.in_(ids))).all() if ids else []
    )
    locations = {state.campaign_id: state.current_location for state in states}
    return [
        {
            **CampaignOut.model_validate(campaign).model_dump(),
            "current_location": locations.get(campaign.id, "Campaign opening"),
            "last_activity_at": campaign.updated_at,
        }
        for campaign in rows
    ]


@router.patch("/{campaign_id}", response_model=CampaignOut)
def update_campaign(
    campaign_id: int,
    payload: CampaignUpdate,
    session: Session = Depends(get_session),
) -> Campaign:
    campaign = session.get(Campaign, campaign_id)
    if not campaign:
        raise NotFoundError("Campaign not found.")
    if payload.title is not None:
        campaign.title = payload.title
    if payload.archived is not None:
        campaign.archived_at = now_utc() if payload.archived else None
    campaign.updated_at = now_utc()
    session.add(campaign)
    session.commit()
    session.refresh(campaign)
    return campaign


@router.get("/{campaign_id}/export")
def export_campaign(
    campaign_id: int,
    session: Session = Depends(get_session),
) -> Response:
    campaign = session.get(Campaign, campaign_id)
    if not campaign:
        raise NotFoundError("Campaign not found.")
    encounters = session.exec(
        select(Encounter).where(Encounter.campaign_id == campaign_id).order_by(Encounter.id)
    ).all()
    encounter_ids = [row.id for row in encounters if row.id is not None]
    combatants = (
        session.exec(
            select(Combatant)
            .where(Combatant.encounter_id.in_(encounter_ids))
            .order_by(Combatant.encounter_id, Combatant.id)
        ).all()
        if encounter_ids
        else []
    )
    lore = session.exec(
        select(LoreDocument)
        .join(CampaignLore, CampaignLore.lore_document_id == LoreDocument.id)
        .where(CampaignLore.campaign_id == campaign_id)
        .order_by(LoreDocument.filename)
    ).all()
    opening = session.exec(
        select(CampaignOpening).where(CampaignOpening.campaign_id == campaign_id)
    ).first()
    payload = {
        "schema_version": 1,
        "exported_at": now_utc(),
        "campaign": campaign,
        "opening": (
            {
                "status": opening.status,
                "error_code": opening.error_code,
                "error_message": opening.error_message,
                "updated_at": opening.updated_at,
            }
            if opening
            else None
        ),
        "characters": session.exec(
            select(Character).where(Character.campaign_id == campaign_id).order_by(Character.id)
        ).all(),
        "world_state": session.exec(
            select(WorldState).where(WorldState.campaign_id == campaign_id)
        ).first(),
        "quests": session.exec(
            select(Quest).where(Quest.campaign_id == campaign_id).order_by(Quest.id)
        ).all(),
        "turns": session.exec(
            select(Turn).where(Turn.campaign_id == campaign_id).order_by(Turn.id)
        ).all(),
        "encounters": encounters,
        "combatants": combatants,
        "lore": [
            {
                "filename": document.filename,
                "size_bytes": document.size_bytes,
                "page_count": document.page_count,
                "status": document.status,
            }
            for document in lore
        ],
    }
    safe_title = (
        "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in campaign.title.lower()
        ).strip("-")
        or f"campaign-{campaign_id}"
    )
    return JSONResponse(
        content=jsonable_encoder(payload),
        headers={"Content-Disposition": f'attachment; filename="{safe_title}.json"'},
    )


@router.post("", response_model=CampaignOut, status_code=201)
async def create(
    payload: CampaignCreate,
    session: Session = Depends(get_session),
) -> CampaignOut:
    campaign = create_campaign(
        session,
        title=payload.title,
        setting=payload.setting,
        tone=payload.tone,
        protagonist_id=payload.protagonist_id,
        companion_ids=payload.companion_ids,
        lore_document_ids=payload.lore_document_ids,
    )
    return CampaignOut.model_validate(campaign)


@router.get("/{campaign_id}", response_model=CampaignDetailOut)
def detail(
    campaign_id: int,
    cursor: str | None = Query(default=None, min_length=1, max_length=512),
    limit: int | None = Query(default=None, ge=20, le=500),
    resources: AppResources = Depends(get_resources),
    session: Session = Depends(get_session),
) -> dict[str, object]:
    result = campaign_detail(
        session,
        campaign_id,
        turn_limit=limit or 100,
        cursor=cursor,
    )
    pending = result["pending_roll"]
    if pending is not None:
        result["pending_roll"] = asdict(pending)
    recoverable = result["recoverable_operation"]
    if isinstance(recoverable, dict) and recoverable.get("pending_roll") is not None:
        recoverable["pending_roll"] = asdict(recoverable["pending_roll"])
    return result


@router.post("/{campaign_id}/rest")
def rest(
    campaign_id: int,
    payload: RestRequest,
    session: Session = Depends(get_session),
) -> dict[str, object]:
    return {
        "kind": payload.kind,
        "events": rest_party(session, campaign_id, payload.kind, payload.hit_dice),
    }


@router.post("/{campaign_id}/quests/{quest_id}/complete")
def finish_quest(
    campaign_id: int,
    quest_id: int,
    session: Session = Depends(get_session),
) -> dict[str, object]:
    return complete_quest(session, campaign_id, quest_id)
