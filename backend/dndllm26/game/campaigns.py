from __future__ import annotations

import base64
import binascii
from datetime import datetime
import json
from typing import Any, Sequence

from sqlalchemy import and_, or_
from sqlmodel import Session, select

from dndllm26.core.errors import NotFoundError, ValidationError
from dndllm26.db.models import (
    ActionRequest,
    Campaign,
    CampaignOpening,
    CampaignLore,
    Character,
    DiceRoll,
    GameEvent,
    Hero,
    LoreDocument,
    PendingRoll,
    Quest,
    Turn,
    WorldState,
    now_utc,
)
from dndllm26.game.text import clean_choice_list
from dndllm26.game.catalog import ABILITIES, build_hero_sheet
from dndllm26.game.combat import encounter_detail
from dndllm26.game.types import (
    CampaignContext,
    CharacterSnapshot,
    PendingRollSnapshot,
    TurnSnapshot,
    WorldSnapshot,
)
from dndllm26.rag.store import LoreReference

DEFAULT_CHOICES = (
    "Look for work or rumours.",
    "Find a safe place to rest.",
    "Study the local trouble.",
)


def add_event(
    session: Session,
    campaign_id: int,
    event_type: str,
    payload: dict[str, Any],
    *,
    action_request_id: int | None = None,
) -> GameEvent:
    event = GameEvent(
        campaign_id=campaign_id,
        action_request_id=action_request_id,
        event_type=event_type,
        payload_json=json.dumps(payload, ensure_ascii=False, default=str),
    )
    session.add(event)
    return event


def touch_campaign(session: Session, campaign_id: int) -> None:
    campaign = session.get(Campaign, campaign_id)
    if campaign:
        campaign.updated_at = now_utc()
        session.add(campaign)


def choices_from_state(state: WorldState) -> list[str]:
    try:
        parsed = json.loads(state.choices_json)
    except (json.JSONDecodeError, TypeError):
        return []
    return clean_choice_list(parsed)


def pending_roll_snapshot(pending: PendingRoll) -> PendingRollSnapshot:
    if pending.id is None:
        raise ValueError("Pending roll has not been persisted")
    return PendingRollSnapshot(
        id=pending.id,
        campaign_id=pending.campaign_id,
        action_request_id=pending.action_request_id,
        action_text=pending.action_text,
        formula=pending.formula,
        ability=pending.ability,
        skill=pending.skill,
        dc=pending.dc,
        reason=pending.reason,
        narration=pending.narration,
        status=pending.status,
    )


def get_pending_roll(session: Session, campaign_id: int) -> PendingRoll | None:
    return session.exec(
        select(PendingRoll)
        .where(PendingRoll.campaign_id == campaign_id)
        .where(PendingRoll.status.in_(["pending", "resolving"]))
        .order_by(PendingRoll.created_at.desc(), PendingRoll.id.desc())
    ).first()


def get_recoverable_operation(session: Session, campaign_id: int) -> dict[str, object] | None:
    request = session.exec(
        select(ActionRequest)
        .where(ActionRequest.campaign_id == campaign_id)
        .where(ActionRequest.status.in_(["failed", "interrupted"]))
        .where(ActionRequest.retryable.is_(True))
        .order_by(ActionRequest.updated_at.desc(), ActionRequest.id.desc())
    ).first()
    if not request:
        return None
    pending = session.get(PendingRoll, request.pending_roll_id) if request.pending_roll_id else None
    roll = session.get(DiceRoll, request.dice_roll_id) if request.dice_roll_id else None
    rolls: list[int] = []
    if roll:
        try:
            parsed = json.loads(roll.rolls_json)
            rolls = [int(value) for value in parsed if isinstance(value, int)]
        except (json.JSONDecodeError, TypeError):
            rolls = []
    return {
        "kind": request.operation,
        "idempotency_key": request.idempotency_key,
        "action_text": request.action_text,
        "error": request.error_message
        or "The operation was interrupted and can be retried safely.",
        "pending_roll": pending_roll_snapshot(pending) if pending else None,
        "dice_roll": (
            {
                "id": roll.id,
                "pending_roll_id": roll.pending_roll_id,
                "formula": roll.formula,
                "rolls": rolls,
                "modifier": roll.modifier,
                "total": roll.total,
                "dc": roll.dc,
                "outcome": roll.outcome,
                "reason": roll.reason,
            }
            if roll
            else None
        ),
    }


def get_or_create_world_state(session: Session, campaign_id: int) -> WorldState:
    state = session.exec(select(WorldState).where(WorldState.campaign_id == campaign_id)).first()
    if state:
        return state
    state = WorldState(campaign_id=campaign_id, choices_json=json.dumps(DEFAULT_CHOICES))
    session.add(state)
    session.flush()
    return state


def get_lore_references(session: Session, campaign_id: int) -> tuple[LoreReference, ...]:
    rows = session.exec(
        select(LoreDocument)
        .join(CampaignLore, CampaignLore.lore_document_id == LoreDocument.id)
        .where(CampaignLore.campaign_id == campaign_id)
        # A refresh writes a new inactive index while the previous key remains usable.
        # Retrieval therefore follows the active key, not the transient worker status.
        .where(LoreDocument.index_key != "")
        .order_by(LoreDocument.filename)
    ).all()
    return tuple(
        LoreReference(
            document_id=document.id or 0,
            filename=document.filename,
            index_key=document.index_key,
            index_table=document.index_table,
            embed_model=document.embed_model,
            embed_dimension=document.embed_dimension,
        )
        for document in rows
        if document.id is not None and document.index_key
    )


def build_campaign_context(
    session: Session,
    campaign_id: int,
    *,
    turn_limit: int,
    exclude_turn_id: int | None = None,
) -> CampaignContext:
    campaign = session.get(Campaign, campaign_id)
    if not campaign:
        raise NotFoundError("Campaign not found.")
    state = get_or_create_world_state(session, campaign_id)
    characters = session.exec(
        select(Character).where(Character.campaign_id == campaign_id).order_by(Character.id)
    ).all()
    statement = select(Turn).where(Turn.campaign_id == campaign_id)
    if exclude_turn_id is not None:
        statement = statement.where(Turn.id != exclude_turn_id)
    newest = session.exec(
        statement.order_by(Turn.created_at.desc(), Turn.id.desc()).limit(turn_limit)
    ).all()
    turns = list(reversed(newest))
    return CampaignContext(
        campaign_id=campaign_id,
        title=campaign.title,
        setting=campaign.setting,
        tone=campaign.tone,
        characters=tuple(
            CharacterSnapshot(
                name=character.name,
                ancestry=character.ancestry,
                character_class=character.character_class,
                backstory=character.backstory,
                role=character.role,
                level=character.level,
                abilities=tuple(
                    (ability, int(getattr(character, ability))) for ability in ABILITIES
                ),
                current_hp=character.current_hp,
                max_hp=character.max_hp,
                armor_class=character.armor_class,
                skills=tuple(json.loads(character.skills_json)),
                inventory=tuple(json.loads(character.inventory_json)),
                conditions=tuple(json.loads(character.conditions_json)),
            )
            for character in characters
        ),
        turns=tuple(TurnSnapshot(speaker=turn.speaker, content=turn.content) for turn in turns),
        world=WorldSnapshot(
            current_location=state.current_location,
            active_objective=state.active_objective,
            scene_summary=state.scene_summary,
            choices=tuple(choices_from_state(state)),
            facts=tuple(json.loads(state.facts_json)),
            npcs=tuple(json.loads(state.npcs_json)),
        ),
        lore_documents=get_lore_references(session, campaign_id),
    )


def seed_default_heroes(session: Session) -> None:
    if session.exec(select(Hero.id)).first() is not None:
        return
    sheet = build_hero_sheet("Human", "Rogue")
    session.add(
        Hero(
            name="Mira Voss",
            ancestry="Human",
            character_class="Rogue",
            backstory=(
                "A careful scout who knows the city roofs and owes a debt to a vanished archivist."
            ),
            inventory_json=json.dumps(["shortbow", "lockpicks", "hooded lantern"]),
            **{ability: sheet[ability] for ability in ABILITIES},
            max_hp=sheet["max_hp"],
            armor_class=sheet["armor_class"],
            speed=sheet["speed"],
            skills_json=json.dumps(sheet["skills"]),
            saves_json=json.dumps(sheet["saves"]),
            spells_json=json.dumps(sheet["spells"]),
            resources_json=json.dumps(sheet["resources"]),
        )
    )
    session.commit()


def create_campaign(
    session: Session,
    *,
    title: str,
    setting: str,
    tone: str,
    protagonist_id: int,
    companion_ids: Sequence[int],
    lore_document_ids: Sequence[int],
) -> Campaign:
    campaign = Campaign(title=title, setting=setting, tone=tone)
    session.add(campaign)
    session.flush()
    if campaign.id is None:
        raise RuntimeError("Campaign id was not generated")

    session.add(
        Turn(
            campaign_id=campaign.id,
            speaker="System",
            content=f"Campaign created. Setting: {setting} Tone: {tone}",
        )
    )
    session.add(
        WorldState(
            campaign_id=campaign.id,
            current_location="Campaign opening",
            active_objective="Establish the first scene.",
            scene_summary=f"{setting} Tone: {tone}",
            choices_json=json.dumps(DEFAULT_CHOICES),
        )
    )
    session.add(CampaignOpening(campaign_id=campaign.id))
    session.add(
        Quest(
            campaign_id=campaign.id,
            title="Answer the opening call",
            objective="Discover the immediate threat and choose how to confront it.",
            status="active",
            milestone_reward=1,
        )
    )

    protagonist = session.get(Hero, protagonist_id)
    if not protagonist:
        raise NotFoundError("A valid protagonist is required.")
    hero_roles = [
        (protagonist_id, "protagonist"),
        *[(value, "companion") for value in companion_ids],
    ]
    seen_heroes: set[int] = set()
    for hero_id, role in hero_roles:
        if hero_id in seen_heroes:
            continue
        hero = session.get(Hero, hero_id)
        if not hero:
            continue
        seen_heroes.add(hero_id)
        session.add(
            Character(
                campaign_id=campaign.id,
                name=hero.name,
                ancestry=hero.ancestry,
                character_class=hero.character_class,
                backstory=hero.backstory,
                inventory_json=hero.inventory_json,
                role=role,
                level=hero.level,
                **{ability: getattr(hero, ability) for ability in ABILITIES},
                max_hp=hero.max_hp,
                current_hp=hero.max_hp,
                armor_class=hero.armor_class,
                speed=hero.speed,
                skills_json=hero.skills_json,
                saves_json=hero.saves_json,
                spells_json=hero.spells_json,
                resources_json=hero.resources_json,
            )
        )

    seen_lore: set[int] = set()
    for document_id in lore_document_ids:
        if document_id in seen_lore:
            continue
        document = session.get(LoreDocument, document_id)
        if not document:
            continue
        seen_lore.add(document_id)
        session.add(CampaignLore(campaign_id=campaign.id, lore_document_id=document_id))

    add_event(
        session,
        campaign.id,
        "campaign_created",
        {"title": title, "setting": setting, "tone": tone},
    )
    session.commit()
    session.refresh(campaign)
    return campaign


def list_campaigns(
    session: Session, *, limit: int = 100, include_archived: bool = False
) -> list[Campaign]:
    statement = select(Campaign)
    if not include_archived:
        statement = statement.where(Campaign.archived_at.is_(None))
    return list(session.exec(statement.order_by(Campaign.updated_at.desc()).limit(limit)).all())


def list_turns(
    session: Session,
    campaign_id: int,
    *,
    limit: int,
    cursor: str | None = None,
) -> tuple[list[Turn], bool, str | None]:
    statement = select(Turn).where(Turn.campaign_id == campaign_id)
    if cursor is not None:
        before, before_id = decode_turn_cursor(cursor)
        statement = statement.where(
            or_(
                Turn.created_at < before,
                and_(Turn.created_at == before, Turn.id < before_id),
            )
        )
    rows = session.exec(
        statement.order_by(Turn.created_at.desc(), Turn.id.desc()).limit(limit + 1)
    ).all()
    has_more = len(rows) > limit
    selected = list(reversed(rows[:limit]))
    next_cursor = encode_turn_cursor(selected[0]) if has_more and selected else None
    return selected, has_more, next_cursor


def encode_turn_cursor(turn: Turn) -> str:
    if turn.id is None:
        raise ValueError("Cannot create a cursor for an unpersisted turn.")
    payload = json.dumps(
        {"created_at": turn.created_at.isoformat(), "id": turn.id, "v": 1},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_turn_cursor(value: str) -> tuple[datetime, int]:
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.b64decode(padded, altchars=b"-_", validate=True))
        if not isinstance(payload, dict) or payload.get("v") != 1:
            raise ValueError
        created_at = datetime.fromisoformat(payload["created_at"])
        turn_id = payload["id"]
        if not isinstance(turn_id, int) or isinstance(turn_id, bool) or turn_id <= 0:
            raise ValueError
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, binascii.Error) as exc:
        raise ValidationError("The history cursor is malformed or unsupported.") from exc
    return created_at, turn_id


def campaign_detail(
    session: Session,
    campaign_id: int,
    *,
    turn_limit: int,
    cursor: str | None = None,
) -> dict[str, object]:
    campaign = session.get(Campaign, campaign_id)
    if not campaign:
        raise NotFoundError("Campaign not found.")
    state = get_or_create_world_state(session, campaign_id)
    if state in session.new:
        session.commit()
        session.refresh(state)
    characters = session.exec(
        select(Character).where(Character.campaign_id == campaign_id).order_by(Character.id)
    ).all()
    turns, has_more, next_cursor = list_turns(
        session,
        campaign_id,
        limit=turn_limit,
        cursor=cursor,
    )
    pending = get_pending_roll(session, campaign_id)
    recoverable = get_recoverable_operation(session, campaign_id)
    opening = session.exec(
        select(CampaignOpening).where(CampaignOpening.campaign_id == campaign_id)
    ).first()
    last_roll = session.exec(
        select(DiceRoll)
        .where(DiceRoll.campaign_id == campaign_id)
        .order_by(DiceRoll.created_at.desc(), DiceRoll.id.desc())
    ).first()
    quests = session.exec(
        select(Quest).where(Quest.campaign_id == campaign_id).order_by(Quest.status, Quest.id)
    ).all()
    return {
        "campaign": campaign,
        "characters": characters,
        "turns": turns,
        "world_state": state,
        "choices": choices_from_state(state),
        "pending_roll": pending_roll_snapshot(pending) if pending else None,
        "recoverable_operation": recoverable,
        "opening": {
            "status": opening.status if opening else "needed",
            "idempotency_key": opening.idempotency_key if opening else None,
            "error": opening.error_message if opening else None,
            "retryable": not opening or opening.status in {"needed", "failed"},
        },
        "last_roll": (
            {
                "id": last_roll.id,
                "pending_roll_id": last_roll.pending_roll_id,
                "formula": last_roll.formula,
                "rolls": list(json.loads(last_roll.rolls_json)),
                "modifier": last_roll.modifier,
                "total": last_roll.total,
                "dc": last_roll.dc,
                "outcome": last_roll.outcome,
                "reason": last_roll.reason,
            }
            if last_roll
            else None
        ),
        "encounter": encounter_detail(session, campaign_id),
        "quests": [
            {
                "id": quest.id,
                "title": quest.title,
                "objective": quest.objective,
                "status": quest.status,
                "milestone_reward": quest.milestone_reward,
            }
            for quest in quests
        ],
        "turn_page": {
            "has_more": has_more,
            "next_cursor": next_cursor,
            "limit": turn_limit,
        },
    }
