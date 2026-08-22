from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from dndllm26.db.models import ActionRequest, CampaignOpening, Character, DiceRoll, Hero, Turn
from dndllm26.game.catalog import ABILITIES, build_hero_sheet
from dndllm26.game.campaigns import campaign_detail, create_campaign
from dndllm26.game.play import (
    create_pending_roll,
    mark_request_failed,
    persist_narrative,
    prepare_action,
    prepare_roll_resolution,
    recover_stale_operations,
)
from dndllm26.game.schemas import RollDecisionOutput
from dndllm26.game.types import WorldUpdate


class FixedRandom:
    def randint(self, _minimum: int, _maximum: int) -> int:
        return 12


def make_campaign(session: Session) -> int:
    sheet = build_hero_sheet("Human", "Fighter")
    hero = Hero(
        name="Test Hero",
        ancestry="Human",
        character_class="Fighter",
        backstory="A test hero.",
        **{ability: sheet[ability] for ability in ABILITIES},
        max_hp=sheet["max_hp"],
        armor_class=sheet["armor_class"],
        speed=sheet["speed"],
    )
    session.add(hero)
    session.flush()
    assert hero.id is not None
    campaign = create_campaign(
        session,
        title="Test Campaign",
        setting="A test setting",
        tone="focused",
        protagonist_id=hero.id,
        companion_ids=[],
        lore_document_ids=[],
    )
    assert campaign.id is not None
    opening = session.exec(
        select(CampaignOpening).where(CampaignOpening.campaign_id == campaign.id)
    ).one()
    opening.status = "complete"
    session.add(opening)
    session.commit()
    return campaign.id


def test_foreign_keys_are_enforced(engine) -> None:
    with Session(engine) as session:
        session.add(
            Character(
                campaign_id=999,
                name="Orphan",
                ancestry="Human",
                character_class="Fighter",
                backstory="None",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_failed_action_retry_does_not_duplicate_player_turn(engine) -> None:
    with Session(engine) as session:
        campaign_id = make_campaign(session)
        first = prepare_action(
            session,
            campaign_id=campaign_id,
            idempotency_key="action-retry-1",
            action="Open the gate",
            prompt_history_limit=10,
            stale_seconds=900,
        )
        mark_request_failed(
            session,
            request_id=first.request_id,
            code="model_unavailable",
            message="Unavailable",
            retryable=True,
        )
        second = prepare_action(
            session,
            campaign_id=campaign_id,
            idempotency_key="action-retry-1",
            action="Open the gate",
            prompt_history_limit=10,
            stale_seconds=900,
        )
        assert second.request_id == first.request_id
        player_turns = session.exec(
            select(Turn).where(Turn.campaign_id == campaign_id).where(Turn.speaker == "Player")
        ).all()
        assert len(player_turns) == 1


def test_roll_retry_reuses_the_stored_dice_result(engine) -> None:
    with Session(engine) as session:
        campaign_id = make_campaign(session)
        action = prepare_action(
            session,
            campaign_id=campaign_id,
            idempotency_key="action-roll-1",
            action="Climb the wall",
            prompt_history_limit=10,
            stale_seconds=900,
        )
        pending = create_pending_roll(
            session,
            request_id=action.request_id,
            decision=RollDecisionOutput(
                requires_roll=True,
                formula="1d20+2",
                ability="Strength",
                skill="Athletics",
                dc=13,
                reason="Climb the wall",
                narration="",
            ),
        )
        first = prepare_roll_resolution(
            session,
            campaign_id=campaign_id,
            pending_roll_id=pending.id,
            idempotency_key="roll-retry-1",
            prompt_history_limit=10,
            stale_seconds=900,
            random_source=FixedRandom(),
        )
        mark_request_failed(
            session,
            request_id=first.request_id,
            code="model_unavailable",
            message="Unavailable",
            retryable=True,
        )
        detail = campaign_detail(session, campaign_id, turn_limit=20)
        recovery = detail["recoverable_operation"]
        assert isinstance(recovery, dict)
        assert recovery["idempotency_key"] == "roll-retry-1"
        assert recovery["pending_roll"].status == "resolved"
        assert recovery["dice_roll"]["total"] == 15

        second = prepare_roll_resolution(
            session,
            campaign_id=campaign_id,
            pending_roll_id=pending.id,
            idempotency_key="roll-retry-1",
            prompt_history_limit=10,
            stale_seconds=900,
        )
        assert second.dice_roll.id == first.dice_roll.id
        assert second.dice_roll.total == 15
        rolls = session.exec(select(DiceRoll).where(DiceRoll.pending_roll_id == pending.id)).all()
        assert len(rolls) == 1


def test_narrative_finalisation_is_idempotent(engine) -> None:
    with Session(engine) as session:
        campaign_id = make_campaign(session)
        action = prepare_action(
            session,
            campaign_id=campaign_id,
            idempotency_key="action-complete-1",
            action="Wait",
            prompt_history_limit=10,
            stale_seconds=900,
        )
        update = WorldUpdate(
            location="Old Gate",
            objective="Wait for the guard",
            summary="The party waits by the gate.",
            choices=("Ask the guard a question.",),
        )
        persist_narrative(
            session, request_id=action.request_id, dm_text="The guard arrives.", world_update=update
        )
        persist_narrative(
            session, request_id=action.request_id, dm_text="Duplicate.", world_update=update
        )
        dm_turns = session.exec(
            select(Turn).where(Turn.campaign_id == campaign_id).where(Turn.speaker == "DM")
        ).all()
        assert [turn.content for turn in dm_turns] == ["The guard arrives."]


def test_narrative_finalisation_can_roll_back_as_one_unit(engine) -> None:
    with Session(engine) as session:
        campaign_id = make_campaign(session)
        action = prepare_action(
            session,
            campaign_id=campaign_id,
            idempotency_key="action-atomic-1",
            action="Open the gate",
            prompt_history_limit=10,
            stale_seconds=900,
        )
        update = WorldUpdate(
            location="Gate",
            objective="Enter",
            summary="The gate opens.",
            choices=("Enter.",),
        )
        persist_narrative(
            session,
            request_id=action.request_id,
            dm_text="The gate opens.",
            world_update=update,
        )
        session.rollback()
        request = session.get(ActionRequest, action.request_id)
        assert request is not None and request.status == "processing"
        dm_turns = session.exec(
            select(Turn).where(Turn.campaign_id == campaign_id).where(Turn.speaker == "DM")
        ).all()
        assert dm_turns == []


def test_restart_immediately_exposes_durable_action_recovery(engine) -> None:
    with Session(engine) as session:
        campaign_id = make_campaign(session)
        prepared = prepare_action(
            session,
            campaign_id=campaign_id,
            idempotency_key="restart-action-1",
            action="Open the sealed door",
            prompt_history_limit=10,
            stale_seconds=900,
        )
    assert recover_stale_operations(engine) == 1
    with Session(engine) as session:
        detail = campaign_detail(session, campaign_id, turn_limit=20)
        recovery = detail["recoverable_operation"]
        assert isinstance(recovery, dict)
        assert recovery["idempotency_key"] == "restart-action-1"
        retried = prepare_action(
            session,
            campaign_id=campaign_id,
            idempotency_key="restart-action-1",
            action="Open the sealed door",
            prompt_history_limit=10,
            stale_seconds=900,
        )
        assert retried.request_id == prepared.request_id
