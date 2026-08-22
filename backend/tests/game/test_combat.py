from __future__ import annotations

import pytest
from sqlmodel import Session

from dndllm26.core.errors import ConflictError
from dndllm26.db.models import Combatant, DiceRoll, Hero
from dndllm26.game.campaigns import create_campaign
from dndllm26.game.catalog import ABILITIES, build_hero_sheet
from dndllm26.game.combat import (
    apply_combat_roll,
    assess_combat,
    balanced_enemy_keys,
    encounter_detail,
    prepare_combat_action,
    start_encounter,
    _action_roll_formula,
    _effective_defense,
)
import dndllm26.game.combat as combat_module


def test_mage_armor_and_restrained_modify_engine_calculations() -> None:
    combatant = Combatant(
        encounter_id=1,
        name="Wizard",
        side="party",
        lane="back",
        max_hp=8,
        current_hp=8,
        armor_class=12,
        dexterity_modifier=2,
        attack_bonus=5,
        conditions_json='["mage_armor", "restrained"]',
    )
    assert _effective_defense(combatant) == 13
    assert _action_roll_formula(combatant, {"kind": "attack"}) == "1d20+3"


def test_combat_action_applies_once_and_automates_round(engine) -> None:
    with Session(engine) as session:
        sheet = build_hero_sheet("Dwarf", "Fighter")
        hero = Hero(
            name="Borin",
            ancestry="Dwarf",
            character_class="Fighter",
            backstory="A steadfast guardian.",
            **{ability: sheet[ability] for ability in ABILITIES},
            max_hp=sheet["max_hp"],
            armor_class=sheet["armor_class"],
            speed=sheet["speed"],
        )
        session.add(hero)
        session.flush()
        campaign = create_campaign(
            session,
            title="Combat test",
            setting="A ruined road",
            tone="heroic",
            protagonist_id=hero.id or 0,
            companion_ids=[],
            lore_document_ids=[],
        )
        encounter = start_encounter(session, campaign.id or 0, ["bandit"])
        detail = encounter_detail(session, campaign.id or 0)
        assert detail is not None
        actions = {item["id"]: item for item in detail["legal_actions"]}
        assert actions["second_wind"]["target_type"] == "self"
        assert "action_surge" not in actions
        target = next(item for item in detail["combatants"] if item["side"] == "enemy")
        pending, replayed = prepare_combat_action(
            session,
            campaign_id=campaign.id or 0,
            encounter_id=encounter.id or 0,
            idempotency_key="combat-action-1",
            action_id="weapon_attack",
            target_id=int(target["id"]),
            destination_lane=None,
        )
        assert replayed is False
        replay, replayed = prepare_combat_action(
            session,
            campaign_id=campaign.id or 0,
            encounter_id=encounter.id or 0,
            idempotency_key="combat-action-1",
            action_id="weapon_attack",
            target_id=int(target["id"]),
            destination_lane=None,
        )
        assert replay.id == pending.id
        assert replayed is True
        with pytest.raises(ConflictError, match="different combat action"):
            prepare_combat_action(
                session,
                campaign_id=campaign.id or 0,
                encounter_id=encounter.id or 0,
                idempotency_key="combat-action-1",
                action_id="dodge",
                target_id=None,
                destination_lane=None,
            )
        roll = DiceRoll(
            campaign_id=campaign.id or 0,
            pending_roll_id=pending.id,
            formula=pending.formula,
            rolls_json="[20]",
            modifier=5,
            total=25,
            dc=pending.dc,
            outcome="success",
            reason=pending.reason,
        )
        session.add(roll)
        session.flush()
        first_events = apply_combat_roll(session, pending.id or 0, roll)
        second_events = apply_combat_roll(session, pending.id or 0, roll)
        assert first_events == second_events
        assert any("hits Bandit" in event for event in first_events)
        refreshed = encounter_detail(session, campaign.id or 0)
        assert refreshed is not None
        assert refreshed["recent_events"][-len(first_events) :] == first_events


def test_non_hostile_model_enemy_proposal_does_not_start_combat() -> None:
    assert not assess_combat(
        "Talk to a nearby merchant",
        "The merchant lowers his voice and glances toward the guards.",
        model_enemy_keys=["goblin"] * 6,
    ).starts
    assert assess_combat(
        "Attack the goblin",
        "The goblin bares its teeth.",
        model_enemy_keys=["goblin"],
    ).starts


def test_committed_hostility_does_not_depend_on_model_candidates() -> None:
    known = (
        "Soldiers (mismatched armor)",
        "Soldier (mismatched armor, charging)",
        "Barmaid (grim expression)",
    )
    assessment = assess_combat("Engage the soldier", known_npcs=known)
    assert assessment.starts
    assert assessment.enemy_keys == ("bandit",)
    assert assessment.display_names == ("Soldier",)
    assert assessment.trigger == "player_commitment"


def test_explicit_role_target_becomes_display_identity_without_npc_memory() -> None:
    assessment = assess_combat("Attack the closest guard")
    assert assessment.starts
    assert assessment.enemy_keys == ("bandit",)
    assert assessment.display_names == ("Guard",)


def test_duel_is_limited_to_named_opponent() -> None:
    assessment = assess_combat(
        "Accept the one-on-one duel",
        model_enemy_keys=("bandit", "bandit", "cultist"),
        model_names=("Captain Rusk", "Guard", "Watcher"),
        known_npcs=("Captain Rusk (scarred guard captain)", "Guard", "Watcher"),
        context_text="Captain Rusk awaits the answer to his challenge.",
    )
    assert assessment.starts
    assert assessment.duel
    assert assessment.enemy_keys == ("bandit",)
    assert assessment.display_names == ("Captain Rusk",)


@pytest.mark.parametrize(
    "action",
    [
        "Draw my sword and warn the soldier.",
        "Challenge the soldier to a duel.",
        "Look for an opening to strike.",
        "Fight for the workers' rights.",
    ],
)
def test_preparatory_or_nonviolent_actions_remain_roleplay(action: str) -> None:
    assert not assess_combat(action, known_npcs=("Soldier",)).starts


def test_npc_attack_in_narration_starts_combat_without_model_candidate() -> None:
    assessment = assess_combat(
        "Try to reason with the soldier",
        "The soldier rejects the appeal and lunges at Mira with his spear.",
        known_npcs=("Soldier (mismatched armor)",),
    )
    assert assessment.starts
    assert assessment.trigger == "npc_attack"
    assert assessment.display_names == ("Soldier",)


def test_solo_level_one_threat_budget_limits_enemy_count(engine) -> None:
    with Session(engine) as session:
        sheet = build_hero_sheet("Human", "Rogue")
        hero = Hero(
            name="Solo",
            ancestry="Human",
            character_class="Rogue",
            backstory="Alone.",
            **{ability: sheet[ability] for ability in ABILITIES},
            max_hp=sheet["max_hp"],
            armor_class=sheet["armor_class"],
            speed=sheet["speed"],
        )
        session.add(hero)
        session.flush()
        campaign = create_campaign(
            session,
            title="Solo",
            setting="Road",
            tone="tense",
            protagonist_id=hero.id or 0,
            companion_ids=[],
            lore_document_ids=[],
        )
        assert balanced_enemy_keys(session, campaign.id or 0, ["goblin"] * 6) == ["goblin"]


def test_encounter_preserves_named_npc_identity(engine) -> None:
    with Session(engine) as session:
        sheet = build_hero_sheet("Human", "Fighter")
        hero = Hero(
            name="Mira Voss",
            ancestry="Human",
            character_class="Fighter",
            backstory="A vigilant guardian.",
            **{ability: sheet[ability] for ability in ABILITIES},
            max_hp=sheet["max_hp"],
            armor_class=sheet["armor_class"],
            speed=sheet["speed"],
        )
        session.add(hero)
        session.flush()
        campaign = create_campaign(
            session,
            title="Named foe",
            setting="A guarded road",
            tone="tense",
            protagonist_id=hero.id or 0,
            companion_ids=[],
            lore_document_ids=[],
        )
        encounter = start_encounter(
            session,
            campaign.id or 0,
            ["bandit"],
            ["Captain Rusk"],
        )
        detail = encounter_detail(session, campaign.id or 0)
        assert detail is not None
        enemies = [item for item in detail["combatants"] if item["side"] == "enemy"]
        assert encounter.name == "Conflict with Captain Rusk"
        assert enemies[0]["name"] == "Captain Rusk"


def test_encounter_resolves_enemies_ahead_of_hero_in_initiative(engine, monkeypatch) -> None:
    rolls = iter((1, 20))

    class FixedInitiative:
        def randint(self, _minimum: int, _maximum: int) -> int:
            return next(rolls)

    monkeypatch.setattr(combat_module, "SystemRandom", FixedInitiative)
    with Session(engine) as session:
        sheet = build_hero_sheet("Human", "Fighter")
        hero = Hero(
            name="Mira Voss",
            ancestry="Human",
            character_class="Fighter",
            backstory="A vigilant guardian.",
            **{ability: sheet[ability] for ability in ABILITIES},
            max_hp=sheet["max_hp"],
            armor_class=sheet["armor_class"],
            speed=sheet["speed"],
        )
        session.add(hero)
        session.flush()
        campaign = create_campaign(
            session,
            title="Initiative test",
            setting="A tavern duel",
            tone="tense",
            protagonist_id=hero.id or 0,
            companion_ids=[],
            lore_document_ids=[],
        )
        opening_events: list[str] = []
        start_encounter(
            session,
            campaign.id or 0,
            ["bandit"],
            ["Soldier"],
            opening_events,
        )
        detail = encounter_detail(session, campaign.id or 0)
        assert detail is not None
        assert detail["current_combatant_id"] == detail["protagonist_combatant_id"]
        assert opening_events
