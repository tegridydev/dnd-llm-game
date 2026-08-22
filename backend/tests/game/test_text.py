from __future__ import annotations

from dndllm26.game.prompts import (
    NARRATION_SYSTEM,
    build_dm_prompt,
    build_world_update_messages,
    combat_narration_is_grounded,
    combat_transition_violations,
    protagonist_identity_violations,
    safe_combat_transition,
    safe_second_person_narration,
)
from dndllm26.game.text import (
    MAX_DM_CHARS,
    MAX_DM_WORDS,
    clean_choice_list,
    clean_extracted_text,
    extract_choices,
    trim_dm_text,
)
from dndllm26.game.types import CampaignContext, CharacterSnapshot, WorldSnapshot


def test_extract_choices_rejects_prompt_placeholders() -> None:
    text = """A locked gate shudders in the wind.\n\nChoices:\n1. Inspect the rusted lock.\n2. specific playable action\n3. Ask Mira about the sigil.\n"""
    assert extract_choices(text) == ["Inspect the rusted lock.", "Ask Mira about the sigil."]


def test_choice_cleanup_deduplicates_case_insensitively() -> None:
    assert clean_choice_list(["1. Open the door", "open the door", {"action": "Wait"}]) == [
        "Open the door",
        "Wait",
    ]


def test_placeholder_state_falls_back() -> None:
    assert (
        clean_extracted_text(
            "short current location",
            "Old Gate",
            limit=120,
            default="Unknown",
        )
        == "Old Gate"
    )


def test_dm_output_is_bounded_and_meta_removed() -> None:
    source = "Here is a possible response: " + "word " * 400
    output = trim_dm_text(source)
    assert len(output) <= MAX_DM_CHARS
    assert len(output.split()) <= MAX_DM_WORDS
    assert not output.lower().startswith("here is")


def test_dm_output_removes_leaked_choices_and_incomplete_tail() -> None:
    source = (
        "The merchant lowers his voice. The square grows quiet as a patrol passes.\n\n"
        "You have the following options:\n1. Ask about the ruins.\n2. Follow"
    )
    assert trim_dm_text(source) == (
        "The merchant lowers his voice. The square grows quiet as a patrol passes."
    )


def test_authoritative_narration_contract_forbids_choices() -> None:
    lower = NARRATION_SYSTEM.casefold()
    assert "no choices" in lower
    assert "3000" in lower
    assert "260 words" in lower
    assert "end with" not in lower


def test_combat_narration_requires_authoritative_ledger_and_rejects_redirected_attack() -> None:
    context = CampaignContext(
        campaign_id=1,
        title="The Gate",
        setting="A frontier inn",
        tone="heroic",
        characters=(),
        turns=(),
        world=WorldSnapshot(
            current_location="Red Griffin Inn",
            active_objective="Survive the fight",
            scene_summary="A bandit attacks.",
            choices=(),
            npcs=("Bartender", "Patrons of the Red Griffin Inn"),
        ),
        lore_documents=(),
    )
    events = ("Mira Voss hits Bandit for 6 damage.",)
    assert combat_narration_is_grounded(
        context,
        "Mira Voss hits Bandit for 6 damage. The bandit reels into a table.",
        events,
    )
    assert not combat_narration_is_grounded(
        context,
        "Your short sword slices into the Bartender as the patrons gasp.",
        events,
    )
    assert not combat_narration_is_grounded(
        context,
        "Mira Voss hits Bandit for 6 damage. You then stab the Bartender.",
        events,
    )


def test_combat_transition_sanitizer_stops_before_resolved_exchange() -> None:
    unsafe = (
        "Mira closes the distance and raises her blade. "
        "The guard blocks her strike and forces her backward. Another soldier attacks."
    )
    assert combat_transition_violations(unsafe) == (
        "resolved hit or defense",
        "resolved impact or forced movement",
        "unregistered attacker",
    )
    assert (
        safe_combat_transition(
            unsafe,
            hero_name="Mira",
            opponent_names=("Guard",),
        )
        == "Mira closes the distance and raises her blade."
    )


def test_combat_transition_sanitizer_has_deterministic_fallback() -> None:
    result = safe_combat_transition(
        "The guard hits Mira for six damage.",
        hero_name="Mira",
        opponent_names=("Guard",),
    )
    assert "You close on Guard" in result
    assert not combat_transition_violations(result)


def test_second_person_repair_removes_independent_protagonist() -> None:
    text = "Mira Voss signals to you, then Mira draws her blade."
    assert protagonist_identity_violations(text, "Mira Voss")
    assert safe_second_person_narration(text, "Mira Voss") == (
        "You signal, then you draw your blade."
    )


def test_prompt_marks_protagonist_as_player_and_hides_corrupt_npc_memory() -> None:
    hero = CharacterSnapshot(
        name="Mira Voss",
        role="protagonist",
        level=1,
        ancestry="Human",
        character_class="Rogue",
        backstory="A scout.",
        abilities=(),
        current_hp=9,
        max_hp=9,
        armor_class=14,
        skills=(),
        inventory=(),
        conditions=(),
    )
    context = CampaignContext(
        campaign_id=2,
        title="Arena",
        setting="A crowded arena",
        tone="heroic",
        characters=(hero,),
        turns=(),
        world=WorldSnapshot(
            current_location="Arena",
            active_objective="Advance",
            scene_summary="The crowd waits.",
            choices=(),
            npcs=("Mira Voss: hooded scout", "Arena Guard"),
        ),
        lore_documents=(),
    )
    _, prompt = build_dm_prompt(context, "Approach the guard", ())
    assert "PLAYER-CONTROLLED PROTAGONIST" in prompt
    assert '"you" and "your" mean Mira Voss' in prompt
    assert "Known NPCs: Arena Guard" in prompt
    assert "Known NPCs: Mira Voss" not in prompt
    _, world_prompt = build_world_update_messages(
        "You approach the guard.",
        context.world,
        protagonist="Mira Voss",
        party_names=("Mira Voss",),
    )
    assert "Known NPCs: Arena Guard" in world_prompt
    assert "Known NPCs: Mira Voss" not in world_prompt
