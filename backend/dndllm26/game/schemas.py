from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from dndllm26.game.dice import normalize_formula
from dndllm26.game.text import clean_choice_list, clean_choice_value, clean_roll_text


class RollDecisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    requires_roll: StrictBool
    narration: str = ""
    formula: str = "1d20"
    ability: str = "Ability"
    skill: str | None = None
    dc: StrictInt = Field(default=10, ge=5, le=25)
    reason: str = ""
    encounter_enemies: list[Literal["bandit", "goblin", "wolf", "skeleton", "cultist", "ogre"]] = (
        Field(default_factory=list, max_length=6)
    )
    encounter_names: list[str] = Field(default_factory=list, max_length=6)
    action_summary: str = ""
    lore_query: str = ""

    @field_validator("formula")
    @classmethod
    def valid_formula(cls, value: str) -> str:
        return normalize_formula(value)

    @field_validator("ability")
    @classmethod
    def valid_ability(cls, value: str) -> str:
        return clean_choice_value(value)[:80] or "Ability"

    @field_validator("skill")
    @classmethod
    def valid_skill(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = clean_choice_value(value)[:80]
        return cleaned or None

    @field_validator("narration", "reason", "action_summary", "lore_query")
    @classmethod
    def bounded_text(cls, value: str) -> str:
        return value.strip()[:180]

    @field_validator("encounter_names")
    @classmethod
    def valid_encounter_names(cls, values: list[str]) -> list[str]:
        return [clean_choice_value(value)[:120] for value in values if clean_choice_value(value)][
            :6
        ]


class WorldUpdateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    location: str
    objective: str
    summary: str
    choices: list[str]
    facts: list[str] = Field(default_factory=list, max_length=20)
    npcs: list[str] = Field(default_factory=list, max_length=20)
    resolved_facts: list[str] = Field(default_factory=list, max_length=20)
    location_changed: StrictBool = False
    objective_changed: StrictBool = False

    @field_validator("location")
    @classmethod
    def location_text(cls, value: str) -> str:
        return value.strip()[:120]

    @field_validator("objective")
    @classmethod
    def objective_text(cls, value: str) -> str:
        return value.strip()[:180]

    @field_validator("summary")
    @classmethod
    def summary_text(cls, value: str) -> str:
        return value.strip()[:260]

    @field_validator("choices")
    @classmethod
    def valid_choices(cls, value: list[str]) -> list[str]:
        return clean_choice_list(value)

    @field_validator("facts", "npcs", "resolved_facts")
    @classmethod
    def bounded_memory(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            item = clean_choice_value(value)[:160]
            if item and item.casefold() not in {existing.casefold() for existing in cleaned}:
                cleaned.append(item)
        return cleaned[:20]


def concise_roll_reason(action: str, skill: str | None) -> str:
    label = skill or "Check"
    clean_action = clean_choice_value(action).rstrip(".")
    if len(clean_action) > 90:
        clean_action = clean_action[:90].rsplit(" ", 1)[0]
    return f"{label}: {clean_action}"[:180]


def fallback_roll_decision(action: str, known_npcs: tuple[str, ...] = ()) -> RollDecisionOutput:
    lower = action.casefold()
    combat_words = (
        "attack",
        "engage",
        "strike",
        "stab",
        "slash",
        "shoot",
        "fire at",
        "grapple",
        "tackle",
        "punch",
        "kick",
        "accept duel",
        "accept the duel",
    )
    preparatory = ("look for an opening", "wait for an opening", "prepare", "ready")
    if any(word in lower for word in combat_words) and not any(
        phrase in lower for phrase in preparatory
    ):
        target_names: list[str] = []
        for raw_npc in known_npcs:
            name = raw_npc.split(",", 1)[0].split(":", 1)[0].split("(", 1)[0].strip()
            if name and name.casefold().rstrip("s") in lower:
                target_names = [name]
                break
        return RollDecisionOutput(
            requires_roll=False,
            action_summary=clean_choice_value(action),
            lore_query=clean_choice_value(action),
            encounter_enemies=["bandit"],
            encounter_names=target_names,
        )
    checks = [
        (("sneak", "hide", "stealth"), "Dexterity", "Stealth", 13),
        (
            ("persuade", "convince", "lie", "deceive", "rumour", "rumor"),
            "Charisma",
            "Persuasion",
            12,
        ),
        (("search", "inspect", "investigate", "study"), "Intelligence", "Investigation", 12),
        (("listen", "notice", "watch", "spot"), "Wisdom", "Perception", 12),
        (("climb", "force", "break", "lift"), "Strength", "Athletics", 13),
    ]
    for words, ability, skill, dc in checks:
        if any(word in lower for word in words):
            enemies = ["bandit"] if skill == "Attack" else []
            target_names: list[str] = []
            if enemies:
                for raw_npc in known_npcs:
                    name = raw_npc.split(",", 1)[0].split(":", 1)[0].strip()
                    if name and name.casefold() in lower:
                        target_names = [name]
                        break
            return RollDecisionOutput(
                requires_roll=True,
                narration=action[:140],
                formula="1d20+2",
                ability=ability,
                skill=skill,
                dc=dc,
                reason=concise_roll_reason(action, skill),
                encounter_enemies=enemies,
                encounter_names=target_names,
                action_summary=clean_choice_value(action),
                lore_query=clean_choice_value(action),
            )
    return RollDecisionOutput(
        requires_roll=False,
        narration="",
        formula="1d20",
        ability="Ability",
        skill=None,
        dc=10,
        reason="",
        action_summary=clean_choice_value(action),
        lore_query=clean_choice_value(action),
    )


def normalise_roll_decision(
    decision: RollDecisionOutput,
    *,
    action: str,
) -> RollDecisionOutput:
    fallback = fallback_roll_decision(action)
    return decision.model_copy(
        update={
            "reason": clean_roll_text(
                decision.reason,
                concise_roll_reason(action, decision.skill or decision.ability),
            ),
            "narration": clean_roll_text(decision.narration, action[:140])
            if decision.requires_roll
            else "",
            "formula": decision.formula if decision.requires_roll else fallback.formula,
            "action_summary": decision.action_summary or clean_choice_value(action),
            "lore_query": decision.lore_query or clean_choice_value(action),
            "encounter_names": decision.encounter_names[: len(decision.encounter_enemies)],
        }
    )
