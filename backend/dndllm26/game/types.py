from __future__ import annotations

from dataclasses import dataclass
from dndllm26.rag.store import LoreReference


@dataclass(frozen=True, slots=True)
class CharacterSnapshot:
    name: str
    ancestry: str
    character_class: str
    backstory: str
    role: str
    level: int
    abilities: tuple[tuple[str, int], ...]
    current_hp: int
    max_hp: int
    armor_class: int
    skills: tuple[str, ...]
    inventory: tuple[str, ...]
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TurnSnapshot:
    speaker: str
    content: str


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    current_location: str
    active_objective: str
    scene_summary: str
    choices: tuple[str, ...]
    facts: tuple[str, ...] = ()
    npcs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CampaignContext:
    campaign_id: int
    title: str
    setting: str
    tone: str
    characters: tuple[CharacterSnapshot, ...]
    turns: tuple[TurnSnapshot, ...]
    world: WorldSnapshot
    lore_documents: tuple[LoreReference, ...]


@dataclass(frozen=True, slots=True)
class PendingRollSnapshot:
    id: int
    campaign_id: int
    action_request_id: int | None
    action_text: str
    formula: str
    ability: str
    skill: str | None
    dc: int
    reason: str
    narration: str
    status: str


@dataclass(frozen=True, slots=True)
class DiceRollSnapshot:
    id: int
    pending_roll_id: int | None
    formula: str
    rolls: tuple[int, ...]
    modifier: int
    total: int
    dc: int | None
    outcome: str
    reason: str


@dataclass(frozen=True, slots=True)
class ActionPreparation:
    request_id: int
    campaign_id: int
    idempotency_key: str
    action_text: str
    context: CampaignContext
    replay_status: str | None = None
    pending_roll: PendingRollSnapshot | None = None


@dataclass(frozen=True, slots=True)
class RollPreparation:
    request_id: int
    campaign_id: int
    idempotency_key: str
    pending_roll: PendingRollSnapshot
    dice_roll: DiceRollSnapshot
    context: CampaignContext
    replay_status: str | None = None


@dataclass(frozen=True, slots=True)
class WorldUpdate:
    location: str
    objective: str
    summary: str
    choices: tuple[str, ...]
    facts: tuple[str, ...] = ()
    npcs: tuple[str, ...] = ()
