from __future__ import annotations

from datetime import datetime
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dndllm26.game.catalog import ABILITIES, ANCESTRIES, CLASSES

_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}$")
PositiveId = Annotated[int, Field(gt=0)]


def validate_idempotency_key(value: str) -> str:
    cleaned = value.strip()
    if not _IDEMPOTENCY_RE.fullmatch(cleaned):
        raise ValueError(
            "Idempotency-Key must contain 8-128 letters, digits, dots, underscores, colons, or hyphens."
        )
    return cleaned


def validate_model_name(value: str) -> str:
    cleaned = value.strip()
    if not _MODEL_RE.fullmatch(cleaned) or ".." in cleaned:
        raise ValueError("Invalid Ollama model name.")
    return cleaned


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class CampaignCreate(BaseModel):
    title: str = Field(default="The Shattered Gate", min_length=1, max_length=120)
    setting: str = Field(
        default="A frontier city built over sealed ruins.", min_length=1, max_length=2_000
    )
    tone: str = Field(default="dangerous heroic fantasy", min_length=1, max_length=160)
    protagonist_id: int = Field(gt=0)
    companion_ids: list[PositiveId] = Field(default_factory=list, max_length=5)
    lore_document_ids: list[PositiveId] = Field(default_factory=list, max_length=100)

    @field_validator("title", "setting", "tone")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("companion_ids", "lore_document_ids")
    @classmethod
    def unique_positive_ids(cls, values: list[int]) -> list[int]:
        return list(dict.fromkeys(values))


class CampaignUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    archived: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_empty_or_null_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not value:
            raise ValueError("Provide a campaign title or archive state.")
        null_fields = sorted(key for key, item in value.items() if item is None)
        if null_fields:
            raise ValueError("Campaign fields cannot be null: " + ", ".join(null_fields))
        return value

    @field_validator("title")
    @classmethod
    def strip_optional_title(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class HeroCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    ancestry: str = Field(default="Human", min_length=1, max_length=80)
    character_class: str = Field(default="Fighter", min_length=1, max_length=80)
    backstory: str = Field(
        default="An adventurer looking for a reason to risk everything.",
        min_length=1,
        max_length=4_000,
    )
    inventory: list[Annotated[str, Field(min_length=1, max_length=120)]] = Field(
        default_factory=lambda: ["torch", "rations", "dagger"], max_length=100
    )
    strength: int | None = Field(default=None, ge=3, le=18)
    dexterity: int | None = Field(default=None, ge=3, le=18)
    constitution: int | None = Field(default=None, ge=3, le=18)
    intelligence: int | None = Field(default=None, ge=3, le=18)
    wisdom: int | None = Field(default=None, ge=3, le=18)
    charisma: int | None = Field(default=None, ge=3, le=18)

    @field_validator("name", "ancestry", "character_class", "backstory")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("ancestry")
    @classmethod
    def supported_ancestry(cls, value: str) -> str:
        if value not in ANCESTRIES:
            raise ValueError("Choose a supported ancestry.")
        return value

    @field_validator("character_class")
    @classmethod
    def supported_class(cls, value: str) -> str:
        if value not in CLASSES:
            raise ValueError("Choose a supported class.")
        return value

    @field_validator("inventory")
    @classmethod
    def clean_inventory(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @model_validator(mode="after")
    def complete_ability_scores(self) -> "HeroCreate":
        supplied = [getattr(self, ability) is not None for ability in ABILITIES]
        if any(supplied) and not all(supplied):
            raise ValueError("Provide all six ability scores or omit all of them.")
        return self


class PlayerAction(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Action cannot be empty.")
        return cleaned


class CombatAction(BaseModel):
    action_id: str = Field(min_length=1, max_length=80)
    target_id: int | None = Field(default=None, gt=0)
    destination_lane: Literal["front", "back"] | None = None


class RestRequest(BaseModel):
    kind: Literal["short", "long"]
    hit_dice: int = Field(default=1, ge=0, le=5)


class CampaignOut(ORMModel):
    id: int
    title: str
    setting: str
    tone: str
    milestone_points: int
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None


class CampaignSummaryOut(CampaignOut):
    current_location: str
    last_activity_at: datetime


class CharacterOut(ORMModel):
    id: int
    campaign_id: int
    name: str
    ancestry: str
    character_class: str
    backstory: str
    inventory_json: str
    role: Literal["protagonist", "companion"]
    level: int
    strength: int
    dexterity: int
    constitution: int
    intelligence: int
    wisdom: int
    charisma: int
    max_hp: int
    current_hp: int
    armor_class: int
    speed: int
    skills_json: str
    saves_json: str
    spells_json: str
    resources_json: str
    conditions_json: str


class HeroOut(ORMModel):
    id: int
    name: str
    ancestry: str
    character_class: str
    backstory: str
    inventory_json: str
    level: int
    strength: int
    dexterity: int
    constitution: int
    intelligence: int
    wisdom: int
    charisma: int
    max_hp: int
    armor_class: int
    speed: int
    skills_json: str
    saves_json: str
    spells_json: str
    resources_json: str
    created_at: datetime
    updated_at: datetime


class TurnOut(ORMModel):
    id: int
    campaign_id: int
    speaker: str
    content: str
    created_at: datetime


class WorldStateOut(ORMModel):
    id: int
    campaign_id: int
    current_location: str
    active_objective: str
    scene_summary: str
    choices_json: str
    facts_json: str
    npcs_json: str
    updated_at: datetime


class PendingRollOut(BaseModel):
    id: int
    campaign_id: int
    action_text: str
    formula: str
    ability: str
    skill: str | None
    dc: int
    reason: str
    narration: str
    status: Literal["pending", "resolving", "resolved", "cancelled", "failed"]


class DiceRollOut(BaseModel):
    id: int
    pending_roll_id: int | None
    formula: str
    rolls: list[int]
    modifier: int
    total: int
    dc: int | None
    outcome: str
    reason: str


class RecoverableOperationOut(BaseModel):
    kind: Literal["action", "roll_resolution", "combat_action"]
    idempotency_key: str
    action_text: str
    error: str
    pending_roll: PendingRollOut | None
    dice_roll: DiceRollOut | None


class CampaignOpeningOut(BaseModel):
    status: Literal["needed", "processing", "failed", "complete"]
    idempotency_key: str | None
    error: str | None
    retryable: bool


class TurnPageOut(BaseModel):
    has_more: bool
    next_cursor: str | None
    limit: int


class CampaignDetailOut(BaseModel):
    campaign: CampaignOut
    characters: list[CharacterOut]
    turns: list[TurnOut]
    world_state: WorldStateOut
    choices: list[str]
    pending_roll: PendingRollOut | None
    recoverable_operation: RecoverableOperationOut | None
    opening: CampaignOpeningOut
    encounter: dict[str, object] | None
    quests: list[dict[str, object]]
    turn_page: TurnPageOut
    last_roll: DiceRollOut | None


class LoreDocumentOut(ORMModel):
    id: int
    filename: str
    status: Literal["queued", "indexing", "ready", "error", "deleting"]
    chunks: int
    size_bytes: int
    page_count: int
    attempts: int
    embed_model: str
    created_at: datetime
    updated_at: datetime
    error: str | None


class HealthComponent(BaseModel):
    status: Literal["ok", "degraded", "error"]
    detail: str | None = None


class WorkerHealthOut(HealthComponent):
    running: bool
    active_document_id: int | None
    queued_count: int


class ModelRuntimeComponent(BaseModel):
    status: Literal["unverified", "healthy", "failed"]
    detail: str | None = None
    updated_at: datetime | None = None


class ModelRuntimeOut(BaseModel):
    narrator: ModelRuntimeComponent
    utility: ModelRuntimeComponent
    embeddings: ModelRuntimeComponent


class HealthOut(BaseModel):
    status: Literal["ok", "degraded", "error"]
    database: HealthComponent
    filesystem: HealthComponent
    worker: WorkerHealthOut
    ollama: HealthComponent
    model_runtime: ModelRuntimeOut
    chat_model: str
    utility_model: str
    embed_model: str
    request_max_chars: int


class ModelOptionOut(BaseModel):
    name: str
    capabilities: list[str]


class ModelSettingsUpdate(BaseModel):
    chat_model: str
    utility_model: str
    embed_model: str
    narration_style: Literal["focused", "balanced", "cinematic"]
    confirm_lore_reindex: bool = False

    @field_validator("chat_model", "utility_model", "embed_model")
    @classmethod
    def valid_model(cls, value: str) -> str:
        return validate_model_name(value)


class ModelSettingsOut(BaseModel):
    chat_model: str
    utility_model: str
    embed_model: str
    narration_style: Literal["focused", "balanced", "cinematic"]
    models: list[ModelOptionOut]
    model_runtime: ModelRuntimeOut
    lore_document_count: int
    reindex_queued: int = 0
