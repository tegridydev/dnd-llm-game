from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import CheckConstraint, Index, UniqueConstraint, text
from sqlmodel import Field, SQLModel


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Campaign(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    title: str = Field(min_length=1, max_length=120)
    setting: str = Field(
        default="A dangerous frontier realm full of ruins, factions, and secrets.",
        max_length=2_000,
    )
    tone: str = Field(default="heroic fantasy", max_length=160)
    milestone_points: int = Field(default=0, ge=0, le=20)
    created_at: datetime = Field(default_factory=now_utc, index=True)
    updated_at: datetime = Field(default_factory=now_utc, index=True)
    archived_at: datetime | None = Field(default=None, index=True)


class CampaignOpening(SQLModel, table=True):
    __table_args__ = (
        CheckConstraint(
            "status IN ('needed', 'processing', 'failed', 'complete')",
            name="ck_campaign_opening_status",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    campaign_id: int = Field(unique=True, index=True, foreign_key="campaign.id")
    status: str = Field(default="needed", max_length=20, index=True)
    idempotency_key: str | None = Field(default=None, max_length=128, unique=True)
    error_code: str | None = Field(default=None, max_length=80)
    error_message: str | None = Field(default=None, max_length=500)
    created_at: datetime = Field(default_factory=now_utc, index=True)
    updated_at: datetime = Field(default_factory=now_utc, index=True)


class RuntimePreference(SQLModel, table=True):
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_runtime_preference_singleton"),
        CheckConstraint(
            "narration_style IN ('focused', 'balanced', 'cinematic')",
            name="ck_runtime_preference_narration_style",
        ),
    )

    id: int = Field(default=1, primary_key=True)
    chat_model: str = Field(max_length=160)
    utility_model: str = Field(max_length=160)
    embed_model: str = Field(max_length=160)
    narration_style: str = Field(default="balanced", max_length=20)
    updated_at: datetime = Field(default_factory=now_utc)


class Character(SQLModel, table=True):
    __table_args__ = (
        CheckConstraint("role IN ('protagonist', 'companion')", name="ck_character_role"),
        Index(
            "uq_character_campaign_protagonist",
            "campaign_id",
            unique=True,
            sqlite_where=text("role = 'protagonist'"),
        ),
    )
    id: int | None = Field(default=None, primary_key=True)
    campaign_id: int = Field(index=True, foreign_key="campaign.id")
    name: str = Field(min_length=1, max_length=100)
    ancestry: str = Field(default="Human", max_length=80)
    character_class: str = Field(default="Fighter", max_length=80)
    backstory: str = Field(
        default="An adventurer looking for a reason to risk everything.",
        max_length=4_000,
    )
    inventory_json: str = "[]"
    role: str = Field(default="companion", max_length=20, index=True)
    level: int = Field(default=1, ge=1, le=5)
    strength: int = Field(default=10, ge=3, le=18)
    dexterity: int = Field(default=10, ge=3, le=18)
    constitution: int = Field(default=10, ge=3, le=18)
    intelligence: int = Field(default=10, ge=3, le=18)
    wisdom: int = Field(default=10, ge=3, le=18)
    charisma: int = Field(default=10, ge=3, le=18)
    max_hp: int = Field(default=8, ge=1, le=200)
    current_hp: int = Field(default=8, ge=0, le=200)
    armor_class: int = Field(default=10, ge=1, le=30)
    speed: int = Field(default=30, ge=15, le=50)
    skills_json: str = "[]"
    saves_json: str = "[]"
    spells_json: str = "[]"
    resources_json: str = "{}"
    conditions_json: str = "[]"


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(min_length=1, max_length=100)
    ancestry: str = Field(default="Human", max_length=80)
    character_class: str = Field(default="Fighter", max_length=80)
    backstory: str = Field(
        default="An adventurer looking for a reason to risk everything.",
        max_length=4_000,
    )
    inventory_json: str = "[]"
    level: int = Field(default=1, ge=1, le=5)
    strength: int = Field(default=10, ge=3, le=18)
    dexterity: int = Field(default=10, ge=3, le=18)
    constitution: int = Field(default=10, ge=3, le=18)
    intelligence: int = Field(default=10, ge=3, le=18)
    wisdom: int = Field(default=10, ge=3, le=18)
    charisma: int = Field(default=10, ge=3, le=18)
    max_hp: int = Field(default=8, ge=1, le=200)
    armor_class: int = Field(default=10, ge=1, le=30)
    speed: int = Field(default=30, ge=15, le=50)
    skills_json: str = "[]"
    saves_json: str = "[]"
    spells_json: str = "[]"
    resources_json: str = "{}"
    created_at: datetime = Field(default_factory=now_utc, index=True)
    updated_at: datetime = Field(default_factory=now_utc, index=True)


class ActionRequest(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint(
            "campaign_id",
            "idempotency_key",
            "operation",
            name="uq_action_request_idempotency",
        ),
        CheckConstraint(
            "operation IN ('action', 'roll_resolution', 'combat_action')",
            name="ck_action_request_operation",
        ),
        CheckConstraint(
            "status IN ('processing', 'roll_required', 'complete', 'failed', 'interrupted')",
            name="ck_action_request_status",
        ),
        CheckConstraint("json_valid(request_payload_json)", name="ck_action_request_payload_json"),
        Index(
            "uq_action_request_active_campaign",
            "campaign_id",
            unique=True,
            sqlite_where=text("status = 'processing'"),
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    campaign_id: int = Field(index=True, foreign_key="campaign.id")
    idempotency_key: str = Field(min_length=8, max_length=128, index=True)
    operation: str = Field(default="action", max_length=32)
    status: str = Field(default="processing", max_length=32, index=True)
    action_text: str = Field(default="", max_length=20_000)
    request_payload_json: str = Field(default="{}", max_length=20_000)
    pending_roll_id: int | None = Field(default=None, index=True, foreign_key="pendingroll.id")
    dice_roll_id: int | None = Field(default=None, index=True, foreign_key="diceroll.id")
    player_turn_id: int | None = Field(default=None, index=True, foreign_key="turn.id")
    dm_turn_id: int | None = Field(default=None, index=True, foreign_key="turn.id")
    error_code: str | None = Field(default=None, max_length=80)
    error_message: str | None = Field(default=None, max_length=500)
    retryable: bool = False
    created_at: datetime = Field(default_factory=now_utc, index=True)
    updated_at: datetime = Field(default_factory=now_utc, index=True)
    completed_at: datetime | None = None


class Turn(SQLModel, table=True):
    __table_args__ = (
        CheckConstraint(
            "speaker IN ('System', 'Player', 'DM', 'Roll')",
            name="ck_turn_speaker",
        ),
        Index("ix_turn_campaign_created_id", "campaign_id", "created_at", "id"),
    )

    id: int | None = Field(default=None, primary_key=True)
    campaign_id: int = Field(index=True, foreign_key="campaign.id")
    action_request_id: int | None = Field(default=None, index=True, foreign_key="actionrequest.id")
    speaker: str = Field(max_length=16)
    content: str = Field(max_length=20_000)
    created_at: datetime = Field(default_factory=now_utc, index=True)


class LoreDocument(SQLModel, table=True):
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'indexing', 'ready', 'error', 'deleting')",
            name="ck_lore_document_status",
        ),
        CheckConstraint("length(content_sha256) = 64", name="ck_lore_document_sha256"),
        UniqueConstraint("content_sha256", name="uq_lore_document_content_sha256"),
    )

    id: int | None = Field(default=None, primary_key=True)
    filename: str = Field(max_length=255)  # Safe display/original name.
    storage_name: str = Field(default="", max_length=96, unique=True, index=True)
    content_sha256: str = Field(max_length=64, index=True)
    size_bytes: int = Field(default=0, ge=0)
    page_count: int = Field(default=0, ge=0)
    status: str = Field(default="queued", max_length=20, index=True)
    chunks: int = Field(default=0, ge=0)
    attempts: int = Field(default=0, ge=0)
    embed_model: str = Field(default="", max_length=160)
    embed_dimension: int = Field(default=0, ge=0)
    index_version: int = Field(default=1, ge=1)
    index_table: str = Field(default="", max_length=160)
    index_key: str = Field(default="", max_length=240, index=True)
    created_at: datetime = Field(default_factory=now_utc, index=True)
    updated_at: datetime = Field(default_factory=now_utc, index=True)
    error: str | None = Field(default=None, max_length=1_000)


class CampaignLore(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("campaign_id", "lore_document_id", name="uq_campaign_lore_pair"),
    )

    id: int | None = Field(default=None, primary_key=True)
    campaign_id: int = Field(index=True, foreign_key="campaign.id")
    lore_document_id: int = Field(index=True, foreign_key="loredocument.id")


class GameEvent(SQLModel, table=True):
    __table_args__ = (Index("ix_game_event_campaign_created", "campaign_id", "created_at"),)

    id: int | None = Field(default=None, primary_key=True)
    campaign_id: int = Field(index=True, foreign_key="campaign.id")
    action_request_id: int | None = Field(default=None, index=True, foreign_key="actionrequest.id")
    event_type: str = Field(index=True, max_length=80)
    payload_json: str = "{}"
    created_at: datetime = Field(default_factory=now_utc, index=True)


class PendingRoll(SQLModel, table=True):
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'resolving', 'resolved', 'cancelled', 'failed')",
            name="ck_pending_roll_status",
        ),
        CheckConstraint("dc >= 1 AND dc <= 40", name="ck_pending_roll_dc"),
        Index(
            "uq_pending_roll_active_campaign",
            "campaign_id",
            unique=True,
            sqlite_where=text("status IN ('pending', 'resolving')"),
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    campaign_id: int = Field(index=True, foreign_key="campaign.id")
    action_request_id: int | None = Field(default=None, index=True, foreign_key="actionrequest.id")
    action_text: str = Field(max_length=20_000)
    formula: str = Field(default="1d20", max_length=32)
    ability: str = Field(default="Ability", max_length=80)
    skill: str | None = Field(default=None, max_length=80)
    dc: int = 10
    reason: str = Field(max_length=500)
    narration: str = Field(default="", max_length=500)
    purpose: str = Field(default="ability_check", max_length=40)
    context_json: str = "{}"
    status: str = Field(default="pending", max_length=20, index=True)
    created_at: datetime = Field(default_factory=now_utc, index=True)
    resolved_at: datetime | None = None


class DiceRoll(SQLModel, table=True):
    __table_args__ = (
        CheckConstraint("dc IS NULL OR (dc >= 1 AND dc <= 40)", name="ck_dice_roll_dc"),
        CheckConstraint(
            "outcome IN ('rolled', 'success', 'failure')",
            name="ck_dice_roll_outcome",
        ),
        Index(
            "uq_dice_roll_pending",
            "pending_roll_id",
            unique=True,
            sqlite_where=text("pending_roll_id IS NOT NULL"),
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    campaign_id: int = Field(index=True, foreign_key="campaign.id")
    pending_roll_id: int | None = Field(default=None, index=True, foreign_key="pendingroll.id")
    formula: str = Field(max_length=32)
    rolls_json: str
    modifier: int = 0
    total: int
    dc: int | None = None
    outcome: str = Field(default="rolled", max_length=20)
    reason: str = Field(default="", max_length=500)
    created_at: datetime = Field(default_factory=now_utc, index=True)


class WorldState(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    campaign_id: int = Field(index=True, foreign_key="campaign.id", unique=True)
    current_location: str = Field(default="Unknown", max_length=120)
    active_objective: str = Field(default="Find an adventure.", max_length=180)
    scene_summary: str = Field(default="", max_length=500)
    choices_json: str = "[]"
    facts_json: str = "[]"
    npcs_json: str = "[]"
    updated_at: datetime = Field(default_factory=now_utc, index=True)


class Quest(SQLModel, table=True):
    __table_args__ = (
        CheckConstraint("status IN ('active', 'complete', 'failed')", name="ck_quest_status"),
    )

    id: int | None = Field(default=None, primary_key=True)
    campaign_id: int = Field(index=True, foreign_key="campaign.id")
    title: str = Field(max_length=160)
    objective: str = Field(max_length=500)
    status: str = Field(default="active", max_length=20, index=True)
    milestone_reward: int = Field(default=1, ge=0, le=1)
    reward_claimed: bool = False
    created_at: datetime = Field(default_factory=now_utc, index=True)
    updated_at: datetime = Field(default_factory=now_utc, index=True)


class Encounter(SQLModel, table=True):
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'victory', 'defeat', 'fled')", name="ck_encounter_status"
        ),
        Index(
            "uq_encounter_active_campaign",
            "campaign_id",
            unique=True,
            sqlite_where=text("status = 'active'"),
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    campaign_id: int = Field(index=True, foreign_key="campaign.id")
    name: str = Field(default="Encounter", max_length=160)
    status: str = Field(default="active", max_length=20, index=True)
    round_number: int = Field(default=1, ge=1)
    turn_index: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=now_utc, index=True)
    updated_at: datetime = Field(default_factory=now_utc, index=True)


class Combatant(SQLModel, table=True):
    __table_args__ = (
        CheckConstraint("side IN ('party', 'enemy')", name="ck_combatant_side"),
        CheckConstraint("lane IN ('front', 'back')", name="ck_combatant_lane"),
    )

    id: int | None = Field(default=None, primary_key=True)
    encounter_id: int = Field(index=True, foreign_key="encounter.id")
    character_id: int | None = Field(default=None, index=True, foreign_key="character.id")
    name: str = Field(max_length=120)
    side: str = Field(max_length=16, index=True)
    lane: str = Field(default="front", max_length=16)
    initiative: int = Field(default=0, index=True)
    max_hp: int = Field(ge=1, le=500)
    current_hp: int = Field(ge=0, le=500)
    armor_class: int = Field(ge=1, le=30)
    dexterity_modifier: int = Field(default=0, ge=-5, le=10)
    attack_bonus: int = Field(default=2, ge=-5, le=20)
    damage_formula: str = Field(default="1d6", max_length=32)
    save_dc: int = Field(default=10, ge=5, le=25)
    conditions_json: str = "[]"
    resources_json: str = "{}"
    death_successes: int = Field(default=0, ge=0, le=3)
    death_failures: int = Field(default=0, ge=0, le=3)
    defeated: bool = False
