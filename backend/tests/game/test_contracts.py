from __future__ import annotations

import asyncio
from datetime import datetime

import pytest
from pydantic import ValidationError as PydanticValidationError
from sqlmodel import Session

from dndllm26.api.schemas import CampaignCreate, HeroCreate
from dndllm26.db.models import Campaign, Turn
from dndllm26.game.campaigns import list_turns
from dndllm26.game.schemas import WorldUpdateOutput
from dndllm26.game.types import WorldSnapshot
from dndllm26.game.world import analyse_world_update


class EmptyWorldModel:
    utility_model = "test"

    async def chat_structured(self, *_args, **_kwargs) -> WorldUpdateOutput:
        return WorldUpdateOutput(
            location="Gate",
            objective="Wait",
            summary="Nothing changes.",
            choices=["Keep watch."],
            facts=[],
            npcs=[],
        )


class ConfusedWorldModel:
    utility_model = "test"

    async def chat_structured(self, *_args, **_kwargs) -> WorldUpdateOutput:
        return WorldUpdateOutput(
            location="Arena",
            objective="Advance",
            summary="Mira Voss signals to you.",
            choices=["Signal to Mira Voss", "Approach the guard"],
            facts=[],
            npcs=["Mira Voss: hooded scout", "Arena Guard"],
        )


def test_empty_model_memory_cannot_erase_established_entries() -> None:
    current = WorldSnapshot(
        current_location="Gate",
        active_objective="Wait",
        scene_summary="The party waits.",
        choices=("Keep watch.",),
        facts=("The gate is sealed.",),
        npcs=("Mira, the guard",),
    )
    result = asyncio.run(
        analyse_world_update(EmptyWorldModel(), "The party waits.", current)  # type: ignore[arg-type]
    )
    assert result.facts == current.facts
    assert result.npcs == current.npcs


def test_world_update_excludes_party_from_new_npcs_and_choices() -> None:
    current = WorldSnapshot(
        current_location="Arena",
        active_objective="Advance",
        scene_summary="The crowd waits.",
        choices=(),
        facts=(),
        npcs=(),
    )
    result = asyncio.run(
        analyse_world_update(
            ConfusedWorldModel(),  # type: ignore[arg-type]
            "The guard approaches.",
            current,
            protagonist="Mira Voss",
            party_names=("Mira Voss",),
        )
    )
    assert result.npcs == ("Arena Guard",)
    assert result.choices == ("Approach the guard",)
    assert result.summary == "You signal."


def test_gameplay_schema_rejects_unknown_null_partial_and_non_positive_values() -> None:
    with pytest.raises(PydanticValidationError):
        HeroCreate(name="Nox", character_class="Necromancer")
    with pytest.raises(PydanticValidationError):
        HeroCreate(name="Nox", strength=12)
    with pytest.raises(PydanticValidationError):
        HeroCreate.model_validate({"ancestry": None})
    with pytest.raises(PydanticValidationError):
        CampaignCreate(protagonist_id=1, companion_ids=[2, 0])


def test_composite_cursor_pages_equal_timestamps_without_loss(engine) -> None:
    created_at = datetime(2026, 8, 21, 12, 0, 0)
    with Session(engine) as session:
        campaign = Campaign(title="Cursor", setting="Test", tone="Test")
        session.add(campaign)
        session.flush()
        for index in range(5):
            session.add(
                Turn(
                    campaign_id=campaign.id or 0,
                    speaker="DM",
                    content=f"Turn {index}",
                    created_at=created_at,
                )
            )
        session.commit()
        cursor = None
        ids: list[int] = []
        while True:
            rows, has_more, cursor = list_turns(session, campaign.id or 0, limit=2, cursor=cursor)
            ids.extend(int(row.id) for row in rows if row.id is not None)
            if not has_more:
                break
        assert len(ids) == 5
        assert len(set(ids)) == 5
