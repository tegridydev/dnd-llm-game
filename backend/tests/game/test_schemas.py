from __future__ import annotations

import pytest
from pydantic import ValidationError

from dndllm26.game.schemas import RollDecisionOutput, WorldUpdateOutput


def test_roll_decision_does_not_coerce_string_booleans() -> None:
    with pytest.raises(ValidationError):
        RollDecisionOutput.model_validate(
            {
                "requires_roll": "false",
                "narration": "",
                "formula": "1d20",
                "ability": "Wisdom",
                "skill": None,
                "dc": 10,
                "reason": "",
            }
        )


def test_world_update_rejects_object_choices() -> None:
    with pytest.raises(ValidationError):
        WorldUpdateOutput.model_validate(
            {
                "location": "Old Gate",
                "objective": "Open the gate",
                "summary": "A gate blocks the road.",
                "choices": [{"action": "Inspect it"}],
            }
        )
