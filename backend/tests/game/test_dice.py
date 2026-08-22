from __future__ import annotations

import pytest

from dndllm26.game.dice import normalize_formula, outcome_for, roll_formula


class FixedRandom:
    def __init__(self, values: list[int]) -> None:
        self.values = iter(values)

    def randint(self, _minimum: int, _maximum: int) -> int:
        return next(self.values)


def test_normalize_formula() -> None:
    assert normalize_formula(" d20 + 3 ") == "1d20+3"
    assert normalize_formula("2d6-1") == "2d6-1"


@pytest.mark.parametrize("formula", ["", "0d20", "21d6", "1d1", "1d20+101", "drop table"])
def test_invalid_formulas_are_rejected(formula: str) -> None:
    with pytest.raises(ValueError):
        normalize_formula(formula)


def test_roll_formula_supports_deterministic_sources() -> None:
    result = roll_formula("2d6+2", random_source=FixedRandom([3, 5]))
    assert result.rolls == [3, 5]
    assert result.total == 10
    assert outcome_for(result.total, 10) == "success"
