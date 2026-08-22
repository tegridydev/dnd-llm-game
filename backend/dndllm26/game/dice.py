from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Protocol

ROLL_RE = re.compile(r"^\s*(?:(\d{1,2})?d)?(\d{1,4})([+-]\d{1,3})?\s*$", re.IGNORECASE)


class RandomSource(Protocol):
    def randint(self, a: int, b: int) -> int: ...


@dataclass(frozen=True, slots=True)
class RollResult:
    formula: str
    rolls: list[int]
    modifier: int
    total: int


def normalize_formula(formula: str) -> str:
    compact = re.sub(r"\s+", "", formula)
    match = ROLL_RE.match(compact)
    if not match:
        raise ValueError("Use dice notation like d20, 1d20+3, or 2d6.")
    count = int(match.group(1) or "1")
    sides = int(match.group(2))
    modifier = int(match.group(3) or "0")
    if not 1 <= count <= 20:
        raise ValueError("Roll count must be between 1 and 20.")
    if not 2 <= sides <= 1_000:
        raise ValueError("Dice sides must be between 2 and 1000.")
    if not -100 <= modifier <= 100:
        raise ValueError("Dice modifier must be between -100 and 100.")
    modifier_text = f"{modifier:+d}" if modifier else ""
    return f"{count}d{sides}{modifier_text}"


def roll_formula(formula: str, *, random_source: RandomSource | None = None) -> RollResult:
    normalized = normalize_formula(formula)
    match = ROLL_RE.match(normalized)
    if not match:  # pragma: no cover - normalize_formula guarantees this
        raise ValueError("Invalid dice formula.")
    source = random_source or random.SystemRandom()
    count = int(match.group(1) or "1")
    sides = int(match.group(2))
    modifier = int(match.group(3) or "0")
    rolls = [source.randint(1, sides) for _ in range(count)]
    return RollResult(
        formula=normalized,
        rolls=rolls,
        modifier=modifier,
        total=sum(rolls) + modifier,
    )


def outcome_for(total: int, dc: int | None) -> str:
    if dc is None:
        return "rolled"
    return "success" if total >= dc else "failure"
