from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Optional

@dataclass
class GameState:
    """
    Tracks the current turn, phase, narrative history, available options,
    and most recent player choice.
    """
    turn: int = 0
    phase: Literal["start", "intro", "choice", "dm_response"] = "start"
    intro_text: Optional[str] = None
    story: List[str] = field(default_factory=list)         # DM and Player lines
    current_options: List[str] = field(default_factory=list)
    last_choice: Optional[str] = None
