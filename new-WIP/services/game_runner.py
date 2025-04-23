import logging
from typing import Dict

from core.models import GameState
from services.rag_utils import (
    generate_party_sync,
    start_adventure_sync,
    generate_options_sync,
    dm_turn_sync,
)

logger = logging.getLogger(__name__)

class GameRunner:
    def __init__(self):
        self.party: Dict[str, object] | None = None
        self.state: GameState = GameState()

    def new_party(self) -> Dict[str, object]:
        self.party = generate_party_sync()
        self.state = GameState(turn=0, phase="start")
        logger.info("Party generated: %s", list(self.party.keys()))
        return self.party

    def start_adventure(self) -> GameState:
        if not self.party:
            raise RuntimeError("Generate party first.")
        intro = start_adventure_sync(self.party)
        self.state.turn = 1
        self.state.phase = "intro"
        self.state.intro_text = intro
        self.state.story = [intro]
        return self.state

    def request_options(self) -> GameState:
        if self.state.phase not in ("intro", "dm_response"):
            raise RuntimeError("Cannot request options now.")
        opts = generate_options_sync(self.state.__dict__)
        self.state.current_options = opts
        self.state.phase = "choice"
        return self.state

    def process_player_choice(self, idx: int) -> GameState:
        opts = self.state.current_options
        choice = opts[idx]
        self.state.last_choice = choice
        self.state.story.append(f"Player: {choice}")
        self.state.phase = "dm_response"
        return self.state

    def run_dm_turn(self) -> GameState:
        dm_text = dm_turn_sync(self.state.__dict__)
        self.state.story.append(f"DM: {dm_text}")
        self.state.turn += 1
        # stay in dm_response until UI moves back to request_options()
        return self.state
