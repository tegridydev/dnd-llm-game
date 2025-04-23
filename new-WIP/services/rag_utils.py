import json
import logging
import re
from typing import Dict, List

from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_fixed

from core.utils import retrieve, last_sentences
from services.ollama_client import ollama_client
from core.settings import settings

logger = logging.getLogger(__name__)

# ——— Character schema ——————————————————————————————————

class Character(BaseModel):
    name: str
    race: str
    class_: str = Field(..., alias="class")
    backstory: str
    items: List[str]
    personality: str

    model_config = {"populate_by_name": True}

# ——— JSON extraction ——————————————————————————————————

def _extract_json(raw: str) -> str:
    # Strip fences (case-insensitive), then grab first {...} non-greedily
    cleaned = re.sub(r"```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    m = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
    return m.group(0) if m else cleaned

# ——— Generation params ——————————————————————————————————

CHAR_PROMPT = (
    "SYSTEM: You are a D&D character creator. "
    "Output exactly one JSON object with keys: name, race, class, backstory, items, personality.\n"
    "USER: Generate one unique character."
)
CHAR_MAX = 200
CHAR_TEMP = 0.7

DM_INTRO_PROMPT = (
    "SYSTEM: You are the Dungeon Master. "
    "Describe a scene (200–300 words) and end with a clear challenge.\n"
    "USER: Start an epic adventure with: {names}."
)
DM_TURN_PROMPT = (
    "SYSTEM: You are the Dungeon Master. Continue the narrative (150–250 words), "
    "summarizing what happened and presenting the next challenge.\n"
    "USER: {context}"
)
DM_MAX = 300
DM_TEMP = 0.8

PLAYER_PROMPT = (
    "SYSTEM: You are a player character. Stay in character, describe only your single action (100–150 words).\n"
    "USER: {context}"
)
PLAYER_MAX = 150
PLAYER_TEMP = 0.6

OPTIONS_PROMPT = (
    "SYSTEM: You are the Dungeon Master. Given recent events, output exactly 3 possible action options as a JSON array.\n"
    "USER: {context}"
)
OPTIONS_MAX = 150
OPTIONS_TEMP = 0.6

# ——— Character generation with retry ——————————————————————

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
def generate_character_sync() -> Character:
    resp = ollama_client.generate(
        prompt=CHAR_PROMPT,
        max_tokens=CHAR_MAX,
        temperature=CHAR_TEMP
    )
    raw = getattr(resp, "response", "") or ""
    js = _extract_json(raw)
    try:
        data = json.loads(js)
        return Character.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning("Parse error (retrying): %s\nRaw: %s", e, raw)
        raise

def generate_party_sync() -> Dict[str, Character]:
    return {f"Player {i+1}": generate_character_sync() for i in range(4)}

def start_adventure_sync(party: Dict[str, Character]) -> str:
    names = ", ".join(party.keys())
    prompt = DM_INTRO_PROMPT.format(names=names)
    resp = ollama_client.generate(
        prompt=prompt,
        max_tokens=DM_MAX,
        temperature=DM_TEMP
    )
    return getattr(resp, "response", "").strip()

def player_turn_sync(state: Dict, name: str, info: Character) -> str:
    recent = last_sentences(" ".join(state["story"]), 3)
    lore  = retrieve(info.backstory + " " + recent)
    ctxt  = f"Character: {info.model_dump_json()}\nRecent: {recent}\nLore: {' | '.join(lore)}"
    prompt=PLAYER_PROMPT.format(context=ctxt)
    resp  = ollama_client.generate(prompt=prompt, max_tokens=PLAYER_MAX, temperature=PLAYER_TEMP)
    return getattr(resp, "response", "").strip()

def dm_turn_sync(state: Dict) -> str:
    recent = last_sentences(" ".join(state["story"]), 5)
    lore   = retrieve(recent)
    ctxt   = f"Recent events: {recent}\nLore: {' | '.join(lore)}"
    prompt = DM_TURN_PROMPT.format(context=ctxt)
    resp   = ollama_client.generate(prompt=prompt, max_tokens=DM_MAX, temperature=DM_TEMP)
    return getattr(resp, "response", "").strip()

def generate_options_sync(state: Dict) -> List[str]:
    recent = last_sentences(" ".join(state["story"]), 3)
    ctxt   = f"Recent events: {recent}"
    prompt = OPTIONS_PROMPT.format(context=ctxt)
    resp   = ollama_client.generate(prompt=prompt, max_tokens=OPTIONS_MAX, temperature=OPTIONS_TEMP)
    raw    = getattr(resp, "response", "") or ""
    js     = _extract_json(raw)
    try:
        opts = json.loads(js)
        if isinstance(opts, list) and all(isinstance(o, str) for o in opts):
            return opts
    except Exception:
        logger.error("Options parse error, raw: %s", raw)
    return ["Continue forward", "Inspect surroundings", "Rest and recover"]
