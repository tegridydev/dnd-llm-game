from __future__ import annotations

import logging
import re

from dndllm26.core.errors import ModelResponseError, ModelUnavailableError
from dndllm26.game.prompts import build_world_update_messages, safe_second_person_narration
from dndllm26.game.schemas import WorldUpdateOutput
from dndllm26.game.combat import normalise_npc_identity
from dndllm26.game.text import (
    clean_choice_list,
    clean_extracted_text,
    extract_choices,
    fallback_summary,
)
from dndllm26.game.types import WorldSnapshot, WorldUpdate
from dndllm26.llm.ollama_client import OllamaService

logger = logging.getLogger(__name__)


def _merge_memory(
    existing: tuple[str, ...], extracted: list[str], *, limit: int = 20
) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for raw in (*existing, *extracted):
        value = clean_extracted_text(raw, "", limit=160, default="")
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            merged.append(value)
        if len(merged) == limit:
            break
    return tuple(merged)


def _remove_resolved(existing: tuple[str, ...], resolved: list[str]) -> tuple[str, ...]:
    removed = {item.strip().casefold() for item in resolved if item.strip()}
    return tuple(item for item in existing if item.casefold() not in removed)


def _matches_party(value: str, party_names: tuple[str, ...]) -> bool:
    identity = normalise_npc_identity(value)
    aliases = {
        alias.casefold()
        for name in party_names
        for alias in (name.strip(), name.split()[0].strip() if name.strip() else "")
        if len(alias) >= 3
    }
    return any(re.search(rf"\b{re.escape(alias)}\b", identity, re.I) for alias in aliases)


def _merge_npcs(
    existing: tuple[str, ...], extracted: list[str], party_names: tuple[str, ...] = ()
) -> tuple[str, ...]:
    merged: list[str] = []
    positions: dict[str, int] = {}
    for raw in existing:
        value = clean_extracted_text(raw, "", limit=160, default="")
        key = normalise_npc_identity(value).casefold()
        if value and key and key not in positions:
            positions[key] = len(merged)
            merged.append(value)
    for raw in extracted:
        value = clean_extracted_text(raw, "", limit=160, default="")
        if not value or _matches_party(value, party_names):
            continue
        key = normalise_npc_identity(value).casefold()
        if not key:
            continue
        match = positions.get(key)
        if match is None:
            positions[key] = len(merged)
            merged.append(value)
        else:
            merged[match] = value
    return tuple(merged[:20])


async def analyse_world_update(
    ollama: OllamaService,
    dm_text: str,
    current: WorldSnapshot,
    *,
    action: str = "",
    authoritative_events: tuple[str, ...] = (),
    bootstrap: bool = False,
    protagonist: str = "",
    party_names: tuple[str, ...] = (),
) -> WorldUpdate:
    parsed_choices = extract_choices(dm_text)
    fallback_choices = (
        parsed_choices
        or list(current.choices)
        or [
            "Ask a follow-up question.",
            "Inspect the immediate area.",
            "Move carefully onward.",
        ]
    )
    fallback = WorldUpdateOutput(
        location=current.current_location or "Unknown location",
        objective=current.active_objective or "Choose the next move.",
        summary=fallback_summary(dm_text) or current.scene_summary or "The scene is unfolding.",
        choices=fallback_choices,
        facts=list(current.facts),
        npcs=list(current.npcs),
    )
    system, user = build_world_update_messages(
        dm_text,
        current,
        action=action,
        authoritative_events=authoritative_events,
        protagonist=protagonist,
        party_names=party_names,
    )
    try:
        extracted = await ollama.chat_structured(
            system,
            user,
            WorldUpdateOutput,
            model=ollama.utility_model,
            temperature=0.0,
            num_predict=300,
        )
    except (ModelUnavailableError, ModelResponseError) as exc:
        logger.warning("Utility world-state analysis fell back to deterministic parsing: %s", exc)
        extracted = fallback

    choices = clean_choice_list(
        [choice for choice in extracted.choices if not _matches_party(choice, party_names)]
    ) or clean_choice_list(
        [choice for choice in fallback.choices if not _matches_party(choice, party_names)]
    )
    if not choices:
        choices = ["Assess the immediate situation.", "Proceed cautiously."]
    summary = clean_extracted_text(
        extracted.summary,
        fallback.summary,
        limit=260,
        default="The scene is unfolding.",
    )
    if protagonist:
        summary = safe_second_person_narration(summary, protagonist)
    return WorldUpdate(
        location=clean_extracted_text(
            extracted.location
            if bootstrap or extracted.location_changed
            else current.current_location,
            current.current_location,
            limit=120,
            default="Unknown location",
        ),
        objective=clean_extracted_text(
            extracted.objective
            if bootstrap or extracted.objective_changed
            else current.active_objective,
            current.active_objective,
            limit=180,
            default="Choose the next move.",
        ),
        summary=summary,
        choices=tuple(choices[:4]),
        facts=_merge_memory(
            _remove_resolved(current.facts, extracted.resolved_facts), extracted.facts
        ),
        npcs=_merge_npcs(current.npcs, extracted.npcs, party_names),
    )
