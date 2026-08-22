from __future__ import annotations

import logging

from dndllm26.core.errors import ModelResponseError, ModelUnavailableError
from dndllm26.game.prompts import (
    build_identity_retry_messages,
    protagonist_identity_violations,
    safe_second_person_narration,
)
from dndllm26.game.text import MAX_DM_CHARS, trim_dm_text
from dndllm26.llm.ollama_client import OllamaService

logger = logging.getLogger(__name__)


async def collect_dm_narration(
    ollama: OllamaService,
    system: str,
    prompt: str,
    *,
    narration_style: str,
) -> str:
    parts: list[str] = []
    character_count = 0
    async for token in ollama.stream_dm(system, prompt, narration_style=narration_style):
        if character_count >= MAX_DM_CHARS:
            break
        fragment = token[: MAX_DM_CHARS - character_count]
        if fragment:
            parts.append(fragment)
            character_count += len(fragment)
    return trim_dm_text("".join(parts))


async def ensure_second_person(
    ollama: OllamaService,
    text: str,
    protagonist: str,
) -> str:
    violations = protagonist_identity_violations(text, protagonist)
    if not violations:
        return text
    system, prompt = build_identity_retry_messages(text, protagonist)
    try:
        retry = await collect_dm_narration(
            ollama,
            system,
            prompt,
            narration_style="focused",
        )
    except (ModelUnavailableError, ModelResponseError):
        logger.warning("Identity rewrite failed; applying deterministic second-person repair")
        retry = ""
    if retry and not protagonist_identity_violations(retry, protagonist):
        return retry
    return safe_second_person_narration(retry or text, protagonist)
