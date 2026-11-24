from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing
import logging

from dndllm26.core.errors import ModelResponseError, ModelUnavailableError
from dndllm26.game.prompts import (
    build_identity_retry_messages,
    protagonist_identity_violations,
    safe_second_person_narration,
)
from dndllm26.game.text import MAX_DM_CHARS, trim_dm_text
from dndllm26.llm.client import ModelService

logger = logging.getLogger(__name__)


async def bounded_narration(
    models: ModelService, system: str, prompt: str, *, narration_style: str
) -> AsyncIterator[str]:
    remaining = MAX_DM_CHARS
    async with aclosing(
        models.stream_dm(system, prompt, narration_style=narration_style)
    ) as stream:
        async for token in stream:
            fragment = token[:remaining]
            if fragment:
                remaining -= len(fragment)
                yield fragment
            if remaining == 0:
                return


async def collect_dm_narration(
    models: ModelService, system: str, prompt: str, *, narration_style: str
) -> str:
    return trim_dm_text(
        "".join(
            [
                fragment
                async for fragment in bounded_narration(
                    models, system, prompt, narration_style=narration_style
                )
            ]
        )
    )


async def ensure_second_person(
    models: ModelService,
    text: str,
    protagonist: str,
) -> str:
    violations = protagonist_identity_violations(text, protagonist)
    if not violations:
        return text
    system, prompt = build_identity_retry_messages(text, protagonist)
    try:
        retry = await collect_dm_narration(
            models,
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
