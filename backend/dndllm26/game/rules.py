from __future__ import annotations

import logging

from dndllm26.core.errors import ModelResponseError, ModelUnavailableError
from dndllm26.game.prompts import build_roll_decision_messages
from dndllm26.game.schemas import (
    RollDecisionOutput,
    fallback_roll_decision,
    normalise_roll_decision,
)
from dndllm26.game.types import CampaignContext
from dndllm26.llm.ollama_client import OllamaService

logger = logging.getLogger(__name__)


async def decide_roll(
    ollama: OllamaService,
    context: CampaignContext,
    action: str,
) -> RollDecisionOutput:
    fallback = fallback_roll_decision(action, context.world.npcs)
    system, user = build_roll_decision_messages(context, action)
    try:
        decision = await ollama.chat_structured(
            system,
            user,
            RollDecisionOutput,
            model=ollama.utility_model,
            temperature=0.0,
            num_predict=260,
        )
    except (ModelUnavailableError, ModelResponseError) as exc:
        logger.warning("Utility roll decision fell back to deterministic rules: %s", exc)
        decision = fallback
    normalised = normalise_roll_decision(decision, action=action)
    if fallback.encounter_enemies and not normalised.encounter_enemies:
        return normalised.model_copy(
            update={
                "requires_roll": False,
                "encounter_enemies": fallback.encounter_enemies,
                "encounter_names": fallback.encounter_names,
            }
        )
    if normalised.encounter_enemies and not normalised.encounter_names:
        return normalised.model_copy(update={"encounter_names": fallback.encounter_names})
    return normalised
