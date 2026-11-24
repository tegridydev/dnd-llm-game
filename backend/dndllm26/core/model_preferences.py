from __future__ import annotations

from hashlib import sha256

from sqlmodel import Session

from dndllm26.core.settings import Settings
from dndllm26.db.models import ModelProfilePreference, RuntimePreference, now_utc
from dndllm26.llm.providers import configuration


def get_or_create_model_preferences(
    session: Session, settings: Settings
) -> RuntimePreference | ModelProfilePreference:
    config = configuration(settings)
    models = dict(
        chat_model=config.chat_model,
        utility_model=config.utility_model,
        embed_model=config.embed_model,
        narration_style="balanced",
        updated_at=now_utc(),
    )
    if (
        config.chat.identity == "ollama:http://127.0.0.1:11434"
        and config.embeddings
        and config.embeddings.identity == config.chat.identity
    ):
        # Reuse saved model roles from the original local Ollama configuration.
        preference = session.get(RuntimePreference, 1)
        if preference:
            return preference
        preference = RuntimePreference(**models)
    else:
        endpoints = "|".join(
            (config.chat.identity, config.embeddings.identity if config.embeddings else "")
        )
        preference_id = sha256(endpoints.encode()).hexdigest()
        previous = session.get(ModelProfilePreference, preference_id)
        if previous:
            return previous
        preference = ModelProfilePreference(id=preference_id, **models)
    session.add(preference)
    session.commit()
    session.refresh(preference)
    return preference
