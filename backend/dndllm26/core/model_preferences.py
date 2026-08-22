from __future__ import annotations

from sqlmodel import Session

from dndllm26.core.settings import Settings
from dndllm26.db.models import RuntimePreference, now_utc


def get_or_create_model_preferences(session: Session, settings: Settings) -> RuntimePreference:
    preference = session.get(RuntimePreference, 1)
    if preference:
        return preference
    preference = RuntimePreference(
        id=1,
        chat_model=settings.ollama_chat_model,
        utility_model=settings.ollama_utility_model or settings.ollama_chat_model,
        embed_model=settings.ollama_embed_model,
        narration_style="balanced",
        updated_at=now_utc(),
    )
    session.add(preference)
    session.commit()
    session.refresh(preference)
    return preference
