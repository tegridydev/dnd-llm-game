from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from dndllm26.api.deps import get_resources, get_session
from dndllm26.api.schemas import ModelSettingsOut, ModelSettingsUpdate
from dndllm26.core.errors import ConflictError, ValidationError
from dndllm26.core.model_preferences import get_or_create_model_preferences
from dndllm26.core.resources import AppResources
from dndllm26.db.models import ActionRequest, LoreDocument, now_utc

router = APIRouter(prefix="/settings", tags=["settings"])


async def _settings_payload(
    resources: AppResources,
    session: Session,
    *,
    reindex_queued: int = 0,
) -> ModelSettingsOut:
    preference = get_or_create_model_preferences(session, resources.settings)
    documents = session.exec(select(LoreDocument).where(LoreDocument.status != "deleting")).all()
    return ModelSettingsOut(
        chat_model=preference.chat_model,
        utility_model=preference.utility_model,
        embed_model=preference.embed_model,
        narration_style=preference.narration_style,
        models=await resources.ollama.model_catalog(),
        model_runtime=resources.ollama.runtime_status(),
        lore_document_count=len(documents),
        reindex_queued=reindex_queued,
    )


@router.get("/models", response_model=ModelSettingsOut)
async def get_model_settings(
    resources: AppResources = Depends(get_resources),
    session: Session = Depends(get_session),
) -> ModelSettingsOut:
    return await _settings_payload(resources, session)


@router.patch("/models", response_model=ModelSettingsOut)
async def update_model_settings(
    payload: ModelSettingsUpdate,
    resources: AppResources = Depends(get_resources),
    session: Session = Depends(get_session),
) -> ModelSettingsOut:
    active = session.exec(
        select(ActionRequest.id).where(ActionRequest.status == "processing")
    ).first()
    if active is not None:
        raise ConflictError("Wait for the current gameplay operation before changing models.")

    preference = get_or_create_model_preferences(session, resources.settings)
    embed_changed = payload.embed_model != preference.embed_model
    documents = session.exec(select(LoreDocument).where(LoreDocument.status != "deleting")).all()
    if embed_changed and resources.lore_worker.active_document_id is not None:
        raise ConflictError("Wait for lore indexing to finish before changing embedding models.")
    if embed_changed and documents and not payload.confirm_lore_reindex:
        raise ConflictError(
            f"Changing the embedding model requires reindexing {len(documents)} lore documents."
        )

    catalog = await resources.ollama.model_catalog()
    capabilities = {str(item["name"]): set(item["capabilities"]) for item in catalog}
    for name, capability, role in (
        (payload.chat_model, "completion", "Dungeon Master"),
        (payload.utility_model, "completion", "rules and memory"),
        (payload.embed_model, "embedding", "embedding"),
    ):
        if name not in capabilities:
            raise ValidationError(f"The selected {role} model is not installed.")
        if capability not in capabilities[name]:
            raise ValidationError(f"The selected {role} model does not support {capability}.")

    preference.chat_model = payload.chat_model
    preference.utility_model = payload.utility_model
    preference.embed_model = payload.embed_model
    preference.narration_style = payload.narration_style
    preference.updated_at = now_utc()
    session.add(preference)
    if embed_changed:
        for document in documents:
            document.status = "queued"
            document.attempts = 0
            document.error = None
            document.updated_at = now_utc()
            session.add(document)
    session.commit()

    resources.ollama.configure_models(
        chat_model=payload.chat_model,
        utility_model=payload.utility_model,
        embed_model=payload.embed_model,
    )
    resources.narration_style = payload.narration_style
    if embed_changed:
        resources.lore_worker.enqueue_pending()
    return await _settings_payload(
        resources,
        session,
        reindex_queued=len(documents) if embed_changed else 0,
    )
