from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlmodel import Session, select

from dndllm26.api.deps import get_resources, get_session
from dndllm26.api.schemas import ModelSettingsOut, ModelSettingsUpdate, ProviderInfo
from dndllm26.core.errors import AppError, ConflictError, ValidationError
from dndllm26.core.model_preferences import get_or_create_model_preferences
from dndllm26.core.resources import AppResources
from dndllm26.db.models import ActionRequest, CampaignOpening, LoreDocument, now_utc
from dndllm26.llm.client import ModelService

router = APIRouter(prefix="/settings", tags=["settings"])


def provider_info(models: ModelService) -> ProviderInfo:
    embed = models.embedding_endpoint
    return ProviderInfo(
        name=models.endpoint.name,
        label=models.endpoint.label,
        remote=models.endpoint.remote,
        embedding_label=embed.label if embed else None,
        embedding_remote=bool(embed and embed.remote),
    )


async def _settings_payload(
    resources: AppResources, session: Session, *, refresh: bool = False, reindex_queued: int = 0
) -> ModelSettingsOut:
    preference = get_or_create_model_preferences(session, resources.settings)
    values = (
        preference.chat_model,
        preference.utility_model,
        preference.embed_model,
        preference.narration_style,
    )
    count = session.exec(
        select(func.count()).select_from(LoreDocument).where(LoreDocument.status != "deleting")
    ).one()
    session.rollback()

    models = resources.models
    warning = None
    available = []
    embedding_models = []
    try:
        available = await models.model_catalog(refresh=refresh)
    except AppError as exc:
        warning = exc.message
    try:
        embedding_models = (
            available
            if models.embedding_endpoint == models.endpoint
            else await models.model_catalog(embeddings=True, refresh=refresh)
        )
    except AppError as exc:
        warning = warning or exc.message
    return ModelSettingsOut(
        chat_model=values[0],
        utility_model=values[1],
        embed_model=values[2],
        narration_style=values[3],
        models=available,
        embedding_models=embedding_models,
        discovery_error=warning,
        provider=provider_info(models),
        model_runtime=models.runtime_status(),
        lore_document_count=count,
        reindex_queued=reindex_queued,
    )


@router.get("/models", response_model=ModelSettingsOut)
async def get_model_settings(
    refresh: bool = False,
    resources: AppResources = Depends(get_resources),
    session: Session = Depends(get_session),
) -> ModelSettingsOut:
    return await _settings_payload(resources, session, refresh=refresh)


@router.patch("/models", response_model=ModelSettingsOut)
async def update_model_settings(
    payload: ModelSettingsUpdate,
    resources: AppResources = Depends(get_resources),
    session: Session = Depends(get_session),
) -> ModelSettingsOut:
    if not payload.chat_model:
        raise ValidationError("Choose a Dungeon Master model.")
    if payload.embed_model and resources.models.embedding_endpoint is None:
        raise ValidationError("Embeddings are disabled in the server configuration.")
    busy = (
        session.exec(select(ActionRequest.id).where(ActionRequest.status == "processing")).first()
        is not None
        or session.exec(
            select(CampaignOpening.id).where(CampaignOpening.status == "processing")
        ).first()
        is not None
    )
    if busy:
        raise ConflictError("Wait for the current gameplay operation before changing models.")

    preference = get_or_create_model_preferences(session, resources.settings)
    embed_changed = payload.embed_model != preference.embed_model
    documents = []
    if embed_changed:
        if resources.lore_worker.active_document_id is not None:
            raise ConflictError(
                "Wait for lore indexing to finish before changing embedding models."
            )
        documents = session.exec(
            select(LoreDocument).where(LoreDocument.status != "deleting")
        ).all()
        if documents and payload.embed_model and not payload.confirm_lore_reindex:
            raise ConflictError(
                f"Changing the embedding model requires reindexing {len(documents)} lore documents."
            )

    preference.chat_model = payload.chat_model
    preference.utility_model = payload.utility_model or payload.chat_model
    preference.embed_model = payload.embed_model
    preference.narration_style = payload.narration_style
    preference.updated_at = now_utc()
    session.add(preference)
    reindex = len(documents) if embed_changed and payload.embed_model else 0
    if reindex:
        for document in documents:
            document.status = "queued"
            document.attempts = 0
            document.error = None
            document.updated_at = now_utc()
    session.commit()

    resources.models.configure_models(
        chat_model=payload.chat_model,
        utility_model=payload.utility_model or payload.chat_model,
        embed_model=payload.embed_model,
    )
    resources.narration_style = payload.narration_style
    if reindex:
        resources.lore_worker.enqueue_pending()
    return await _settings_payload(resources, session, reindex_queued=reindex)
