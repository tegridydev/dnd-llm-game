from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, Response
from sqlalchemy import text
from sqlmodel import Session

from dndllm26.api.deps import get_resources, get_session
from dndllm26.api.schemas import HealthComponent, HealthOut, WorkerHealthOut
from dndllm26.core.errors import AppError
from dndllm26.api.routes.settings import provider_info
from dndllm26.core.resources import AppResources

router = APIRouter(tags=["health"])


def _filesystem_status(paths: list[Path]) -> HealthComponent:
    unavailable = [str(path) for path in paths if not path.exists() or not os.access(path, os.W_OK)]
    if unavailable:
        return HealthComponent(status="error", detail="Unwritable paths: " + ", ".join(unavailable))
    return HealthComponent(status="ok")


async def _health_payload(resources: AppResources, session: Session) -> HealthOut:
    try:
        session.exec(text("SELECT 1")).one()
        database = HealthComponent(status="ok")
    except Exception as exc:
        database = HealthComponent(status="error", detail=str(exc)[:300])

    filesystem = _filesystem_status(
        [
            resources.settings.database_path.parent
            if resources.settings.database_path
            else resources.settings.resolved_upload_dir,
            resources.settings.resolved_upload_dir,
            resources.settings.resolved_lore_index_dir,
        ]
    )
    worker = WorkerHealthOut(
        status="ok" if resources.lore_worker.running else "error",
        detail=(
            f"active={resources.lore_worker.active_document_id}; "
            f"queued={resources.lore_worker.queued_count}"
        ),
        running=resources.lore_worker.running,
        active_document_id=resources.lore_worker.active_document_id,
        queued_count=resources.lore_worker.queued_count,
    )
    service = resources.models
    try:
        models = await service.list_models()
        missing = service.endpoint.name == "ollama" and service.chat_model not in models
        connection = HealthComponent(
            status="degraded" if missing or not service.chat_model else "ok",
            detail=(
                "Choose a Dungeon Master model in Settings."
                if not service.chat_model
                else f"Model is missing: {service.chat_model}"
                if missing
                else None
            ),
        )
    except AppError as exc:
        connection = HealthComponent(status="degraded", detail=exc.message)
    core_error = any(component.status == "error" for component in (database, filesystem, worker))
    model_runtime = service.runtime_status()
    degraded = connection.status != "ok" or model_runtime["narrator"]["status"] == "failed"
    status = "error" if core_error else "degraded" if degraded else "ok"
    return HealthOut(
        status=status,
        database=database,
        filesystem=filesystem,
        worker=worker,
        connection=connection,
        provider=provider_info(service),
        model_runtime=model_runtime,
        chat_model=resources.models.chat_model,
        utility_model=resources.models.utility_model,
        embed_model=resources.models.embed_model,
        request_max_chars=resources.settings.request_max_chars,
    )


@router.get("/health/ready", response_model=HealthOut)
async def ready(
    response: Response,
    resources: AppResources = Depends(get_resources),
    session: Session = Depends(get_session),
) -> HealthOut:
    payload = await _health_payload(resources, session)
    if payload.status == "error":
        response.status_code = 503
    return payload
