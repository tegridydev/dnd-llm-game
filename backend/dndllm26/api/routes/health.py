from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import APIRouter, Depends, Response
from sqlalchemy import text
from sqlmodel import Session

from dndllm26.api.deps import get_resources, get_session
from dndllm26.api.schemas import HealthComponent, HealthOut, WorkerHealthOut
from dndllm26.core.errors import ModelUnavailableError
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
            resources.settings.resolved_lancedb_dir,
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
    models: list[str] = []
    try:
        models = await asyncio.wait_for(
            resources.ollama.list_models(),
            timeout=resources.settings.ollama_health_timeout_seconds,
        )
        configured = {
            resources.ollama.chat_model,
            resources.ollama.utility_model,
            resources.ollama.embed_model,
        }
        missing = sorted(model for model in configured if model not in models)
        ollama = HealthComponent(
            status="degraded" if missing else "ok",
            detail=("Missing configured models: " + ", ".join(missing)) if missing else None,
        )
    except (TimeoutError, ModelUnavailableError) as exc:
        ollama = HealthComponent(status="degraded", detail=str(exc)[:300])

    core_error = any(component.status == "error" for component in (database, filesystem, worker))
    degraded = ollama.status != "ok"
    model_runtime = resources.ollama.runtime_status()
    runtime_failed = any(item["status"] == "failed" for item in model_runtime.values())
    degraded = degraded or runtime_failed
    status = "error" if core_error else "degraded" if degraded else "ok"
    return HealthOut(
        status=status,
        database=database,
        filesystem=filesystem,
        worker=worker,
        ollama=ollama,
        model_runtime=model_runtime,
        chat_model=resources.ollama.chat_model,
        utility_model=resources.ollama.utility_model,
        embed_model=resources.ollama.embed_model,
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
