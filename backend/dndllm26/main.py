from __future__ import annotations

from contextlib import asynccontextmanager, suppress
import logging
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from sqlmodel import Session
from starlette.exceptions import HTTPException as StarletteHTTPException

from dndllm26.api.middleware import (
    LocalOnlyMiddleware,
    RequestContextMiddleware,
    UploadBodyLimitMiddleware,
)
from dndllm26.api.routes import router
from dndllm26.core.errors import AppError
from dndllm26.core.logging import configure_logging
from dndllm26.core.model_preferences import get_or_create_model_preferences
from dndllm26.core.resources import AppResources
from dndllm26.core.settings import Settings
from dndllm26.db.models import Hero  # noqa: F401 - ensures SQLModel metadata is registered
from dndllm26.db.session import initialise_database
from dndllm26.game.campaigns import seed_default_heroes
from dndllm26.game.openings import recover_openings
from dndllm26.game.play import recover_stale_operations
from dndllm26.llm.ollama_client import OllamaService
from dndllm26.rag.store import RagStore
from dndllm26.rag.worker import LoreIndexWorker

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved_settings.log_level)
        engine = initialise_database(resolved_settings)
        ollama = OllamaService(resolved_settings)
        with Session(engine) as session:
            preference = get_or_create_model_preferences(session, resolved_settings)
        ollama.configure_models(
            chat_model=preference.chat_model,
            utility_model=preference.utility_model,
            embed_model=preference.embed_model,
        )
        rag = RagStore(resolved_settings, ollama)
        worker = LoreIndexWorker(resolved_settings, engine, rag)
        resources = AppResources(
            settings=resolved_settings,
            engine=engine,
            ollama=ollama,
            rag=rag,
            lore_worker=worker,
            narration_style=preference.narration_style,
        )
        app.state.resources = resources
        with Session(engine) as session:
            seed_default_heroes(session)
        recovered = recover_stale_operations(engine)
        if recovered:
            logger.warning(
                "Recovered interrupted gameplay operations", extra={"duration_ms": recovered}
            )
        recovered_openings = recover_openings(engine)
        if recovered_openings:
            logger.warning(
                "Recovered interrupted campaign openings",
                extra={"duration_ms": recovered_openings},
            )
        await worker.start()
        try:
            yield
        finally:
            with suppress(Exception):
                await worker.stop()
            with suppress(Exception):
                await ollama.close()
            engine.dispose()
            app.state.resources = None

    app = FastAPI(
        title="DNDLLM26",
        version="0.3.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = resolved_settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Idempotency-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
        max_age=600,
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]", "::1", "testserver"],
    )
    app.add_middleware(LocalOnlyMiddleware, settings=resolved_settings)
    app.add_middleware(UploadBodyLimitMiddleware, settings=resolved_settings)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(router, prefix="/api")

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": exc.code,
                "message": exc.message,
                "retryable": exc.retryable,
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "code": "request_validation_error",
                "message": "The request did not satisfy the API contract.",
                "retryable": False,
                "request_id": _request_id(request),
                "errors": [
                    {
                        "type": error.get("type"),
                        "location": list(error.get("loc", ())),
                        "message": error.get("msg"),
                    }
                    for error in exc.errors()
                ],
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {
            404: "not_found",
            409: "conflict",
            413: "upload_too_large",
            422: "validation_error",
        }
        message = (
            exc.detail if isinstance(exc.detail, str) else "The request could not be completed."
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": codes.get(exc.status_code, "http_error"),
                "message": message,
                "retryable": exc.status_code >= 500,
                "request_id": _request_id(request),
            },
            headers=exc.headers,
        )

    return app


app = create_app()


def run() -> None:
    settings: Settings = app.state.settings
    uvicorn.run(
        "dndllm26.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_config=None,
    )


if __name__ == "__main__":
    run()
