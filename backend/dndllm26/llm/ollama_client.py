from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from typing import Any, TypeVar

from ollama import AsyncClient
from pydantic import BaseModel, ValidationError

from dndllm26.core.errors import ModelResponseError, ModelUnavailableError
from dndllm26.core.settings import Settings

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


@dataclass(slots=True)
class ModelRoleState:
    status: str = "unverified"
    detail: str | None = None
    updated_at: str | None = None


class OllamaService:
    """Thin, validated adapter around the official asynchronous Ollama client."""

    def __init__(self, settings: Settings) -> None:
        self.chat_model = settings.ollama_chat_model
        self.utility_model = settings.ollama_utility_model or settings.ollama_chat_model
        self.embed_model = settings.ollama_embed_model
        self.host = settings.ollama_host
        self.timeout = settings.ollama_timeout_seconds
        self._client_instance: AsyncClient | None = None
        self._runtime = {
            "narrator": ModelRoleState(),
            "utility": ModelRoleState(),
            "embeddings": ModelRoleState(),
        }

    def configure_models(self, *, chat_model: str, utility_model: str, embed_model: str) -> None:
        changed = {
            "narrator": self.chat_model != chat_model,
            "utility": self.utility_model != utility_model,
            "embeddings": self.embed_model != embed_model,
        }
        self.chat_model = chat_model
        self.utility_model = utility_model
        self.embed_model = embed_model
        for role, did_change in changed.items():
            if did_change:
                self._runtime[role] = ModelRoleState()

    def runtime_status(self) -> dict[str, dict[str, str | None]]:
        return {
            role: {
                "status": state.status,
                "detail": state.detail,
                "updated_at": state.updated_at,
            }
            for role, state in self._runtime.items()
        }

    def _mark_runtime(self, role: str, status: str, detail: str | None = None) -> None:
        self._runtime[role] = ModelRoleState(
            status=status,
            detail=detail[:300] if detail else None,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _client(self) -> AsyncClient:
        if self._client_instance is None:
            self._client_instance = AsyncClient(host=self.host, timeout=self.timeout)
        return self._client_instance

    @staticmethod
    def error_message(exc: BaseException) -> str:
        error = getattr(exc, "error", None)
        status = getattr(exc, "status_code", None)
        if error and status:
            return f"{error} (status {status})"
        if error:
            return str(error)
        message = str(exc).strip()
        return message or exc.__class__.__name__

    def unavailable(self, exc: BaseException) -> ModelUnavailableError:
        return ModelUnavailableError(
            "Ollama could not complete the request. Confirm that Ollama is running and the configured model is available.",
            detail=self.error_message(exc),
        )

    async def close(self) -> None:
        client = self._client_instance
        self._client_instance = None
        if client is None:
            return
        await client.close()

    async def list_models(self) -> list[str]:
        try:
            response = await self._client().list()
        except Exception as exc:
            raise self.unavailable(exc) from exc
        return sorted({model.model for model in response.models if model.model})

    async def model_catalog(self) -> list[dict[str, object]]:
        names = await self.list_models()

        async def inspect_model(name: str) -> dict[str, object]:
            try:
                response = await self._client().show(model=name)
                capabilities = response.capabilities or []
                return {"name": name, "capabilities": sorted({str(item) for item in capabilities})}
            except Exception:
                return {"name": name, "capabilities": []}

        return list(await asyncio.gather(*(inspect_model(name) for name in names)))

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client().embed(model=self.embed_model, input=list(texts))
        except Exception as exc:
            self._mark_runtime("embeddings", "failed", self.error_message(exc))
            raise self.unavailable(exc) from exc
        raw = response.embeddings or []
        embeddings = [[float(value) for value in vector] for vector in raw if vector]
        if len(embeddings) != len(texts):
            self._mark_runtime("embeddings", "failed", "Unexpected vector count")
            raise ModelResponseError(
                "The embedding model returned an unexpected number of vectors.",
                detail=f"expected={len(texts)} received={len(embeddings)}",
            )
        dimensions = {len(vector) for vector in embeddings}
        if not dimensions or len(dimensions) != 1 or 0 in dimensions:
            self._mark_runtime("embeddings", "failed", "Invalid vector dimensions")
            raise ModelResponseError("The embedding model returned invalid vector dimensions.")
        self._mark_runtime("embeddings", "healthy")
        return embeddings

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def stream_dm(
        self, system: str, prompt: str, *, narration_style: str = "balanced"
    ) -> AsyncIterator[str]:
        profiles = {
            "focused": {"temperature": 0.65, "num_predict": 320, "top_p": 0.88},
            "balanced": {"temperature": 0.75, "num_predict": 460, "top_p": 0.9},
            "cinematic": {"temperature": 0.85, "num_predict": 620, "top_p": 0.92},
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            stream = await self._client().chat(
                model=self.chat_model,
                messages=messages,
                stream=True,
                options=profiles.get(narration_style, profiles["balanced"]),
                keep_alive="10m",
                think=False,
            )
            emitted = False
            async for chunk in stream:
                content = chunk.message.content or ""
                if content:
                    if not emitted:
                        self._mark_runtime("narrator", "healthy")
                    emitted = True
                    yield str(content)
        except Exception as exc:
            self._mark_runtime("narrator", "failed", self.error_message(exc))
            raise self.unavailable(exc) from exc
        if not emitted:
            message = f"{self.chat_model} returned no playable narration."
            self._mark_runtime("narrator", "failed", message)
            raise ModelResponseError(
                "The narrator returned no playable text.",
                detail=message,
            )

    async def chat_text(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        num_predict: int = 500,
        format_schema: dict[str, Any] | str | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model or self.utility_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature, "num_predict": num_predict},
            "keep_alive": "10m",
            "think": False,
        }
        if format_schema is not None:
            kwargs["format"] = format_schema
        try:
            response = await self._client().chat(**kwargs)
        except Exception as exc:
            self._mark_runtime("utility", "failed", self.error_message(exc))
            raise self.unavailable(exc) from exc
        content = str(response.message.content or "").strip()
        if not content:
            message = f"{model or self.utility_model} returned an empty response."
            self._mark_runtime("utility", "failed", message)
            raise ModelResponseError("The model returned no usable content.", detail=message)
        self._mark_runtime("utility", "healthy")
        return content

    async def chat_structured(
        self,
        system: str,
        user: str,
        schema: type[StructuredModel],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        num_predict: int = 500,
    ) -> StructuredModel:
        raw = await self.chat_text(
            system,
            user,
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            format_schema=schema.model_json_schema(),
        )
        try:
            return schema.model_validate_json(raw)
        except ValidationError as first_error:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if match:
                try:
                    return schema.model_validate(json.loads(match.group(0)))
                except (ValidationError, json.JSONDecodeError):
                    pass
            raise ModelResponseError(
                "The utility model returned data that did not match the required schema.",
                detail=str(first_error),
            ) from first_error
