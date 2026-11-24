from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from datetime import datetime, timezone
import json
import math
import ssl
from time import monotonic
from typing import Any, TypeVar

import certifi
import httpx2 as httpx
from pydantic import BaseModel, ValidationError

from dndllm26.core.errors import ModelResponseError, ModelUnavailableError
from dndllm26.core.settings import Settings
from dndllm26.llm.providers import Endpoint, configuration, model_id

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)
_ROLES = ("narrator", "utility", "embeddings")
_PROFILES = {"focused": (0.65, 320), "balanced": (0.75, 460), "cinematic": (0.85, 620)}
_MAX_REPLY_BYTES = 16 * 1024 * 1024
_MAX_STREAM_LINE = 256 * 1024


def _object(raw: bytes | str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise ModelResponseError("The endpoint returned invalid JSON.") from exc
    if not isinstance(value, dict):
        raise ModelResponseError("The endpoint returned an unexpected response.")
    if "error" in value or value.get("type") == "error":
        raise ModelUnavailableError("The endpoint rejected the model request.")
    return value


async def _records(response: httpx.Response, *, sse: bool) -> AsyncIterator[str]:
    buffer = bytearray()
    data_lines: list[bytes] = []
    frame_size = 0
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > _MAX_REPLY_BYTES:
            raise ModelResponseError("The model stream exceeded its size limit.")
        buffer.extend(chunk)
        offset = 0
        while (end := buffer.find(b"\n", offset)) >= 0:
            if end - offset > _MAX_STREAM_LINE:
                raise ModelResponseError("The model stream contained an oversized record.")
            line = bytes(buffer[offset:end]).rstrip(b"\r")
            offset = end + 1
            if not sse:
                if line.strip():
                    yield line.decode("utf-8")
                continue
            if not line and data_lines:
                yield b"\n".join(data_lines).decode("utf-8")
                data_lines.clear()
                frame_size = 0
            elif line.startswith(b"data:"):
                frame_size += len(line)
                if frame_size > _MAX_STREAM_LINE:
                    raise ModelResponseError("The model stream contained an oversized event.")
                data_lines.append(line[5:].removeprefix(b" "))
        del buffer[:offset]
        if len(buffer) > _MAX_STREAM_LINE:
            raise ModelResponseError("The model stream contained an oversized record.")
    if not sse and buffer.strip():
        yield buffer.decode("utf-8")


def _assistant_text(protocol: str, data: dict[str, Any], *, stream: bool) -> str:
    if protocol == "ollama":
        content = (data.get("message") or {}).get("content")
    elif protocol == "anthropic":
        if stream:
            delta = data.get("delta") or {}
            content = delta.get("text") if delta.get("type") == "text_delta" else None
        else:
            content = "".join(
                str(block.get("text") or "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            )
    else:
        choices = data.get("choices") or []
        message = choices[0].get("delta" if stream else "message") if choices else None
        content = (message or {}).get("content")
    return content if isinstance(content, str) else ""


def _chat_request(
    endpoint: Endpoint,
    model: str,
    system: str,
    prompt: str,
    *,
    stream: bool,
    temperature: float,
    tokens: int,
    schema: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if schema is not None and endpoint.protocol != "ollama":
        system += "\nReturn only one JSON object matching this schema:\n" + json.dumps(schema)
    user = {"role": "user", "content": prompt}
    payload: dict[str, Any] = {"model": model, "stream": stream}
    if endpoint.protocol == "anthropic":
        return "/messages", {**payload, "system": system, "messages": [user], "max_tokens": tokens}
    payload["messages"] = [{"role": "system", "content": system}, user]
    if endpoint.protocol == "ollama":
        payload.update(
            options={"temperature": temperature, "num_predict": tokens},
            think=False,
            keep_alive="10m",
        )
        if schema is not None:
            payload["format"] = schema
        return "/api/chat", payload
    token_key = "max_completion_tokens" if endpoint.name in {"openai", "azure"} else "max_tokens"
    payload[token_key] = tokens
    if not endpoint.remote:
        payload["temperature"] = temperature
    return "/chat/completions", payload


class ModelService:
    """One transport for native Ollama, Claude and OpenAI-compatible APIs."""

    def __init__(
        self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        config = configuration(settings)
        self.endpoint = config.chat
        self.embedding_endpoint = config.embeddings
        self.chat_model = config.chat_model
        self.utility_model = config.utility_model
        self.embed_model = config.embed_model
        self.timeout = settings.llm_timeout_seconds or settings.ollama_timeout_seconds
        self.catalog_timeout = settings.ollama_health_timeout_seconds
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._runtime = {
            role: {"status": "unverified", "detail": None, "updated_at": None} for role in _ROLES
        }
        self._catalog: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._catalog_lock = asyncio.Lock()

    @property
    def embedding_identity(self) -> str:
        return self.embedding_endpoint.identity if self.embedding_endpoint else ""

    def configure_models(self, *, chat_model: str, utility_model: str, embed_model: str) -> None:
        for role, field, value in (
            ("narrator", "chat_model", chat_model),
            ("utility", "utility_model", utility_model),
            ("embeddings", "embed_model", embed_model),
        ):
            if getattr(self, field) != value:
                self._runtime[role] = {"status": "unverified", "detail": None, "updated_at": None}
                setattr(self, field, value)

    def runtime_status(self) -> dict[str, dict[str, str | None]]:
        return {key: value.copy() for key, value in self._runtime.items()}

    def _mark(self, role: str, status: str, detail: str | None = None) -> None:
        self._runtime[role] = {
            "status": status,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            context = ssl.create_default_context(cafile=certifi.where())
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            self._client = httpx.AsyncClient(
                verify=context,
                trust_env=False,
                follow_redirects=False,
                timeout=httpx.Timeout(self.timeout, connect=5.0),
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
                headers={"Accept-Encoding": "identity"},
                transport=self._transport,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @asynccontextmanager
    async def _response(self, endpoint: Endpoint, path: str, payload: dict | None = None):
        try:
            async with self._http().stream(
                "GET" if payload is None else "POST",
                endpoint.base_url + path,
                headers=endpoint.headers,
                json=payload,
            ) as response:
                code = response.status_code
                if not 200 <= code < 300:
                    if code in {401, 403}:
                        reason = "Check the server-side API key and permissions."
                    elif code == 429:
                        reason = "The endpoint is rate limited or has exhausted its quota."
                    elif code in {400, 404, 422}:
                        reason = "Check the model ID and the endpoint configuration."
                    else:
                        reason = "Check the endpoint status and retry."
                    raise ModelUnavailableError(f"{endpoint.label} returned HTTP {code}. {reason}")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise ModelResponseError(
                        "The endpoint returned an unsupported content encoding."
                    )
                yield response
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                f"Could not reach {endpoint.label}. Check its endpoint and try again."
            ) from exc

    async def _json(
        self,
        endpoint: Endpoint,
        path: str,
        payload: dict | None = None,
        *,
        timeout: float | None = None,
    ):
        try:
            async with (
                asyncio.timeout(timeout or self.timeout),
                self._response(endpoint, path, payload) as response,
            ):
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > _MAX_REPLY_BYTES:
                        raise ModelResponseError("The model reply exceeded its size limit.")
                    body.extend(chunk)
                return _object(bytes(body))
        except TimeoutError as exc:
            raise ModelUnavailableError(f"{endpoint.label} did not respond in time.") from exc

    async def model_catalog(
        self, *, embeddings: bool = False, refresh: bool = False
    ) -> list[dict[str, Any]]:
        endpoint = self.embedding_endpoint if embeddings else self.endpoint
        if endpoint is None:
            return []
        async with self._catalog_lock:
            cache_key = endpoint.identity + (
                ":embed" if embeddings and endpoint != self.endpoint else ""
            )
            cached = self._catalog.get(cache_key)
            if not refresh and cached and monotonic() - cached[0] < 45:
                return cached[1]
            native = endpoint.protocol == "ollama"
            result = await self._json(
                endpoint, "/api/tags" if native else "/models", timeout=self.catalog_timeout
            )
            values = result.get("models" if native else "data")
            if not isinstance(values, list):
                raise ModelResponseError("The endpoint did not return a model catalogue.")
            names = set()
            for entry in values[:2000]:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("model" if native else "id") or entry.get("name")
                if isinstance(name, str):
                    try:
                        names.add(model_id(name))
                    except ValueError:
                        pass
            # Catalogue results are suggestions. Some servers return incomplete capabilities or use aliases.
            models = [{"name": name, "capabilities": []} for name in sorted(names)[:500]]
            self._catalog[cache_key] = (monotonic(), models)
            return models

    async def list_models(self) -> list[str]:
        return [item["name"] for item in await self.model_catalog()]

    async def stream_dm(
        self, system: str, prompt: str, *, narration_style: str = "balanced"
    ) -> AsyncIterator[str]:
        if not self.chat_model:
            raise ModelUnavailableError("Choose a Dungeon Master model in Settings first.")
        temperature, limit = _PROFILES.get(narration_style, _PROFILES["balanced"])
        path, body = _chat_request(
            self.endpoint,
            self.chat_model,
            system,
            prompt,
            stream=True,
            temperature=temperature,
            tokens=limit,
        )
        emitted = False
        terminal = False
        deadline = asyncio.get_running_loop().time() + self.timeout
        try:
            # Do not leave an asyncio timeout armed across a yielded token.
            async with AsyncExitStack() as stack:
                async with asyncio.timeout_at(deadline):
                    response = await stack.enter_async_context(
                        self._response(self.endpoint, path, body)
                    )
                async with aclosing(
                    _records(response, sse=self.endpoint.protocol != "ollama")
                ) as records:
                    while True:
                        async with asyncio.timeout_at(deadline):
                            raw = await anext(records, None)
                        if raw is None:
                            break
                        if raw.strip() == "[DONE]":
                            terminal = True
                            break
                        data = _object(raw)
                        choice = (data.get("choices") or [{}])[0]
                        terminal = terminal or bool(
                            data.get("done")
                            or data.get("type") == "message_stop"
                            or choice.get("finish_reason")
                        )
                        content = _assistant_text(self.endpoint.protocol, data, stream=True)
                        if content:
                            if not emitted:
                                self._mark("narrator", "healthy")
                            emitted = True
                            yield content.encode("utf-8", errors="replace").decode("utf-8")
                        if terminal and self.endpoint.protocol != "openai":
                            break
            if not emitted:
                raise ModelResponseError("The narrator returned no playable text.")
            if not terminal:
                raise ModelResponseError("The narrator stream ended early. Retry this turn.")
        except (ValueError, AttributeError, TypeError, IndexError, KeyError) as exc:
            failure = ModelResponseError("The narrator returned an invalid stream.")
            self._mark("narrator", "failed", failure.message)
            raise failure from exc
        except TimeoutError as exc:
            failure = ModelUnavailableError("The narrator timed out. Retry this turn.")
            self._mark("narrator", "failed", failure.message)
            raise failure from exc
        except (ModelUnavailableError, ModelResponseError) as exc:
            self._mark("narrator", "failed", exc.message)
            raise
        self._mark("narrator", "healthy")

    async def chat_text(
        self,
        system: str,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        num_predict: int = 500,
        format_schema: dict[str, Any] | None = None,
    ) -> str:
        selected = model or self.utility_model
        try:
            if not selected:
                raise ModelUnavailableError("The utility model is unset.")
            path, body = _chat_request(
                self.endpoint,
                selected,
                system,
                prompt,
                stream=False,
                temperature=temperature,
                tokens=num_predict,
                schema=format_schema,
            )
            content = _assistant_text(
                self.endpoint.protocol, await self._json(self.endpoint, path, body), stream=False
            ).strip()
            if not content or len(content) > 65_536:
                raise ModelResponseError("The utility model returned no usable content.")
        except (AttributeError, TypeError, IndexError) as exc:
            failure = ModelResponseError("The utility model returned an invalid response.")
            self._mark("utility", "failed", failure.message)
            raise failure from exc
        except (ModelResponseError, ModelUnavailableError) as exc:
            self._mark("utility", "failed", exc.message)
            raise
        self._mark("utility", "healthy")
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
            if raw.startswith("```"):
                _, _, raw = raw.partition("\n")
                raw = raw.removesuffix("```").strip()
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            message = "The utility model returned data outside the required schema."
            self._mark("utility", "failed", message)
            raise ModelResponseError(message) from exc

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        endpoint = self.embedding_endpoint
        try:
            if not endpoint or not self.embed_model:
                raise ModelUnavailableError("Choose an embedding model in Settings to index lore.")
            native = endpoint.protocol == "ollama"
            result = await self._json(
                endpoint,
                "/api/embed" if native else "/embeddings",
                {"model": self.embed_model, "input": list(texts)},
            )
            if native:
                ordered = result["embeddings"]
            else:
                rows = sorted(result["data"], key=lambda item: item["index"])
                if [item["index"] for item in rows] != list(range(len(texts))):
                    raise ValueError("invalid indices")
                ordered = [item["embedding"] for item in rows]
            if not isinstance(ordered, list) or len(ordered) != len(texts):
                raise ValueError("invalid vector count")
            dimension = len(ordered[0])
            if not 0 < dimension <= 8192:
                raise ValueError("invalid dimensions")
            vectors = []
            for values in ordered:
                if (
                    not isinstance(values, list)
                    or len(values) != dimension
                    or any(type(n) not in {int, float} or not math.isfinite(n) for n in values)
                ):
                    raise ValueError("invalid vector")
                vectors.append([float(n) for n in values])
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
            failure = ModelResponseError("The embedding endpoint returned invalid vectors.")
            self._mark("embeddings", "failed", failure.message)
            raise failure from exc
        except (ModelResponseError, ModelUnavailableError) as exc:
            self._mark("embeddings", "failed", exc.message)
            raise
        self._mark("embeddings", "healthy")
        return vectors

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]
