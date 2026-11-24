from __future__ import annotations

import asyncio
import json

import httpx2 as httpx
import pytest
from pydantic import BaseModel

from dndllm26.core.errors import ModelResponseError, ModelUnavailableError
from dndllm26.core.settings import Settings
from dndllm26.llm.client import ModelService


def stream_response(*events):
    wire = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=wire)


@pytest.mark.parametrize(
    ("preset", "route", "stream", "auth"),
    [
        ("ollama", "/api/chat", '{"message":{"content":"You arrive."},"done":true}\n', None),
        ("openai", "/v1/chat/completions", None, "authorization"),
        ("lmstudio", "/v1/chat/completions", None, None),
        ("gemini", "/v1beta/openai/chat/completions", None, "authorization"),
        ("anthropic", "/v1/messages", None, "x-api-key"),
    ],
)
def test_streaming_protocols(preset, route, stream, auth) -> None:
    def endpoint(request):
        assert request.url.path == route
        assert json.loads(request.content)["stream"] is True
        if auth:
            assert request.headers[auth].endswith("example-key")
        if preset == "ollama":
            assert json.loads(request.content)["think"] is False
            return httpx.Response(200, text=stream)
        if preset == "anthropic":
            assert request.headers["anthropic-version"] == "2023-06-01"
            return stream_response(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "You arrive."},
                },
                {"type": "message_stop"},
            )
        return stream_response(
            {"choices": [{"delta": {"content": "You arrive."}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )

    async def run():
        service = ModelService(
            Settings(
                llm_provider=preset,
                llm_chat_model="test-model",
                llm_allow_remote=True,
                llm_api_key="example-key" if auth else "",
            ),
            transport=httpx.MockTransport(endpoint),
        )
        try:
            assert [text async for text in service.stream_dm("system", "prompt")] == ["You arrive."]
            assert service.runtime_status()["narrator"]["status"] == "healthy"
        finally:
            await service.close()

    asyncio.run(run())


def test_utility_validates_schema_locally_and_embeddings_restore_order():
    class Decision(BaseModel):
        location: str

    def endpoint(request):
        payload = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            assert payload["input"] == ["one", "two"]
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                },
            )
        assert "JSON object matching" in payload["messages"][0]["content"]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"location":"The camp"}'}}]}
        )

    async def run():
        service = ModelService(
            Settings(
                llm_provider="lmstudio",
                llm_chat_model="story",
                llm_embed_provider="same",
                llm_embed_model="vectors",
            ),
            transport=httpx.MockTransport(endpoint),
        )
        try:
            assert (
                await service.chat_structured("system", "prompt", Decision)
            ).location == "The camp"
            assert await service.embed_batch(["one", "two"]) == [[1, 0], [0, 1]]
        finally:
            await service.close()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["truncated", "redirect"])
def test_truncated_output_and_redirect_cannot_commit_or_forward_key(failure):
    hits = []

    def endpoint(request):
        hits.append(str(request.url))
        if failure == "redirect":
            return httpx.Response(307, headers={"location": "https://example.invalid/"})
        return stream_response({"choices": [{"delta": {"content": "Unfinished"}}]})

    async def run():
        service = ModelService(
            Settings(
                llm_provider="openai",
                llm_api_key="example-key",
                llm_allow_remote=True,
                llm_chat_model="story",
            ),
            transport=httpx.MockTransport(endpoint),
        )
        try:
            with pytest.raises(
                (ModelResponseError, ModelUnavailableError), match="ended early|HTTP 307"
            ):
                _ = [text async for text in service.stream_dm("system", "prompt")]
        finally:
            await service.close()

    asyncio.run(run())
    assert hits == ["https://api.openai.com/v1/chat/completions"]
