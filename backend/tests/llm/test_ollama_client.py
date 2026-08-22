from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from dndllm26.core.errors import ModelResponseError
from dndllm26.llm.ollama_client import OllamaService


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.kwargs = None

    async def chat(self, **kwargs):
        self.kwargs = kwargs
        return self.responses


async def _empty_stream():
    if False:
        yield {}


def test_streaming_narration_disables_thinking_and_rejects_empty_output(settings) -> None:
    service = OllamaService(settings)
    client = FakeClient(_empty_stream())
    service._client_instance = client

    async def consume() -> None:
        async for _ in service.stream_dm("system", "prompt"):
            pass

    with pytest.raises(ModelResponseError, match="no playable text"):
        asyncio.run(consume())
    assert client.kwargs["think"] is False
    assert service.runtime_status()["narrator"]["status"] == "failed"


def test_utility_calls_disable_thinking_and_record_success(settings) -> None:
    service = OllamaService(settings)
    client = FakeClient(SimpleNamespace(message=SimpleNamespace(content="usable")))
    service._client_instance = client
    assert asyncio.run(service.chat_text("system", "prompt")) == "usable"
    assert client.kwargs["think"] is False
    assert service.runtime_status()["utility"]["status"] == "healthy"
