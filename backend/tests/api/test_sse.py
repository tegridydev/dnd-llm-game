from __future__ import annotations

import json

from dndllm26.api.sse import SseEmitter, encode_sse


def test_sse_encodes_multiline_data_safely() -> None:
    value = encode_sse("message", {"text": "first\nsecond"})
    assert value.startswith("event: message\n")
    assert value.endswith("\n\n")
    data_line = next(line for line in value.splitlines() if line.startswith("data: "))
    assert json.loads(data_line.removeprefix("data: "))["text"] == "first\nsecond"


def test_emitter_adds_request_identity_and_sequence() -> None:
    emitter = SseEmitter("req-1", 42)
    first = emitter.emit("phase", {"status": "checking_action"})
    second = emitter.emit("done", {"status": "complete", "authoritative_refresh": True})
    assert '"sequence":1' in first
    assert '"sequence":2' in second
    assert '"campaign_id":42' in first
    assert '"protocol_version":1' in first
