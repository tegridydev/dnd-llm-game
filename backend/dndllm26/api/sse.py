from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any

SSE_PROTOCOL_VERSION = 1


def encode_sse(event: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    lines = [f"event: {event}"]
    lines.extend(f"data: {line}" for line in data.splitlines() or [""])
    return "\n".join(lines) + "\n\n"


@dataclass(slots=True)
class SseEmitter:
    request_id: str
    campaign_id: int
    sequence: int = 0

    def emit(self, event: str, data: dict[str, Any]) -> str:
        self.sequence += 1
        return encode_sse(
            event,
            {
                "protocol_version": SSE_PROTOCOL_VERSION,
                "request_id": self.request_id,
                "campaign_id": self.campaign_id,
                "sequence": self.sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data,
            },
        )


def stream_headers(request_id: str) -> dict[str, str]:
    return {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "X-Request-ID": request_id,
    }
