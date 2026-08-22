from __future__ import annotations

from ipaddress import ip_address
import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from dndllm26.core.logging import request_id_var
from dndllm26.core.settings import Settings


class UploadBodyLimitMiddleware:
    """Bound lore request bodies before Starlette's multipart parser can spool them."""

    def __init__(self, app, settings: Settings) -> None:  # type: ignore[no-untyped-def]
        self.app = app
        self.limit = settings.max_upload_request_bytes

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[no-untyped-def]
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != "/api/lore/upload"
        ):
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared:
            try:
                if int(declared) > self.limit:
                    await self._reject(send)
                    return
            except ValueError:
                await self._reject(send)
                return
        messages: list[dict[str, object]] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            total += len(body)
            if total > self.limit:
                await self._reject(send)
                return
            messages.append(message)
            if not message.get("more_body", False):
                break
        position = 0

        async def replay() -> dict[str, object]:
            nonlocal position
            if position < len(messages):
                message = messages[position]
                position += 1
                return message
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay, send)

    async def _reject(self, send) -> None:  # type: ignore[no-untyped-def]
        import json

        body = json.dumps(
            {
                "code": "upload_too_large",
                "message": "The upload request exceeds the configured size limit.",
                "retryable": False,
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _loopback(value: str | None) -> bool:
    if not value:
        return False
    host = value.strip().strip("[]").lower()
    if host == "localhost" or host == "testclient":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", "").strip()
        if not request_id or len(request_id) > 128:
            request_id = uuid4().hex
        token = request_id_var.set(request_id)
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cross-Origin-Resource-Policy"] = "same-site"
        response.headers["Server-Timing"] = f"app;dur={(time.perf_counter() - started) * 1000:.1f}"
        return response


class LocalOnlyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.allowed_origins = set(settings.allowed_origins)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_host = request.client.host if request.client else None
        if client_host not in {"testclient", None} and not _loopback(client_host):
            return JSONResponse(
                status_code=403,
                content={
                    "code": "local_only",
                    "message": "This API accepts loopback clients only.",
                },
            )
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            if origin.rstrip("/") not in self.allowed_origins:
                return JSONResponse(
                    status_code=403,
                    content={
                        "code": "origin_rejected",
                        "message": "The request origin is not allowed by the local API.",
                    },
                )
        return await call_next(request)
