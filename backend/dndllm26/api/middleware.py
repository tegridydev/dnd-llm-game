from __future__ import annotations

import asyncio
from ipaddress import ip_address
import time
from urllib.parse import urlsplit
from uuid import uuid4

from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from dndllm26.core.logging import request_id_var
from dndllm26.core.settings import Settings


class RequestLimitsMiddleware:
    """Check request size as bytes arrive, before body or multipart buffering."""

    def __init__(self, app, settings: Settings) -> None:
        self.app = app
        self.upload_limit = settings.max_upload_request_bytes
        self.active_uploads = 0

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["method"] in {"GET", "HEAD", "OPTIONS"}:
            await self.app(scope, receive, send)
            return
        upload = scope["method"] == "POST" and scope["path"] == "/api/lore/upload"
        limit = self.upload_limit if upload else 128 * 1024
        code = "upload_too_large" if upload else "body_too_large"
        message = "The request exceeds its size limit."
        headers = dict(scope.get("headers", []))
        declared = headers.get(b"content-length")
        if declared is not None and (
            not declared.isdigit() or len(declared) > 12 or int(declared) > limit
        ):
            await JSONResponse({"code": code, "message": message}, status_code=413)(
                scope, receive, send
            )
            return
        if upload and self.active_uploads >= 2:
            await JSONResponse(
                {"code": "upload_busy", "message": "Two lore uploads are already in progress."},
                status_code=429,
            )(scope, receive, send)
            return
        total = 0
        more = True
        deadline = asyncio.get_running_loop().time() + 60

        async def bounded_receive():
            nonlocal total, more
            try:
                async with asyncio.timeout_at(deadline if more else None):
                    item = await receive()
            except TimeoutError as exc:
                raise HTTPException(
                    status_code=408, detail="The upload did not finish in time."
                ) from exc
            if item["type"] == "http.request":
                total += len(item.get("body", b""))
                more = item.get("more_body", False)
                if total > limit:
                    raise HTTPException(status_code=413, detail=message)
            return item

        self.active_uploads += int(upload)
        try:
            await self.app(scope, bounded_receive, send)
        finally:
            self.active_uploads -= int(upload)


def _loopback(value: str | None) -> bool:
    if not value:
        return False
    host = value.strip().strip("[]").lower()
    if host in {"localhost", "ip6-localhost"}:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", "").strip()
        if (
            not request_id
            or len(request_id) > 128
            or not all(c.isascii() and c.isprintable() for c in request_id)
        ):
            request_id = uuid4().hex
        token = request_id_var.set(request_id)
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers.setdefault("Cache-Control", "no-store")
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
        if client_host != "testclient" and not _loopback(client_host):
            return JSONResponse(
                status_code=403,
                content={
                    "code": "local_only",
                    "message": "This API accepts loopback clients only.",
                },
            )
        host = request.headers.get("host", "")
        try:
            target = urlsplit("//" + host)
            _ = target.port
            allowed_host = (
                target.netloc == host
                and not target.path
                and not target.query
                and not target.fragment
                and target.username is None
                and (
                    _loopback(target.hostname)
                    or (target.hostname == "testserver" and client_host == "testclient")
                )
            )
        except ValueError:
            allowed_host = False
        if not allowed_host:
            return JSONResponse(
                {"code": "host_rejected", "message": "Use the local app address."}, status_code=400
            )
        origin = request.headers.get("origin")
        cross_site = request.headers.get("sec-fetch-site") == "cross-site"
        if (origin is not None and origin not in self.allowed_origins) or (
            cross_site and origin is None
        ):
            return JSONResponse(
                {"code": "origin_rejected", "message": "Open the app from its local address."},
                status_code=403,
            )
        return await call_next(request)
