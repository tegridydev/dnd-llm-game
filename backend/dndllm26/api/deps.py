from __future__ import annotations

from collections.abc import Generator

from fastapi import Request
from sqlmodel import Session

from dndllm26.core.resources import AppResources


def get_resources(request: Request) -> AppResources:
    resources = getattr(request.app.state, "resources", None)
    if resources is None:
        raise RuntimeError("Application resources are not initialised")
    return resources


def get_session(request: Request) -> Generator[Session, None, None]:
    resources = get_resources(request)
    with Session(resources.engine) as session:
        yield session
