from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from dndllm26.core.settings import Settings

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        app_root=tmp_path,
        database_url=f"sqlite:///{tmp_path / 'data' / 'test.db'}",
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "data" / "uploads",
        lancedb_dir=tmp_path / "data" / "lancedb",
        api_host="127.0.0.1",
        frontend_host="127.0.0.1",
    )


@pytest.fixture
def engine(settings: Settings) -> Iterator[Engine]:
    # Imported lazily so dependency-independent unit tests can run in minimal
    # environments. Normal project installs always include SQLModel.
    from dndllm26.db.session import initialise_database

    database_engine = initialise_database(settings)
    try:
        yield database_engine
    finally:
        database_engine.dispose()
