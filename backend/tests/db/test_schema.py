from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dndllm26.core.settings import Settings
from dndllm26.db.schema import SCHEMA_VERSION, validate_existing_database
from dndllm26.db.session import initialise_database


def test_fresh_database_is_versioned_and_constrained(tmp_path: Path) -> None:
    settings = Settings(
        app_root=tmp_path,
        database_url=f"sqlite:///{tmp_path / 'data' / 'test.db'}",
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "data" / "uploads",
        lancedb_dir=tmp_path / "data" / "lancedb",
    )
    engine = initialise_database(settings)
    engine.dispose()
    validate_existing_database(settings.database_path)
    connection = sqlite3.connect(settings.database_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='pendingroll'"
        ).fetchone()[0]
        assert "ck_pending_roll_status" in sql
        assert "ck_pending_roll_dc" in sql
    finally:
        connection.close()


def test_old_database_is_rejected_without_modification(tmp_path: Path) -> None:
    database = tmp_path / "old.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE campaign (id INTEGER PRIMARY KEY)")
    connection.execute("PRAGMA user_version=2")
    connection.commit()
    connection.close()
    before = database.read_bytes()
    with pytest.raises(RuntimeError, match="Archive or remove"):
        validate_existing_database(database)
    assert database.read_bytes() == before


def test_schema_v7_is_rejected_without_modification(tmp_path: Path) -> None:
    settings = Settings(
        app_root=tmp_path,
        database_url=f"sqlite:///{tmp_path / 'data' / 'test.db'}",
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "data" / "uploads",
        lancedb_dir=tmp_path / "data" / "lancedb",
    )
    engine = initialise_database(settings)
    engine.dispose()
    connection = sqlite3.connect(settings.database_path)
    connection.execute("DROP TABLE campaignopening")
    connection.execute("PRAGMA user_version=7")
    connection.commit()
    connection.close()

    before = settings.database_path.read_bytes()
    with pytest.raises(RuntimeError, match="expected 8"):
        initialise_database(settings)
    assert settings.database_path.read_bytes() == before


def test_current_database_with_missing_partial_index_is_rejected_read_only(tmp_path: Path) -> None:
    settings = Settings(
        app_root=tmp_path,
        database_url=f"sqlite:///{tmp_path / 'data' / 'test.db'}",
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "data" / "uploads",
        lancedb_dir=tmp_path / "data" / "lancedb",
    )
    engine = initialise_database(settings)
    engine.dispose()
    connection = sqlite3.connect(settings.database_path)
    connection.execute("DROP INDEX uq_character_campaign_protagonist")
    connection.commit()
    connection.close()
    before = settings.database_path.read_bytes()
    with pytest.raises(RuntimeError, match="structure is incompatible"):
        validate_existing_database(settings.database_path)
    assert settings.database_path.read_bytes() == before
