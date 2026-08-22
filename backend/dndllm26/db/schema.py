from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlmodel import SQLModel, create_engine


SCHEMA_VERSION = 8


def _normalise_sql(value: str | None) -> str | None:
    return " ".join(value.split()).casefold() if value else None


def _schema_manifest(connection: Any) -> dict[str, object]:
    tables = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        if not str(row[0]).startswith("sqlite_")
    }
    table_definitions: dict[str, object] = {}
    for table in sorted(tables):
        create_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()[0]
        columns = tuple(
            (row[1], row[2].casefold(), row[3], row[4], row[5], row[6])
            for row in connection.execute(f'PRAGMA table_xinfo("{table}")')
        )
        foreign_keys = tuple(
            sorted(
                tuple(row[2:8]) for row in connection.execute(f'PRAGMA foreign_key_list("{table}")')
            )
        )
        indexes: list[object] = []
        for row in connection.execute(f'PRAGMA index_list("{table}")'):
            name = str(row[1])
            index_sql_row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (name,)
            ).fetchone()
            index_columns = tuple(
                (item[1], item[2], item[5])
                for item in connection.execute(f'PRAGMA index_xinfo("{name}")')
            )
            indexes.append(
                (
                    name,
                    row[2],
                    row[3],
                    row[4],
                    index_columns,
                    _normalise_sql(index_sql_row[0] if index_sql_row else None),
                )
            )
        table_definitions[table] = {
            "sql": _normalise_sql(create_sql),
            "columns": columns,
            "foreign_keys": foreign_keys,
            "indexes": tuple(sorted(indexes, key=lambda value: str(value[0]))),
        }
    return {"tables": tuple(sorted(tables)), "definitions": table_definitions}


@lru_cache(maxsize=1)
def _canonical_schema_manifest() -> dict[str, object]:
    # Import registers every first-party table without maintaining a parallel manifest.
    from dndllm26.db import models as _models  # noqa: F401

    engine = create_engine("sqlite://")
    try:
        SQLModel.metadata.create_all(engine)
        raw = engine.raw_connection()
        try:
            return _schema_manifest(raw)
        finally:
            raw.close()
    finally:
        engine.dispose()


def validate_existing_database(database_path: Path | None) -> None:
    """Validate a populated database against the complete current schema."""

    if database_path is None or not database_path.exists() or database_path.stat().st_size == 0:
        return
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            if not str(row[0]).startswith("sqlite_")
        }
        if not tables:
            return
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version != SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {version} is incompatible with this release "
                f"(expected {SCHEMA_VERSION}). Archive or remove {database_path} manually "
                "and restart to create a fresh database. The existing file was not modified."
            )
        expected = _canonical_schema_manifest()
        actual = _schema_manifest(connection)
        missing = sorted(set(expected["tables"]) - tables)
        if missing:
            raise RuntimeError(
                "The database claims the current schema version but is incomplete; missing tables: "
                + ", ".join(missing)
            )
        unexpected = sorted(tables - set(expected["tables"]))
        if unexpected:
            raise RuntimeError(
                "The database claims the current schema version but contains unexpected tables: "
                + ", ".join(unexpected)
            )
        if actual != expected:
            differing = sorted(
                table
                for table in tables
                if actual["definitions"].get(table) != expected["definitions"].get(table)
            )
            raise RuntimeError(
                "The database claims the current schema version but its structure is incompatible"
                + (": " + ", ".join(differing) if differing else ".")
            )
        integrity = [str(row[0]) for row in connection.execute("PRAGMA integrity_check")]
        if integrity != ["ok"]:
            raise RuntimeError("SQLite integrity check failed: " + "; ".join(integrity))
        foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
        if foreign_keys:
            raise RuntimeError(f"SQLite foreign-key violations were found: {foreign_keys!r}")
    finally:
        connection.close()
