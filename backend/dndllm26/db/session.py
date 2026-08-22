from __future__ import annotations

from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlmodel import SQLModel, create_engine

from dndllm26.core.settings import Settings
from dndllm26.db.schema import SCHEMA_VERSION, validate_existing_database
from dndllm26.db import models as _models  # noqa: F401 - register SQLModel tables


def create_database_engine(settings: Settings) -> Engine:
    engine = create_engine(
        settings.resolved_database_url,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    return engine


def initialise_database(settings: Settings) -> Engine:
    settings.prepare_filesystem()
    validate_existing_database(settings.database_path)
    engine = create_database_engine(settings)
    SQLModel.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text(f"PRAGMA user_version={SCHEMA_VERSION}"))
        connection.execute(text("PRAGMA journal_mode=WAL"))
        connection.execute(text("PRAGMA synchronous=NORMAL"))
        enabled = connection.execute(text("PRAGMA foreign_keys")).scalar_one()
        if int(enabled) != 1:
            raise RuntimeError("SQLite foreign-key enforcement could not be enabled")
    return engine
