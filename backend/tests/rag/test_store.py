from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dndllm26.core.settings import Settings
from dndllm26.rag.store import RagStore


class FakeOllama:
    embed_model = "test-embed"

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [[float(index), 1.0] for index, _text in enumerate(texts)]


def make_store(tmp_path: Path) -> RagStore:
    settings = Settings(
        app_root=tmp_path,
        database_url=f"sqlite:///{tmp_path / 'data' / 'test.db'}",
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "data" / "uploads",
        lancedb_dir=tmp_path / "data" / "lancedb",
    )
    return RagStore(settings, FakeOllama())  # type: ignore[arg-type]


def test_each_index_attempt_uses_a_fresh_inactive_key(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    rows: list[list[dict[str, object]]] = []
    store._add_rows_sync = lambda _table, values: rows.append(values)  # type: ignore[method-assign]
    first = asyncio.run(
        store.index_chunks(
            document_id=7,
            filename="realm.pdf",
            content_sha256="a" * 64,
            chunks=["one"],
            batch_size=1,
            index_version=1,
        )
    )
    second = asyncio.run(
        store.index_chunks(
            document_id=7,
            filename="realm.pdf",
            content_sha256="a" * 64,
            chunks=["one"],
            batch_size=1,
            index_version=2,
        )
    )
    assert first.index_key != second.index_key
    assert rows[0][0]["index_key"] == first.index_key
    assert rows[1][0]["index_key"] == second.index_key


def test_failed_index_attempt_only_cleans_its_new_key(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    deleted: list[tuple[str, str]] = []

    def fail_add(_table: str, _rows: list[dict[str, object]]) -> None:
        raise RuntimeError("write failed")

    store._add_rows_sync = fail_add  # type: ignore[method-assign]
    store._delete_index_sync = lambda table, key: deleted.append((table, key))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="write failed"):
        asyncio.run(
            store.index_chunks(
                document_id=7,
                filename="realm.pdf",
                content_sha256="a" * 64,
                chunks=["one"],
                batch_size=1,
                index_version=2,
            )
        )
    assert len(deleted) == 1
    assert deleted[0][1].startswith("7:")


def test_delete_document_removes_all_versions(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    deleted: list[tuple[int, str | None]] = []
    store._cleanup_document_sync = lambda document_id, active: deleted.append(  # type: ignore[method-assign]
        (document_id, active)
    )
    asyncio.run(store.delete_document(9))
    assert deleted == [(9, None)]
