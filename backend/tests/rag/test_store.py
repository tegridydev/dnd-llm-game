from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dndllm26.core.errors import LoreProcessingError
from dndllm26.core.settings import Settings
from dndllm26.rag.store import LoreReference, RagStore


class FakeModels:
    embed_model = "test-embed"
    embedding_identity = "test"

    async def embed_batch(self, texts):
        return [[1.0, 0.0] if "winter" in text else [0.0, 1.0] for text in texts]

    async def embed(self, text):
        return (await self.embed_batch([text]))[0]


def make_store(tmp_path: Path) -> RagStore:
    return RagStore(Settings(app_root=tmp_path, lancedb_dir=tmp_path), FakeModels())


def test_only_active_compatible_index_is_searched_and_deletion_is_final(tmp_path):
    async def run():
        store = make_store(tmp_path)
        args = dict(document_id=7, filename="realm.pdf", content_sha256="a" * 64, batch_size=1)
        old = await store.index_chunks(**args, chunks=["old winter"], index_version=1)
        new = await store.index_chunks(**args, chunks=["summer", "new winter"], index_version=2)
        assert old.index_key != new.index_key
        old_reference = LoreReference(
            7, "realm.pdf", old.index_key, old.table_name, store.embed_model, 2
        )
        new_reference = LoreReference(
            7, "realm.pdf", new.index_key, new.table_name, store.embed_model, 2
        )
        report = await store.search("winter", limit=2, documents=[new_reference])
        assert [row["text"] for row in report] == ["new winter", "summer"]
        store._models.embedding_identity = "different-provider"
        assert await store.search("winter", limit=2, documents=[new_reference]) == []
        store._models.embedding_identity = "test"
        await store.cleanup_stale(document_id=7, active_index_key=new.index_key)
        assert await store.search("winter", limit=2, documents=[old_reference]) == []
        await store.delete_document(7)
        assert await store.search("winter", limit=2, documents=[new_reference]) == []

    asyncio.run(run())


def test_invalid_vector_does_not_open_index(tmp_path):
    async def run():
        store = make_store(tmp_path)

        async def zeros(_):
            return [[0.0, 0.0]]

        store._models.embed_batch = zeros
        with pytest.raises(LoreProcessingError, match="zero"):
            await store.index_chunks(
                document_id=7,
                filename="realm.pdf",
                content_sha256="a" * 64,
                batch_size=1,
                chunks=["invalid"],
                index_version=1,
            )
        assert not (tmp_path / "lore.sqlite3").exists()

    asyncio.run(run())
