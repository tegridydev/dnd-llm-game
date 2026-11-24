from __future__ import annotations

from array import array
import asyncio
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from heapq import nlargest
import math
import sqlite3
import sys
from uuid import uuid4

from dndllm26.core.errors import LoreProcessingError
from dndllm26.core.settings import Settings
from dndllm26.llm.client import ModelService


@dataclass(frozen=True, slots=True)
class LoreReference:
    document_id: int
    filename: str
    index_key: str
    index_table: str
    embed_model: str
    embed_dimension: int


@dataclass(frozen=True, slots=True)
class IndexResult:
    chunks: int
    dimension: int
    index_key: str
    table_name: str


def _unit(vector: Sequence[float]) -> array:
    magnitude = math.hypot(*vector)
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise LoreProcessingError("The embedding model returned a zero or invalid vector.")
    return array("f", (value / magnitude for value in vector))


def _little_endian_vector(vector: array) -> array:
    if sys.byteorder != "little":
        vector.byteswap()
    return vector


class RagStore:
    """SQLite index for bounded, exact cosine retrieval over campaign lore."""

    def __init__(self, settings: Settings, models: ModelService) -> None:
        self._path = settings.resolved_lore_index_dir / "lore.sqlite3"
        self._models = models

    @property
    def embed_model(self) -> str:
        return self._models.embed_model

    @property
    def table_name(self) -> str:
        identity = f"{self._models.embedding_identity}\0{self.embed_model}"
        return "sqlite_v1_" + sha256(identity.encode()).hexdigest()[:16]

    @property
    def enabled(self) -> bool:
        return bool(self._models.embedding_identity and self.embed_model)

    @contextmanager
    def _database(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=5)
        try:
            if write:
                with conn:
                    conn.execute(
                        """CREATE TABLE IF NOT EXISTS lore_passages (
                            index_key TEXT NOT NULL,
                            chunk_index INTEGER NOT NULL,
                            document_id INTEGER NOT NULL,
                            filename TEXT NOT NULL,
                            passage TEXT NOT NULL,
                            vector BLOB NOT NULL,
                            PRIMARY KEY (index_key, chunk_index)
                        ) WITHOUT ROWID"""
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS lore_by_document ON lore_passages (document_id)"
                    )
                    yield conn
            else:
                yield conn
        finally:
            conn.close()

    def _add_rows(self, rows: list[tuple]) -> None:
        with self._database(write=True) as conn:
            conn.executemany("INSERT INTO lore_passages VALUES (?, ?, ?, ?, ?, ?)", rows)

    async def index_chunks(
        self,
        *,
        document_id: int,
        filename: str,
        content_sha256: str,
        chunks: Sequence[str],
        batch_size: int,
        index_version: int,
    ) -> IndexResult:
        index_key = f"{document_id}:{content_sha256}:{index_version}:{uuid4().hex}"
        table_name = self.table_name
        rows: list[tuple] = []
        dimension = 0
        for start in range(0, len(chunks), batch_size):
            passages = chunks[start : start + batch_size]
            vectors = await self._models.embed_batch(passages)
            for number, (text, vector) in enumerate(zip(passages, vectors, strict=True), start):
                dimension = dimension or len(vector)
                if len(vector) != dimension:
                    raise LoreProcessingError(
                        "The embedding model changed vector dimensions during indexing."
                    )
                packed = _little_endian_vector(_unit(vector)).tobytes()
                rows.append((index_key, number, document_id, filename, text, packed))
        if not rows:
            raise LoreProcessingError("No embeddings were produced for this document.")

        # Make cancellation wait for the transaction before disposing of inactive rows.
        write = asyncio.create_task(asyncio.to_thread(self._add_rows, rows))
        try:
            await asyncio.shield(write)
        except asyncio.CancelledError:
            try:
                await write
            finally:
                await self.delete_index(table_name=table_name, index_key=index_key)
            raise
        return IndexResult(len(rows), dimension, index_key, table_name)

    def _delete(self, condition: str, values: tuple) -> None:
        if self._path.exists():
            with self._database(write=True) as conn:
                conn.execute("DELETE FROM lore_passages WHERE " + condition, values)

    async def delete_index(self, *, table_name: str, index_key: str) -> None:
        if table_name.startswith("sqlite_v1_"):
            await asyncio.to_thread(self._delete, "index_key = ?", (index_key,))

    async def cleanup_stale(self, *, document_id: int, active_index_key: str) -> None:
        await asyncio.to_thread(
            self._delete, "document_id = ? AND index_key != ?", (document_id, active_index_key)
        )

    async def delete_document(self, document_id: int) -> None:
        await asyncio.to_thread(self._delete, "document_id = ?", (document_id,))

    def _search(self, vector: array, keys: list[str], limit: int) -> list[dict]:
        per_document = max(1, 8192 // len(keys))
        sumprod = getattr(
            math, "sumprod", lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))
        )
        if not self._path.exists():
            return []
        with self._database() as conn:
            conn.execute("BEGIN")

            def scores():
                for key in keys:
                    for index, raw in conn.execute(
                        "SELECT chunk_index, vector FROM lore_passages "
                        "WHERE index_key = ? ORDER BY chunk_index LIMIT ?",
                        (key, per_document),
                    ):
                        if len(raw) != len(vector) * vector.itemsize:
                            continue
                        candidate = array("f")
                        candidate.frombytes(raw)
                        _little_endian_vector(candidate)
                        yield sumprod(vector, candidate), key, index

            results = []
            for score, key, index in nlargest(limit, scores()):
                document_id, filename, text = conn.execute(
                    """SELECT document_id, filename, passage FROM lore_passages
                    WHERE index_key = ? AND chunk_index = ?""",
                    (key, index),
                ).fetchone()
                results.append(
                    {
                        "document_id": document_id,
                        "filename": filename,
                        "chunk_index": index,
                        "text": text,
                        "score": max(0.0, 1 - score),
                    }
                )
            return results

    async def search(
        self, query: str, *, limit: int, documents: Sequence[LoreReference]
    ) -> list[dict]:
        compatible = [
            document
            for document in documents
            if document.index_key
            and document.index_table == self.table_name
            and document.embed_model == self.embed_model
            and document.embed_dimension > 0
        ]
        if not self.enabled or not compatible:
            return []
        vector = _unit(await self._models.embed(query))
        keys = [row.index_key for row in compatible if row.embed_dimension == len(vector)]
        try:
            return await asyncio.to_thread(self._search, vector, keys, limit) if keys else []
        except sqlite3.Error as exc:
            raise LoreProcessingError("Local lore search is unavailable.") from exc
