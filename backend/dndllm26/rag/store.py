from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
from typing import Any, Sequence
from uuid import uuid4

import lancedb

from dndllm26.core.settings import Settings
from dndllm26.llm.ollama_client import OllamaService


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


def _escape_filter_string(value: str) -> str:
    return value.replace("'", "''")


def _table_name(model: str, dimension: int) -> str:
    fingerprint = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
    return f"lore_v1_{fingerprint}_{dimension}"


class RagStore:
    """LanceDB adapter with logically atomic, versioned document replacement."""

    def __init__(self, settings: Settings, ollama: OllamaService) -> None:
        self._db_path = settings.resolved_lancedb_dir
        self._ollama = ollama
        self._db_instance: Any | None = None
        self._lock = asyncio.Lock()

    @property
    def embed_model(self) -> str:
        return self._ollama.embed_model

    def _connect_sync(self) -> Any:
        if self._db_instance is None:
            self._db_instance = lancedb.connect(str(self._db_path))
        return self._db_instance

    def _table_names_sync(self) -> list[str]:
        db = self._connect_sync()
        names = db.table_names()
        return list(names) if not isinstance(names, list) else names

    def _add_rows_sync(self, table_name: str, rows: list[dict[str, Any]]) -> None:
        db = self._connect_sync()
        if table_name not in self._table_names_sync():
            db.create_table(table_name, data=rows)
            return
        table = db.open_table(table_name)
        table.add(rows)

    def _delete_index_sync(self, table_name: str, index_key: str) -> None:
        db = self._connect_sync()
        if table_name not in self._table_names_sync():
            return
        table = db.open_table(table_name)
        table.delete(f"index_key = '{_escape_filter_string(index_key)}'")

    def _cleanup_document_sync(self, document_id: int, active_index_key: str | None) -> None:
        db = self._connect_sync()
        for table_name in self._table_names_sync():
            if not table_name.startswith("lore_v1_"):
                continue
            table = db.open_table(table_name)
            expression = f"document_id = {int(document_id)}"
            if active_index_key:
                expression += f" AND index_key != '{_escape_filter_string(active_index_key)}'"
            table.delete(expression)

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
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            vectors.extend(await self._ollama.embed_batch(chunks[start : start + batch_size]))
        if not vectors:
            raise ValueError("No embeddings were produced for the document.")
        dimension = len(vectors[0])
        table_name = _table_name(self._ollama.embed_model, dimension)
        model_key = hashlib.sha256(self._ollama.embed_model.encode()).hexdigest()[:12]
        index_key = f"{document_id}:{content_sha256}:{model_key}:{index_version}:{uuid4().hex}"
        rows = [
            {
                "vector": vector,
                "document_id": int(document_id),
                "filename": filename,
                "chunk_index": index,
                "text": chunks[index],
                "index_key": index_key,
                "embed_model": self._ollama.embed_model,
                "content_sha256": content_sha256,
            }
            for index, vector in enumerate(vectors)
        ]
        try:
            async with self._lock:
                await asyncio.to_thread(self._add_rows_sync, table_name, rows)
        except Exception:
            async with self._lock:
                await asyncio.to_thread(self._delete_index_sync, table_name, index_key)
            raise
        return IndexResult(
            chunks=len(rows),
            dimension=dimension,
            index_key=index_key,
            table_name=table_name,
        )

    async def delete_index(self, *, table_name: str, index_key: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._delete_index_sync, table_name, index_key)

    async def cleanup_stale(
        self,
        *,
        document_id: int,
        active_index_key: str,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._cleanup_document_sync,
                document_id,
                active_index_key,
            )

    async def delete_document(self, document_id: int) -> None:
        async with self._lock:
            await asyncio.to_thread(self._cleanup_document_sync, document_id, None)

    def _search_table_sync(
        self,
        table_name: str,
        vector: list[float],
        index_keys: Sequence[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        db = self._connect_sync()
        if table_name not in self._table_names_sync():
            return []
        table = db.open_table(table_name)
        quoted = ", ".join(f"'{_escape_filter_string(key)}'" for key in index_keys)
        query = table.search(vector)
        if quoted:
            expression = f"index_key IN ({quoted})"
            try:
                query = query.where(expression, prefilter=True)
            except TypeError:  # Compatibility with older supported LanceDB query signatures.
                query = query.where(expression)
        return list(query.limit(limit).to_list())

    async def search(
        self,
        query: str,
        *,
        limit: int,
        documents: Sequence[LoreReference],
    ) -> list[dict[str, Any]]:
        compatible = [
            document
            for document in documents
            if document.index_key
            and document.index_table
            and document.embed_model == self._ollama.embed_model
            and document.embed_dimension > 0
        ]
        if not compatible:
            return []
        vector = await self._ollama.embed(query)
        grouped: dict[str, list[LoreReference]] = {}
        for document in compatible:
            if document.embed_dimension == len(vector):
                grouped.setdefault(document.index_table, []).append(document)
        rows: list[dict[str, Any]] = []
        async with self._lock:
            for table_name, refs in grouped.items():
                rows.extend(
                    await asyncio.to_thread(
                        self._search_table_sync,
                        table_name,
                        vector,
                        [ref.index_key for ref in refs],
                        max(limit, limit * 2),
                    )
                )
        rows.sort(key=lambda row: float(row.get("_distance", float("inf"))))
        return [
            {
                "document_id": row.get("document_id"),
                "filename": row.get("filename"),
                "chunk_index": row.get("chunk_index"),
                "text": row.get("text"),
                "score": row.get("_distance"),
            }
            for row in rows[:limit]
        ]
