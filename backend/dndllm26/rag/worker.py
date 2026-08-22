from __future__ import annotations

import asyncio
import logging
from time import perf_counter

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from dndllm26.core.logging import log_event
from dndllm26.core.settings import Settings
from dndllm26.db.models import LoreDocument, now_utc
from dndllm26.rag.deletion import finalize_lore_deletion
from dndllm26.rag.pdf import chunk_text, extract_pdf_text
from dndllm26.rag.store import RagStore

logger = logging.getLogger(__name__)
_STOP = -1


class LoreIndexWorker:
    """Single-process bounded worker for durable local lore indexing jobs."""

    def __init__(self, settings: Settings, engine: Engine, rag: RagStore) -> None:
        self._settings = settings
        self._engine = engine
        self._rag = rag
        self._queue: asyncio.Queue[int] = asyncio.Queue(maxsize=settings.lore_worker_queue_size)
        self._queued: set[int] = set()
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._active_document_id: int | None = None

    @property
    def running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    @property
    def active_document_id(self) -> int | None:
        return self._active_document_id

    @property
    def queued_count(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        if self.running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="dndllm26-lore-index-worker")
        await self.recover()

    def enqueue_pending(self) -> int:
        if not self.running:
            return 0
        available = max(0, self._queue.maxsize - self._queue.qsize())
        if available == 0:
            return 0
        with Session(self._engine) as session:
            records = session.exec(
                select(LoreDocument)
                .where(LoreDocument.status == "queued")
                .order_by(LoreDocument.created_at)
            ).all()
        added = 0
        for record in records:
            if added >= available or record.id is None:
                break
            if record.id in self._queued or record.id == self._active_document_id:
                continue
            self._queue.put_nowait(record.id)
            self._queued.add(record.id)
            added += 1
        return added

    async def stop(self) -> None:
        if not self._task:
            return
        self._running = False
        try:
            self._queue.put_nowait(_STOP)
        except asyncio.QueueFull:
            self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            self._queued.clear()
            self._active_document_id = None

    async def recover(self) -> None:
        with Session(self._engine) as session:
            deleting = session.exec(
                select(LoreDocument).where(LoreDocument.status == "deleting")
            ).all()
            records = session.exec(
                select(LoreDocument).where(LoreDocument.status.in_(["queued", "indexing"]))
            ).all()
            for record in records:
                if record.status == "indexing":
                    record.status = "queued"
                    record.error = "Indexing was interrupted by an application restart and has been queued again."
                    record.updated_at = now_utc()
                    session.add(record)
            session.commit()
            deleting_ids = [record.id for record in deleting if record.id is not None]
        for document_id in deleting_ids:
            try:
                await finalize_lore_deletion(
                    document_id,
                    engine=self._engine,
                    settings=self._settings,
                    rag=self._rag,
                )
            except Exception:
                logger.exception(
                    "Failed to recover pending lore deletion",
                    extra={"document_id": document_id},
                )
        self.enqueue_pending()

    def enqueue(self, document_id: int, *, force: bool = False) -> bool:
        if document_id in self._queued or document_id == self._active_document_id:
            return True
        if not self.running:
            return False
        if force:
            with Session(self._engine) as session:
                record = session.get(LoreDocument, document_id)
                if record and record.status != "deleting":
                    record.status = "queued"
                    record.error = None
                    record.updated_at = now_utc()
                    session.add(record)
                    session.commit()
        try:
            self._queue.put_nowait(document_id)
        except asyncio.QueueFull:
            return False
        self._queued.add(document_id)
        return True

    async def _run(self) -> None:
        while self._running:
            document_id = await self._queue.get()
            if document_id == _STOP:
                self._queue.task_done()
                break
            self._queued.discard(document_id)
            self._active_document_id = document_id
            try:
                await self._process(document_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Unhandled lore worker failure", extra={"document_id": document_id}
                )
            finally:
                self._active_document_id = None
                self._queue.task_done()
                self.enqueue_pending()

    async def _process(self, document_id: int) -> None:
        started = perf_counter()
        result = None
        activated = False
        with Session(self._engine) as session:
            record = session.get(LoreDocument, document_id)
            if not record:
                return
            if record.status == "deleting":
                return
            if (
                record.attempts >= self._settings.lore_worker_max_attempts
                and record.status == "error"
            ):
                return
            path = self._settings.resolved_upload_dir / record.storage_name
            if not record.storage_name or not path.exists() or not path.is_file():
                record.status = "error"
                record.error = "Uploaded PDF is missing from the managed upload directory."
                record.updated_at = now_utc()
                session.add(record)
                session.commit()
                return
            record.status = "indexing"
            record.attempts += 1
            record.error = None
            record.updated_at = now_utc()
            filename = record.filename
            content_sha256 = record.content_sha256
            next_index_version = record.index_version + 1 if record.index_key else 1
            session.add(record)
            session.commit()

        try:
            extracted = await asyncio.to_thread(
                extract_pdf_text,
                path,
                max_pages=self._settings.max_pdf_pages,
                max_characters=self._settings.max_pdf_characters,
            )
            chunks = await asyncio.to_thread(
                chunk_text,
                extracted.text,
                words_per_chunk=self._settings.lore_chunk_words,
                overlap_words=self._settings.lore_chunk_overlap_words,
                max_chunks=self._settings.max_lore_chunks,
            )
            if not chunks:
                raise ValueError("No usable lore chunks were produced.")
            result = await self._rag.index_chunks(
                document_id=document_id,
                filename=filename,
                content_sha256=content_sha256,
                chunks=chunks,
                batch_size=self._settings.lore_embed_batch_size,
                index_version=next_index_version,
            )
            with Session(self._engine) as session:
                record = session.get(LoreDocument, document_id)
                if not record:
                    raise RuntimeError("Lore document was deleted before index activation.")
                record.status = "ready"
                record.chunks = result.chunks
                record.page_count = extracted.page_count
                record.embed_model = self._rag.embed_model
                record.embed_dimension = result.dimension
                record.index_version = next_index_version
                record.index_table = result.table_name
                record.index_key = result.index_key
                record.error = (
                    "Text extraction reached the configured character limit."
                    if extracted.truncated
                    else None
                )
                record.updated_at = now_utc()
                session.add(record)
                session.commit()
                activated = True
            try:
                await self._rag.cleanup_stale(
                    document_id=document_id,
                    active_index_key=result.index_key,
                )
            except Exception:
                logger.warning(
                    "Lore index activated but stale-version cleanup failed",
                    exc_info=True,
                    extra={"document_id": document_id, "index_key": result.index_key},
                )
            log_event(
                logger,
                logging.INFO,
                "Lore document indexed",
                document_id=document_id,
                job_stage="complete",
                duration_ms=round((perf_counter() - started) * 1000),
            )
        except asyncio.CancelledError:
            if result is not None and not activated:
                await self._rag.delete_index(
                    table_name=result.table_name,
                    index_key=result.index_key,
                )
            with Session(self._engine) as session:
                record = session.get(LoreDocument, document_id)
                if record:
                    record.status = "queued"
                    record.error = "Indexing was interrupted during shutdown."
                    record.updated_at = now_utc()
                    session.add(record)
                    session.commit()
            raise
        except Exception as exc:
            if result is not None and not activated:
                try:
                    await self._rag.delete_index(
                        table_name=result.table_name,
                        index_key=result.index_key,
                    )
                except Exception:
                    logger.exception(
                        "Failed to clean inactive lore index",
                        extra={"document_id": document_id, "index_key": result.index_key},
                    )
            with Session(self._engine) as session:
                record = session.get(LoreDocument, document_id)
                if record:
                    record.status = "error"
                    record.error = str(exc)[:1_000]
                    record.updated_at = now_utc()
                    session.add(record)
                    session.commit()
            log_event(
                logger,
                logging.ERROR,
                "Lore document indexing failed",
                document_id=document_id,
                job_stage="failed",
                duration_ms=round((perf_counter() - started) * 1000),
                error_code=exc.__class__.__name__,
            )
