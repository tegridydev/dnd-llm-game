from __future__ import annotations

import asyncio

import pytest
from sqlmodel import Session

from dndllm26.db.models import LoreDocument
from dndllm26.rag.deletion import finalize_lore_deletion


class FakeRag:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.deleted: list[int] = []

    async def delete_document(self, document_id: int) -> None:
        self.deleted.append(document_id)
        if self.fail:
            raise RuntimeError("vector cleanup failed")


def make_deleting_document(engine, settings) -> int:
    settings.prepare_filesystem()
    path = settings.resolved_upload_dir / "stored.pdf"
    path.write_bytes(b"%PDF-1.7\n")
    with Session(engine) as session:
        document = LoreDocument(
            filename="lore.pdf",
            storage_name="stored.pdf",
            content_sha256="a" * 64,
            size_bytes=9,
            status="deleting",
        )
        session.add(document)
        session.commit()
        session.refresh(document)
        return document.id or 0


def test_deletion_finalization_is_idempotent(engine, settings) -> None:
    document_id = make_deleting_document(engine, settings)
    rag = FakeRag()
    assert asyncio.run(
        finalize_lore_deletion(
            document_id,
            engine=engine,
            settings=settings,
            rag=rag,  # type: ignore[arg-type]
        )
    )
    assert (
        asyncio.run(
            finalize_lore_deletion(
                document_id,
                engine=engine,
                settings=settings,
                rag=rag,  # type: ignore[arg-type]
            )
        )
        is False
    )
    with Session(engine) as session:
        assert session.get(LoreDocument, document_id) is None


def test_failed_deletion_keeps_durable_retry_state(engine, settings) -> None:
    document_id = make_deleting_document(engine, settings)
    with pytest.raises(RuntimeError, match="vector cleanup failed"):
        asyncio.run(
            finalize_lore_deletion(
                document_id,
                engine=engine,
                settings=settings,
                rag=FakeRag(fail=True),  # type: ignore[arg-type]
            )
        )
    with Session(engine) as session:
        document = session.get(LoreDocument, document_id)
        assert document is not None
        assert document.status == "deleting"
        assert document.error and "pending retry" in document.error
