from __future__ import annotations

import asyncio

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from dndllm26.core.settings import Settings
from dndllm26.db.models import CampaignLore, LoreDocument, now_utc
from dndllm26.rag.store import RagStore
from dndllm26.rag.uploads import managed_pdf_path


async def finalize_lore_deletion(
    document_id: int,
    *,
    engine: Engine,
    settings: Settings,
    rag: RagStore,
) -> bool:
    """Idempotently finish a deletion whose durable intent is already committed."""
    with Session(engine) as session:
        document = session.get(LoreDocument, document_id)
        if not document:
            return False
        if document.status != "deleting":
            return False
        storage_name = document.storage_name
    try:
        await rag.delete_document(document_id)
        path = managed_pdf_path(settings, storage_name)
        if path is not None:
            await asyncio.to_thread(path.unlink, missing_ok=True)
        with Session(engine) as session:
            document = session.get(LoreDocument, document_id)
            if not document:
                return True
            links = session.exec(
                select(CampaignLore).where(CampaignLore.lore_document_id == document_id)
            ).all()
            for link in links:
                session.delete(link)
            session.delete(document)
            session.commit()
        return True
    except Exception as exc:
        with Session(engine) as session:
            document = session.get(LoreDocument, document_id)
            if document:
                document.error = f"Deletion is pending retry: {exc}"[:1_000]
                document.updated_at = now_utc()
                session.add(document)
                session.commit()
        raise
