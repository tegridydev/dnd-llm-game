from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from dndllm26.api.deps import get_resources, get_session
from dndllm26.api.schemas import LoreDocumentOut
from dndllm26.core.errors import ConflictError, ValidationError
from dndllm26.core.resources import AppResources
from dndllm26.db.models import LoreDocument, now_utc
from dndllm26.rag.deletion import finalize_lore_deletion
from dndllm26.rag.uploads import store_pdf_upload

router = APIRouter(prefix="/lore", tags=["lore"])


@router.get("", response_model=list[LoreDocumentOut])
def list_lore(session: Session = Depends(get_session)) -> list[LoreDocument]:
    return list(
        session.exec(
            select(LoreDocument)
            .where(LoreDocument.status != "deleting")
            .order_by(LoreDocument.created_at.desc())
        ).all()
    )


@router.post("/upload", response_model=LoreDocumentOut, status_code=201)
async def upload_lore(
    file: UploadFile = File(...),
    resources: AppResources = Depends(get_resources),
    session: Session = Depends(get_session),
) -> LoreDocument:
    try:
        stored = await store_pdf_upload(file, resources.settings)
    except ValidationError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The PDF could not be stored safely.") from exc

    existing = session.exec(
        select(LoreDocument).where(LoreDocument.content_sha256 == stored.sha256)
    ).first()
    if existing:
        if existing.status == "deleting":
            stored.path.unlink(missing_ok=True)
            raise ConflictError("An identical lore document is currently being deleted.")
        if existing.status == "error":
            existing_path = resources.settings.resolved_upload_dir / existing.storage_name
            if not existing_path.is_file():
                stored.path.replace(existing_path)
            else:
                stored.path.unlink(missing_ok=True)
            existing.status = "queued"
            existing.error = None
            existing.attempts = 0
            existing.updated_at = now_utc()
            session.add(existing)
            session.commit()
            session.refresh(existing)
            resources.lore_worker.enqueue(existing.id, force=True)
        else:
            stored.path.unlink(missing_ok=True)
        return existing

    document = LoreDocument(
        filename=stored.display_name,
        storage_name=stored.storage_name,
        content_sha256=stored.sha256,
        size_bytes=stored.size_bytes,
        status="queued",
    )
    try:
        session.add(document)
        session.commit()
        session.refresh(document)
    except IntegrityError:
        session.rollback()
        stored.path.unlink(missing_ok=True)
        winner = session.exec(
            select(LoreDocument).where(LoreDocument.content_sha256 == stored.sha256)
        ).first()
        if winner:
            if winner.status == "deleting":
                raise ConflictError("An identical lore document is currently being deleted.")
            return winner
        raise
    except Exception:
        session.rollback()
        stored.path.unlink(missing_ok=True)
        raise
    if document.id is not None and not resources.lore_worker.enqueue(document.id):
        document.error = "Index queue is currently full. Use Refresh Index to retry."
        document.updated_at = now_utc()
        session.add(document)
        session.commit()
        session.refresh(document)
    return document


@router.post("/refresh-index")
def refresh_lore_index(
    force: bool = False,
    resources: AppResources = Depends(get_resources),
    session: Session = Depends(get_session),
) -> dict[str, object]:
    statement = select(LoreDocument).order_by(LoreDocument.created_at)
    documents = session.exec(statement.where(LoreDocument.status != "deleting")).all()
    queued: list[int] = []
    skipped: list[int] = []
    for document in documents:
        if document.id is None:
            continue
        should_queue = force or document.status in {"queued", "error"}
        if not should_queue:
            continue
        if document.attempts >= resources.settings.lore_worker_max_attempts and not force:
            skipped.append(document.id)
            continue
        if resources.lore_worker.enqueue(document.id, force=force):
            queued.append(document.id)
        else:
            skipped.append(document.id)
    return {"status": "queued", "queued": queued, "skipped": skipped}


@router.delete("/{document_id}")
async def delete_lore_document(
    document_id: int,
    resources: AppResources = Depends(get_resources),
    session: Session = Depends(get_session),
) -> dict[str, object]:
    document = session.get(LoreDocument, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Lore document not found.")
    if document.status == "indexing" or resources.lore_worker.active_document_id == document_id:
        raise HTTPException(status_code=409, detail="Wait for indexing to finish before deleting.")
    if document.status != "deleting":
        document.status = "deleting"
        document.error = None
        document.updated_at = now_utc()
        session.add(document)
        session.commit()
    try:
        await finalize_lore_deletion(
            document_id,
            engine=resources.engine,
            settings=resources.settings,
            rag=resources.rag,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Lore deletion is pending and will be retried at startup.",
        ) from exc
    return {"status": "deleted", "id": document_id}
