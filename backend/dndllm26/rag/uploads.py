from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from uuid import uuid4

from fastapi import UploadFile

from dndllm26.core.errors import ValidationError
from dndllm26.core.settings import Settings

_READ_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class StoredUpload:
    display_name: str
    storage_name: str
    sha256: str
    size_bytes: int
    path: Path


def safe_display_name(value: str | None) -> str:
    original = (value or "lore.pdf").replace("\\", "/").split("/")[-1]
    original = "".join(character for character in original if character.isprintable())
    original = re.sub(r"\s+", " ", original).strip().strip(".")
    if not original:
        original = "lore.pdf"
    if not original.lower().endswith(".pdf"):
        raise ValidationError("Lore uploads must use a .pdf filename.")
    stem = original[:-4].rstrip(".") or "lore"
    return stem[:251] + ".pdf"


async def store_pdf_upload(upload: UploadFile, settings: Settings) -> StoredUpload:
    display_name = safe_display_name(upload.filename)
    storage_name = f"{uuid4().hex}.pdf"
    destination = settings.resolved_upload_dir / storage_name
    temporary = destination.with_suffix(".pdf.part")
    digest = hashlib.sha256()
    size = 0
    header = bytearray()

    try:
        with temporary.open("xb") as handle:
            while True:
                chunk = await upload.read(_READ_SIZE)
                if not chunk:
                    break
                if len(header) < 1024:
                    header.extend(chunk[: 1024 - len(header)])
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise ValidationError(
                        f"PDF exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB upload limit."
                    )
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if size == 0:
            raise ValidationError("Uploaded PDF is empty.")
        if b"%PDF-" not in bytes(header):
            raise ValidationError("Uploaded file does not contain a valid PDF header.")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    return StoredUpload(
        display_name=display_name,
        storage_name=storage_name,
        sha256=digest.hexdigest(),
        size_bytes=size,
        path=destination,
    )
