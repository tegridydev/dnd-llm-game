from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile

from dndllm26.core.errors import ValidationError
from dndllm26.core.settings import Settings
from dndllm26.rag.uploads import safe_display_name, store_pdf_upload


def test_display_name_removes_path_components() -> None:
    assert safe_display_name("../../outside/realm.pdf") == "realm.pdf"
    assert safe_display_name(r"..\\outside\\realm.pdf") == "realm.pdf"


def test_display_name_preserves_suffix_and_rejects_named_non_pdf() -> None:
    assert safe_display_name("x" * 300 + ".pdf").endswith(".pdf")
    assert len(safe_display_name("x" * 300 + ".pdf")) == 255
    assert safe_display_name("...") == "lore.pdf"
    assert safe_display_name(None) == "lore.pdf"
    with pytest.raises(ValidationError):
        safe_display_name("notes.txt")


def test_pdf_upload_uses_generated_storage_name(tmp_path: Path) -> None:
    settings = Settings(
        app_root=tmp_path,
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "data" / "uploads",
        lancedb_dir=tmp_path / "data" / "lancedb",
        database_url=f"sqlite:///{tmp_path / 'data' / 'test.db'}",
    )
    settings.prepare_filesystem()
    upload = UploadFile(filename="../../realm.pdf", file=BytesIO(b"%PDF-1.7\nexample"))
    stored = asyncio.run(store_pdf_upload(upload, settings))
    assert stored.display_name == "realm.pdf"
    assert stored.storage_name.endswith(".pdf")
    assert "/" not in stored.storage_name
    assert stored.path.parent == settings.resolved_upload_dir
    assert stored.path.read_bytes().startswith(b"%PDF-")


def test_non_pdf_signature_is_rejected(tmp_path: Path) -> None:
    settings = Settings(
        app_root=tmp_path,
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "data" / "uploads",
        lancedb_dir=tmp_path / "data" / "lancedb",
        database_url=f"sqlite:///{tmp_path / 'data' / 'test.db'}",
    )
    settings.prepare_filesystem()
    upload = UploadFile(filename="fake.pdf", file=BytesIO(b"not actually a pdf"))
    with pytest.raises(ValidationError):
        asyncio.run(store_pdf_upload(upload, settings))
    assert list(settings.resolved_upload_dir.iterdir()) == []
