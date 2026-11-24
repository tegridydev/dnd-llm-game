from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
import sys

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from dndllm26.core.errors import LoreProcessingError
from dndllm26.rag.pdf import extract_pdf_isolated


def write_lore_pdf(path: Path) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica"),
                        }
                    ),
                }
            ),
        }
    )
    contents = DecodedStreamObject()
    contents.set_data(b"BT /F1 18 Tf 30 600 Td (A bell rings at the Lantern Gate.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(contents)
    with path.open("wb") as output:
        writer.write(output)


def test_worker_extracts_real_pdf_text(tmp_path: Path) -> None:
    path = tmp_path / "lore.pdf"
    write_lore_pdf(path)
    result = asyncio.run(extract_pdf_isolated(path, max_pages=3, max_characters=300, timeout=6))
    assert "Lantern Gate" in result.text
    assert result.page_count == 1


def test_extraction_kills_stalled_worker(tmp_path: Path, monkeypatch) -> None:
    create = asyncio.create_subprocess_exec
    children = []

    async def launch(*_args, **_kwargs):
        child = await create(
            sys.executable, "-c", "import time; time.sleep(30)", stdout=subprocess.PIPE
        )
        children.append(child)
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)

    async def run():
        with pytest.raises(LoreProcessingError, match="time limit"):
            await extract_pdf_isolated(
                tmp_path / "lore.pdf", max_pages=3, max_characters=300, timeout=0.4
            )
        assert children[0].returncode is not None

    asyncio.run(run())
