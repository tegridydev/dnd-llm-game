from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from dndllm26.core.errors import LoreProcessingError


@dataclass(frozen=True, slots=True)
class ExtractedPdf:
    text: str
    page_count: int
    truncated: bool


def extract_pdf_text(path: Path, *, max_pages: int, max_characters: int) -> ExtractedPdf:
    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            result = reader.decrypt("")
            if not result:
                raise LoreProcessingError("Password-protected PDFs are not supported.")
        page_count = len(reader.pages)
        if page_count > max_pages:
            raise LoreProcessingError(
                f"PDF contains {page_count} pages; the configured limit is {max_pages}."
            )
        parts: list[str] = []
        total = 0
        truncated = False
        for page in reader.pages:
            value = page.extract_text() or ""
            remaining = max_characters - total
            if remaining <= 0:
                truncated = True
                break
            if len(value) > remaining:
                value = value[:remaining]
                truncated = True
            parts.append(value)
            total += len(value)
        text = "\n".join(parts).strip()
    except LoreProcessingError:
        raise
    except Exception as exc:
        raise LoreProcessingError("The PDF could not be parsed safely.", detail=str(exc)) from exc

    if not text:
        raise LoreProcessingError(
            "No extractable text was found. Scanned/image-only PDFs require OCR and are not supported."
        )
    return ExtractedPdf(text=text, page_count=page_count, truncated=truncated)


def chunk_text(
    text: str,
    *,
    words_per_chunk: int,
    overlap_words: int,
    max_chunks: int,
) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words) and len(chunks) < max_chunks:
        end = min(start + words_per_chunk, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks
