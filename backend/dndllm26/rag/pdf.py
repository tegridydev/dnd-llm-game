from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys

from pypdf import PdfReader

from dndllm26.core.errors import LoreProcessingError


@dataclass(frozen=True, slots=True)
class ExtractedPdf:
    text: str
    page_count: int
    truncated: bool


def extract_pdf_text(path: Path, *, max_pages: int, max_characters: int) -> ExtractedPdf:
    try:
        with path.open("rb") as stream:
            reader = PdfReader(stream, strict=False)
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
                remaining = max_characters - total - bool(parts)
                if remaining <= 0:
                    truncated = True
                    break
                if len(value) > remaining:
                    value = value[:remaining]
                    truncated = True
                total += len(value) + bool(parts)
                parts.append(value)
            text = "\n".join(parts).strip()
    except LoreProcessingError:
        raise
    except Exception as exc:
        raise LoreProcessingError("The PDF could not be parsed safely.") from exc

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
    if not 0 <= overlap_words < words_per_chunk:
        raise ValueError("Lore chunk overlap must be below the chunk size.")
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


async def extract_pdf_isolated(
    path: Path, *, max_pages: int, max_characters: int, timeout: float
) -> ExtractedPdf:
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL"}
    }
    worker = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-m",
        "dndllm26.rag.pdf",
        str(path),
        str(max_pages),
        str(max_characters),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
    )
    transfer = asyncio.create_task(worker.communicate())
    try:
        output, _ = await asyncio.wait_for(asyncio.shield(transfer), timeout)
    except TimeoutError as exc:
        raise LoreProcessingError("PDF extraction exceeded its time limit.") from exc
    finally:
        if worker.returncode is None:
            with suppress(ProcessLookupError):
                worker.kill()
            await transfer
    try:
        result = json.loads(output)
    except (ValueError, UnicodeError) as exc:
        raise LoreProcessingError("The PDF could not be parsed within the worker limits.") from exc
    if worker.returncode != 0:
        raise LoreProcessingError(result.get("error", "The PDF could not be parsed safely."))
    return ExtractedPdf(**result)


def _run_worker() -> int:
    if sys.platform == "linux":
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = min(n for n in (soft, hard, 768 * 1024 * 1024) if n >= 0)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    try:
        result = extract_pdf_text(
            Path(sys.argv[1]), max_pages=int(sys.argv[2]), max_characters=int(sys.argv[3])
        )
    except LoreProcessingError as exc:
        print(json.dumps({"error": exc.message}))
        return 1
    print(json.dumps(asdict(result), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_worker())
