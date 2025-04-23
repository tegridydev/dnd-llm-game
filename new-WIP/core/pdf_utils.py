import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import pdfplumber

from .settings import settings

logger = logging.getLogger(__name__)

def list_pdfs() -> List[Path]:
    folder = settings.pdf_folder
    if not folder.is_dir():
        logger.warning("%s is not a directory", folder)
        return []
    return [p for p in folder.iterdir() if p.suffix.lower()==".pdf"]

def extract_text_from_pdf(pdf_path: Path) -> List[str]:
    pages: List[str] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
    except Exception as e:
        logger.exception("PDF read error %s: %s", pdf_path, e)
    return pages

def load_all_pdf_texts() -> List[str]:
    files = list_pdfs()
    if not files:
        return []
    texts: List[str] = []
    with ThreadPoolExecutor() as exe:
        futures = {exe.submit(extract_text_from_pdf, p): p for p in files}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                texts.extend(fut.result())
            except Exception as e:
                logger.error("Error %s: %s", p, e)
    return texts
