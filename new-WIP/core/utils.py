import logging
import pickle
import re
from pathlib import Path
from typing import List

import hnswlib
import numpy as np

from .pdf_utils import load_all_pdf_texts
from .embeddings import embed_texts, EMBED_DIM
from .settings import settings

logger = logging.getLogger(__name__)

# ——— Context utilities —————————————————————————————————

def last_sentences(text: str, n: int) -> str:
    """
    Grab the last n sentences (naïve split) for context truncation.
    """
    sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
    return " ".join(sentences[-n:])

# (Optional) naive char-based fallback
def semantic_truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[-max_chars:]


# ——— RAG index management —————————————————————————————

INDEX_FILE = "hnsw_index.bin"
TEXTS_FILE = "texts.pkl"

_index: hnswlib.Index | None = None
_texts: List[str] = []
_last_mtimes: dict[str, float] = {}

def _needs_rebuild() -> bool:
    rebuild = False
    for pdf in settings.pdf_folder.glob("*.pdf"):
        m = pdf.stat().st_mtime
        if pdf.name not in _last_mtimes or _last_mtimes[pdf.name] < m:
            _last_mtimes[pdf.name] = m
            rebuild = True
    return rebuild

def build_index() -> None:
    """
    Load or (re)build the HNSW index of PDF embeddings.
    """
    global _index, _texts
    docs = load_all_pdf_texts()
    if not docs:
        logger.info("No PDFs to index.")
        return

    idx_dir = settings.vector_index_dir
    idx_path = idx_dir / INDEX_FILE
    txts_path = idx_dir / TEXTS_FILE

    # Load existing
    if idx_path.exists() and txts_path.exists() and not _needs_rebuild():
        try:
            _texts = pickle.loads(txts_path.read_bytes())
            _index = hnswlib.Index(space='l2', dim=EMBED_DIM)
            _index.load_index(str(idx_path))
            logger.info("Loaded vector index (%d entries)", _index.get_current_count())
            return
        except Exception as e:
            logger.warning("Could not load index: %s. Rebuilding.", e)

    # Build new index
    embs = embed_texts(docs)
    arr = np.array(embs, dtype="float32")
    _index = hnswlib.Index(space='l2', dim=arr.shape[1])
    _index.init_index(max_elements=len(arr), ef_construction=200, M=16)
    _index.add_items(arr, np.arange(len(arr)))
    _index.set_ef(50)
    _texts = docs

    try:
        with open(txts_path, "wb") as f:
            pickle.dump(_texts, f)
        _index.save_index(str(idx_path))
        logger.info("Built and saved vector index (%d chunks)", len(_texts))
    except Exception as e:
        logger.exception("Failed to save index: %s", e)

def retrieve(query: str, k: int = 3) -> List[str]:
    """
    Return the top-k PDF text chunks for `query`, if RAG is enabled.
    """
    if not settings.enable_rag:
        return []
    global _index, _texts
    if _index is None or _needs_rebuild():
        build_index()

    try:
        q_emb = np.array(embed_texts([query]), dtype="float32")
        labels, _ = _index.knn_query(q_emb, k=k)
        return [ _texts[i] for i in labels[0] ]
    except Exception as e:
        logger.exception("Retrieve error for %r: %s", query, e)
        return []
