import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model = None
for repo in ("intfloat/e5-small", "sentence-transformers/all-MiniLM-L6-v2"):
    try:
        _model = SentenceTransformer(repo, local_files_only=True)
        logger.info("Loaded embedding model %s", repo)
        break
    except Exception:
        logger.debug("Could not load %s", repo)

if _model is None:
    logger.error("No embedding model loaded; RAG will fallback to zeros")

EMBED_DIM = _model.get_sentence_embedding_dimension() if _model else 384

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    if _model is None:
        return [[0.0]*EMBED_DIM for _ in texts]
    try:
        return _model.encode(texts, show_progress_bar=False, batch_size=64).tolist()
    except Exception as e:
        logger.exception("Embed error: %s", e)
        return [[0.0]*EMBED_DIM for _ in texts]
