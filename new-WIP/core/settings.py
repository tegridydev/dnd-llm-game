import logging
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import HttpUrl

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    ollama_host: HttpUrl = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    pdf_folder: Path = Path("pdf")
    vector_index_dir: Path = Path("vector_index")
    turn_limit: int = 10
    chunk_size: int = 500
    chunk_overlap: int = 50
    enable_rag: bool = True

    class Config:
        env_file = ".env"
        validate_assignment = True

settings = Settings()

# Ensure data dirs exist
for folder in (settings.pdf_folder, settings.vector_index_dir):
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("Failed to create folder %s", folder)
