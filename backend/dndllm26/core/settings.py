from __future__ import annotations

from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_ROOT = Path(__file__).resolve().parents[3]
_LOOPBACK_NAMES = {"localhost", "ip6-localhost"}


def _is_loopback_host(value: str) -> bool:
    host = value.strip().strip("[]").lower()
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


class Settings(BaseSettings):
    """Validated runtime configuration.

    Construction is side-effect free. Directories and database resources are created
    explicitly by the FastAPI lifespan handler.
    """

    app_root: Path = APP_ROOT
    log_level: str = "INFO"

    database_url: str = "sqlite:///data/dndllm26.db"
    upload_dir: Path = Path("data/uploads")
    lancedb_dir: Path = Path("data/lancedb")

    ollama_host: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "llama3.2:3b"
    ollama_utility_model: str = "llama3.2:1b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout_seconds: float = Field(default=120.0, ge=5.0, le=600.0)
    ollama_health_timeout_seconds: float = Field(default=3.0, ge=0.5, le=30.0)

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8765, ge=1024, le=65535)
    api_reload: bool = False
    frontend_host: str = "127.0.0.1"
    frontend_port: int = Field(default=5173, ge=1024, le=65535)
    request_max_chars: int = Field(default=2_000, ge=100, le=20_000)

    max_upload_bytes: int = Field(default=32 * 1024 * 1024, ge=1024, le=512 * 1024 * 1024)
    max_pdf_pages: int = Field(default=400, ge=1, le=5_000)
    max_pdf_characters: int = Field(default=4_000_000, ge=10_000, le=50_000_000)
    max_lore_chunks: int = Field(default=2_000, ge=10, le=20_000)
    lore_chunk_words: int = Field(default=700, ge=100, le=2_000)
    lore_chunk_overlap_words: int = Field(default=100, ge=0, le=500)
    lore_embed_batch_size: int = Field(default=16, ge=1, le=128)
    lore_worker_queue_size: int = Field(default=64, ge=1, le=1_000)
    lore_worker_max_attempts: int = Field(default=3, ge=1, le=20)

    model_config = SettingsConfigDict(
        env_file=APP_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    @field_validator("ollama_chat_model", "ollama_embed_model")
    @classmethod
    def non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @field_validator("log_level")
    @classmethod
    def valid_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        return level

    @field_validator("api_host")
    @classmethod
    def loopback_api_only(cls, value: str) -> str:
        if not _is_loopback_host(value):
            raise ValueError(
                "API_HOST must be a loopback address. DNDLLM26 has no remote authentication layer."
            )
        return value.strip()

    @field_validator("frontend_host")
    @classmethod
    def loopback_frontend_only(cls, value: str) -> str:
        if not _is_loopback_host(value):
            raise ValueError("FRONTEND_HOST must be a loopback address")
        return value.strip()

    @field_validator("ollama_host")
    @classmethod
    def valid_ollama_url(cls, value: str) -> str:
        parsed = urlparse(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("OLLAMA_HOST must be an absolute http(s) URL")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.lore_chunk_overlap_words >= self.lore_chunk_words:
            raise ValueError("LORE_CHUNK_OVERLAP_WORDS must be smaller than LORE_CHUNK_WORDS")
        return self

    def resolve_path(self, value: Path) -> Path:
        path = value.expanduser()
        return path.resolve() if path.is_absolute() else (self.app_root / path).resolve()

    @property
    def resolved_upload_dir(self) -> Path:
        return self.resolve_path(self.upload_dir)

    @property
    def resolved_lancedb_dir(self) -> Path:
        return self.resolve_path(self.lancedb_dir)

    @property
    def resolved_database_url(self) -> str:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise ValueError("Only SQLite database URLs are supported")
        raw = self.database_url[len(prefix) :]
        if raw == ":memory:":
            return "sqlite:///:memory:"
        path = Path(raw).expanduser()
        resolved = path.resolve() if path.is_absolute() else (self.app_root / path).resolve()
        return f"sqlite:///{resolved.as_posix()}"

    @property
    def database_path(self) -> Path | None:
        url = self.resolved_database_url
        if url.endswith(":memory:"):
            return None
        return Path(url.removeprefix("sqlite:///"))

    @property
    def allowed_origins(self) -> list[str]:
        return [
            f"http://127.0.0.1:{self.frontend_port}",
            f"http://localhost:{self.frontend_port}",
            f"http://[::1]:{self.frontend_port}",
        ]

    @property
    def max_upload_request_bytes(self) -> int:
        """Allow bounded multipart framing in addition to the configured PDF bytes."""
        return self.max_upload_bytes + 256 * 1024

    def prepare_filesystem(self) -> None:
        for directory in (
            self.resolved_upload_dir,
            self.resolved_lancedb_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if self.database_path is not None:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
