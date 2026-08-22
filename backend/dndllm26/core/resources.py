from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy.engine import Engine

from dndllm26.core.settings import Settings

if TYPE_CHECKING:
    from dndllm26.llm.ollama_client import OllamaService
    from dndllm26.rag.store import RagStore
    from dndllm26.rag.worker import LoreIndexWorker


@dataclass(slots=True)
class AppResources:
    settings: Settings
    engine: Engine
    ollama: "OllamaService"
    rag: "RagStore"
    lore_worker: "LoreIndexWorker"
    narration_style: str = "balanced"
