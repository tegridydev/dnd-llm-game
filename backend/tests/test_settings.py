from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dndllm26.core.settings import Settings


def test_settings_reject_remote_api_binding(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Settings(app_root=tmp_path, api_host="0.0.0.0")


def test_settings_resolve_paths_against_app_root(tmp_path: Path) -> None:
    settings = Settings(app_root=tmp_path, upload_dir=Path("runtime/uploads"))
    assert settings.resolved_upload_dir == (tmp_path / "runtime/uploads").resolve()
    assert settings.database_path == (tmp_path / "data" / "dndllm26.db").resolve()


def test_remote_chat_and_remote_embeddings_have_separate_opt_ins() -> None:
    with pytest.raises(ValidationError, match="remote opt-in"):
        Settings(llm_provider="anthropic", llm_api_key="example-key")
    with pytest.raises(ValidationError, match="LLM_ALLOW_REMOTE_EMBEDDINGS"):
        Settings(
            llm_provider="openai",
            llm_api_key="example-key",
            llm_allow_remote=True,
            llm_embed_provider="same",
        )
    assert Settings(
        llm_provider="openai",
        llm_api_key="example-key",
        llm_allow_remote=True,
        llm_embed_provider="same",
        llm_allow_remote_embeddings=True,
    ).llm_allow_remote_embeddings


def test_only_local_http_and_fixed_hosts_for_named_cloud_presets() -> None:
    with pytest.raises(ValidationError, match="remote opt-in"):
        Settings(
            llm_provider="compatible", llm_base_url="http://example.net/v1", llm_allow_remote=True
        )
    with pytest.raises(ValidationError, match="custom hosted"):
        Settings(
            llm_provider="openai", llm_base_url="https://example.net/v1", llm_allow_remote=True
        )
    with pytest.raises(ValidationError, match="without credentials"):
        Settings(llm_provider="compatible", llm_base_url="http://someone@localhost:1234/v1")
