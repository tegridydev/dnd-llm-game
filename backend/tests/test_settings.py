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
