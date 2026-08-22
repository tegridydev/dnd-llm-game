from __future__ import annotations

import json
from pathlib import Path
import tomllib


def test_package_and_documented_versions_match() -> None:
    root = Path(__file__).resolve().parents[2]
    python_version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    root_version = json.loads((root / "package.json").read_text())["version"]
    frontend_version = json.loads((root / "frontend" / "package.json").read_text())["version"]
    security = (root / "SECURITY.md").read_text()
    assert python_version == root_version == frontend_version
    assert f"v{python_version}" in security
