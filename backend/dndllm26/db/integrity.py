from __future__ import annotations

import json
import sqlite3

from dndllm26.core.settings import Settings


def check_integrity() -> dict[str, object]:
    settings = Settings()
    path = settings.database_path
    if path is None:
        return {"database": ":memory:", "integrity": ["ok"], "foreign_keys": []}
    if not path.exists():
        return {"database": str(path), "status": "missing"}
    connection = sqlite3.connect(path)
    try:
        integrity = [str(row[0]) for row in connection.execute("PRAGMA integrity_check")]
        foreign_keys = [list(row) for row in connection.execute("PRAGMA foreign_key_check")]
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        return {
            "database": str(path),
            "schema_version": version,
            "integrity": integrity,
            "foreign_keys": foreign_keys,
            "ok": integrity == ["ok"] and not foreign_keys,
        }
    finally:
        connection.close()


def main() -> None:
    result = check_integrity()
    print(json.dumps(result, indent=2))
    if result.get("ok") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
