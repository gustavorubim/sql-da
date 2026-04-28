"""Read-only SQL execution helpers."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

FORBIDDEN_SQL = {
    "alter",
    "attach",
    "create",
    "delete",
    "detach",
    "drop",
    "insert",
    "pragma",
    "replace",
    "truncate",
    "update",
    "vacuum",
}


class SQLSafetyError(ValueError):
    """Raised when a SQL statement violates read-only safety rules."""


class ReadOnlySQLRunner:
    """A small SQLite runner that only accepts bounded SELECT-style queries."""

    def __init__(self, db_path: str | Path, max_rows: int = 100):
        self.db_path = Path(db_path)
        self.max_rows = max_rows

    def validate(self, sql: str) -> str:
        cleaned = _strip_sql_comments(sql).strip().rstrip(";")
        if not cleaned:
            raise SQLSafetyError("SQL query is empty")
        if ";" in cleaned:
            raise SQLSafetyError("Only one SQL statement is allowed")
        first_token = cleaned.split(maxsplit=1)[0].lower()
        if first_token not in {"select", "with"}:
            raise SQLSafetyError("Only SELECT or WITH queries are allowed")
        tokens = {token.lower() for token in re.findall(r"[A-Za-z_]+", cleaned)}
        forbidden = sorted(tokens & FORBIDDEN_SQL)
        if forbidden:
            raise SQLSafetyError(f"Forbidden SQL keyword(s): {', '.join(forbidden)}")
        return cleaned

    def run(self, sql: str) -> list[dict[str, Any]]:
        cleaned = self.validate(sql)
        limited = f"SELECT * FROM ({cleaned}) AS safe_query LIMIT {int(self.max_rows)}"
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(limited).fetchall()]

    def schema(self) -> dict[str, list[str]]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
            schema: dict[str, list[str]] = {}
            for (name,) in rows:
                schema[name] = [row[1] for row in conn.execute(f"PRAGMA table_info({name})")]
            return schema


def _strip_sql_comments(sql: str) -> str:
    without_line_comments = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    return re.sub(r"/\*.*?\*/", "", without_line_comments, flags=re.DOTALL)
