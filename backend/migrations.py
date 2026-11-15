"""Numbered schema migrations for the local SQLite database (no external tool).

How it works
- `schema_version` holds the number of the last migration applied.
- A brand-new database is created by `Base.metadata.create_all` with the current models, so it is stamped with
  the latest version and nothing runs.
- An existing database runs every migration newer than its version, in order, each in its own transaction.
  Before the first one runs, the database file is copied to `<db>.bak-v<old version>`.
- A database from before migrations existed (tables but no `schema_version`) counts as version 0.

Adding a migration: change `backend/models.py`, then append `Migration(<next number>, "<what>", <function>)` to
MIGRATIONS. The function gets a SQLAlchemy Connection inside a transaction. New *tables* are created by
create_all already; migrations are for changes to tables that already exist (new columns, data fixes).
Use the helpers below so a migration is safe to run on a database that already has the change.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Migration:
    version: int
    description: str
    apply: Callable[[Connection], None]


# ---------------------------------------------------------------------------- helpers for migrations

def has_column(conn: Connection, table: str, column: str) -> bool:
    return any(c["name"] == column for c in inspect(conn).get_columns(table))


def add_column(conn: Connection, table: str, column: str, ddl: str) -> None:
    """ALTER TABLE ... ADD COLUMN, skipped if the column is already there. `ddl` e.g. "INTEGER NOT NULL DEFAULT 0".
    Also skipped when the table doesn't exist: create_all builds new tables with every current column."""
    if inspect(conn).has_table(table) and not has_column(conn, table, column):
        conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {ddl}'))


# ---------------------------------------------------------------------------- the migrations

MIGRATIONS: list[Migration] = [
    Migration(1, "quiz_attempts.title (quiz history shows the topic)",
              lambda c: add_column(c, "quiz_attempts", "title", "VARCHAR(200)")),
    Migration(2, "subjects.exam_date and subjects.minutes_per_day (personalised revision plan)",
              lambda c: (add_column(c, "subjects", "exam_date", "DATE"),
                         add_column(c, "subjects", "minutes_per_day", "INTEGER"))),
]


# ---------------------------------------------------------------------------- runner

def latest_version(migrations: list[Migration] | None = None) -> int:
    migrations = MIGRATIONS if migrations is None else migrations
    return max((m.version for m in migrations), default=0)


def current_version(conn: Connection) -> int | None:
    """None when the database has no schema_version table yet."""
    if not inspect(conn).has_table("schema_version"):
        return None
    row = conn.execute(text("SELECT version FROM schema_version")).first()
    return int(row[0]) if row else 0


def _set_version(conn: Connection, version: int) -> None:
    conn.execute(text("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"))
    conn.execute(text("DELETE FROM schema_version"))
    conn.execute(text("INSERT INTO schema_version (version) VALUES (:v)"), {"v": version})


def _backup(engine: Engine, version: int) -> Path | None:
    db = engine.url.database
    if engine.url.get_backend_name() != "sqlite" or not db or db == ":memory:":
        return None
    src = Path(db)
    if not src.is_file():
        return None
    dst = src.with_name(f"{src.name}.bak-v{version}")
    shutil.copy2(src, dst)
    return dst


def migrate(engine: Engine, had_tables: bool, migrations: list[Migration] | None = None) -> int:
    """Bring the database up to the latest schema. `had_tables`: did it contain tables before create_all ran?
    Returns the schema version afterwards."""
    migrations = sorted(MIGRATIONS if migrations is None else migrations, key=lambda m: m.version)
    versions = [m.version for m in migrations]
    if len(set(versions)) != len(versions) or any(v < 1 for v in versions):
        raise ValueError(f"Migration numbers must be unique and >= 1, got {versions}")
    latest = latest_version(migrations)

    with engine.begin() as conn:
        version = current_version(conn)
        if version is None:
            # Fresh database: create_all just built the latest schema. Old database: baseline is 0.
            version = 0 if had_tables else latest
            _set_version(conn, version)

    pending = [m for m in migrations if m.version > version]
    if not pending:
        return version
    if version > latest:
        log.warning("Database schema v%d is newer than this code (v%d)", version, latest)
        return version

    backup = _backup(engine, version)
    log.info("Migrating database from v%d to v%d (backup: %s)", version, latest, backup)
    for m in pending:
        # Explicit BEGIN/COMMIT: Python's sqlite3 driver otherwise commits DDL (ALTER TABLE) on its own, so a
        # failing migration would leave half its changes behind. SQLite itself rolls back DDL correctly.
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.exec_driver_sql("BEGIN")
            try:
                m.apply(conn)
                _set_version(conn, m.version)
            except Exception:
                conn.exec_driver_sql("ROLLBACK")
                log.error("Migration %d (%s) failed; database left at v%d", m.version, m.description, version)
                raise
            conn.exec_driver_sql("COMMIT")
        log.info("Applied migration %d: %s", m.version, m.description)
        version = m.version
    return version
