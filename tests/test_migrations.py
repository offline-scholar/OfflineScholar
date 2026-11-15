"""backend/migrations.py: numbered SQLite schema migrations."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from backend.config import Settings
from backend.database import configure_database
from backend.migrations import MIGRATIONS, Migration, add_column, current_version, has_column, latest_version, migrate


def _engine(tmp_path, name="t.db"):
    return create_engine(f"sqlite:///{(tmp_path / name).as_posix()}")


def _old_db(tmp_path):
    """A database from before the change: a `notes` table without the `pinned` column, and one row."""
    eng = _engine(tmp_path)
    with eng.begin() as c:
        c.execute(text("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)"))
        c.execute(text("INSERT INTO notes (body) VALUES ('keep me')"))
    return eng


ADD_PINNED = [Migration(1, "add notes.pinned", lambda c: add_column(c, "notes", "pinned", "INTEGER NOT NULL DEFAULT 0"))]


def test_fresh_database_is_stamped_latest_and_runs_nothing(tmp_path):
    ran = []
    migrations = [Migration(1, "x", lambda c: ran.append(1)), Migration(2, "y", lambda c: ran.append(2))]
    eng = _engine(tmp_path)
    assert migrate(eng, had_tables=False, migrations=migrations) == 2
    assert ran == []
    with eng.connect() as c:
        assert current_version(c) == 2


def test_existing_database_is_upgraded_in_place_with_a_backup(tmp_path):
    eng = _old_db(tmp_path)
    assert migrate(eng, had_tables=True, migrations=ADD_PINNED) == 1
    with eng.connect() as c:
        assert has_column(c, "notes", "pinned")
        assert c.execute(text("SELECT body, pinned FROM notes")).one() == ("keep me", 0)  # data kept
        assert current_version(c) == 1
    assert (tmp_path / "t.db.bak-v0").is_file()


def test_running_again_is_a_no_op(tmp_path):
    eng = _old_db(tmp_path)
    migrate(eng, had_tables=True, migrations=ADD_PINNED)
    (tmp_path / "t.db.bak-v0").unlink()
    assert migrate(eng, had_tables=True, migrations=ADD_PINNED) == 1
    assert not (tmp_path / "t.db.bak-v0").exists()  # nothing to do, so no backup either


def test_only_newer_migrations_run(tmp_path):
    eng = _old_db(tmp_path)
    migrate(eng, had_tables=True, migrations=ADD_PINNED)
    second = ADD_PINNED + [Migration(2, "add notes.colour", lambda c: add_column(c, "notes", "colour", "VARCHAR(20)"))]
    assert migrate(eng, had_tables=True, migrations=second) == 2
    with eng.connect() as c:
        assert has_column(c, "notes", "colour")


def test_failed_migration_rolls_back_and_keeps_the_version(tmp_path):
    def broken(c):
        add_column(c, "notes", "half_done", "INTEGER")
        raise RuntimeError("boom")

    eng = _old_db(tmp_path)
    migrate(eng, had_tables=True, migrations=ADD_PINNED)
    with pytest.raises(RuntimeError):
        migrate(eng, had_tables=True, migrations=ADD_PINNED + [Migration(2, "broken", broken)])
    with eng.connect() as c:
        assert current_version(c) == 1 and not has_column(c, "notes", "half_done")


def test_duplicate_numbers_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        migrate(_engine(tmp_path), had_tables=False, migrations=[Migration(1, "a", lambda c: None)] * 2)


def test_app_database_gets_a_schema_version(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", embedding_backend="hashing", _env_file=None)
    eng = configure_database(settings)
    with eng.connect() as c:
        assert current_version(c) == latest_version(MIGRATIONS)
