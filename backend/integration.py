"""Bridge to Member 2's exam_engine. We only call it; we never reimplement its algorithms.

Expected contract (see docs/INTEGRATION.md): the package `exam_engine.api` exposes

    analyze_subject(subject_id: int, db: sqlalchemy.orm.Session) -> dict      # JSON-serialisable
    recommend_revision(subject_id: int, db: sqlalchemy.orm.Session) -> dict   # JSON-serialisable

Analysis and revision re-read every document of a subject and take ~10 s on a laptop CPU, so their results are
cached until the subject's data changes (see `data_fingerprint`).
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import logging
import threading
from collections import OrderedDict
from typing import Any, Callable

from sqlalchemy import func
from sqlalchemy.orm import Session

from .errors import AppError
from .models import Document, QuizAttempt, RevisionProgress

log = logging.getLogger(__name__)


def _call(func_name: str, subject_id: int, db: Session, **kwargs: Any) -> Any:
    try:
        module = importlib.import_module("exam_engine.api")
    except ModuleNotFoundError as exc:
        if exc.name in ("exam_engine", "exam_engine.api"):
            raise AppError(501, "exam_engine_unavailable",
                           "The exam engine (Member 2) is not installed yet.") from exc
        raise
    fn = getattr(module, func_name, None)
    if fn is None:
        raise AppError(501, "exam_engine_unavailable", f"exam_engine.api.{func_name} is not implemented yet.")
    return fn(subject_id, db=db, **kwargs)


# ----------------------------------------------------------------------------- result cache

def data_fingerprint(subject_id: int, db: Session) -> str:
    """Changes whenever anything the exam engine reads for this subject changes: a document is added, removed or
    finishes processing (status, chunk count), or a quiz result is submitted. Two cheap SQL queries."""
    docs = (db.query(Document.id, Document.status, Document.chunk_count, Document.doc_type)
            .filter(Document.subject_id == subject_id).order_by(Document.id).all())
    quizzes = db.query(func.count(QuizAttempt.id), func.max(QuizAttempt.id)).filter(
        QuizAttempt.subject_id == subject_id).one()
    progress = db.query(func.count(RevisionProgress.id), func.coalesce(func.sum(RevisionProgress.minutes), 0),
                        func.max(RevisionProgress.done_at)).filter(RevisionProgress.subject_id == subject_id).one()
    raw = repr((docs, tuple(quizzes), tuple(progress)))
    return hashlib.sha1(raw.encode()).hexdigest()


class ResultCache:
    """Small in-process LRU cache. One computation per key at a time: concurrent requests for the same key
    (the dashboard asks for analysis and revision together) wait for the first instead of repeating it."""

    def __init__(self, max_entries: int = 32):
        self.max_entries = max_entries
        self._data: OrderedDict[tuple, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._key_locks: dict[tuple, threading.Lock] = {}
        self.hits = self.misses = 0

    def get_or_compute(self, key: tuple, compute: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return copy.deepcopy(self._data[key])  # callers must not mutate the cached copy
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._lock:  # another request may have filled it while we waited
                if key in self._data:
                    self.hits += 1
                    return copy.deepcopy(self._data[key])
            value = compute()
            with self._lock:
                self.misses += 1
                self._data[key] = copy.deepcopy(value)
                while len(self._data) > self.max_entries:
                    self._data.popitem(last=False)
                self._key_locks.pop(key, None)
            return value

    def invalidate(self, key: tuple) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            _saved_revision_key.clear()


cache = ResultCache()
# Which data state each subject's revision_recommendations rows were last written for.
_saved_revision_key: dict[int, tuple] = {}


def _cached(func_name: str, subject_id: int, db: Session) -> Any:
    key = (func_name, subject_id, data_fingerprint(subject_id, db))
    return cache.get_or_compute(key, lambda: _call(func_name, subject_id, db))


def _progress_kwargs(subject_id: int, db: Session) -> dict:
    """Minutes studied per topic, for exam_engine versions whose recommend_revision accepts `progress`
    ({"Topic": {"minutes_done": 30}}, the format exam_engine.revision_engine already understands)."""
    try:
        fn = importlib.import_module("exam_engine.api").recommend_revision
        if "progress" not in inspect.signature(fn).parameters:
            return {}
    except (ImportError, AttributeError, TypeError, ValueError):
        return {}
    from .routes.progress import minutes_by_topic

    return {"progress": {t: {"minutes_done": m} for t, m in minutes_by_topic(db, subject_id).items()}}


def get_analysis(subject_id: int, db: Session) -> Any:
    return _cached("analyze_subject", subject_id, db)


def get_revision(subject_id: int, db: Session) -> Any:
    """recommend_revision also saves the plan to revision_recommendations. A cached plan is only reused when
    those rows were last written for this same data; otherwise (e.g. a document was added, then deleted again)
    it is recomputed so the table matches what the API returns."""
    key = ("recommend_revision", subject_id, data_fingerprint(subject_id, db))
    if _saved_revision_key.get(subject_id) != key:
        cache.invalidate(key)
    result = cache.get_or_compute(key, lambda: _call("recommend_revision", subject_id, db, **_progress_kwargs(subject_id, db)))
    _saved_revision_key[subject_id] = key
    return result


def generate_quiz(subject_id: int, topic: str, n_questions: int, db: Session) -> Any:
    return _call("generate_quiz_for_subject", subject_id, db, topic=topic, n_questions=n_questions)  # never cached
