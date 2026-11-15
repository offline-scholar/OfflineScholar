"""Revision progress: schedule sessions the student ticked as done. Feeds the revision plan."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.errors import AppError
from backend.models import RevisionProgress, Subject
from backend.schemas import ProgressOut, ProgressSessionOut, ProgressUpdate

router = APIRouter(prefix="/subjects", tags=["progress"])


def _require_subject(db: Session, subject_id: int) -> None:
    if db.get(Subject, subject_id) is None:
        raise AppError(404, "subject_not_found", f"Subject {subject_id} not found")


def minutes_by_topic(db: Session, subject_id: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in db.query(RevisionProgress).filter(RevisionProgress.subject_id == subject_id):
        out[row.topic] = out.get(row.topic, 0) + row.minutes
    return out


def _progress(db: Session, subject_id: int) -> ProgressOut:
    rows = (db.query(RevisionProgress).filter(RevisionProgress.subject_id == subject_id)
            .order_by(RevisionProgress.day, RevisionProgress.topic).all())
    return ProgressOut(
        subject_id=subject_id,
        sessions=[ProgressSessionOut(day=r.day, topic=r.topic, minutes=r.minutes, done_at=r.done_at) for r in rows],
        minutes_by_topic=minutes_by_topic(db, subject_id),
    )


@router.get("/{subject_id}/progress", response_model=ProgressOut)
def get_progress(subject_id: int, db: Session = Depends(get_db)):
    """Sessions ticked as done, and total minutes studied per topic."""
    _require_subject(db, subject_id)
    return _progress(db, subject_id)


@router.put("/{subject_id}/progress", response_model=ProgressOut)
def set_session(subject_id: int, body: ProgressUpdate, db: Session = Depends(get_db)):
    """Tick (`done: true`) or untick (`done: false`) one schedule session. Idempotent."""
    _require_subject(db, subject_id)
    topic = body.topic.strip()
    row = db.query(RevisionProgress).filter_by(subject_id=subject_id, day=body.day, topic=topic).one_or_none()
    if body.done:
        if row is None:
            db.add(RevisionProgress(subject_id=subject_id, day=body.day, topic=topic, minutes=body.minutes))
        else:
            row.minutes = body.minutes
    elif row is not None:
        db.delete(row)
    db.commit()
    return _progress(db, subject_id)
