"""The single local student profile (name and study preferences), stored in SQLite instead of the browser."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.errors import AppError
from backend.models import Profile
from backend.schemas import ProfileOut, ProfileUpdate

router = APIRouter(prefix="/profile", tags=["profile"])


def get_or_create_profile(db: Session) -> Profile:
    profile = db.get(Profile, 1)
    if profile is None:
        profile = Profile(id=1)
        db.add(profile)
        db.commit()
    return profile


@router.get("", response_model=ProfileOut)
def read_profile(db: Session = Depends(get_db)):
    return get_or_create_profile(db)


@router.patch("", response_model=ProfileOut)
def update_profile(body: ProfileUpdate, db: Session = Depends(get_db)):
    profile = get_or_create_profile(db)
    for field in body.model_fields_set:
        value = getattr(body, field)
        if value is None:
            raise AppError(422, "invalid_profile", f"{field} cannot be null")
        setattr(profile, field, value)
    db.commit()
    return profile
