"""Personalised revision plan: exam date and daily minutes per subject."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from backend import integration
from exam_engine import api as exam_api

TODAY = date(2026, 11, 16)


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr(integration, "today", lambda: TODAY)


@pytest.fixture
def engine_calls(monkeypatch):
    calls = []

    def recommend_revision(subject_id, db, days=7, minutes_per_day=90):
        calls.append((days, minutes_per_day))
        return {"summary": {"days": days, "minutes_per_day": minutes_per_day}}

    monkeypatch.setattr(exam_api, "recommend_revision", recommend_revision)
    return calls


def _subject(client, name="S") -> int:
    return client.post("/api/subjects", json={"name": name}).json()["id"]


def test_set_partial_update_and_clear(client):
    sid = _subject(client)
    exam = (TODAY + timedelta(days=12)).isoformat()
    r = client.patch(f"/api/subjects/{sid}", json={"exam_date": exam, "minutes_per_day": 60})
    assert r.status_code == 200 and r.json() == {"id": sid, "name": "S", "exam_date": exam, "minutes_per_day": 60}
    r = client.patch(f"/api/subjects/{sid}", json={"minutes_per_day": 45})  # exam date untouched
    assert r.json()["exam_date"] == exam and r.json()["minutes_per_day"] == 45
    r = client.patch(f"/api/subjects/{sid}", json={"exam_date": None})
    assert r.json()["exam_date"] is None and r.json()["minutes_per_day"] == 45
    assert client.get("/api/subjects").json()[0]["minutes_per_day"] == 45


def test_validation_and_unknown_subject(client):
    sid = _subject(client)
    assert client.patch(f"/api/subjects/{sid}", json={"minutes_per_day": 5}).status_code == 422
    assert client.patch(f"/api/subjects/{sid}", json={"exam_date": "not a date"}).status_code == 422
    assert client.patch("/api/subjects/999", json={"minutes_per_day": 60}).status_code == 404


@pytest.mark.parametrize("days_away, days, capped, passed", [
    (None, 7, False, False),   # no exam date: default week
    (10, 10, False, False),    # plan until the exam
    (75, 30, True, False),     # far away: first 30 days
    (0, 7, False, True),       # exam today or past: default week, flagged
    (-3, 7, False, True),
])
def test_plan_length_follows_the_exam_date(client, engine_calls, days_away, days, capped, passed):
    sid = _subject(client)
    body = {"minutes_per_day": 50}
    if days_away is not None:
        body["exam_date"] = (TODAY + timedelta(days=days_away)).isoformat()
    client.patch(f"/api/subjects/{sid}", json=body)
    plan = client.get(f"/api/revision/{sid}").json()
    assert engine_calls == [(days, 50)]
    s = plan["plan_settings"]
    assert (s["days"], s["minutes_per_day"], s["capped"], s["exam_passed"]) == (days, 50, capped, passed)
    assert s["days_until_exam"] == days_away


def test_changing_settings_or_the_day_refreshes_the_cached_plan(client, engine_calls, monkeypatch):
    sid = _subject(client)
    client.patch(f"/api/subjects/{sid}", json={"exam_date": (TODAY + timedelta(days=9)).isoformat()})
    client.get(f"/api/revision/{sid}")
    client.get(f"/api/revision/{sid}")
    assert engine_calls == [(9, 90)]  # cached
    client.patch(f"/api/subjects/{sid}", json={"minutes_per_day": 120})
    client.get(f"/api/revision/{sid}")
    assert engine_calls[-1] == (9, 120)
    monkeypatch.setattr(integration, "today", lambda: TODAY + timedelta(days=1))  # next morning
    client.get(f"/api/revision/{sid}")
    assert engine_calls[-1] == (8, 120)
