"""Quiz history and revision progress stored in the backend (not just the browser)."""
from __future__ import annotations

from sqlalchemy import create_engine, text

from backend.migrations import MIGRATIONS, has_column, migrate
from exam_engine import api as exam_api


def _subject(client, name="S") -> int:
    return client.post("/api/subjects", json={"name": name}).json()["id"]


def _submit(client, sid, topic, results, title=None):
    body = {"subject_id": sid, "answers": [{"topic": topic, "correct": c} for c in results]}
    if title:
        body["title"] = title
    r = client.post("/api/quiz/submit", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def test_quiz_history_newest_first_with_title_and_topics(client):
    sid = _subject(client)
    _submit(client, sid, "SVM", [True, False], title="Support vector machines")
    _submit(client, sid, "Perceptron", [True, True, True])
    hist = client.get(f"/api/quiz/history?subject_id={sid}").json()
    assert [h["title"] for h in hist] == [None, "Support vector machines"]
    assert hist[0]["correct"] == 3 and hist[0]["topics"] == ["Perceptron"] and hist[0]["score"] == 1.0
    assert hist[1]["total"] == 2 and hist[1]["created_at"]
    assert client.get(f"/api/quiz/history?subject_id={sid}&limit=1").json()[0]["topics"] == ["Perceptron"]


def test_history_is_per_subject_and_unknown_subject_is_404(client):
    a, b = _subject(client, "A"), _subject(client, "B")
    _submit(client, a, "SVM", [True])
    assert client.get(f"/api/quiz/history?subject_id={b}").json() == []
    r = client.get("/api/quiz/history?subject_id=999")
    assert r.status_code == 404 and r.json()["error"]["code"] == "subject_not_found"


def test_tick_untick_and_minutes_per_topic(client):
    sid = _subject(client)
    url = f"/api/subjects/{sid}/progress"
    assert client.get(url).json() == {"subject_id": sid, "sessions": [], "minutes_by_topic": {}}
    client.put(url, json={"day": 1, "topic": "SVM", "minutes": 40})
    client.put(url, json={"day": 2, "topic": "SVM", "minutes": 20})
    p = client.put(url, json={"day": 1, "topic": "Perceptron", "minutes": 30}).json()
    assert p["minutes_by_topic"] == {"SVM": 60, "Perceptron": 30}
    assert [(s["day"], s["topic"]) for s in p["sessions"]] == [(1, "Perceptron"), (1, "SVM"), (2, "SVM")]
    again = client.put(url, json={"day": 1, "topic": "SVM", "minutes": 40}).json()  # idempotent
    assert len(again["sessions"]) == 3
    p = client.put(url, json={"day": 2, "topic": "SVM", "done": False}).json()
    assert p["minutes_by_topic"] == {"SVM": 40, "Perceptron": 30} and len(p["sessions"]) == 2


def test_progress_validation_and_404(client):
    sid = _subject(client)
    assert client.put(f"/api/subjects/{sid}/progress", json={"day": 0, "topic": "x"}).status_code == 422
    assert client.put(f"/api/subjects/{sid}/progress", json={"day": 1, "topic": ""}).status_code == 422
    assert client.get("/api/subjects/999/progress").status_code == 404


def test_deleting_a_subject_removes_its_history_and_progress(client):
    sid = _subject(client)
    _submit(client, sid, "SVM", [True])
    client.put(f"/api/subjects/{sid}/progress", json={"day": 1, "topic": "SVM", "minutes": 10})
    assert client.delete(f"/api/subjects/{sid}").status_code == 204
    sid2 = _subject(client, "S")  # same name, new subject: starts empty
    assert client.get(f"/api/subjects/{sid2}/progress").json()["sessions"] == []
    assert client.get(f"/api/quiz/history?subject_id={sid2}").json() == []


def test_progress_reaches_the_engine_when_it_supports_it_and_refreshes_the_plan(client, monkeypatch):
    seen = []

    def with_progress(subject_id, db, progress=None, **kwargs):
        seen.append(progress)
        return {"run": len(seen)}

    monkeypatch.setattr(exam_api, "recommend_revision", with_progress)
    sid = _subject(client)
    assert client.get(f"/api/revision/{sid}").json()["run"] == 1
    client.put(f"/api/subjects/{sid}/progress", json={"day": 1, "topic": "SVM", "minutes": 40})
    assert client.get(f"/api/revision/{sid}").json()["run"] == 2  # ticking a session refreshes the plan
    assert seen == [{}, {"SVM": {"minutes_done": 40}}]


def test_engine_without_progress_parameter_still_works(client, monkeypatch):
    """Today's exam_engine.api.recommend_revision takes days and minutes_per_day but not progress."""
    def current_engine(subject_id, db, days=7, minutes_per_day=90):
        return {"days": days, "minutes_per_day": minutes_per_day}

    monkeypatch.setattr(exam_api, "recommend_revision", current_engine)
    sid = _subject(client)
    client.put(f"/api/subjects/{sid}/progress", json={"day": 1, "topic": "SVM", "minutes": 40})
    r = client.get(f"/api/revision/{sid}").json()
    assert (r["days"], r["minutes_per_day"]) == (7, 90)


def test_migration_1_adds_title_to_an_existing_database(tmp_path):
    eng = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with eng.begin() as c:  # quiz_attempts as it was before this change
        c.execute(text("CREATE TABLE quiz_attempts (id INTEGER PRIMARY KEY, subject_id INTEGER, total INTEGER, "
                       "correct INTEGER, score FLOAT, created_at DATETIME)"))
        c.execute(text("INSERT INTO quiz_attempts (subject_id, total, correct, score) VALUES (1, 4, 3, 0.75)"))
    assert migrate(eng, had_tables=True, migrations=MIGRATIONS) >= 1
    with eng.connect() as c:
        assert has_column(c, "quiz_attempts", "title")
        assert c.execute(text("SELECT score, title FROM quiz_attempts")).one() == (0.75, None)
