"""Caching of /analysis and /revision: reuse results until the subject's data changes."""
from __future__ import annotations

import threading
import time

import pytest

from backend.integration import ResultCache
from exam_engine import api as exam_api

from .conftest import make_pdf, upload


@pytest.fixture
def calls(monkeypatch):
    """Replace the (slow) exam-engine entry points with counting stand-ins."""
    counts = {"analyze_subject": 0, "recommend_revision": 0}

    def fake(name):
        def run(subject_id, db, **kwargs):
            counts[name] += 1
            return {"subject": {"id": subject_id}, "run": counts[name], "notes": []}
        return run

    monkeypatch.setattr(exam_api, "analyze_subject", fake("analyze_subject"))
    monkeypatch.setattr(exam_api, "recommend_revision", fake("recommend_revision"))
    return counts


def _subject(client, name="S") -> int:
    return client.post("/api/subjects", json={"name": name}).json()["id"]


def test_second_request_is_served_from_cache(client, calls):
    sid = _subject(client)
    first = client.get(f"/api/analysis/{sid}").json()
    second = client.get(f"/api/analysis/{sid}").json()
    assert first == second and calls["analyze_subject"] == 1
    client.get(f"/api/revision/{sid}")
    client.get(f"/api/revision/{sid}")
    assert calls["recommend_revision"] == 1


def test_new_or_deleted_document_recomputes(client, calls, tmp_path):
    sid = _subject(client)
    client.get(f"/api/analysis/{sid}")
    doc = upload(client, make_pdf(tmp_path / "Lecture.pdf", ["Gradient descent updates the weights."]),
                 subject_id=sid, doc_type="lecture")
    assert client.get(f"/api/analysis/{sid}").json()["run"] == 2
    assert client.get(f"/api/analysis/{sid}").json()["run"] == 2  # unchanged data: cached again
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    # Back to the original data: the analysis computed for exactly that data is reused.
    assert client.get(f"/api/analysis/{sid}").json()["run"] == 1 and calls["analyze_subject"] == 2


def test_revision_is_recomputed_when_its_saved_plan_is_for_other_data(client, calls, tmp_path):
    """revision_recommendations must match the returned plan, so add-then-delete recomputes instead of reusing."""
    sid = _subject(client)
    assert client.get(f"/api/revision/{sid}").json()["run"] == 1
    doc = upload(client, make_pdf(tmp_path / "Lecture.pdf", ["Gradient descent updates the weights."]),
                 subject_id=sid, doc_type="lecture")
    assert client.get(f"/api/revision/{sid}").json()["run"] == 2
    client.delete(f"/api/documents/{doc['id']}")
    assert client.get(f"/api/revision/{sid}").json()["run"] == 3
    assert client.get(f"/api/revision/{sid}").json()["run"] == 3  # now unchanged: cached


def test_quiz_result_recomputes_revision(client, calls):
    sid = _subject(client)
    client.get(f"/api/revision/{sid}")
    r = client.post("/api/quiz/submit", json={"subject_id": sid, "answers": [{"topic": "SVM", "correct": False}]})
    assert r.status_code == 201
    assert client.get(f"/api/revision/{sid}").json()["run"] == 2


def test_subjects_are_cached_separately(client, calls):
    a, b = _subject(client, "A"), _subject(client, "B")
    assert client.get(f"/api/analysis/{a}").json()["subject"]["id"] == a
    assert client.get(f"/api/analysis/{b}").json()["subject"]["id"] == b
    assert calls["analyze_subject"] == 2


def test_cached_value_cannot_be_changed_by_a_caller():
    c = ResultCache()
    v = c.get_or_compute(("k",), lambda: {"topics": [1]})
    v["topics"].append(2)
    assert c.get_or_compute(("k",), lambda: {"topics": ["recomputed"]}) == {"topics": [1]}


def test_concurrent_requests_compute_once_and_cache_is_bounded():
    c = ResultCache(max_entries=2)
    runs = []

    def slow():
        runs.append(1)
        time.sleep(0.2)
        return {"ok": True}

    threads = [threading.Thread(target=c.get_or_compute, args=(("same",), slow)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(runs) == 1 and c.hits == 3
    for k in ("a", "b", "c"):
        c.get_or_compute((k,), lambda: k)
    assert ("same",) not in c._data and len(c._data) == 2  # least recently used entry dropped
