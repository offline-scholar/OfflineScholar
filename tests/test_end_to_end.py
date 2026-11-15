"""The whole student journey through the real API and the real exam engine (hashing embedder + fake LLM).

Upload notes, past papers and a syllabus -> ask a question -> analysis -> quiz -> results -> personalised plan ->
tick a session -> delete a paper. Each step checks that the next one sees the change.
"""
from __future__ import annotations

from datetime import date, timedelta

from .conftest import make_pdf, upload
from .exam_fixtures import LECTURES
from .test_exam_engine_api import PAPERS, SYLLABUS


def test_student_journey(client, tmp_path, llm, settings):
    settings.min_score = 0.0  # the hashing test embedder scores real paraphrases low

    # 1. A subject with three past papers, a syllabus and lecture notes, all uploaded through the API.
    sid = client.post("/api/subjects", json={"name": "Machine Learning"}).json()["id"]
    papers = [upload(client, make_pdf(tmp_path / name, pages), subject_id=sid, doc_type="past_paper")
              for name, pages in PAPERS.items()]
    upload(client, make_pdf(tmp_path / "Syllabus.pdf", SYLLABUS), subject_id=sid, doc_type="syllabus")
    upload(client, make_pdf(tmp_path / "Lecture notes.pdf", [c["text"] for c in LECTURES]),
           subject_id=sid, doc_type="lecture")
    docs = client.get(f"/api/documents?subject_id={sid}").json()
    assert len(docs) == 5 and {d["status"] for d in docs} == {"ready"}

    # 2. A grounded answer that cites the lecture notes.
    ans = client.post("/api/chat/ask", json={"question": "What does an activation function do in a neuron?",
                                             "subject_id": sid}).json()
    assert ans["grounded"] and any(s["document_name"] == "Lecture notes.pdf" for s in ans["sources"])

    # 3. Analysis from the real exam engine: papers, coverage, and a hint that quizzes are missing.
    analysis = client.get(f"/api/analysis/{sid}").json()
    assert analysis["past_papers"]["topic_frequency"]["total_papers"] == 3
    cov = {t["topic"]: t["status"] for t in analysis["syllabus_coverage"]["topics"]}
    assert cov["Neural networks"] == "Covered" and cov["Reinforcement learning"] == "Not covered"
    assert analysis["weak_topics"] is None

    # 4. A quiz from the lecture notes (Ollama "down": extractive questions), answered badly, then saved.
    llm.down = True
    quiz = client.post("/api/quiz/generate", json={"subject_id": sid, "topic": "Neural networks",
                                                  "n_questions": 2}).json()
    assert quiz["questions"] and all(q["source"]["document_name"] == "Lecture notes.pdf" for q in quiz["questions"])
    answers = [{"topic": "Neural networks", "correct": False, "question_ref": f"q{i}"} for i in range(4)]
    r = client.post("/api/quiz/submit", json={"subject_id": sid, "title": "Neural networks", "answers": answers})
    assert r.status_code == 201
    history = client.get(f"/api/quiz/history?subject_id={sid}").json()
    assert history[0]["title"] == "Neural networks" and history[0]["score"] == 0.0

    # 5. The quiz result reaches the analysis (cache refreshed) and the plan.
    analysis = client.get(f"/api/analysis/{sid}").json()
    assert "Neural networks" in analysis["weak_topics"]["weak_topics"]
    plan = client.get(f"/api/revision/{sid}").json()
    nn = next(t for t in plan["priority_topics"] if t["topic"] == "Neural networks")
    assert nn["signals"]["weakness"] is not None and nn["rank"] <= 3
    assert plan["plan_settings"]["days"] == 7

    # 6. An exam date 10 days away at 45 min a day reshapes the plan.
    exam = (date.today() + timedelta(days=10)).isoformat()
    client.patch(f"/api/subjects/{sid}", json={"exam_date": exam, "minutes_per_day": 45})
    plan = client.get(f"/api/revision/{sid}").json()
    assert plan["plan_settings"]["days"] == 10 and len(plan["schedule"]) == 10
    assert all(day["minutes"] <= 45 for day in plan["schedule"])

    # 7. Ticking the first session is saved and refreshes the plan.
    first = plan["schedule"][0]["sessions"][0]
    prog = client.put(f"/api/subjects/{sid}/progress",
                      json={"day": 1, "topic": first["topic"], "minutes": first["minutes"]}).json()
    assert prog["minutes_by_topic"] == {first["topic"]: first["minutes"]}

    # 8. Deleting a past paper changes the analysis, not a stale cached copy.
    assert client.delete(f"/api/documents/{papers[0]['id']}").status_code == 204
    analysis = client.get(f"/api/analysis/{sid}").json()
    assert analysis["past_papers"]["topic_frequency"]["total_papers"] == 2

    # 9. Health is fine throughout.
    assert client.get("/api/health").json()["database"]["ok"]
