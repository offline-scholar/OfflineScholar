"""POST /api/chat/ask/stream (answer word by word) and OllamaClient.chat_stream."""
from __future__ import annotations

import json

import httpx
import pytest

from ai_engine.llm import LLMUnavailableError, OllamaClient

from .conftest import make_pdf, upload

NOTES = ["The perceptron updates its weights only when it misclassifies an example.",
         "A support vector machine chooses the separating hyperplane with the largest margin."]


def _events(r) -> list[dict]:
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in r.text.splitlines() if line.strip()]


def _subject(client, tmp_path) -> int:
    sid = client.post("/api/subjects", json={"name": "ML"}).json()["id"]
    upload(client, make_pdf(tmp_path / "Notes.pdf", NOTES), subject_id=sid, doc_type="lecture")
    return sid


def test_stream_ends_with_the_same_result_as_ask(client, tmp_path):
    sid = _subject(client, tmp_path)
    body = {"question": "When does the perceptron update its weights?", "subject_id": sid}
    events = _events(client.post("/api/chat/ask/stream", json=body))
    final = events[-1]
    assert final["type"] == "final" and all(e["type"] == "token" for e in events[:-1]) and len(events) >= 2
    plain = client.post("/api/chat/ask", json=body).json()
    assert {k: final[k] for k in ("answer", "grounded", "sources", "confidence")} == plain
    assert final["grounded"] and final["sources"][0]["document_name"] == "Notes.pdf"


def test_streaming_llm_sends_pieces_in_order(client, tmp_path, llm):
    pieces = ["The perceptron ", "updates when it ", "misclassifies an example [1]."]
    llm.chat_stream = lambda system, user: iter(pieces)  # a model that streams
    sid = _subject(client, tmp_path)
    events = _events(client.post("/api/chat/ask/stream",
                                 json={"question": "When does the perceptron update?", "subject_id": sid}))
    assert [e["text"] for e in events if e["type"] == "token"] == pieces
    assert events[-1]["answer"] == "".join(pieces) and events[-1]["grounded"]


def test_unsupported_answer_is_streamed_then_withdrawn(client, tmp_path, llm):
    llm.chat_stream = lambda system, user: iter(["Photosynthesis happens in chloroplasts ", "using sunlight [1]."])
    sid = _subject(client, tmp_path)
    events = _events(client.post("/api/chat/ask/stream",
                                 json={"question": "When does the perceptron update?", "subject_id": sid}))
    assert any(e["type"] == "token" for e in events)
    assert events[-1]["grounded"] is False and "couldn't find" in events[-1]["answer"]


def test_nothing_relevant_sends_only_the_final_refusal(client, tmp_path, llm):
    sid = _subject(client, tmp_path)
    events = _events(client.post("/api/chat/ask/stream", json={"question": "zzzz qqqq xxxx", "subject_id": sid}))
    assert [e["type"] for e in events] == ["final"] and events[0]["grounded"] is False
    assert llm.calls == []  # the model was never asked


def test_errors_before_and_during_the_stream(client, tmp_path, llm):
    r = client.post("/api/chat/ask/stream", json={"question": "anything at all", "subject_id": 999})
    assert r.status_code == 404 and r.json()["error"]["code"] == "subject_not_found"
    sid = _subject(client, tmp_path)
    llm.down = True
    events = _events(client.post("/api/chat/ask/stream",
                                 json={"question": "When does the perceptron update?", "subject_id": sid}))
    assert events[-1] == {"type": "error", "error": {"code": "llm_unavailable", "message": "Cannot reach Ollama"}}


# ---------------------------------------------------------------- OllamaClient.chat_stream with a mock transport

def _ollama(handler) -> OllamaClient:
    return OllamaClient("http://localhost:11434", "qwen2.5:3b", transport=httpx.MockTransport(handler))


def test_chat_stream_yields_ollama_pieces():
    lines = [{"message": {"content": "Hel"}, "done": False}, {"message": {"content": "lo [1]"}, "done": False},
             {"message": {"content": ""}, "done": True}]
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())

    assert list(_ollama(handler).chat_stream("sys", "user")) == ["Hel", "lo [1]"]
    assert seen[0]["stream"] is True and seen[0]["options"]["num_predict"] == 400


@pytest.mark.parametrize("response, match", [
    (httpx.Response(404, json={"error": "model not found"}), "not installed"),
    (httpx.Response(500, text="boom"), "HTTP 500"),
    (httpx.Response(200, content=json.dumps({"error": "out of memory"}).encode()), "out of memory"),
])
def test_chat_stream_errors(response, match):
    with pytest.raises(LLMUnavailableError, match=match):
        list(_ollama(lambda r: response).chat_stream("sys", "user"))


def test_chat_stream_connection_refused():
    def refuse(request):
        raise httpx.ConnectError("refused")
    with pytest.raises(LLMUnavailableError, match="Cannot reach Ollama"):
        list(_ollama(refuse).chat_stream("sys", "user"))
