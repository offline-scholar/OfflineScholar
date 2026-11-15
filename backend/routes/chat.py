from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ai_engine.embeddings import EmbeddingModelUnavailableError
from ai_engine.llm import LLMUnavailableError
from ai_engine.rag import answer_question, answer_question_stream
from backend.database import get_db
from backend.errors import AppError
from backend.models import Subject
from backend.schemas import AskRequest, AskResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, db: Session = Depends(get_db)):
    """Answer a question from the uploaded materials, with document/page citations.

    If the materials don't contain the answer, `grounded` is false and `sources` is empty.
    """
    if req.subject_id is not None and db.get(Subject, req.subject_id) is None:
        raise AppError(404, "subject_not_found", f"Subject {req.subject_id} not found")
    result = answer_question(req.question, req.subject_id, req.top_k)  # may raise LLMUnavailableError -> 503
    return AskResponse(
        answer=result.answer, grounded=result.grounded,
        sources=[asdict(s) for s in result.sources], confidence=result.confidence,
    )


@router.post("/ask/stream")
def ask_stream(req: AskRequest, db: Session = Depends(get_db)):
    """Same as /ask, but the answer arrives word by word as newline-delimited JSON (application/x-ndjson):
    `{"type":"token","text":...}` lines while the model writes, then one `{"type":"final", ...}` line with the
    checked result (same fields as /ask). Show the tokens as they come, then replace them with `final.answer`:
    when `final.grounded` is false the streamed words were not supported by the notes.

    Errors before the stream starts use the normal error responses (404, 422). Errors during it arrive as a
    last line `{"type":"error","error":{"code","message"}}`, because the 200 status has already been sent.
    """
    if req.subject_id is not None and db.get(Subject, req.subject_id) is None:
        raise AppError(404, "subject_not_found", f"Subject {req.subject_id} not found")

    def lines() -> Iterator[str]:
        try:
            for event in answer_question_stream(req.question, req.subject_id, req.top_k):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except LLMUnavailableError as exc:
            yield json.dumps({"type": "error", "error": {"code": "llm_unavailable", "message": str(exc)}}) + "\n"
        except EmbeddingModelUnavailableError as exc:
            yield json.dumps({"type": "error", "error": {"code": "embedding_model_unavailable",
                                                         "message": str(exc)}}) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
