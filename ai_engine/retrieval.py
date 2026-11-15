"""Semantic retrieval and chunk access (the reusable API consumed by Member 2)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from backend.database import session_scope
from backend.models import Chunk, Document, Subject

from .runtime import Services, get_services


class DocumentNotFoundError(LookupError):
    pass


@dataclass
class ChunkRecord:
    chunk_id: int
    document_id: int
    document_name: str
    subject_id: int
    doc_type: str
    page_number: int
    chunk_index: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievedChunk(ChunkRecord):
    score: float = 0.0


def generate_embeddings(texts: list[str], services: Services | None = None) -> np.ndarray:
    """Embed texts with the shared model. Returns normalised float32 array (n, dim)."""
    return (services or get_services()).embedder.embed(list(texts))


def get_document_chunks(document_id: int) -> list[ChunkRecord]:
    """All chunks of a document in reading order. Raises DocumentNotFoundError."""
    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            raise DocumentNotFoundError(f"Document {document_id} not found")
        rows = db.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.chunk_index).all()
        return [ChunkRecord(r.id, doc.id, doc.filename, doc.subject_id, doc.doc_type,
                            r.page_number, r.chunk_index, r.text) for r in rows]


def retrieve_context(
    question: str,
    subject_id: int | None = None,
    top_k: int = 4,
    *,
    min_score: float | None = None,
    doc_types: tuple[str, ...] | list[str] | None = None,
    services: Services | None = None,
) -> list[RetrievedChunk]:
    """Top-k passages for a question, best first, each with document name, page and cosine score.

    subject_id=None searches every subject. Passages scoring below `min_score` (default from
    settings) are dropped, so an empty list means "nothing relevant in the uploaded material".
    `doc_types` (e.g. ("lecture",)) keeps only passages from those document types and still returns up to
    top_k of them: without it, past-paper questions on the same topic can fill the top k and push the
    lecture notes out (the quiz generator then had too little material).
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    svc = services or get_services()
    threshold = svc.settings.min_score if min_score is None else min_score

    with session_scope() as db:
        subject_ids = [subject_id] if subject_id is not None else [s.id for s in db.query(Subject.id).all()]
        if not subject_ids:
            return []
        qvec = svc.embedder.embed([question.strip()])[0]
        wanted = set(doc_types) if doc_types else None
        # With a type filter, look further down the ranking so top_k passages of that type remain.
        k = top_k if wanted is None else max(top_k * 8, 50)
        hits = [h for h in svc.vector_store.search(qvec, subject_ids, k) if h[1] >= threshold]
        # Equal scores (common with short or unrelated passages) come back from FAISS in an order that depends
        # on k. Break ties by chunk id (reading order) so the same question always gets the same passages.
        hits.sort(key=lambda h: (-round(h[1], 6), h[0]))
        if not hits:
            return []
        query = db.query(Chunk, Document).join(Document).filter(
            Chunk.id.in_([h[0] for h in hits]), Document.status == "ready")
        if wanted is not None:
            query = query.filter(Document.doc_type.in_(wanted))
        rows = {c.id: (c, d) for c, d in query.all()}
        return [
            RetrievedChunk(c.id, d.id, d.filename, d.subject_id, d.doc_type, c.page_number,
                           c.chunk_index, c.text, score=round(score, 4))
            for cid, score in hits if cid in rows for c, d in [rows[cid]]
        ][:top_k]
