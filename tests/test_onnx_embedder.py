"""ONNX embedder (default runtime, no PyTorch) and clear failures when an embedding runtime can't load."""
from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai_engine.embeddings import (EmbeddingModelUnavailableError, OnnxEmbedder, SentenceTransformerEmbedder,
                                  build_embedder)
from ai_engine.runtime import build_services
from backend.config import Settings
from backend.main import create_app

ROOT = Path(__file__).resolve().parent.parent
REAL_MODEL = ROOT / "data" / "models" / "all-MiniLM-L6-v2"


def _settings(tmp_path, backend: str) -> Settings:
    return Settings(data_dir=tmp_path / "data", embedding_backend=backend, _env_file=None)


def _fake_model_dir(settings: Settings, with_onnx: bool) -> Path:
    d = settings.models_dir / "all-MiniLM-L6-v2"
    (d / "onnx").mkdir(parents=True)
    if with_onnx:
        (d / "onnx" / "model.onnx").write_bytes(b"not loaded in this test")
        (d / "tokenizer.json").write_text("{}")
    return d


def test_auto_prefers_onnx_when_its_model_file_is_saved(tmp_path):
    s = _settings(tmp_path, "auto")
    _fake_model_dir(s, with_onnx=True)
    emb = build_embedder(s)
    assert isinstance(emb, OnnxEmbedder) and emb.available and not emb.loaded


def test_auto_falls_back_to_sentence_transformers_without_the_onnx_file(tmp_path):
    s = _settings(tmp_path, "auto")
    _fake_model_dir(s, with_onnx=False)
    assert isinstance(build_embedder(s), SentenceTransformerEmbedder)


def test_missing_onnx_model_is_a_clear_error(tmp_path):
    emb = OnnxEmbedder(tmp_path / "nowhere")
    assert not emb.available
    with pytest.raises(EmbeddingModelUnavailableError, match="download_models.py"):
        emb.embed(["x"])


def test_blocked_runtime_gives_503_with_a_fix_and_health_says_why(tmp_path, monkeypatch):
    """Windows Smart App Control blocking a PyTorch DLL used to surface as a bare 500 on every question."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise OSError('[WinError 4551] An Application Control policy has blocked this file. Error loading "shm.dll"')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    s = _settings(tmp_path, "sentence-transformers")
    svc = build_services(s)
    with TestClient(create_app(s, svc)) as client:
        sid = client.post("/api/subjects", json={"name": "S"}).json()["id"]
        r = client.post("/api/chat/ask", json={"question": "What is a margin?", "subject_id": sid})
        assert r.status_code == 503
        err = r.json()["error"]
        assert err["code"] == "embedding_model_unavailable" and "Smart App Control" in err["message"]
        h = client.get("/api/health").json()
        assert h["status"] == "degraded" and h["embeddings"]["available"] is False
        assert "Application Control" in h["embeddings"]["error"]


@pytest.mark.skipif(not OnnxEmbedder.model_file(REAL_MODEL).is_file(), reason="real ONNX model not downloaded")
def test_real_onnx_model_gives_normalised_semantic_vectors():
    emb = OnnxEmbedder(REAL_MODEL)
    v = emb.embed(["The support vector machine maximises the margin.",
                   "An SVM finds the separator with the largest margin.",
                   "Bake the bread for forty minutes."])
    assert v.shape == (3, 384) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2] + 0.3  # paraphrases are close, unrelated text is not
    long = emb.embed(["word " * 2000])  # longer than the 256-token limit: truncated, not an error
    assert long.shape == (1, 384)
