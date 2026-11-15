"""Embedding backends.

Production: all-MiniLM-L6-v2 run with ONNX Runtime (no PyTorch: smaller, faster to start, and its DLLs are
signed, so Windows Smart App Control doesn't block it). Fallback: the same model through Sentence Transformers
(PyTorch). Tests/CI: a hashing embedder.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from backend.config import Settings


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 array of shape (len(texts), dim)."""


class EmbeddingModelUnavailableError(RuntimeError):
    """The embedding model can't be used: not on disk (offline) or its runtime failed to load."""


def _load_failure(model: str, exc: BaseException) -> EmbeddingModelUnavailableError:
    """Turn an import/DLL error into a clear, actionable 503 instead of a 500."""
    msg = str(exc)
    hint = ""
    if "Application Control" in msg or "4551" in msg:
        hint = (" Windows Smart App Control blocked a library file. Run `python scripts/download_models.py` "
                "so the ONNX version of the model (no PyTorch) is used instead.")
    return EmbeddingModelUnavailableError(f"Could not load the embedding model '{model}': {msg[:300]}.{hint}")


class SentenceTransformerEmbedder:
    """Lazy-loads the model on first use (loading takes a few seconds)."""

    def __init__(self, model: str, offline: bool = False):
        self.name = model
        self._offline = offline
        self._model = None
        self._lock = threading.Lock()
        self._dim: int | None = None
        self.error: str | None = None  # why the last load failed, for /api/health

    @property
    def available(self) -> bool:
        """False when offline and the model isn't saved locally, or when loading it already failed."""
        if self.error:
            return False
        return self._model is not None or not self._offline or Path(self.name).is_dir()

    def _load(self):
        with self._lock:
            if self._model is None:
                if not self.available:
                    # Fail fast with a fix, instead of Hugging Face's "check your internet connection"
                    # after a ~13 s retry loop.
                    raise EmbeddingModelUnavailableError(
                        f"Embedding model '{self.name}' is not saved in data/models and offline mode "
                        "(OS_OFFLINE=1) forbids downloading it. Run `python scripts/download_models.py` "
                        "once while online.")
                if self._offline:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                try:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(self.name, device="cpu")
                except (ImportError, OSError) as exc:  # e.g. a PyTorch DLL blocked by the OS
                    self.error = str(exc)
                    raise _load_failure(self.name, exc) from exc
                self._model = model
                get_dim = getattr(self._model, "get_embedding_dimension", None) or \
                    self._model.get_sentence_embedding_dimension  # name before sentence-transformers 6
                self._dim = int(get_dim())
        return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def dim(self) -> int:  # type: ignore[override]
        self._load()
        return self._dim  # type: ignore[return-value]

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vecs = model.encode(
            texts, batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


class OnnxEmbedder:
    """all-MiniLM-L6-v2 through ONNX Runtime, matching Sentence Transformers' output.

    Same steps as the saved Sentence Transformers pipeline (modules.json: Transformer -> mean Pooling ->
    Normalize): tokenize (max 256 tokens, pad to the longest text in the batch), run the model, mean-pool
    the token vectors over the attention mask, L2-normalise. Needs `<model dir>/onnx/model.onnx` and the
    tokenizer files that scripts/download_models.py saves.
    """

    def __init__(self, model_dir, max_tokens: int = 256, threads: int | None = None):
        self.dir = Path(model_dir)
        self.name = str(self.dir)
        self.max_tokens = max_tokens
        self.threads = threads
        self._session = None
        self._tokenizer = None
        self._inputs: set[str] = set()
        self._lock = threading.Lock()
        self.dim = 384
        self.error: str | None = None

    @staticmethod
    def model_file(model_dir) -> Path:
        return Path(model_dir) / "onnx" / "model.onnx"

    @property
    def available(self) -> bool:
        return self.error is None and self.model_file(self.dir).is_file() and (self.dir / "tokenizer.json").is_file()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def _load(self):
        with self._lock:
            if self._session is None:
                if not self.available:
                    raise EmbeddingModelUnavailableError(
                        self.error or f"ONNX model not found in {self.dir}. Run `python scripts/download_models.py`.")
                try:
                    import onnxruntime as ort
                    from tokenizers import Tokenizer
                except (ImportError, OSError) as exc:
                    self.error = str(exc)
                    raise _load_failure(self.name, exc) from exc
                tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
                tok.enable_truncation(max_length=self.max_tokens)
                pad_id = tok.token_to_id("[PAD]")
                tok.enable_padding(pad_id=0 if pad_id is None else pad_id, pad_token="[PAD]")  # pad to longest
                opts = ort.SessionOptions()
                if self.threads:
                    opts.intra_op_num_threads = self.threads
                sess = ort.InferenceSession(str(self.model_file(self.dir)), opts, providers=["CPUExecutionProvider"])
                self._inputs = {i.name for i in sess.get_inputs()}
                dim = sess.get_outputs()[0].shape[-1]
                if isinstance(dim, int):
                    self.dim = dim
                self._tokenizer, self._session = tok, sess
        return self._session

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        sess = self._load()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for start in range(0, len(texts), batch_size):
            enc = self._tokenizer.encode_batch(texts[start:start + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            tokens = sess.run(None, feeds)[0]  # (batch, seq, dim)
            m = mask[..., None].astype(np.float32)
            pooled = (tokens * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            out.append(pooled / np.clip(norms, 1e-12, None))
        return np.vstack(out).astype(np.float32)


_STOP = frozenset(
    "a an the of to in on at for and or is are was were be been it this that these those with as by from "
    "what which who how why when where do does did can could should would explain define describe tell me about".split()
)


class HashingEmbedder:
    """Bag-of-words hashing embedder. Deterministic, no downloads: NOT semantic. Tests/CI only."""

    loaded = True
    available = True

    def __init__(self, dim: int = 384):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                if tok in _STOP or len(tok) < 2:
                    continue
                tok = tok.rstrip("s") if len(tok) > 3 else tok
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def build_embedder(settings: Settings) -> Embedder:
    """`auto` (default): ONNX when its model file is saved locally, else Sentence Transformers."""
    backend = settings.embedding_backend
    if backend == "hashing":
        return HashingEmbedder()
    local = settings.resolved_embedding_model
    if backend == "onnx" or (backend == "auto" and OnnxEmbedder.model_file(local).is_file()):
        return OnnxEmbedder(local)
    return SentenceTransformerEmbedder(local, offline=settings.offline)
