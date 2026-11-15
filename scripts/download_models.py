"""One-time setup (needs internet). After this, Offline Scholar runs fully offline.

    python scripts/download_models.py [--force]

1. Saves the embedding model into data/models/ (skipped if already there): the ONNX file used at runtime
   (no PyTorch needed) plus the Sentence Transformers files used as a fallback. Pinned to a known revision.
2. Pulls the Ollama LLM through Ollama's HTTP API (Ollama must be running; the `ollama` CLI does not
   need to be on PATH, which matters on Windows right after installing it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from backend.config import get_settings  # noqa: E402


# Revision of sentence-transformers/all-MiniLM-L6-v2 whose ONNX export was checked against the PyTorch model
# (cosine 1.000000 on 120 lecture chunks, docs/EVALUATION.md). Pinning keeps every install identical.
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_FILES = [
    "onnx/model.onnx", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json",
    "config.json", "modules.json", "sentence_bert_config.json", "config_sentence_transformers.json",
    "1_Pooling/config.json", "model.safetensors",
]


def download_embedding_model(force: bool) -> None:
    """Fetch the files with huggingface_hub (no PyTorch import, so this also works where it's blocked)."""
    from huggingface_hub import hf_hub_download

    s = get_settings()
    target = s.models_dir / s.embedding_model.split("/")[-1]
    missing = [f for f in EMBEDDING_FILES if force or not (target / f).is_file()]
    if not missing:
        print(f"[1/2] Embedding model already present: {target}")
        return
    print(f"[1/2] Downloading {len(missing)} embedding-model file(s) from {s.embedding_model} -> {target}")
    revision = EMBEDDING_REVISION if s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2" else None
    for f in missing:
        print(f"  {f}")
        hf_hub_download(s.embedding_model, f, revision=revision, local_dir=target)


def pull_ollama_model() -> int:
    s = get_settings()
    print(f"[2/2] Pulling Ollama model {s.ollama_model} from {s.ollama_url}")
    last = ""
    try:
        with httpx.stream("POST", f"{s.ollama_url.rstrip('/')}/api/pull",
                          json={"model": s.ollama_model}, timeout=httpx.Timeout(30.0, read=None)) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    print(f"  Ollama error: {msg['error']}")
                    return 1
                status = msg.get("status", "")
                if msg.get("total"):
                    status += f" {100 * msg.get('completed', 0) // msg['total']}%"
                if status != last:
                    print(f"  {status}", end="\r" if "%" in status else "\n", flush=True)
                    last = status
    except httpx.ConnectError:
        print(f"  Cannot reach Ollama at {s.ollama_url}. Install it from https://ollama.com and make sure "
              "it is running (Windows/macOS: start the Ollama app; Linux: `ollama serve`).")
        return 1
    print()
    return 0 if last == "success" else 1


def main() -> int:
    download_embedding_model(force="--force" in sys.argv)
    rc = pull_ollama_model()
    if rc == 0:
        print("\nDone. Optionally set OS_OFFLINE=1 in .env to forbid any network use from now on.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
