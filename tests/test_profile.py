"""The single local profile: GET/PATCH /api/profile."""
from __future__ import annotations

from backend.config import Settings
from backend.main import create_app
from ai_engine.runtime import build_services
from fastapi.testclient import TestClient


def test_defaults_on_first_read(client):
    assert client.get("/api/profile").json() == {"name": "Student", "quiz_length": 5, "show_sources": True}


def test_partial_update_and_trimmed_name(client):
    r = client.patch("/api/profile", json={"name": "  Malaika  "})
    assert r.status_code == 200 and r.json() == {"name": "Malaika", "quiz_length": 5, "show_sources": True}
    r = client.patch("/api/profile", json={"quiz_length": 10, "show_sources": False})
    assert r.json() == {"name": "Malaika", "quiz_length": 10, "show_sources": False}
    assert client.get("/api/profile").json()["quiz_length"] == 10


def test_validation(client):
    assert client.patch("/api/profile", json={"name": "   "}).status_code == 422
    assert client.patch("/api/profile", json={"quiz_length": 0}).status_code == 422
    assert client.patch("/api/profile", json={"quiz_length": 21}).status_code == 422
    r = client.patch("/api/profile", json={"name": None})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_profile"


def test_profile_survives_a_restart(tmp_path, llm):
    settings = Settings(data_dir=tmp_path / "data", embedding_backend="hashing", _env_file=None)
    with TestClient(create_app(settings, build_services(settings, llm=llm))) as c:
        c.patch("/api/profile", json={"name": "Essa"})
    with TestClient(create_app(settings, build_services(settings, llm=llm))) as c:
        assert c.get("/api/profile").json()["name"] == "Essa"
