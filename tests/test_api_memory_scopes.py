from fastapi.testclient import TestClient
from app.main import app
from app.services import memory_service, scope_service

client = TestClient(app)


def test_list_notes_and_content_and_save(monkeypatch):
    sample = [{"title": "Note A", "path": "a.md"}]

    monkeypatch.setattr(memory_service, "get_refined_notes_list", lambda: sample)
    monkeypatch.setattr(memory_service, "read_note_content", lambda p: "CONTENT" if p == "a.md" else (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(memory_service, "save_note_content", lambda p, c: True)

    r = client.get("/api/memory/notes")
    assert r.status_code == 200
    assert r.json() == sample

    r = client.get("/api/memory/notes/content", params={"path": "a.md"})
    assert r.status_code == 200
    assert r.json() == {"content": "CONTENT"}

    r = client.post("/api/memory/notes/save", json={"path": "b.md", "content": "X"})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    monkeypatch.setattr(memory_service, "load_session_memory", lambda: [{"q":"1"}])
    r = client.get("/api/memory/sessions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_scope_config_and_validate(monkeypatch):
    default = scope_service.ScopeProcessor.validate_scope_config({})
    r = client.get("/api/scopes/config")
    assert r.status_code == 200
    assert r.json() == default

    # validate payload
    payload = {"doc_text": "hello #tag", "metadata": {"tags": "#tag"}, "scope": {"include_tags": ["#tag"]}}
    r = client.post("/api/scopes/validate", json=payload)
    assert r.status_code == 200
    assert r.json()["ok"] in (True, False)
