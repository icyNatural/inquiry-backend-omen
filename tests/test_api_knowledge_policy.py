from fastapi.testclient import TestClient
from app.main import app
from app.models.inquiry import KnowledgePolicy
import app.services.legacy_inquiry_engine as engine

client = TestClient(app)


def _stub_investigate(user_query, mode, scope, knowledge_policy):
    # Ensure the engine receives the canonical enum
    assert isinstance(knowledge_policy, KnowledgePolicy)
    return {
        "status": "ok",
        "query": user_query,
        "mode": mode,
        "model": "stub-model",
        "answer": "stub",
        "organized_signals": "",
        "search_queries": [],
        "retrieved_count": 0,
        "selected_count": 0,
        "sources": [],
    }


def test_api_accepts_memory_plus_model():
    orig = engine.investigate
    engine.investigate = _stub_investigate

    try:
        resp = client.post(
            "/api/cognition/reason",
            json={
                "query": "What is homeorhesis?",
                "mode": "recall",
                "scope": {},
                "knowledge_policy": "memory_plus_model",
            },
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ok"
    finally:
        engine.investigate = orig


def test_api_accepts_memory_only():
    orig = engine.investigate
    engine.investigate = _stub_investigate

    try:
        resp = client.post(
            "/api/cognition/reason",
            json={
                "query": "What is homeorhesis?",
                "mode": "recall",
                "scope": {},
                "knowledge_policy": "memory_only",
            },
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ok"
    finally:
        engine.investigate = orig
