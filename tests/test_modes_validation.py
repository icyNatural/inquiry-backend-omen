import pytest
from app.services import legacy_inquiry_engine as engine

QUERY = "Test query"
SCOPE = {"namespaces": ["this_should_not_exist"]}

MODES = ["recall", "synthesis", "framework", "raw", "grounding", "mapping"]


def test_supported_modes_memory_only_no_ollama_call():
    called = {"count": 0}

    def stub_chat(model, messages):
        called["count"] += 1
        return {"message": {"content": "MODEL_RESPONSE"}}

    orig = engine.llm_service.chat
    engine.llm_service.chat = stub_chat

    try:
        for m in MODES:
            res = engine.investigate(QUERY, mode=m, scope=SCOPE, knowledge_policy="memory_only")
            assert res["retrieved_count"] == 0
            assert res["selected_count"] == 0
            assert res["sources"] == []
            # memory_only must not call Ollama
            assert called["count"] == 0
    finally:
        engine.llm_service.chat = orig


def test_supported_modes_memory_plus_model_calls_ollama_and_labels():
    called = {"count": 0}

    def stub_chat(model, messages):
        called["count"] += 1
        return {"message": {"content": "Model generated answer."}}

    orig = engine.llm_service.chat
    engine.llm_service.chat = stub_chat

    try:
        for m in MODES:
            res = engine.investigate(QUERY, mode=m, scope=SCOPE, knowledge_policy="memory_plus_model")
            assert res["retrieved_count"] == 0
            assert res["selected_count"] == 0
            assert res["sources"] == []
            assert called["count"] >= 1
            # organized_signals should indicate no matching memories
            assert "No matching memories were found" in res["organized_signals"]
            assert res["answer"].startswith("No matching memories were found.")
    finally:
        engine.llm_service.chat = orig


def test_invalid_mode_rejected():
    with pytest.raises(ValueError) as exc:
        engine.investigate(QUERY, mode="not_a_mode")
    assert "Unsupported mode" in str(exc.value)
