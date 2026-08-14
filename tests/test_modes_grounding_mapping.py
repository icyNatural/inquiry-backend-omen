import json
from app.services import legacy_inquiry_engine as engine

QUERY = "What does my memory say about the Inquiry Engine?"
SCOPE = {"namespaces": ["this_should_not_exist"]}


def test_grounding_memory_only_no_ollama_call():
    called = {"count": 0}

    def stub_chat(model, messages):
        called["count"] += 1
        return {"message": {"content": "MODEL_RESPONSE"}}

    # patch
    orig = engine.llm_service.chat
    engine.llm_service.chat = stub_chat

    try:
        res = engine.investigate(QUERY, mode="grounding", scope=SCOPE, knowledge_policy="memory_only")

        assert res["retrieved_count"] == 0
        assert res["selected_count"] == 0
        assert res["sources"] == []
        assert isinstance(res["answer"], str)
        # memory_only must not call Ollama
        assert called["count"] == 0
    finally:
        engine.llm_service.chat = orig


def test_grounding_memory_plus_model_calls_ollama_and_labels():
    called = {"count": 0}

    def stub_chat(model, messages):
        called["count"] += 1
        return {"message": {"content": "This is a model-generated grounding answer."}}

    orig = engine.llm_service.chat
    engine.llm_service.chat = stub_chat

    try:
        res = engine.investigate(QUERY, mode="grounding", scope=SCOPE, knowledge_policy="memory_plus_model")

        assert res["retrieved_count"] == 0
        assert res["selected_count"] == 0
        assert res["sources"] == []
        assert called["count"] >= 1
        assert "No matching memories were found" in res["organized_signals"]
        assert res["answer"].startswith("No matching memories were found.")
    finally:
        engine.llm_service.chat = orig


def test_mapping_memory_only_no_ollama_call():
    called = {"count": 0}

    def stub_chat(model, messages):
        called["count"] += 1
        return {"message": {"content": "MODEL_RESPONSE"}}

    orig = engine.llm_service.chat
    engine.llm_service.chat = stub_chat

    try:
        res = engine.investigate(QUERY, mode="mapping", scope=SCOPE, knowledge_policy="memory_only")

        assert res["retrieved_count"] == 0
        assert res["selected_count"] == 0
        assert res["sources"] == []
        assert isinstance(res["answer"], str)
        assert called["count"] == 0
    finally:
        engine.llm_service.chat = orig


def test_mapping_memory_plus_model_calls_ollama_and_labels():
    called = {"count": 0}

    def stub_chat(model, messages):
        called["count"] += 1
        return {"message": {"content": "This is a model-generated mapping answer."}}

    orig = engine.llm_service.chat
    engine.llm_service.chat = stub_chat

    try:
        res = engine.investigate(QUERY, mode="mapping", scope=SCOPE, knowledge_policy="memory_plus_model")

        assert res["retrieved_count"] == 0
        assert res["selected_count"] == 0
        assert res["sources"] == []
        assert called["count"] >= 1
        assert "No matching memories were found" in res["organized_signals"]
        assert res["answer"].startswith("No matching memories were found.")
    finally:
        engine.llm_service.chat = orig
