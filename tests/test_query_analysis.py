import app.services.legacy_inquiry_engine as engine


def _stub_chat(model, messages):
    return {"message": {"content": "STUB"}}


def test_normalization_and_classification_factual():
    orig = engine.llm_service.chat
    engine.llm_service.chat = _stub_chat
    try:
        res = engine.investigate("what is homeothersis", mode="recall", scope={}, knowledge_policy="memory_plus_model")

        assert res["original_query"] == "what is homeothersis"
        assert res["normalized_query"] == "homeorhesis" or "homeorhesis" in res["normalized_query"]
        assert res["query_changed"] is True
        assert res["question_type"] == "factual_concept"
        assert res["source_plan"]["personal_memory"] is False
        assert res["source_plan"]["external_web"] is True
    finally:
        engine.llm_service.chat = orig


def test_personal_recall_classification():
    orig = engine.llm_service.chat
    engine.llm_service.chat = _stub_chat
    try:
        res = engine.investigate("what did I say about homeorhesis?", mode="recall", scope={}, knowledge_policy="memory_only")
        assert res["question_type"] == "personal_recall"
        assert res["source_plan"]["personal_memory"] is True
    finally:
        engine.llm_service.chat = orig


def test_comparison_requires_both_sources():
    orig = engine.llm_service.chat
    engine.llm_service.chat = _stub_chat
    try:
        res = engine.investigate("compare my notes on homeorhesis with current research", mode="recall", scope={}, knowledge_policy="memory_plus_model")
        assert res["question_type"] == "comparison"
        assert res["source_plan"]["personal_memory"] is True
        assert res["source_plan"]["external_web"] is True
    finally:
        engine.llm_service.chat = orig
