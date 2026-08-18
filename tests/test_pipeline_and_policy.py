import sys

# Provide a lightweight dummy chromadb module to avoid heavy runtime dependency during tests.
if "chromadb" not in sys.modules:
    import types

    chromadb = types.SimpleNamespace()

    class _DummyClient:
        def __init__(self, path=None):
            self.path = path

        def get_or_create_collection(self, name=None):
            class _Collection:
                def query(self, **kwargs):
                    return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

                def add(self, **kwargs):
                    return None

                def delete(self, **kwargs):
                    return None

            return _Collection()

    chromadb.PersistentClient = _DummyClient
    sys.modules["chromadb"] = chromadb

if "sentence_transformers" not in sys.modules:
    import types

    st = types.SimpleNamespace()

    class _DummyST:
        def __init__(self, model_name=None):
            self.model_name = model_name

        def encode(self, texts):
            # return zero vectors of length 1 for simplicity
            return [[0.0] for _ in texts]

    st.SentenceTransformer = _DummyST
    sys.modules["sentence_transformers"] = st

if "ollama" not in sys.modules:
    import types

    ollama = types.SimpleNamespace()

    def dummy_chat(model=None, messages=None):
        return {"message": {"content": "OLLAMA_PLACEHOLDER"}}

    def dummy_list():
        return {"models": []}

    ollama.chat = dummy_chat
    ollama.list = dummy_list
    sys.modules["ollama"] = ollama

from app.services import legacy_inquiry_engine as engine
from app.services import retrieval_service, cognition_service

QUERY = "Pipeline test query"


def test_pipeline_successful_memory_backed_flow(monkeypatch):
    # Prepare fake merged results (as returned by merge_multi_query_results)
    merged = [
        {"document": "Doc A content", "metadata": {"title": "A", "conversation_index": 1, "chunk_index": 0, "source": "refined_notes", "domain": "general"}, "distance": 0.1, "source_query": "q1"},
        {"document": "Doc B content", "metadata": {"title": "B", "conversation_index": 2, "chunk_index": 0, "source": "refined_notes", "domain": "general"}, "distance": 0.2, "source_query": "q1"},
    ]

    # Stubs
    monkeypatch.setattr(retrieval_service, "merge_multi_query_results", lambda sq: merged)
    monkeypatch.setattr(retrieval_service, "diversify_results", lambda mr, top_k: mr[:1])
    monkeypatch.setattr(retrieval_service, "rerank_results", lambda q, sel, top_k: sel)
    monkeypatch.setattr(cognition_service, "interpret_sources", lambda q, sel: "INTERPRETED SIGNALS")

    called = {"count": 0, "messages": None}

    def stub_chat(model, messages):
        called["count"] += 1
        called["messages"] = messages
        return {"message": {"content": "FINAL ANSWER"}}

    monkeypatch.setattr(engine.llm_service, "chat", stub_chat)

    res = engine.investigate(QUERY, mode="recall", scope=None, knowledge_policy="memory_plus_model")

    assert res["retrieved_count"] == len(merged)
    assert res["selected_count"] == 1
    assert len(res["sources"]) == 1
    assert res["organized_signals"] == "INTERPRETED SIGNALS"
    assert res["answer"].endswith("FINAL ANSWER") or "FINAL ANSWER" in res["answer"]
    assert called["count"] == 1


def test_knowledge_policy_memory_only_no_ollama(monkeypatch):
    # No results
    monkeypatch.setattr(retrieval_service, "merge_multi_query_results", lambda sq: [])
    called = {"count": 0}

    def stub_chat(model, messages):
        called["count"] += 1
        return {"message": {"content": "SHOULD NOT HAPPEN"}}

    monkeypatch.setattr(engine.llm_service, "chat", stub_chat)

    res = engine.investigate("Q", mode="recall", scope={"namespaces": ["this_should_not_exist"]}, knowledge_policy="memory_only")

    assert res["retrieved_count"] == 0
    assert res["selected_count"] == 0
    assert res["sources"] == []
    assert res["answer"] == "No memories matched the requested scope."
    assert called["count"] == 0


def test_knowledge_policy_memory_plus_model_fallback(monkeypatch):
    # No results
    monkeypatch.setattr(retrieval_service, "merge_multi_query_results", lambda sq: [])
    called = {"count": 0}

    def stub_chat(model, messages):
        called["count"] += 1
        return {"message": {"content": "ModelFallbackAnswer"}}

    monkeypatch.setattr(engine.llm_service, "chat", stub_chat)

    res = engine.investigate("Q", mode="recall", scope={"namespaces": ["this_should_not_exist"]}, knowledge_policy="memory_plus_model")

    assert res["retrieved_count"] == 0
    assert res["selected_count"] == 0
    assert res["sources"] == []
    assert "No matching memories were found" in res["organized_signals"]
    assert res["answer"].startswith("No matching memories were found.")
    assert called["count"] == 1
