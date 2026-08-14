from app.services import legacy_inquiry_engine as engine
from app.services import memory_service


def test_engines_use_memory_service():
    # The legacy orchestration should obtain its collection/embedding from memory_service
    assert engine.collection is memory_service.get_collection()
    assert engine.embed_model is memory_service.get_embedding_model()
