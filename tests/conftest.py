import sys
import types

# Provide lightweight dummy chromadb if not installed
if "chromadb" not in sys.modules:
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

# Dummy sentence_transformers
if "sentence_transformers" not in sys.modules:
    st = types.SimpleNamespace()

    class _DummyST:
        def __init__(self, model_name=None):
            self.model_name = model_name

        class _Emb:
            def __init__(self, arr):
                self._arr = arr

            def tolist(self):
                return self._arr

        def encode(self, texts):
            # return an object with a tolist() method like numpy arrays
            return self._Emb([[0.0] for _ in texts])

    st.SentenceTransformer = _DummyST
    sys.modules["sentence_transformers"] = st

# Dummy ollama
if "ollama" not in sys.modules:
    ollama = types.SimpleNamespace()

    def dummy_chat(model=None, messages=None):
        return {"message": {"content": "OLLAMA_PLACEHOLDER"}}

    def dummy_list():
        return {"models": []}

    ollama.chat = dummy_chat
    ollama.list = dummy_list
    sys.modules["ollama"] = ollama
