import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'app' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import legacy_inquiry_engine as engine

# Patch ollama.chat with a stub that records calls and returns a predictable response
called = {"count": 0}

def stub_chat(model, messages):
    called["count"] += 1
    return {"message": {"content": "This is a model-generated general knowledge answer."}}

# Save original
_original_chat = getattr(engine.ollama, 'chat', None)
engine.ollama.chat = stub_chat

query = "What does my memory say about the Inquiry Engine?"
scope = {"namespaces": ["this_should_not_exist"]}

print("Running Test 1: memory_only (should NOT call Ollama)")
called["count"] = 0
res1 = engine.investigate(query, mode="recall", scope=scope, knowledge_policy="memory_only")
print(json.dumps({"called": called["count"], "result": res1}, indent=2))

print("\nRunning Test 2: memory_plus_model (should call Ollama)")
called["count"] = 0
res2 = engine.investigate(query, mode="recall", scope=scope, knowledge_policy="memory_plus_model")
print(json.dumps({"called": called["count"], "result": res2}, indent=2))

# Restore original
if _original_chat is not None:
    engine.ollama.chat = _original_chat
else:
    delattr(engine.ollama, 'chat')
