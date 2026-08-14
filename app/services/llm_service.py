import os
import ollama

OLLAMA_MODEL = os.getenv("INQUIRY_OLLAMA_MODEL", "gemma3:4b")


def chat(model: str | None, messages: list) -> dict:
    """Call Ollama chat with basic fallback behavior.
    Returns the raw response dict from ollama.chat to preserve compatibility.
    """
    model_name = model or OLLAMA_MODEL
    try:
        response = ollama.chat(model=model_name, messages=messages)
        return response
    except Exception as e:
        err_msg = str(e)
        # Fallback: try to list available models and use first available
        try:
            models_list = ollama.list()
            available = [m.get("name") for m in models_list.get("models", []) if m.get("name")]
            if available:
                fallback = available[0]
                if fallback != model_name:
                    response = ollama.chat(model=fallback, messages=messages)
                    return {"message": {"content": f"[FALLBACK TO {fallback}]\n\n{response['message']['content'].strip()}"}}
        except Exception:
            pass
        raise
