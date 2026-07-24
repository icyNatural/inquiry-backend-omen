import json
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path("raw/notes_ready.json")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "chat_notes"

MAX_CHARS = 1200
OVERLAP_MESSAGES = 1
BATCH_SIZE = 2000

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"Deleted old collection: {COLLECTION_NAME}")
except Exception:
    print(f"No existing collection to delete: {COLLECTION_NAME}")

collection = client.get_or_create_collection(name=COLLECTION_NAME)
model = SentenceTransformer("all-MiniLM-L6-v2")


def split_messages(text):
    parts = re.split(r"(USER:|ASSISTANT:)", text)
    messages = []
    current_role = None

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part == "USER:":
            current_role = "user"
        elif part == "ASSISTANT:":
            current_role = "assistant"
        else:
            messages.append({
                "speaker": current_role or "unknown",
                "text": part
            })

    return messages


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def guess_chunk_type(text):
    lower = text.lower()

    if "```" in text or "function " in lower or "import " in lower or "const " in lower:
        return "code_or_technical"

    if "framework" in lower or "model" in lower or "architecture" in lower:
        return "framework"

    if "how do i" in lower or "where do i" in lower or "error" in lower:
        return "troubleshooting"

    return "conversation"


def build_message_aware_chunks(messages):
    chunks = []
    current = []
    current_len = 0

    for msg in messages:
        speaker = msg["speaker"]
        text = clean_text(msg["text"])

        if not text:
            continue

        block = f"{speaker.upper()}: {text}"
        block_len = len(block)

        if current and current_len + block_len > MAX_CHARS:
            chunks.append(current)

            current = current[-OVERLAP_MESSAGES:] if OVERLAP_MESSAGES > 0 else []
            current_len = sum(len(x["block"]) for x in current)

        current.append({
            "speaker": speaker,
            "block": block,
            "text": text
        })

        current_len += block_len

    if current:
        chunks.append(current)

    return chunks


def speaker_mix(chunk_messages):
    speakers = {m["speaker"] for m in chunk_messages}

    if speakers == {"user"}:
        return "user_only"

    if speakers == {"assistant"}:
        return "assistant_only"

    return "mixed"


def chunk_to_text(chunk_messages):
    return "\n\n".join(m["block"] for m in chunk_messages)


with INPUT_FILE.open("r", encoding="utf-8") as f:
    conversations = json.load(f)

batch_documents = []
batch_metadatas = []
batch_ids = []

doc_id = 0
total_chunks = 0


def flush_batch():
    global batch_documents, batch_metadatas, batch_ids, total_chunks

    if not batch_documents:
        return

    print(f"Embedding batch of {len(batch_documents)} chunks...")

    embeddings = model.encode(
        batch_documents,
        show_progress_bar=True
    ).tolist()

    collection.add(
        documents=batch_documents,
        embeddings=embeddings,
        metadatas=batch_metadatas,
        ids=batch_ids
    )

    total_chunks += len(batch_documents)
    print(f"Saved {total_chunks} chunks so far.")

    batch_documents = []
    batch_metadatas = []
    batch_ids = []


for convo_index, convo in enumerate(conversations):
    title = convo.get("title", "Untitled")
    text = convo.get("text", "")

    if not text.strip():
        continue

    messages = split_messages(text)

    if not messages:
        continue

    chunks = build_message_aware_chunks(messages)

    for chunk_index, chunk_messages in enumerate(chunks):
        chunk_text = chunk_to_text(chunk_messages)
        chunk_kind = guess_chunk_type(chunk_text)

        batch_documents.append(chunk_text)

        batch_metadatas.append({
            "title": title,
            "conversation_index": convo_index,
            "chunk_index": chunk_index,
            "source": "chatgpt_export",
            "chunk_type": chunk_kind,
            "speaker_mix": speaker_mix(chunk_messages),
            "message_count": len(chunk_messages)
        })

        batch_ids.append(f"doc_{doc_id}")
        doc_id += 1

        if len(batch_documents) >= BATCH_SIZE:
            flush_batch()

flush_batch()

print("Done.")
print(f"Saved {total_chunks} message-aware chunks to Chroma at: {CHROMA_DIR}")