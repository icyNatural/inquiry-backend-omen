import re
from pathlib import Path
from typing import List, Tuple

from app.services.memory_service import get_collection, get_embedding_model
from app.services import llm_service

TOP_K_FINAL = 5
TOP_K_PER_QUERY = 8
SNIPPET_LEN = 220


def make_snippet(text: str, max_len: int = SNIPPET_LEN) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def generate_search_queries(user_query: str, mode: str):
    boosters = {
        "energy engine": [
            "Energy Engine concept recovery usable energy baseline",
            "Energy Engine HRV sleep heart rate activity wake time",
            "Energy Engine raw physiology recovery energy model"
        ],
        "stable node": [
            "Stable Node calm reference point clarity pressure",
            "Stable Node grounding overload avoidance"
        ],
        "language execution": [
            "Language Execution Framework language behavior outcome",
            "internal language instruction behavior environment"
        ],
        "world understanding engine": [
            "World Understanding Engine global patterns country systems",
            "WUE culture economics geography relationships"
        ],
        "inquiry engine": [
            "Inquiry Engine adaptive questioning memory patterns",
            "semantic branching followup questions life patterns"
        ],
    }

    lower_query = user_query.lower().strip()
    queries = [user_query.strip()]

    for key, extra_queries in boosters.items():
        if key in lower_query:
            for q in extra_queries:
                if q.lower() not in {x.lower() for x in queries}:
                    queries.append(q)

    fallback = [
        f"{user_query} concept meaning",
        f"{user_query} key ideas",
        f"{user_query} related notes"
    ]

    for q in fallback:
        if q.lower() not in {x.lower() for x in queries}:
            queries.append(q)

    return queries[:4]


def retrieve_for_query(query: str, top_k: int = TOP_K_PER_QUERY) -> Tuple[List[str], List[dict], List[float]]:
    collection = get_collection()
    embed_model = get_embedding_model()

    query_embedding = embed_model.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]
    return documents, metadatas, distances


def merge_multi_query_results(search_queries):
    merged = []
    seen_keys = set()

    for source_query in search_queries:
        documents, metadatas, distances = retrieve_for_query(source_query, TOP_K_PER_QUERY)

        for doc, meta, dist in zip(documents, metadatas, distances):
            key = (
                meta.get("conversation_index"),
                meta.get("chunk_index"),
                meta.get("title", "Untitled")
            )

            if key in seen_keys:
                continue

            seen_keys.add(key)
            merged.append({
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "source_query": source_query
            })

    merged.sort(key=lambda x: x["distance"])
    return merged


def diversify_results(merged_results, final_k: int = TOP_K_FINAL):
    selected = []
    seen_conversations = set()
    seen_titles = set()

    for item in merged_results:
        meta = item["metadata"]
        convo_id = meta.get("conversation_index")
        title = meta.get("title", "Untitled")

        if convo_id in seen_conversations:
            continue
        if title in seen_titles:
            continue

        selected.append(item)
        seen_conversations.add(convo_id)
        seen_titles.add(title)

        if len(selected) >= final_k:
            return selected

    for item in merged_results:
        if len(selected) >= final_k:
            break

        meta = item["metadata"]
        convo_id = meta.get("conversation_index")
        chunk_id = meta.get("chunk_index")

        already_selected = any(
            s["metadata"].get("conversation_index") == convo_id
            and s["metadata"].get("chunk_index") == chunk_id
            for s in selected
        )

        if already_selected:
            continue

        if convo_id in seen_conversations:
            continue

        selected.append(item)
        seen_conversations.add(convo_id)

    for item in merged_results:
        if len(selected) >= final_k:
            break

        meta = item["metadata"]
        convo_id = meta.get("conversation_index")
        chunk_id = meta.get("chunk_index")

        already_selected = any(
            s["metadata"].get("conversation_index") == convo_id
            and s["metadata"].get("chunk_index") == chunk_id
            for s in selected
        )

        if already_selected:
            continue

        selected.append(item)

    return selected[:final_k]


def rerank_results(user_query: str, selected_results, top_k: int = TOP_K_FINAL):
    if not selected_results:
        return []

    prompt_parts = []

    for i, item in enumerate(selected_results, start=1):
        meta = item["metadata"]
        doc = make_snippet(item["document"], 500)

        prompt_parts.append(
            f"""[{i}]
Title: {meta.get("title", "Untitled")}
Snippet:
{doc}
"""
        )

    joined = "\n\n".join(prompt_parts)

    prompt = f"""You are reranking semantic search results for a personal knowledge AI.

User query:
{user_query}

Below are retrieved note candidates.

Rank the most relevant candidates for answering the user's query.

Rules:
- Prefer directly relevant conceptual matches
- Penalize vague metaphor overlap
- Penalize weak keyword coincidence
- Prefer notes that actually discuss the target concept
- Return 5 result numbers if possible
- If only 1-2 are strong, still include the next best related candidates
- One line only
- Example format: 2,5,1,4,3

Candidates:

{joined}
"""

    ranked = []

    try:
        response = llm_service.chat(
            None,
            [{"role": "user", "content": prompt}],
        )

        text = response["message"]["content"]
        import re

        matches = re.findall(r"\d+", text)

        for match in matches:
            idx = int(match) - 1

            if 0 <= idx < len(selected_results):
                ranked.append(selected_results[idx])

    except Exception:
        ranked = []

    final = []
    seen = set()

    for item in ranked:
        key = (
            item["metadata"].get("conversation_index"),
            item["metadata"].get("chunk_index")
        )

        if key in seen:
            continue

        seen.add(key)
        final.append(item)

        if len(final) >= top_k:
            return final

    for item in selected_results:
        key = (
            item["metadata"].get("conversation_index"),
            item["metadata"].get("chunk_index")
        )

        if key in seen:
            continue

        seen.add(key)
        final.append(item)

        if len(final) >= top_k:
            break

    return final[:top_k]
