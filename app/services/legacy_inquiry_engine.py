import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from app.services import llm_service

from app.services.scope_service import ScopeProcessor
from app.services.query_analysis_service import analyze_query
from app.services.mode_prompts import MODE_PROMPTS, SUPPORTED_MODES
from app.models.inquiry import KnowledgePolicy
from app.services import retrieval_service
from app.services import cognition_service

CHROMA_DIR = os.getenv("INQUIRY_CHROMA_PATH", r"C:\Users\justc\OneDrive\Documents\ai_brain_notes - Copy\chroma_db")
COLLECTION_NAME = os.getenv("INQUIRY_CHROMA_COLLECTION", "chat_notes")
OLLAMA_MODEL = os.getenv("INQUIRY_OLLAMA_MODEL", "gemma3:4b")

TOP_K_FINAL = 5
TOP_K_PER_QUERY = 8
MAX_HISTORY_TURNS = 0
SNIPPET_LEN = 220

REFINED_ROOT = Path(os.getenv("INQUIRY_REFINED_NOTES_PATH", r"C:\Users\justc\OneDrive\Documents\ai_brain_notes - Copy\inquiry_engine_step3_complete\refined_notes"))

# Use memory_service where possible for collection/embedding access (centralized in Phase 3)
from app.services.memory_service import get_collection, get_embedding_model

# Keep a module-level reference for backward compatibility with code that expects
# a `collection` and `embed_model` variables; these are obtained from memory_service.
collection = get_collection()
embed_model = get_embedding_model()

history = []
current_mode = "recall"
show_sources = True
debug_mode = False

last_query = None
last_answer = None
last_documents = None
last_metadatas = None
last_tags = None
last_saved_path = None
last_search_queries = None
last_selected_results = None

# Mode prompts are centralized in app.services.mode_prompts


def ensure_refined_dirs():
    for mode in MODE_PROMPTS.keys():
        (REFINED_ROOT / mode).mkdir(parents=True, exist_ok=True)


def slugify(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if not text:
        text = "note"
    return text[:max_len].strip("-")


def unique_path(base_dir: Path, slug: str) -> Path:
    path = base_dir / f"{slug}.md"
    if not path.exists():
        return path

    counter = 2
    while True:
        candidate = base_dir / f"{slug}-{counter}.md"
        if not candidate.exists():
            return candidate
        counter += 1


def make_snippet(text: str, max_len: int = SNIPPET_LEN) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def normalize_tag(tag: str) -> str:
    tag = tag.strip().lower()
    if not tag:
        return "#untagged"
    if not tag.startswith("#"):
        tag = f"#{tag}"
    return tag


def generate_search_queries(user_query: str, mode: str):
    """
    Simple query expansion.
    No LLM rewriting.
    No builder-style fallback.
    Only light canonical boosters.
    """

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


def retrieve_for_query(query: str, top_k: int = TOP_K_PER_QUERY):
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
    """
    Retrieve for each query and merge results.
    Lower distance is better.
    """
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
    """
    Diversify retrieval results:
    - avoid one conversation dominating
    - avoid one title repeating too much
    - preserve strongest semantic matches
    """

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
    """
    Soft LLM reranking pass.
    The LLM chooses the strongest matches first.
    Then we backfill with embedding-ranked results so we still keep enough context.
    """

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
            OLLAMA_MODEL,
            [{"role": "user", "content": prompt}]
        )

        text = response["message"]["content"]
        matches = re.findall(r"\d+", text)

        for match in matches:
            idx = int(match) - 1

            if 0 <= idx < len(selected_results):
                ranked.append(selected_results[idx])

    except Exception:
        ranked = []

    # dedupe ranked picks
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

    # backfill with original embedding/diversified order
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


def interpret_sources(user_query: str, selected_results):
    """
    Middle layer:
    Converts messy retrieved chunks into clean memory signals before final answer.
    """

    source_blocks = []

    for i, item in enumerate(selected_results, start=1):
        meta = item["metadata"]
        doc = make_snippet(item["document"], 900)

        source_blocks.append(
            f"""[Source {i}]
Title: {meta.get("title", "Untitled")}
Retrieved via: {item.get("source_query", "unknown")}
Text:
{doc}
"""
        )

    joined = "\n\n".join(source_blocks)

    prompt = f"""You are cleaning retrieved memory chunks for a personal memory AI.

User question:
{user_query}

Retrieved chunks:
{joined}

Task:
Turn the retrieved chunks into clean source signals.

Rules:
- Do NOT answer the user yet.
- Do NOT give advice.
- Do NOT continue old conversations.
- Do NOT ask for files, app.py, code, screenshots, or data.
- Extract only what is relevant to the user's current question.
- Ignore workflow language, old next steps, and old assistant instructions.
- Prefer concepts, definitions, claims, patterns, and decisions.

Return this exact structure:

Clean memory signals:
1. Title -> useful signal
2. Title -> useful signal
3. Title -> useful signal

Relevant concepts:
- concept
- concept
- concept

Ignore:
- old workflow instruction if any
- unrelated tangent if any
"""

    try:
        response = llm_service.chat(
            OLLAMA_MODEL,
            [{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except Exception:
        fallback = []
        for i, item in enumerate(selected_results, start=1):
            meta = item["metadata"]
            fallback.append(
                f"{i}. {meta.get('title', 'Untitled')} -> {make_snippet(item['document'], 250)}"
            )
        return "\n".join(fallback)

def build_context(documents, metadatas):
    parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        parts.append(
            f"[Source {i}]\n"
            f"Title: {meta.get('title', 'Untitled')}\n"
            f"Conversation Index: {meta.get('conversation_index')}\n"
            f"Chunk Index: {meta.get('chunk_index')}\n"
            f"Content:\n{doc}"
        )
    return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(parts)


def get_final_answer_prompt(mode: str) -> str:
    prompts = {
        "recall": """You are answering in RECALL mode.

Goal:
Explain what the user's memory says about the current question.

Behavior:
- Give a clear answer, not just tiny extraction.
- Stay grounded in the organized memory signals.
- Do not give advice or next steps unless asked.
- Do not build anything.
- Do not continue old conversations.
- Mild synthesis is allowed only to explain the memory clearly.

Return:
Direct answer:
<clear memory-based explanation>

Key memory points:
- point
- point
- point

Relevant source signals:
- title -> signal
- title -> signal
""",

        "synthesis": """You are answering in SYNTHESIS mode.

Goal:
Connect patterns across the organized memory signals.

Behavior:
- Find relationships between sources.
- Explain the larger pattern.
- You may interpret, but stay grounded.
- Do not invent unsupported claims.
- Do not continue old conversations.

Return:
Direct answer:
<integrated answer>

Main patterns:
- pattern
- pattern
- pattern

Cross-note synthesis:
<how the memories connect>

Relevant source signals:
- title -> signal
- title -> signal
""",

        "raw": """You are answering in RAW mode.

Goal:
Show the retrieved memory signals with minimal interpretation.

Behavior:
- Keep interpretation low.
- Prefer direct extracted signals.
- Do not synthesize heavily.
- Do not give advice.
- Do not continue old conversations.

Return:
Direct answer:
<1-2 line answer>

Raw memory signals:
- title -> signal
- title -> signal
- title -> signal

Minimal synthesis:
<one sentence or None>
""",

        "framework": """You are answering in FRAMEWORK mode.

Goal:
Turn the organized memory signals into a reusable model.

Behavior:
- Extract principles, mechanisms, patterns, failure modes, and applications.
- Stay grounded in the organized memory signals.
- Do not create unrelated frameworks.
- Do not continue old conversations.

Return:
Direct answer:
<short framework summary>

Framework:
- Principle:
- Mechanism:
- Pattern:
- Failure mode:
- Application:

Relevant source signals:
- title -> signal
- title -> signal
"""
    }

    return prompts.get(mode, prompts["recall"])


def build_messages(user_query: str, context: str):

    mode_instructions = {
        "recall": """Mode behavior:
- Explain what the memory says clearly.
- Give enough context to be useful, but do not overbuild.
- Do not create plans or next steps.
- Stay close to the retrieved memory signals.
""",

        "synthesis": """Mode behavior:
- Connect patterns across retrieved memory signals.
- Explain relationships between ideas.
- You may interpret, but stay grounded.
- Do not invent unsupported claims.
""",

        "raw": """Mode behavior:
- Show the retrieved signals with minimal interpretation.
- Keep wording close to the source signals.
- Do not expand unless necessary.
""",

        "framework": """Mode behavior:
- Convert retrieved memory into a reusable framework.
- Extract principles, mechanisms, patterns, failure modes, and applications.
- Stay grounded in the source signals.
"""
    }

    system_prompt = f"""You are a personal memory assistant.

You answer from organized memory signals, not raw old conversations.

Universal rules:
- Answer the current question directly.
- Do not continue old conversations.
- Do not ask for files, app.py, code, screenshots, raw data, or next steps.
- Do not build anything unless the current question asks you to build.
- Stay grounded in the organized memory signals.

{mode_instructions.get(current_mode, mode_instructions["recall"])}
"""

    messages = [{"role": "system", "content": system_prompt}]

    output_formats = {
        "recall": """Return:

Direct recall:
<clear explanation from memory>

Key memory points:
- point
- point
- point

Relevant source signals:
- title -> signal
- title -> signal
""",

        "synthesis": """Return:

Synthesis:
<integrated explanation>

Patterns:
- pattern
- pattern
- pattern

Cross-links:
- idea -> related idea
- idea -> related idea
""",

        "raw": """Return:

Raw memory signals:

1.
Title:
Retrieved via:
Signal:

2.
Title:
Retrieved via:
Signal:

3.
Title:
Retrieved via:
Signal:

Rules:
- Keep wording close to retrieved memory.
- Do not heavily summarize.
- Do not explain unless necessary.
- Preserve useful terminology from memory.
- Prefer multiple compact signals over one compressed sentence.
""",

        "framework": """Return:

Framework:
<name or short description>

Principles:
- principle
- principle

Mechanisms:
- mechanism
- mechanism

Failure modes:
- failure mode
- failure mode

Applications:
- application
- application
"""
    }

    messages.append({
        "role": "user",
        "content": f"""Current question:
{user_query}

Organized memory signals:
{context}

Answer using the current mode: {current_mode}

{output_formats.get(current_mode, output_formats["recall"])}
"""
    })

    return messages


def render_sources_for_save(documents, metadatas, selected_results=None) -> str:
    lines = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        lines.append(f"## Source {i}")
        lines.append(f"- Title: {meta.get('title', 'Untitled')}")
        lines.append(f"- Conversation Index: {meta.get('conversation_index')}")
        lines.append(f"- Chunk Index: {meta.get('chunk_index')}")
        if selected_results and i - 1 < len(selected_results):
            lines.append(f"- Retrieved Via: {selected_results[i - 1].get('source_query', 'unknown')}")
        lines.append(f"- Signal: {make_snippet(doc, 300)}")
        lines.append("")
    return "\n".join(lines).strip()


def generate_tags(answer_text: str, query_text: str, mode: str) -> str:
    return "#untagged"


def save_last_answer(custom_name=None):
    global last_query, last_answer, last_documents, last_metadatas, last_tags, last_saved_path, last_selected_results

    if not last_answer or not last_query:
        print("Nothing to save yet.\n")
        return

    base_dir = REFINED_ROOT / current_mode
    base_dir.mkdir(parents=True, exist_ok=True)

    title = custom_name.strip() if custom_name else last_query
    slug = slugify(title)
    path = unique_path(base_dir, slug)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sources_block = render_sources_for_save(
        last_documents or [],
        last_metadatas or [],
        last_selected_results or []
    )
    tags = last_tags if last_tags else generate_tags(last_answer, last_query, current_mode)

    content = f"""# {title}

- Saved: {timestamp}
- Mode: {current_mode}
- Query: {last_query}
- Tags: {tags}

## AI Answer

{last_answer}

## Note Signals Used

{sources_block if sources_block else "None."}
"""

    path.write_text(content, encoding="utf-8")
    last_saved_path = path
    print(f"Saved to {path}\n")


def get_all_refined_notes():
    if not REFINED_ROOT.exists():
        return []

    files = []
    for path in REFINED_ROOT.rglob("*.md"):
        if path.is_file():
            files.append(path)

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def extract_tags_from_file(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []

    match = re.search(r"^- Tags:\s*(.+)$", text, re.MULTILINE)
    if not match:
        return []

    raw_tags = match.group(1).strip()
    return [normalize_tag(tag) for tag in raw_tags.split() if tag.strip().startswith("#")]


def list_notes(limit: int = 20):
    notes = get_all_refined_notes()

    if not notes:
        print("No refined notes found.\n")
        return

    print("\n" + "=" * 80)
    print("REFINED NOTES")
    print("=" * 80 + "\n")

    for i, path in enumerate(notes[:limit], start=1):
        rel = path.relative_to(REFINED_ROOT.parent)
        mode = path.parent.name
        print(f"{i}. [{mode}] {rel}")

    if len(notes) > limit:
        print(f"\nShowing {limit} of {len(notes)} notes.\n")
    else:
        print()


def find_tag(tag: str):
    tag = normalize_tag(tag)

    notes = get_all_refined_notes()
    matches = []

    for path in notes:
        tags = [t.lower() for t in extract_tags_from_file(path)]
        if tag in tags:
            matches.append((path, tags))

    if not matches:
        print(f"No refined notes found with tag {tag}\n")
        return

    print("\n" + "=" * 80)
    print(f"NOTES WITH TAG {tag}")
    print("=" * 80 + "\n")

    for i, (path, tags) in enumerate(matches, start=1):
        rel = path.relative_to(REFINED_ROOT.parent)
        print(f"{i}. {rel}")
        print(f"   tags: {' '.join(tags)}")

    print()


def list_tags():
    notes = get_all_refined_notes()
    counter = Counter()

    for path in notes:
        for tag in extract_tags_from_file(path):
            counter[tag.lower()] += 1

    if not counter:
        print("No tags found yet.\n")
        return

    print("\n" + "=" * 80)
    print("ALL TAGS")
    print("=" * 80 + "\n")

    for tag, count in counter.most_common():
        print(f"{tag} ({count})")

    print()


def find_mode(mode_name: str):
    mode_name = mode_name.strip().lower()
    mode_dir = REFINED_ROOT / mode_name

    if mode_name not in MODE_PROMPTS:
        print("Unknown mode. Use: recall, synthesis, raw, framework\n")
        return

    if not mode_dir.exists():
        print(f"No directory found for mode {mode_name}\n")
        return

    notes = sorted(
        [p for p in mode_dir.glob("*.md") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not notes:
        print(f"No notes found in mode {mode_name}\n")
        return

    print("\n" + "=" * 80)
    print(f"NOTES IN MODE {mode_name}")
    print("=" * 80 + "\n")

    for i, path in enumerate(notes, start=1):
        rel = path.relative_to(REFINED_ROOT.parent)
        print(f"{i}. {rel}")

    print()


def open_last():
    global last_saved_path

    if not last_saved_path:
        print("No note has been saved in this session yet.\n")
        return

    if not last_saved_path.exists():
        print("Last saved file no longer exists.\n")
        return

    text = last_saved_path.read_text(encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"OPEN LAST: {last_saved_path}")
    print("=" * 80 + "\n")
    print(text)
    print()


def open_note(name: str):
    name = name.strip().lower()
    if not name:
        print("Provide a note name after 'open note'.\n")
        return

    notes = get_all_refined_notes()
    matches = []

    for path in notes:
        stem = path.stem.lower()
        rel = str(path.relative_to(REFINED_ROOT.parent)).lower()
        if name in stem or name in rel:
            matches.append(path)

    if not matches:
        print(f"No note found matching: {name}\n")
        return

    chosen = matches[0]
    text = chosen.read_text(encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"OPEN NOTE: {chosen}")
    print("=" * 80 + "\n")
    print(text)
    print()
def investigate(
    user_query: str,
    mode: str = "recall",
    scope: dict | None = None,
    knowledge_policy: str | KnowledgePolicy = "memory_only",
) -> dict:
    """
    Run one complete Inquiry Engine request without terminal input.

    Pipeline:
    query expansion
    -> Chroma retrieval
    -> result diversification
    -> Ollama reranking
    -> source interpretation
    -> mode-specific response
    """

    global current_mode

    clean_query = user_query.strip()
    clean_mode = mode.strip().lower()

    if not clean_query:
        raise ValueError("Question cannot be empty.")

    if clean_mode not in MODE_PROMPTS:
        raise ValueError(
            "Unsupported mode. Use recall, synthesis, framework, raw, grounding, or mapping."
        )

    current_mode = clean_mode
    ensure_refined_dirs()

    # Run query analysis (normalization + classification) before retrieval
    qa = analyze_query(clean_query)

    # Use normalized query for search/retrieval
    normalized_for_search = qa.normalized_query

    search_queries = retrieval_service.generate_search_queries(
        normalized_for_search,
        current_mode,
    )

    merged_results = retrieval_service.merge_multi_query_results(search_queries)

    clean_scope = scope or {}

    if clean_scope:
        merged_results = ScopeProcessor.filter_results(
            merged_results,
            clean_scope,
        )

    selected_results = retrieval_service.diversify_results(
        merged_results,
        TOP_K_FINAL,
    )

    selected_results = retrieval_service.rerank_results(
        clean_query,
        selected_results,
        TOP_K_FINAL,
    )

    # Normalize and validate knowledge policy to the canonical `KnowledgePolicy` enum.
    if isinstance(knowledge_policy, KnowledgePolicy):
        kp = knowledge_policy
    else:
        try:
            kp = KnowledgePolicy(knowledge_policy)
        except Exception:
            raise ValueError("Unsupported knowledge_policy. Use 'memory_only' or 'memory_plus_model'.")

    # Handle no-memory cases according to policy
    if not selected_results:
        if kp == KnowledgePolicy.memory_only:
            # include query analysis trace in early return
            return {
                "status": "ok",
                "query": clean_query,
                "original_query": qa.original_query,
                "normalized_query": qa.normalized_query,
                "query_changed": qa.changed,
                "question_type": qa.question_type,
                "source_plan": {
                    "personal_memory": qa.source_plan.personal_memory,
                    "external_web": qa.source_plan.external_web,
                    "scientific_sources": qa.source_plan.scientific_sources,
                    "model_knowledge": qa.source_plan.model_knowledge,
                },
                "reasoning_depth": qa.reasoning_depth,
                "mode": current_mode,
                "model": OLLAMA_MODEL,
                "answer": "No memories matched the requested scope.",
                "organized_signals": "",
                "search_queries": search_queries,
                "retrieved_count": len(merged_results),
                "selected_count": 0,
                "sources": [],
            }

        # memory_plus_model: prepare an organized_signals note indicating fallback
        organized_signals = (
            "No matching memories were found.\n\n"
            "The following answer is based on the model's general knowledge."
        )
    else:
        organized_signals = cognition_service.interpret_sources(
            clean_query,
            selected_results,
        )

    messages = cognition_service.build_messages(
        clean_query,
        organized_signals,
        current_mode,
    )

    response = llm_service.chat(OLLAMA_MODEL, messages)
    answer = response["message"]["content"].strip()

    if kp == KnowledgePolicy.memory_plus_model and not selected_results:
        answer = (
            "No matching memories were found.\n\n"
            "The following answer is based on the model's general knowledge.\n\n"
            + answer
        )

    sources = []
    for item in selected_results:
        metadata = item["metadata"]

        sources.append(
            {
                "title": metadata.get("title", "Untitled"),
                "conversation_index": metadata.get("conversation_index"),
                "chunk_index": metadata.get("chunk_index"),
                "source": metadata.get("source"),
                "domain": metadata.get("domain"),
                "distance": item.get("distance"),
                "retrieved_via": item.get("source_query"),
                "signal": make_snippet(
                    item["document"],
                    SNIPPET_LEN,
                ),
                "raw_text": item["document"],
                "metadata": metadata,
            }
        )

    result = {
        "status": "ok",
        "query": clean_query,
        # Query analysis trace
        "original_query": qa.original_query,
        "normalized_query": qa.normalized_query,
        "query_changed": qa.changed,
        "question_type": qa.question_type,
        "source_plan": {
            "personal_memory": qa.source_plan.personal_memory,
            "external_web": qa.source_plan.external_web,
            "scientific_sources": qa.source_plan.scientific_sources,
            "model_knowledge": qa.source_plan.model_knowledge,
        },
        "reasoning_depth": qa.reasoning_depth,
        "mode": current_mode,
        "model": OLLAMA_MODEL,
        "answer": answer,
        "organized_signals": organized_signals,
        "search_queries": search_queries,
        "retrieved_count": len(merged_results),
        "selected_count": len(selected_results),
        "sources": sources,
    }

    return result
