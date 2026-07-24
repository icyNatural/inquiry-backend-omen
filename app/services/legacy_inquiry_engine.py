import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
import ollama

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "chat_notes"
OLLAMA_MODEL = "gemma3:4b"

TOP_K_FINAL = 5
TOP_K_PER_QUERY = 8
MAX_HISTORY_TURNS = 0
SNIPPET_LEN = 220

REFINED_ROOT = Path("refined_notes")

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=COLLECTION_NAME)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

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

MODE_PROMPTS = {
    "recall": """You are a simple personal memory recall assistant.

Use the retrieved notes to answer the current question directly.

Rules:
- Keep it short.
- Recall what the notes say.
- Do not continue old conversations.
- Do not give next steps unless asked.
""",
    "synthesis": """You are the user's personal knowledge synthesis assistant.

Goal:
Combine retrieved notes into a more integrated higher-level answer.

Return this structure:

Direct answer:
<short paragraph>

Main patterns:
- bullet
- bullet
- bullet

Cross-note synthesis:
<short paragraph>

Key note signals:
- title -> short signal
- title -> short signal
""",
    "raw": """You are the user's raw note retrieval assistant.

Goal:
Minimize interpretation. Show the strongest retrieved signals directly.

Return this structure:

Direct answer:
<1-2 lines max>

Raw signals:
- title -> quoted or near-quoted signal
- title -> quoted or near-quoted signal
- title -> quoted or near-quoted signal
- title -> quoted or near-quoted signal

Minimal synthesis:
<one short sentence or 'None.'>

Rules:
- Be as literal as possible.
- Prefer extraction over explanation.
""",
    "framework": """You are the user's framework extraction assistant.

Goal:
Turn retrieved notes into a reusable structured model.

Return this structure:

Direct answer:
<short paragraph>

Framework elements:
- principle:
- mechanism:
- pattern:
- failure mode:
- application:

Relevant note signals:
- title -> short signal
- title -> short signal

Optional synthesis:
<short only if useful>
"""
}


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
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
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
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
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


ensure_refined_dirs()

print("Personal Knowledge AI is ready.")
print("Type your question and press Enter.")
print("Commands: exit, quit, clear, sources off, sources on, save, save as <name>")
print("Library: list notes, find tag <tag>, list tags, find mode <mode>, open last, open note <name>")
print("Modes: mode recall, mode synthesis, mode raw, mode framework")
print("Debug: debug on, debug off\n")

while True:
    try:
        query = input(f"Ask your AI [{current_mode}]: ").strip()
    except KeyboardInterrupt:
        print("\nGoodbye.")
        break

    if not query:
        continue

    cmd = query.lower()

    if cmd in {"exit", "quit"}:
        print("Goodbye.")
        break

    if cmd == "clear":
        history.clear()
        print("Conversation history cleared.\n")
        continue

    if cmd == "sources off":
        show_sources = False
        print("Source display turned off.\n")
        continue

    if cmd == "sources on":
        show_sources = True
        print("Source display turned on.\n")
        continue

    if cmd == "debug on":
        debug_mode = True
        print("Debug mode turned on.\n")
        continue

    if cmd == "debug off":
        debug_mode = False
        print("Debug mode turned off.\n")
        continue

    if cmd == "save":
        save_last_answer()
        continue

    if cmd.startswith("save as "):
        custom_name = query[8:].strip()
        if not custom_name:
            print("Provide a name after 'save as'.\n")
        else:
            save_last_answer(custom_name)
        continue

    if cmd == "list notes":
        list_notes()
        continue

    if cmd.startswith("find tag "):
        tag = query[9:].strip()
        if not tag:
            print("Provide a tag after 'find tag'.\n")
        else:
            find_tag(tag)
        continue

    if cmd == "list tags":
        list_tags()
        continue

    if cmd.startswith("find mode "):
        mode_name = query[10:].strip()
        if not mode_name:
            print("Provide a mode after 'find mode'.\n")
        else:
            find_mode(mode_name)
        continue

    if cmd == "open last":
        open_last()
        continue

    if cmd.startswith("open note "):
        name = query[10:].strip()
        open_note(name)
        continue

    if cmd.startswith("mode "):
        new_mode = cmd.replace("mode ", "", 1).strip()
        if new_mode in MODE_PROMPTS:
            current_mode = new_mode
            print(f"Mode changed to: {current_mode}\n")
        else:
            print("Unknown mode. Use: recall, synthesis, raw, framework\n")
        continue

    # Phase 4 retrieval upgrade
    search_queries = generate_search_queries(query, current_mode)
    merged_results = merge_multi_query_results(search_queries)
    selected_results = diversify_results(merged_results, TOP_K_FINAL)
    selected_results = rerank_results(query, selected_results, TOP_K_FINAL)

    documents = [item["document"] for item in selected_results]
    metadatas = [item["metadata"] for item in selected_results]

    context = interpret_sources(query, selected_results)
    messages = build_messages(query, context)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages
    )

    answer = response["message"]["content"]
    tags = generate_tags(answer, query, current_mode)

    history.append({
        "user": query,
        "assistant": answer
    })

    last_query = query
    last_answer = answer
    last_documents = documents
    last_metadatas = metadatas
    last_tags = tags
    last_search_queries = search_queries
    last_selected_results = selected_results

    if debug_mode:
        print("\n" + "=" * 80)
        print("RETRIEVAL DEBUG")
        print("=" * 80 + "\n")
        print("Search queries used:")
        for i, q in enumerate(search_queries, start=1):
            print(f"{i}. {q}")
        print()

    print("\n" + "=" * 80)
    print("AI ANSWER")
    print("=" * 80 + "\n")
    print(answer)

    print(f"\nTags: {tags}")

    if show_sources:
        print("\n" + "=" * 80)
        print("NOTE SIGNALS USED")
        print("=" * 80 + "\n")
        for i, item in enumerate(selected_results, start=1):
            doc = item["document"]
            meta = item["metadata"]
            source_query = item.get("source_query", "unknown")
            print(f"{i}. {meta.get('title', 'Untitled')}")
            print(f"   signal: {make_snippet(doc)}")
            print(f"   location: convo {meta.get('conversation_index')} | chunk {meta.get('chunk_index')}")
            print(f"   retrieved_via: {source_query}")
            print()

    print()























