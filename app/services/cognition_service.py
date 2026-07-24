import re
import ollama
from typing import Dict, Any, List

import memory_manager
from scope_processor import ScopeProcessor

# Default model configuration
DEFAULT_MODEL = "gemma3:12b"

# Mode prompts for final answer reasoning
MODE_REASONING_PROMPTS = {
    "recall": """You are in RECALL mode.
Goal: Answer what the user's memory says about their question, directly and clearly.

Instructions:
- Stay strictly grounded in the interpreted memory signals.
- Keep it compact and direct.
- Do not make recommendations, plans, next steps, or continue old conversations.
- Do not write code or propose actions unless explicitly requested by the query.

Format:
Direct Answer:
<direct memory-based explanation>

Key Points:
- <point 1>
- <point 2>
""",

    "synthesis": """You are in SYNTHESIS mode.
Goal: Connect patterns, ideas, and observations across multiple memory signals.

Instructions:
- Analyze how different notes and observations connect.
- Detail the underlying themes.
- Interpret the relationships while remaining strictly grounded in the sources.

Format:
Direct Answer:
<integrated summary>

Core Themes:
- <theme 1>: <explanation>
- <theme 2>: <explanation>

Cross-Note Connections:
<how the different notes and concepts interface and balance each other>
""",

    "framework": """You are in FRAMEWORK mode.
Goal: Transform retrieved signals into a reusable model, principle, or operational structure.

Instructions:
- Extract core structural components.
- Do not suggest external tools unless they exist in the memory.
- Stay grounded in the source signals.

Format:
Direct Answer:
<short explanation of the framework>

Framework Architecture:
- **Principle**: <core belief or truth from memory>
- **Mechanism**: <how it operates, step-by-step>
- **Pattern**: <recurring behavior or state>
- **Failure Mode**: <how it breaks or loops unproductively>
- **Application**: <when and how to apply this model>
""",

    "raw": """You are in RAW mode.
Goal: Expose source signals directly with absolute minimum interpretation.

Instructions:
- Act as an extraction mechanism rather than a summarizing assistant.
- Do not add comments, advice, or heavy summaries.
- Keep wording extremely close to the interpreted source signals.

Format:
Direct Answer:
<1-2 lines maximum statement>

Extracted Signals:
- [Source Title]: <near-quoted signal or core claim>
- [Source Title]: <near-quoted signal or core claim>
""",

    "grounding": """You are in GROUNDING mode.
Goal: Resolve ambiguity, identify abstract terms, and stabilize meaning before reasoning.

Instructions:
- Examine the query and the interpreted memory signals.
- Identify any ambiguous terms (e.g. "flow", "energy", "engine") that have multiple meanings, abstract contexts, or split definitions.
- Distinguish between these contexts (e.g. physiological vs productivity vs philosophical).
- Formulate clarifying questions if the meaning remains unstable.

Format:
Ambiguous Terms Identified:
- <term>: <brief explanation of splits/meanings found in memory>

Grounding Definition:
<how these terms are defined and stabilized in the context of the memory>

Clarifying Questions (Optional):
- <question 1 to resolve splits>
- <question 2 to resolve splits>

Grounded Answer:
<grounded response to the query using the stabilized definition>
""",

    "mapping": """You are in MAPPING mode.
Goal: Extract concepts, relationships, dependencies, lineages, and conceptual topology.

Instructions:
- Focus on how ideas are structured and flow into one another.
- Identify directed connections (Concept A -> [relationship] -> Concept B).
- Identify dependencies (Concept A depends on Concept B).
- Map temporal lineages (Concept A evolved from Concept B).

Format:
Core Concepts:
- <concept 1>: <brief meaning>
- <concept 2>: <brief meaning>

Directed Relationships:
- <Concept A> -> [relationship type] -> <Concept B> (e.g., "Attention shaping -> depends_on -> Zero-state thinking")

Lineage & Evolution:
<description of how these ideas branched, merged, or evolved over time>
"""
}

class CognitionLayer:
    @staticmethod
    def call_ollama(model_name: str, messages: List[Dict[str, str]]) -> str:
        """
        Model-agnostic Ollama runner.
        Falls back to first available model if the requested model is not downloaded.
        """
        try:
            response = ollama.chat(model=model_name, messages=messages)
            return response["message"]["content"].strip()
        except Exception as e:
            err_msg = str(e)
            print(f"Ollama execution error for model '{model_name}': {e}")
            
            # Fallback routine if model not found
            if "not found" in err_msg.lower() or "404" in err_msg:
                try:
                    models_list = ollama.list()
                    available_models = [m['name'] for m in models_list.get('models', [])]
                    if available_models:
                        fallback_model = available_models[0]
                        # avoid infinite loop
                        if fallback_model != model_name:
                            print(f"Warning: Model '{model_name}' not found. Falling back to '{fallback_model}'")
                            response = ollama.chat(model=fallback_model, messages=messages)
                            return f"[FALLBACK TO {fallback_model}]\n\n{response['message']['content'].strip()}"
                except Exception as ex:
                    print(f"Failed to query available models for fallback: {ex}")
            raise RuntimeError(f"Ollama inference failed: {err_msg}")

    @classmethod
    def expand_query(cls, query: str, model_name: str) -> List[str]:
        """
        Light query expansion. Generates booster terms to retrieve more diverse notes.
        """
        prompt = f"""You are generating semantic search queries to retrieve personal knowledge notes.
User query:
{query}

Task:
Generate 3 distinct search terms or phrases (booster queries) that would help retrieve notes related to this topic.
- Use different terminology, synonyms, or conceptual matches.
- Keep them short (3-6 words).
- Do not explain why you chose them.
- Output ONLY the 3 queries, one per line.

Examples:
Input: flow state
Output:
mental flow and state setup
rhythmic focus practices
attention shaping routines

Output:
"""
        messages = [{"role": "user", "content": prompt}]
        response_text = cls.call_ollama(model_name, messages)
        
        # parse lines
        queries = [query.strip()]
        for line in response_text.split("\n"):
            line = re.sub(r"^\d+[\.\-\)]\s*", "", line).strip() # clean bullet numbers
            if line and line.lower() not in {q.lower() for q in queries}:
                queries.append(line)
                
        return queries[:4]

    @classmethod
    def rerank_results(cls, query: str, results: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
        """
        Soft LLM reranking pass. Ranks matching candidates based on conceptual overlap,
        penalizing vague metaphor matching.
        """
        if not results:
            return []
            
        candidates_str = []
        for i, item in enumerate(results, start=1):
            doc_snippet = item["document"][:400].replace("\n", " ")
            meta = item["metadata"]
            candidates_str.append(
                f"[{i}] Title: {meta.get('title', 'Untitled')} | Source: {meta.get('source')} | Snippet: {doc_snippet}"
            )
            
        candidates_joined = "\n".join(candidates_str)
        
        prompt = f"""You are reranking semantic search results for a personal knowledge AI.
User query:
{query}

Rank these note candidates from most relevant to least relevant:
- Prefer conceptual alignment and direct answers.
- Penalize vague word overlap or metaphor similarity.
- Return the indices of the ranked candidates as a comma-separated list on a single line.
- Example output: 3,1,4,2,5

Candidates:
{candidates_joined}

Output:
"""
        messages = [{"role": "user", "content": prompt}]
        try:
            response_text = cls.call_ollama(model_name, messages)
            matches = re.findall(r"\d+", response_text)
            
            ranked_results = []
            seen = set()
            for m in matches:
                idx = int(m) - 1
                if 0 <= idx < len(results):
                    key = (results[idx]["metadata"].get("path") or results[idx]["metadata"].get("conversation_index"), results[idx]["metadata"].get("chunk_index"))
                    if key not in seen:
                        seen.add(key)
                        ranked_results.append(results[idx])
                        
            # Backfill any missing ones
            for item in results:
                key = (item["metadata"].get("path") or item["metadata"].get("conversation_index"), item["metadata"].get("chunk_index"))
                if key not in seen:
                    seen.add(key)
                    ranked_results.append(item)
                    
            return ranked_results
        except Exception as e:
            print(f"Reranking error (using embedding order): {e}")
            return results

    @classmethod
    def interpret_sources(cls, query: str, results: List[Dict[str, Any]], model_name: str) -> str:
        """
        Source Interpretation Layer.
        Converts messy raw retrieved chunks into clean structured memory signals.
        RAW CHUNKS MUST NEVER BE SENT TO THE FINAL ANSWER GENERATOR.
        """
        if not results:
            return "No memory signals retrieved."
            
        blocks = []
        for i, item in enumerate(results, start=1):
            meta = item["metadata"]
            blocks.append(
                f"""[Source {i}]
Title: {meta.get('title', 'Untitled')}
Namespace: {meta.get('source', 'unknown')}
Domain: {meta.get('domain', 'general')}
Text:
{item['document']}
"""
            )
            
        joined_sources = "\n\n".join(blocks)
        
        prompt = f"""You are an intermediate source interpreter for a personal cognition system.
Your job is to translate messy retrieved conversation fragments or notes into clean, structured semantic signals.

User query:
{query}

Retrieved Chunks:
{joined_sources}

Task:
Extract only the key insights, definitions, principles, patterns, and claims relevant to the query.
- Do NOT answer the user query directly.
- Do NOT summarize unrelated tangents.
- Strip conversational headers, assistant noise, and workflow instructions.

Format:
Clean memory signals:
1. [Source Title] -> <core relevant claim or insight>
2. [Source Title] -> <core relevant claim or insight>

Relevant concepts:
- <concept name>: <relevance to query>
- <concept name>: <relevance to query>
"""
        messages = [{"role": "user", "content": prompt}]
        return cls.call_ollama(model_name, messages)

    @classmethod
    def execute_cognition_pipeline(cls, query: str, mode: str, scope: Dict[str, Any], model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
        """
        The main five-layer cognition pipeline execution:
        1. Retrieval: semantic search + scope filters (handled by memory_manager & scope_processor).
        2. Rerank: soft LLM reranker.
        3. Source Interpretation: intermediate clean-up (raw chunks are compressed).
        4. Mode-Specific Reasoning: processes interpreted signals into a formatted answer.
        """
        mode = mode.lower()
        if mode not in MODE_REASONING_PROMPTS:
            mode = "recall"
            
        # 1. RETRIEVAL (Scoped)
        collection = memory_manager.get_collection()
        model = memory_manager.get_embedding_model()
        
        # A. Query Expansion
        expanded_queries = cls.expand_query(query, model_name)
        
        # B. Multi-Query Retrieval
        raw_candidates = []
        seen_keys = set()
        
        where_filter = ScopeProcessor.build_chroma_where(scope)
        
        for eq in expanded_queries:
            eq_emb = model.encode([eq]).tolist()[0]
            
            if where_filter:
                res = collection.query(query_embeddings=[eq_emb], n_results=10, where=where_filter)
            else:
                res = collection.query(query_embeddings=[eq_emb], n_results=10)
                
            docs = res["documents"][0] if res["documents"] else []
            metas = res["metadatas"][0] if res["metadatas"] else []
            dists = res.get("distances", [[]])[0]
            
            for doc, meta, dist in zip(docs, metas, dists):
                key = (meta.get("path") or meta.get("conversation_index"), meta.get("chunk_index"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    raw_candidates.append({
                        "document": doc,
                        "metadata": meta,
                        "distance": dist,
                        "booster_query": eq
                    })
                    
        # Sort by distance
        raw_candidates.sort(key=lambda x: x["distance"])
        
        # C. Apply Scope Layer Filters (include/exclude tags, terms, etc.)
        scoped_candidates = ScopeProcessor.filter_results(raw_candidates, scope)
        
        # Identify excluded ones for diagnostics
        excluded_candidates = []
        for item in raw_candidates:
            is_ok, reason = ScopeProcessor.matches_scope(item["document"], item["metadata"], scope)
            if not is_ok:
                excluded_candidates.append({
                    "title": item["metadata"].get("title", "Untitled"),
                    "source": item["metadata"].get("source", "unknown"),
                    "reason": reason
                })
                
        # 2. RERANKING
        top_candidates = scoped_candidates[:8] # rerank top 8
        reranked_candidates = cls.rerank_results(query, top_candidates, model_name)
        
        # Final set to interpret (top 4)
        selected_candidates = reranked_candidates[:4]
        
        # 3. SOURCE INTERPRETATION (RAW CHUNKS COMPRESSED TO SIGNALS)
        interpreted_signals = cls.interpret_sources(query, selected_candidates, model_name)
        
        # 4. MODE-SPECIFIC REASONING & FINAL ANSWER
        mode_instruction = MODE_REASONING_PROMPTS[mode]
        
        final_prompt = f"""You are in the final reasoning step of a personal knowledge assistant.
Answer behavior:
{mode_instruction}

User query:
{query}

Interpreted Memory Signals:
{interpreted_signals}

Instructions:
- Base your entire response ONLY on the Interpreted Memory Signals provided above.
- Do NOT use any external or raw conversational text.
- Follow the formatting rules of the selected mode.

Output:
"""
        messages = [
            {"role": "system", "content": f"You are a personal semantic memory assistant. Mode: {mode}."},
            {"role": "user", "content": final_prompt}
        ]
        
        final_answer = cls.call_ollama(model_name, messages)
        
        # 5. DIAGNOSTICS ASSEMBLY
        sources_used = []
        for item in selected_candidates:
            sources_used.append({
                "title": item["metadata"].get("title"),
                "source": item["metadata"].get("source"),
                "path": item["metadata"].get("path"),
                "distance": item["distance"]
            })
            
        return {
            "answer": final_answer,
            "mode": mode,
            "model": model_name,
            "interpreted_signals": interpreted_signals,
            "sources": sources_used,
            "scope": scope,
            "excluded_sources": excluded_candidates[:10] # limit diagnostics size
        }
