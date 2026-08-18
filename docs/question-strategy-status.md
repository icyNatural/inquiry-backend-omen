Question Strategy Status

This document reports the exact presence/state of the requested question-strategy symbols in the repository.

Checked symbols:
- clarify
- compare
- action
- challenge
- deepen
- choose_strategy
- generate_followup
- find_related_entries
- related_entry_ids
- session_id
- next_question

Summary:

- None of the above symbols exist as top-level functions or methods in the current `app/` runtime code.
- There are relevant archived implementations and related helpers in `app/legacy/` that provide comparable capabilities under different names (for example: `expand_query`, `rerank_results`, `interpret_sources`, `execute_cognition_pipeline` in `app/legacy/cognition_layer_original.py`).

Per-symbol status

- `clarify` : NOT PRESENT
  - Exact filename: None
  - Function/class name: None
  - Inputs: N/A
  - Outputs: N/A
  - Persistence: N/A
  - Dependencies: N/A
  - Safe to port: Not applicable (no implementation found)

- `compare` : NOT PRESENT
  - Exact filename: None
  - ... (same as above)

- `action` : NOT PRESENT

- `challenge` : NOT PRESENT

- `deepen` : NOT PRESENT

- `choose_strategy` : NOT PRESENT

- `generate_followup` : NOT PRESENT

- `find_related_entries` : NOT PRESENT

- `related_entry_ids` : NOT PRESENT

- `session_id` : NOT PRESENT (no top-level `session_id` function; session identifiers may exist in archived code patterns)

- `next_question` : NOT PRESENT

Archived / Related Implementations

- `app/legacy/cognition_layer_original.py`
  - `expand_query(query, model_name)` — generates booster queries (inputs: `query`, `model_name`; outputs: list[str]). This can serve `generate_followup` style capabilities.
  - `rerank_results(query, results, model_name)` — LLM-driven reranking (inputs: query, results, model_name; outputs: ordered results).
  - `interpret_sources(query, results, model_name)` — source interpretation (inputs: query and retrieved chunks; outputs: structured signals string).
  - `execute_cognition_pipeline(query, mode, scope, model_name)` — a monolithic pipeline that runs retrieval, rerank, interpretation and final reasoning.
  - Persistence: archived code expects `memory_manager` and `scope_processor` in the legacy environment.
  - Dependencies: ollama, memory_manager, scope_processor (legacy modules)
  - Safe to port: Parts are safe and already partially ported (retrieval, rerank, interpret were extracted). The legacy code uses different I/O and module boundaries; port carefully.

Recommendation

- Treat all listed symbols as NOT PRESENT for now. Use the extracted utilities (`retrieval_service`, `cognition_service`) and the archived `cognition_layer_original.py` as source material when implementing specific strategy functions (clarify, deepen, choose_strategy, generate_followup, etc.).

If you want, I can:
- Map each NOT PRESENT symbol to a concrete implementation plan (which legacy functions to port and tests to add).
- Start porting one strategy (e.g., `generate_followup`) using `cognition_layer_original.expand_query` as a starting point.

*** End of report
