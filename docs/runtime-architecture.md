**Runtime Architecture**

- **API entry**: [app/main.py](app/main.py#L1) — FastAPI application and router inclusion. Runtime: RUNTIME CANONICAL
- **Inquiry API routes**: [app/api/inquiry.py](app/api/inquiry.py#L1) — `cognition_reason` and `investigate` endpoints forwarding to `investigate()`. Runtime: RUNTIME CANONICAL
- **Investigation CRUD routes**: [app/api/investigations.py](app/api/investigations.py#L1) — create/list/get/update/delete. Runtime: RUNTIME CANONICAL
- **Inquiry orchestration**: `app.services.legacy_inquiry_engine.investigate()` ([app/services/legacy_inquiry_engine.py](app/services/legacy_inquiry_engine.py#L1078)) — main runtime orchestration pipeline. Runtime: RUNTIME CANONICAL
- **Mode definitions**:
  - `app.services.mode_prompts.MODE_PROMPTS` ([app/services/mode_prompts.py](app/services/mode_prompts.py#L1)) — centralized mode prompt texts used by the runtime orchestration. Runtime: RUNTIME CANONICAL
  - `app.services.cognition_service.MODE_REASONING_PROMPTS` ([app/services/cognition_service.py](app/services/cognition_service.py#L1)) — similar reasoning prompts in the cognition module. Runtime: DUPLICATED BUT UNUSED by the legacy orchestration
- **Query expansion**:
  - `app.services.legacy_inquiry_engine.generate_search_queries` ([app/services/legacy_inquiry_engine.py](app/services/legacy_inquiry_engine.py#L...)) — lightweight expansion used by legacy pipeline. Runtime: RUNTIME CANONICAL
  - `app.services.cognition_service.CognitionLayer.expand_query` ([app/services/cognition_service.py](app/services/cognition_service.py#L1)) — separate expansion routine in the cognition layer. Runtime: DUPLICATED BUT UNUSED
- **Chroma collection access**: `app.services.memory_service.get_collection()` ([app/services/memory_service.py](app/services/memory_service.py#L1)). Runtime: RUNTIME CANONICAL
- **Memory retrieval**:
  - Legacy: `legacy_inquiry_engine.merge_multi_query_results` and related functions in the same module. Runtime: RUNTIME CANONICAL
  - Cognition layer: `cognition_layer_original.execute_cognition_pipeline` (app/legacy/cognition_layer_original.py) — Legacy implementation kept for compatibility. Runtime: COMPATIBILITY ONLY
- **Scope filtering**: `app.services.scope_service.ScopeProcessor` ([app/services/scope_service.py](app/services/scope_service.py#L1)). Runtime: RUNTIME CANONICAL
- **Diversification**: `legacy_inquiry_engine.diversify_results` (module-local). Runtime: RUNTIME CANONICAL
- **Reranking**:
  - Legacy: `legacy_inquiry_engine.rerank_results` (module-local) — soft LLM reranker used by the legacy pipeline. Runtime: RUNTIME CANONICAL
  - Cognition: `cognition_service.CognitionLayer.rerank_results` — similar implementation exists in the cognition module. Runtime: DUPLICATED BUT UNUSED
- **Source interpretation**:
  - `legacy_inquiry_engine.interpret_sources` (module-local) and `cognition_service.CognitionLayer.interpret_sources` both present. Runtime: RUNTIME CANONICAL (legacy pipeline uses module-local interpreter)
- **Ollama calls**:
  - `app.services.llm_service.chat` ([app/services/llm_service.py](app/services/llm_service.py#L1)) — centralized LLM call wrapper; `legacy_inquiry_engine` now invokes this service. Runtime: RUNTIME CANONICAL
  - `cognition_service.CognitionLayer.call_ollama` provides a similar fallback wrapper. Runtime: DUPLICATED BUT UNUSED
- **Knowledge policy & model fallback**: implemented in `legacy_inquiry_engine.investigate()` (memory_only vs memory_plus_model). Runtime: RUNTIME CANONICAL
- **Investigation persistence**: `app.repositories.investigation_repository` and `app/api/investigations.py`. Runtime: RUNTIME CANONICAL
-- **Session/history state**: `app.services.memory_service.load_session_memory` is the canonical session storage/loader. The `history` global in `legacy_inquiry_engine.py` is present but dormant (not used by the current FastAPI request path). Runtime: RUNTIME CANONICAL for `memory_service`; `history` in `legacy_inquiry_engine.py` is DORMANT

Notes:
- Many cognition functions exist in both `app/services/cognition_service.py` and `app/services/legacy_inquiry_engine.py` (expansion, rerank, interpret). This pass preserves both implementations and treats the `legacy_inquiry_engine` pipeline as the active runtime orchestration.
