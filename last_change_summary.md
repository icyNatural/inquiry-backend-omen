# Last Change Summary

## Commit
- **Message:** Canonicalize inquiry memory llm and modes
- **Commit:** `b02d395`

## What changed
This change focuses on normalizing the inquiry flow around memory, LLM behavior, and mode handling. The update introduces a more consistent structure across the inquiry and investigation APIs, aligns the inquiry model and repository logic, and expands supporting services and documentation.

### Key updates
- Updated inquiry and investigation API layers to support the canonical request/response flow.
- Added and refined inquiry model data structures for memory and mode-driven behavior.
- Extended the investigation repository and related service layer to support the new runtime flow.
- Refreshed the legacy inquiry engine and cognition service to align with the new canonical model.
- Added/updated supporting docs and tests covering memory scope behavior, mode grounding, and runtime validation.

### Files touched
- `app/api/inquiry.py`
- `app/api/investigations.py`
- `app/models/inquiry.py`
- `app/repositories/investigation_repository.py`
- `app/services/cognition_service.py`
- `app/services/legacy_inquiry_engine.py`
- `app/services/scope_service.py`
- `docs/question-strategy-status.md`
- `tests/test_api_memory_scopes.py`
- `tests/test_canonical_runtime.py`
- `tests/test_modes_grounding_mapping.py`
- `tests/test_modes_validation.py`
- `tests/test_pipeline_and_policy.py`

## Overall intent
The project appears to be consolidating its runtime behavior around a canonical inquiry model so that memory handling, LLM prompts, and mode logic work together more consistently across the application.
