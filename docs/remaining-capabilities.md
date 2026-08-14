**Remaining Legacy Capabilities Inventory**

Scanned: `app/legacy/` and migrated services.

- memory note listing
  - `app.services.memory_service.get_refined_notes_list` — WORKING AND EXPOSED
  - `app/legacy/memory_manager_original.py` — COMPATIBILITY ONLY

- memory note reading
  - `app.services.memory_service.read_note_content` — WORKING AND EXPOSED
  - legacy: `app/legacy/memory_manager_original.py` — COMPATIBILITY ONLY

- memory note saving
  - `app.services.memory_service.save_note_content` / `index_refined_note` — WORKING AND EXPOSED

- session listing
  - `app.services.memory_service.load_session_memory` / `append_session_entry` — WORKING AND EXPOSED

- scope configuration
  - `app.services.scope_service.ScopeProcessor.validate_scope_config` — WORKING AND EXPOSED
  - `app/legacy/scope_processor_original.py` — COMPATIBILITY ONLY

- scope validation
  - `app.services.scope_service.ScopeProcessor.matches_scope` — WORKING AND EXPOSED

- question strategies / clarify / compare / action / challenge / deepen strategies
 - question strategies / clarify / compare / action / challenge / deepen strategies
  - These higher-level strategies are referenced only in prompt text and legacy documentation; there are no concrete strategy functions named `clarify`, `compare`, `action`, `challenge`, or `deepen` implemented as callable runtime code in the current repository. Classification: NOT PRESENT (only described in prompts / legacy prose)

- related-entry lookup
 - related-entry lookup
  - Only mentioned in legacy materials and prompts; no runnable helper named `related_entry` or equivalent is exposed in current runtime: NOT PRESENT

- inquiry lineage / unresolved-thread tracking / theme evolution / continuity scaffolding / graph-related data structures
  - Appears in legacy materials and notes, e.g. `legacy_inquiry_engine_before_api.py`, but not fully wired into the canonical runtime: PARTIALLY IMPLEMENTED or PLANNED ONLY depending on file

Summary recommendations (non-destructive):
- The legacy folder contains useful capabilities (clarify/compare strategies, lineage tracking) that are WORKING BUT NOT EXPOSED; consider exposing them incrementally after parity tests.
- Do not remove these files; they act as a backlog of capabilities to selectively re-expose.
