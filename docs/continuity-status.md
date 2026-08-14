**Continuity Status**

This document records what continuity-related features exist in the current runtime and what is only metadata or planned.

Categories: IMPLEMENTED | PARTIAL | METADATA ONLY | PLANNED | NOT PRESENT

- theme evolution: NOT PRESENT (only mentioned in legacy notes and prompts)
- unresolved thread tracking: NOT PRESENT
- parent/child lineage: PARTIAL — Investigation model/repository supports parent relationships (see Investigation model and repository), but there is no general lineage graph implementation in runtime services.
- investigation lineage: PARTIAL — repository persists investigations and can store parent references; no graph traversal utilities exposed.
- belief revision: NOT PRESENT
- persistent session continuity: IMPLEMENTED (basic session history is saved/loaded via `app.services.memory_service.load_session_memory` and `save_session_memory`)
- graph schemas: NOT PRESENT (no formal graph schema implementation in runtime)

Files inspected:
- `app/repositories/investigation_repository.py` — supports basic investigation CRUD and parent field storage (PARTIAL)
- `app/services/memory_service.py` — session persistence helpers (IMPLEMENTED)
- `app/legacy/*` — contains conceptual notes and prompts referencing continuity but no runnable, integrated continuity layer (METADATA ONLY)

Notes:
- The current Investigation persistence supports parent references but lacks exploration/graph utilities for lineage.
- Many continuity concepts exist in legacy prose and prompt templates but are not yet implemented as services.
