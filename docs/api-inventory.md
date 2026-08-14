**API Inventory**

- `GET /health` — [app/main.py](app/main.py#L1)
  - Implementation: `health_check()` in `app/main.py`
  - Status: RUNTIME CANONICAL
  - Purpose: basic health endpoint

- `POST /api/cognition/reason` — [app/api/inquiry.py](app/api/inquiry.py#L1)
  - Implementation: `cognition_reason()` -> `run_investigation()` -> `investigate()`
  - Status: RUNTIME CANONICAL
  - Purpose: Run an inquiry/cognition request with payload `InquiryRequest`

- `POST /api/investigate` — [app/api/inquiry.py](app/api/inquiry.py#L1)
  - Implementation: `investigate_alias()` -> `run_investigation()` -> `investigate()`
  - Status: RUNTIME CANONICAL
  - Purpose: Alias for cognition reasoning endpoint

- Investigation CRUD
  - `POST /api/investigations` — create investigation: [app/api/investigations.py](app/api/investigations.py#L1)
  - `GET /api/investigations` — list investigations
  - `GET /api/investigations/{id}` — get single
  - `PATCH /api/investigations/{id}` — update
  - `DELETE /api/investigations/{id}` — delete
  - Implementation: `app.repositories.investigation_repository.InvestigationRepository`
  - Status: RUNTIME CANONICAL
  - Purpose: persist and manage investigation records

- Legacy / Not currently exposed routes (present in `app/legacy/server_original.py`)
  - Several older endpoints used for interactive workflows and developer tools exist in legacy server code but are not included in the current FastAPI app: LEGACY / NOT CURRENTLY EXPOSED

- Memory & Scopes (new canonical routes)
  - `GET /api/memory/notes` — lists refined notes: [app/api/memory.py](app/api/memory.py#L1) (RUNTIME CANONICAL)
  - `GET /api/memory/notes/content` — read refined note content: [app/api/memory.py](app/api/memory.py#L1) (RUNTIME CANONICAL)
  - `POST /api/memory/notes/save` — save refined note content and index: [app/api/memory.py](app/api/memory.py#L1) (RUNTIME CANONICAL)
  - `GET /api/memory/sessions` — list saved session memory entries: [app/api/memory.py](app/api/memory.py#L1) (RUNTIME CANONICAL)
  - `GET /api/scopes/config` — returns default scope config structure: [app/api/scopes.py](app/api/scopes.py#L1) (RUNTIME CANONICAL)
  - `POST /api/scopes/validate` — validate a doc+metadata against a scope: [app/api/scopes.py](app/api/scopes.py#L1) (RUNTIME CANONICAL)

Notes:
- The active inquiry endpoints delegate directly to `app.services.legacy_inquiry_engine.investigate()` for orchestration. The legacy cognition server files provide historical context but are not wired into the current FastAPI app.
