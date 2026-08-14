from fastapi import APIRouter
from pydantic import BaseModel
from app.services import scope_service

router = APIRouter(prefix="/api/scopes", tags=["Scopes"])


@router.get("/config")
def get_scope_config():
    return scope_service.ScopeProcessor.validate_scope_config({})


class ValidatePayload(BaseModel):
    doc_text: str
    metadata: dict
    scope: dict


@router.post("/validate")
def validate_scope(payload: ValidatePayload):
    ok, reason = scope_service.ScopeProcessor.matches_scope(payload.doc_text, payload.metadata, payload.scope)
    return {"ok": ok, "reason": reason}
