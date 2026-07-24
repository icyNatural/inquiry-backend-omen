from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InvestigationCreate(BaseModel):
    original_question: str = Field(min_length=1)
    objective: str | None = None
    workspace_id: str = "default"
    brief: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0, le=1)
    parent_investigation_id: str | None = None


class InvestigationUpdate(BaseModel):
    original_question: str | None = Field(default=None, min_length=1)
    objective: str | None = None
    brief: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    parent_investigation_id: str | None = None
    version: int | None = Field(default=None, ge=1)


class Investigation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str = "default"
    original_question: str
    objective: str | None = None
    brief: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    parent_investigation_id: str | None = None
    version: int = 1
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)