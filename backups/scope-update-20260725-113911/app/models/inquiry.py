from typing import Any

from pydantic import BaseModel, Field


class InquiryRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str = "recall"


class InquirySource(BaseModel):
    title: str
    conversation_index: Any | None = None
    chunk_index: Any | None = None
    source: Any | None = None
    domain: Any | None = None
    distance: float | None = None
    retrieved_via: str | None = None
    signal: str
    raw_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class InquiryResponse(BaseModel):
    status: str
    query: str
    mode: str
    model: str
    answer: str
    organized_signals: str
    search_queries: list[str]
    retrieved_count: int
    selected_count: int
    sources: list[InquirySource]
