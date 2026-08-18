from pydantic import BaseModel, Field
from typing import List, Any


class Scope(BaseModel):
    namespaces: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    include_tags: List[str] = Field(default_factory=list)
    exclude_tags: List[str] = Field(default_factory=list)
    include_terms: List[str] = Field(default_factory=list)
    exclude_terms: List[str] = Field(default_factory=list)
