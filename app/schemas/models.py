from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")


class QueryResponse(BaseModel):
    answer: str
    context: list[str] = Field(default_factory=list, description="Retrieved chunks")
    steps: list[str] = Field(default_factory=list, description="Agent steps (debug)")

