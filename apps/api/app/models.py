"""Pydantic models for API requests, responses, profiles and suggestions."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.core.ops import CleaningOp


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing: int
    missing_pct: float
    unique: int
    sample: list[Any] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class DatasetProfile(BaseModel):
    rows: int
    columns: int
    duplicate_rows: int
    memory_bytes: int
    columns_meta: list[ColumnProfile] = Field(default_factory=list)
    ydata_report: Optional[str] = None


class Suggestion(BaseModel):
    id: str
    op: CleaningOp
    rationale: str = ""
    confidence: float = 0.5
    evidence: dict[str, Any] = Field(default_factory=dict)
    source: Literal["llm", "heuristic", "manual"] = "heuristic"
    status: Literal["pending", "approved", "rejected"] = "pending"


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    profile: DatasetProfile
    preview: "PreviewResponse"


class PreviewResponse(BaseModel):
    columns: list[str]
    dtypes: dict[str, str]
    rows: list[dict[str, Any]]
    total_rows: int


class SuggestRequest(BaseModel):
    use_llm: bool = True
    max_suggestions: int = 20


class SuggestResponse(BaseModel):
    dataset_id: str
    provider: str
    suggestions: list[Suggestion]
    warning: Optional[str] = None


class ApplyRequest(BaseModel):
    ops: list[CleaningOp]


class ApplyResponse(BaseModel):
    dataset_id: str
    preview: PreviewResponse
    applied: int
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int


class ExportRequest(BaseModel):
    ops: list[CleaningOp]
    read_source: str = "input.csv"
    write_target: str = "cleaned.csv"


class ExportScriptResponse(BaseModel):
    dataset_id: str
    filename: str
    script: str


UploadResponse.model_rebuild()
