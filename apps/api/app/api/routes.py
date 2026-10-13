"""HTTP API routes."""

from __future__ import annotations

import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.models import (
    ApplyRequest,
    ApplyResponse,
    DatasetProfile,
    ExportRequest,
    ExportScriptResponse,
    PreviewResponse,
    SuggestRequest,
    SuggestResponse,
    UploadResponse,
)
from app.services import ingest, pipeline
from app.services.llm import LLMError, OllamaProvider, heuristic_suggestions
from app.services.profiler import profile_dataframe, write_ydata_report
from app.storage import store

router = APIRouter(prefix="/api")


def _get_record(dataset_id: str):
    record = store.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return record


@router.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "model": settings.ollama_model, "backend": "ollama"}


@router.post("/datasets", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> UploadResponse:
    settings = get_settings()
    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_upload_mb}MB limit",
        )
    try:
        df = ingest.load_dataframe(file.filename or "upload.csv", content)
    except ingest.IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    profile = profile_dataframe(df)
    profile.ydata_report = write_ydata_report(df, "pending")
    record = store.add(file.filename or "upload.csv", df, profile)
    preview = pipeline.build_preview(df, settings.preview_rows)
    return UploadResponse(
        dataset_id=record.id,
        filename=record.filename,
        profile=profile,
        preview=preview,
    )


@router.get("/datasets/{dataset_id}/profile", response_model=DatasetProfile)
def get_profile(dataset_id: str) -> DatasetProfile:
    return _get_record(dataset_id).profile


@router.get("/datasets/{dataset_id}/preview", response_model=PreviewResponse)
def get_preview(dataset_id: str) -> PreviewResponse:
    record = _get_record(dataset_id)
    return pipeline.build_preview(record.df, get_settings().preview_rows)


@router.post("/datasets/{dataset_id}/suggest", response_model=SuggestResponse)
async def suggest(dataset_id: str, body: SuggestRequest) -> SuggestResponse:
    record = _get_record(dataset_id)
    warning = None
    provider = "heuristic"

    if body.use_llm:
        try:
            client = OllamaProvider()
            suggestions = await client.suggest(record.profile)
            provider = f"ollama:{client.model}"
        except LLMError as exc:
            warning = f"LLM unavailable, using heuristics: {exc}"
            suggestions = heuristic_suggestions(record.df, record.profile)
        except Exception as exc:  # noqa: BLE001 - defensive, still fall back
            warning = f"LLM error, using heuristics: {exc}"
            suggestions = heuristic_suggestions(record.df, record.profile)
    else:
        suggestions = heuristic_suggestions(record.df, record.profile)

    if not suggestions and provider == "heuristic":
        warnings_note = "No cleaning operations suggested."
        warning = f"{warning} {warnings_note}" if warning else warnings_note

    suggestions = suggestions[: body.max_suggestions]
    record.suggestions = suggestions
    return SuggestResponse(
        dataset_id=dataset_id,
        provider=provider,
        suggestions=suggestions,
        warning=warning,
    )


@router.post("/datasets/{dataset_id}/apply", response_model=ApplyResponse)
def apply(dataset_id: str, body: ApplyRequest) -> ApplyResponse:
    record = _get_record(dataset_id)
    before = record.df
    try:
        cleaned = pipeline.apply_ops(before, body.ops)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ApplyResponse(
        dataset_id=dataset_id,
        preview=pipeline.build_preview(cleaned, get_settings().preview_rows),
        applied=len(body.ops),
        rows_before=int(before.shape[0]),
        rows_after=int(cleaned.shape[0]),
        columns_before=int(before.shape[1]),
        columns_after=int(cleaned.shape[1]),
    )


@router.post("/datasets/{dataset_id}/export/script", response_model=ExportScriptResponse)
def export_script(dataset_id: str, body: ExportRequest) -> ExportScriptResponse:
    _get_record(dataset_id)
    script = pipeline.generate_script(body.ops, body.read_source, body.write_target)
    return ExportScriptResponse(
        dataset_id=dataset_id,
        filename="clean.py",
        script=script,
    )


@router.post("/datasets/{dataset_id}/export/data")
def export_data(dataset_id: str, body: ExportRequest) -> StreamingResponse:
    record = _get_record(dataset_id)
    try:
        cleaned = pipeline.apply_ops(record.df, body.ops)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    buffer = io.StringIO()
    cleaned.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="cleaned.csv"'},
    )
