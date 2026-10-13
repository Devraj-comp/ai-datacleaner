"""Profiling service.

Produces a compact, LLM-friendly statistical profile of a DataFrame. When the
optional ``ydata-profiling`` package is installed and enabled, a full HTML
report is generated alongside the lightweight JSON profile.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.config import get_settings
from app.models import ColumnProfile, DatasetProfile

_MAX_SAMPLE = 8


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if pd.isna(value):
        return None
    return str(value)


def profile_column(series: pd.Series) -> ColumnProfile:
    total = len(series)
    missing = int(series.isna().sum())
    sample = [_jsonable(v) for v in series.dropna().head(_MAX_SAMPLE).tolist()]
    stats: dict[str, Any] = {}

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        desc = series.describe()
        stats = {
            "min": _jsonable(desc.get("min")),
            "max": _jsonable(desc.get("max")),
            "mean": _jsonable(desc.get("mean")),
            "std": _jsonable(desc.get("std")),
        }
    else:
        counts = series.dropna().astype("string").value_counts().head(5)
        stats = {"top_values": {str(k): int(v) for k, v in counts.items()}}

    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        missing=missing,
        missing_pct=round((missing / total * 100) if total else 0.0, 2),
        unique=int(series.nunique(dropna=True)),
        sample=sample,
        stats=stats,
    )


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    return DatasetProfile(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        duplicate_rows=int(df.duplicated().sum()),
        memory_bytes=int(df.memory_usage(deep=True).sum()),
        columns_meta=[profile_column(df[c]) for c in df.columns],
    )


def write_ydata_report(df: pd.DataFrame, dataset_id: str) -> Optional[str]:
    """Generate a ydata-profiling HTML report if the package is available."""
    settings = get_settings()
    if not settings.enable_ydata_profile:
        return None
    try:
        from ydata_profiling import ProfileReport  # type: ignore
    except ImportError:
        return None

    os.makedirs(settings.storage_dir, exist_ok=True)
    path = os.path.join(settings.storage_dir, f"{dataset_id}_profile.html")
    try:
        ProfileReport(df, title=f"Profile {dataset_id}", minimal=True).to_file(path)
    except Exception:  # noqa: BLE001 - profiling is best-effort
        return None
    return path


def profile_to_prompt(profile: DatasetProfile) -> str:
    """Render the profile as compact text for the LLM prompt."""
    lines = [
        f"rows={profile.rows}, columns={profile.columns}, duplicate_rows={profile.duplicate_rows}"
    ]
    for c in profile.columns_meta:
        stats = json.dumps(c.stats, default=str)
        sample = json.dumps(c.sample, default=str)
        lines.append(
            f"- {c.name}: dtype={c.dtype}, missing={c.missing} ({c.missing_pct}%), "
            f"unique={c.unique}, stats={stats}, sample={sample}"
        )
    return "\n".join(lines)
