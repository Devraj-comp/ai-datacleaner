"""Dataset ingestion: parse uploaded CSV/JSON bytes into a pandas DataFrame."""

from __future__ import annotations

import io
import json

import pandas as pd


class IngestError(ValueError):
    """Raised when an uploaded payload cannot be parsed."""


def _read_csv(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(content), sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the API
        raise IngestError(f"Could not parse CSV: {exc}") from exc


def _read_json(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_json(io.BytesIO(content))
    except Exception:
        pass
    try:
        payload = json.loads(content.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise IngestError(f"Could not parse JSON: {exc}") from exc

    if isinstance(payload, list):
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        return pd.json_normalize(payload)
    raise IngestError("JSON payload must be an object or array of objects")


def load_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    """Parse raw bytes into a DataFrame based on the file extension."""
    if not content:
        raise IngestError("Uploaded file is empty")

    lower = filename.lower()
    if lower.endswith(".csv") or lower.endswith(".tsv"):
        df = _read_csv(content)
    elif lower.endswith(".json"):
        df = _read_json(content)
    else:
        # Fall back to CSV parsing, which is the most common tabular upload.
        df = _read_csv(content)

    df.columns = [str(c) for c in df.columns]
    return df
