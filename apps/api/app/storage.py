"""In-memory dataset store (single-user, single-process).

Datasets live for the lifetime of the process. This keeps the first version
dependency-free; swap for Redis/SQLite when persistence or multi-user support
is needed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pandas as pd

from app.models import DatasetProfile, Suggestion


@dataclass
class DatasetRecord:
    id: str
    filename: str
    df: pd.DataFrame
    profile: DatasetProfile
    suggestions: list[Suggestion] = field(default_factory=list)


class DatasetStore:
    def __init__(self) -> None:
        self._items: dict[str, DatasetRecord] = {}

    def add(self, filename: str, df: pd.DataFrame, profile: DatasetProfile) -> DatasetRecord:
        dataset_id = uuid.uuid4().hex[:12]
        record = DatasetRecord(id=dataset_id, filename=filename, df=df, profile=profile)
        self._items[dataset_id] = record
        return record

    def get(self, dataset_id: str) -> DatasetRecord | None:
        return self._items.get(dataset_id)

    def list_ids(self) -> list[str]:
        return list(self._items.keys())


store = DatasetStore()
