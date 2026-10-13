"""LLM provider (Ollama) plus a deterministic heuristic fallback.

The model never writes executable code directly. It emits *cleaning operations*
that are validated against the op DSL and reviewed by a human before use.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pandas as pd
from pydantic import ValidationError

from app.config import get_settings
from app.core.ops import OP_REGISTRY, OpAdapter
from app.models import DatasetProfile, Suggestion
from app.services.profiler import profile_to_prompt


class LLMError(RuntimeError):
    """Raised when the LLM backend cannot be reached or returns junk."""


def ops_catalog() -> str:
    """Render the available ops and their parameters from the Pydantic models."""
    lines: list[str] = []
    for name, model in OP_REGISTRY.items():
        schema = model.model_json_schema()
        props = schema.get("properties", {})
        params = []
        for field, spec in props.items():
            if field == "op":
                continue
            type_hint = spec.get("type") or spec.get("anyOf") or spec.get("$ref", "any")
            if isinstance(type_hint, list):
                type_hint = "/".join(
                    t.get("type", "any") if isinstance(t, dict) else str(t) for t in type_hint
                )
            default = spec.get("default")
            suffix = f" (default: {default})" if default is not None else ""
            params.append(f"{field}: {type_hint}{suffix}")
        lines.append(f"- {name}: {{{', '.join(params)}}}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are a meticulous data-cleaning assistant.
Given a statistical profile of a tabular dataset, propose cleaning operations.

Available operations (emit ONLY these, with exact field names):
{ops_catalog()}

Rules:
- Only reference columns that exist in the profile.
- Return JSON of the form:
  {{"suggestions": [{{"op": "<name>", "params": {{...}}, "rationale": "<why>", "confidence": 0.0-1.0}}]}}
- "params" must contain the op's fields EXCEPT the "op" field.
- Base each suggestion on concrete evidence from the profile (missing %, dtype, ranges, samples).
- Prefer the smallest safe change. Do not invent values that are not implied by the data.
- If nothing needs cleaning, return {{"suggestions": []}}.
- Output JSON only, no markdown, no commentary.
"""


class OllamaProvider:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = settings.ollama_timeout
        self.temperature = settings.ollama_temperature

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        content = data.get("message", {}).get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model did not return valid JSON: {content[:200]}") from exc

    async def suggest(self, profile: DatasetProfile) -> list[Suggestion]:
        user = (
            "Dataset profile:\n"
            f"{profile_to_prompt(profile)}\n\n"
            "Propose cleaning operations as JSON."
        )
        payload = await self.complete_json(SYSTEM_PROMPT, user)
        raw = payload.get("suggestions", [])
        return parse_suggestions(raw, source="llm")


def parse_suggestions(raw: list[Any], source: str) -> list[Suggestion]:
    """Validate raw model output into typed suggestions, dropping invalid ones."""
    suggestions: list[Suggestion] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or "op" not in item:
            continue
        params = item.get("params", {}) or {}
        candidate = {"op": item["op"], **params}
        try:
            op = OpAdapter.validate_python(candidate)
        except ValidationError:
            continue
        confidence = item.get("confidence", 0.5)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.5
        suggestions.append(
            Suggestion(
                id=f"{source}-{index}",
                op=op,
                rationale=str(item.get("rationale", "")),
                confidence=confidence,
                source=source,  # type: ignore[arg-type]
            )
        )
    return suggestions


def heuristic_suggestions(df: pd.DataFrame, profile: DatasetProfile) -> list[Suggestion]:
    """Deterministic suggestions derived directly from the profile.

    Used when the LLM is unavailable or disabled, so the tool always works.
    """
    out: list[dict[str, Any]] = []
    rows = max(profile.rows, 1)

    if profile.duplicate_rows > 0:
        out.append(
            {
                "op": "drop_duplicates",
                "params": {"keep": "first"},
                "rationale": f"{profile.duplicate_rows} fully duplicated rows detected",
                "confidence": 0.9,
            }
        )

    for c in profile.columns_meta:
        name = c.name
        is_numeric = any(t in c.dtype for t in ("int", "float", "Int", "Float"))
        is_object = _is_text_dtype(c.dtype)

        if c.missing_pct >= 60:
            out.append(
                {
                    "op": "drop_column",
                    "params": {"column": name},
                    "rationale": f"'{name}' is {c.missing_pct}% empty",
                    "confidence": 0.5,
                }
            )
        elif c.missing_pct > 0:
            if is_numeric:
                strategy = "median"
            else:
                strategy = "mode"
            out.append(
                {
                    "op": "fill_null",
                    "params": {"column": name, "strategy": strategy},
                    "rationale": f"'{name}' has {c.missing_pct}% missing values",
                    "confidence": round(min(0.9, 0.5 + c.missing_pct / 100), 2),
                }
            )

        if is_object and _looks_numeric(c):
            out.append(
                {
                    "op": "cast_type",
                    "params": {"column": name, "dtype": "float"},
                    "rationale": f"'{name}' stores numeric-looking values as text",
                    "confidence": 0.7,
                }
            )

        if is_object and _name_suggests_date(name):
            out.append(
                {
                    "op": "parse_date",
                    "params": {"column": name},
                    "rationale": f"'{name}' looks like a date/time field",
                    "confidence": 0.6,
                }
            )

        if is_object and 1 < c.unique <= 20 and c.unique / rows < 0.5 and not _name_suggests_date(name):
            out.append(
                {
                    "op": "cast_type",
                    "params": {"column": name, "dtype": "category"},
                    "rationale": f"'{name}' has only {c.unique} distinct values",
                    "confidence": 0.55,
                }
            )

        if is_numeric and c.stats:
            mean, std = c.stats.get("mean"), c.stats.get("std")
            lo, hi = c.stats.get("min"), c.stats.get("max")
            if (
                isinstance(mean, (int, float))
                and isinstance(std, (int, float))
                and std
                and isinstance(hi, (int, float))
                and hi > mean + 3 * std
            ):
                out.append(
                    {
                        "op": "clip_outliers",
                        "params": {"column": name, "method": "iqr"},
                        "rationale": f"'{name}' max is far above its mean ({hi} vs {mean:.2f})",
                        "confidence": 0.5,
                    }
                )

    if _has_whitespace(df):
        cols = _has_whitespace(df, return_columns=True)
        out.append(
            {
                "op": "trim_whitespace",
                "params": {"columns": cols},
                "rationale": "Values with leading/trailing whitespace detected",
                "confidence": 0.8,
            }
        )

    suggestions: list[Suggestion] = []
    for index, item in enumerate(out):
        try:
            op = OpAdapter.validate_python({"op": item["op"], **item["params"]})
        except ValidationError:
            continue
        suggestions.append(
            Suggestion(
                id=f"heuristic-{index}",
                op=op,
                rationale=item["rationale"],
                confidence=float(item.get("confidence", 0.5)),
                source="heuristic",
            )
        )
    return suggestions


def _is_text_dtype(dtype: str) -> bool:
    """True for object/str/string dtypes (pandas 3.x uses 'str' for text)."""
    lowered = str(dtype).lower()
    return lowered in {"object", "str", "string"} or "object" in lowered


def _looks_numeric(c) -> bool:
    for value in c.sample:
        if value is None:
            continue
        try:
            float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return False
    return bool(c.sample)


def _name_suggests_date(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("date", "time", "timestamp", "created", "updated"))


def _has_whitespace(df: pd.DataFrame, return_columns: bool = False):
    cols: list[str] = []
    for c in df.columns:
        if _is_text_dtype(str(df[c].dtype)):
            try:
                stripped = df[c].dropna().astype(str).str.strip()
                original = df[c].dropna().astype(str)
                if not stripped.equals(original):
                    cols.append(str(c))
            except Exception:  # noqa: BLE001
                continue
    if return_columns:
        return cols
    return bool(cols)
