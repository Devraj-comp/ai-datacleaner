"""Cleaning operation DSL.

Every cleaning action is a typed, serializable operation. Each op knows how to
(a) apply itself to a pandas DataFrame and (b) emit equivalent pandas code.
Because both paths derive from the same object, the generated script and the
in-process preview can never drift apart.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, TypeAdapter

DType = Literal["int", "float", "bool", "string", "datetime", "category"]

_TRUTHY = {"1", "true", "t", "yes", "y"}
_FALSY = {"0", "false", "f", "no", "n", ""}


def to_bool(value: Any) -> Any:
    """Best-effort boolean coercion used by both apply and codegen."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return pd.NA


def col(name: str) -> str:
    """Render a column reference as a safe Python expression."""
    return f"df[{json.dumps(str(name))}]"


class BaseOp(BaseModel):
    """Base class shared by every cleaning operation."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    def emit_pandas(self) -> str:  # pragma: no cover
        raise NotImplementedError

    def summary(self) -> str:  # pragma: no cover
        raise NotImplementedError


class CastTypeOp(BaseOp):
    op: Literal["cast_type"] = "cast_type"
    column: str
    dtype: DType

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = self.column
        if self.dtype == "int":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        elif self.dtype == "float":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float64")
        elif self.dtype == "bool":
            df[c] = df[c].map(to_bool).astype("boolean")
        elif self.dtype == "string":
            df[c] = df[c].astype("string")
        elif self.dtype == "datetime":
            df[c] = pd.to_datetime(df[c], errors="coerce")
        elif self.dtype == "category":
            df[c] = df[c].astype("category")
        return df

    def emit_pandas(self) -> str:
        target = col(self.column)
        if self.dtype == "int":
            return f'{target} = pd.to_numeric({target}, errors="coerce").astype("Int64")'
        if self.dtype == "float":
            return f'{target} = pd.to_numeric({target}, errors="coerce").astype("Float64")'
        if self.dtype == "bool":
            return f'{target} = {target}.map(_to_bool).astype("boolean")'
        if self.dtype == "string":
            return f'{target} = {target}.astype("string")'
        if self.dtype == "datetime":
            return f'{target} = pd.to_datetime({target}, errors="coerce")'
        return f'{target} = {target}.astype("category")'

    def summary(self) -> str:
        return f"Cast '{self.column}' to {self.dtype}"


class FillNullOp(BaseOp):
    op: Literal["fill_null"] = "fill_null"
    column: str
    strategy: Literal["mean", "median", "mode", "constant", "ffill", "bfill"] = "constant"
    value: Optional[Union[str, int, float, bool]] = None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = self.column
        if self.strategy == "mean":
            df[c] = df[c].fillna(df[c].mean())
        elif self.strategy == "median":
            df[c] = df[c].fillna(df[c].median())
        elif self.strategy == "mode":
            mode = df[c].mode()
            if not mode.empty:
                df[c] = df[c].fillna(mode.iat[0])
        elif self.strategy == "constant":
            df[c] = df[c].fillna(self.value)
        elif self.strategy == "ffill":
            df[c] = df[c].ffill()
        elif self.strategy == "bfill":
            df[c] = df[c].bfill()
        return df

    def emit_pandas(self) -> str:
        target = col(self.column)
        if self.strategy == "mean":
            return f"{target} = {target}.fillna({target}.mean())"
        if self.strategy == "median":
            return f"{target} = {target}.fillna({target}.median())"
        if self.strategy == "mode":
            return f"{target} = {target}.fillna({target}.mode().iat[0])"
        if self.strategy == "constant":
            return f"{target} = {target}.fillna({json.dumps(self.value)})"
        if self.strategy == "ffill":
            return f"{target} = {target}.ffill()"
        return f"{target} = {target}.bfill()"

    def summary(self) -> str:
        detail = self.strategy if self.strategy != "constant" else f"constant={self.value!r}"
        return f"Fill nulls in '{self.column}' ({detail})"


class DropDuplicatesOp(BaseOp):
    op: Literal["drop_duplicates"] = "drop_duplicates"
    subset: Optional[list[str]] = None
    keep: Literal["first", "last"] = "first"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=self.subset, keep=self.keep)

    def emit_pandas(self) -> str:
        subset = json.dumps(self.subset) if self.subset else "None"
        return f'df = df.drop_duplicates(subset={subset}, keep="{self.keep}")'

    def summary(self) -> str:
        scope = ", ".join(self.subset) if self.subset else "all columns"
        return f"Drop duplicate rows ({scope})"


class ClipOutliersOp(BaseOp):
    op: Literal["clip_outliers"] = "clip_outliers"
    column: str
    method: Literal["iqr", "zscore"] = "iqr"
    factor: float = 1.5
    threshold: float = 3.0

    def _bounds(self, series: pd.Series) -> tuple[float, float]:
        if self.method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            return q1 - self.factor * iqr, q3 + self.factor * iqr
        mean, std = series.mean(), series.std()
        return mean - self.threshold * std, mean + self.threshold * std

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = self.column
        lower, upper = self._bounds(df[c])
        df[c] = df[c].clip(lower=lower, upper=upper)
        return df

    def emit_pandas(self) -> str:
        target = col(self.column)
        if self.method == "iqr":
            return (
                f"_q1 = {target}.quantile(0.25); _q3 = {target}.quantile(0.75); "
                f"_iqr = _q3 - _q1\n"
                f"{target} = {target}.clip(lower=_q1 - {self.factor} * _iqr, "
                f"upper=_q3 + {self.factor} * _iqr)"
            )
        return (
            f"_mean = {target}.mean(); _std = {target}.std()\n"
            f"{target} = {target}.clip(lower=_mean - {self.threshold} * _std, "
            f"upper=_mean + {self.threshold} * _std)"
        )

    def summary(self) -> str:
        return f"Clip outliers in '{self.column}' using {self.method}"


class TrimWhitespaceOp(BaseOp):
    op: Literal["trim_whitespace"] = "trim_whitespace"
    columns: list[str]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.columns:
            df[c] = df[c].astype("string").str.strip()
        return df

    def emit_pandas(self) -> str:
        return "\n".join(
            f'{col(c)} = {col(c)}.astype("string").str.strip()' for c in self.columns
        )

    def summary(self) -> str:
        return f"Trim whitespace in {', '.join(self.columns)}"


class NormalizeCaseOp(BaseOp):
    op: Literal["normalize_case"] = "normalize_case"
    columns: list[str]
    case: Literal["lower", "upper", "title"] = "lower"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.columns:
            s = df[c].astype("string")
            df[c] = getattr(s.str, self.case)()
        return df

    def emit_pandas(self) -> str:
        return "\n".join(
            f'{col(c)} = {col(c)}.astype("string").str.{self.case}()' for c in self.columns
        )

    def summary(self) -> str:
        return f"Normalize case ({self.case}) in {', '.join(self.columns)}"


class StandardizeCategoriesOp(BaseOp):
    op: Literal["standardize_categories"] = "standardize_categories"
    column: str
    mapping: dict[str, str]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.column] = df[self.column].astype("string").replace(self.mapping)
        return df

    def emit_pandas(self) -> str:
        mapping = json.dumps(self.mapping)
        return f'{col(self.column)} = {col(self.column)}.astype("string").replace({mapping})'

    def summary(self) -> str:
        return f"Standardize {len(self.mapping)} categories in '{self.column}'"


class ParseDateOp(BaseOp):
    op: Literal["parse_date"] = "parse_date"
    column: str
    format: Optional[str] = None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.column] = pd.to_datetime(df[self.column], format=self.format, errors="coerce")
        return df

    def emit_pandas(self) -> str:
        fmt = json.dumps(self.format) if self.format else "None"
        return f'{col(self.column)} = pd.to_datetime({col(self.column)}, format={fmt}, errors="coerce")'

    def summary(self) -> str:
        return f"Parse '{self.column}' as datetime"


class DropColumnOp(BaseOp):
    op: Literal["drop_column"] = "drop_column"
    column: str

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[self.column])

    def emit_pandas(self) -> str:
        return f"df = df.drop(columns={json.dumps([self.column])})"

    def summary(self) -> str:
        return f"Drop column '{self.column}'"


class RenameColumnOp(BaseOp):
    op: Literal["rename_column"] = "rename_column"
    column: str
    new_name: str

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={self.column: self.new_name})

    def emit_pandas(self) -> str:
        mapping = json.dumps({self.column: self.new_name})
        return f"df = df.rename(columns={mapping})"

    def summary(self) -> str:
        return f"Rename '{self.column}' to '{self.new_name}'"


class ReplaceValuesOp(BaseOp):
    op: Literal["replace_values"] = "replace_values"
    column: str
    mapping: dict[str, Any]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.column] = df[self.column].replace(self.mapping)
        return df

    def emit_pandas(self) -> str:
        return f"{col(self.column)} = {col(self.column)}.replace({json.dumps(self.mapping)})"

    def summary(self) -> str:
        return f"Replace {len(self.mapping)} values in '{self.column}'"


class ValidateRegexOp(BaseOp):
    op: Literal["validate_regex"] = "validate_regex"
    column: str
    pattern: str

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        match = df[self.column].astype("string").str.fullmatch(self.pattern)
        df.loc[~match.fillna(False), self.column] = pd.NA
        return df

    def emit_pandas(self) -> str:
        target = col(self.column)
        pattern = json.dumps(self.pattern)
        return (
            f"_match = {target}.astype(\"string\").str.fullmatch({pattern})\n"
            f'{target}.loc[~_match.fillna(False)] = pd.NA'
        )

    def summary(self) -> str:
        return f"Null values in '{self.column}' that fail /{self.pattern}/"


CleaningOp = Annotated[
    Union[
        CastTypeOp,
        FillNullOp,
        DropDuplicatesOp,
        ClipOutliersOp,
        TrimWhitespaceOp,
        NormalizeCaseOp,
        StandardizeCategoriesOp,
        ParseDateOp,
        DropColumnOp,
        RenameColumnOp,
        ReplaceValuesOp,
        ValidateRegexOp,
    ],
    Field(discriminator="op"),
]

OpAdapter: TypeAdapter[CleaningOp] = TypeAdapter(CleaningOp)

OP_REGISTRY: dict[str, type[BaseOp]] = {
    "cast_type": CastTypeOp,
    "fill_null": FillNullOp,
    "drop_duplicates": DropDuplicatesOp,
    "clip_outliers": ClipOutliersOp,
    "trim_whitespace": TrimWhitespaceOp,
    "normalize_case": NormalizeCaseOp,
    "standardize_categories": StandardizeCategoriesOp,
    "parse_date": ParseDateOp,
    "drop_column": DropColumnOp,
    "rename_column": RenameColumnOp,
    "replace_values": ReplaceValuesOp,
    "validate_regex": ValidateRegexOp,
}
