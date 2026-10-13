from __future__ import annotations

import pandas as pd
import pytest

from app.core.ops import (
    CastTypeOp,
    ClipOutliersOp,
    DropDuplicatesOp,
    FillNullOp,
    ParseDateOp,
    StandardizeCategoriesOp,
    TrimWhitespaceOp,
    ValidateRegexOp,
)
from app.services.pipeline import apply_ops, generate_script


@pytest.fixture
def messy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [" Alice ", "bob", "bob", "carol"],
            "age": ["25", "30", "30", "200"],
            "score": [88.5, None, None, 72.0],
            "city": ["NYC", "NYC", "NYC", "nyc"],
            "signup": ["2021-01-02", "2021-03-04", "2021-03-04", "not-a-date"],
        }
    )


def test_trim_whitespace() -> None:
    df = pd.DataFrame({"a": [" x ", "y"]})
    out = TrimWhitespaceOp(columns=["a"]).apply(df)
    assert out["a"].tolist() == ["x", "y"]


def test_cast_type_numeric() -> None:
    df = pd.DataFrame({"a": ["1", "2", None]})
    out = CastTypeOp(column="a", dtype="int").apply(df)
    assert str(out["a"].dtype) == "Int64"
    assert out["a"].tolist()[:2] == [1, 2]


def test_fill_null_median() -> None:
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    out = FillNullOp(column="a", strategy="median").apply(df)
    assert out["a"].tolist() == [1.0, 2.0, 3.0]


def test_fill_null_constant() -> None:
    df = pd.DataFrame({"a": ["x", None]})
    out = FillNullOp(column="a", strategy="constant", value="unknown").apply(df)
    assert out["a"].tolist() == ["x", "unknown"]


def test_drop_duplicates() -> None:
    df = pd.DataFrame({"a": [1, 1, 2]})
    out = DropDuplicatesOp().apply(df)
    assert len(out) == 2


def test_clip_outliers_iqr() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 1000.0]})
    out = ClipOutliersOp(column="a", method="iqr").apply(df)
    assert out["a"].max() < 1000


def test_parse_date() -> None:
    df = pd.DataFrame({"d": ["2021-01-02", "bad"]})
    out = ParseDateOp(column="d").apply(df)
    assert str(out["d"].dtype).startswith("datetime")
    assert pd.isna(out["d"].iloc[1])


def test_standardize_categories() -> None:
    df = pd.DataFrame({"c": ["NYC", "nyc", "NyC"]})
    out = StandardizeCategoriesOp(column="c", mapping={"nyc": "NYC"}).apply(df)
    assert out["c"].tolist() == ["NYC", "NYC", "NyC"]


def test_validate_regex_nulls_invalid() -> None:
    df = pd.DataFrame({"code": ["AB12", "bad!", "CD34"]})
    out = ValidateRegexOp(column="code", pattern=r"[A-Z]{2}\d{2}").apply(df)
    assert pd.isna(out["code"].iloc[1])
    assert out["code"].iloc[0] == "AB12"


def test_generated_script_matches_in_process_apply(messy_df: pd.DataFrame) -> None:
    ops = [
        TrimWhitespaceOp(columns=["name"]),
        CastTypeOp(column="age", dtype="int"),
        FillNullOp(column="score", strategy="median"),
        StandardizeCategoriesOp(column="city", mapping={"nyc": "NYC"}),
        DropDuplicatesOp(),
    ]
    expected = apply_ops(messy_df, ops)

    script = generate_script(ops, read_source="in.csv", write_target="out.csv")
    namespace: dict = {}
    exec(compile(script, "clean.py", "exec"), namespace)
    produced = namespace["clean"](messy_df.copy())

    pd.testing.assert_frame_equal(produced.reset_index(drop=True), expected.reset_index(drop=True))


def test_generated_script_is_valid_python(messy_df: pd.DataFrame) -> None:
    ops = [
        CastTypeOp(column="age", dtype="float"),
        ClipOutliersOp(column="score", method="zscore"),
        ParseDateOp(column="signup"),
        TrimWhitespaceOp(columns=["name", "city"]),
    ]
    script = generate_script(ops)
    compile(script, "clean.py", "exec")
    assert "def clean(df: pd.DataFrame)" in script
