from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services.llm import heuristic_suggestions, parse_suggestions
from app.services.profiler import profile_dataframe

client = TestClient(app)

CSV = """name,age,score,city
 Alice ,25,88.5,NYC
bob,30,,NYC
bob,30,,NYC
carol,200,72.0,nyc
"""


def _upload() -> str:
    response = client.post(
        "/api/datasets",
        files={"file": ("data.csv", CSV, "text/csv")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["profile"]["rows"] == 4
    assert body["profile"]["duplicate_rows"] == 1
    return body["dataset_id"]


def test_health() -> None:
    assert client.get("/api/health").json()["status"] == "ok"


def test_upload_and_profile() -> None:
    dataset_id = _upload()
    profile = client.get(f"/api/datasets/{dataset_id}/profile").json()
    names = {c["name"] for c in profile["columns_meta"]}
    assert names == {"name", "age", "score", "city"}


def test_suggest_heuristics_and_apply() -> None:
    dataset_id = _upload()
    suggest = client.post(
        f"/api/datasets/{dataset_id}/suggest", json={"use_llm": False}
    ).json()
    assert suggest["provider"] == "heuristic"
    assert len(suggest["suggestions"]) > 0

    ops = [s["op"] for s in suggest["suggestions"]]
    apply = client.post(f"/api/datasets/{dataset_id}/apply", json={"ops": ops})
    assert apply.status_code == 200, apply.text
    body = apply.json()
    assert body["rows_after"] <= body["rows_before"]
    assert body["applied"] == len(ops)


def test_export_script() -> None:
    dataset_id = _upload()
    ops = [
        {"op": "trim_whitespace", "columns": ["name"]},
        {"op": "drop_duplicates"},
    ]
    response = client.post(
        f"/api/datasets/{dataset_id}/export/script",
        json={"ops": ops, "read_source": "data.csv", "write_target": "clean.csv"},
    )
    assert response.status_code == 200, response.text
    script = response.json()["script"]
    assert "def clean(df: pd.DataFrame)" in script
    compile(script, "clean.py", "exec")


def test_export_data_csv() -> None:
    dataset_id = _upload()
    response = client.post(
        f"/api/datasets/{dataset_id}/export/data",
        json={"ops": [{"op": "drop_duplicates"}]},
    )
    assert response.status_code == 200
    assert "name,age,score,city" in response.text


def test_apply_invalid_op_returns_422() -> None:
    dataset_id = _upload()
    response = client.post(
        f"/api/datasets/{dataset_id}/apply",
        json={"ops": [{"op": "cast_type", "column": "missing", "dtype": "int"}]},
    )
    assert response.status_code == 422


def test_heuristics_cover_text_columns() -> None:
    df = pd.DataFrame({"name": [" Alice ", "bob"], "amount": ["1,200.50", "990.99"]})
    suggestions = heuristic_suggestions(df, profile_dataframe(df))
    ops = {s.op.op for s in suggestions}
    assert "trim_whitespace" in ops
    assert any(s.op.op == "cast_type" and s.op.dtype == "float" for s in suggestions)


def test_parse_suggestions_drops_invalid() -> None:
    raw = [
        {"op": "drop_duplicates", "params": {}, "rationale": "dupes", "confidence": 0.9},
        {"op": "not_real", "params": {}},
        {"op": "cast_type", "params": {"column": "age", "dtype": "int"}},
        "garbage",
        {"op": "cast_type", "params": {"column": "age", "dtype": "not-a-dtype"}},
    ]
    parsed = parse_suggestions(raw, source="llm")
    assert len(parsed) == 2
    assert all(s.source == "llm" for s in parsed)
