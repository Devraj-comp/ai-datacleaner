# datacleaner — API

Backend for the AI-powered data cleaning tool. Ingests CSV/JSON, profiles the
data, asks an LLM (Ollama) for structured **cleaning operations**, applies the
approved ops with pandas, and exports a standalone cleaning script.

The model never writes executable code. It emits ops from a typed DSL
(`app/core/ops.py`); each op can both `apply()` itself and `emit_pandas()`, so
the preview and the exported script are always identical.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with `llama3:latest`
  (`ollama pull llama3:latest`). The API still works without it via the
  heuristic fallback.

## Setup

```bash
cd apps/api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Docs: http://localhost:8000/docs

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET  | `/api/health` | Liveness + configured model |
| POST | `/api/datasets` | Upload CSV/JSON, returns profile + preview |
| GET  | `/api/datasets/{id}/profile` | Statistical profile |
| GET  | `/api/datasets/{id}/preview` | First N rows |
| POST | `/api/datasets/{id}/suggest` | LLM/heuristic cleaning suggestions |
| POST | `/api/datasets/{id}/apply` | Apply approved ops, return preview + diff counts |
| POST | `/api/datasets/{id}/export/script` | Standalone `.py` cleaning script |
| POST | `/api/datasets/{id}/export/data` | Cleaned CSV download |

## The op DSL

`cast_type`, `fill_null`, `drop_duplicates`, `clip_outliers`, `trim_whitespace`,
`normalize_case`, `standardize_categories`, `parse_date`, `drop_column`,
`rename_column`, `replace_values`, `validate_regex`.

`validate_regex` nulls values that fail to match (non-destructive).

## Optional: pandas-profiling

```bash
pip install ydata-profiling
DC_ENABLE_YDATA_PROFILE=true uvicorn app.main:app --reload
```

Writes `storage/{dataset_id}_profile.html` on upload.

## Tests

```bash
cd apps/api && pytest
```

## Layout

```
app/
├─ core/ops.py        # DSL: apply + emit_pandas per op
├─ services/
│  ├─ ingest.py       # CSV/JSON parsing
│  ├─ profiler.py     # profile + optional ydata report
│  ├─ llm.py          # Ollama provider + heuristic fallback
│  └─ pipeline.py     # apply ops, preview, script codegen
├─ api/routes.py      # HTTP endpoints
├─ models.py          # Pydantic models
├─ storage.py         # in-memory dataset store
└─ main.py            # FastAPI app
```
