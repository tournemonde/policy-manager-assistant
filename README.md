# My Insurance Policy Manager Assistant

AI-powered policy assistant that answers insurance questions using Retrieval-Augmented Generation (RAG) over uploaded or baseline PDF policy documents.

## What it does

- Parses policy PDFs into structured chunks using `landingai-ade`
- Embeds and indexes chunks in local ChromaDB
- Answers user questions with `gpt-4o` grounded on retrieved excerpts
- Shows source citations and highlighted PDF locations in the UI
- Supports two UI modes:
  - Integrated (chat + expandable source previews)
  - Side by side (chat left, document viewer right)
- Supports document upload at runtime

## Tech stack

- Python 3.12
- Poetry
- Streamlit
- Landing.ai ADE (`landingai-ade`)
- OpenAI (`gpt-4o`, `text-embedding-3-small`)
- ChromaDB (`0.6.3`)
- PyMuPDF (`pymupdf`) for page rendering/highlighting
- Google Cloud Storage + Cloud Run (deployment mode)

## Project structure

See full commented structure and deep internals in `docs/ARCHITECTURE.md`.

## Local setup

1. Install dependencies:

```bash
poetry install
```

2. Create `.env` from `.env.example` and fill keys:

```env
OPENAI_API_KEY=...
LANDINGAI_API_KEY=...
# Optional
# OPENAI_MODEL=gpt-4o
```

3. Place baseline PDFs in `data/1_raw/` (or use existing files).

## Build index pipeline (local)

1. Parse raw PDFs to chunks:

```bash
poetry run python -m chatbot.ingest
```

2. Index chunks into ChromaDB:

```bash
poetry run python -m chatbot.index
```

3. Optional CLI RAG test:

```bash
poetry run python -m chatbot.rag
```

## Run the app (local)

```bash
poetry run streamlit run app.py
```

## Deployment model (Cloud Run)

This app uses a **baseline + uploads archive** model:

- `baseline/` (GCS): canonical startup dataset (raw PDFs, chunks, ChromaDB)
- `uploads/` (GCS): user-uploaded files/chunks archive

On Cloud Run startup:

1. `startup.py` syncs `baseline/` into local `data/`
2. App starts from that baseline only

At runtime:

- Uploaded documents are parsed/indexed for the running instance
- Upload artifacts are archived to `uploads/`
- On next restart, app reloads baseline (not archived uploads)

## Key environment variables

- `OPENAI_API_KEY` (required)
- `LANDINGAI_API_KEY` (required)
- `OPENAI_MODEL` (optional, default `gpt-4o`)
- `GOOGLE_PROJECT_ID` (required in Cloud Run mode)
- `GCS_BUCKET_NAME` (enables GCS sync/archive)
- `GCS_BASELINE_PREFIX` (default `baseline`)
- `GCS_UPLOADS_PREFIX` (default `uploads`)

## Quality checks

```bash
poetry run ruff check .
poetry run pytest
```

## Documentation

- Architecture and file/function-level documentation: `docs/ARCHITECTURE.md`

