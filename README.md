# Policy Manager Assistant

AI-powered **document chat** for insurance policies: users ask natural-language questions and get answers **grounded in uploaded PDF wording**, with **citations** and **highlighted source pages** in the UI.

**Live demo:** [Policy Manager Assistant on Cloud Run](https://policy-manager-assistant-981308040437.europe-west1.run.app/)

## What it does

- Parses policy PDFs into structured chunks with [Landing.ai ADE](https://landing.ai/)
- Embeds chunks and stores them in **ChromaDB** (cosine similarity)
- Answers with **OpenAI** (`gpt-4o` by default) using retrieval-augmented generation (RAG)
- Shows **source document + page** and renders **bbox highlights** on the PDF (PyMuPDF)
- **Upload** new PDFs at runtime (parse → index → chat)
- Two layouts: integrated chat + expanders, or side-by-side chat + document viewer
- **Cloud Run** deployment with optional **GCS** baseline sync and upload archive

## Tech stack

| Layer | Tools |
|-------|--------|
| UI | Streamlit |
| Packaging | Poetry, Python 3.12 |
| Ingestion | `landingai-ade` |
| Embeddings / LLM | OpenAI (`text-embedding-3-small`, `gpt-4o`) |
| Vector store | ChromaDB 0.6.3 |
| Rendering | PyMuPDF |
| Cloud | Docker, Cloud Run, Google Cloud Storage |

## Architecture (summary)

```text
PDF → ADE parse → chunks JSON → OpenAI embeddings → ChromaDB
User question → embed → retrieve top-k → LLM answer + citation metadata → Streamlit UI
```

Detailed flows, component diagram, and module reference: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Local setup

1. Install dependencies:

```bash
poetry install
```

2. Copy environment template and set keys:

```bash
cp .env.example .env
```

Required locally:

- `OPENAI_API_KEY`
- `LANDINGAI_API_KEY`

3. Add your own policy PDFs under `data/1_raw/` (not included in this repo).

## Index pipeline

```bash
poetry run python -m chatbot.ingest
poetry run python -m chatbot.index
```

Optional CLI RAG smoke test:

```bash
poetry run python -m chatbot.rag
```

## Run the app

```bash
poetry run streamlit run app.py
```

## Cloud Run deployment

Container entrypoint: `startup.py` (baseline sync from GCS, then Streamlit on `$PORT`).

| Concept | Behavior |
|---------|----------|
| `baseline/` on GCS | Canonical dataset restored on each container start |
| `uploads/` on GCS | Runtime uploads archived separately |
| Restart | Reloads baseline only (uploads are not auto-loaded) |

Optional variables: `GOOGLE_PROJECT_ID`, `GCS_BUCKET_NAME`, `GCS_BASELINE_PREFIX`, `GCS_UPLOADS_PREFIX`, `OPENAI_MODEL`.

Build and deploy using your own GCP project; see `Dockerfile` and `docs/ARCHITECTURE.md`.

## Demo questions

Example prompts for policy wording documents: [`docs/questions.md`](docs/questions.md).

## Quality checks

```bash
poetry run ruff check .
poetry run pytest
```

## Data and secrets

- **Do not commit** `.env`, API keys, or customer PDFs.
- `data/` (raw PDFs, chunks, ChromaDB) is gitignored; regenerate locally or from your own GCS baseline.
- Use `.env.example` only as a template.

## Personal initiative

This repository demonstrates end-to-end **document RAG**: structured PDF parsing, vector search, grounded generation, citation UX, and production-style deployment patterns—not a managed SaaS product.
