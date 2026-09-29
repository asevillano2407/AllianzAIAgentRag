# Allianz Claims Copilot

Agentic RAG for explaining CIDE, ASCIDE and CICOS criteria and analysing motor
claim scenarios against the supplied manual. The system retrieves page-level
evidence, asks an LLM for a structured answer, validates citations and returns a
safe fallback when the manual does not support a conclusion.

> Interview project. The output supports claims handlers; it is not a legal or
> coverage decision and requires human review.

## What this demonstrates

- Reproducible PDF ingestion and deterministic, page-aware chunking.
- Multilingual semantic retrieval with a persistent Chroma collection.
- Explicit LangGraph state, deterministic routing and bounded retries.
- Structured LLM output with evidence and citation validation.
- FastAPI and Streamlit interfaces backed by the same application service.
- Offline retrieval and response-quality evaluation.
- Unit tests, Docker packaging, CI and operational documentation.

## Architecture

```text
Question or accident description
        |
        v
Input validation and deterministic routing
        |
        v
Chroma retrieval over page-aware chunks
        |
        v
LLM structured generation (Vertex AI Gemini)
        |
        v
Citation and output validation -- one bounded retry
        |
        v
Evidence-backed answer or explicit insufficient-evidence response
```

See [the technical approach](docs/technical_approach.md) for the detailed
design and [the five-day plan](docs/delivery_plan.md) for scope and milestones.

## Quick start

### 1. Configure

```bash
cp .env.example .env
```

For Vertex AI, authenticate with Application Default Credentials and set
`GOOGLE_CLOUD_PROJECT`. The default model is `gemini-2.5-flash`.

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

### 3. Build the index

Place the supplied manual at `data/raw/Manual-cide-ascide-y-cicos.pdf`, then:

```bash
allianz-rag ingest
```

### 4. Run the demo

```bash
uvicorn allianz_rag.api:app --reload
streamlit run app/streamlit_app.py
```

The API documentation is available at `http://localhost:8000/docs` and the UI
at `http://localhost:8501`.

### 5. Evaluate

```bash
allianz-rag evaluate --dataset evaluation/golden_dataset.jsonl
pytest
```

Retrieval evaluation does not call the LLM. Full answer evaluation requires
Vertex AI credentials and is enabled with `--include-generation`.

## Docker

```bash
docker compose up --build
```

The compose stack mounts `data/` so the vector index persists between runs.

## Repository map

```text
src/allianz_rag/       Core application, graph, ingestion and evaluation
app/                   Streamlit demo
data/raw/              Supplied source document
evaluation/            Versioned golden questions and expected evidence
tests/                 Unit tests without external API calls
docs/                  Architecture, plan and interview runbook
deliverables/          Technical document and interview presentation
```

## Key design decisions

- Retrieval uses a local multilingual embedding model, so the manual does not
  leave the development environment during indexing.
- The LLM only receives retrieved excerpts and treats them as untrusted data.
- Responsibility under the agreements is kept separate from legal liability.
- All state transitions are explicit and retries are capped at one.
- Provider and model settings live in environment variables; no credential is
  stored in the repository.

## Known limitations

- The legacy PDF text layer contains damaged accented characters. Cleaning is
  deterministic, while page images remain the source of truth for human review.
- The supplied manual dates from 2004. The application surfaces this limitation
  and must not imply that its content reflects current law or current agreements.
- The demo does not automate a payment, rejection or coverage decision.
