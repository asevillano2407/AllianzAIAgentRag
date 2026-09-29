# Technical approach

## Executive summary

The solution uses retrieval augmented generation rather than fine-tuning. The
manual is small, the answer must show its evidence and the source may change.
The application extracts page text, creates deterministic chunks, embeds them
locally and stores them in Chroma. A bounded LangGraph workflow retrieves the
most relevant evidence, asks Gemini on Vertex AI for a typed response and checks
that every citation points to a retrieved chunk.

The system supports a claims handler. It does not decide legal liability,
coverage, payment or rejection. This distinction is important because the
manual defines operational criteria between insurers and dates from 2004.

## Requirements interpretation

The interview brief asks for an LLM-based RAG system over the supplied manual,
live responses to factual questions and accident descriptions, basic quality
evaluation, source code, a technical document and a presentation. The five
example accidents also require the answer to identify parties, circumstances,
agreement responsibility and missing evidence.

## Components

### Ingestion

`pypdf` extracts every page. Cleaning removes page-number-only lines, soft
hyphens, repeated whitespace and unrecoverable glyph markers. Chunks never cross
a page boundary. Each chunk stores the source file, PDF page, detected section
and a stable SHA-256-derived identifier.

The legacy PDF visibly renders correctly but its embedded text replaces some
accented Spanish characters. Guessing those characters could alter legal terms,
so the MVP uses deterministic cleaning. OCR with a Spanish model and a rendered
page checksum is the next production improvement.

### Retrieval

`paraphrase-multilingual-MiniLM-L12-v2` produces local embeddings. Chroma uses a
persistent cosine-distance HNSW index. The workflow requests six candidates and
drops results beyond the configured distance threshold. Both values are
configuration, not hardcoded business logic.

### Agent workflow

LangGraph holds explicit state containing the query, route, retrieved chunks,
answer, validation feedback, warnings and retry count. Routing between a factual
question and an accident narrative uses deterministic lexical rules. An LLM is
not needed for this low-risk decision.

The graph performs these steps:

1. Validate input and choose the route.
2. Retrieve and threshold evidence.
3. Generate a Pydantic-constrained response.
4. Validate source, page and chunk identifiers.
5. Retry generation once with validation feedback.
6. Return a low-confidence fallback if validation still fails.

The bounded retry prevents an infinite agent loop. Every node has one
responsibility and can be tested separately.

### Generation and prompt security

Gemini receives a system policy, the user request and only the retrieved
excerpts. The prompt marks excerpts as untrusted reference data and forbids them
from overriding application instructions. The schema separates the plain answer,
parties, agreement responsibility, applicable framework, key facts, missing
information, citations, confidence and limitations.

The model must distinguish agreement responsibility from legal liability,
coverage, compensation and criminal responsibility. It must state when evidence
is insufficient and must not automate a material claim decision.

### Interfaces

FastAPI exposes a health check and `POST /v1/analyze`. Streamlit calls the API
instead of duplicating orchestration logic. The command line handles ingestion,
single queries and evaluation. Docker Compose runs the API and UI while mounting
the persistent data directory.

## Evaluation

The versioned JSON Lines dataset covers the five interview examples and manual
topics such as semaphores, reverse motion, direct collision and priority. Offline
metrics include Recall@K and mean reciprocal rank. When model credentials are
available, evaluation also checks citation validity and required structured
fields.

Retrieval and generation are measured separately because a fluent answer cannot
compensate for missing evidence. The next iteration should add expert-labelled
answer correctness, faithfulness and decision usefulness.

## Testing strategy

Unit tests use in-memory fakes and never call Vertex AI. They cover cleaning,
stable chunk identifiers, page boundaries, routing, citation rejection, safe
fallbacks, retrieval metrics and missing inputs. CI runs Ruff and pytest. A small
integration test with a temporary Chroma collection can be added after pinning
the deployment image and embedding cache.

## Privacy, security and operations

No credentials are committed. Vertex AI uses Application Default Credentials.
The MVP only indexes the supplied public-style manual and does not process claim
personal data. A production service would add identity-aware access, encrypted
storage, region controls, prompt and response redaction, audit logs, model and
prompt versioning, monitoring and an approval checkpoint before any downstream
action.

Key operational signals are request latency, retrieval distance, fallback rate,
invalid-citation rate, token use, model errors and user feedback. Logs must avoid
raw personal data and complete prompts.

## Trade-offs and remaining risks

The local embedding model keeps ingestion private and reproducible, but it adds
image size and startup cost. Chroma is appropriate for a single-document demo;
a managed vector service would improve availability and access control at scale.
Vertex AI aligns with enterprise deployment, but live inference depends on
credentials and network access.

The largest knowledge risk is source currency. The UI and response schema expose
the 2004 limitation. A production system needs an owner, update process and
effective-date metadata before claims handlers can rely on it operationally.
