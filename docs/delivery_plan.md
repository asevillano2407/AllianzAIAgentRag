# Five day delivery plan

## Goal and definition of done

The deliverable is an interview-ready assistant that answers questions about the
supplied CIDE, ASCIDE and CICOS manual and analyses accident descriptions with
traceable evidence. It must run locally, expose an API and a simple UI, fail
safely, and include a reproducible evaluation.

Done means that a reviewer can clone the repository, configure Vertex AI, build
the index, run the sample cases and inspect the cited PDF pages. The repository
also contains tests, architecture rationale, a presentation and a demo runbook.

## Scope

### Included

- One supplied PDF and one persistent Chroma collection.
- Spanish questions and accident narratives.
- Page-aware semantic retrieval.
- Structured Gemini responses with citation validation.
- Human review warning and explicit insufficient-evidence behavior.
- Ten golden examples covering the interview hints and key manual sections.

### Deferred

- Production claims-system integration and personal data ingestion.
- Automated coverage, payment or rejection decisions.
- OCR pipeline and document-version comparison.
- Authentication, tenant isolation and full cloud infrastructure as code.
- Fine-tuning. RAG is more appropriate because the source is small, traceability
  matters and agreement content may change.

## Milestones

| Day | Outcome | Acceptance check |
| --- | --- | --- |
| 1 | Requirements, manual inspection, architecture and repository skeleton | Source pages extract with metadata and all decisions are recorded |
| 2 | Ingestion and retrieval | Golden queries return relevant pages in the top six results |
| 3 | Agent graph, structured generation and guardrails | Citations validate and failures reach a bounded fallback |
| 4 | API, UI, Docker and evaluation | Example cases run end to end without external test dependencies |
| 5 | Tests, presentation, runbook and rehearsal | Clean setup, timed demo and documented limitations |

## Main risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Legacy PDF loses accented glyphs | Lower retrieval quality | Multilingual embeddings, deterministic cleaning, page citations and OCR backlog |
| Manual dates from 2004 | Outdated operational or legal conclusion | Visible source-date warning and human review requirement |
| LLM invents a citation | Untraceable answer | Structured citations checked against retrieved chunk metadata |
| Live model or network fails | Demo interruption | Pre-ingested index, health check and a prepared failure/fallback explanation |
| Five-day time limit | Incomplete breadth | One strong vertical slice and explicitly deferred production features |

## Demo success measures

- Retrieval Recall@6 and MRR over the versioned golden dataset.
- Citation validity and structured response completeness.
- One normal question, one ambiguous accident and one out-of-scope case.
- Explanation of a controlled failure without hiding the limitation.
