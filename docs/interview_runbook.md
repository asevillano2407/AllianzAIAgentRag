# Interview runbook for 30 to 45 minutes

## Suggested timing

| Minutes | Topic |
| --- | --- |
| 0 to 4 | Problem, user and safety boundary |
| 4 to 10 | Requirements, assumptions and five-day scope |
| 10 to 18 | Ingestion, retrieval and source-quality challenge |
| 18 to 26 | Agent graph, structured output and guardrails |
| 26 to 34 | Live demo with two cases and one controlled limitation |
| 34 to 39 | Evaluation, tests and observability |
| 39 to 45 | Trade-offs, production roadmap and questions |

## Demo sequence

1. Show repository structure and `.env.example` without opening credentials.
2. Run the health endpoint and confirm the index size.
3. Ask whether alcoholemia alone excludes the agreements. Open the cited page.
4. Analyse the rear-end collision at a red light. Explain retrieved evidence,
   missing facts and the distinction between agreement responsibility and law.
5. Use the unidentified parked-car case. Highlight the low-confidence or
   insufficient-evidence behavior rather than forcing a conclusion.
6. Show the latest evaluation JSON and one citation-validation unit test.

## Questions to expect

**Why RAG instead of fine-tuning?** The task needs current, inspectable evidence.
The corpus is small and changes should be reflected by re-indexing, not training.

**Why is this agentic?** It has explicit state and multiple controlled decisions:
routing, retrieval, thresholding, generation, validation, retry and fallback.
The deterministic decisions stay outside the LLM.

**How do you prevent hallucinations?** The model sees only retrieved evidence,
returns a schema and must cite immutable metadata. Code rejects unknown citations
and permits only one retry before a safe fallback.

**How would this scale?** Move embeddings and vectors to managed regional
services, add document versioning and access control, use asynchronous ingestion,
cache frequent queries and trace each graph node.

**What would you improve with more time?** Spanish OCR, expert-labelled cases,
hybrid lexical and vector retrieval, reranking, identity controls, PII redaction,
observability dashboards and a formal document update workflow.

## Demo fallback

If Vertex AI is unavailable, show retrieval evaluation and the graph tests, then
walk through the saved structured schema. Do not replace the live result with an
undisclosed canned response.
