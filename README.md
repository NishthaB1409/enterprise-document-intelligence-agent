# Enterprise Document-Intelligence Agent

An agentic RAG system for enterprise document workflows. It ingests PDFs and contracts, answers questions **with chunk-level citations**, flags internal inconsistencies, and routes low-confidence cases to a human reviewer.

Retrieval here is a *decision the agent makes* — not a fixed pipeline step. The agent can skip retrieval, retrieve multiple times for multi-hop questions, grade whether retrieved chunks are actually relevant, and rewrite-and-retry when they are not. That self-correcting loop is what separates this from a one-pass RAG pipeline.

---

## Why this exists

Enterprise document automation (contract review, compliance, knowledge retrieval) demands three things that basic RAG doesn't provide: **traceable citations** for every claim, **confidence-gated human review** for high-stakes answers, and **measurable quality** you can defend. This project is built around those three requirements.

---

## Architecture

```
                 ┌─────────────┐
   query ───────▶│    ROUTE    │  retrieve or answer directly?
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  RETRIEVE   │  hybrid: dense (embeddings) + sparse (BM25) → rerank
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐     not relevant
                 │ GRADE DOCS  │──────────────┐
                 └──────┬──────┘              ▼
                        │ relevant     ┌─────────────┐
                        │              │  REWRITE Q  │
                        │              └──────┬──────┘
                        │                     │ retry
                        ▼                     ▲
                 ┌─────────────┐              │
                 │  GENERATE   │              │
                 │ + citations │              │
                 └──────┬──────┘──────────────┘
                        ▼
                 ┌─────────────┐
                 │  CRITIQUE   │  self-check + cross-chunk contradiction detection
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐   low confidence / flagged
                 │ HITL GATE   │──────────────▶  human review queue
                 └──────┬──────┘
                        ▼
                     answer
```

The graph is orchestrated with **LangGraph**, which models the flow as a stateful directed graph with explicit state, checkpointing, and `interrupt_before` human-in-the-loop pauses.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | Async, typed, standard for Python ML backends |
| Retrieval / chunking | LlamaIndex | Deepest retrieval + indexing module library |
| Orchestration | LangGraph | Stateful cyclic graphs, native checkpointing, HITL pauses |
| Vector DB | Qdrant | Fast, open-source, strong hybrid-search support |
| RAG metrics | Ragas | Canonical reference-free RAG metric suite |
| CI quality gates | DeepEval | pytest-style metric assertions for CI/CD |
| Tracing / observability | Langfuse | Open-source, traces every node next to its eval score |
| Packaging | Docker Compose | One-command reproducible run |

---

## Key features

- **Hybrid retrieval** — dense embeddings + BM25 sparse, fused and reranked.
- **Agentic control flow** — the agent routes, grades, rewrites, and retries instead of blindly stuffing top-k chunks.
- **Chunk-level citations** — every claim in an answer maps to a source span.
- **Contradiction detection** — a critique node cross-checks retrieved chunks for inconsistencies.
- **Human-in-the-loop gate** — low-confidence or flagged answers pause for approval via LangGraph `interrupt_before`.
- **Full evaluation harness** — Ragas metrics on a golden dataset, DeepEval gates in CI, all scores traced in Langfuse.

---

## Getting started

### Prerequisites
- Docker and Docker Compose
- An LLM API key (set in `.env`)

### Run

```bash
git clone <your-repo-url>
cd enterprise-doc-agent
cp .env.example .env        # add your API key
docker compose up
```

This starts Qdrant, Langfuse, and the FastAPI app. The API is available at `http://localhost:8000`.

### Ingest documents

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@path/to/document.pdf"
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the termination notice period?"}'
```

The response includes the answer, per-claim citations, a confidence score, and — if the answer was gated — a review-queue reference.

---

## Evaluation

Quality is measured against a golden dataset of question–answer pairs.

| Metric | Target |
|---|---|
| Faithfulness | ≥ 0.90 |
| Answer relevancy | ≥ 0.85 |
| Context precision | ≥ 0.80 |
| Context recall | ≥ 0.80 |

Run the evaluation suite:

```bash
# Ragas metrics over the golden dataset
python -m eval.run_ragas

# DeepEval quality gates (used in CI)
pytest eval/
```

The CI workflow blocks any pull request that drops a metric below its threshold, so quality regressions are caught before merge. Every eval score is attached to its Langfuse trace for drill-down.

---

## Project structure

```
enterprise-doc-agent/
├── app/
│   ├── main.py            # FastAPI entrypoint
│   ├── ingest/            # PDF parsing, chunking, embedding
│   ├── retrieval/         # hybrid retriever + reranker
│   ├── graph/             # LangGraph nodes and state definition
│   └── tracing.py         # Langfuse setup
├── eval/
│   ├── golden_dataset.json
│   ├── run_ragas.py
│   └── test_gates.py      # DeepEval pytest gates
├── docker-compose.yml
├── CLAUDE.md              # conventions + architecture for Claude Code sessions
├── .env.example
└── README.md
```

---

## Roadmap

- [ ] Baseline RAG with citations
- [ ] Hybrid retrieval + reranker (with measured before/after lift)
- [ ] Agentic graph: route → grade → rewrite → generate → critique
- [ ] Human-in-the-loop gate + contradiction detection
- [ ] Ragas + DeepEval + Langfuse evaluation harness
- [ ] One-command Docker packaging

---

## License

MIT
