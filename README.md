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
| Answer generation | Claude or OpenAI | Swapped by one setting; both held to the same citation schema |
| Embeddings | FastEmbed (local ONNX) | No vendor key, no per-document cost, re-indexing is free |
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
- An API key for one answer provider, set in `.env`: [Anthropic](https://console.anthropic.com/settings/keys) (default) or [OpenAI](https://platform.openai.com/api-keys). Embeddings run locally, so this is the only vendor key needed — and only for answering; ingestion works without it.

### Run without Docker

Qdrant also runs embedded, in-process, against a directory — same engine, same code path, nothing to install or start:

```ini
QDRANT_PATH=./data/qdrant
```

```powershell
uv run uvicorn app.main:app
```

It holds an exclusive lock on that directory, so one process only — run without `--reload`, and use the Compose setup above for anything with more than one worker.

### Choosing an answer provider

```ini
LLM_PROVIDER=openai        # or: anthropic
OPENAI_API_KEY=sk-proj-... # or: ANTHROPIC_API_KEY=sk-ant-...
```

Only generation changes. Embeddings are local either way, so switching providers re-indexes nothing and costs nothing. Both implementations sit behind the same `Answerer` protocol and share one prompt and one JSON schema, so the citation contract is identical whichever you pick — an OpenAI model just has to support strict structured outputs (`gpt-4o-mini` or newer). `ANSWER_MODEL` defaults to `claude-opus-5` or `gpt-4o-mini` to match the provider.

### Run

```bash
git clone <your-repo-url>
cd enterprise-doc-agent
cp .env.example .env        # add your ANTHROPIC_API_KEY
docker compose up --build
```

This starts Qdrant and the FastAPI app. The API is at `http://localhost:8000`, interactive docs at `http://localhost:8000/docs`, and the Qdrant dashboard at `http://localhost:6333/dashboard`.

Tracing is optional: set `LANGFUSE_*` in `.env` to send traces to Langfuse Cloud or your own instance. Left blank, the app runs identically and discards spans.

### Ingest documents

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@path/to/document.pdf"
```

```json
{"doc_id": "9f86d081...", "source": "contract.pdf", "pages": 14, "chunks": 37, "trace_id": "..."}
```

The `doc_id` is the SHA-256 of the file's contents, so re-uploading the same document replaces its chunks instead of duplicating them. Scans with no text layer are rejected with a 422 rather than silently indexed as empty.

### Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the termination notice period?"}'
```

```json
{
  "answer": "Either party may terminate on thirty days written notice.",
  "answerable": true,
  "claims": [
    {
      "text": "Either party may terminate on thirty days written notice.",
      "citations": [
        {
          "chunk_id": "6f9619ff-...",
          "doc_id": "9f86d081...",
          "source": "contract.pdf",
          "page": 12,
          "char_start": 1840,
          "char_end": 1993,
          "score": 0.81,
          "text": "Section 14.2. Either party may terminate this agreement..."
        }
      ]
    }
  ],
  "citations": ["... every source used, deduplicated, in rank order ..."],
  "unsupported_claims": [],
  "trace_id": "..."
}
```

Three fields carry the phase-1 guarantee:

- **`claims`** — the answer decomposed into individual assertions, each with the spans that support it. Prose with `[1]` markers would read the same but could not be checked; a claim with an empty `citations` list can be.
- **`unsupported_claims`** — assertions whose citations did not resolve to a source we actually retrieved. Out-of-range source numbers are dropped, never clamped to a nearby one. A non-empty list is the signal the phase-4 review gate will key on.
- **`answerable`** — `false` means the retrieved documents do not contain the answer. Distinguishing that from a short answer is the difference between "go find the right document" and "read this one".

`top_k` may be passed per request to override the configured default.

---

## Tests

```bash
uv sync            # installs the dev group
uv run pytest
```

The suite needs no Qdrant, no API key, and no network: `tests/fakes.py` supplies a hashed bag-of-words embedder, an in-memory vector store, and a scripted model, and the app is built with those. Everything above them — parsing, chunking, the pipeline, retrieval, citation grounding, both routes — is the code that runs in production.

---

## Evaluation

> Planned for phase 5 — the harness below is not wired up yet.

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
│   ├── main.py            # FastAPI entrypoint + app factory
│   ├── config.py          # settings, loaded from the environment
│   ├── services.py        # composition root: picks the concrete implementations
│   ├── api/routes/        # /health, /ingest, /query
│   ├── ingest/            # PDF parsing, chunking, embedding, the pipeline
│   ├── vectorstore/       # VectorStore protocol + the Qdrant implementation
│   ├── retrieval/         # dense retriever (hybrid + reranker land in phase 2)
│   ├── generation/        # cited-answer generation and citation grounding
│   ├── graph/             # LangGraph nodes and state definition (phase 3)
│   └── observability/     # Langfuse client, tracing middleware
├── tests/                 # in-process stand-ins for the embedder, store, and model
├── eval/                  # Ragas + DeepEval harness (phase 5)
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

Every layer below the API sits behind a Protocol (`Embedder`, `VectorStore`, `Answerer`), and `app/services.py` is the only module that names a concrete one. That is what lets the test suite run the production pipeline end to end with no container, no network, and no API key.

---

## Roadmap

- [x] Baseline RAG with citations
- [ ] Hybrid retrieval + reranker (with measured before/after lift)
- [ ] Agentic graph: route → grade → rewrite → generate → critique
- [ ] Human-in-the-loop gate + contradiction detection
- [ ] Ragas + DeepEval + Langfuse evaluation harness
- [ ] One-command Docker packaging

---

## License

MIT
