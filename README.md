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
| Generation | OpenAI | Strict structured outputs, which the citation contract depends on |
| Embeddings | FastEmbed (local ONNX) | No vendor key, no per-document cost, re-indexing is free |
| Retrieval / chunking | LlamaIndex | Deepest retrieval + indexing module library |
| Orchestration | LangGraph | Stateful cyclic graphs, native checkpointing, HITL pauses |
| Vector DB | Qdrant | Fast, open-source, strong hybrid-search support |
| RAG metrics | Ragas | Canonical RAG metric suite; judges faithfulness, relevancy and correctness |
| CI quality gate | `answer_eval --gate` | The same Ragas scores the report uses, so the gate and the report cannot disagree |
| Tracing / observability | Langfuse | Open-source, traces every node next to its eval score |
| Packaging | Docker Compose | One-command reproducible run |

---

## Key features

- **Hybrid retrieval** — dense embeddings + BM25 sparse, fused by reciprocal rank,
  with an optional cross-encoder reranker. Both off by default: on the evaluation
  corpus they measured *worse* than dense alone, and the numbers are in
  [`eval/README.md`](eval/README.md) rather than assumed.
- **Chunk-level citations** — every claim maps to a source span, and a claim whose
  citations don't resolve is reported rather than silently dropped or silently kept.
- **Agentic control flow** (`AGENT_ENABLED=true`) — a LangGraph state machine that
  routes off-topic questions away instead of answering them from model memory,
  drops retrieved chunks that can't support an answer *before* they can be cited,
  retries once with the question rephrased in the documents' vocabulary, and
  scores its own answer against the sources it used. Costs 3–5 model calls per
  question against the single pass's 1, which is why it is opt-in — and why every
  node can be switched off on its own.
- **Every node fails open** — routing, grading, and critique are additions to a
  pipeline that worked without them. If one is unreachable the query still
  completes: routing failure searches anyway, grading failure keeps every chunk,
  and a dead critic reports *unreviewed* rather than a clean bill of health.
- **Human-in-the-loop gate** — an answer with an uncited claim, a failed or
  doubtful critique, or sources that contradict each other is held, not
  returned. The requester gets a `202` and a review id. A reviewer sees the full
  answer, its citations, and why it was held, and approves or rejects it. The
  rules read signals the pipeline already produced, so the gate adds no model
  calls and runs on the single-pass pipeline too. The queue is SQLite, so it
  survives a restart.
- **Contradiction detection** (`AGENT_ENABLED=true`) — the critic also reports
  retrieved sources that disagree on the point in question (a contract saying
  thirty days and its schedule saying forty-five, for example), each resolved to
  the two spans. It runs inside the critique call that already happens, so it
  costs no extra call.
- **Full evaluation harness** — retrieval, routing, contradiction detection, and
  end-to-end answer quality each have a runner and a written result, negative
  results included ([`eval/ANSWERS.md`](eval/ANSWERS.md)). The answer eval
  doubles as a CI regression gate.

---

## Getting started

### Prerequisites
- Docker and Docker Compose
- An [OpenAI API key](https://platform.openai.com/api-keys) in `.env`. Embeddings, BM25, and reranking all run locally, so this is the only vendor key involved — and only for answering. Ingestion works without it.

### Run without Docker

Qdrant also runs embedded, in-process, against a directory — same engine, same code path, nothing to install or start:

```ini
QDRANT_PATH=./data/qdrant
```

```powershell
uv run uvicorn app.main:app
```

It holds an exclusive lock on that directory, so one process only — run without `--reload`, and use the Compose setup above for anything with more than one worker.

### Choosing a model

```ini
ANSWER_MODEL=gpt-4o-mini   # writes the cited answer
AGENT_MODEL=gpt-4o-mini    # the graph's classification steps, when enabled
```

The answering model must support **strict structured outputs** (`gpt-4o-mini` or
newer). That is not a preference: the citation contract depends on
schema-constrained JSON, and without strict mode the schema is a hint — an answer
whose claims parse most of the time cannot be verified, which is the same as not
being verifiable at all.

`AGENT_MODEL` is separate so the agentic graph's four extra calls — routing,
grading, rewriting, critique — can run on something cheaper than the model that
writes the answer. They are yes/no judgements and paraphrase; running them on the
answering model is the easiest way to make the graph cost several times what it
should for no measurable gain.

Only generation touches a vendor at all. Embeddings run locally, so changing
either model re-indexes nothing and costs nothing.

### Run

```bash
git clone <your-repo-url>
cd enterprise-doc-agent
cp .env.example .env        # add your OPENAI_API_KEY
docker compose up --build
```

This starts Qdrant and the FastAPI app. Open `http://localhost:8000` — it redirects to the interactive docs at `/docs`, where you can upload a PDF and ask questions without leaving the browser. The Qdrant dashboard is at `http://localhost:6333/dashboard`.

> **On Windows PowerShell**, `curl` is an alias for `Invoke-WebRequest` and rejects the flags below with *"A parameter cannot be found that matches parameter name 'F'"*. Use `curl.exe` — the real one, in `System32` — or the PowerShell form shown after each example.

Tracing is optional: set `LANGFUSE_*` in `.env` to send traces to Langfuse Cloud or your own instance. Left blank, the app runs identically and discards spans.

### Ingest documents

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@path/to/document.pdf"
```

```powershell
curl.exe -F "file=@path/to/document.pdf" http://localhost:8000/api/v1/ingest
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

```powershell
$body = @{ question = "What is the termination notice period?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/v1/query `
  -Method Post -ContentType "application/json" -Body $body
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
  "contradictions": [],
  "critique": null,
  "steps": ["retrieve", "generate"],
  "trace_id": "...",
  "review": null
}
```

Three fields carry the phase-1 guarantee:

- **`claims`** — the answer decomposed into individual assertions, each with the spans that support it. Prose with `[1]` markers would read the same but could not be checked; a claim with an empty `citations` list can be.
- **`unsupported_claims`** — assertions whose citations did not resolve to a source we actually retrieved. Out-of-range source numbers are dropped, never clamped to a nearby one. A non-empty list holds the answer for review.
- **`answerable`** — `false` means the retrieved documents do not contain the answer. Distinguishing that from a short answer is the difference between "go find the right document" and "read this one".

`top_k` may be passed per request to override the configured default.

### Human review

An answer that trips a review rule comes back as `202 Accepted`. The content
fields are withheld, and a `review` object says why:

```json
{
  "answer": "This answer has been held for human review before release. ...",
  "claims": [], "citations": [], "unsupported_claims": [], "contradictions": [],
  "review": {
    "id": "a38b7a55e67344c58f87b8a6ddf3b051",
    "status": "pending",
    "reasons": ["claim cites no retrieved source: 'Penalties accrue at 5% monthly.'"]
  }
}
```

What holds an answer: a claim citing no retrieved source; answering "from the
documents" while citing nothing; a critic that could not be reached; a critic
verdict of *not supported*, or confidence below `REVIEW_MIN_CONFIDENCE` (0.7);
and sources that contradict each other. What does not: an honest "no document
covers this". The reasoning for each rule is in
[`app/review/gate.py`](app/review/gate.py).

```powershell
# the queue, oldest first (?status=approved|rejected for decided ones)
Invoke-RestMethod http://localhost:8000/api/v1/reviews

# one held answer: the full response as it would have been released, and why it was held
Invoke-RestMethod http://localhost:8000/api/v1/reviews/<id>

# decide it; a decision is final, and a second one gets 409
$body = @{ decision = "approve"; reviewer = "legal-ops"; note = "checked schedule B" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/v1/reviews/<id>/decision `
  -Method Post -ContentType "application/json" -Body $body
```

The requester polls the same `GET /reviews/{id}` for the outcome. Once it is
approved, `response` is exactly what the reviewer read. The queue lives in
SQLite at `REVIEW_DB_PATH`, on a volume under Compose.

Two caveats. **There is no authentication** here, as on every other route. The
`reviewer` name is recorded but not verified, and anyone who can reach
`/reviews/{id}` can read a held answer. Treat the gate as a workflow step, not an
access control, until the routes sit behind your identity provider. And
`REVIEW_ENABLED=false` returns every answer as before. The trace still records
what the gate *would* have held, which is the cheapest way to choose a threshold
before switching it on.

*Why not LangGraph `interrupt_before`?* It was the original plan. But the gate
has to cover the single-pass pipeline, which has no graph. And the signal it
needs most, `unsupported_claims`, is computed after the graph finishes. Pausing a
graph with nothing left to run after the pause would have been a queue with
extra steps.

---

## Tests

```bash
uv sync            # installs the dev group
uv run pytest
```

The suite needs no Qdrant, no API key, and no network: `tests/fakes.py` supplies a hashed bag-of-words embedder, an in-memory vector store, and a scripted model, and the app is built with those. Everything above them — parsing, chunking, the pipeline, retrieval, citation grounding, both routes — is the code that runs in production.

---

## Evaluation

### Retrieval — shipped

A golden set of 20 questions over a 13-document corpus, scored on hit@k, MRR, and
nDCG@5. Runs offline and needs no API key.

```bash
python -m eval.retrieval_eval --json eval/results.json
```

It exists to answer one question per change: did this actually help? For phase 2
the answer was **no** — hybrid retrieval and cross-encoder reranking both measured
*worse* than the dense baseline on this corpus, so both ship off by default. The
full result, the diagnosis, and the caveats are in
**[`eval/README.md`](eval/README.md)**.

Relevance is anchored to verbatim gold spans rather than chunk ids, so re-tuning
the chunker cannot silently invalidate the ground truth;
`tests/test_eval_corpus.py` enforces that the spans stay unique and reachable.

### Routing — shipped

Twenty questions run repeatedly through the router, scoring how often an
answerable question actually reaches retrieval. Needs an API key, because it
calls `AGENT_MODEL`.

```bash
python -m eval.routing_eval --trials 12
```

This exists because the router is the only node whose failure is silent: a wrong
refusal ends the graph before retrieval and looks exactly like a right one. Its
first live run refused a question the document answered, eleven times out of
twelve. The regression, the fix, and what the numbers do and don't prove are in
**[`eval/ROUTING.md`](eval/ROUTING.md)**.

### Contradiction detection — runner shipped, not yet measured

Twelve critique cases: six where two sources state the same term differently,
and six hard negatives (different terms with figures in the same unit, and
exceptions that refine a term). Scores detections and false alarms separately,
and prints the critic's confidence, which is what `REVIEW_MIN_CONFIDENCE` should
be set from. Needs an API key.

```bash
python -m eval.review_eval --trials 6
```

What it measures, and why the negatives matter as much as the positives, is in
**[`eval/REVIEW.md`](eval/REVIEW.md)**.

### Answer quality — shipped

The golden set plus six unanswerable questions, run through both pipelines and
scored with Ragas (faithfulness, answer relevancy) and a correctness judge
against the gold spans. Context precision and recall are computed exactly from
the gold spans, so no judge is needed there. Needs an API key and `uv sync --extra eval`.

```bash
python -m eval.answer_eval                          # both pipelines, ~440 calls
python -m eval.answer_eval --pipelines simple --gate   # the CI gate
```

**Result: on this corpus the agentic graph does not beat the single pass.**

| | single pass | agent |
|---|---|---|
| answer correctness | 0.675 | 0.725 |
| faithfulness | 0.805 | 0.894 |
| answer relevancy | 0.766 | 0.698 |
| unanswerable questions declined | 6/6 | 6/6 |
| model calls / question | 1.0 | 4.0 |

Correctness differs by one question in twenty, which is inside the noise. The
agent declined two answerable questions and costs 4× the calls. What it adds is
review signals (critique, contradictions, holds), not better answers, so
`AGENT_ENABLED` stays off by default. The eval also found a router regression on
technical-policy questions (fixed, 0/12 → 12/12) and two Ragas metrics that
misgrade fragment references.

The full results, all six findings and the caveats are in
**[`eval/ANSWERS.md`](eval/ANSWERS.md)**.

The README's original targets (faithfulness ≥ 0.90, relevancy ≥ 0.85, precision
≥ 0.80, recall ≥ 0.80) were measured for the first time here, and the single pass
misses three of them. `--gate` therefore enforces **regression floors** set just
below today's scores. A gate that fails every build gets switched off, so the
targets stay as goals.

---|---|
| Faithfulness | ≥ 0.90 |
| Answer relevancy | ≥ 0.85 |
| Context precision | ≥ 0.80 |
| Context recall | ≥ 0.80 |

The intent is that CI blocks a pull request that drops a metric below threshold, and
that every score is attached to its Langfuse trace for drill-down.

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
│   ├── review/            # review gate rules, contradiction resolution, the queue
│   └── observability/     # Langfuse client, tracing middleware
├── tests/                 # in-process stand-ins for the embedder, store, and model
├── eval/                  # retrieval, routing, review and answer-quality evals
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

Every layer below the API sits behind a Protocol (`Embedder`, `VectorStore`, `Answerer`), and `app/services.py` is the only module that names a concrete one. That is what lets the test suite run the production pipeline end to end with no container, no network, and no API key.

---

## Roadmap

- [x] Baseline RAG with citations
- [x] Hybrid retrieval + reranker (measured: [no lift on this corpus](eval/README.md))
- [x] Agentic graph: route → grade → rewrite → generate → critique (opt-in; [routing measured](eval/ROUTING.md), [end to end: no lift](eval/ANSWERS.md))
- [x] Human-in-the-loop gate + contradiction detection ([runner shipped](eval/REVIEW.md), not yet measured)
- [x] Answer-quality eval with Ragas + CI regression gate ([measured: agent does not beat the single pass](eval/ANSWERS.md))
- [ ] Attach eval scores to Langfuse traces; run the gate in CI
- [ ] One-command Docker packaging

---

## License

MIT
