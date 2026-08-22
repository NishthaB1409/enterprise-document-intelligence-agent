# Retrieval evaluation — hybrid search and reranking

**Headline: neither change improved retrieval on this corpus, and both are off by default.**
Dense retrieval alone scored best. The code for both ships, behind settings, because
the result is corpus-specific and the next corpus may well go the other way.

```bash
python -m eval.retrieval_eval --json eval/results.json
```

Runs offline once the ONNX models are cached. No API key — this measures retrieval
only, so no generation model is called.

---

## What was measured

| | |
|---|---|
| Corpus | 13 documents, 30 chunks (5 hold answers, 8 are distractors) |
| Questions | 20, split 10 exact-token / 10 paraphrased |
| Embeddings | `BAAI/bge-small-en-v1.5`, 384-dim, local |
| Sparse | `Qdrant/bm25`, IDF applied by the engine |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder |
| Fusion | Reciprocal rank (Qdrant server-side), `k=2` |
| `top_k` / `candidates` | 5 / 30 |

Relevance is binary and comes from a **gold span** — a verbatim phrase that answers
the question. A chunk counts as relevant if it contains that span, whitespace and
case normalised. Spans rather than chunk ids, because ids move whenever the chunk
size changes and would make a chunking tweak look like a retrieval regression.
`tests/test_eval_corpus.py` asserts each span resolves to exactly one passage, that
no distractor contains one, and that none straddle a chunk boundary.

The four configurations are a 2×2, so BM25 and reranking can be read independently.
Reporting only `dense` against `hybrid+rerank` would have produced one combined
number and hidden that the two changes pull in opposite directions.

---

## Results

**Overall** (MRR delta against the dense baseline)

| configuration | hit@1 | hit@3 | hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| **dense** (baseline) | **0.60** | 0.80 | **0.85** | **0.702** | **0.739** |
| dense+rerank | 0.55 | 0.80 | 0.85 | 0.679 (−0.022) | 0.723 |
| hybrid | 0.55 | 0.80 | 0.80 | 0.658 (−0.043) | 0.695 |
| hybrid+rerank | 0.55 | 0.80 | 0.85 | 0.679 (−0.022) | 0.723 |

**Exact-token questions** — clause numbers, `SOC 2 Type II`, `AES-256`, `99.95%`

| configuration | hit@1 | hit@3 | MRR |
|---|---|---|---|
| dense | 0.80 | 1.00 | 0.900 |
| hybrid | 0.80 | 1.00 | 0.900 (=) |
| dense+rerank | 0.70 | 1.00 | 0.850 (−0.050) |
| hybrid+rerank | 0.70 | 1.00 | 0.850 (−0.050) |

**Paraphrased questions** — question and answer share almost no vocabulary

| configuration | hit@1 | hit@3 | MRR |
|---|---|---|---|
| dense | 0.40 | 0.60 | 0.503 |
| dense+rerank | 0.40 | 0.60 | 0.508 (+0.005) |
| hybrid+rerank | 0.40 | 0.60 | 0.508 (+0.005) |
| hybrid | 0.30 | 0.60 | 0.417 (−0.087) |

---

## Why hybrid didn't help

The case for hybrid search is that dense embeddings blur rare tokens — a clause
number or a certification name gets averaged into a 384-dimensional summary
competing with every other word in the chunk — while BM25 treats exactly that
rarity as its strongest signal.

**That premise did not hold here.** Ranking each arm separately over the whole
corpus shows why:

| | dense | BM25 | fused |
|---|---|---|---|
| MRR over the full corpus | 0.717 | 0.537 | 0.678 |
| Gold chunk ranked #1 | 12/20 | 9/20 | 11/20 |

BM25 is working correctly — on the exact-token questions it puts the gold chunk
first in 9 of 10 cases. The problem is that **dense already does too**. There was
no question where dense failed and BM25 rescued it, so fusion had no wins
available to contribute.

Meanwhile BM25 found *nothing* on 5 of the 10 paraphrased questions — the gold
chunk never appeared in its ranking at any depth. RRF still gives its top-ranked
result full rank-1 credit, so an arbitrary lexical match displaces a correct dense
one. Four questions got worse and one got better:

| question | dense | BM25 | fused |
|---|---|---|---|
| How long do staff have to file receipts…? | 3 | not found | **7** |
| If the vendor breaks the deal badly…? | 1 | not found | **2** |
| What happens to our information…? | 2 | 7 | **3** |
| When would the platform be taken offline…? | 15 | not found | **19** |
| How often does someone check who can get in? | 5 | 3 | **3** ✅ |

This is RRF's known weakness: it assumes both arms are roughly equally
trustworthy. When one has no signal for a query, its noise still arrives at full
weight.

**Not a tuning artifact.** Candidate depth was swept, and hybrid loses at every
setting:

| candidates | 5 | 10 | 15 | 20 | 30 | dense |
|---|---|---|---|---|---|---|
| MRR | 0.668 | 0.658 | 0.658 | 0.658 | 0.658 | **0.702** |

## Why reranking didn't help

The cross-encoder cost 0.05 MRR on exact-token questions and returned 0.005 on
paraphrased ones. It reorders a shortlist that was already well ordered, and on
this corpus it was more likely to demote a correct top hit than to promote one.
A reranker earns its keep when retrieval returns the right chunk somewhere in the
top 20 but ranks it poorly — here dense retrieval already put it first 60% of the
time.

---

## What this does and doesn't establish

**Take seriously:**

- BM25 and the fusion path work; they are pinned by `tests/test_hybrid_retrieval.py`
  and `tests/test_qdrant_store.py` against a real Qdrant engine.
- On *this* corpus, with *this* embedding model, neither addition pays for itself.
  Shipping them on by default would have been a regression, sold as an upgrade.
- Modern small embedding models handle exact identifiers better than the standard
  argument for hybrid search assumes. That argument dates from weaker encoders and
  deserves re-testing rather than repeating.

**Do not take seriously:**

- **The sample is too small for the differences to be significant.** With 20
  questions each contributes at most 0.05 to MRR, so the −0.043 gap is roughly one
  question's worth of rank change. Treat the ordering as "no measured improvement",
  not as a quantified regression.
- **30 chunks is a small index.** At `candidates=30` each arm ranks the entire
  corpus, which is not what fusion faces in production. On a corpus of thousands,
  BM25's top-30 is a far more selective signal, and the result could reverse.
- **The corpus is synthetic**, written by the same person who wrote the questions.
  Real documents are messier, and real questions are worse-formed.
- **Retrieval only.** Whether these rankings produce better *answers* is phase 5.

## What would change the recommendation

Re-run `python -m eval.retrieval_eval` against your own documents. Turn hybrid on
(`RETRIEVAL_MODE=hybrid`) if you see any of:

- A corpus in the thousands of chunks, where dense recall drops and BM25's
  selectivity starts to matter.
- Identifiers the embedding model has never seen — internal part numbers, ticket
  ids, customer-specific codes. The corpus here uses *public* conventions
  (`SOC 2`, `TLS 1.3`) that bge-small has seen in training, which is precisely
  why dense handled them.
- Multilingual or domain-specific text where the embedding model is weak.

Enable reranking (`RERANK_ENABLED=true`) when hit@5 is comfortably above hit@1 —
that gap is the headroom a reranker has to work with. Here it was 0.25, and the
cross-encoder still could not convert it.

## Reproducing

```bash
python -m eval.retrieval_eval --json eval/results.json   # full run, ~1 min cached
pytest tests/test_eval_corpus.py                          # ground-truth integrity
```

`eval/results.json` holds the full output including per-configuration misses.
Files: `corpus.py` (documents with answers), `distractors.py` (documents without),
`questions.py` (the golden set), `metrics.py` (hit@k, MRR, nDCG), `retrieval_eval.py`
(the harness).
