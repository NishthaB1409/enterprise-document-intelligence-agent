"""Application settings, loaded from the environment (and `.env` in development)."""

from functools import lru_cache
from typing import ClassVar, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "edia"
    # Tags every trace so dev/staging/prod runs stay separable in one Langfuse project.
    environment: str = "development"
    # Set from the git SHA in CI so a regression can be traced back to a build.
    release: str | None = None

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"
    # Embedded mode: a directory instead of a server. `qdrant-client` runs the
    # engine in-process, so the whole stack works with no Docker at all — which
    # is the difference between "can demo this" and "cannot" on a machine
    # without it. Takes precedence over `qdrant_url` when set.
    #
    # It holds an exclusive lock on the directory, so exactly one process may
    # open it: fine for a demo or a test, not for more than one worker.
    qdrant_path: str | None = None

    # Local ONNX embeddings (384-dim). Chosen over an API embedder so ingestion
    # needs no second vendor key and re-indexing the corpus costs nothing.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Where the model weights are cached. Unset uses the library default, which
    # is a temp directory — fine locally, a re-download per restart in a container.
    embedding_cache_dir: str | None = None

    # Generation is the only thing that needs a vendor key: embeddings, BM25,
    # and reranking all run locally.
    openai_api_key: str | None = None
    # For Azure OpenAI, a proxy, or any OpenAI-compatible gateway. Unset uses
    # api.openai.com.
    openai_base_url: str | None = None

    # The model that writes the cited answer. It must support strict structured
    # outputs (gpt-4o-mini or newer) — the citation contract depends on
    # schema-constrained JSON, not on parsing prose out of a paragraph.
    answer_model: str = "gpt-4o-mini"
    # Reasoning tokens count against this too, so a tight budget truncates the
    # JSON rather than shortening the prose.
    answer_max_tokens: int = 8000

    # How many chunks an answer may draw on. Every one of them is sent to the
    # model on every query, so this trades recall against cost and latency.
    retrieval_top_k: int = 5

    # "hybrid" adds BM25 alongside the dense vectors and fuses the two rankings;
    # "dense" is embeddings only.
    #
    # Defaults to dense because that is what measured best here: on the corpus
    # in `eval/`, hybrid scored 0.658 MRR against dense's 0.702, at every
    # candidate depth tried. bge-small already ranks the exact-token questions
    # first, so BM25 found nothing new while adding noise on paraphrased ones.
    # That result is corpus-specific and the sample is small — re-run
    # `python -m eval.retrieval_eval` against your own documents before
    # trusting it. See `eval/README.md` for the caveats.
    retrieval_mode: Literal["dense", "hybrid"] = "dense"

    # BM25 term weights. Pulls a tokenizer and stopword list (~KB), not a model.
    sparse_model: str = "Qdrant/bm25"

    # How deep each arm of a hybrid search looks before the rankings are fused.
    # Must exceed retrieval_top_k to be worth anything: the results hybrid
    # exists to surface are the ones ranked well by one arm and poorly by the
    # other, and those only appear if both arms looked past k.
    retrieval_candidates: int = 30

    # A cross-encoder reads question and chunk together, so it catches the
    # near-miss chunk that sits close in vector space while answering a
    # neighbouring question. Costs one forward pass per candidate, which is why
    # it runs on the shortlist rather than the corpus.
    #
    # Off by default on the same evidence: it cost 0.05 MRR on the exact-token
    # questions and returned 0.005 on the paraphrased ones — a loss overall, and
    # ~80MB of model plus a forward pass per candidate to get it. Worth
    # re-testing on a corpus where dense retrieval is not already near the
    # ceiling.
    rerank_enabled: bool = False
    # ~80MB ONNX. The L-12 variant is slower and slightly better; the BAAI and
    # jina rerankers are stronger again at 1GB+.
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Measured in whitespace words, not BPE tokens: the splitter's default
    # tokenizer downloads its vocabulary on first use, and ingestion must not
    # depend on a network round-trip. 350 words is ~450-470 BPE tokens, which
    # stays inside the 512-token window bge-small truncates at, while still
    # holding a whole contract clause. The overlap keeps a clause that straddles
    # a boundary from being cut in half.
    chunk_size_words: int = 350
    chunk_overlap_words: int = 50

    # --- Agentic graph ------------------------------------------------------
    # Route -> retrieve -> grade -> (rewrite -> retry) -> generate -> critique,
    # instead of the single retrieve-then-generate pass. Costs three to five
    # model calls per question rather than one, so it is opt-in: see
    # `app/graph/pipeline.py` for what each node buys.
    agent_enabled: bool = False

    # Individually disableable, because they are not equally worth their cost on
    # every corpus. Routing earns its keep when users ask off-topic questions;
    # grading when retrieval returns near-misses; critique when someone acts on
    # the answer. Each is one extra model call per query.
    agent_route: bool = True
    agent_grade: bool = True
    agent_critique: bool = True

    # How many times a question may be rewritten and re-retrieved when grading
    # finds nothing relevant. Bounded because a loop that can retry forever
    # will: each retry is two more model calls, and the second rewrite of a
    # question is usually further from what was asked, not closer.
    agent_max_rewrites: int = 1

    # The model the graph's classification steps use. Routing, grading, and
    # rewriting are yes/no judgements and paraphrase, not analysis — running
    # them on the answering model is the easiest way to make the graph cost
    # several times what it should for no measurable gain. Point this at the
    # same model as ANSWER_MODEL if you would rather not run two.
    agent_model: str = "gpt-4o-mini"

    # --- Human review -------------------------------------------------------
    # Hold answers that trip a review rule (see `app/review/gate.py`) until a
    # reviewer approves them. The requester gets a review id instead of the
    # answer. The rules read signals the pipeline already produced, so this
    # costs no model calls — off only when something downstream does its own
    # review.
    review_enabled: bool = True

    # Critic confidence below this holds the answer. Only applies when the
    # agentic graph runs its critic. On the eval cases the critic never scored a
    # conflict above 0.40 or a clean answer below 0.80 (eval/REVIEW.md), so 0.7
    # sits in that gap. Re-check with `python -m eval.review_eval` on your own
    # documents: twelve cases are a sanity check, not a calibration.
    review_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    # Where the review queue lives. A file rather than memory, because a
    # pending review lost on restart is a promise to a requester silently
    # broken. Created on first use, not at startup.
    review_db_path: str = "data/reviews.sqlite3"

    # Bounds the memory a single upload can claim, since parsing loads the file.
    max_upload_bytes: int = 25 * 1024 * 1024

    # Paths that should never open a trace. Liveness probes fire constantly and
    # would otherwise dominate the trace volume.
    untraced_paths: tuple[str, ...] = ("/", "/health", "/health/live", "/health/ready")

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def generation_configured(self) -> bool:
        """Ingestion needs no LLM key; answering does. Checked up front so
        `/query` fails with a clear 503 instead of an SDK error mid-request."""
        return bool(self.openai_api_key)

    # Named in the 503 so the fix is obvious without reading the config.
    generation_key_variable: ClassVar[str] = "OPENAI_API_KEY"


@lru_cache
def get_settings() -> Settings:
    return Settings()
