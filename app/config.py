"""Application settings, loaded from the environment (and `.env` in development)."""

from functools import lru_cache

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

    # Local ONNX embeddings (384-dim). Chosen over an API embedder so ingestion
    # needs no second vendor key and re-indexing the corpus costs nothing.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Where the model weights are cached. Unset uses the library default, which
    # is a temp directory — fine locally, a re-download per restart in a container.
    embedding_cache_dir: str | None = None

    anthropic_api_key: str | None = None
    answer_model: str = "claude-opus-5"
    # low | medium | high | xhigh | max. Reading a handful of retrieved chunks
    # and citing them is not a reasoning-heavy task; medium keeps /query
    # interactive. Raise it if answers start missing cross-chunk implications.
    answer_effort: str = "medium"
    # Thinking counts against this too, so a tight budget truncates the JSON
    # rather than shortening the prose.
    answer_max_tokens: int = 8000

    # How many chunks an answer may draw on. Every one of them is sent to the
    # model on every query, so this trades recall against cost and latency.
    retrieval_top_k: int = 5

    # Measured in whitespace words, not BPE tokens: the splitter's default
    # tokenizer downloads its vocabulary on first use, and ingestion must not
    # depend on a network round-trip. 350 words is ~450-470 BPE tokens, which
    # stays inside the 512-token window bge-small truncates at, while still
    # holding a whole contract clause. The overlap keeps a clause that straddles
    # a boundary from being cut in half.
    chunk_size_words: int = 350
    chunk_overlap_words: int = 50

    # Bounds the memory a single upload can claim, since parsing loads the file.
    max_upload_bytes: int = 25 * 1024 * 1024

    # Paths that should never open a trace. Liveness probes fire constantly and
    # would otherwise dominate the trace volume.
    untraced_paths: tuple[str, ...] = ("/health", "/health/live", "/health/ready")

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def generation_configured(self) -> bool:
        """Ingestion needs no LLM key; answering does. Checked up front so
        `/query` fails with a clear 503 instead of an SDK error mid-request."""
        return bool(self.anthropic_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
