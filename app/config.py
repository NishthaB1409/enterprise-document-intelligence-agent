"""Application settings, loaded from the environment (and `.env` in development)."""

from functools import lru_cache
from typing import Literal

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

    # Which vendor writes the answer. Only generation is affected — embeddings
    # are local either way, so switching costs nothing and re-indexes nothing.
    llm_provider: Literal["anthropic", "openai"] = "anthropic"

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    # For Azure OpenAI, a proxy, or any OpenAI-compatible gateway. Unset uses
    # api.openai.com.
    openai_base_url: str | None = None

    # Unset resolves to the provider's default (see `resolved_answer_model`),
    # so switching providers doesn't strand a model name that belongs to the
    # other one.
    answer_model: str | None = None
    # Anthropic only: low | medium | high | xhigh | max. Reading a handful of
    # retrieved chunks and citing them is not a reasoning-heavy task; medium
    # keeps /query interactive. Raise it if answers start missing cross-chunk
    # implications.
    answer_effort: str = "medium"
    # Thinking or reasoning tokens count against this too, so a tight budget
    # truncates the JSON rather than shortening the prose.
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
        `/query` fails with a clear 503 instead of an SDK error mid-request.

        Only the *selected* provider's key counts — holding an unused key for
        the other vendor must not make an unconfigured deployment look healthy.
        """
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        return bool(self.anthropic_api_key)

    @property
    def generation_key_variable(self) -> str:
        """Named in the 503 so the fix is obvious without reading the config."""
        return "OPENAI_API_KEY" if self.llm_provider == "openai" else "ANTHROPIC_API_KEY"

    @property
    def resolved_answer_model(self) -> str:
        if self.answer_model:
            return self.answer_model
        # gpt-4o-mini is the cheapest OpenAI model that supports strict
        # structured outputs, which this pipeline depends on. Override
        # ANSWER_MODEL for a stronger one.
        return "gpt-4o-mini" if self.llm_provider == "openai" else "claude-opus-5"


@lru_cache
def get_settings() -> Settings:
    return Settings()
