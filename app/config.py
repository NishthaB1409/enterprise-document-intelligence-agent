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


@lru_cache
def get_settings() -> Settings:
    return Settings()
