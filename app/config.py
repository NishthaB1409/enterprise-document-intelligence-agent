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

    # Paths that should never open a trace. Liveness probes fire constantly and
    # would otherwise dominate the trace volume.
    untraced_paths: tuple[str, ...] = ("/health", "/health/live", "/health/ready")

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
