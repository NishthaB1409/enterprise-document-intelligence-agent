"""Langfuse client lifecycle.

The rest of the codebase never constructs a Langfuse client directly — it calls
`langfuse.get_client()` (or uses `@observe`), which resolves the singleton this
module initialises at startup. That keeps instrumentation decoupled from config:
a module deep in the retrieval layer can open a span without importing settings.

When credentials are absent the client is constructed with `tracing_enabled=False`.
Spans still get created, they just become no-ops — so the app boots and behaves
identically with or without Langfuse, and there is no `if tracing_enabled:` branch
scattered through the business logic.
"""

import logging

from langfuse import Langfuse, get_client

from app.config import Settings

logger = logging.getLogger(__name__)


def init_langfuse(settings: Settings) -> Langfuse:
    """Construct the process-wide Langfuse client. Call once, at startup."""
    if not settings.langfuse_configured:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set - tracing is disabled. "
            "Spans will be created but discarded."
        )

    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        environment=settings.environment,
        release=settings.release,
        tracing_enabled=settings.langfuse_configured,
    )

    if settings.langfuse_configured:
        # Surfaces bad credentials at boot instead of silently dropping every span.
        # Non-fatal: a Langfuse outage must not take the API down with it.
        try:
            if client.auth_check():
                logger.info("Langfuse tracing enabled (host=%s)", settings.langfuse_host)
            else:
                logger.error(
                    "Langfuse credentials rejected by %s - traces will not be recorded.",
                    settings.langfuse_host,
                )
        except Exception:
            logger.exception(
                "Could not reach Langfuse at %s - traces may not be recorded.",
                settings.langfuse_host,
            )

    return client


def shutdown_langfuse() -> None:
    """Flush buffered spans and stop the exporter. Call once, at shutdown.

    Spans are batched, so without this the last requests before a deploy or a
    container restart are lost.
    """
    try:
        get_client().shutdown()
    except Exception:
        logger.exception("Error shutting down the Langfuse client.")
