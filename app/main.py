"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import echo, health, ingest, query
from app.config import Settings, get_settings
from app.observability.langfuse_client import init_langfuse, shutdown_langfuse
from app.observability.middleware import TracingMiddleware
from app.services import Services, build_services

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_langfuse(app.state.settings)

    services: Services = app.state.services
    try:
        services.store.ensure_collection(services.embedder.dimension)
    except Exception:
        # Non-fatal, deliberately: the vector store may still be starting up
        # beside us, and the first ingest creates the collection anyway. Failing
        # to boot here would mean a Qdrant restart takes the API down with it.
        logger.exception("Could not reach the vector store at startup.")

    yield
    # Spans are batched; without this flush the final requests before a restart
    # never reach Langfuse.
    shutdown_langfuse()


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """`services` is an injection point for tests, which supply in-memory
    stand-ins for the embedder, vector store, and model."""
    settings = settings or get_settings()

    app = FastAPI(
        title="Enterprise Document-Intelligence Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.services = services or build_services(settings)

    app.add_middleware(TracingMiddleware, settings=settings)

    app.include_router(health.router)
    app.include_router(echo.router, prefix=API_PREFIX)
    app.include_router(ingest.router, prefix=API_PREFIX)
    app.include_router(query.router, prefix=API_PREFIX)

    return app


app = create_app()
