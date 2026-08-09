"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import echo, health
from app.config import Settings, get_settings
from app.observability.langfuse_client import init_langfuse, shutdown_langfuse
from app.observability.middleware import TracingMiddleware

logging.basicConfig(level=logging.INFO)

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_langfuse(app.state.settings)
    yield
    # Spans are batched; without this flush the final requests before a restart
    # never reach Langfuse.
    shutdown_langfuse()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Enterprise Document-Intelligence Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(TracingMiddleware, settings=settings)

    app.include_router(health.router)
    app.include_router(echo.router, prefix=API_PREFIX)

    return app


app = create_app()
