"""Liveness/readiness probes. Excluded from tracing via `Settings.untraced_paths`."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    tracing_enabled: bool


@router.get("/health")
def health() -> HealthResponse:
    from app.config import get_settings

    return HealthResponse(status="ok", tracing_enabled=get_settings().langfuse_configured)
