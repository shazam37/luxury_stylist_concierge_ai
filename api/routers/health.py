"""
Health and readiness endpoints.
GET /health  → liveness probe
GET /ready   → readiness probe (checks all services)
GET /info    → app metadata
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter

from api.schemas import HealthResponse
from config.settings import get_settings

router = APIRouter(tags=["Health"])
logger = structlog.get_logger(__name__)


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health():
    """Basic liveness check — returns 200 if the app is running."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        services={"app": "ok"},
    )


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
async def readiness():
    """
    Readiness check — verifies all downstream services are reachable.
    Returns 200 only if Qdrant, Postgres, and Redis are all healthy.
    """
    settings = get_settings()
    services = {}

    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=3)
        client.get_collections()
        services["qdrant"] = "ok"
    except Exception as e:
        services["qdrant"] = f"error: {str(e)[:50]}"

    # Check Postgres
    try:
        from db.postgres import get_engine
        import sqlalchemy
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
        services["postgres"] = "ok"
    except Exception as e:
        services["postgres"] = f"error: {str(e)[:50]}"

    # Check Redis
    try:
        import redis.asyncio as aioredis
        r = await aioredis.from_url(settings.redis_url)
        await r.ping()
        await r.aclose()
        services["redis"] = "ok"
    except Exception as e:
        services["redis"] = f"error: {str(e)[:50]}"

    overall = "ok" if all(v == "ok" for v in services.values()) else "degraded"
    services["app"] = "ok"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        services=services,
    )


@router.get("/info", summary="App info")
async def info():
    """Returns app metadata including active LLM provider and model."""
    settings = get_settings()
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "qdrant_collection": settings.qdrant_collection,
    }