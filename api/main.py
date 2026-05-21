"""
Stylist Concierge — FastAPI Application Entry Point

Startup sequence:
  1. Init structured logging
  2. Connect to PostgreSQL
  3. Ensure Qdrant collection exists
  4. Mount all routers
  5. Apply middleware (CORS, rate limiting, request logging)
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings
from db.postgres import init_db, close_db
from embeddings.indexer import get_qdrant_manager
from api.routers import health

# ── Lazy router imports (avoid circular deps at parse time) ───
# Routers are imported here after all modules initialise.

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
#  Logging Setup
# ─────────────────────────────────────────────

def configure_logging(debug: bool = False) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if debug else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            10 if debug else 20  # DEBUG : INFO
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ─────────────────────────────────────────────
#  Lifespan (startup / shutdown)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle — runs startup then yields, then teardown."""
    settings = get_settings()
    configure_logging(debug=settings.debug)
    logger.info("app.starting", name=settings.app_name, version=settings.app_version)

    # 1. PostgreSQL
    try:
        await init_db()
        logger.info("startup.postgres_ok")
    except Exception as e:
        logger.error("startup.postgres_failed", error=str(e))

    # 2. Qdrant
    try:
        qdrant = get_qdrant_manager()
        await qdrant.ensure_collection()
        info = await qdrant.get_collection_info()
        logger.info("startup.qdrant_ok", **info)
    except Exception as e:
        logger.error("startup.qdrant_failed", error=str(e))

    # 3. Redis (cache)
    try:
        from cache.semantic_cache import get_semantic_cache
        cache = get_semantic_cache()
        await cache._get_redis()
        logger.info("startup.cache_ok")
    except Exception as e:
        logger.warning("startup.cache_warning", error=str(e))

    logger.info("app.ready", host=settings.app_host, port=settings.app_port)
    yield

    # Teardown
    await close_db()
    logger.info("app.shutdown")


# ─────────────────────────────────────────────
#  App Factory
# ─────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
## Luxury AI Stylist Concierge API

An agentic fashion recommendation system powered by LangGraph + RAG.

### Features
- **POST /api/v1/style-me** — Core styling endpoint: send a natural language prompt, get a complete outfit recommendation
- **Catalog Management** — Browse, search, and filter the scraped fashion catalog
- **Wardrobe** — Save and manage items you already own (referenced by the agent)
- **Scraper Control** — Trigger on-demand scrape jobs
- **Semantic Cache** — Near-duplicate prompts return instantly without LLM calls

### Model-Agnostic
Switch between Groq, OpenAI, Anthropic, or Google by changing `LLM_PROVIDER` in `.env`.
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else ["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID + Timing Middleware ─────────
    @app.middleware("http")
    async def request_middleware(request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception("request.unhandled_error", path=request.url.path, error=str(exc))
            response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id},
            )

        latency_ms = int((time.perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{latency_ms}ms"

        logger.info(
            "request.complete",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=latency_ms,
        )
        return response

    # ── Routers ────────────────────────────────
    app.include_router(health.router)

    # Import and mount feature routers
    from api.routers import style, catalog, wardrobe, scraper
    app.include_router(style.router,    prefix="/api/v1", tags=["Styling"])
    app.include_router(catalog.router,  prefix="/api/v1", tags=["Catalog"])
    app.include_router(wardrobe.router, prefix="/api/v1", tags=["Wardrobe"])
    app.include_router(scraper.router,  prefix="/api/v1", tags=["Scraper"])

    # ── Global exception handler ───────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("unhandled_exception", path=str(request.url), error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "An unexpected error occurred", "detail": str(exc)},
        )

    return app


# ─────────────────────────────────────────────
#  App instance
# ─────────────────────────────────────────────

app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )