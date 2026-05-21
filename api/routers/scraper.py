"""
/api/v1/scraper — On-demand scrape job control.
POST /api/v1/scraper/trigger   → kick off a scrape job (background)
POST /api/v1/scraper/seed      → seed catalog with mock data (dev/testing)
GET  /api/v1/scraper/jobs      → list recent jobs
GET  /api/v1/scraper/jobs/{id} → job status
"""

from __future__ import annotations

import asyncio
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import ScraperTriggerRequest, ScraperJobResponse, ScraperJobStatusResponse
from db.postgres import get_db
from db.models import ScrapeJob

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post(
    "/scraper/trigger",
    response_model=ScraperJobResponse,
    summary="Trigger a scrape job",
    description=(
        "Kick off a background scraping job for one or all sources. "
        "Returns a job ID to poll for status via GET /api/v1/scraper/jobs/{job_id}."
    ),
    status_code=202,
)
async def trigger_scrape(
    request: ScraperTriggerRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Trigger a background scrape job for zara | hm | myntra | all."""
    job = ScrapeJob(
        source=request.source,
        status="pending",
        triggered_by="api",
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    job_id = str(job.id)
    logger.info("scraper.job_created", job_id=job_id, source=request.source)

    background_tasks.add_task(
        _run_scrape_job,
        job_id=job_id,
        source=request.source,
        reindex=request.reindex,
    )

    return ScraperJobResponse(
        job_id=job_id,
        source=request.source,
        status="pending",
        message=(
            f"Scrape job queued for source: '{request.source}'. "
            f"Poll /api/v1/scraper/jobs/{job_id} for status."
        ),
    )


@router.post(
    "/scraper/seed",
    summary="Seed catalog with mock data (dev/testing)",
    description=(
        "Populates the catalog with realistic mock fashion items without running real scrapers. "
        "Useful for development, demos, and testing. Runs as a background task."
    ),
    status_code=202,
)
async def seed_catalog(
    background_tasks: BackgroundTasks,
    count: int = Query(default=120, ge=10, le=500, description="Number of mock items to generate"),
):
    """Seed the vector DB and Postgres with realistic mock fashion items."""
    background_tasks.add_task(_do_seed, count=count)
    return {
        "status": "seeding",
        "count_requested": count,
        "message": (
            f"Seeding {count} mock items in background. "
            "Check GET /api/v1/catalog/stats for progress."
        ),
    }


@router.get("/scraper/jobs", summary="List recent scrape jobs")
async def list_jobs(
    limit: int = Query(default=10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """List the most recent scrape/seed jobs with their status."""
    result = await db.execute(
        select(ScrapeJob).order_by(ScrapeJob.created_at.desc()).limit(limit)
    )
    jobs = result.scalars().all()
    return {"jobs": [_to_status(j) for j in jobs], "count": len(jobs)}


@router.get(
    "/scraper/jobs/{job_id}",
    response_model=ScraperJobStatusResponse,
    summary="Get scrape job status",
)
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    """Poll a specific scrape job for its current status and item counts."""
    result = await db.execute(select(ScrapeJob).where(ScrapeJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _to_status(job)


# ─────────────────────────────────────────────
#  Background task implementations
# ─────────────────────────────────────────────

async def _run_scrape_job(job_id: str, source: str, reindex: bool) -> None:
    """Background task: run the real scraper pipeline and update job record."""
    from db.postgres import get_db_session
    from db.models import ScrapeJob
    from datetime import datetime, timezone

    async with get_db_session() as db:
        result = await db.execute(select(ScrapeJob).where(ScrapeJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        await db.commit()

    try:
        from scraper.pipeline import run_pipeline
        stats = await run_pipeline(source=source, reindex=reindex)

        async with get_db_session() as db:
            result = await db.execute(select(ScrapeJob).where(ScrapeJob.id == job_id))
            job = result.scalar_one_or_none()
            if job:
                job.status = "completed"
                job.items_scraped = stats.get("scraped", 0)
                job.items_indexed = stats.get("indexed", 0)
                job.errors = stats.get("errors", [])
                job.completed_at = datetime.now(timezone.utc)
                await db.commit()

        logger.info("scraper.job_completed", job_id=job_id, stats=stats)

    except Exception as e:
        logger.exception("scraper.job_failed", job_id=job_id, error=str(e))
        async with get_db_session() as db:
            result = await db.execute(select(ScrapeJob).where(ScrapeJob.id == job_id))
            job = result.scalar_one_or_none()
            if job:
                job.status = "failed"
                job.errors = [str(e)]
                from datetime import datetime, timezone
                job.completed_at = datetime.now(timezone.utc)
                await db.commit()


async def _do_seed(count: int) -> None:
    """Background task: seed the catalog with mock data."""
    try:
        from scraper.pipeline import seed_mock_catalog
        result = await seed_mock_catalog(count=count)
        logger.info("seed.complete", **{k: v for k, v in result.items() if k != "errors"})
    except Exception as e:
        logger.exception("seed.failed", error=str(e))


# ─────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────

def _to_status(job: ScrapeJob) -> ScraperJobStatusResponse:
    return ScraperJobStatusResponse(
        job_id=str(job.id),
        source=job.source or "unknown",
        status=job.status or "unknown",
        items_scraped=job.items_scraped or 0,
        items_indexed=job.items_indexed or 0,
        errors=[str(e) for e in (job.errors or [])],
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )