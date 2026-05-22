"""
POST /api/v1/style-me — Core styling endpoint.
Full agentic implementation added in Phase 3 (LangGraph agent).
This file registers the router; logic lives in agents/graph.py.
"""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import StyleMeRequest, StyleMeResponse, FeedbackRequest, OutfitOption, TokenUsage
from db.postgres import get_db

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post(
    "/style-me",
    response_model=StyleMeResponse,
    summary="Get an AI-powered outfit recommendation",
    description="""
Submit a natural language styling request and receive a complete outfit recommendation.

**Examples:**
- "I have dark navy chinos. What t-shirt and shoes for a summer yacht party?"
- "Suggest a full casual Friday office outfit, budget under $200"
- "I own a white linen shirt — complete the look for a beach wedding"

The agent will:
1. Parse your intent and context
2. Check the semantic cache (fast return if similar query exists)  
3. Query the fashion catalog using RAG
4. Apply fashion rules and color theory
5. Return a primary outfit + 2 alternatives with a luxurious Stylist Note
    """,
)
async def style_me(
    request: StyleMeRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    logger.info("style_me.request", request_id=request_id, prompt=request.prompt[:80])

    try:
        # Import here to avoid circular imports at module load time
        from agents.graph import run_stylist_agent

        result = await run_stylist_agent(
            prompt=request.prompt,
            user_id=request.user_id,
            session_id=request.session_id,
            gender=request.gender,
            budget=request.budget,
            style_preference=request.style_preference,
            include_alternatives=request.include_alternatives,
            include_accessories=request.include_accessories,
            db=db,
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Persist request log in background (non-blocking)
        background_tasks.add_task(
            _log_style_request,
            db=db,
            request_id=request_id,
            request=request,
            result=result,
            latency_ms=latency_ms,
        )

        return StyleMeResponse(
            request_id=request_id,
            user_prompt=request.prompt,
            cache_hit=result.get("cache_hit", False),
            parsed_intent=result.get("parsed_intent", {}),
            outfit=result["outfit"],
            alternatives=result.get("alternatives", []),
            budget_summary=result.get("budget_summary"),
            token_usage=result.get("token_usage", TokenUsage()),
            agent_trace=result.get("agent_trace", []),
            latency_ms=latency_ms,
            model_used=result.get("model_used"),
        )

    except Exception as e:
        logger.exception("style_me.error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@router.post(
    "/style-me/feedback",
    summary="Submit feedback on a styling recommendation",
)
async def submit_feedback(
    feedback: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """Rate a styling recommendation (1-5 stars)."""
    from sqlalchemy import update
    from db.models import StyleRequest

    await db.execute(
        update(StyleRequest)
        .where(StyleRequest.id == feedback.request_id)
        .values(feedback_score=feedback.score)
    )
    await db.commit()

    logger.info("feedback.received", request_id=feedback.request_id, score=feedback.score)
    return {"status": "ok", "message": "Feedback recorded"}


async def _log_style_request(db, request_id, request, result, latency_ms):
    """Background task: persist the style request to Postgres."""
    try:
        from db.models import StyleRequest
        record = StyleRequest(
            id=request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            prompt=request.prompt,
            parsed_intent=result.get("parsed_intent", {}),
            outfit_response={"outfit": result.get("outfit", {}), "alternatives": result.get("alternatives", [])},
            cache_hit=result.get("cache_hit", False),
            token_usage=result.get("token_usage", {}).model_dump() if hasattr(result.get("token_usage", {}), "model_dump") else {},
            agent_trace=result.get("agent_trace", []),
            latency_ms=latency_ms,
        )
        db.add(record)
        await db.commit()
    except Exception as e:
        logger.warning("style_request.log_failed", error=str(e))