"""
Node 2: Cache Manager
Checks semantic cache before running the full pipeline.
On hit: returns cached response immediately. On miss: continues.
"""
from __future__ import annotations
import structlog
from agents.state import StylistState

logger = structlog.get_logger(__name__)


async def run_cache_check(state: StylistState) -> dict:
    """Check semantic cache for a similar prior request."""
    logger.info("node.cache_check.start")

    vector = state.get("query_vector", [])
    if not vector:
        return {
            "cache_hit": False,
            "agent_trace": state.get("agent_trace", []) + ["cache_skip_no_vector"],
        }

    try:
        from cache.semantic_cache import get_semantic_cache
        cache = get_semantic_cache()
        cached = await cache.get(vector)

        if cached:
            logger.info("node.cache_check.hit")
            return {
                "cache_hit": True,
                "cached_response": cached,
                "agent_trace": state.get("agent_trace", []) + ["cache_hit"],
            }

        logger.info("node.cache_check.miss")
        return {
            "cache_hit": False,
            "agent_trace": state.get("agent_trace", []) + ["cache_miss"],
        }

    except Exception as e:
        logger.warning("node.cache_check.error", error=str(e))
        return {
            "cache_hit": False,
            "agent_trace": state.get("agent_trace", []) + ["cache_error"],
        }

