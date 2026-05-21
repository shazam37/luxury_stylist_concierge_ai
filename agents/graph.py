"""
LangGraph Stylist Agent — Full Graph Implementation.

Graph Flow:
  START
    → intent_parser       (parse prompt → structured intent + query vector)
    → cache_check         (semantic similarity check)
    → [cache_hit?]
        YES → response_formatter (return cached, skip LLM)
        NO  → rag_retriever → fashion_reasoner → response_formatter
  END

All nodes are async. State is immutable between nodes (each returns partial updates).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

import structlog
from langgraph.graph import StateGraph, START, END

from agents.state import StylistState
from agents.nodes import (
    run_intent_parser,
    run_cache_check,
    run_rag_retriever,
    run_fashion_reasoner,
    run_response_formatter,
)

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
#  Conditional edge: cache hit or miss?
# ─────────────────────────────────────────────

def route_after_cache(state: StylistState) -> Literal["rag_retriever", "response_formatter"]:
    """Route based on cache hit/miss."""
    if state.get("cache_hit") and state.get("cached_response"):
        logger.info("graph.routing", decision="cache_hit → formatter")
        return "response_formatter"
    logger.info("graph.routing", decision="cache_miss → rag")
    return "rag_retriever"


# ─────────────────────────────────────────────
#  Cache-hit formatter (uses cached_response directly)
# ─────────────────────────────────────────────

async def format_cached_response(state: StylistState) -> dict:
    """When cache hits, load cached response directly into final_response."""
    cached = state.get("cached_response", {})
    from api.schemas import OutfitOption, TokenUsage

    # Re-hydrate OutfitOption from cached dict
    outfit_raw = cached.get("outfit", {})
    if isinstance(outfit_raw, dict):
        outfit = OutfitOption(**outfit_raw)
    else:
        outfit = outfit_raw

    alternatives_raw = cached.get("alternatives", [])
    alternatives = []
    for a in alternatives_raw:
        if isinstance(a, dict):
            alternatives.append(OutfitOption(**a))
        else:
            alternatives.append(a)

    token_raw = cached.get("token_usage", {})
    token_usage = TokenUsage(**token_raw) if isinstance(token_raw, dict) else token_raw

    return {
        "final_response": {
            "cache_hit": True,
            "parsed_intent": cached.get("parsed_intent", {}),
            "outfit": outfit,
            "alternatives": alternatives,
            "token_usage": token_usage,
            "agent_trace": (state.get("agent_trace", []) + ["served_from_cache"]),
            "model_used": cached.get("model_used", "cached"),
        }
    }


# ─────────────────────────────────────────────
#  Load wardrobe from DB (pre-graph step)
# ─────────────────────────────────────────────

async def load_wardrobe(state: StylistState) -> dict:
    """Load user's wardrobe items from Postgres if user_id is provided."""
    user_id = state.get("user_id")
    if not user_id:
        return {"wardrobe_items": []}

    try:
        from db.postgres import get_db_session
        from db.models import WardrobeItem
        from sqlalchemy import select

        async with get_db_session() as db:
            result = await db.execute(
                select(WardrobeItem).where(WardrobeItem.user_id == user_id).limit(20)
            )
            items = result.scalars().all()
            wardrobe = [
                {
                    "name": i.name,
                    "category": i.category,
                    "color": i.color,
                    "brand": i.brand,
                }
                for i in items
            ]
        logger.info("graph.wardrobe_loaded", count=len(wardrobe))
        return {"wardrobe_items": wardrobe}
    except Exception as e:
        logger.warning("graph.wardrobe_load_failed", error=str(e))
        return {"wardrobe_items": []}


# ─────────────────────────────────────────────
#  Build the graph
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph stylist agent."""
    graph = StateGraph(StylistState)

    # Register nodes
    graph.add_node("load_wardrobe",       load_wardrobe)
    graph.add_node("intent_parser",       run_intent_parser)
    graph.add_node("cache_check",         run_cache_check)
    graph.add_node("rag_retriever",       run_rag_retriever)
    graph.add_node("fashion_reasoner",    run_fashion_reasoner)
    graph.add_node("response_formatter",  run_response_formatter)
    graph.add_node("cache_formatter",     format_cached_response)

    # Edges
    graph.add_edge(START,              "load_wardrobe")
    graph.add_edge("load_wardrobe",    "intent_parser")
    graph.add_edge("intent_parser",    "cache_check")

    # Conditional: cache hit → cache_formatter, miss → rag
    graph.add_conditional_edges(
        "cache_check",
        route_after_cache,
        {
            "rag_retriever":      "rag_retriever",
            "response_formatter": "cache_formatter",
        },
    )

    graph.add_edge("rag_retriever",      "fashion_reasoner")
    graph.add_edge("fashion_reasoner",   "response_formatter")
    graph.add_edge("response_formatter", END)
    graph.add_edge("cache_formatter",    END)

    return graph.compile()


# Singleton compiled graph
_compiled_graph = None

def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ─────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────

async def run_stylist_agent(
    prompt: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    gender: Optional[str] = None,
    budget: Optional[Any] = None,
    style_preference: Optional[str] = None,
    include_alternatives: bool = True,
    include_accessories: bool = False,
    db: Optional[Any] = None,
) -> dict:
    """
    Main entry point. Runs the full LangGraph pipeline.
    Returns a dict matching the StyleMeResponse schema.
    """
    # Build initial state
    initial_state: StylistState = {
        "user_prompt": prompt,
        "user_id": user_id,
        "session_id": session_id,
        "gender": gender,
        "budget_min": float(budget.min_price) if budget and budget.min_price else None,
        "budget_max": float(budget.max_price) if budget and budget.max_price else None,
        "style_preference": style_preference,
        "include_alternatives": include_alternatives,
        "include_accessories": include_accessories,
        "wardrobe_items": [],
        "parsed_intent": {},
        "query_vector": [],
        "cache_hit": False,
        "cached_response": None,
        "retrieved_items": {},
        "ranked_items": {},
        "outfit_primary": {},
        "outfit_alternatives": [],
        "stylist_note": "",
        "final_response": {},
        "token_usage": {},
        "agent_trace": [],
        "model_used": "",
        "errors": [],
    }

    graph = get_graph()

    logger.info("agent.starting", prompt_preview=prompt[:80])

    try:
        final_state = await graph.ainvoke(initial_state)
        result = final_state.get("final_response", {})
        logger.info(
            "agent.complete",
            cache_hit=result.get("cache_hit"),
            trace=result.get("agent_trace"),
        )
        return result
    except Exception as e:
        logger.exception("agent.fatal_error", error=str(e))
        from api.schemas import OutfitOption, TokenUsage
        return {
            "cache_hit": False,
            "parsed_intent": {},
            "outfit": OutfitOption(stylist_note=f"Agent error: {str(e)}", total_price=0.0),
            "alternatives": [],
            "token_usage": TokenUsage(),
            "agent_trace": ["fatal_error"],
            "model_used": "error",
        }