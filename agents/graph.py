"""
LangGraph Stylist Agent — Full Graph.

Updated flow with query_planner and price_optimizer:

  START
    → load_wardrobe        fetch owned items from Postgres
    → intent_parser        LLM: prompt → ParsedIntent + embed query vector
    → cache_check          cosine similarity vs Redis cache
    │
    ├── [HIT]  → cache_formatter → END
    │
    └── [MISS] → query_planner      deterministic: ParsedIntent → QueryPlan
                 → rag_retriever    execute QueryPlan against Qdrant (per-category)
                 → fashion_reasoner LLM: apply fashion rules → select outfit
                 → price_optimizer  deterministic: budget compliance + value tagging
                 → response_formatter schema assembly + cache write
                 → END
"""
from __future__ import annotations

from typing import Any, Literal, Optional

import structlog
from langgraph.graph import StateGraph, START, END

from agents.state import StylistState
from agents.nodes import (
    run_intent_parser,
    run_cache_check,
    run_query_planner,
    run_rag_retriever,
    run_fashion_reasoner,
    run_price_optimizer,
    run_response_formatter,
)

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
#  Conditional edge: cache hit or miss?
# ─────────────────────────────────────────────

def route_after_cache(state: StylistState) -> Literal["query_planner", "cache_formatter"]:
    if state.get("cache_hit") and state.get("cached_response"):
        logger.info("graph.routing", decision="cache_hit → formatter")
        return "cache_formatter"
    logger.info("graph.routing", decision="cache_miss → query_planner")
    return "query_planner"


# ─────────────────────────────────────────────
#  Cache-hit formatter
# ─────────────────────────────────────────────

async def format_cached_response(state: StylistState) -> dict:
    """Serve a cached response directly — skips all LLM nodes."""
    cached = state.get("cached_response", {})
    from api.schemas import OutfitOption, TokenUsage, BudgetSummary

    def _hydrate_outfit(raw):
        return OutfitOption(**raw) if isinstance(raw, dict) else raw

    outfit = _hydrate_outfit(cached.get("outfit", {}))
    alternatives = [_hydrate_outfit(a) for a in cached.get("alternatives", [])]
    token_raw = cached.get("token_usage", {})
    token_usage = TokenUsage(**token_raw) if isinstance(token_raw, dict) else token_raw

    budget_raw = cached.get("budget_summary")
    budget_summary = BudgetSummary(**budget_raw) if isinstance(budget_raw, dict) else None

    return {
        "final_response": {
            "cache_hit": True,
            "parsed_intent": cached.get("parsed_intent", {}),
            "outfit": outfit,
            "alternatives": alternatives,
            "budget_summary": budget_summary,
            "token_usage": token_usage,
            "agent_trace": state.get("agent_trace", []) + ["served_from_cache"],
            "model_used": cached.get("model_used", "cached"),
        }
    }


# ─────────────────────────────────────────────
#  Wardrobe loader (pre-graph)
# ─────────────────────────────────────────────

async def load_wardrobe(state: StylistState) -> dict:
    """Load user's saved wardrobe items from Postgres."""
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
                {"name": i.name, "category": i.category, "color": i.color, "brand": i.brand}
                for i in items
            ]
        logger.info("graph.wardrobe_loaded", count=len(wardrobe))
        return {"wardrobe_items": wardrobe}
    except Exception as e:
        logger.warning("graph.wardrobe_load_failed", error=str(e))
        return {"wardrobe_items": []}


# ─────────────────────────────────────────────
#  Build & compile the graph
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(StylistState)

    # ── Nodes ──────────────────────────────────────────────────────
    graph.add_node("load_wardrobe",      load_wardrobe)
    graph.add_node("intent_parser",      run_intent_parser)
    graph.add_node("cache_check",        run_cache_check)
    graph.add_node("query_planner",      run_query_planner)      # NEW
    graph.add_node("rag_retriever",      run_rag_retriever)
    graph.add_node("fashion_reasoner",   run_fashion_reasoner)
    graph.add_node("price_optimizer",    run_price_optimizer)    # NEW
    graph.add_node("response_formatter", run_response_formatter)
    graph.add_node("cache_formatter",    format_cached_response)

    # ── Edges ───────────────────────────────────────────────────────
    graph.add_edge(START,                "load_wardrobe")
    graph.add_edge("load_wardrobe",      "intent_parser")
    graph.add_edge("intent_parser",      "cache_check")

    graph.add_conditional_edges(
        "cache_check",
        route_after_cache,
        {
            "query_planner":  "query_planner",
            "cache_formatter": "cache_formatter",
        },
    )

    graph.add_edge("query_planner",      "rag_retriever")
    graph.add_edge("rag_retriever",      "fashion_reasoner")
    graph.add_edge("fashion_reasoner",   "price_optimizer")
    graph.add_edge("price_optimizer",    "response_formatter")
    graph.add_edge("response_formatter", END)
    graph.add_edge("cache_formatter",    END)

    return graph.compile()


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
        "query_plan": {},
        "cache_hit": False,
        "cached_response": None,
        "retrieved_items": {},
        "ranked_items": {},
        "outfit_primary": {},
        "outfit_alternatives": [],
        "budget_summary": {},
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
        logger.info("agent.complete", cache_hit=result.get("cache_hit"), trace=result.get("agent_trace"))
        return result
    except Exception as e:
        logger.exception("agent.fatal_error", error=str(e))
        from api.schemas import OutfitOption, TokenUsage
        return {
            "cache_hit": False,
            "parsed_intent": {},
            "outfit": OutfitOption(stylist_note=f"Agent error: {str(e)}", total_price=0.0),
            "alternatives": [],
            "budget_summary": None,
            "token_usage": TokenUsage(),
            "agent_trace": ["fatal_error"],
            "model_used": "error",
        }