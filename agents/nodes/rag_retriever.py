"""
Node 3: RAG Retriever

Consumes the QueryPlan produced by query_planner and executes the
multi-category semantic search against Qdrant. Pure execution — zero
retrieval-strategy decisions made here.

Changes from original:
  - Reads categories, limits, score_threshold from query_plan
  - Uses per-category query strings from query_plan.category_queries
    (embedded on-the-fly) for sharper per-category retrieval
  - Falls back to shared query_vector if per-category embedding fails
  - Still applies gender + price filters from parsed_intent / state
"""
from __future__ import annotations

import asyncio
import structlog
from agents.state import StylistState

logger = structlog.get_logger(__name__)


async def run_rag_retriever(state: StylistState) -> dict:
    """Execute retrieval plan against Qdrant. Returns retrieved_items dict."""
    logger.info("node.rag_retriever.start")

    plan = state.get("query_plan") or {}
    parsed = state.get("parsed_intent") or {}
    shared_vector = state.get("query_vector") or []

    # ── Determine categories and limits from plan (with safe fallback) ────────
    categories = plan.get("categories") or parsed.get("requested_categories") or ["top", "shoes"]
    limits = plan.get("limits") or {cat: 8 for cat in categories}
    score_threshold = plan.get("score_threshold") or 0.0
    category_queries = plan.get("category_queries") or {}

    if not shared_vector:
        return {
            "retrieved_items": {},
            "errors": state.get("errors", []) + ["No query vector for RAG retrieval"],
            "agent_trace": state.get("agent_trace", []) + ["rag_skip_no_vector"],
        }

    from embeddings.indexer import get_qdrant_manager
    from embeddings.embedder import get_embedder
    qdrant = get_qdrant_manager()
    embedder = get_embedder()

    # ── Build filters ─────────────────────────────────────────────────────────
    filters: dict = {}
    gender = parsed.get("gender") or state.get("gender")
    if gender and gender != "unisex":
        filters["gender"] = gender

    budget_max = parsed.get("budget_max") or state.get("budget_max")
    budget_min = parsed.get("budget_min") or state.get("budget_min")
    if budget_max:
        filters["max_price"] = float(budget_max)
    if budget_min:
        filters["min_price"] = float(budget_min)

    # ── Embed per-category queries concurrently ───────────────────────────────
    # Each category gets its own targeted query string for higher precision.
    async def _embed_cat(cat: str) -> tuple[str, list[float]]:
        q = category_queries.get(cat)
        if q:
            try:
                vec = await embedder.embed_query(q)
                return cat, vec
            except Exception:
                pass
        return cat, shared_vector  # fallback to shared vector

    embed_results = await asyncio.gather(*[_embed_cat(cat) for cat in categories])
    cat_vectors: dict[str, list[float]] = dict(embed_results)

    # ── Execute per-category searches concurrently ────────────────────────────
    async def _search_cat(cat: str) -> tuple[str, list[dict]]:
        vec = cat_vectors.get(cat, shared_vector)
        limit = limits.get(cat, 8)
        try:
            results = await qdrant.search(
                query_vector=vec,
                category=cat,
                gender=filters.get("gender"),
                min_price=filters.get("min_price"),
                max_price=filters.get("max_price"),
                limit=limit,
                score_threshold=score_threshold,
            )
            return cat, results
        except Exception as e:
            logger.warning("rag_retriever.cat_failed", category=cat, error=str(e)[:80])
            return cat, []

    try:
        search_results = await asyncio.gather(*[_search_cat(cat) for cat in categories])
        retrieved: dict[str, list[dict]] = dict(search_results)

        total = sum(len(v) for v in retrieved.values())
        logger.info(
            "node.rag_retriever.done",
            categories=list(retrieved.keys()),
            per_cat={cat: len(items) for cat, items in retrieved.items()},
            total=total,
        )

        return {
            "retrieved_items": retrieved,
            "agent_trace": state.get("agent_trace", []) + [f"retrieved_{total}_items"],
        }

    except Exception as e:
        logger.exception("node.rag_retriever.error", error=str(e))
        return {
            "retrieved_items": {},
            "errors": state.get("errors", []) + [f"RAG retrieval failed: {str(e)[:80]}"],
            "agent_trace": state.get("agent_trace", []) + ["rag_failed"],
        }