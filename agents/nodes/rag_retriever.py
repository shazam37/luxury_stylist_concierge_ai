"""
Node 3: RAG Retriever
Runs multi-category semantic search on Qdrant.
Retrieves tops, bottoms, shoes (and accessories if requested) in parallel.
"""
from __future__ import annotations
import structlog
from agents.state import StylistState

logger = structlog.get_logger(__name__)


async def run_rag_retriever(state: StylistState) -> dict:
    """Retrieve fashion items from Qdrant using semantic + filter search."""
    logger.info("node.rag_retriever.start")

    parsed = state.get("parsed_intent", {})
    vector = state.get("query_vector", [])

    if not vector:
        return {
            "retrieved_items": {},
            "errors": state.get("errors", []) + ["No query vector available for RAG"],
            "agent_trace": state.get("agent_trace", []) + ["rag_skip_no_vector"],
        }

    from embeddings.indexer import get_qdrant_manager
    qdrant = get_qdrant_manager()

    # Build filter dict from parsed intent + request params
    filters = {}
    gender = parsed.get("gender") or state.get("gender")
    if gender and gender != "unisex":
        filters["gender"] = gender

    budget_max = parsed.get("budget_max") or state.get("budget_max")
    budget_min = parsed.get("budget_min") or state.get("budget_min")
    if budget_max:
        filters["max_price"] = float(budget_max)
    if budget_min:
        filters["min_price"] = float(budget_min)

    # Determine categories to retrieve
    requested = parsed.get("requested_categories", ["top", "shoes"])
    if state.get("include_accessories") and "accessory" not in requested:
        requested = list(requested) + ["accessory"]

    # Always ensure we have at least tops and shoes
    if not requested:
        requested = ["top", "bottom", "shoes"]

    try:
        retrieved = await qdrant.multi_category_search(
            query_vector=vector,
            categories=requested,
            filters=filters,
            limit_per_category=8,
        )

        total = sum(len(v) for v in retrieved.values())
        logger.info("node.rag_retriever.done", categories=list(retrieved.keys()), total_items=total)

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