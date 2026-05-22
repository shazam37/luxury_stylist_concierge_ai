"""
Node 2.5: Query Planner

Sits between intent_parser and rag_retriever.
Transforms ParsedIntent into an explicit, inspectable QueryPlan that drives
all downstream retrieval decisions.

Responsibilities:
  - Decide which categories to retrieve and how many results per category
  - Build per-category semantic query strings (more specific than the raw prompt)
  - Select a retrieval sort strategy (relevance vs price_asc vs price_desc)
  - Set confidence thresholds per category based on occasion + formality
  - Produce the query_plan state key consumed by rag_retriever

This separation means rag_retriever becomes a pure execution node with zero
decision logic — making both nodes independently testable.
"""

from __future__ import annotations

import structlog
from agents.state import StylistState, QueryPlan

logger = structlog.get_logger(__name__)


# ── Occasion → retrieval strategy mapping ────────────────────────────────────

# How many results to fetch per category for each formality level.
# Higher formality = fewer but more precise results needed.
_FORMALITY_LIMITS: dict[str, int] = {
    "casual":          10,
    "smart_casual":    8,
    "business_casual": 8,
    "formal":          6,
}

# Score threshold below which a retrieved item is considered a poor match.
# Higher = stricter. Qdrant uses cosine similarity (0–1).
_FORMALITY_THRESHOLDS: dict[str, float] = {
    "casual":          0.55,
    "smart_casual":    0.62,
    "business_casual": 0.65,
    "formal":          0.70,
}

# Per-occasion hints injected into category-level semantic queries.
_OCCASION_HINTS: dict[str, dict[str, str]] = {
    "yacht party":     {"top": "linen shirt summer nautical",   "shoes": "espadrilles loafers boat shoes canvas"},
    "beach":           {"top": "lightweight cotton casual",      "shoes": "sandals slides casual"},
    "office":          {"top": "structured shirt formal",        "shoes": "leather oxford derby loafer"},
    "date night":      {"top": "elegant evening fitted",         "shoes": "leather dress boot chelsea"},
    "casual friday":   {"top": "relaxed smart polo shirt",       "shoes": "clean sneaker loafer"},
    "wedding":         {"top": "formal dress shirt suit",        "shoes": "oxford brogue leather formal"},
    "streetwear":      {"top": "graphic oversized urban",        "shoes": "chunky sneaker trainer"},
    "gym":             {"top": "athletic performance moisture",   "shoes": "running trainer gym"},
    "festival":        {"top": "bohemian printed relaxed",       "shoes": "boot sandal comfortable"},
    "brunch":          {"top": "smart casual light relaxed",     "shoes": "loafer clean sneaker mule"},
}

_DEFAULT_OCCASION_HINTS: dict[str, str] = {
    "top":       "stylish versatile",
    "bottom":    "well-fitted comfortable",
    "shoes":     "clean versatile",
    "accessory": "understated elegant",
    "outerwear": "classic well-structured",
}


async def run_query_planner(state: StylistState) -> dict:
    """
    Produce a QueryPlan from ParsedIntent.
    No LLM call — this is pure deterministic logic, fast and free.
    """
    logger.info("node.query_planner.start")

    parsed = state.get("parsed_intent", {})
    occasion_raw = (parsed.get("occasion") or "").lower().strip()
    formality = (parsed.get("formality") or "smart_casual").lower()
    style_keywords = parsed.get("style_keywords") or []
    color_palette = parsed.get("color_palette") or []
    season = parsed.get("season") or "all-season"
    owned_items = parsed.get("owned_items") or state.get("wardrobe_items") or []

    # ── 1. Determine categories to retrieve ──────────────────────────────────
    requested_cats = list(parsed.get("requested_categories") or [])
    if not requested_cats:
        # Default: anything not already owned
        owned_cats = {i.get("category", "").lower() for i in owned_items}
        requested_cats = [c for c in ["top", "bottom", "shoes"] if c not in owned_cats]
        if not requested_cats:
            requested_cats = ["top", "shoes"]

    if state.get("include_accessories") and "accessory" not in requested_cats:
        requested_cats = list(requested_cats) + ["accessory"]

    # ── 2. Per-category retrieval limits ─────────────────────────────────────
    base_limit = _FORMALITY_LIMITS.get(formality, 8)
    # Give shoes fewer slots since there are fewer options typically
    limits: dict[str, int] = {}
    for cat in requested_cats:
        if cat == "accessory":
            limits[cat] = min(base_limit, 5)
        elif cat == "shoes":
            limits[cat] = min(base_limit, 6)
        else:
            limits[cat] = base_limit

    # ── 3. Per-category semantic query strings ────────────────────────────────
    # Find the best-matching occasion hint
    occasion_hints = _DEFAULT_OCCASION_HINTS.copy()
    for occ_key, hints in _OCCASION_HINTS.items():
        if occ_key in occasion_raw:
            occasion_hints.update(hints)
            break

    # Build owned-items color context to guide complementary retrieval
    owned_color_ctx = ""
    if owned_items:
        owned_colors = [i.get("color", "") for i in owned_items if i.get("color")]
        if owned_colors:
            owned_color_ctx = f"complement {' '.join(owned_colors[:3])}"

    style_ctx = " ".join(style_keywords[:4])
    season_ctx = season if season != "all-season" else ""

    category_queries: dict[str, str] = {}
    for cat in requested_cats:
        hint = occasion_hints.get(cat, _DEFAULT_OCCASION_HINTS.get(cat, ""))
        parts = [p for p in [hint, style_ctx, season_ctx, owned_color_ctx] if p]
        # Primary query: occasion hint + style keywords + color context
        category_queries[cat] = " ".join(parts) if parts else cat

    # ── 4. Sort strategy ─────────────────────────────────────────────────────
    budget_max = parsed.get("budget_max") or state.get("budget_max")
    budget_min = parsed.get("budget_min") or state.get("budget_min")

    if budget_max and budget_max < 100:
        sort_strategy = "price_asc"   # tight budget → show cheapest first
    elif formality == "formal":
        sort_strategy = "relevance"   # formal → quality over price
    else:
        sort_strategy = "relevance"   # default

    # ── 5. Score threshold ───────────────────────────────────────────────────
    score_threshold = _FORMALITY_THRESHOLDS.get(formality, 0.60)

    # ── 6. Assemble the plan ─────────────────────────────────────────────────
    plan: QueryPlan = {
        "categories": requested_cats,
        "category_queries": category_queries,
        "limits": limits,
        "sort_strategy": sort_strategy,
        "score_threshold": score_threshold,
        "occasion": occasion_raw or "general",
        "formality": formality,
        "season": season,
        "color_palette": color_palette,
        "owned_colors": [i.get("color") for i in owned_items if i.get("color")],
        "style_keywords": style_keywords,
    }

    logger.info(
        "node.query_planner.done",
        categories=requested_cats,
        strategy=sort_strategy,
        threshold=score_threshold,
        occasion=plan["occasion"],
    )

    return {
        "query_plan": plan,
        "agent_trace": state.get("agent_trace", []) + [
            f"plan:{','.join(requested_cats)}|strategy:{sort_strategy}"
        ],
    }