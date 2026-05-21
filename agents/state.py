"""
LangGraph State Schema for the Stylist Agent.

All data passed between nodes lives in StylistState.
Nodes read from state, compute, and return partial state updates.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class ParsedIntent(TypedDict, total=False):
    """Structured extraction from user prompt."""
    occasion: str                    # e.g. "yacht party", "office", "date night"
    owned_items: list[dict]          # items user already has
    requested_categories: list[str]  # top, bottom, shoes, etc.
    style_keywords: list[str]        # e.g. ["nautical", "summer", "elegant"]
    color_palette: list[str]         # extracted / inferred colors
    gender: str
    season: str                      # summer, winter, etc.
    formality: str                   # casual, smart_casual, formal
    budget_min: Optional[float]
    budget_max: Optional[float]


class StylistState(TypedDict, total=False):
    """
    Full agent state flowing through the LangGraph graph.
    Every node receives this and returns a partial update.
    """
    # ── Input ────────────────────────────────
    user_prompt: str
    user_id: Optional[str]
    session_id: Optional[str]
    gender: Optional[str]
    budget_min: Optional[float]
    budget_max: Optional[float]
    style_preference: Optional[str]
    include_alternatives: bool
    include_accessories: bool

    # ── Wardrobe context ─────────────────────
    wardrobe_items: list[dict]        # items user already owns

    # ── Intent parsing ───────────────────────
    parsed_intent: ParsedIntent
    query_vector: list[float]         # embedding of the enriched query

    # ── Cache ────────────────────────────────
    cache_hit: bool
    cached_response: Optional[dict]

    # ── RAG retrieval ────────────────────────
    retrieved_items: dict[str, list[dict]]   # category → list of items

    # ── Fashion reasoning ────────────────────
    ranked_items: dict[str, list[dict]]      # after coherence + rule scoring
    outfit_primary: dict                     # selected primary outfit
    outfit_alternatives: list[dict]          # 2 alternatives

    # ── Output ───────────────────────────────
    stylist_note: str
    final_response: dict

    # ── Metadata ─────────────────────────────
    token_usage: dict
    agent_trace: list[str]
    model_used: str
    errors: list[str]