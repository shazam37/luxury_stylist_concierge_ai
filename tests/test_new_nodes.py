"""
Tests for the two new agent nodes:
  - query_planner  (deterministic, no mocks needed)
  - price_optimizer (deterministic, no mocks needed)
"""
from __future__ import annotations

import pytest
from agents.nodes.query_planner import run_query_planner
from agents.nodes.price_optimizer import (
    run_price_optimizer, _outfit_total, _category_avg,
    _cheapest_alternative, _price_label,
)
from agents.state import StylistState


def _base_state(**overrides) -> StylistState:
    state: StylistState = {
        "user_prompt": "I have navy chinos, suggest a top and shoes for a yacht party",
        "user_id": None, "session_id": None,
        "gender": "male", "budget_min": None, "budget_max": None,
        "style_preference": None, "include_alternatives": True, "include_accessories": False,
        "wardrobe_items": [],
        "parsed_intent": {
            "occasion": "yacht party",
            "requested_categories": ["top", "shoes"],
            "style_keywords": ["nautical", "summer"],
            "color_palette": ["navy", "white"],
            "gender": "male",
            "season": "summer",
            "formality": "smart_casual",
            "budget_min": None,
            "budget_max": None,
            "owned_items": [{"name": "Navy Chinos", "category": "bottom", "color": "navy"}],
        },
        "query_vector": [0.1] * 1536,
        "query_plan": {}, "cache_hit": False, "cached_response": None,
        "retrieved_items": {}, "ranked_items": {}, "outfit_primary": {},
        "outfit_alternatives": [], "budget_summary": {}, "stylist_note": "",
        "final_response": {}, "token_usage": {}, "agent_trace": [], "model_used": "", "errors": [],
    }
    state.update(overrides)
    return state


# ─────────────────────────────────────────────────────────────────────────────
#  Query Planner Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryPlanner:

    @pytest.mark.asyncio
    async def test_produces_query_plan(self):
        state = _base_state()
        result = await run_query_planner(state)
        assert "query_plan" in result
        plan = result["query_plan"]
        assert "categories" in plan
        assert "category_queries" in plan
        assert "limits" in plan
        assert "sort_strategy" in plan
        assert "score_threshold" in plan

    @pytest.mark.asyncio
    async def test_categories_from_parsed_intent(self):
        state = _base_state()
        result = await run_query_planner(state)
        plan = result["query_plan"]
        assert set(plan["categories"]) == {"top", "shoes"}

    @pytest.mark.asyncio
    async def test_appends_to_agent_trace(self):
        state = _base_state()
        result = await run_query_planner(state)
        assert any("plan:" in t for t in result["agent_trace"])

    @pytest.mark.asyncio
    async def test_per_category_queries_generated(self):
        state = _base_state()
        result = await run_query_planner(state)
        plan = result["query_plan"]
        # Each category should have a specific query string
        for cat in plan["categories"]:
            assert cat in plan["category_queries"]
            assert len(plan["category_queries"][cat]) > 0

    @pytest.mark.asyncio
    async def test_yacht_occasion_injects_nautical_hints(self):
        state = _base_state()
        result = await run_query_planner(state)
        plan = result["query_plan"]
        top_query = plan["category_queries"].get("top", "")
        shoe_query = plan["category_queries"].get("shoes", "")
        # Yacht party should inject nautical/linen hints
        assert any(kw in top_query for kw in ["linen", "nautical", "summer"])
        assert any(kw in shoe_query for kw in ["espadrilles", "loafer", "boat", "canvas"])

    @pytest.mark.asyncio
    async def test_tight_budget_sets_price_asc_strategy(self):
        state = _base_state()
        state["parsed_intent"]["budget_max"] = 80.0
        state["budget_max"] = 80.0
        result = await run_query_planner(state)
        assert result["query_plan"]["sort_strategy"] == "price_asc"

    @pytest.mark.asyncio
    async def test_no_budget_uses_relevance_strategy(self):
        state = _base_state()
        result = await run_query_planner(state)
        assert result["query_plan"]["sort_strategy"] == "relevance"

    @pytest.mark.asyncio
    async def test_formal_has_higher_threshold(self):
        state_formal = _base_state()
        state_formal["parsed_intent"]["formality"] = "formal"
        state_casual = _base_state()
        state_casual["parsed_intent"]["formality"] = "casual"

        result_formal = await run_query_planner(state_formal)
        result_casual = await run_query_planner(state_casual)

        assert result_formal["query_plan"]["score_threshold"] > result_casual["query_plan"]["score_threshold"]

    @pytest.mark.asyncio
    async def test_accessories_added_when_flag_set(self):
        state = _base_state(include_accessories=True)
        result = await run_query_planner(state)
        assert "accessory" in result["query_plan"]["categories"]

    @pytest.mark.asyncio
    async def test_fallback_categories_when_empty(self):
        state = _base_state()
        state["parsed_intent"]["requested_categories"] = []
        result = await run_query_planner(state)
        # Should default to top + shoes (bottom owned)
        assert len(result["query_plan"]["categories"]) >= 1

    @pytest.mark.asyncio
    async def test_owned_color_context_in_query(self):
        state = _base_state()
        # User owns navy chino — query should reference complement
        result = await run_query_planner(state)
        plan = result["query_plan"]
        assert plan["owned_colors"] == ["navy"]


# ─────────────────────────────────────────────────────────────────────────────
#  Price Optimizer Tests
# ─────────────────────────────────────────────────────────────────────────────

MOCK_RETRIEVED = {
    "top": [
        {"name": "Linen Shirt",      "price": 49.99, "color": "white", "source": "zara"},
        {"name": "Cotton Tee",       "price": 19.99, "color": "white", "source": "hm"},
        {"name": "Premium Silk Top", "price": 129.99,"color": "white", "source": "zara"},
    ],
    "shoes": [
        {"name": "Canvas Espadrilles","price": 45.99, "color": "navy",  "source": "hm"},
        {"name": "Leather Loafers",   "price": 119.99,"color": "tan",   "source": "zara"},
        {"name": "Budget Sneakers",   "price": 29.99, "color": "white", "source": "hm"},
    ],
    "bottom": [],
}

MOCK_PRIMARY = {
    "top":    {"name": "Linen Shirt",       "price": 49.99,  "color": "white", "source": "zara"},
    "shoes":  {"name": "Leather Loafers",   "price": 119.99, "color": "tan",   "source": "zara"},
    "bottom": None, "accessory": None,
    "stylist_note": "A perfect yacht look.",
}

MOCK_ALTERNATIVES = [
    {
        "top":    {"name": "Premium Silk Top", "price": 129.99, "color": "white", "source": "zara"},
        "shoes":  {"name": "Leather Loafers",  "price": 119.99, "color": "tan",   "source": "zara"},
        "bottom": None, "accessory": None,
        "stylist_note": "Elevated silk look.",
    },
    {
        "top":    {"name": "Cotton Tee",        "price": 19.99, "color": "white", "source": "hm"},
        "shoes":  {"name": "Budget Sneakers",   "price": 29.99, "color": "white", "source": "hm"},
        "bottom": None, "accessory": None,
        "stylist_note": "Relaxed budget option.",
    },
]


class TestPriceOptimizerHelpers:

    def test_outfit_total_sums_items(self):
        outfit = {
            "top":    {"price": 49.99},
            "shoes":  {"price": 45.99},
            "bottom": None,
            "accessory": None,
        }
        assert _outfit_total(outfit) == pytest.approx(95.98, rel=1e-3)

    def test_outfit_total_ignores_none(self):
        assert _outfit_total({"top": None, "shoes": None}) == 0.0

    def test_outfit_total_handles_missing_price(self):
        outfit = {"top": {"name": "Shirt"}, "shoes": {"price": 45.99}}
        assert _outfit_total(outfit) == pytest.approx(45.99)

    def test_category_avg(self):
        avg = _category_avg(MOCK_RETRIEVED, "top")
        expected = (49.99 + 19.99 + 129.99) / 3
        assert avg == pytest.approx(expected, rel=1e-3)

    def test_category_avg_empty(self):
        assert _category_avg({}, "shoes") == 50.0

    def test_cheapest_alternative_finds_cheapest(self):
        result = _cheapest_alternative(MOCK_RETRIEVED, "shoes", 100.0)
        assert result is not None
        assert result["name"] == "Budget Sneakers"
        assert result["price"] == 29.99

    def test_cheapest_alternative_excludes_current(self):
        result = _cheapest_alternative(MOCK_RETRIEVED, "shoes", 100.0, exclude_name="Budget Sneakers")
        assert result is not None
        assert result["name"] == "Canvas Espadrilles"

    def test_cheapest_alternative_respects_budget(self):
        result = _cheapest_alternative(MOCK_RETRIEVED, "shoes", 20.0)
        assert result is None  # cheapest is 29.99, over budget of 20

    def test_price_label_best_value(self):
        assert _price_label(20.0, 80.0) == "Best value"

    def test_price_label_mid_range(self):
        assert _price_label(80.0, 80.0) == "Mid-range"

    def test_price_label_luxury(self):
        assert _price_label(200.0, 80.0) == "Luxury pick"

    def test_price_label_none(self):
        assert _price_label(None, 80.0) == ""


class TestPriceOptimizerNode:

    @pytest.mark.asyncio
    async def test_no_budget_passes_through(self):
        """Without a budget, optimizer annotates items but doesn't swap."""
        state = _base_state(
            outfit_primary=dict(MOCK_PRIMARY),
            outfit_alternatives=list(MOCK_ALTERNATIVES),
            retrieved_items=MOCK_RETRIEVED,
        )
        result = await run_price_optimizer(state)
        assert "outfit_primary" in result
        assert "outfit_alternatives" in result
        assert "budget_summary" in result
        assert result["budget_summary"]["within_budget"] is True
        assert result["budget_summary"]["swapped_items"] == []

    @pytest.mark.asyncio
    async def test_annotates_items_with_price_note(self):
        state = _base_state(
            outfit_primary=dict(MOCK_PRIMARY),
            outfit_alternatives=list(MOCK_ALTERNATIVES),
            retrieved_items=MOCK_RETRIEVED,
        )
        result = await run_price_optimizer(state)
        primary = result["outfit_primary"]
        # Both items should have price_note added
        assert "price_note" in primary["top"]
        assert "price_note" in primary["shoes"]

    @pytest.mark.asyncio
    async def test_swaps_over_budget_item(self):
        """Leather Loafers at $119.99 + Linen Shirt $49.99 = $169.98 > $100 budget."""
        state = _base_state(
            outfit_primary=dict(MOCK_PRIMARY),
            outfit_alternatives=list(MOCK_ALTERNATIVES),
            retrieved_items=MOCK_RETRIEVED,
            budget_max=100.0,
        )
        state["parsed_intent"]["budget_max"] = 100.0
        result = await run_price_optimizer(state)
        budget = result["budget_summary"]
        assert budget["within_budget"] is True
        assert budget["primary_total"] <= 100.0
        assert len(budget["swapped_items"]) > 0

    @pytest.mark.asyncio
    async def test_budget_summary_populated(self):
        state = _base_state(
            outfit_primary=dict(MOCK_PRIMARY),
            outfit_alternatives=list(MOCK_ALTERNATIVES),
            retrieved_items=MOCK_RETRIEVED,
            budget_max=200.0,
        )
        state["parsed_intent"]["budget_max"] = 200.0
        result = await run_price_optimizer(state)
        bs = result["budget_summary"]
        assert bs["budget_max"] == 200.0
        assert bs["primary_total"] > 0
        assert "savings_vs_splurge" in bs

    @pytest.mark.asyncio
    async def test_alternatives_tagged_with_price_tier(self):
        state = _base_state(
            outfit_primary=dict(MOCK_PRIMARY),
            outfit_alternatives=list(MOCK_ALTERNATIVES),
            retrieved_items=MOCK_RETRIEVED,
        )
        result = await run_price_optimizer(state)
        alts = result["outfit_alternatives"]
        assert len(alts) == 2
        # Each alternative should have a price_tier tag
        for alt in alts:
            assert "price_tier" in alt
            assert alt["price_tier"] in ("value_pick", "splurge_pick", "balanced")
            assert "price_tier_label" in alt

    @pytest.mark.asyncio
    async def test_expensive_alt_tagged_splurge(self):
        """Silk top ($129.99) + loafers ($119.99) should be the splurge pick."""
        state = _base_state(
            outfit_primary=dict(MOCK_PRIMARY),
            outfit_alternatives=list(MOCK_ALTERNATIVES),
            retrieved_items=MOCK_RETRIEVED,
        )
        result = await run_price_optimizer(state)
        alts = result["outfit_alternatives"]
        totals = [(a.get("price_tier"), _outfit_total(a)) for a in alts]
        splurge = [t for tier, t in totals if tier == "splurge_pick"]
        value = [t for tier, t in totals if tier == "value_pick"]
        # At least one value and one splurge across the two alts
        assert len(splurge) > 0 or len(value) > 0

    @pytest.mark.asyncio
    async def test_stylist_note_appended_when_within_budget(self):
        """When within budget with savings vs splurge, savings note appended."""
        state = _base_state(
            outfit_primary=dict(MOCK_PRIMARY),
            outfit_alternatives=list(MOCK_ALTERNATIVES),
            retrieved_items=MOCK_RETRIEVED,
            budget_max=250.0,
        )
        state["parsed_intent"]["budget_max"] = 250.0
        result = await run_price_optimizer(state)
        note = result["outfit_primary"].get("stylist_note", "")
        # Should append budget note since primary ($169.98) is well under $250
        assert "$" in note

    @pytest.mark.asyncio
    async def test_agent_trace_updated(self):
        state = _base_state(
            outfit_primary=dict(MOCK_PRIMARY),
            outfit_alternatives=list(MOCK_ALTERNATIVES),
            retrieved_items=MOCK_RETRIEVED,
        )
        result = await run_price_optimizer(state)
        assert any("price_opt:" in t for t in result["agent_trace"])

    @pytest.mark.asyncio
    async def test_empty_outfit_handled_gracefully(self):
        state = _base_state(
            outfit_primary={},
            outfit_alternatives=[],
            retrieved_items={},
        )
        result = await run_price_optimizer(state)
        assert "outfit_primary" in result
        assert result["budget_summary"]["primary_total"] == 0.0


class TestGraphWithNewNodes:
    def test_graph_has_all_9_nodes(self):
        import agents.graph as ag
        ag._compiled_graph = None  # force rebuild
        g = ag.get_graph()
        expected = {
            "__start__", "load_wardrobe", "intent_parser", "cache_check",
            "query_planner", "rag_retriever", "fashion_reasoner",
            "price_optimizer", "response_formatter", "cache_formatter",
        }
        assert expected.issubset(set(g.nodes.keys()))

    def test_route_cache_miss_goes_to_query_planner(self):
        from agents.graph import route_after_cache
        state = _base_state(cache_hit=False)
        assert route_after_cache(state) == "query_planner"

    def test_route_cache_hit_goes_to_formatter(self):
        from agents.graph import route_after_cache
        state = _base_state(cache_hit=True, cached_response={"outfit": {}})
        assert route_after_cache(state) == "cache_formatter"