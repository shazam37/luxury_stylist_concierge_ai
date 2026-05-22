"""Tests for agent nodes and graph logic."""
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.state import StylistState
from agents.nodes.cache_manager import run_cache_check
from agents.nodes.rag_retriever import run_rag_retriever
from agents.nodes.response_formatter import run_response_formatter, _aggregate_token_usage
from api.schemas import TokenUsage


def _base_state(**overrides) -> StylistState:
    state: StylistState = {
        "user_prompt": "I have navy chinos, what top for a yacht party?",
        "user_id": None, "session_id": None, "gender": "male",
        "budget_min": None, "budget_max": None,
        "style_preference": None, "include_alternatives": True, "include_accessories": False,
        "wardrobe_items": [],
        "parsed_intent": {
            "occasion": "yacht party", "requested_categories": ["top", "shoes"],
            "style_keywords": ["nautical"], "gender": "male",
            "season": "summer", "formality": "smart_casual",
        },
        "query_vector": [0.1] * 1536, "cache_hit": False, "cached_response": None,
        "retrieved_items": {}, "ranked_items": {}, "outfit_primary": {},
        "outfit_alternatives": [], "stylist_note": "", "final_response": {},
        "token_usage": {}, "agent_trace": [], "model_used": "", "errors": [],
    }
    state.update(overrides)
    return state


class TestCacheManager:
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value=None)
        with patch("cache.semantic_cache._cache", mock_cache):
            result = await run_cache_check(_base_state())
        assert result["cache_hit"] is False
        assert "cache_miss" in result["agent_trace"]

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        cached_data = {"outfit": {"stylist_note": "Cached look"}, "cache_hit": True}
        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value=cached_data)
        with patch("cache.semantic_cache._cache", mock_cache):
            result = await run_cache_check(_base_state())
        assert result["cache_hit"] is True
        assert result["cached_response"] == cached_data

    @pytest.mark.asyncio
    async def test_no_vector_skips_cache(self):
        result = await run_cache_check(_base_state(query_vector=[]))
        assert result["cache_hit"] is False
        assert "cache_skip_no_vector" in result["agent_trace"]

    @pytest.mark.asyncio
    async def test_cache_error_graceful(self):
        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(side_effect=ConnectionError("Redis down"))
        with patch("cache.semantic_cache._cache", mock_cache):
            result = await run_cache_check(_base_state())
        assert result["cache_hit"] is False
        assert "cache_error" in result["agent_trace"]


class TestRagRetriever:
    @pytest.mark.asyncio
    async def test_retrieves_multiple_categories(self):
        mock_qdrant = AsyncMock()
        mock_qdrant.search = AsyncMock(return_value=[])
        mock_embedder = AsyncMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 1536)
        state = _base_state()
        state["query_plan"] = {"categories": ["top", "shoes"], "limits": {"top": 8, "shoes": 6},
                                "score_threshold": 0.6, "category_queries": {"top": "linen shirt", "shoes": "espadrilles"}}
        with patch("embeddings.indexer.get_qdrant_manager", return_value=mock_qdrant),              patch("embeddings.embedder.get_embedder", return_value=mock_embedder):
            result = await run_rag_retriever(state)
        assert "retrieved_items" in result
        assert mock_qdrant.search.call_count == 2  # once per category

    @pytest.mark.asyncio
    async def test_no_vector_returns_empty(self):
        result = await run_rag_retriever(_base_state(query_vector=[]))
        assert result["retrieved_items"] == {}
        assert "rag_skip_no_vector" in result["agent_trace"]

    @pytest.mark.asyncio
    async def test_passes_gender_filter(self):
        mock_qdrant = AsyncMock()
        mock_qdrant.search = AsyncMock(return_value=[])
        mock_embedder = AsyncMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 1536)
        state = _base_state(gender="female")
        state["parsed_intent"]["gender"] = "female"
        state["query_plan"] = {"categories": ["top"], "limits": {"top": 8},
                                "score_threshold": 0.6, "category_queries": {"top": "blouse"}}
        with patch("embeddings.indexer.get_qdrant_manager", return_value=mock_qdrant),              patch("embeddings.embedder.get_embedder", return_value=mock_embedder):
            await run_rag_retriever(state)
        call_kwargs = mock_qdrant.search.call_args.kwargs
        assert call_kwargs.get("gender") == "female"

    @pytest.mark.asyncio
    async def test_passes_budget_filter(self):
        mock_qdrant = AsyncMock()
        mock_qdrant.search = AsyncMock(return_value=[])
        mock_embedder = AsyncMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 1536)
        state = _base_state(budget_max=150.0)
        state["parsed_intent"]["budget_max"] = 150.0
        state["query_plan"] = {"categories": ["top"], "limits": {"top": 8},
                                "score_threshold": 0.6, "category_queries": {"top": "shirt"}}
        with patch("embeddings.indexer.get_qdrant_manager", return_value=mock_qdrant),              patch("embeddings.embedder.get_embedder", return_value=mock_embedder):
            await run_rag_retriever(state)
        call_kwargs = mock_qdrant.search.call_args.kwargs
        assert call_kwargs.get("max_price") == 150.0

    @pytest.mark.asyncio
    async def test_qdrant_error_handled(self):
        mock_qdrant = AsyncMock()
        mock_qdrant.search = AsyncMock(side_effect=Exception("Qdrant down"))
        mock_embedder = AsyncMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 1536)
        state = _base_state()
        state["query_plan"] = {"categories": ["top"], "limits": {"top": 8},
                                "score_threshold": 0.6, "category_queries": {"top": "shirt"}}
        with patch("embeddings.indexer.get_qdrant_manager", return_value=mock_qdrant),              patch("embeddings.embedder.get_embedder", return_value=mock_embedder):
            result = await run_rag_retriever(state)
        # Per-category errors are caught; returns empty dict not fatal
        assert "retrieved_items" in result


class TestResponseFormatter:
    def test_aggregate_token_usage_sum(self):
        usage = {
            "intent_parser": {"prompt_tokens": 200, "completion_tokens": 100},
            "fashion_reasoner": {"prompt_tokens": 800, "completion_tokens": 400},
        }
        result = _aggregate_token_usage(usage)
        assert result.prompt_tokens == 1000
        assert result.completion_tokens == 500
        assert result.total_tokens == 1500

    def test_aggregate_empty(self):
        assert _aggregate_token_usage({}).total_tokens == 0

    @pytest.mark.asyncio
    async def test_formats_primary_outfit(self):
        mock_cache = AsyncMock()
        mock_cache.set = AsyncMock(return_value="cache-id")
        outfit_data = {
            "top": {"name": "Linen Shirt", "brand": "Zara", "price": 49.99, "color": "white",
                    "image_url": "https://example.com/shirt.jpg",
                    "product_url": "https://zara.com/shirt", "source": "zara"},
            "shoes": {"name": "Espadrilles", "brand": "H&M", "price": 45.99, "color": "navy",
                      "image_url": "https://example.com/shoes.jpg",
                      "product_url": "https://hm.com/shoes", "source": "hm"},
            "bottom": None, "accessory": None,
            "stylist_note": "Perfect yacht club look.",
        }
        state = _base_state(outfit_primary=outfit_data, outfit_alternatives=[])
        with patch("cache.semantic_cache._cache", mock_cache):
            result = await run_response_formatter(state)
        final = result["final_response"]
        assert final["outfit"].top.name == "Linen Shirt"
        assert final["outfit"].shoes.name == "Espadrilles"
        assert final["outfit"].total_price == pytest.approx(95.98, rel=1e-2)

    @pytest.mark.asyncio
    async def test_stores_in_cache_on_miss(self):
        mock_cache = AsyncMock()
        mock_cache.set = AsyncMock(return_value="cache-id")
        state = _base_state(
            outfit_primary={"stylist_note": "Nice look.", "top": None, "bottom": None, "shoes": None, "accessory": None},
            outfit_alternatives=[], cache_hit=False, query_vector=[0.1] * 1536,
        )
        with patch("cache.semantic_cache._cache", mock_cache):
            await run_response_formatter(state)
        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_cache_write_on_hit(self):
        mock_cache = AsyncMock()
        state = _base_state(
            outfit_primary={"stylist_note": "Nice.", "top": None, "bottom": None, "shoes": None, "accessory": None},
            outfit_alternatives=[], cache_hit=True,
        )
        with patch("cache.semantic_cache._cache", mock_cache):
            await run_response_formatter(state)
        mock_cache.set.assert_not_called()


class TestGraphRouting:
    def test_graph_has_all_nodes(self):
        import agents.graph as ag; ag._compiled_graph = None
        g = ag.get_graph()
        expected = {"__start__", "load_wardrobe", "intent_parser", "cache_check",
                    "query_planner", "rag_retriever", "fashion_reasoner",
                    "price_optimizer", "response_formatter", "cache_formatter"}
        assert expected.issubset(set(g.nodes.keys()))

    def test_route_cache_miss(self):
        from agents.graph import route_after_cache
        assert route_after_cache(_base_state(cache_hit=False)) == "query_planner"

    def test_route_cache_hit(self):
        from agents.graph import route_after_cache
        state = _base_state(cache_hit=True, cached_response={"outfit": {}})
        assert route_after_cache(state) == "cache_formatter"

    def test_route_cache_hit_no_data(self):
        from agents.graph import route_after_cache
        state = _base_state(cache_hit=True, cached_response=None)
        assert route_after_cache(state) == "query_planner"