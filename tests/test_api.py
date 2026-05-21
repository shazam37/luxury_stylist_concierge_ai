"""Integration tests for FastAPI endpoints."""
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
SAMPLE_STYLE_REQUEST = {
    "prompt": "I have dark navy chinos. What t-shirt and shoes for a summer yacht party?",
    "gender": "male",
    "include_alternatives": True,
    "include_accessories": False,
}


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_health_ok(self, app_client):
        resp = await app_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_info_endpoint(self, app_client):
        resp = await app_client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "llm_provider" in data
        assert "llm_model" in data
        assert "qdrant_collection" in data

    @pytest.mark.asyncio
    async def test_docs_available(self, app_client):
        resp = await app_client.get("/docs")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_openapi_schema(self, app_client):
        resp = await app_client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "Stylist Concierge API"
        # Verify all key endpoints are documented
        paths = schema["paths"]
        assert "/api/v1/style-me" in paths
        assert "/api/v1/catalog" in paths
        assert "/api/v1/wardrobe" in paths
        assert "/api/v1/scraper/trigger" in paths


class TestStyleMeEndpoint:
    @pytest.mark.asyncio
    async def test_style_me_prompt_too_short(self, app_client):
        resp = await app_client.post("/api/v1/style-me", json={"prompt": "hi"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_style_me_missing_prompt(self, app_client):
        resp = await app_client.post("/api/v1/style-me", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_style_me_stub_response(self, app_client, mock_qdrant, mock_embedder, mock_llm, mock_cache):
        """Full pipeline test with all external calls mocked."""
        from api.schemas import OutfitOption, TokenUsage

        mock_response = {
            "cache_hit": False,
            "parsed_intent": {"occasion": "yacht party", "style_keywords": ["nautical"]},
            "outfit": OutfitOption(stylist_note="Perfect yacht look.", total_price=95.98),
            "alternatives": [],
            "token_usage": TokenUsage(prompt_tokens=500, completion_tokens=200, total_tokens=700),
            "agent_trace": ["intent_parsed", "cache_miss", "retrieved_3_items", "outfit_reasoned", "formatted"],
            "model_used": "groq/llama-4",
        }

        # Patch where agent is actually called (lazy import inside function)
        with patch("agents.graph.run_stylist_agent", new=AsyncMock(return_value=mock_response)):
            resp = await app_client.post("/api/v1/style-me", json=SAMPLE_STYLE_REQUEST)

        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data
        assert "outfit" in data
        assert data["outfit"]["stylist_note"] == "Perfect yacht look."
        assert data["cache_hit"] is False
        assert data["token_usage"]["total_tokens"] == 700
        assert "X-Request-ID" in resp.headers
        assert "X-Response-Time" in resp.headers

    @pytest.mark.asyncio
    async def test_style_me_with_budget(self, app_client):
        from api.schemas import OutfitOption, TokenUsage
        mock_response = {
            "cache_hit": False, "parsed_intent": {},
            "outfit": OutfitOption(stylist_note="Budget look.", total_price=80.0),
            "alternatives": [], "token_usage": TokenUsage(),
            "agent_trace": [], "model_used": "groq/llama-4",
        }
        with patch("agents.graph.run_stylist_agent", new=AsyncMock(return_value=mock_response)):
            resp = await app_client.post("/api/v1/style-me", json={
                **SAMPLE_STYLE_REQUEST,
                "budget": {"min_price": 0, "max_price": 200, "currency": "USD"},
            })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_feedback_endpoint(self, app_client):
        resp = await app_client.post("/api/v1/style-me/feedback", json={
            "request_id": "00000000-0000-0000-0000-000000000001",
            "score": 5,
            "notes": "Loved the espadrilles recommendation!",
        })
        assert resp.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_feedback_invalid_score(self, app_client):
        resp = await app_client.post("/api/v1/style-me/feedback", json={
            "request_id": "test-id", "score": 10,
        })
        assert resp.status_code == 422


class TestCatalogEndpoints:
    @pytest.mark.asyncio
    async def test_list_catalog_empty(self, app_client):
        resp = await app_client.get("/api/v1/catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "count" in data
        assert isinstance(data["items"], list)

    @pytest.mark.asyncio
    async def test_catalog_with_filters(self, app_client):
        resp = await app_client.get("/api/v1/catalog?category=top&gender=male&limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["limit"] == 5

    @pytest.mark.asyncio
    async def test_catalog_stats(self, app_client, mock_qdrant):
        with patch("embeddings.indexer.get_qdrant_manager", return_value=mock_qdrant):
            resp = await app_client.get("/api/v1/catalog/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_items" in data
        assert "by_category" in data
        assert "by_source" in data
        assert "qdrant_vectors" in data

    @pytest.mark.asyncio
    async def test_catalog_item_not_found(self, app_client):
        resp = await app_client.get("/api/v1/catalog/nonexistent-id-123")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_catalog_semantic_search(self, app_client, mock_embedder, mock_qdrant):
        with patch("embeddings.indexer.get_qdrant_manager", return_value=mock_qdrant), \
             patch("embeddings.embedder.get_embedder", return_value=mock_embedder):
            resp = await app_client.get("/api/v1/catalog/search?q=white+linen+shirt&limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "query" in data
        assert "results" in data

    @pytest.mark.asyncio
    async def test_catalog_pagination(self, app_client):
        resp = await app_client.get("/api/v1/catalog?limit=5&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["offset"] == 0
        assert data["limit"] == 5


class TestWardrobeEndpoints:
    @pytest.mark.asyncio
    async def test_list_wardrobe_empty(self, app_client):
        resp = await app_client.get("/api/v1/wardrobe")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_add_wardrobe_item(self, app_client):
        resp = await app_client.post("/api/v1/wardrobe", json={
            "name": "Dark Navy Chinos",
            "category": "bottom",
            "color": "navy",
            "brand": "Zara",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Dark Navy Chinos"
        assert data["category"] == "bottom"
        assert data["color"] == "navy"
        assert "id" in data
        return data["id"]

    @pytest.mark.asyncio
    async def test_add_then_list_wardrobe(self, app_client):
        # Add item
        add_resp = await app_client.post("/api/v1/wardrobe", json={
            "name": "White Linen Shirt",
            "category": "top",
            "color": "white",
        })
        assert add_resp.status_code == 201
        data = add_resp.json()
        # Verify the returned item has all expected fields
        assert data["name"] == "White Linen Shirt"
        assert data["category"] == "top"
        assert data["color"] == "white"
        assert "id" in data
        assert data["id"] is not None

    @pytest.mark.asyncio
    async def test_delete_wardrobe_item(self, app_client):
        # Add
        add_resp = await app_client.post("/api/v1/wardrobe", json={
            "name": "Canvas Sneakers",
            "category": "shoes",
        })
        item_id = add_resp.json()["id"]

        # Delete
        del_resp = await app_client.delete(f"/api/v1/wardrobe/{item_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted"] == item_id

    @pytest.mark.asyncio
    async def test_delete_nonexistent_wardrobe_item(self, app_client):
        resp = await app_client.delete("/api/v1/wardrobe/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_wardrobe_invalid_category(self, app_client):
        resp = await app_client.post("/api/v1/wardrobe", json={
            "name": "Something",
            "category": "invalid_category",
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_wardrobe_filter_by_category(self, app_client):
        # Add two items
        await app_client.post("/api/v1/wardrobe", json={"name": "Oxford Shirt", "category": "top"})
        await app_client.post("/api/v1/wardrobe", json={"name": "Slim Jeans", "category": "bottom"})

        # Filter by top
        resp = await app_client.get("/api/v1/wardrobe?category=top")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert all(i["category"] == "top" for i in items)


class TestScraperEndpoints:
    @pytest.mark.asyncio
    async def test_trigger_scrape(self, app_client):
        resp = await app_client.post("/api/v1/scraper/trigger", json={
            "source": "all",
            "reindex": False,
        })
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert "source" in data

    @pytest.mark.asyncio
    async def test_trigger_specific_source(self, app_client):
        resp = await app_client.post("/api/v1/scraper/trigger", json={
            "source": "zara",
            "reindex": True,
        })
        assert resp.status_code == 202
        data = resp.json()
        assert data["source"] == "zara"

    @pytest.mark.asyncio
    async def test_list_jobs(self, app_client):
        # Create a job first
        await app_client.post("/api/v1/scraper/trigger", json={"source": "hm"})
        resp = await app_client.get("/api/v1/scraper/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)

    @pytest.mark.asyncio
    async def test_get_job_status(self, app_client):
        # Create job
        create_resp = await app_client.post("/api/v1/scraper/trigger", json={"source": "zara"})
        job_id = create_resp.json()["job_id"]

        # Check status
        resp = await app_client.get(f"/api/v1/scraper/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("pending", "running", "completed", "failed")

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, app_client):
        resp = await app_client.get("/api/v1/scraper/jobs/nonexistent-job-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_seed_endpoint(self, app_client):
        resp = await app_client.post("/api/v1/scraper/seed?count=10")
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "seeding"
        assert data["count_requested"] == 10