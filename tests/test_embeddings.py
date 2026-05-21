"""Tests for embedder and Qdrant indexer utilities."""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from embeddings.embedder import build_item_text, Embedder


class TestBuildItemText:
    def test_full_item(self):
        item = {
            "name": "Classic Linen Shirt",
            "brand": "Zara",
            "category": "top",
            "sub_category": "shirt",
            "color": "white",
            "description": "A breathable summer shirt",
            "tags": ["summer", "linen"],
            "gender": "male",
        }
        text = build_item_text(item)
        assert "Classic Linen Shirt" in text
        assert "Zara" in text
        assert "white" in text
        assert "summer" in text

    def test_minimal_item(self):
        text = build_item_text({"name": "White Shirt"})
        assert "White Shirt" in text

    def test_long_description_truncated(self):
        item = {"name": "Shirt", "description": "x" * 500}
        text = build_item_text(item)
        assert len(text) < 1000  # truncated

    def test_empty_item(self):
        text = build_item_text({})
        assert text == ""


class TestEmbedder:
    @pytest.mark.asyncio
    async def test_embed_query(self):
        mock_model = MagicMock()
        mock_model.embed_query = MagicMock(return_value=[0.1] * 1536)

        embedder = Embedder()
        embedder._model = mock_model

        result = await embedder.embed_query("white linen shirt for yacht party")
        assert len(result) == 1536
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_embed_documents_batching(self):
        mock_model = MagicMock()
        # 5 texts, batch_size=3 → 2 batches; each batch call returns vectors for that batch
        mock_model.embed_documents = MagicMock(side_effect=[
            [[0.1] * 1536] * 3,  # batch 1: 3 items
            [[0.1] * 1536] * 2,  # batch 2: 2 items
        ])

        embedder = Embedder()
        embedder._model = mock_model

        texts = ["item " + str(i) for i in range(5)]
        result = await embedder.embed_documents(texts, batch_size=3)
        assert len(result) == 5
        assert mock_model.embed_documents.call_count == 2

    def test_cosine_similarity_identical(self):
        embedder = Embedder()
        v = [1.0, 0.0, 0.0]
        assert embedder.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        embedder = Embedder()
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert embedder.cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        embedder = Embedder()
        v1 = [0.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert embedder.cosine_similarity(v1, v2) == 0.0

    @pytest.mark.asyncio
    async def test_embed_items_uses_build_item_text(self):
        mock_model = MagicMock()
        mock_model.embed_documents = MagicMock(return_value=[[0.2] * 1536] * 2)

        embedder = Embedder()
        embedder._model = mock_model

        items = [
            {"name": "Navy Shirt", "brand": "Zara", "category": "top"},
            {"name": "White Jeans", "brand": "H&M", "category": "bottom"},
        ]
        result = await embedder.embed_items(items)
        assert len(result) == 2
        mock_model.embed_documents.assert_called_once()


class TestSemanticCache:
    def test_cosine_similarity_above_threshold(self):
        from cache.semantic_cache import SemanticCache
        cache = SemanticCache()
        v1 = [1.0] + [0.0] * 1535
        v2 = [1.0] + [0.0] * 1535
        assert cache._cosine_similarity(v1, v2) == pytest.approx(1.0)

    def test_memory_fallback_set_get(self):
        from cache.semantic_cache import SemanticCache
        import time
        cache = SemanticCache()
        cache._threshold = 0.95

        v = [1.0] + [0.0] * 1535
        cache._memory_set("entry1", {
            "id": "entry1",
            "prompt": "test",
            "vector": v,
            "response": {"outfit": "test"},
            "created_at": time.time(),
        })

        result = cache._memory_get(v)
        assert result == {"outfit": "test"}

    def test_memory_fallback_miss(self):
        from cache.semantic_cache import SemanticCache
        cache = SemanticCache()
        cache._threshold = 0.99

        v1 = [1.0] + [0.0] * 1535
        v2 = [0.0, 1.0] + [0.0] * 1534  # orthogonal
        import time
        cache._memory_set("entry1", {
            "id": "entry1",
            "prompt": "test",
            "vector": v1,
            "response": {"outfit": "test"},
            "created_at": time.time(),
        })

        result = cache._memory_get(v2)
        assert result is None