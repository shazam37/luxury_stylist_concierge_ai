"""
Model-agnostic embedder.
Generates embeddings for fashion catalog items and queries.
Provider is set in config (EMBEDDING_PROVIDER).
"""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import structlog

from config.settings import get_settings, get_embedding_model, EmbeddingProvider

logger = structlog.get_logger(__name__)


def build_item_text(item: dict) -> str:
    """
    Constructs a rich text representation of a fashion item for embedding.
    More informative text → better semantic search quality.
    """
    parts = []

    if item.get("name"):
        parts.append(item["name"])
    if item.get("brand"):
        parts.append(f"by {item['brand']}")
    if item.get("category"):
        parts.append(item["category"])
    if item.get("sub_category"):
        parts.append(item["sub_category"])
    if item.get("color"):
        parts.append(f"in {item['color']}")
    if item.get("description"):
        # Truncate long descriptions
        desc = item["description"][:300]
        parts.append(desc)
    if item.get("tags"):
        parts.append(" ".join(item["tags"][:10]))
    if item.get("gender"):
        parts.append(f"for {item['gender']}")

    return ". ".join(parts)


class Embedder:
    """
    Wraps LangChain embedding models to provide batch async embedding.
    """

    def __init__(self):
        self._model = None
        self._settings = get_settings()

    def _get_model(self):
        if self._model is None:
            self._model = get_embedding_model(self._settings)
        return self._model

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (for search)."""
        model = self._get_model()
        loop = asyncio.get_event_loop()
        # LangChain embedding models are sync; run in thread pool
        vector = await loop.run_in_executor(None, model.embed_query, text)
        return vector

    async def embed_documents(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """
        Embed a list of documents in batches.
        Batching prevents rate limit errors on large catalogs.
        """
        model = self._get_model()
        loop = asyncio.get_event_loop()
        all_vectors = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vectors = await loop.run_in_executor(None, model.embed_documents, batch)
            all_vectors.extend(vectors)
            logger.info(
                "embedder.batch_complete",
                batch=i // batch_size + 1,
                total_batches=(len(texts) + batch_size - 1) // batch_size,
                embedded=len(all_vectors),
            )

        return all_vectors

    async def embed_items(self, items: list[dict], batch_size: int = 100) -> list[list[float]]:
        """
        Build rich text for each item and embed them.
        Returns vectors in same order as items.
        """
        texts = [build_item_text(item) for item in items]
        logger.info("embedder.embedding_items", count=len(texts))
        return await self.embed_documents(texts, batch_size=batch_size)

    def cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Compute cosine similarity between two vectors (for cache lookup)."""
        a = np.array(v1)
        b = np.array(v2)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)


# ─────────────────────────────────────────────
#  Singleton
# ─────────────────────────────────────────────

_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder