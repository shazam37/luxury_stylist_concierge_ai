"""
Qdrant collection management.
Handles: collection creation, schema, upsert, search, delete.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import structlog
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
    FieldCondition,
    MatchValue,
    Range,
    Filter,
    PointStruct,
    SearchRequest,
)

from config.settings import get_settings

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
#  Qdrant Payload Schema
#  Every indexed fashion item carries these fields.
# ─────────────────────────────────────────────

PAYLOAD_SCHEMA = {
    "source":       PayloadSchemaType.KEYWORD,
    "category":     PayloadSchemaType.KEYWORD,
    "sub_category": PayloadSchemaType.KEYWORD,
    "gender":       PayloadSchemaType.KEYWORD,
    "color":        PayloadSchemaType.KEYWORD,
    "brand":        PayloadSchemaType.KEYWORD,
    "price":        PayloadSchemaType.FLOAT,
    "is_available": PayloadSchemaType.BOOL,
}

# Payload keys stored per point (full metadata)
FULL_PAYLOAD_KEYS = [
    "pg_id", "source", "name", "category", "sub_category",
    "price", "currency", "color", "colors_available",
    "brand", "description", "image_url", "product_url",
    "sizes_available", "gender", "is_available", "tags",
]


class QdrantManager:
    """
    Manages the Qdrant vector collection for fashion catalog items.
    Use as a singleton via get_qdrant_manager().
    """

    def __init__(self):
        settings = get_settings()
        self.collection = settings.qdrant_collection
        self.dimension = settings.embedding_dimension

        # Async client for production use
        self._async_client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,
        )
        # Sync client for one-time setup operations
        self._sync_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,
        )

    # ─────────────────────────────────────────
    #  Collection Lifecycle
    # ─────────────────────────────────────────

    async def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist, with all payload indexes."""
        collections = await self._async_client.get_collections()
        existing = [c.name for c in collections.collections]

        if self.collection not in existing:
            await self._async_client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("qdrant.collection_created", name=self.collection, dimension=self.dimension)

            # Create payload indexes for fast filtered search
            for field, schema_type in PAYLOAD_SCHEMA.items():
                await self._async_client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema_type,
                )
            logger.info("qdrant.indexes_created", fields=list(PAYLOAD_SCHEMA.keys()))
        else:
            logger.info("qdrant.collection_exists", name=self.collection)

    async def drop_collection(self) -> None:
        """Drop and recreate collection (use for full re-index)."""
        await self._async_client.delete_collection(self.collection)
        logger.warning("qdrant.collection_dropped", name=self.collection)

    async def get_collection_info(self) -> dict:
        info = await self._async_client.get_collection(self.collection)
        return {
            "name": self.collection,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }

    # ─────────────────────────────────────────
    #  Upsert
    # ─────────────────────────────────────────

    async def upsert_items(self, items: list[dict], vectors: list[list[float]]) -> int:
        """
        Upsert fashion items with their embedding vectors.
        items: list of payload dicts (must contain pg_id)
        vectors: corresponding embedding vectors
        Returns count of upserted items.
        """
        if not items:
            return 0

        points = []
        for item, vector in zip(items, vectors):
            point_id = item.get("qdrant_id") or str(uuid.uuid4())
            payload = {k: item.get(k) for k in FULL_PAYLOAD_KEYS if item.get(k) is not None}
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        await self._async_client.upsert(
            collection_name=self.collection,
            points=points,
            wait=True,
        )
        logger.info("qdrant.upserted", count=len(points))
        return len(points)

    # ─────────────────────────────────────────
    #  Search
    # ─────────────────────────────────────────

    async def search(
        self,
        query_vector: list[float],
        category: Optional[str] = None,
        gender: Optional[str] = None,
        color: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        source: Optional[str] = None,
        only_available: bool = True,
        limit: int = 10,
        score_threshold: float = 0.3,
    ) -> list[dict]:
        """
        Semantic search with optional metadata pre-filters.
        Filters are applied BEFORE vector search (Qdrant's filtered search).
        """
        conditions: list[Any] = []

        if only_available:
            conditions.append(FieldCondition(key="is_available", match=MatchValue(value=True)))
        if category:
            conditions.append(FieldCondition(key="category", match=MatchValue(value=category.lower())))
        if gender and gender.lower() != "unisex":
            # Accept exact gender OR unisex
            conditions.append(
                qmodels.Filter(
                    should=[
                        FieldCondition(key="gender", match=MatchValue(value=gender.lower())),
                        FieldCondition(key="gender", match=MatchValue(value="unisex")),
                    ]
                )
            )
        if color:
            conditions.append(FieldCondition(key="color", match=MatchValue(value=color.lower())))
        if source:
            conditions.append(FieldCondition(key="source", match=MatchValue(value=source.lower())))
        if min_price is not None or max_price is not None:
            conditions.append(
                FieldCondition(
                    key="price",
                    range=Range(
                        gte=min_price if min_price is not None else None,
                        lte=max_price if max_price is not None else None,
                    ),
                )
            )

        query_filter = Filter(must=conditions) if conditions else None

        results = await self._async_client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        return [
            {**r.payload, "score": r.score, "qdrant_id": str(r.id)}
            for r in results
        ]

    async def multi_category_search(
        self,
        query_vector: list[float],
        categories: list[str],
        filters: dict,
        limit_per_category: int = 5,
    ) -> dict[str, list[dict]]:
        """
        Runs parallel searches for multiple categories (top, bottom, shoes).
        Returns dict keyed by category.
        """
        import asyncio

        async def _search_category(cat: str) -> tuple[str, list[dict]]:
            results = await self.search(
                query_vector=query_vector,
                category=cat,
                limit=limit_per_category,
                **{k: v for k, v in filters.items() if k != "category"},
            )
            return cat, results

        tasks = [_search_category(cat) for cat in categories]
        results = await asyncio.gather(*tasks)
        return dict(results)

    # ─────────────────────────────────────────
    #  Delete / Update
    # ─────────────────────────────────────────

    async def delete_by_source(self, source: str) -> int:
        """Delete all items from a specific source (use before re-scraping)."""
        result = await self._async_client.delete(
            collection_name=self.collection,
            points_selector=qmodels.FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                )
            ),
        )
        logger.info("qdrant.deleted_by_source", source=source)
        return result.status

    async def close(self):
        await self._async_client.close()


# ─────────────────────────────────────────────
#  Singleton
# ─────────────────────────────────────────────

_qdrant_manager: QdrantManager | None = None


def get_qdrant_manager() -> QdrantManager:
    global _qdrant_manager
    if _qdrant_manager is None:
        _qdrant_manager = QdrantManager()
    return _qdrant_manager