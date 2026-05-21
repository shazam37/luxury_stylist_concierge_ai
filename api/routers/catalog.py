"""
GET /api/v1/catalog  — Browse and search the fashion catalog.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import select, func, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import CatalogItemResponse, CatalogSearchRequest, CatalogStatsResponse
from db.postgres import get_db
from db.models import CatalogItem

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get("/catalog", summary="List catalog items with optional filters")
async def list_catalog(
    category: str | None = Query(default=None, description="top | bottom | shoes | accessory | outerwear"),
    gender: str | None = Query(default=None),
    color: str | None = Query(default=None),
    source: str | None = Query(default=None, description="zara | hm | myntra"),
    min_price: float | None = Query(default=None, ge=0),
    max_price: float | None = Query(default=None, ge=0),
    only_available: bool = Query(default=True),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Browse catalog items with filters. Supports pagination."""
    q = select(CatalogItem)

    if category:
        q = q.where(CatalogItem.category == category.lower())
    if gender:
        q = q.where(CatalogItem.gender.in_([gender.lower(), "unisex"]))
    if color:
        q = q.where(CatalogItem.color.ilike(f"%{color}%"))
    if source:
        q = q.where(CatalogItem.source == source.lower())
    if min_price is not None:
        q = q.where(CatalogItem.price >= min_price)
    if max_price is not None:
        q = q.where(CatalogItem.price <= max_price)
    if only_available:
        q = q.where(CatalogItem.is_available == True)

    q = q.limit(limit).offset(offset).order_by(CatalogItem.scraped_at.desc())

    result = await db.execute(q)
    items = result.scalars().all()

    return {
        "items": [_to_response(item) for item in items],
        "count": len(items),
        "offset": offset,
        "limit": limit,
    }


@router.get("/catalog/search", summary="Semantic search over the catalog")
async def semantic_search(
    q: str = Query(..., description="Natural language search query"),
    category: str | None = Query(default=None),
    gender: str | None = Query(default=None),
    min_price: float | None = Query(default=None),
    max_price: float | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
):
    """Semantic vector search over the fashion catalog."""
    from embeddings.embedder import get_embedder
    from embeddings.indexer import get_qdrant_manager

    embedder = get_embedder()
    qdrant = get_qdrant_manager()

    vector = await embedder.embed_query(q)
    results = await qdrant.search(
        query_vector=vector,
        category=category,
        gender=gender,
        min_price=min_price,
        max_price=max_price,
        limit=limit,
    )

    return {"query": q, "results": results, "count": len(results)}


@router.get("/catalog/stats", response_model=CatalogStatsResponse, summary="Catalog statistics")
async def catalog_stats(db: AsyncSession = Depends(get_db)):
    """Returns counts, breakdowns, and vector DB stats for the catalog."""
    from embeddings.indexer import get_qdrant_manager
    qdrant = get_qdrant_manager()

    total_q = await db.execute(select(func.count(CatalogItem.id)))
    total = total_q.scalar() or 0

    # By category
    cat_q = await db.execute(
        select(CatalogItem.category, func.count(CatalogItem.id))
        .group_by(CatalogItem.category)
    )
    by_category = {row[0]: row[1] for row in cat_q.fetchall() if row[0]}

    # By source
    src_q = await db.execute(
        select(CatalogItem.source, func.count(CatalogItem.id))
        .group_by(CatalogItem.source)
    )
    by_source = {row[0]: row[1] for row in src_q.fetchall() if row[0]}

    # By gender
    gen_q = await db.execute(
        select(CatalogItem.gender, func.count(CatalogItem.id))
        .group_by(CatalogItem.gender)
    )
    by_gender = {row[0] or "unknown": row[1] for row in gen_q.fetchall()}

    # Price range
    price_q = await db.execute(
        select(func.min(CatalogItem.price), func.max(CatalogItem.price), func.avg(CatalogItem.price))
        .where(CatalogItem.price.isnot(None))
    )
    price_row = price_q.fetchone()
    price_range = {
        "min": float(price_row[0] or 0),
        "max": float(price_row[1] or 0),
        "avg": round(float(price_row[2] or 0), 2),
    }

    # Qdrant vector count
    try:
        qdrant_info = await qdrant.get_collection_info()
        qdrant_vectors = qdrant_info.get("points_count", 0)
    except Exception:
        qdrant_vectors = -1

    return CatalogStatsResponse(
        total_items=total,
        by_category=by_category,
        by_source=by_source,
        by_gender=by_gender,
        price_range=price_range,
        qdrant_vectors=qdrant_vectors,
    )


@router.get("/catalog/{item_id}", response_model=CatalogItemResponse, summary="Get item by ID")
async def get_catalog_item(item_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(CatalogItem).where(CatalogItem.id == item_id))
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return _to_response(item)


def _to_response(item: CatalogItem) -> CatalogItemResponse:
    return CatalogItemResponse(
        id=str(item.id),
        source=item.source,
        name=item.name,
        category=item.category,
        sub_category=item.sub_category,
        price=float(item.price) if item.price else None,
        currency=item.currency or "USD",
        color=item.color,
        brand=item.brand,
        description=item.description,
        image_url=item.image_url,
        product_url=item.product_url,
        gender=item.gender,
        sizes_available=item.sizes_available or [],
        is_available=item.is_available,
        tags=item.tags or [],
    )