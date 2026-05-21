"""
/api/v1/wardrobe — User wardrobe management.
Users can save items they already own; the agent references these when making recommendations.
"""
from __future__ import annotations

import json
import uuid
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import WardrobeItemCreate, WardrobeItemResponse
from db.postgres import get_db
from db.models import WardrobeItem

router = APIRouter()
logger = structlog.get_logger(__name__)

ANON_USER_ID = "00000000-0000-0000-0000-000000000000"


def _safe_tags(tags) -> list:
    """Deserialize tags whether stored as list or JSON string."""
    if not tags:
        return []
    if isinstance(tags, list):
        return tags
    if isinstance(tags, str):
        try:
            return json.loads(tags)
        except Exception:
            return []
    return []


@router.get("/wardrobe", summary="List wardrobe items")
async def list_wardrobe(
    user_id: str = Query(default=ANON_USER_ID),
    category: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    q = select(WardrobeItem).where(WardrobeItem.user_id == user_id)
    if category:
        q = q.where(WardrobeItem.category == category.lower())
    q = q.order_by(WardrobeItem.created_at.desc())
    result = await db.execute(q)
    items = result.scalars().all()
    return {"items": [_to_response(i) for i in items], "count": len(items)}


@router.post("/wardrobe", response_model=WardrobeItemResponse, summary="Add item to wardrobe", status_code=201)
async def add_wardrobe_item(
    item: WardrobeItemCreate,
    user_id: str = Query(default=ANON_USER_ID),
    db: AsyncSession = Depends(get_db),
):
    # Serialize tags as JSON string for cross-DB compatibility (Postgres ARRAY / SQLite TEXT)
    tags_value = item.tags if item.tags is not None else []

    record = WardrobeItem(
        id=str(uuid.uuid4()),
        user_id=user_id,
        name=item.name,
        category=item.category.value,
        color=item.color,
        brand=item.brand,
        description=item.description,
        image_url=item.image_url,
        tags=tags_value,
    )
    db.add(record)
    try:
        await db.commit()
        await db.refresh(record)
    except Exception:
        # Fallback: use raw SQL for SQLite compatibility
        await db.rollback()
        item_id = str(uuid.uuid4())
        await db.execute(
            text("""
                INSERT INTO wardrobe_items
                  (id, user_id, name, category, color, brand, description, image_url, tags)
                VALUES
                  (:id, :user_id, :name, :category, :color, :brand, :description, :image_url, :tags)
            """),
            {
                "id": item_id,
                "user_id": user_id,
                "name": item.name,
                "category": item.category.value,
                "color": item.color,
                "brand": item.brand,
                "description": item.description,
                "image_url": item.image_url,
                "tags": json.dumps(tags_value),
            },
        )
        await db.commit()
        # Build synthetic response
        from datetime import datetime, timezone
        return WardrobeItemResponse(
            id=item_id,
            name=item.name,
            category=item.category.value,
            color=item.color,
            brand=item.brand,
            description=item.description,
            image_url=item.image_url,
            tags=tags_value,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    logger.info("wardrobe.item_added", user_id=user_id, name=item.name)
    return _to_response(record)


@router.delete("/wardrobe/{item_id}", summary="Remove item from wardrobe")
async def delete_wardrobe_item(
    item_id: str,
    user_id: str = Query(default=ANON_USER_ID),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(WardrobeItem).where(
            WardrobeItem.id == item_id,
            WardrobeItem.user_id == user_id,
        )
    )
    item = result.scalar_one_or_none()
    if not item:
        # Also try raw query (for SQLite fallback rows)
        raw = await db.execute(
            text("SELECT id FROM wardrobe_items WHERE id = :id AND user_id = :uid"),
            {"id": item_id, "uid": user_id},
        )
        if not raw.fetchone():
            raise HTTPException(status_code=404, detail="Item not found")
        await db.execute(
            text("DELETE FROM wardrobe_items WHERE id = :id"), {"id": item_id}
        )
        await db.commit()
        return {"status": "ok", "deleted": item_id}

    await db.delete(item)
    await db.commit()
    return {"status": "ok", "deleted": item_id}


def _to_response(item: WardrobeItem) -> WardrobeItemResponse:
    return WardrobeItemResponse(
        id=str(item.id),
        name=item.name,
        category=item.category,
        color=item.color,
        brand=item.brand,
        description=item.description,
        image_url=item.image_url,
        tags=_safe_tags(item.tags),
        created_at=item.created_at.isoformat() if item.created_at else None,
    )