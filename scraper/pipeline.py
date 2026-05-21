"""
Scraper Pipeline — orchestrates all scrapers end-to-end.

Flow:
  1. Run Zara, H&M, Myntra scrapers concurrently
  2. Deduplicate scraped items
  3. Persist raw items to PostgreSQL (catalog_items table)
  4. Embed each item using the configured embedding model
  5. Upsert vectors to Qdrant with full metadata payload
  6. Return scrape statistics

Designed to run as a background job triggered via POST /api/v1/scraper/trigger.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from typing import Optional

import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
#  Source Registry
# ─────────────────────────────────────────────

def _get_scraper(source: str):
    """Return instantiated scraper for the given source name."""
    if source == "zara":
        from scraper.zara_scraper import ZaraScraper
        return ZaraScraper()
    elif source == "hm":
        from scraper.hm_scraper import HMScraper
        return HMScraper()
    elif source == "myntra":
        from scraper.myntra_scraper import MyntraScraper
        return MyntraScraper()
    else:
        raise ValueError(f"Unknown scraper source: {source}")


KNOWN_SOURCES = ["zara", "hm", "myntra"]


# ─────────────────────────────────────────────
#  Deduplication
# ─────────────────────────────────────────────

def _item_fingerprint(item: dict) -> str:
    """Create a stable hash for deduplication based on name + source + price."""
    key = f"{item.get('source', '')}|{(item.get('name') or '').lower().strip()}|{item.get('price', '')}"
    return hashlib.md5(key.encode()).hexdigest()


def deduplicate(items: list[dict]) -> list[dict]:
    """Remove duplicate items based on name+source+price fingerprint."""
    seen = set()
    unique = []
    for item in items:
        fp = _item_fingerprint(item)
        if fp not in seen:
            seen.add(fp)
            unique.append(item)
    return unique


# ─────────────────────────────────────────────
#  Postgres Persistence
# ─────────────────────────────────────────────

async def _persist_to_postgres(items: list[dict], reindex: bool = False) -> list[dict]:
    """
    Upsert scraped items to the catalog_items table.
    Returns items with their pg_id populated (needed for Qdrant cross-ref).
    """
    from db.postgres import get_db_session
    from db.models import CatalogItem
    from sqlalchemy import select

    enriched = []

    async with get_db_session() as db:
        for item in items:
            try:
                # Check if item already exists (by external_id)
                existing = None
                if item.get("external_id"):
                    result = await db.execute(
                        select(CatalogItem).where(
                            CatalogItem.external_id == item["external_id"],
                            CatalogItem.source == item["source"],
                        )
                    )
                    existing = result.scalar_one_or_none()

                if existing and not reindex:
                    # Keep existing record, just update qdrant_id if needed
                    item["pg_id"] = str(existing.id)
                    item["qdrant_id"] = existing.qdrant_id or str(uuid.uuid4())
                    enriched.append(item)
                    continue

                # Assign a Qdrant ID now (UUID string, stored in both tables)
                qdrant_id = str(uuid.uuid4())
                item["qdrant_id"] = qdrant_id

                if existing and reindex:
                    # Update existing record
                    for field in [
                        "name", "category", "sub_category", "price", "currency",
                        "color", "brand", "description", "image_url", "product_url",
                        "sizes_available", "gender", "is_available", "tags",
                        "colors_available", "raw_data",
                    ]:
                        if item.get(field) is not None:
                            setattr(existing, field, item[field])
                    existing.qdrant_id = qdrant_id
                    item["pg_id"] = str(existing.id)
                else:
                    # Insert new record
                    record = CatalogItem(
                        id=str(uuid.uuid4()),
                        external_id=item.get("external_id"),
                        source=item["source"],
                        name=item["name"],
                        category=item.get("category", "top"),
                        sub_category=item.get("sub_category"),
                        price=item.get("price"),
                        currency=item.get("currency", "USD"),
                        color=item.get("color"),
                        colors_available=item.get("colors_available", []),
                        brand=item.get("brand"),
                        description=item.get("description"),
                        image_url=item.get("image_url"),
                        product_url=item.get("product_url"),
                        sizes_available=item.get("sizes_available", []),
                        gender=item.get("gender", "unisex"),
                        is_available=item.get("is_available", True),
                        tags=item.get("tags", []),
                        raw_data=item.get("raw_data", {}),
                        qdrant_id=qdrant_id,
                    )
                    db.add(record)
                    item["pg_id"] = str(record.id)

                enriched.append(item)

            except Exception as e:
                logger.warning("pipeline.postgres_item_error", error=str(e)[:100], name=item.get("name", "?"))
                continue

        await db.commit()

    logger.info("pipeline.postgres_persisted", count=len(enriched))
    return enriched


# ─────────────────────────────────────────────
#  Embedding + Qdrant Upsert
# ─────────────────────────────────────────────

async def _embed_and_index(items: list[dict], batch_size: int = 50) -> int:
    """
    Embed all items and upsert them to Qdrant.
    Returns count of successfully indexed items.
    """
    from embeddings.embedder import get_embedder
    from embeddings.indexer import get_qdrant_manager

    embedder = get_embedder()
    qdrant = get_qdrant_manager()

    indexed = 0

    # Process in batches to manage memory and API rate limits
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]

        try:
            vectors = await embedder.embed_items(batch)

            count = await qdrant.upsert_items(batch, vectors)
            indexed += count

            logger.info(
                "pipeline.batch_indexed",
                batch=i // batch_size + 1,
                total_batches=(len(items) + batch_size - 1) // batch_size,
                indexed_so_far=indexed,
            )

            # Small pause between batches to avoid rate limits
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error("pipeline.batch_error", batch_start=i, error=str(e)[:150])

    return indexed


# ─────────────────────────────────────────────
#  Main Pipeline Entry Point
# ─────────────────────────────────────────────

async def run_pipeline(source: str = "all", reindex: bool = False) -> dict:
    """
    Main entry point for the scraper pipeline.

    Args:
        source: "all" | "zara" | "hm" | "myntra"
        reindex: If True, drop and re-index existing items for this source.

    Returns:
        dict with scraped/indexed counts and any errors.
    """
    start = time.perf_counter()
    logger.info("pipeline.starting", source=source, reindex=reindex)

    # ── 1. Determine which scrapers to run ────
    sources_to_run = KNOWN_SOURCES if source == "all" else [source]
    all_errors: list[str] = []

    # ── 2. Run scrapers concurrently ──────────
    scrape_tasks = [_run_scraper(s) for s in sources_to_run]
    scraper_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

    all_items: list[dict] = []
    for i, result in enumerate(scraper_results):
        src = sources_to_run[i]
        if isinstance(result, Exception):
            msg = f"Scraper {src} failed: {str(result)[:100]}"
            logger.error("pipeline.scraper_failed", source=src, error=str(result))
            all_errors.append(msg)
        elif isinstance(result, list):
            logger.info("pipeline.scraper_done", source=src, count=len(result))
            all_items.extend(result)

    scraped_count = len(all_items)
    logger.info("pipeline.all_scraped", total=scraped_count)

    if not all_items:
        return {
            "scraped": 0,
            "indexed": 0,
            "errors": all_errors,
            "message": "No items scraped. Check scraper logs.",
            "elapsed_s": round(time.perf_counter() - start, 2),
        }

    # ── 3. Deduplicate ────────────────────────
    all_items = deduplicate(all_items)
    logger.info("pipeline.deduplicated", unique=len(all_items), dropped=scraped_count - len(all_items))

    # ── 4. Persist to Postgres ────────────────
    try:
        all_items = await _persist_to_postgres(all_items, reindex=reindex)
    except Exception as e:
        msg = f"Postgres persistence failed: {str(e)[:150]}"
        logger.error("pipeline.postgres_failed", error=str(e))
        all_errors.append(msg)

    # ── 5. Embed + index in Qdrant ────────────
    indexed_count = 0
    try:
        indexed_count = await _embed_and_index(all_items)
    except Exception as e:
        msg = f"Qdrant indexing failed: {str(e)[:150]}"
        logger.error("pipeline.qdrant_failed", error=str(e))
        all_errors.append(msg)

    elapsed = round(time.perf_counter() - start, 2)

    summary = {
        "scraped": scraped_count,
        "unique": len(all_items),
        "indexed": indexed_count,
        "sources": sources_to_run,
        "errors": all_errors,
        "elapsed_s": elapsed,
        "message": (
            f"Pipeline complete: {scraped_count} scraped, "
            f"{len(all_items)} unique, {indexed_count} indexed in {elapsed}s."
        ),
    }

    logger.info("pipeline.complete", **{k: v for k, v in summary.items() if k != "errors"})
    return summary


async def _run_scraper(source: str) -> list[dict]:
    """Run a single scraper and return items."""
    scraper = _get_scraper(source)
    return await scraper.scrape(target_count=60)


# ─────────────────────────────────────────────
#  Seed catalog with mock data (dev/testing)
# ─────────────────────────────────────────────

async def seed_mock_catalog(count: int = 120) -> dict:
    """
    Seed the catalog with realistic mock fashion items for development.
    Useful when you can't run the real scrapers (rate limits, offline, etc.).
    """
    import random

    mock_tops = [
        {"name": "Classic White Oxford Shirt", "color": "white", "sub_category": "shirt", "price": 49.99},
        {"name": "Navy Linen Shirt", "color": "navy", "sub_category": "shirt", "price": 59.99},
        {"name": "Slim Fit White T-Shirt", "color": "white", "sub_category": "t-shirt", "price": 19.99},
        {"name": "Striped Breton T-Shirt", "color": "blue", "sub_category": "t-shirt", "price": 24.99},
        {"name": "Relaxed Graphic Tee", "color": "black", "sub_category": "t-shirt", "price": 22.99},
        {"name": "Slim Fit Polo Shirt", "color": "navy", "sub_category": "polo", "price": 34.99},
        {"name": "Linen Blend Casual Shirt", "color": "beige", "sub_category": "shirt", "price": 44.99},
        {"name": "Oversized Cotton T-Shirt", "color": "grey", "sub_category": "t-shirt", "price": 27.99},
        {"name": "Printed Camp Collar Shirt", "color": "white", "sub_category": "shirt", "price": 39.99},
        {"name": "Ribbed Knit Polo", "color": "cream", "sub_category": "polo", "price": 42.99},
        {"name": "Lightweight Merino Sweater", "color": "grey", "sub_category": "sweater", "price": 89.99},
        {"name": "Cotton Crew Neck Tee", "color": "white", "sub_category": "t-shirt", "price": 18.99},
        {"name": "Floral Print Casual Shirt", "color": "blue", "sub_category": "shirt", "price": 35.99},
        {"name": "Textured Knit Polo", "color": "olive", "sub_category": "polo", "price": 45.99},
        {"name": "Essential Crewneck Sweatshirt", "color": "grey", "sub_category": "sweatshirt", "price": 55.99},
    ]

    mock_bottoms = [
        {"name": "Slim Fit Dark Navy Chinos", "color": "navy", "sub_category": "chinos", "price": 59.99},
        {"name": "Tapered Khaki Chinos", "color": "khaki", "sub_category": "chinos", "price": 54.99},
        {"name": "Slim Straight Jeans", "color": "blue", "sub_category": "jeans", "price": 69.99},
        {"name": "Linen Blend Trousers", "color": "beige", "sub_category": "trousers", "price": 64.99},
        {"name": "Jogger Trousers", "color": "grey", "sub_category": "joggers", "price": 44.99},
        {"name": "Relaxed Linen Shorts", "color": "white", "sub_category": "shorts", "price": 34.99},
        {"name": "Slim Fit Dress Trousers", "color": "black", "sub_category": "trousers", "price": 79.99},
        {"name": "Cargo Pants", "color": "olive", "sub_category": "cargo", "price": 59.99},
        {"name": "Stretch Slim Chinos", "color": "grey", "sub_category": "chinos", "price": 62.99},
        {"name": "Dark Wash Skinny Jeans", "color": "blue", "sub_category": "jeans", "price": 74.99},
        {"name": "Straight Leg Trousers", "color": "beige", "sub_category": "trousers", "price": 55.99},
        {"name": "Pull-On Linen Shorts", "color": "navy", "sub_category": "shorts", "price": 39.99},
    ]

    mock_shoes = [
        {"name": "White Leather Sneakers", "color": "white", "sub_category": "sneakers", "price": 89.99},
        {"name": "Suede Loafers", "color": "tan", "sub_category": "loafers", "price": 119.99},
        {"name": "Classic Canvas Espadrilles", "color": "navy", "sub_category": "espadrilles", "price": 45.99},
        {"name": "Leather Derby Shoes", "color": "brown", "sub_category": "derby", "price": 139.99},
        {"name": "Slip-On Boat Shoes", "color": "tan", "sub_category": "boat shoes", "price": 79.99},
        {"name": "Running Sneakers", "color": "white", "sub_category": "sneakers", "price": 99.99},
        {"name": "Chelsea Boots", "color": "black", "sub_category": "boots", "price": 149.99},
        {"name": "Leather Sandals", "color": "tan", "sub_category": "sandals", "price": 69.99},
    ]

    sources = ["zara", "hm", "myntra"]
    brands_by_source = {"zara": "Zara", "hm": "H&M", "myntra": None}
    genders = ["male", "female", "unisex"]

    items = []
    all_templates = (
        [(t, "top") for t in mock_tops] +
        [(t, "bottom") for t in mock_bottoms] +
        [(t, "shoes") for t in mock_shoes]
    )

    for i in range(count):
        template, cat = random.choice(all_templates)
        source = random.choice(sources)
        gender = random.choice(genders[:2])  # male or female

        item = {
            "external_id": f"{source}_mock_{i:04d}",
            "source": source,
            "brand": brands_by_source[source] or random.choice(["Allen Solly", "US Polo", "Arrow", "Van Heusen"]),
            "name": template["name"],
            "category": cat,
            "sub_category": template.get("sub_category", ""),
            "color": template.get("color"),
            "price": template["price"] * (1 + random.uniform(-0.2, 0.3)),
            "currency": "INR" if source == "myntra" else "USD",
            "description": (
                f"A premium quality {template['name'].lower()} crafted for modern style. "
                f"Features a comfortable fit with exceptional fabric quality. "
                f"Perfect for {random.choice(['casual outings', 'office wear', 'weekend getaways', 'evening events'])}."
            ),
            "image_url": f"https://via.placeholder.com/400x500?text={template['name'].replace(' ', '+')}",
            "product_url": f"https://{source}.com/products/{source}-{i:04d}",
            "sizes_available": random.sample(["XS", "S", "M", "L", "XL", "XXL"], k=random.randint(2, 5)),
            "gender": gender,
            "is_available": True,
            "colors_available": [template.get("color", "white")],
            "tags": [],
            "raw_data": {"mock": True},
        }

        # Round price
        item["price"] = round(item["price"], 2)
        items.append(item)

    # Remove duplicates and run through pipeline stages
    items = deduplicate(items)

    # Validate each item (apply auto-tags etc.)
    from scraper.base_scraper import BaseScraper
    class _TempScraper(BaseScraper):
        SOURCE_NAME = "mock"
        async def _scrape_source(self, target_count): return []

    tmp = _TempScraper()
    items = [tmp._validate_and_enrich(i) for i in items]
    items = [i for i in items if i]

    # Persist to postgres
    items = await _persist_to_postgres(items, reindex=True)

    # Embed and index
    indexed = await _embed_and_index(items)

    return {
        "scraped": len(items),
        "indexed": indexed,
        "message": f"Mock catalog seeded: {len(items)} items, {indexed} indexed.",
        "errors": [],
    }