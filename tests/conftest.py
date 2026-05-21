"""
Pytest configuration and shared fixtures.
Uses an in-memory SQLite engine and mocked external services
so tests run without Docker, API keys, or network.
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

# ── Use SQLite for tests (no Postgres needed) ──────────────────
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Single event loop for the whole test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """
    Create in-memory SQLite engine with all tables.
    Replaces PostgreSQL-only types (JSONB, ARRAY) with SQLite-compatible equivalents.
    """
    import sqlalchemy
    from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID, ARRAY
    from sqlalchemy import JSON, String, Text

    # Monkey-patch PG-only types to SQLite-safe equivalents for testing
    import db.models as models_module
    original_jsonb = None
    try:
        # Replace JSONB → JSON and ARRAY → Text in the metadata at DDL time
        from sqlalchemy import event
        from sqlalchemy.engine import Engine

        @event.listens_for(Engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            pass  # no-op; WAL not needed for tests

    except Exception:
        pass

    engine = create_async_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create tables using raw SQL to avoid PG-specific type issues
    async with engine.begin() as conn:
        await conn.execute(sqlalchemy.text("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                style_profile TEXT DEFAULT '{}',
                budget_range TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """))
        await conn.execute(sqlalchemy.text("""
            CREATE TABLE IF NOT EXISTS wardrobe_items (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                color TEXT,
                brand TEXT,
                description TEXT,
                image_url TEXT,
                tags TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """))
        await conn.execute(sqlalchemy.text("""
            CREATE TABLE IF NOT EXISTS catalog_items (
                id TEXT PRIMARY KEY,
                external_id TEXT,
                source TEXT NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                sub_category TEXT,
                price REAL,
                currency TEXT DEFAULT 'USD',
                color TEXT,
                colors_available TEXT DEFAULT '[]',
                brand TEXT,
                description TEXT,
                image_url TEXT,
                product_url TEXT,
                sizes_available TEXT DEFAULT '[]',
                gender TEXT,
                is_available INTEGER DEFAULT 1,
                tags TEXT DEFAULT '[]',
                raw_data TEXT DEFAULT '{}',
                qdrant_id TEXT,
                scraped_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """))
        await conn.execute(sqlalchemy.text("""
            CREATE TABLE IF NOT EXISTS scrape_jobs (
                id TEXT PRIMARY KEY,
                source TEXT,
                status TEXT DEFAULT 'pending',
                triggered_by TEXT DEFAULT 'system',
                items_scraped INTEGER DEFAULT 0,
                items_indexed INTEGER DEFAULT 0,
                errors TEXT DEFAULT '[]',
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """))
        await conn.execute(sqlalchemy.text("""
            CREATE TABLE IF NOT EXISTS style_requests (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                prompt TEXT NOT NULL,
                parsed_intent TEXT DEFAULT '{}',
                outfit_response TEXT DEFAULT '{}',
                cache_hit INTEGER DEFAULT 0,
                token_usage TEXT DEFAULT '{}',
                agent_trace TEXT DEFAULT '[]',
                latency_ms INTEGER,
                feedback_score INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """))
        await conn.execute(sqlalchemy.text("""
            CREATE TABLE IF NOT EXISTS saved_outfits (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                style_request_id TEXT,
                outfit_data TEXT NOT NULL,
                name TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """))
        # Seed anonymous user
        await conn.execute(sqlalchemy.text("""
            INSERT OR IGNORE INTO users (id, email, name)
            VALUES ('00000000-0000-0000-0000-000000000000', 'anonymous@stylist.local', 'Anonymous')
        """))

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide a fresh DB session per test, rolled back after."""
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant manager so tests don't need a running Qdrant instance."""
    mock = AsyncMock()
    mock.ensure_collection = AsyncMock(return_value=None)
    mock.get_collection_info = AsyncMock(return_value={
        "name": "fashion_catalog", "points_count": 100,
        "vectors_count": 100, "indexed_vectors_count": 100, "status": "green"
    })
    mock.search = AsyncMock(return_value=MOCK_SEARCH_RESULTS["top"])
    mock.multi_category_search = AsyncMock(return_value=MOCK_SEARCH_RESULTS)
    mock.upsert_items = AsyncMock(return_value=5)
    mock.delete_by_source = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_embedder():
    """Mock embedder so tests don't need an embedding API key."""
    mock = AsyncMock()
    mock.embed_query = AsyncMock(return_value=[0.1] * 1536)
    mock.embed_documents = AsyncMock(return_value=[[0.1] * 1536] * 10)
    mock.embed_items = AsyncMock(return_value=[[0.1] * 1536] * 10)
    mock.cosine_similarity = MagicMock(return_value=0.5)
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM so tests don't need API keys."""
    mock = AsyncMock()
    intent_response = MagicMock()
    intent_response.content = json.dumps({
        "occasion": "yacht party",
        "owned_items": [{"name": "dark navy chinos", "category": "bottom", "color": "navy"}],
        "requested_categories": ["top", "shoes"],
        "style_keywords": ["nautical", "summer", "elegant"],
        "color_palette": ["navy", "white", "tan"],
        "gender": "male",
        "season": "summer",
        "formality": "smart_casual",
        "budget_min": None,
        "budget_max": None,
        "vibe": "effortlessly chic yacht club aesthetic"
    })
    intent_response.usage_metadata = {"input_tokens": 200, "output_tokens": 150}

    fashion_response = MagicMock()
    fashion_response.content = json.dumps({
        "primary_outfit": {
            "top": {
                "name": "Classic White Linen Shirt",
                "brand": "Zara",
                "price": 49.99,
                "color": "white",
                "image_url": "https://example.com/shirt.jpg",
                "product_url": "https://zara.com/shirt",
                "source": "zara",
                "reason": "Crisp white linen is the quintessential yacht club top"
            },
            "shoes": {
                "name": "Canvas Espadrilles Navy",
                "brand": "H&M",
                "price": 45.99,
                "color": "navy",
                "image_url": "https://example.com/shoes.jpg",
                "product_url": "https://hm.com/shoes",
                "source": "hm",
                "reason": "Espadrilles are the perfect nautical shoe"
            },
            "bottom": None,
            "accessory": None,
            "stylist_note": (
                "The crisp white linen shirt against your navy chinos is the platonic ideal "
                "of yacht-club dressing. The espadrilles ground the look with authentic "
                "Mediterranean flair without sacrificing comfort on deck."
            )
        },
        "alternatives": [
            {
                "top": {
                    "name": "Striped Breton T-Shirt",
                    "brand": "H&M",
                    "price": 24.99,
                    "color": "blue",
                    "image_url": "https://example.com/tee.jpg",
                    "product_url": "https://hm.com/tee",
                    "source": "hm",
                    "reason": "A Breton stripe is the most nautical pattern"
                },
                "shoes": {
                    "name": "Suede Loafers Tan",
                    "brand": "Zara",
                    "price": 119.99,
                    "color": "tan",
                    "image_url": "https://example.com/loafers.jpg",
                    "product_url": "https://zara.com/loafers",
                    "source": "zara",
                    "reason": "Loafers elevate the casual stripe"
                },
                "bottom": None,
                "stylist_note": "A relaxed Breton stripe with tan loafers — effortlessly coastal."
            }
        ],
        "style_tags": ["nautical", "summer", "smart_casual", "linen"],
        "color_story": "Navy and white with tan accents — a timeless maritime palette."
    })
    fashion_response.usage_metadata = {"input_tokens": 800, "output_tokens": 400}

    mock.ainvoke = AsyncMock(side_effect=[intent_response, fashion_response])
    return mock


@pytest.fixture
def mock_cache():
    """Mock semantic cache."""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)  # default: cache miss
    mock.set = AsyncMock(return_value="test-cache-id")
    mock.invalidate_all = AsyncMock(return_value=0)
    return mock


@pytest_asyncio.fixture
async def app_client(db_session, mock_qdrant, mock_embedder, mock_cache):
    """Full async test client with mocked external services."""
    from api.main import create_app
    from db.postgres import get_db

    app = create_app()

    # Override DB dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with patch("embeddings.indexer.get_qdrant_manager", return_value=mock_qdrant), \
         patch("cache.semantic_cache.get_semantic_cache", return_value=mock_cache), \
         patch("embeddings.embedder.get_embedder", return_value=mock_embedder):

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client


# ─────────────────────────────────────────────
#  Shared mock data
# ─────────────────────────────────────────────

MOCK_SEARCH_RESULTS = {
    "top": [
        {
            "name": "Classic White Linen Shirt",
            "brand": "Zara",
            "category": "top",
            "sub_category": "shirt",
            "color": "white",
            "price": 49.99,
            "currency": "USD",
            "image_url": "https://example.com/shirt.jpg",
            "product_url": "https://zara.com/shirt",
            "source": "zara",
            "gender": "male",
            "is_available": True,
            "score": 0.92,
            "qdrant_id": "test-top-1",
            "tags": ["linen", "summer", "white"],
        },
        {
            "name": "Striped Breton T-Shirt",
            "brand": "H&M",
            "category": "top",
            "sub_category": "t-shirt",
            "color": "blue",
            "price": 24.99,
            "currency": "USD",
            "image_url": "https://example.com/tee.jpg",
            "product_url": "https://hm.com/tee",
            "source": "hm",
            "gender": "male",
            "is_available": True,
            "score": 0.87,
            "qdrant_id": "test-top-2",
            "tags": ["nautical", "summer"],
        },
    ],
    "shoes": [
        {
            "name": "Canvas Espadrilles Navy",
            "brand": "H&M",
            "category": "shoes",
            "sub_category": "espadrilles",
            "color": "navy",
            "price": 45.99,
            "currency": "USD",
            "image_url": "https://example.com/shoes.jpg",
            "product_url": "https://hm.com/shoes",
            "source": "hm",
            "gender": "male",
            "is_available": True,
            "score": 0.89,
            "qdrant_id": "test-shoe-1",
            "tags": ["nautical", "canvas"],
        },
    ],
    "bottom": [],
}

SAMPLE_STYLE_REQUEST = {
    "prompt": "I have dark navy chinos. What t-shirt and shoes for a summer yacht party?",
    "gender": "male",
    "include_alternatives": True,
    "include_accessories": False,
}