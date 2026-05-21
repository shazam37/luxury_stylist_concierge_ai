# Architecture — Luxury AI Stylist Concierge

## Overview

A fully agentic, model-agnostic fashion concierge built on **FastAPI + LangGraph + Qdrant**.
Users send natural language prompts and receive complete outfit recommendations backed by
real scraped inventory, semantic search, and LLM fashion reasoning.

---

## Folder Structure

```
stylist-concierge/
├── config/
│   └── settings.py          # All config, LLM factory, embedding factory
├── scraper/
│   ├── base_scraper.py      # Abstract base: retry, rate-limit, normalisation
│   ├── zara_scraper.py      # Zara internal API + HTML fallback
│   ├── hm_scraper.py        # H&M internal API + HTML fallback
│   ├── myntra_scraper.py    # Myntra search API (bonus 3rd source)
│   └── pipeline.py          # Orchestrator: scrape → deduplicate → embed → index
├── embeddings/
│   ├── embedder.py          # Model-agnostic batch embedder
│   └── indexer.py           # Qdrant collection manager (CRUD + search)
├── agents/
│   ├── state.py             # StylistState TypedDict (all inter-node data)
│   ├── graph.py             # LangGraph StateGraph wiring + entry point
│   └── nodes/
│       ├── intent_parser.py     # Node 1: NLP → structured ParsedIntent
│       ├── cache_manager.py     # Node 2: Semantic cache lookup
│       ├── rag_retriever.py     # Node 3: Qdrant multi-category search
│       ├── fashion_reasoner.py  # Node 4: LLM fashion rules + outfit selection
│       └── response_formatter.py# Node 5: Final schema assembly + cache write
├── api/
│   ├── main.py              # FastAPI app factory, lifespan, middleware
│   ├── routers/
│   │   ├── style.py         # POST /api/v1/style-me (core)
│   │   ├── catalog.py       # GET/SEARCH catalog
│   │   ├── wardrobe.py      # Wardrobe CRUD
│   │   ├── scraper.py       # Scrape job control + seed
│   │   └── health.py        # /health, /ready, /info
│   └── schemas/
│       └── requests.py      # All Pydantic v2 request + response models
├── cache/
│   └── semantic_cache.py    # Redis-backed semantic dedup cache
├── db/
│   ├── postgres.py          # Async SQLAlchemy engine + session
│   └── models.py            # ORM models: User, WardrobeItem, CatalogItem, etc.
├── scripts/
│   └── init_db.sql          # DB schema + indexes + seed anonymous user
├── docker-compose.yml       # Qdrant + Postgres + Redis + App
└── Dockerfile
```

---

## Agent State Machine (LangGraph)

```
START
  │
  ▼ load_wardrobe        (fetch user's owned items from Postgres)
  │
  ▼ intent_parser        (LLM: prompt → ParsedIntent + embed query vector)
  │
  ▼ cache_check          (cosine similarity vs Redis cache entries)
  │
  ├── [cache HIT]  ──→  cache_formatter  ──→  END   (zero LLM cost)
  │
  └── [cache MISS] ──→  rag_retriever
                           │
                           ▼ fashion_reasoner   (LLM: color theory + occasion rules)
                           │
                           ▼ response_formatter (schema assembly + cache write)
                           │
                          END
```

**State schema** (`StylistState`): a single TypedDict that all nodes read and partially update.
LangGraph merges partial updates automatically — nodes only return what they change.

---

## Database Schema

### PostgreSQL tables

| Table | Purpose |
|-------|---------|
| `users` | User profiles with `style_profile` JSONB and `budget_range` JSONB |
| `wardrobe_items` | Items a user owns (referenced by the agent at query time) |
| `catalog_items` | All scraped fashion items with full metadata + `qdrant_id` cross-ref |
| `scrape_jobs` | Job records for background scrape runs (status tracking) |
| `style_requests` | Every agent invocation logged with prompt, response, token usage |
| `saved_outfits` | User-favourited outfits for future reference |

All primary keys are UUIDs. `catalog_items.qdrant_id` links each row to its vector in Qdrant.
GIN trigram index on `catalog_items.name` enables fast fuzzy text search.

### Qdrant collection: `fashion_catalog`

| Field | Type | Purpose |
|-------|------|---------|
| vector | float[1536] | OpenAI `text-embedding-3-small` embedding of rich item text |
| `source` | keyword | zara / hm / myntra — filterable before vector search |
| `category` | keyword | top / bottom / shoes / accessory — **primary filter** |
| `gender` | keyword | male / female / unisex |
| `color` | keyword | primary colour |
| `price` | float | range filter for budget-aware search |
| `is_available` | bool | exclude sold-out items |

All payload fields are **indexed** — Qdrant applies filters *before* ANN search (not post-filter),
so budget + category + gender filtering costs almost nothing.

---

## Model-Agnostic LLM Config

Switching providers requires changing **two** `.env` values:

```env
# Groq (default, fastest)
LLM_PROVIDER=groq
LLM_MODEL=meta-llama/llama-4-maverick-17b-128e-instruct
GROQ_API_KEY=gsk_...

# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-...

# Google
LLM_PROVIDER=google
LLM_MODEL=gemini-2.0-flash
GOOGLE_API_KEY=...
```

`config/settings.py::get_llm()` returns the correct LangChain `BaseChatModel` subclass.
No other code changes needed anywhere.

---

## Prompt Optimisation (Frugal Mindset)

### 1. Semantic Cache
- Every prompt is embedded and stored in Redis with its response.
- Incoming prompts are compared via cosine similarity (threshold: 0.92).
- Cache hits return in **<50ms** with **zero LLM token cost**.
- TTL: 1 hour (configurable). Falls back to in-memory dict if Redis is unavailable.

### 2. Catalog pre-filtering before LLM call
- Qdrant filters by `category`, `gender`, `price`, `is_available` *before* vector search.
- The LLM receives at most **6 items per category** (18 items total) — not the full catalog.
- This keeps the fashion reasoner prompt under ~2000 tokens.

### 3. Intent parser = cheap, reasoner = expensive
- Intent parsing uses a structured JSON prompt with a small output — fast and cheap.
- The fashion reasoner is the only "creative" LLM call; it receives pre-filtered, concise input.

### 4. Batch embeddings
- Items are embedded in batches of 50/100 during the pipeline — avoids per-item API calls.

---

## Scraper Design

Each scraper extends `BaseScraper` and overrides `_scrape_source()`.

**Rate limit bypass strategies:**
- Random `User-Agent` rotation from a pool of 6 real browser UAs
- Randomised delays between requests (1–3s configurable)
- Exponential backoff retry (tenacity) on HTTP errors
- Concurrent per-category scraping (asyncio.gather) with a concurrency cap
- HTML fallback when the primary API endpoint fails

**Output normalisation:**
- Every scraper produces the same dict schema (defined in `make_empty_item()`)
- `_validate_and_enrich()` in BaseScraper fills defaults, infers category/gender, auto-tags
- Items are deduplicated by MD5(source + name + price) before indexing

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/style-me` | Core: NL prompt → outfit recommendation |
| POST | `/api/v1/style-me/feedback` | Rate a recommendation (1-5) |
| GET | `/api/v1/catalog` | Browse catalog with filters + pagination |
| GET | `/api/v1/catalog/search` | Semantic search over catalog |
| GET | `/api/v1/catalog/stats` | Catalog counts, breakdowns, price range |
| GET | `/api/v1/catalog/{id}` | Single item detail |
| GET | `/api/v1/wardrobe` | List user's wardrobe |
| POST | `/api/v1/wardrobe` | Add item to wardrobe |
| DELETE | `/api/v1/wardrobe/{id}` | Remove wardrobe item |
| POST | `/api/v1/scraper/trigger` | Kick off a scrape job |
| POST | `/api/v1/scraper/seed` | Seed mock catalog (dev) |
| GET | `/api/v1/scraper/jobs` | List scrape jobs |
| GET | `/api/v1/scraper/jobs/{id}` | Job status |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe (all services) |
| GET | `/info` | App metadata + active model |

---

## Infrastructure

All services run via `docker-compose up`:

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `qdrant` | qdrant/qdrant:v1.9.2 | 6333/6334 | Vector search |
| `postgres` | postgres:16-alpine | 5432 | Relational data |
| `redis` | redis:7.2-alpine | 6379 | Semantic cache |
| `app` | (local build) | 8000 | FastAPI + LangGraph |

Health checks on all services; `app` waits for all three to be healthy before starting.