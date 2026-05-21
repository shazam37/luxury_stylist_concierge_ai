# 👔 Stylist Concierge — Luxury AI Fashion Recommendation API

An agentic, model-agnostic fashion concierge powered by **LangGraph + FastAPI + Qdrant**.

Send a natural language styling prompt → receive a complete, beautifully reasoned outfit recommendation backed by real scraped inventory and LLM fashion intelligence.

---

## Quick Demo

```bash
curl -X POST http://localhost:8000/api/v1/style-me \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I have dark navy chinos. What t-shirt and shoes for a summer yacht party?"}'
```

```json
{
  "request_id": "a1b2c3d4-...",
  "cache_hit": false,
  "outfit": {
    "top": {
      "name": "Classic Linen Shirt",
      "brand": "Zara",
      "price": 49.99,
      "color": "white",
      "image_url": "https://...",
      "product_url": "https://..."
    },
    "shoes": {
      "name": "Canvas Espadrilles Navy",
      "brand": "H&M",
      "price": 45.99,
      "color": "navy"
    },
    "total_price": 95.98,
    "stylist_note": "The crisp white linen shirt against your navy chinos is the platonic ideal of yacht-club dressing. The espadrilles ground the look with authentic Mediterranean flair.",
    "style_tags": ["nautical", "summer", "smart_casual", "linen"]
  },
  "alternatives": [...],
  "token_usage": { "prompt_tokens": 812, "completion_tokens": 340, "total_tokens": 1152 },
  "latency_ms": 1840,
  "agent_trace": ["wardrobe_loaded", "intent_parsed", "cache_miss", "retrieved_18_items", "outfit_reasoned", "formatted"]
}
```

---

## Features

| Feature | Details |
|---------|---------|
| 🤖 Agentic pipeline | LangGraph stateful graph with 7 nodes |
| 🔀 Model-agnostic | Groq / OpenAI / Anthropic / Google — swap with 2 env vars |
| 🔍 RAG retrieval | Qdrant vector search with pre-filter (category + gender + budget) |
| 🧠 Fashion intelligence | Color theory, occasion dressing, formality matching |
| ⚡ Semantic cache | Redis-backed cosine similarity cache — near-zero latency on repeat queries |
| 👕 Wardrobe memory | Users can store owned items; agent references them automatically |
| 💰 Budget awareness | Optional budget param filters items before LLM sees them |
| 👔 Outfit alternatives | Returns primary + 2 alternate looks per request |
| 🕷️ Multi-source scraper | Zara + H&M + Myntra (concurrent, async, with retry + fallback) |
| 📊 Catalog API | Browse, filter, semantic search, stats endpoints |
| 🔧 On-demand scraping | POST endpoint to trigger fresh scrape jobs |
| 🧪 114 tests | Full unit + integration coverage, no Docker needed |

---

## Architecture

```
User Prompt
    │
    ▼
FastAPI (validation · request ID · timing middleware)
    │
    ▼
LangGraph Agent
    ├─ load_wardrobe   → fetch user's owned items from Postgres
    ├─ intent_parser   → LLM: prompt → ParsedIntent + embed query vector
    ├─ cache_check     → cosine similarity vs Redis (threshold: 0.92)
    │     ├─ HIT  → cache_formatter → END (0 LLM tokens)
    │     └─ MISS → rag_retriever
    │                   │
    │               fashion_reasoner → LLM: color theory + outfit selection
    │                   │
    │               response_formatter → schema + cache write
    ▼
StyleMeResponse { outfit, alternatives, stylist_note, token_usage, agent_trace }
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for full documentation and [docs/flowchart.md](./docs/flowchart.md) for Mermaid diagrams.

---

## Setup

### Prerequisites
- Python 3.12+
- Docker + Docker Compose
- API key for at least one LLM provider (Groq recommended — free tier available)

### 1. Clone and configure

```bash
git clone <repo-url>
cd stylist-concierge
cp .env.example .env
# Edit .env — set your GROQ_API_KEY (or other provider key)
```

### 2. Start infrastructure

```bash
docker compose up -d qdrant postgres redis
# Wait ~15s for services to be healthy
docker compose ps   # all should show "healthy"
```

### 3. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** — full Swagger UI ready.

### 5. Seed the catalog

Before making style requests, populate the catalog:

```bash
# Option A: Seed with realistic mock data (instant, no network needed)
curl -X POST "http://localhost:8000/api/v1/scraper/seed?count=120"

# Option B: Run real scrapers (takes 2–5 minutes, requires network)
curl -X POST http://localhost:8000/api/v1/scraper/trigger \
  -H "Content-Type: application/json" \
  -d '{"source": "all", "reindex": false}'

# Check scrape job status
curl http://localhost:8000/api/v1/scraper/jobs
```

### 6. Make your first style request

```bash
curl -X POST http://localhost:8000/api/v1/style-me \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I have dark navy chinos. What top and shoes for a summer yacht party?",
    "gender": "male",
    "include_alternatives": true
  }'
```

---

## Switching LLM Providers

Edit `.env` — **only two variables change**:

```env
# Groq (default — fastest, cheapest)
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

No code changes needed anywhere.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/style-me` | **Core**: NL prompt → outfit recommendation |
| `POST` | `/api/v1/style-me/feedback` | Rate a recommendation (1–5) |
| `GET`  | `/api/v1/catalog` | Browse catalog with filters + pagination |
| `GET`  | `/api/v1/catalog/search?q=...` | Semantic search over catalog |
| `GET`  | `/api/v1/catalog/stats` | Item counts, breakdowns, Qdrant info |
| `GET`  | `/api/v1/catalog/{id}` | Single item detail |
| `GET`  | `/api/v1/wardrobe` | List user's owned items |
| `POST` | `/api/v1/wardrobe` | Add item to wardrobe |
| `DELETE` | `/api/v1/wardrobe/{id}` | Remove wardrobe item |
| `POST` | `/api/v1/scraper/trigger` | Trigger real scrape job (background) |
| `POST` | `/api/v1/scraper/seed` | Seed mock catalog (dev/testing) |
| `GET`  | `/api/v1/scraper/jobs` | List scrape jobs |
| `GET`  | `/api/v1/scraper/jobs/{id}` | Job status + item counts |
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/ready` | Readiness (checks all services) |
| `GET`  | `/info` | App metadata + active model info |

Full interactive docs at **http://localhost:8000/docs**

---

## Running Tests

```bash
# All 114 tests — no Docker or API keys needed
pytest tests/ -v

# Individual modules
pytest tests/test_schemas.py -v
pytest tests/test_scraper.py -v
pytest tests/test_agent.py -v
pytest tests/test_api.py -v
pytest tests/test_embeddings.py -v
```

---

## Docker (Production)

```bash
# Build and run everything
docker compose up --build

# App only (infrastructure already running)
docker compose up app
```

---

## Project Structure

```
stylist-concierge/
├── config/settings.py          # All config, LLM factory, model-agnostic wiring
├── scraper/
│   ├── base_scraper.py         # Retry, rate-limit, normalisation utilities
│   ├── zara_scraper.py         # Zara internal API + HTML fallback
│   ├── hm_scraper.py           # H&M internal API + HTML fallback
│   ├── myntra_scraper.py       # Myntra search API (bonus 3rd source)
│   └── pipeline.py             # Orchestrator: scrape → deduplicate → embed → index
├── embeddings/
│   ├── embedder.py             # Model-agnostic batch embedder
│   └── indexer.py              # Qdrant collection manager + multi-category search
├── agents/
│   ├── state.py                # StylistState TypedDict
│   ├── graph.py                # LangGraph StateGraph + run_stylist_agent()
│   └── nodes/                  # 5 agent nodes
│       ├── intent_parser.py    # LLM: prompt → ParsedIntent + embed
│       ├── cache_manager.py    # Semantic cache lookup
│       ├── rag_retriever.py    # Qdrant multi-category search
│       ├── fashion_reasoner.py # LLM: fashion rules + outfit selection
│       └── response_formatter.py # Schema assembly + cache write
├── api/
│   ├── main.py                 # FastAPI app + lifespan + middleware
│   ├── routers/                # style · catalog · wardrobe · scraper · health
│   └── schemas/requests.py    # All Pydantic v2 models
├── cache/semantic_cache.py     # Redis-backed cosine similarity cache
├── db/
│   ├── models.py               # 6 ORM models
│   └── postgres.py             # Async SQLAlchemy engine + session
├── tests/                      # 114 tests (unit + integration)
├── docs/flowchart.md           # Mermaid system diagrams
├── ARCHITECTURE.md             # Full design documentation
├── docker-compose.yml          # Qdrant + Postgres + Redis + App
└── Dockerfile                  # Multi-stage build
```

---

## Token Economics (Frugal Mindset)

| Strategy | Saving |
|----------|--------|
| **Semantic cache** | 100% tokens saved on similar queries (threshold: 0.92 cosine) |
| **Pre-filter before LLM** | Qdrant filters by category/gender/price → LLM sees ≤18 items |
| **Structured intent parsing** | Small JSON output prompt keeps intent node cost low |
| **Batch embeddings** | 50 items per API call during indexing, not 1-per-item |
| **Groq default** | Llama-4-Maverick at ~$0.05/M tokens vs $15/M for GPT-4 |

Every response includes `token_usage` + `estimated_cost_usd` for full transparency.