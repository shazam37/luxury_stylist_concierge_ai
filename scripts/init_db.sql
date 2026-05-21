-- ─────────────────────────────────────────────────────────────
--  Stylist Concierge — DB Initialisation Script
--  Runs automatically when Postgres container starts for the first time.
-- ─────────────────────────────────────────────────────────────

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- for fuzzy text search

-- ── Users ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email       VARCHAR(255) UNIQUE NOT NULL,
    name        VARCHAR(255),
    style_profile JSONB DEFAULT '{}',      -- e.g. {"preferred_style": "minimalist", "gender": "male"}
    budget_range JSONB DEFAULT '{}',       -- e.g. {"min": 0, "max": 500, "currency": "USD"}
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── Wardrobe (items users already own) ───────────────────────
CREATE TABLE IF NOT EXISTS wardrobe_items (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        VARCHAR(500) NOT NULL,
    category    VARCHAR(100) NOT NULL,      -- top | bottom | shoes | accessory | outerwear
    color       VARCHAR(100),
    brand       VARCHAR(255),
    description TEXT,
    image_url   TEXT,
    tags        TEXT[] DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_wardrobe_user ON wardrobe_items(user_id);
CREATE INDEX IF NOT EXISTS idx_wardrobe_category ON wardrobe_items(category);

-- ── Catalog Items (scraped inventory) ────────────────────────
CREATE TABLE IF NOT EXISTS catalog_items (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id     VARCHAR(500),           -- source website's product ID
    source          VARCHAR(100) NOT NULL,  -- zara | hm | myntra
    name            VARCHAR(500) NOT NULL,
    category        VARCHAR(100) NOT NULL,  -- top | bottom | shoes | accessory | outerwear
    sub_category    VARCHAR(100),           -- t-shirt | shirt | jeans | chinos | etc.
    price           DECIMAL(10, 2),
    currency        VARCHAR(10) DEFAULT 'USD',
    color           VARCHAR(100),
    colors_available TEXT[] DEFAULT '{}',
    brand           VARCHAR(255),
    description     TEXT,
    image_url       TEXT,
    product_url     TEXT,
    sizes_available TEXT[] DEFAULT '{}',
    gender          VARCHAR(50),            -- male | female | unisex
    is_available    BOOLEAN DEFAULT TRUE,
    tags            TEXT[] DEFAULT '{}',
    raw_data        JSONB DEFAULT '{}',     -- full raw scraped payload
    qdrant_id       VARCHAR(255),           -- ID in Qdrant for cross-ref
    scraped_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_catalog_source ON catalog_items(source);
CREATE INDEX IF NOT EXISTS idx_catalog_category ON catalog_items(category);
CREATE INDEX IF NOT EXISTS idx_catalog_price ON catalog_items(price);
CREATE INDEX IF NOT EXISTS idx_catalog_available ON catalog_items(is_available);
CREATE INDEX IF NOT EXISTS idx_catalog_gender ON catalog_items(gender);
CREATE INDEX IF NOT EXISTS idx_catalog_name_trgm ON catalog_items USING gin(name gin_trgm_ops);

-- ── Scrape Jobs ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scrape_jobs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source          VARCHAR(100),           -- zara | hm | myntra | all
    status          VARCHAR(50) DEFAULT 'pending',  -- pending | running | completed | failed
    triggered_by    VARCHAR(100) DEFAULT 'system',  -- system | user | api
    items_scraped   INTEGER DEFAULT 0,
    items_indexed   INTEGER DEFAULT 0,
    errors          JSONB DEFAULT '[]',
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scrape_jobs_status ON scrape_jobs(status);
CREATE INDEX IF NOT EXISTS idx_scrape_jobs_source ON scrape_jobs(source);

-- ── Style Requests (agent conversation history) ───────────────
CREATE TABLE IF NOT EXISTS style_requests (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id      VARCHAR(255),
    prompt          TEXT NOT NULL,
    parsed_intent   JSONB DEFAULT '{}',
    outfit_response JSONB DEFAULT '{}',
    cache_hit       BOOLEAN DEFAULT FALSE,
    token_usage     JSONB DEFAULT '{}',
    agent_trace     TEXT[] DEFAULT '{}',
    latency_ms      INTEGER,
    feedback_score  SMALLINT,              -- 1-5 user rating
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_style_requests_user ON style_requests(user_id);
CREATE INDEX IF NOT EXISTS idx_style_requests_session ON style_requests(session_id);
CREATE INDEX IF NOT EXISTS idx_style_requests_created ON style_requests(created_at DESC);

-- ── Outfit Saves (favourites) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS saved_outfits (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    style_request_id UUID REFERENCES style_requests(id) ON DELETE SET NULL,
    outfit_data     JSONB NOT NULL,
    name            VARCHAR(255),          -- user-given name, e.g. "Yacht party look"
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_saved_outfits_user ON saved_outfits(user_id);

-- ── updated_at trigger helper ─────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER catalog_updated_at
    BEFORE UPDATE ON catalog_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ── Seed a default anonymous user for unauthenticated requests ─
INSERT INTO users (id, email, name)
VALUES ('00000000-0000-0000-0000-000000000000', 'anonymous@stylist.local', 'Anonymous')
ON CONFLICT (id) DO NOTHING;