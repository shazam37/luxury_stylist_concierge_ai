#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  dev.sh — Start the Stylist Concierge in development mode
#  Usage: ./scripts/dev.sh
# ─────────────────────────────────────────────────────────────
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── 1. Check .env exists ──────────────────────────────────────
if [ ! -f ".env" ]; then
  echo "⚠️  No .env file found. Copying from .env.example..."
  cp .env.example .env
  echo "✅  Created .env — please fill in your API keys before continuing."
  echo "    Required: set at least one of GROQ_API_KEY, OPENAI_API_KEY, etc."
  exit 1
fi

# ── 2. Start infrastructure ───────────────────────────────────
echo "🐳  Starting infrastructure (Qdrant + PostgreSQL + Redis)..."
docker compose up -d qdrant postgres redis

# ── 3. Wait for services ──────────────────────────────────────
echo "⏳  Waiting for services to be healthy..."
sleep 5

MAX_TRIES=20
for i in $(seq 1 $MAX_TRIES); do
  if docker compose ps | grep -q "healthy"; then
    echo "✅  Infrastructure ready"
    break
  fi
  if [ $i -eq $MAX_TRIES ]; then
    echo "❌  Services did not become healthy in time"
    docker compose logs
    exit 1
  fi
  sleep 3
done

# ── 4. Install Python dependencies ───────────────────────────
if [ ! -d ".venv" ]; then
  echo "🐍  Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate
echo "📦  Installing dependencies..."
pip install -q -r requirements.txt

# Install Playwright browsers (for scraper)
playwright install chromium 2>/dev/null || echo "ℹ️  Playwright browsers install skipped (run manually if needed)"

# ── 5. Start FastAPI ──────────────────────────────────────────
echo ""
echo "🚀  Starting Stylist Concierge API..."
echo "    Docs:   http://localhost:8000/docs"
echo "    Qdrant: http://localhost:6333/dashboard"
echo ""

uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level debug