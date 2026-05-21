# ─────────────────────────────────────────────────────────────
#  Stylist Concierge — Multi-stage Dockerfile
#  Stages:
#    base      → shared Python + system deps
#    scraper   → base + Playwright browsers
#    production → base + app only (no browsers, smaller image)
# ─────────────────────────────────────────────────────────────

# ── Stage 1: Base ──────────────────────────────────────────────
FROM python:3.12-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Scraper (base + Playwright browsers) ──────────────
FROM base AS scraper

RUN pip install --no-cache-dir playwright \
    && playwright install chromium \
    && playwright install-deps chromium

COPY . .

# ── Stage 3: Production App ────────────────────────────────────
FROM base AS production

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]