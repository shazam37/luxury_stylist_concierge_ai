# syntax=docker/dockerfile:1.7

# ─────────────────────────────────────────────────────────────
#  Stylist Concierge — Optimized Multi-stage Dockerfile
# ─────────────────────────────────────────────────────────────

# ── Stage 1: Base ────────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

# Python optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HuggingFace / Torch cache locations
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV TORCH_HOME=/root/.cache/torch

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for layer caching
COPY requirements.txt .

# ─────────────────────────────────────────────────────────────
# Install Python dependencies with pip cache mount
# ─────────────────────────────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# ─────────────────────────────────────────────────────────────
# Pre-download heavy ML models/libraries
# This layer gets cached separately
# ─────────────────────────────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/huggingface \
    --mount=type=cache,target=/root/.cache/torch \
    python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

# OPTIONAL:
# Add any additional models here
#
# RUN --mount=type=cache,target=/root/.cache/huggingface \
#     --mount=type=cache,target=/root/.cache/torch \
#     python -c '\
# from transformers import pipeline; \
# pipeline(\"text-generation\", model=\"gpt2\")'

# ── Stage 2: Scraper (Playwright) ────────────────────────────
FROM base AS scraper

# Install Playwright separately so browser layer is cached
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install playwright

# Cache Playwright browser binaries
RUN --mount=type=cache,target=/root/.cache/ms-playwright \
    playwright install chromium && \
    playwright install-deps chromium

COPY . .

# ── Stage 3: Production ──────────────────────────────────────
FROM base AS production

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]