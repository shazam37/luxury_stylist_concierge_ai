"""
Semantic Cache for style requests.

How it works:
  1. Embed the incoming user prompt.
  2. Compare against stored prompt vectors in Redis (cosine similarity).
  3. If similarity >= threshold → return cached response (cache HIT).
  4. Otherwise → run the agent, store result + vector (cache MISS).

This implements the "frugal mindset" requirement by avoiding redundant
LLM calls for semantically equivalent prompts.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np
import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)


class SemanticCache:
    """
    Redis-backed semantic cache using vector similarity.
    Falls back to in-memory dict if Redis is unavailable.
    """

    CACHE_KEY_PREFIX = "stylist:cache:"
    INDEX_KEY = "stylist:cache:index"  # sorted set of all cache entry IDs

    def __init__(self):
        self._settings = get_settings()
        self._threshold = self._settings.cache_similarity_threshold
        self._ttl = self._settings.cache_ttl_seconds
        self._redis: Optional[object] = None
        self._memory_store: dict[str, dict] = {}  # fallback

    async def _get_redis(self):
        """Lazy Redis connection with fallback."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = await aioredis.from_url(
                    self._settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await self._redis.ping()
                logger.info("cache.redis_connected")
            except Exception as e:
                logger.warning("cache.redis_unavailable", error=str(e), fallback="in-memory")
                self._redis = None
        return self._redis

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        a, b = np.array(v1), np.array(v2)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    # ─────────────────────────────────────────
    #  Public Interface
    # ─────────────────────────────────────────

    async def get(self, prompt_vector: list[float]) -> Optional[dict]:
        """
        Look up the cache for a semantically similar prompt.
        Returns cached response dict or None.
        """
        redis = await self._get_redis()

        if redis:
            return await self._redis_get(prompt_vector)
        else:
            return self._memory_get(prompt_vector)

    async def set(self, prompt: str, prompt_vector: list[float], response: dict) -> str:
        """
        Store a prompt + response in the cache.
        Returns the cache entry ID.
        """
        import uuid
        entry_id = str(uuid.uuid4())
        entry = {
            "id": entry_id,
            "prompt": prompt,
            "vector": prompt_vector,
            "response": response,
            "created_at": time.time(),
        }

        redis = await self._get_redis()
        if redis:
            await self._redis_set(entry_id, entry)
        else:
            self._memory_set(entry_id, entry)

        logger.info("cache.stored", entry_id=entry_id, prompt_preview=prompt[:60])
        return entry_id

    async def invalidate_all(self) -> int:
        """Clear all cache entries. Returns count cleared."""
        redis = await self._get_redis()
        if redis:
            keys = await redis.keys(f"{self.CACHE_KEY_PREFIX}*")
            if keys:
                await redis.delete(*keys)
            return len(keys)
        else:
            count = len(self._memory_store)
            self._memory_store.clear()
            return count

    # ─────────────────────────────────────────
    #  Redis Backend
    # ─────────────────────────────────────────

    async def _redis_set(self, entry_id: str, entry: dict) -> None:
        key = f"{self.CACHE_KEY_PREFIX}{entry_id}"
        serialised = json.dumps({
            "id": entry["id"],
            "prompt": entry["prompt"],
            "vector": entry["vector"],
            "response": entry["response"],
            "created_at": entry["created_at"],
        })
        await self._redis.setex(key, self._ttl, serialised)
        # Track all IDs in a sorted set (score = creation time for TTL ordering)
        await self._redis.zadd(self.INDEX_KEY, {entry_id: entry["created_at"]})
        await self._redis.expire(self.INDEX_KEY, self._ttl * 2)

    async def _redis_get(self, prompt_vector: list[float]) -> Optional[dict]:
        # Fetch all cached entry IDs
        entry_ids = await self._redis.zrange(self.INDEX_KEY, 0, -1)
        if not entry_ids:
            return None

        best_score = 0.0
        best_response = None

        for entry_id in entry_ids:
            key = f"{self.CACHE_KEY_PREFIX}{entry_id}"
            raw = await self._redis.get(key)
            if not raw:
                continue
            entry = json.loads(raw)
            sim = self._cosine_similarity(prompt_vector, entry["vector"])
            if sim > best_score:
                best_score = sim
                best_response = entry["response"]

        if best_score >= self._threshold:
            logger.info("cache.hit", similarity=round(best_score, 4))
            return best_response

        logger.info("cache.miss", best_similarity=round(best_score, 4), threshold=self._threshold)
        return None

    # ─────────────────────────────────────────
    #  In-memory Fallback Backend
    # ─────────────────────────────────────────

    def _memory_set(self, entry_id: str, entry: dict) -> None:
        self._memory_store[entry_id] = entry
        # Prune old entries (keep max 200)
        if len(self._memory_store) > 200:
            oldest = sorted(self._memory_store.items(), key=lambda x: x[1]["created_at"])
            for old_id, _ in oldest[:50]:
                del self._memory_store[old_id]

    def _memory_get(self, prompt_vector: list[float]) -> Optional[dict]:
        now = time.time()
        best_score = 0.0
        best_response = None

        for entry_id, entry in list(self._memory_store.items()):
            # Expire old entries
            if now - entry["created_at"] > self._ttl:
                del self._memory_store[entry_id]
                continue
            sim = self._cosine_similarity(prompt_vector, entry["vector"])
            if sim > best_score:
                best_score = sim
                best_response = entry["response"]

        if best_score >= self._threshold:
            logger.info("cache.hit_memory", similarity=round(best_score, 4))
            return best_response

        return None


# ─────────────────────────────────────────────
#  Singleton
# ─────────────────────────────────────────────

_cache: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache