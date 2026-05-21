"""
Base scraper class with:
  - Playwright browser management (headless Chromium)
  - Exponential backoff retry logic via tenacity
  - Human-like random delays between requests
  - Rotating user-agents
  - Graceful error handling with per-item fault isolation
  - Structured output: normalised FashionItem dicts
"""

from __future__ import annotations

import asyncio
import random
import re
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
import structlog
from bs4 import BeautifulSoup
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import get_settings

logger = structlog.get_logger(__name__)

# ─────────────────────────────────────────────
#  Human-like User-Agent pool
# ─────────────────────────────────────────────

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
]

# ─────────────────────────────────────────────
#  Normalised item schema (output of every scraper)
# ─────────────────────────────────────────────

def make_empty_item() -> dict:
    return {
        "external_id": None,
        "source": None,
        "name": None,
        "category": None,        # top | bottom | shoes | accessory | outerwear
        "sub_category": None,    # t-shirt | shirt | jeans | chinos | sneakers
        "price": None,
        "currency": "USD",
        "color": None,
        "colors_available": [],
        "brand": None,
        "description": None,
        "image_url": None,
        "product_url": None,
        "sizes_available": [],
        "gender": "unisex",
        "is_available": True,
        "tags": [],
        "raw_data": {},
    }


def clean_price(text: str) -> Optional[float]:
    """Extract float from price strings like '€29.95', 'Rs. 2,499', '$49'."""
    if not text:
        return None
    # Remove currency symbols and thousands separators
    cleaned = re.sub(r"[^\d.,]", "", str(text).strip())
    # Handle European format: 1.299,99 → 1299.99
    if cleaned.count(",") == 1 and cleaned.count(".") >= 1:
        if cleaned.index(",") > cleaned.index("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
    cleaned = cleaned.replace(",", "")
    try:
        return round(float(cleaned), 2)
    except (ValueError, TypeError):
        return None


def clean_text(text: str, max_length: int = 500) -> str:
    """Clean and truncate text, removing excessive whitespace."""
    if not text:
        return ""
    return " ".join(str(text).split())[:max_length]


def infer_category(name: str, sub_category: str = "") -> str:
    """Infer top-level category from item name/sub_category."""
    text = f"{name} {sub_category}".lower()

    if any(w in text for w in ["shoe", "sneaker", "boot", "loafer", "heel", "sandal", "trainer", "footwear"]):
        return "shoes"
    if any(w in text for w in ["trouser", "pant", "jean", "chino", "short", "skirt", "bottom", "legging"]):
        return "bottom"
    if any(w in text for w in ["t-shirt", "tshirt", "shirt", "top", "blouse", "sweater", "hoodie",
                                "jacket", "coat", "pullover", "sweatshirt", "polo", "tee", "knitwear"]):
        return "top"
    if any(w in text for w in ["bag", "belt", "watch", "hat", "cap", "scarf", "sunglasses", "wallet", "accessory"]):
        return "accessory"
    if any(w in text for w in ["blazer", "suit", "overcoat", "parka", "windbreaker"]):
        return "outerwear"

    return "top"  # safe default


def infer_gender(text: str) -> str:
    text_l = text.lower()
    if any(w in text_l for w in ["woman", "women", "female", "ladies", "girl", "femme"]):
        return "female"
    if any(w in text_l for w in ["man", "men", "male", "boys", "homme", "masculin"]):
        return "male"
    return "unisex"


# ─────────────────────────────────────────────
#  Base Scraper
# ─────────────────────────────────────────────

class BaseScraper(ABC):
    """
    Abstract base for all fashion scrapers.
    Subclasses implement _scrape_source() and return normalised item dicts.
    """

    SOURCE_NAME: str = "unknown"
    BASE_URL: str = ""

    def __init__(self):
        self._settings = get_settings()
        self._items: list[dict] = []
        self._errors: list[str] = []
        self._log = logger.bind(scraper=self.SOURCE_NAME)

    # ─────────────────────────────────────────
    #  Public entry point
    # ─────────────────────────────────────────

    async def scrape(self, target_count: int = 60) -> list[dict]:
        """
        Run the full scrape. Returns a list of normalised item dicts.
        `target_count` is the minimum number of items to attempt to collect.
        """
        self._log.info("scraper.starting", target=target_count)
        start = time.perf_counter()

        try:
            items = await self._scrape_source(target_count=target_count)
            items = [self._validate_and_enrich(item) for item in items if item]
            items = [i for i in items if i]  # drop Nones
        except Exception as e:
            self._log.exception("scraper.fatal_error", error=str(e))
            items = []

        elapsed = round(time.perf_counter() - start, 2)
        self._log.info(
            "scraper.finished",
            items=len(items),
            errors=len(self._errors),
            elapsed_s=elapsed,
        )
        return items

    # ─────────────────────────────────────────
    #  Abstract — subclasses implement this
    # ─────────────────────────────────────────

    @abstractmethod
    async def _scrape_source(self, target_count: int) -> list[dict]:
        """Scrape the source and return raw item dicts (before validation)."""
        ...

    # ─────────────────────────────────────────
    #  HTTP helpers
    # ─────────────────────────────────────────

    def _random_ua(self) -> str:
        return random.choice(USER_AGENTS)

    async def _human_delay(self):
        """Random sleep between requests to mimic human browsing."""
        delay = random.uniform(
            self._settings.scraper_delay_min,
            self._settings.scraper_delay_max,
        )
        await asyncio.sleep(delay)

    async def _fetch_html(self, url: str, session: Optional[httpx.AsyncClient] = None) -> Optional[str]:
        """
        Fetch a URL with retry logic.
        Returns the HTML string or None on failure.
        """
        headers = {
            "User-Agent": self._random_ua(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "no-cache",
        }

        async def _do_fetch():
            if session:
                resp = await session.get(url, headers=headers, follow_redirects=True, timeout=20)
                resp.raise_for_status()
                return resp.text
            else:
                async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    return resp.text

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._settings.scraper_max_retries),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
                reraise=True,
            ):
                with attempt:
                    await self._human_delay()
                    html = await _do_fetch()
                    return html
        except Exception as e:
            self._log.warning("fetch.failed", url=url[:80], error=str(e)[:100])
            self._errors.append(f"Fetch failed: {url[:80]} — {str(e)[:80]}")
            return None

    async def _fetch_json(self, url: str, session: Optional[httpx.AsyncClient] = None) -> Optional[dict | list]:
        """Fetch a JSON API endpoint with retry logic."""
        headers = {
            "User-Agent": self._random_ua(),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
        }

        async def _do_fetch():
            if session:
                resp = await session.get(url, headers=headers, follow_redirects=True, timeout=15)
                resp.raise_for_status()
                return resp.json()
            else:
                async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    return resp.json()

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._settings.scraper_max_retries),
                wait=wait_exponential(multiplier=1, min=2, max=8),
                retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
                reraise=True,
            ):
                with attempt:
                    await self._human_delay()
                    return await _do_fetch()
        except Exception as e:
            self._log.warning("fetch_json.failed", url=url[:80], error=str(e)[:100])
            self._errors.append(f"JSON fetch failed: {url[:80]} — {str(e)[:80]}")
            return None

    def _parse_html(self, html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "lxml")

    def _abs_url(self, path: str) -> str:
        """Convert relative URL to absolute."""
        if not path:
            return ""
        if path.startswith("//"):
            return f"https:{path}"
        if path.startswith("http"):
            return path
        return urljoin(self.BASE_URL, path)

    # ─────────────────────────────────────────
    #  Validation & enrichment
    # ─────────────────────────────────────────

    def _validate_and_enrich(self, item: dict) -> Optional[dict]:
        """
        Validate required fields and fill in defaults.
        Returns None if item is unusable.
        """
        if not item:
            return None

        # Must have a name
        if not item.get("name") or len(str(item["name"]).strip()) < 3:
            return None

        # Assign source
        item["source"] = self.SOURCE_NAME

        # Generate stable external ID if missing
        if not item.get("external_id"):
            name_slug = re.sub(r"[^a-z0-9]", "_", (item.get("name") or "").lower())[:40]
            item["external_id"] = f"{self.SOURCE_NAME}_{name_slug}_{uuid.uuid4().hex[:8]}"

        # Clean text fields
        item["name"] = clean_text(item.get("name", ""), 500)
        item["description"] = clean_text(item.get("description", ""), 1000)

        # Ensure category
        if not item.get("category"):
            item["category"] = infer_category(item["name"], item.get("sub_category", ""))

        # Ensure gender
        if not item.get("gender"):
            full_text = f"{item.get('name', '')} {item.get('description', '')} {item.get('product_url', '')}"
            item["gender"] = infer_gender(full_text)

        # Normalise price
        if item.get("price") and not isinstance(item["price"], float):
            item["price"] = clean_price(str(item["price"]))

        # Ensure lists
        item.setdefault("tags", [])
        item.setdefault("colors_available", [])
        item.setdefault("sizes_available", [])

        # Auto-tag based on category/name
        item["tags"] = list(set(item["tags"] + self._auto_tags(item)))

        return item

    def _auto_tags(self, item: dict) -> list[str]:
        """Generate tags from item attributes for better semantic search."""
        tags = []
        name_lower = (item.get("name") or "").lower()
        desc_lower = (item.get("description") or "").lower()
        combined = f"{name_lower} {desc_lower}"

        # Style tags
        style_keywords = {
            "casual": ["casual", "everyday", "relaxed", "comfortable"],
            "formal": ["formal", "elegant", "dress", "office", "business"],
            "summer": ["summer", "lightweight", "linen", "cotton", "breathable", "beach"],
            "winter": ["winter", "wool", "warm", "thick", "fleece", "thermal"],
            "sport": ["sport", "athletic", "gym", "running", "active", "performance"],
            "streetwear": ["streetwear", "urban", "graphic", "oversized", "baggy"],
            "minimalist": ["minimalist", "minimal", "clean", "simple", "classic"],
            "luxury": ["luxury", "premium", "high-end", "designer", "fine", "crafted"],
        }
        for tag, keywords in style_keywords.items():
            if any(kw in combined for kw in keywords):
                tags.append(tag)

        # Material tags
        materials = ["cotton", "linen", "wool", "silk", "denim", "leather", "synthetic", "polyester", "cashmere"]
        tags.extend(m for m in materials if m in combined)

        # Color from name
        colors = ["white", "black", "blue", "navy", "red", "green", "grey", "gray", "brown",
                  "beige", "cream", "pink", "purple", "yellow", "orange", "khaki", "olive"]
        if not item.get("color"):
            for c in colors:
                if c in name_lower:
                    item["color"] = c
                    break
        if item.get("color"):
            tags.append(item["color"])

        return list(set(tags))