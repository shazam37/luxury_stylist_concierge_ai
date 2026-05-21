"""
Myntra Scraper (bonus 3rd source)
Myntra is India's largest fashion e-commerce platform.
Uses their public search/browse API.
Produces: normalised FashionItem dicts.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx
import structlog

from scraper.base_scraper import (
    BaseScraper, clean_price, clean_text, infer_category, make_empty_item,
)

logger = structlog.get_logger(__name__)


class MyntraScraper(BaseScraper):
    """
    Scrapes Myntra using their product search API.
    Currency is INR — we keep it as-is; agent handles multi-currency.
    """

    SOURCE_NAME = "myntra"
    BASE_URL = "https://www.myntra.com"
    BRAND = None  # Myntra lists multiple brands

    # Myntra category IDs for their browse API
    # (category_id, gender, label, category_hint, target)
    MYNTRA_CATEGORIES = [
        ("2", "male", "men_tshirts", "top", 20),
        ("1329", "male", "men_shirts", "top", 20),
        ("17", "male", "men_jeans", "bottom", 20),
        ("6204", "male", "men_trousers", "bottom", 15),
        ("4082", "male", "men_sneakers", "shoes", 15),
        ("1960", "female", "women_tops", "top", 20),
        ("1889", "female", "women_dresses", "top", 15),
        ("3378", "female", "women_jeans", "bottom", 15),
        ("4084", "female", "women_shoes", "shoes", 15),
    ]

    async def _scrape_source(self, target_count: int = 60) -> list[dict]:
        """Scrape Myntra via their product browse API."""
        all_items: list[dict] = []

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=25,
            headers={
                "User-Agent": self._random_ua(),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.myntra.com/",
                "Origin": "https://www.myntra.com",
                "x-meta-app": '{"appFamily":"Web","appVersion":"1.0.0"}',
            }
        ) as session:
            tasks = [
                self._scrape_category(session, cat_id, gender, label, cat_hint, target)
                for cat_id, gender, label, cat_hint, target in self.MYNTRA_CATEGORIES
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                label = self.MYNTRA_CATEGORIES[i][2]
                self._log.warning("myntra.category_error", label=label, error=str(result)[:100])
            elif isinstance(result, list):
                all_items.extend(result)

        self._log.info("myntra.total_scraped", count=len(all_items))
        return all_items

    async def _scrape_category(
        self,
        session: httpx.AsyncClient,
        category_id: str,
        gender: str,
        label: str,
        category_hint: str,
        target: int,
    ) -> list[dict]:
        """Scrape one Myntra category page."""
        items = []
        page = 1
        page_size = 24

        while len(items) < target:
            # Myntra's product listing API
            url = (
                f"https://www.myntra.com/gateway/v2/product/list/filter?"
                f"category={category_id}"
                f"&p={page}"
                f"&rows={page_size}"
                f"&o={0 if page == 1 else (page - 1) * page_size}"
                f"&plaEnabled=false"
            )

            data = await self._fetch_json(url, session=session)

            if not data or not isinstance(data, dict):
                # Try alternate Myntra API
                url_alt = (
                    f"https://www.myntra.com/gateway/v2/product/list/filter"
                    f"?rawQuery=category:{category_id}"
                    f"&p={page}&rows={page_size}"
                )
                data = await self._fetch_json(url_alt, session=session)

            if not data:
                break

            products = self._extract_products(data)
            if not products:
                break

            for product in products:
                item = self._parse_product(product, gender, category_hint)
                if item:
                    items.append(item)

            total = data.get("data", {}).get("totalCount", 0) if isinstance(data.get("data"), dict) else 0
            page += 1

            if len(items) >= target or page * page_size >= total:
                break

            await self._human_delay()

        self._log.info("myntra.category_done", label=label, count=len(items))
        return items[:target]

    def _extract_products(self, data: dict) -> list[dict]:
        """Extract product list from Myntra API response."""
        if isinstance(data, dict):
            # Try nested structure: data.products
            nested = data.get("data") or {}
            if isinstance(nested, dict):
                products = nested.get("products") or nested.get("product") or []
                if products:
                    return products

            # Direct product list
            return data.get("products") or data.get("items") or []

        return []

    def _parse_product(self, product: dict, gender: str, category_hint: str) -> Optional[dict]:
        """Parse a single Myntra product dict."""
        try:
            item = make_empty_item()
            item["gender"] = gender
            item["currency"] = "INR"

            # Name
            name = product.get("productName") or product.get("name") or ""
            brand = product.get("brand", {})
            if isinstance(brand, dict):
                brand_name = brand.get("name") or ""
            else:
                brand_name = str(brand)

            if brand_name:
                item["brand"] = clean_text(brand_name, 100)
                full_name = f"{brand_name} {name}".strip()
            else:
                full_name = name

            if not full_name:
                return None

            item["name"] = clean_text(full_name, 500)
            item["external_id"] = f"myntra_{product.get('id') or product.get('productId', '')}"

            # Price (Myntra prices are in INR paise or INR directly)
            price_raw = product.get("price") or product.get("discountedPrice") or {}
            if isinstance(price_raw, dict):
                price_val = price_raw.get("discounted") or price_raw.get("mrp") or 0
            else:
                price_val = price_raw

            if isinstance(price_val, str):
                price_val = clean_price(price_val) or 0

            # Myntra sometimes returns paise (x100); detect by magnitude
            if isinstance(price_val, (int, float)) and price_val > 0:
                item["price"] = float(price_val)

            # Description
            desc = product.get("description") or product.get("productDescriptor") or ""
            if not desc:
                # Build from category + attributes
                attributes = product.get("primaryColour") or ""
                desc = f"{item['name']}. {attributes}".strip(". ")
            item["description"] = clean_text(str(desc), 800)

            # Image URL
            images = product.get("images") or []
            if images and isinstance(images[0], dict):
                item["image_url"] = images[0].get("src") or images[0].get("url") or ""
            elif images:
                item["image_url"] = str(images[0])

            if not item["image_url"]:
                # Myntra uses a CDN pattern: https://assets.myntassets.com/h_X,q_90,w_X/v1/{id}/...
                pid = product.get("id") or ""
                if pid:
                    item["image_url"] = f"https://assets.myntassets.com/h_720,q_90,w_540/v1/{pid}/images/1.jpg"

            # Product URL
            slug = product.get("landingPageUrl") or ""
            if slug:
                item["product_url"] = f"https://www.myntra.com/{slug.lstrip('/')}"
            else:
                pid = product.get("id") or ""
                item["product_url"] = f"https://www.myntra.com/products/{pid}" if pid else ""

            # Color
            color = product.get("primaryColour") or product.get("color") or ""
            if isinstance(color, dict):
                color = color.get("name") or ""
            item["color"] = clean_text(str(color), 50).lower() if color else None

            # Sizes
            sizes = product.get("sizes") or product.get("sizeChart") or []
            if isinstance(sizes, list):
                size_names = []
                for s in sizes:
                    if isinstance(s, dict):
                        size_names.append(s.get("label") or s.get("size") or str(s))
                    else:
                        size_names.append(str(s))
                item["sizes_available"] = size_names[:20]

            # Category
            item["category"] = category_hint
            item["sub_category"] = (
                product.get("category", {}).get("primaryCategory")
                or product.get("subCategory")
                or ""
            )

            # Availability
            item["is_available"] = product.get("inventoryInfo") != "OOS"

            # Raw data
            item["raw_data"] = {
                "source_id": product.get("id"),
                "rating": product.get("rating"),
                "ratingCount": product.get("ratingCount"),
            }

            return item

        except Exception as e:
            self._log.warning("myntra.parse_error", error=str(e)[:100])
            return None