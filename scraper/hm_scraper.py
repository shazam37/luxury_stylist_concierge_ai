"""
H&M Scraper
Uses H&M's internal product API (reverse-engineered from their web app).
Targets: Men's and Women's tops, bottoms, and shoes.
Produces: normalised FashionItem dicts.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx
import structlog

from scraper.base_scraper import BaseScraper, clean_price, clean_text, infer_category, make_empty_item

logger = structlog.get_logger(__name__)


class HMScraper(BaseScraper):
    """
    Scrapes H&M using their public category/product APIs.
    No Playwright needed — H&M's API is accessible via standard HTTP.
    """

    SOURCE_NAME = "hm"
    BASE_URL = "https://www2.hm.com"
    BRAND = "H&M"

    # H&M product listing API — returns JSON with product cards
    # pagesize=48 is the max allowed per page
    CATEGORY_URLS = {
        "men_tops": "https://www2.hm.com/en_us/men/products/tops.html",
        "men_bottoms": "https://www2.hm.com/en_us/men/products/bottoms.html",
        "men_shoes": "https://www2.hm.com/en_us/men/products/shoes.html",
        "women_tops": "https://www2.hm.com/en_us/ladies/tops.html",
        "women_bottoms": "https://www2.hm.com/en_us/ladies/bottoms.html",
        "women_shoes": "https://www2.hm.com/en_us/ladies/shoes.html",
    }

    # H&M's internal product API — much more reliable than HTML scraping
    HM_API_BASE = "https://www2.hm.com/en_us/search-results/api/products"

    # Category-to-API parameter mapping
    HM_CATEGORIES = [
        # (category_filter, gender, internal_label, count_target)
        ("Tops", "male", "men_tops", 25),
        ("Trousers%20%26%20Jeans", "male", "men_bottoms", 25),
        ("Shoes%20%26%20Boots", "male", "men_shoes", 15),
        ("Tops", "female", "women_tops", 25),
        ("Trousers%20%26%20Jeans", "female", "women_bottoms", 25),
        ("Shoes%20%26%20Boots", "female", "women_shoes", 15),
    ]

    async def _scrape_source(self, target_count: int = 60) -> list[dict]:
        """
        Scrape H&M via their product search API.
        Runs category scrapes concurrently.
        """
        all_items: list[dict] = []

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=25,
            headers={
                "User-Agent": self._random_ua(),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www2.hm.com/en_us/",
                "Origin": "https://www2.hm.com",
            }
        ) as session:
            tasks = [
                self._scrape_category(session, cat_filter, gender, label, count)
                for cat_filter, gender, label, count in self.HM_CATEGORIES
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                self._log.warning("hm.category_error", error=str(result))
            elif isinstance(result, list):
                all_items.extend(result)

        self._log.info("hm.total_scraped", count=len(all_items))
        return all_items

    async def _scrape_category(
        self,
        session: httpx.AsyncClient,
        category_filter: str,
        gender: str,
        label: str,
        target: int,
    ) -> list[dict]:
        """Scrape one H&M category page using their search API."""
        items = []
        page_size = 24
        offset = 0

        # H&M search API parameters
        # This is the same API their website uses for product listing pages
        gender_filter = "Menswear" if gender == "male" else "Ladies"

        while len(items) < target:
            url = (
                f"https://www2.hm.com/en_us/search-results/api/products"
                f"?q={category_filter}"
                f"&department={gender_filter}"
                f"&page-size={page_size}"
                f"&sort=stock"
                f"&offset={offset}"
                f"&_=1"
            )

            data = await self._fetch_json(url, session=session)

            if not data or not isinstance(data, dict):
                # Fallback: try HTML scraping if API fails
                self._log.info("hm.api_fallback_html", label=label)
                html_items = await self._scrape_html_fallback(session, label, gender, target)
                items.extend(html_items)
                break

            products = data.get("products", [])
            if not products:
                break

            for product in products:
                item = self._parse_product(product, gender)
                if item:
                    items.append(item)

            total_count = data.get("total", 0)
            offset += page_size

            if offset >= total_count or len(items) >= target:
                break

            await self._human_delay()

        self._log.info("hm.category_done", label=label, count=len(items))
        return items[:target]

    async def _scrape_html_fallback(
        self,
        session: httpx.AsyncClient,
        label: str,
        gender: str,
        target: int,
    ) -> list[dict]:
        """
        HTML fallback scraper for H&M product listing pages.
        Used when the API is unavailable or returns empty results.
        """
        items = []
        gender_path = "men" if gender == "male" else "ladies"
        category_map = {
            "men_tops": f"https://www2.hm.com/en_us/{gender_path}/products/tops.html",
            "men_bottoms": f"https://www2.hm.com/en_us/{gender_path}/products/bottoms.html",
            "men_shoes": f"https://www2.hm.com/en_us/{gender_path}/products/shoes.html",
            "women_tops": f"https://www2.hm.com/en_us/{gender_path}/new-arrivals/tops.html",
            "women_bottoms": f"https://www2.hm.com/en_us/{gender_path}/new-arrivals/bottoms.html",
            "women_shoes": f"https://www2.hm.com/en_us/{gender_path}/shoes.html",
        }

        url = category_map.get(label, f"https://www2.hm.com/en_us/{gender_path}/products/tops.html")
        html = await self._fetch_html(url, session=session)
        if not html:
            return []

        soup = self._parse_html(html)

        # H&M product cards use class 'product-item'
        product_cards = soup.select("article.product-item, li.product-item, .product")
        if not product_cards:
            product_cards = soup.select("[class*='product']")

        for card in product_cards[:target]:
            try:
                item = self._parse_html_card(card, gender, label)
                if item:
                    items.append(item)
            except Exception as e:
                self._log.warning("hm.html_parse_error", error=str(e)[:80])

        return items

    def _parse_product(self, product: dict, gender: str) -> Optional[dict]:
        """Parse a single H&M product dict from the API response."""
        try:
            item = make_empty_item()

            # Core fields
            item["external_id"] = f"hm_{product.get('id', '')}"
            item["brand"] = self.BRAND
            item["gender"] = gender

            # Name
            name = product.get("name") or product.get("title") or ""
            if not name:
                return None
            item["name"] = clean_text(name)

            # Price
            price_obj = product.get("price") or {}
            if isinstance(price_obj, dict):
                price_str = price_obj.get("value") or price_obj.get("formatted") or ""
            else:
                price_str = str(price_obj)
            item["price"] = clean_price(price_str)
            item["currency"] = "USD"

            # Description
            item["description"] = clean_text(
                product.get("description") or product.get("longDescription") or "", 800
            )

            # Images
            images = product.get("images") or []
            if images:
                img = images[0]
                if isinstance(img, dict):
                    item["image_url"] = img.get("url") or img.get("src") or ""
                else:
                    item["image_url"] = str(img)
            if item["image_url"] and not item["image_url"].startswith("http"):
                item["image_url"] = f"https:{item['image_url']}" if item["image_url"].startswith("//") else f"https://www2.hm.com{item['image_url']}"

            # Product URL
            url_path = product.get("url") or product.get("link") or ""
            item["product_url"] = self._abs_url(url_path) if url_path else ""

            # Color
            color = product.get("color") or product.get("defaultColor") or ""
            if isinstance(color, dict):
                color = color.get("name") or color.get("label") or ""
            item["color"] = clean_text(str(color), 50).lower() if color else None

            # Sizes
            variants = product.get("variants") or product.get("sizes") or []
            if isinstance(variants, list):
                sizes = []
                for v in variants:
                    if isinstance(v, dict):
                        size = v.get("size") or v.get("name") or v.get("value")
                        if size:
                            sizes.append(str(size))
                item["sizes_available"] = sizes[:20]

            # Sub-category inference
            item["sub_category"] = product.get("subCategory") or product.get("categoryName") or ""
            item["category"] = infer_category(item["name"], item["sub_category"])

            # Availability
            item["is_available"] = product.get("inStock", True) or product.get("sellable", True)

            # Store raw data for debugging
            item["raw_data"] = {
                "source_id": product.get("id"),
                "api_category": product.get("category"),
            }

            return item

        except Exception as e:
            self._log.warning("hm.parse_error", error=str(e)[:80])
            return None

    def _parse_html_card(self, card, gender: str, label: str) -> Optional[dict]:
        """Parse H&M product card from HTML."""
        try:
            item = make_empty_item()
            item["brand"] = self.BRAND
            item["gender"] = gender

            # Name
            name_el = card.select_one("h2, h3, .product-name, [class*='name'], [class*='title']")
            item["name"] = clean_text(name_el.get_text()) if name_el else ""
            if not item["name"]:
                return None

            # Price
            price_el = card.select_one(".price, [class*='price']")
            item["price"] = clean_price(price_el.get_text()) if price_el else None

            # Image
            img_el = card.select_one("img")
            if img_el:
                src = img_el.get("src") or img_el.get("data-src") or img_el.get("data-lazy-src") or ""
                item["image_url"] = self._abs_url(src)

            # Link
            link_el = card.select_one("a")
            if link_el:
                item["product_url"] = self._abs_url(link_el.get("href", ""))

            # Category
            item["category"] = infer_category(item["name"])
            item["sub_category"] = label.split("_")[-1]  # tops | bottoms | shoes

            return item

        except Exception:
            return None