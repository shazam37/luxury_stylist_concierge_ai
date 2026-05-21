"""
Zara Scraper
Uses Zara's internal product catalog API (same endpoints their SPA uses).
Targets: Men's and Women's tops, bottoms, shoes.
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


class ZaraScraper(BaseScraper):
    """
    Scrapes Zara using their internal REST API.
    Zara's SPA loads products via JSON — more stable than HTML scraping.
    """

    SOURCE_NAME = "zara"
    BASE_URL = "https://www.zara.com"
    BRAND = "Zara"

    # Zara's internal category API.
    # These section IDs correspond to real Zara category pages.
    ZARA_API_BASE = "https://www.zara.com/us/en/category"

    # (section_id, gender, label, category_hint, target)
    ZARA_CATEGORIES = [
        # Men
        ("2401", "male", "men_shirts",   "top",    20),
        ("2413", "male", "men_tshirts",  "top",    20),
        ("2404", "male", "men_trousers", "bottom", 20),
        ("2414", "male", "men_jeans",    "bottom", 15),
        ("2406", "male", "men_shoes",    "shoes",  15),
        # Women
        ("4301", "female", "women_tshirts",  "top",    20),
        ("4302", "female", "women_shirts",   "top",    20),
        ("4309", "female", "women_trousers", "bottom", 15),
        ("4315", "female", "women_jeans",    "bottom", 15),
        ("4306", "female", "women_shoes",    "shoes",  15),
    ]

    async def _scrape_source(self, target_count: int = 60) -> list[dict]:
        """
        Scrape Zara via their internal catalog API.
        Runs all category scrapes concurrently.
        """
        all_items: list[dict] = []

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=25,
            headers={
                "User-Agent": self._random_ua(),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.zara.com/us/en/",
                "Origin": "https://www.zara.com",
            }
        ) as session:
            tasks = [
                self._scrape_category(session, section_id, gender, label, cat_hint, target)
                for section_id, gender, label, cat_hint, target in self.ZARA_CATEGORIES
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                label = self.ZARA_CATEGORIES[i][2]
                self._log.warning("zara.category_error", label=label, error=str(result)[:120])
            elif isinstance(result, list):
                all_items.extend(result)

        self._log.info("zara.total_scraped", count=len(all_items))
        return all_items

    async def _scrape_category(
        self,
        session: httpx.AsyncClient,
        section_id: str,
        gender: str,
        label: str,
        category_hint: str,
        target: int,
    ) -> list[dict]:
        """Scrape one Zara category via their API."""

        # Zara's catalog API endpoint
        url = (
            f"https://www.zara.com/us/en/category/{section_id}/products"
            f"?ajax=true"
        )

        data = await self._fetch_json(url, session=session)

        if not data:
            # Try the alternate endpoint format
            url_alt = f"https://www.zara.com/us/en/man-shirts-l737.html?ajax=true&section={section_id}"
            data = await self._fetch_json(url_alt, session=session)

        if not data:
            # Final fallback: HTML scraping
            self._log.info("zara.html_fallback", label=label)
            return await self._scrape_html_fallback(session, label, gender, category_hint, target)

        items = []
        products = self._extract_products_from_response(data)

        for product in products[:target]:
            item = self._parse_product(product, gender, category_hint)
            if item:
                items.append(item)

        self._log.info("zara.category_done", label=label, count=len(items))
        return items

    def _extract_products_from_response(self, data) -> list[dict]:
        """Extract product list from various Zara API response structures."""
        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            # Try common response structures
            for key in ["productGroups", "products", "items", "elements", "data"]:
                if key in data:
                    val = data[key]
                    if isinstance(val, list):
                        # productGroups contains groups with elements
                        if val and isinstance(val[0], dict) and "elements" in val[0]:
                            products = []
                            for group in val:
                                products.extend(group.get("elements", []))
                            return products
                        return val
                    elif isinstance(val, dict):
                        return self._extract_products_from_response(val)

        return []

    def _parse_product(self, product: dict, gender: str, category_hint: str) -> Optional[dict]:
        """Parse a single Zara product dict."""
        try:
            item = make_empty_item()
            item["brand"] = self.BRAND
            item["gender"] = gender

            # Name
            name = product.get("name") or product.get("description") or ""
            if not name:
                return None
            item["name"] = clean_text(name)

            # External ID
            item["external_id"] = f"zara_{product.get('id', '')}"

            # Price — Zara stores prices in cents sometimes
            price_obj = product.get("price")
            if price_obj is None:
                price_obj = product.get("seo", {}).get("price") if isinstance(product.get("seo"), dict) else None
            if isinstance(price_obj, (int, float)):
                # Zara API sometimes returns price * 100 (i.e., 2995 = $29.95)
                item["price"] = price_obj / 100 if price_obj > 1000 else float(price_obj)
            elif price_obj:
                item["price"] = clean_price(str(price_obj))
            item["currency"] = "USD"

            # Description
            desc = product.get("description") or product.get("longDescription") or ""
            if desc == item["name"]:
                desc = product.get("tagline") or product.get("detail") or ""
            item["description"] = clean_text(desc, 800)

            # Images — Zara's image structure
            media_list = (
                product.get("mainImage") or
                product.get("xmedia") or
                product.get("media") or
                []
            )
            if isinstance(media_list, list) and media_list:
                first_media = media_list[0]
                if isinstance(first_media, dict):
                    path = first_media.get("path") or first_media.get("url") or ""
                    name_part = first_media.get("name") or first_media.get("id") or ""
                    timestamp = first_media.get("timestamp") or ""
                    if path and name_part:
                        item["image_url"] = f"https://static.zara.net/assets{path}{name_part}/w/750/{timestamp}.jpg"
                    elif path:
                        item["image_url"] = self._abs_url(path)
            elif isinstance(media_list, dict):
                item["image_url"] = self._abs_url(media_list.get("url") or media_list.get("path") or "")

            # Product URL
            seo = product.get("seo") or {}
            if isinstance(seo, dict):
                seo_keyword = seo.get("keyword") or ""
                seo_id = seo.get("seoProductId") or product.get("id") or ""
                if seo_keyword and seo_id:
                    item["product_url"] = f"https://www.zara.com/us/en/{seo_keyword}-p{seo_id}.html"
            if not item["product_url"]:
                pid = product.get("id") or ""
                item["product_url"] = f"https://www.zara.com/us/en/item-p{pid}.html" if pid else ""

            # Colors
            colors = product.get("detail", {})
            if isinstance(colors, dict):
                color_name = colors.get("colors", [{}])[0].get("name", "") if colors.get("colors") else ""
                item["color"] = clean_text(str(color_name), 50).lower() if color_name else None

                # All available colors
                all_colors = [
                    c.get("name") for c in colors.get("colors", []) if c.get("name")
                ]
                item["colors_available"] = all_colors[:10]

            # Sizes
            sizes_raw = product.get("availableSizes") or []
            if not sizes_raw:
                # Try nested structure
                detail = product.get("detail") or {}
                if isinstance(detail, dict):
                    sizes_raw = detail.get("sizes") or []
            if isinstance(sizes_raw, list):
                sizes = []
                for s in sizes_raw:
                    if isinstance(s, dict):
                        sizes.append(s.get("name") or s.get("value") or str(s))
                    else:
                        sizes.append(str(s))
                item["sizes_available"] = sizes[:20]

            # Category
            item["category"] = category_hint
            item["sub_category"] = product.get("familyName") or product.get("subFamily") or ""

            # Availability
            item["is_available"] = not product.get("outOfStock", False)

            # Raw data
            item["raw_data"] = {
                "source_id": product.get("id"),
                "family": product.get("familyName"),
            }

            return item

        except Exception as e:
            self._log.warning("zara.parse_error", error=str(e)[:100])
            return None

    async def _scrape_html_fallback(
        self,
        session: httpx.AsyncClient,
        label: str,
        gender: str,
        category_hint: str,
        target: int,
    ) -> list[dict]:
        """Fallback HTML scraper for Zara product listing pages."""
        items = []

        # Map label to Zara category URL
        url_map = {
            "men_shirts":    "https://www.zara.com/us/en/man-shirts-l737.html",
            "men_tshirts":   "https://www.zara.com/us/en/man-t-shirts-l856.html",
            "men_trousers":  "https://www.zara.com/us/en/man-trousers-l838.html",
            "men_jeans":     "https://www.zara.com/us/en/man-jeans-l841.html",
            "men_shoes":     "https://www.zara.com/us/en/man-shoes-l769.html",
            "women_tshirts": "https://www.zara.com/us/en/woman-t-shirts-l1362.html",
            "women_shirts":  "https://www.zara.com/us/en/woman-shirts-l1217.html",
            "women_trousers":"https://www.zara.com/us/en/woman-trousers-l1335.html",
            "women_jeans":   "https://www.zara.com/us/en/woman-jeans-l1119.html",
            "women_shoes":   "https://www.zara.com/us/en/woman-shoes-l1251.html",
        }

        url = url_map.get(label, f"https://www.zara.com/us/en/man-shirts-l737.html")
        html = await self._fetch_html(url, session=session)
        if not html:
            return []

        soup = self._parse_html(html)

        # Zara uses JS-rendered content but __PRELOADED_STATE__ is sometimes in the HTML
        import json, re
        preloaded_match = re.search(r"window\.__PRELOADED_STATE__\s*=\s*(\{.+?\});", html, re.DOTALL)
        if preloaded_match:
            try:
                state = json.loads(preloaded_match.group(1))
                products_raw = (
                    state.get("catalog", {}).get("product", {}).get("products", []) or
                    state.get("products", []) or []
                )
                for p in products_raw[:target]:
                    item = self._parse_product(p, gender, category_hint)
                    if item:
                        items.append(item)
                if items:
                    return items
            except Exception:
                pass

        # Pure HTML parsing as last resort
        product_cards = soup.select(".product-grid-product, article[class*='product'], li[class*='product']")
        for card in product_cards[:target]:
            try:
                item = make_empty_item()
                item["brand"] = self.BRAND
                item["gender"] = gender
                item["category"] = category_hint

                name_el = card.select_one("h2, .product-name, [class*='name']")
                item["name"] = clean_text(name_el.get_text()) if name_el else ""
                if not item["name"]:
                    continue

                price_el = card.select_one(".price, [class*='price']")
                item["price"] = clean_price(price_el.get_text()) if price_el else None

                img_el = card.select_one("img")
                if img_el:
                    src = img_el.get("src") or img_el.get("data-src") or ""
                    item["image_url"] = self._abs_url(src)

                link_el = card.select_one("a")
                if link_el:
                    item["product_url"] = self._abs_url(link_el.get("href", ""))

                items.append(item)
            except Exception:
                continue

        return items