"""Tests for scraper utilities and pipeline helpers."""
from __future__ import annotations
import pytest
from scraper.base_scraper import (
    clean_price, clean_text, infer_category, infer_gender,
    make_empty_item, BaseScraper,
)
from scraper.pipeline import deduplicate, _item_fingerprint


class TestCleanPrice:
    def test_usd_dollar_sign(self):
        assert clean_price("$49.99") == 49.99

    def test_euro_sign(self):
        assert clean_price("€29.95") == 29.95

    def test_inr_with_comma(self):
        assert clean_price("₹2,499") == 2499.0

    def test_european_format(self):
        # 1.299,99 → 1299.99
        assert clean_price("1.299,99") == 1299.99

    def test_integer_price(self):
        assert clean_price("49") == 49.0

    def test_none_input(self):
        assert clean_price(None) is None

    def test_empty_string(self):
        assert clean_price("") is None

    def test_text_only(self):
        assert clean_price("Free") is None


class TestCleanText:
    def test_collapses_whitespace(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_truncates(self):
        long = "a" * 600
        assert len(clean_text(long, max_length=100)) == 100

    def test_empty(self):
        assert clean_text("") == ""

    def test_none(self):
        assert clean_text(None) == ""


class TestInferCategory:
    def test_shirt(self):
        assert infer_category("Slim Fit Oxford Shirt") == "top"

    def test_tshirt(self):
        assert infer_category("Graphic T-Shirt") == "top"

    def test_jeans(self):
        assert infer_category("Slim Straight Jeans") == "bottom"

    def test_chinos(self):
        assert infer_category("Navy Chinos", "trousers") == "bottom"

    def test_sneakers(self):
        assert infer_category("White Leather Sneakers") == "shoes"

    def test_loafers(self):
        assert infer_category("Suede Loafers", "shoes") == "shoes"

    def test_bag(self):
        assert infer_category("Leather Weekend Bag") == "accessory"

    def test_default(self):
        assert infer_category("Mystery Item XYZ") == "top"


class TestInferGender:
    def test_men(self):
        assert infer_gender("Men's Classic Shirt") == "male"

    def test_women(self):
        assert infer_gender("Women's Floral Dress") == "female"

    def test_ladies(self):
        assert infer_gender("Ladies Summer Top") == "female"

    def test_unisex(self):
        assert infer_gender("Classic White T-Shirt") == "unisex"


class TestMakeEmptyItem:
    def test_has_required_fields(self):
        item = make_empty_item()
        required = ["external_id", "source", "name", "category", "price",
                    "currency", "color", "brand", "description",
                    "image_url", "product_url", "gender", "is_available", "tags"]
        for field in required:
            assert field in item, f"Missing field: {field}"

    def test_defaults(self):
        item = make_empty_item()
        assert item["currency"] == "USD"
        assert item["gender"] == "unisex"
        assert item["is_available"] is True
        assert item["tags"] == []


class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        items = [
            {"source": "zara", "name": "White Shirt", "price": 49.99},
            {"source": "zara", "name": "White Shirt", "price": 49.99},
            {"source": "hm", "name": "Blue Jeans", "price": 59.99},
        ]
        result = deduplicate(items)
        assert len(result) == 2

    def test_keeps_different_sources(self):
        items = [
            {"source": "zara", "name": "White Shirt", "price": 49.99},
            {"source": "hm", "name": "White Shirt", "price": 49.99},
        ]
        result = deduplicate(items)
        assert len(result) == 2

    def test_keeps_different_prices(self):
        items = [
            {"source": "zara", "name": "White Shirt", "price": 49.99},
            {"source": "zara", "name": "White Shirt", "price": 39.99},
        ]
        result = deduplicate(items)
        assert len(result) == 2

    def test_empty_list(self):
        assert deduplicate([]) == []

    def test_single_item(self):
        items = [{"source": "zara", "name": "Shirt", "price": 49.99}]
        assert deduplicate(items) == items


class TestBaseScrapeValidation:
    """Test BaseScraper._validate_and_enrich via a concrete stub."""

    class _StubScraper(BaseScraper):
        SOURCE_NAME = "test"
        async def _scrape_source(self, target_count): return []

    def setup_method(self):
        self.scraper = self._StubScraper()

    def test_valid_item_passes(self):
        item = {**make_empty_item(), "name": "Nice White Shirt", "source": "test"}
        result = self.scraper._validate_and_enrich(item)
        assert result is not None
        assert result["source"] == "test"

    def test_missing_name_rejected(self):
        item = {**make_empty_item(), "name": "", "source": "test"}
        assert self.scraper._validate_and_enrich(item) is None

    def test_short_name_rejected(self):
        item = {**make_empty_item(), "name": "ab", "source": "test"}
        assert self.scraper._validate_and_enrich(item) is None

    def test_external_id_generated(self):
        item = {**make_empty_item(), "name": "Navy Chinos", "external_id": None}
        result = self.scraper._validate_and_enrich(item)
        assert result["external_id"] is not None
        assert "test_" in result["external_id"]

    def test_category_inferred(self):
        item = {**make_empty_item(), "name": "Classic Oxford Shirt", "category": None}
        result = self.scraper._validate_and_enrich(item)
        assert result["category"] == "top"

    def test_color_extracted_from_name(self):
        # "navy" appears in the color list before "blue" and should match
        item = {**make_empty_item(), "name": "Classic Navy Slim Chinos", "color": None}
        result = self.scraper._validate_and_enrich(item)
        assert result["color"] == "navy"

    def test_tags_generated(self):
        item = {**make_empty_item(), "name": "Summer Linen Casual Shirt"}
        result = self.scraper._validate_and_enrich(item)
        assert "linen" in result["tags"]
        assert "casual" in result["tags"]
        assert "summer" in result["tags"]