"""Tests for Pydantic schema validation."""
from __future__ import annotations
import pytest
from pydantic import ValidationError
from api.schemas import (
    StyleMeRequest, FashionItem, OutfitOption, TokenUsage,
    BudgetRange, WardrobeItemCreate, CatalogSearchRequest,
    ScraperTriggerRequest, CategoryEnum, GenderEnum,
)


class TestStyleMeRequest:
    def test_valid_basic(self):
        r = StyleMeRequest(prompt="I have navy chinos, what shoes for a yacht party?")
        assert r.prompt == "I have navy chinos, what shoes for a yacht party?"
        assert r.include_alternatives is True

    def test_prompt_stripped(self):
        r = StyleMeRequest(prompt="  hello what should I wear?  ")
        assert r.prompt == "hello what should I wear?"

    def test_prompt_too_short(self):
        with pytest.raises(ValidationError):
            StyleMeRequest(prompt="hi")

    def test_prompt_empty(self):
        with pytest.raises(ValidationError):
            StyleMeRequest(prompt="   ")

    def test_with_budget(self):
        r = StyleMeRequest(
            prompt="Casual friday outfit please, something comfortable",
            budget=BudgetRange(min_price=0, max_price=200),
            gender=GenderEnum.MALE,
        )
        assert r.budget.max_price == 200
        assert r.gender == GenderEnum.MALE

    def test_with_all_fields(self):
        r = StyleMeRequest(
            prompt="Style me for a beach wedding, elegant but relaxed",
            user_id="user-123",
            session_id="sess-456",
            gender=GenderEnum.FEMALE,
            budget=BudgetRange(max_price=500),
            include_alternatives=False,
            include_accessories=True,
        )
        assert r.user_id == "user-123"
        assert r.include_accessories is True


class TestFashionItem:
    def test_minimal(self):
        item = FashionItem(name="White Shirt", category="top")
        assert item.currency == "USD"
        assert item.sizes_available == []

    def test_full(self):
        item = FashionItem(
            name="Slim Linen Shirt",
            brand="Zara",
            category="top",
            color="white",
            price=49.99,
            currency="USD",
            image_url="https://example.com/img.jpg",
            source="zara",
        )
        assert item.price == 49.99
        assert item.brand == "Zara"


class TestOutfitOption:
    def test_total_price_with_items(self):
        outfit = OutfitOption(
            top=FashionItem(name="Shirt", category="top", price=49.99),
            shoes=FashionItem(name="Loafers", category="shoes", price=119.99),
            stylist_note="Great look.",
            total_price=169.98,
        )
        assert outfit.total_price == 169.98

    def test_empty_outfit(self):
        outfit = OutfitOption(stylist_note="Nothing available.", total_price=0.0)
        assert outfit.top is None
        assert outfit.shoes is None


class TestTokenUsage:
    def test_defaults(self):
        t = TokenUsage()
        assert t.prompt_tokens == 0
        assert t.total_tokens == 0

    def test_with_values(self):
        t = TokenUsage(prompt_tokens=500, completion_tokens=200, total_tokens=700)
        assert t.total_tokens == 700


class TestWardrobeItemCreate:
    def test_valid(self):
        w = WardrobeItemCreate(name="Navy Chinos", category=CategoryEnum.BOTTOM, color="navy")
        assert w.category == CategoryEnum.BOTTOM

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            WardrobeItemCreate(name="Something", category="invalid_cat")


class TestScraperTriggerRequest:
    def test_defaults(self):
        r = ScraperTriggerRequest()
        assert r.source == "all"
        assert r.reindex is False

    def test_specific_source(self):
        r = ScraperTriggerRequest(source="zara", reindex=True)
        assert r.source == "zara"
        assert r.reindex is True