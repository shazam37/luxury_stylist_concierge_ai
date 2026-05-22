"""
Pydantic v2 schemas for all API request and response bodies.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, field_validator


# ─────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────

class CategoryEnum(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"
    SHOES = "shoes"
    ACCESSORY = "accessory"
    OUTERWEAR = "outerwear"


class GenderEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNISEX = "unisex"


class StyleEnum(str, Enum):
    CASUAL = "casual"
    FORMAL = "formal"
    SMART_CASUAL = "smart_casual"
    STREETWEAR = "streetwear"
    MINIMALIST = "minimalist"
    BOHEMIAN = "bohemian"
    ATHLETIC = "athletic"
    LUXURY = "luxury"


# ─────────────────────────────────────────────
#  Shared Sub-models
# ─────────────────────────────────────────────

class FashionItem(BaseModel):
    """A single fashion item in a recommendation."""
    name: str
    brand: Optional[str] = None
    category: str
    sub_category: Optional[str] = None
    color: Optional[str] = None
    price: Optional[float] = None
    currency: str = "USD"
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    source: Optional[str] = None
    sizes_available: list[str] = []
    score: Optional[float] = None         # semantic similarity score


class OutfitOption(BaseModel):
    """One complete outfit recommendation."""
    top: Optional[FashionItem] = None
    bottom: Optional[FashionItem] = None
    shoes: Optional[FashionItem] = None
    accessory: Optional[FashionItem] = None
    total_price: float = 0.0
    stylist_note: str
    style_tags: list[str] = []
    # Price optimizer fields
    price_tier: Optional[str] = None        # "value_pick" | "splurge_pick" | "balanced"
    price_tier_label: Optional[str] = None  # human-readable label


class BudgetSummary(BaseModel):
    """Budget analysis produced by price_optimizer node."""
    primary_total: float = 0.0
    budget_max: Optional[float] = None
    budget_min: Optional[float] = None
    within_budget: bool = True
    swapped_items: list[str] = []
    savings_vs_splurge: Optional[float] = None
    value_outfit_total: Optional[float] = None
    splurge_outfit_total: Optional[float] = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: Optional[float] = None


# ─────────────────────────────────────────────
#  Style-Me Request
# ─────────────────────────────────────────────

class BudgetRange(BaseModel):
    min_price: Optional[float] = Field(default=None, ge=0)
    max_price: Optional[float] = Field(default=None, ge=0)
    currency: str = "USD"


class StyleMeRequest(BaseModel):
    """
    POST /api/v1/style-me
    The core endpoint request body.
    """
    prompt: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Natural language styling request",
        examples=["I have dark navy chinos. What t-shirt and shoes for a summer yacht party?"],
    )
    user_id: Optional[str] = Field(default=None, description="Optional user ID to load wardrobe + profile")
    session_id: Optional[str] = Field(default=None, description="Conversation session ID for multi-turn")
    gender: Optional[GenderEnum] = None
    budget: Optional[BudgetRange] = None
    style_preference: Optional[StyleEnum] = None
    include_alternatives: bool = Field(default=True, description="Return 2 alternate outfit options")
    include_accessories: bool = Field(default=False, description="Include accessory recommendations")

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()


# ─────────────────────────────────────────────
#  Style-Me Response
# ─────────────────────────────────────────────

class StyleMeResponse(BaseModel):
    """
    POST /api/v1/style-me — Response body.
    """
    request_id: str
    user_prompt: str
    cache_hit: bool = False

    # Parsed understanding of the prompt
    parsed_intent: dict[str, Any] = {}

    # Outfit recommendations
    outfit: OutfitOption
    alternatives: list[OutfitOption] = []

    # Budget analysis (populated when budget is set or always)
    budget_summary: Optional[BudgetSummary] = None

    # Metadata
    token_usage: TokenUsage = TokenUsage()
    agent_trace: list[str] = []
    latency_ms: Optional[int] = None
    model_used: Optional[str] = None


# ─────────────────────────────────────────────
#  Catalog Schemas
# ─────────────────────────────────────────────

class CatalogItemResponse(BaseModel):
    id: str
    source: str
    name: str
    category: str
    sub_category: Optional[str] = None
    price: Optional[float] = None
    currency: str = "USD"
    color: Optional[str] = None
    brand: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    gender: Optional[str] = None
    sizes_available: list[str] = []
    is_available: bool = True
    tags: list[str] = []


class CatalogSearchRequest(BaseModel):
    query: Optional[str] = None
    category: Optional[CategoryEnum] = None
    gender: Optional[GenderEnum] = None
    min_price: Optional[float] = Field(default=None, ge=0)
    max_price: Optional[float] = Field(default=None, ge=0)
    color: Optional[str] = None
    source: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class CatalogStatsResponse(BaseModel):
    total_items: int
    by_category: dict[str, int]
    by_source: dict[str, int]
    by_gender: dict[str, int]
    price_range: dict[str, float]
    qdrant_vectors: int


# ─────────────────────────────────────────────
#  Wardrobe Schemas
# ─────────────────────────────────────────────

class WardrobeItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    category: CategoryEnum
    color: Optional[str] = None
    brand: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    tags: list[str] = []


class WardrobeItemResponse(BaseModel):
    id: str
    name: str
    category: str
    color: Optional[str] = None
    brand: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    tags: list[str] = []
    created_at: Optional[str] = None


# ─────────────────────────────────────────────
#  Scraper Schemas
# ─────────────────────────────────────────────

class ScraperTriggerRequest(BaseModel):
    source: str = Field(
        default="all",
        description="Which source to scrape: zara | hm | myntra | all",
    )
    reindex: bool = Field(default=False, description="Drop and re-index existing items for this source")


class ScraperJobResponse(BaseModel):
    job_id: str
    source: str
    status: str
    message: str


class ScraperJobStatusResponse(BaseModel):
    job_id: str
    source: str
    status: str
    items_scraped: int
    items_indexed: int
    errors: list[str]
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# ─────────────────────────────────────────────
#  Feedback Schema
# ─────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    request_id: str
    score: int = Field(..., ge=1, le=5, description="1=terrible, 5=perfect")
    notes: Optional[str] = None


# ─────────────────────────────────────────────
#  Health / Info
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, str]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None