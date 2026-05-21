"""
SQLAlchemy ORM models for the Stylist Concierge.
Mirrors the schema defined in scripts/init_db.sql.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer,
    Numeric, SmallInteger, String, Text, ARRAY,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


def _uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id            = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    email         = Column(String(255), unique=True, nullable=False)
    name          = Column(String(255))
    style_profile = Column(JSONB, default=dict)
    budget_range  = Column(JSONB, default=dict)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    updated_at    = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    wardrobe_items  = relationship("WardrobeItem", back_populates="user", cascade="all, delete")
    style_requests  = relationship("StyleRequest", back_populates="user")
    saved_outfits   = relationship("SavedOutfit", back_populates="user", cascade="all, delete")


class WardrobeItem(Base):
    __tablename__ = "wardrobe_items"

    id          = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    user_id     = Column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name        = Column(String(500), nullable=False)
    category    = Column(String(100), nullable=False)
    color       = Column(String(100))
    brand       = Column(String(255))
    description = Column(Text)
    image_url   = Column(Text)
    tags        = Column(ARRAY(Text), default=list)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="wardrobe_items")


class CatalogItem(Base):
    __tablename__ = "catalog_items"

    id                = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    external_id       = Column(String(500))
    source            = Column(String(100), nullable=False)
    name              = Column(String(500), nullable=False)
    category          = Column(String(100), nullable=False)
    sub_category      = Column(String(100))
    price             = Column(Numeric(10, 2))
    currency          = Column(String(10), default="USD")
    color             = Column(String(100))
    colors_available  = Column(ARRAY(Text), default=list)
    brand             = Column(String(255))
    description       = Column(Text)
    image_url         = Column(Text)
    product_url       = Column(Text)
    sizes_available   = Column(ARRAY(Text), default=list)
    gender            = Column(String(50))
    is_available      = Column(Boolean, default=True)
    tags              = Column(ARRAY(Text), default=list)
    raw_data          = Column(JSONB, default=dict)
    qdrant_id         = Column(String(255))
    scraped_at        = Column(DateTime(timezone=True), server_default=func.now())
    updated_at        = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ScrapeJob(Base):
    __tablename__ = "scrape_jobs"

    id            = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    source        = Column(String(100))
    status        = Column(String(50), default="pending")
    triggered_by  = Column(String(100), default="system")
    items_scraped = Column(Integer, default=0)
    items_indexed = Column(Integer, default=0)
    errors        = Column(JSONB, default=list)
    started_at    = Column(DateTime(timezone=True))
    completed_at  = Column(DateTime(timezone=True))
    created_at    = Column(DateTime(timezone=True), server_default=func.now())


class StyleRequest(Base):
    __tablename__ = "style_requests"

    id              = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    user_id         = Column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    session_id      = Column(String(255))
    prompt          = Column(Text, nullable=False)
    parsed_intent   = Column(JSONB, default=dict)
    outfit_response = Column(JSONB, default=dict)
    cache_hit       = Column(Boolean, default=False)
    token_usage     = Column(JSONB, default=dict)
    agent_trace     = Column(ARRAY(Text), default=list)
    latency_ms      = Column(Integer)
    feedback_score  = Column(SmallInteger)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="style_requests")


class SavedOutfit(Base):
    __tablename__ = "saved_outfits"

    id                = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    user_id           = Column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    style_request_id  = Column(UUID(as_uuid=False), ForeignKey("style_requests.id", ondelete="SET NULL"), nullable=True)
    outfit_data       = Column(JSONB, nullable=False)
    name              = Column(String(255))
    notes             = Column(Text)
    created_at        = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="saved_outfits")