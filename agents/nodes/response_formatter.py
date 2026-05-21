"""
Node 5: Response Formatter
Assembles the final response, stores result in cache, computes token totals.
"""
from __future__ import annotations
import structlog
from agents.state import StylistState
from api.schemas import FashionItem, OutfitOption, TokenUsage

logger = structlog.get_logger(__name__)


async def run_response_formatter(state: StylistState) -> dict:
    """Build the final structured response from agent state."""
    logger.info("node.response_formatter.start")

    primary_raw = state.get("outfit_primary", {})
    alternatives_raw = state.get("outfit_alternatives", [])
    parsed = state.get("parsed_intent", {})

    # Build OutfitOption objects
    primary_outfit = _build_outfit_option(primary_raw)
    alternatives = [_build_outfit_option(a) for a in alternatives_raw if a]

    # Compute total token usage
    token_usage = _aggregate_token_usage(state.get("token_usage", {}))

    # Build final response dict
    final_response = {
        "cache_hit": state.get("cache_hit", False),
        "parsed_intent": parsed,
        "outfit": primary_outfit,
        "alternatives": alternatives,
        "token_usage": token_usage,
        "agent_trace": state.get("agent_trace", []) + ["formatted"],
        "model_used": _get_model_name(),
    }

    # Store in semantic cache for future requests
    vector = state.get("query_vector", [])
    if vector and not state.get("cache_hit"):
        try:
            from cache.semantic_cache import get_semantic_cache
            cache = get_semantic_cache()
            # Store a serialisable version
            cacheable = {
                "cache_hit": True,
                "parsed_intent": parsed,
                "outfit": primary_outfit.model_dump(),
                "alternatives": [a.model_dump() for a in alternatives],
                "token_usage": token_usage.model_dump(),
                "agent_trace": final_response["agent_trace"] + ["from_cache"],
                "model_used": final_response["model_used"],
            }
            await cache.set(
                prompt=state.get("user_prompt", ""),
                prompt_vector=vector,
                response=cacheable,
            )
        except Exception as e:
            logger.warning("formatter.cache_store_failed", error=str(e))

    return {"final_response": final_response}


def _build_outfit_option(raw: dict) -> OutfitOption:
    """Convert raw LLM output dict to OutfitOption schema."""
    if not raw:
        return OutfitOption(stylist_note="No outfit available.", total_price=0.0)

    def _item(d) -> FashionItem | None:
        if not d or not isinstance(d, dict):
            return None
        return FashionItem(
            name=d.get("name") or "Unknown item",
            brand=d.get("brand"),
            category=d.get("category") or "unknown",
            color=d.get("color"),
            price=float(d["price"]) if d.get("price") else None,
            currency=d.get("currency", "USD"),
            image_url=d.get("image_url"),
            product_url=d.get("product_url"),
            source=d.get("source"),
        )

    top = _item(raw.get("top"))
    bottom = _item(raw.get("bottom"))
    shoes = _item(raw.get("shoes"))
    accessory = _item(raw.get("accessory"))

    total = sum(
        i.price for i in [top, bottom, shoes, accessory]
        if i and i.price is not None
    )

    return OutfitOption(
        top=top,
        bottom=bottom,
        shoes=shoes,
        accessory=accessory,
        total_price=round(total, 2),
        stylist_note=raw.get("stylist_note") or "A carefully curated look for you.",
        style_tags=raw.get("style_tags") or [],
    )


def _aggregate_token_usage(usage_dict: dict) -> TokenUsage:
    """Sum up token usage across all nodes."""
    total_prompt = 0
    total_completion = 0

    for node, usage in usage_dict.items():
        if isinstance(usage, dict):
            total_prompt += usage.get("prompt_tokens", 0)
            total_completion += usage.get("completion_tokens", 0)

    total = total_prompt + total_completion

    # Rough cost estimate (Groq is ~$0.05/M input, $0.08/M output)
    cost = (total_prompt * 0.00000005) + (total_completion * 0.00000008)

    return TokenUsage(
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        total_tokens=total,
        estimated_cost_usd=round(cost, 6) if cost > 0 else None,
    )


def _get_model_name() -> str:
    from config.settings import get_settings
    s = get_settings()
    return f"{s.llm_provider.value}/{s.llm_model}"