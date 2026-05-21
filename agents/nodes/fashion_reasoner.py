"""
Node 4: Fashion Reasoner
The creative brain. Applies fashion rules, color theory, and occasion context
to select the best outfit from retrieved items.
Returns: outfit_primary, outfit_alternatives, stylist_note
"""
from __future__ import annotations
import json
import structlog
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm
from agents.state import StylistState

logger = structlog.get_logger(__name__)

FASHION_SYSTEM_PROMPT = """You are a world-class luxury fashion stylist with deep expertise in color theory, 
occasion dressing, and contemporary menswear/womenswear. Your recommendations are precise, inspired, and practical.

FASHION RULES YOU FOLLOW:
1. Color harmony: analogous colors, complementary pairs, or monochromatic looks
2. Formality matching: all pieces must match the occasion's formality level  
3. Season appropriateness: fabric weight and style should suit the season
4. Proportion balance: slim fits pair with relaxed, structured with casual
5. Shoe-trouser harmony: shoe formality must match or exceed trouser formality
6. The 60-30-10 color rule: dominant (60%), secondary (30%), accent (10%)
7. Nautical/yacht: navy, white, stripes, linen, loafers — no heavy fabrics
8. Business casual: chinos/trousers + collared shirt + clean leather shoes
9. For owned items: COMPLEMENT them, don't clash or repeat colors

RESPONSE FORMAT — return ONLY valid JSON:
{
  "primary_outfit": {
    "top": {"name": "...", "brand": "...", "price": 0.0, "color": "...", "image_url": "...", "product_url": "...", "source": "...", "reason": "why this top works"},
    "bottom": {"name": "...", "brand": "...", "price": 0.0, "color": "...", "image_url": "...", "product_url": "...", "source": "...", "reason": "why this bottom works"} or null,
    "shoes": {"name": "...", "brand": "...", "price": 0.0, "color": "...", "image_url": "...", "product_url": "...", "source": "...", "reason": "why these shoes work"},
    "accessory": null or {...same structure...},
    "stylist_note": "2-3 sentences: luxurious, specific, evocative — why this outfit works together"
  },
  "alternatives": [
    {
      "top": {...} or null,
      "bottom": {...} or null,  
      "shoes": {...},
      "stylist_note": "brief note"
    }
  ],
  "style_tags": ["tag1", "tag2"],
  "color_story": "one sentence about the color palette chosen"
}

Select items ONLY from the provided catalog. Use actual names, prices, and image URLs from the catalog.
If a category has no items, use null for that field.
Generate exactly 2 alternatives."""


async def run_fashion_reasoner(state: StylistState) -> dict:
    """Apply fashion intelligence to select and rank outfit combinations."""
    logger.info("node.fashion_reasoner.start")

    retrieved = state.get("retrieved_items", {})
    parsed = state.get("parsed_intent", {})

    if not retrieved:
        return {
            "outfit_primary": _empty_outfit("No items available in catalog. Please run the scraper first."),
            "outfit_alternatives": [],
            "agent_trace": state.get("agent_trace", []) + ["reason_empty_catalog"],
        }

    # Build context for the LLM
    catalog_context = _format_catalog(retrieved)
    user_context = _format_user_context(state, parsed)

    llm = get_llm()
    messages = [
        SystemMessage(content=FASHION_SYSTEM_PROMPT),
        HumanMessage(content=f"{user_context}\n\nAVAILABLE CATALOG:\n{catalog_context}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        raw = response.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw.strip())

        primary = result.get("primary_outfit", {})
        alternatives = result.get("alternatives", [])[:2]
        style_tags = result.get("style_tags", [])

        # Track token usage
        usage = state.get("token_usage", {})
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage["fashion_reasoner"] = {
                "prompt_tokens": response.usage_metadata.get("input_tokens", 0),
                "completion_tokens": response.usage_metadata.get("output_tokens", 0),
            }

        logger.info("node.fashion_reasoner.done", alternatives=len(alternatives))

        return {
            "outfit_primary": primary,
            "outfit_alternatives": alternatives,
            "token_usage": usage,
            "agent_trace": state.get("agent_trace", []) + ["outfit_reasoned"],
        }

    except Exception as e:
        logger.exception("node.fashion_reasoner.error", error=str(e))
        return {
            "outfit_primary": _empty_outfit(f"Fashion reasoning failed: {str(e)[:80]}"),
            "outfit_alternatives": [],
            "errors": state.get("errors", []) + [str(e)],
            "agent_trace": state.get("agent_trace", []) + ["reason_failed"],
        }


def _format_catalog(retrieved: dict) -> str:
    """Format retrieved items into a concise catalog for the LLM."""
    lines = []
    for category, items in retrieved.items():
        lines.append(f"\n[{category.upper()}] ({len(items)} options):")
        for i, item in enumerate(items[:6], 1):  # Limit to top 6 per category
            price_str = f"${item.get('price', 0):.2f}" if item.get('price') else "Price TBD"
            color_str = f", color: {item['color']}" if item.get('color') else ""
            lines.append(
                f"  {i}. {item.get('name', 'Unknown')} | {item.get('brand', item.get('source', ''))} | "
                f"{price_str}{color_str} | img: {item.get('image_url', '')} | url: {item.get('product_url', '')}"
            )
    return "\n".join(lines)


def _format_user_context(state: StylistState, parsed: dict) -> str:
    """Format user context for the fashion reasoner."""
    lines = [f"USER REQUEST: {state.get('user_prompt', '')}"]

    if parsed.get("occasion"):
        lines.append(f"OCCASION: {parsed['occasion']}")
    if parsed.get("formality"):
        lines.append(f"FORMALITY: {parsed['formality']}")
    if parsed.get("season"):
        lines.append(f"SEASON: {parsed['season']}")
    if parsed.get("gender"):
        lines.append(f"GENDER: {parsed['gender']}")
    if parsed.get("vibe"):
        lines.append(f"VIBE: {parsed['vibe']}")
    if parsed.get("style_keywords"):
        lines.append(f"STYLE KEYWORDS: {', '.join(parsed['style_keywords'])}")

    owned = parsed.get("owned_items", []) or state.get("wardrobe_items", [])
    if owned:
        owned_desc = ", ".join([f"{i.get('name')} ({i.get('color', '')})" for i in owned[:5]])
        lines.append(f"USER ALREADY OWNS: {owned_desc}")
        lines.append("→ Recommend items that COMPLEMENT what they own, not duplicates")

    budget_max = parsed.get("budget_max") or state.get("budget_max")
    if budget_max:
        lines.append(f"BUDGET: max ${budget_max}")

    return "\n".join(lines)


def _empty_outfit(note: str) -> dict:
    return {
        "top": None,
        "bottom": None,
        "shoes": None,
        "accessory": None,
        "stylist_note": note,
    }