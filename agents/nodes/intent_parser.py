"""
Node 1: Intent Parser
Extracts structured intent from the user's natural language prompt.
Output: parsed_intent dict, query_vector
"""
from __future__ import annotations
import json
import structlog
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm, get_settings
from agents.state import StylistState

logger = structlog.get_logger(__name__)

INTENT_SYSTEM_PROMPT = """You are an expert fashion analyst. Extract structured information from the user's styling request.

Return ONLY a valid JSON object with these fields:
{
  "occasion": "string (e.g. yacht party, office, date night, beach, casual friday)",
  "owned_items": [{"name": "item name", "category": "top|bottom|shoes", "color": "color"}],
  "requested_categories": ["top", "bottom", "shoes"],
  "style_keywords": ["keyword1", "keyword2"],
  "color_palette": ["color1", "color2"],
  "gender": "male|female|unisex",
  "season": "summer|winter|spring|autumn|all-season",
  "formality": "casual|smart_casual|business_casual|formal",
  "budget_min": null or number,
  "budget_max": null or number,
  "vibe": "one sentence describing the overall vibe"
}

Rules:
- owned_items: items the user ALREADY HAS (look for "I have", "I own", "I've got")
- requested_categories: what they WANT recommendations for (infer from context)
- If they have bottoms and ask for top/shoes, requested_categories = ["top", "shoes"]
- Always include "shoes" in requested_categories unless they specify otherwise
- Extract color palette from owned items AND occasion
- Be precise about formality level"""


async def run_intent_parser(state: StylistState) -> dict:
    """Parse user prompt into structured intent."""
    logger.info("node.intent_parser.start")

    llm = get_llm()
    settings = get_settings()

    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=f"User request: {state['user_prompt']}"),
    ]

    # Add wardrobe context if available
    wardrobe = state.get("wardrobe_items", [])
    if wardrobe:
        wardrobe_text = "\n".join([f"- {i.get('name')} ({i.get('category')}, {i.get('color', 'unknown color')})" for i in wardrobe[:10]])
        messages.append(HumanMessage(content=f"User's wardrobe:\n{wardrobe_text}"))

    try:
        response = await llm.ainvoke(messages)
        raw = response.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw.strip())

        # Merge with explicit request params
        if state.get("gender"):
            parsed["gender"] = state["gender"]
        if state.get("budget_min"):
            parsed["budget_min"] = state["budget_min"]
        if state.get("budget_max"):
            parsed["budget_max"] = state["budget_max"]

        # Embed the enriched query for RAG
        enriched_query = _build_enriched_query(parsed, state["user_prompt"])
        from embeddings.embedder import get_embedder
        embedder = get_embedder()
        vector = await embedder.embed_query(enriched_query)

        # Track token usage
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "intent_parser": {
                    "prompt_tokens": response.usage_metadata.get("input_tokens", 0),
                    "completion_tokens": response.usage_metadata.get("output_tokens", 0),
                }
            }

        logger.info("node.intent_parser.done", occasion=parsed.get("occasion"), categories=parsed.get("requested_categories"))

        return {
            "parsed_intent": parsed,
            "query_vector": vector,
            "token_usage": usage,
            "agent_trace": state.get("agent_trace", []) + ["intent_parsed"],
        }

    except Exception as e:
        logger.exception("node.intent_parser.error", error=str(e))
        return {
            "parsed_intent": {"occasion": "general", "requested_categories": ["top", "shoes"], "style_keywords": []},
            "query_vector": [],
            "errors": state.get("errors", []) + [f"Intent parsing failed: {str(e)[:80]}"],
            "agent_trace": state.get("agent_trace", []) + ["intent_parse_failed"],
        }


def _build_enriched_query(parsed: dict, original: str) -> str:
    """Build a rich query string for embedding from parsed intent."""
    parts = [original]
    if parsed.get("occasion"):
        parts.append(f"occasion: {parsed['occasion']}")
    if parsed.get("style_keywords"):
        parts.append(" ".join(parsed["style_keywords"]))
    if parsed.get("vibe"):
        parts.append(parsed["vibe"])
    if parsed.get("formality"):
        parts.append(parsed["formality"])
    if parsed.get("season"):
        parts.append(f"{parsed['season']} fashion")
    return " | ".join(parts)