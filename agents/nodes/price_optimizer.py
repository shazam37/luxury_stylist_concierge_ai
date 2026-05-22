"""
Node 4.5: Price Optimizer

Sits between fashion_reasoner and response_formatter.
Adds genuine budget intelligence that the fashion_reasoner (focused on style)
cannot do on its own:

  1. Budget compliance check  — if any selected item breaches budget_max,
     swap it for the cheapest semantically-similar item from retrieved_items
     that stays within budget.

  2. Value vs Splurge tagging — labels each alternative as "value_pick",
     "splurge_pick", or "balanced" based on how its total price relates
     to the primary outfit and budget.

  3. Per-item price annotation — adds "price_note" to each item dict
     (e.g. "Best value", "Premium choice", "Mid-range") used by the
     response formatter and visible in the API response.

  4. Budget summary — adds a budget_summary dict to state:
     { total: float, budget_max: float|None, within_budget: bool,
       savings_vs_splurge: float|None }

  5. Savings note injection — if budget is set and the outfit is within it,
     appends a brief savings line to stylist_note.

No LLM call. Pure arithmetic + dict manipulation. Fast and free.
"""

from __future__ import annotations

import structlog
from agents.state import StylistState

logger = structlog.get_logger(__name__)


# ── Price bracket labels ──────────────────────────────────────────────────────

def _price_label(price: float | None, category_avg: float) -> str:
    """Return a human-readable price note relative to category average."""
    if price is None:
        return ""
    ratio = price / category_avg if category_avg > 0 else 1.0
    if ratio < 0.7:
        return "Best value"
    if ratio < 0.95:
        return "Great price"
    if ratio < 1.3:
        return "Mid-range"
    if ratio < 1.8:
        return "Premium"
    return "Luxury pick"


def _outfit_total(outfit: dict) -> float:
    """Sum prices of all non-null items in an outfit dict."""
    total = 0.0
    for key in ("top", "bottom", "shoes", "accessory"):
        item = outfit.get(key)
        if item and isinstance(item, dict):
            price = item.get("price")
            if price:
                total += float(price)
    return round(total, 2)


def _category_avg(retrieved: dict[str, list[dict]], category: str) -> float:
    """Compute average price of retrieved items for a category."""
    items = retrieved.get(category, [])
    prices = [float(i["price"]) for i in items if i.get("price")]
    return sum(prices) / len(prices) if prices else 50.0


def _cheapest_alternative(
    retrieved: dict[str, list[dict]],
    category: str,
    budget_remaining: float,
    exclude_name: str | None = None,
) -> dict | None:
    """
    Find the cheapest in-budget item for a category from retrieved results.
    Excludes the currently selected item by name.
    """
    candidates = [
        i for i in retrieved.get(category, [])
        if i.get("price") is not None
        and float(i["price"]) <= budget_remaining
        and i.get("name") != exclude_name
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda x: float(x["price"]))


def _annotate_item(item: dict | None, category: str, category_avg: float) -> dict | None:
    """Add price_note to an item dict. Returns None if item is None."""
    if not item or not isinstance(item, dict):
        return item
    annotated = dict(item)
    annotated["price_note"] = _price_label(item.get("price"), category_avg)
    return annotated


async def run_price_optimizer(state: StylistState) -> dict:
    """
    Apply budget intelligence to the fashion_reasoner's outfit selections.
    Returns updated outfit_primary, outfit_alternatives, and budget_summary.
    """
    logger.info("node.price_optimizer.start")

    primary = dict(state.get("outfit_primary") or {})
    alternatives = [dict(a) for a in (state.get("outfit_alternatives") or [])]
    retrieved = state.get("retrieved_items") or {}
    parsed = state.get("parsed_intent") or {}

    budget_max: float | None = parsed.get("budget_max") or state.get("budget_max")
    budget_min: float | None = parsed.get("budget_min") or state.get("budget_min")

    # ── 1. Compute per-category averages ─────────────────────────────────────
    categories = ["top", "bottom", "shoes", "accessory"]
    avgs = {cat: _category_avg(retrieved, cat) for cat in categories}

    # ── 2. Annotate primary outfit items with price labels ────────────────────
    for cat in categories:
        primary[cat] = _annotate_item(primary.get(cat), cat, avgs[cat])

    primary_total = _outfit_total(primary)

    # ── 3. Budget compliance — swap over-budget items ─────────────────────────
    swapped: list[str] = []
    if budget_max and primary_total > budget_max:
        logger.info(
            "node.price_optimizer.over_budget",
            total=primary_total,
            budget_max=budget_max,
        )
        remaining = budget_max

        # Sort by price descending — swap most expensive first
        sortable = [
            (cat, primary.get(cat))
            for cat in ["shoes", "top", "bottom", "accessory"]
            if primary.get(cat) and isinstance(primary.get(cat), dict)
            and primary[cat].get("price")
        ]
        sortable.sort(key=lambda x: float(x[1].get("price", 0)), reverse=True)

        for cat, item in sortable:
            item_price = float(item.get("price", 0))
            # Check if swapping this item would bring total within budget
            total_without = primary_total - item_price
            if total_without + item_price <= budget_max:
                # Already fits — keep it, just update remaining
                remaining = budget_max - total_without
                continue

            # Need a cheaper swap
            budget_for_cat = budget_max - total_without
            if budget_for_cat <= 0:
                # Remove the item entirely
                primary[cat] = None
                primary_total = total_without
                swapped.append(f"removed_{cat}")
                continue

            swap = _cheapest_alternative(
                retrieved, cat, budget_for_cat, exclude_name=item.get("name")
            )
            if swap:
                swap_annotated = _annotate_item(swap, cat, avgs[cat])
                primary[cat] = swap_annotated
                primary_total = _outfit_total(primary)
                swapped.append(f"swapped_{cat}")
                logger.info(
                    "node.price_optimizer.swapped",
                    category=cat,
                    from_price=item_price,
                    to_price=swap.get("price"),
                )

        if swapped:
            # Append a note about the swap to stylist_note
            existing_note = primary.get("stylist_note", "")
            primary["stylist_note"] = (
                existing_note.rstrip(".")
                + f" Adjusted to fit your budget — swapped {', '.join(cat.replace('swapped_','').replace('removed_','') for cat in swapped)} for better-value alternatives."
            )

    # ── 4. Tag and annotate alternatives ─────────────────────────────────────
    all_totals = [primary_total] + [_outfit_total(a) for a in alternatives]
    max_total = max(all_totals) if all_totals else primary_total
    min_total = min(all_totals) if all_totals else primary_total
    spread = max_total - min_total

    tagged_alternatives = []
    for alt in alternatives:
        alt = dict(alt)
        for cat in categories:
            alt[cat] = _annotate_item(alt.get(cat), cat, avgs[cat])
        alt_total = _outfit_total(alt)

        # Tag the alternative
        if spread > 20:
            if alt_total <= min_total + spread * 0.35:
                alt["price_tier"] = "value_pick"
                alt["price_tier_label"] = "Value Pick"
            elif alt_total >= min_total + spread * 0.65:
                alt["price_tier"] = "splurge_pick"
                alt["price_tier_label"] = "Splurge Pick"
            else:
                alt["price_tier"] = "balanced"
                alt["price_tier_label"] = "Balanced"
        else:
            alt["price_tier"] = "balanced"
            alt["price_tier_label"] = "Balanced"

        tagged_alternatives.append(alt)

    # ── 5. Budget summary ─────────────────────────────────────────────────────
    splurge_total = max(all_totals) if all_totals else None
    value_total = min(all_totals) if all_totals else None

    budget_summary = {
        "primary_total": primary_total,
        "budget_max": budget_max,
        "budget_min": budget_min,
        "within_budget": (primary_total <= budget_max) if budget_max else True,
        "swapped_items": swapped,
        "savings_vs_splurge": (
            round(splurge_total - primary_total, 2)
            if splurge_total and splurge_total != primary_total
            else None
        ),
        "value_outfit_total": value_total,
        "splurge_outfit_total": splurge_total,
    }

    # ── 6. Savings note ───────────────────────────────────────────────────────
    if (
        budget_max
        and budget_summary["within_budget"]
        and not swapped  # don't double-annotate
        and budget_summary["savings_vs_splurge"]
        and budget_summary["savings_vs_splurge"] > 10
    ):
        savings = budget_summary["savings_vs_splurge"]
        existing_note = primary.get("stylist_note", "")
        primary["stylist_note"] = (
            existing_note.rstrip(".")
            + f" This look comes in at ${primary_total:.0f} — ${savings:.0f} under your budget."
        )

    logger.info(
        "node.price_optimizer.done",
        primary_total=primary_total,
        within_budget=budget_summary["within_budget"],
        swapped=swapped,
        alternatives_tagged=len(tagged_alternatives),
    )

    return {
        "outfit_primary": primary,
        "outfit_alternatives": tagged_alternatives,
        "budget_summary": budget_summary,
        "agent_trace": state.get("agent_trace", []) + [
            f"price_opt:${primary_total:.0f}"
            + (f"(budget:${budget_max:.0f})" if budget_max else "")
            + (f"[swapped:{len(swapped)}]" if swapped else "")
        ],
    }