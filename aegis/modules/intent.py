"""
Intent matching module.

Determines whether the proposed purchase aligns with the user's original
instruction. This is a critical safety check — an agent may correctly price
and source an item but buy the *wrong* thing entirely.

Scoring approach:
  The module uses multi-signal text analysis to compute alignment:
  - Keyword overlap between the instruction and the item name.
  - Category/type keyword detection.
  - Price constraint extraction and validation.
  - Penalty for items that look categorically different from the instruction.

Score interpretation:
  85–100  Strong alignment — item clearly matches the instruction.
  65–84   Reasonable alignment — likely correct with minor ambiguity.
  45–64   Weak alignment — intent is uncertain; worth flagging.
  25–44   Poor alignment — item probably does not match the instruction.
   0–24   No alignment — item appears unrelated to the instruction.
"""

from __future__ import annotations

import logging
import re
import string

from ..models import Confidence, ModuleResult, PurchaseIntent

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 65.0

# Common English stop words to exclude from keyword matching
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "be", "been",
        "i", "me", "my", "we", "our", "you", "your", "it", "its",
        "this", "that", "these", "those", "get", "buy", "purchase",
        "order", "find", "need", "want", "best", "good", "great", "top",
        "some", "any", "all", "most", "more", "less", "very", "really",
        "please", "can", "could", "would", "should", "will", "do",
    }
)

# Product category keyword clusters used to detect cross-category mismatches
_CATEGORY_CLUSTERS: dict[str, frozenset[str]] = {
    "electronics": frozenset(
        {
            "headphones", "earbuds", "speaker", "laptop", "computer", "phone",
            "tablet", "camera", "tv", "monitor", "keyboard", "mouse", "charger",
            "cable", "battery", "smartwatch", "drone", "printer", "projector",
        }
    ),
    "travel": frozenset(
        {
            "flight", "hotel", "airbnb", "rental", "car", "ticket", "booking",
            "reservation", "accommodation", "hostel", "motel", "cruise",
            "vacation", "trip", "travel",
        }
    ),
    "software": frozenset(
        {
            "software", "subscription", "license", "saas", "app", "tool",
            "service", "plan", "account", "membership", "access", "api",
        }
    ),
    "clothing": frozenset(
        {
            "shirt", "shoes", "jacket", "pants", "dress", "hat", "socks",
            "sneakers", "boots", "coat", "sweater", "hoodie", "jeans",
        }
    ),
    "food": frozenset(
        {
            "food", "coffee", "tea", "snack", "meal", "drink", "grocery",
            "restaurant", "delivery", "supplement", "vitamin",
        }
    ),
    "books": frozenset(
        {
            "book", "ebook", "kindle", "audiobook", "textbook", "novel",
            "magazine", "course", "udemy", "coursera",
        }
    ),
}


async def verify(intent: PurchaseIntent) -> ModuleResult:
    """
    Verify that the proposed purchase item matches the user's original instruction.

    Args:
        intent: The PurchaseIntent submitted by the agent.

    Returns:
        ModuleResult with score, confidence, flags, and reasons.
    """
    reasons: list[str] = []
    flags: list[str] = []
    data: dict = {}

    try:
        instruction_tokens = _tokenize(intent.original_instruction)
        item_tokens = _tokenize(intent.item)
        description_tokens = (
            _tokenize(intent.description) if intent.description else set()
        )
        item_tokens_full = item_tokens | description_tokens

        # --- Keyword overlap score (0–100) ---
        overlap_score, overlap_data = _keyword_overlap_score(
            instruction_tokens, item_tokens_full
        )
        data["overlap"] = overlap_data

        # --- Category mismatch penalty ---
        # Include description tokens when checking item category so that
        # descriptions like "noise-canceling wireless headphones" can trigger
        # the electronics cluster even when item name alone is sparse.
        mismatch_penalty, mismatch_data = _category_mismatch_penalty(
            instruction_tokens, item_tokens_full
        )
        data["category"] = mismatch_data

        # --- Price constraint check ---
        price_constraint_score, price_data = _price_constraint_score(
            intent.original_instruction, intent.price
        )
        data["price_constraint"] = price_data

        # Category match bonus: when instruction and item are in the same
        # detected category, grant a boost to handle semantically equivalent
        # but lexically different descriptions (e.g. "noise-canceling" vs
        # "QuietComfort"). This compensates for the limits of lexical matching.
        category_bonus = 0.0
        if mismatch_data.get("reason") == "category_match":
            category_bonus = 30.0

        # Combine: overlap is primary signal, category bonus for same-category
        # matches, mismatch penalty and price constraint are adjustments.
        raw_score = overlap_score + category_bonus - mismatch_penalty + price_constraint_score
        score = max(0.0, min(100.0, raw_score))

        # Determine confidence based on instruction length and signal quality
        confidence = _assess_confidence(
            instruction_tokens, overlap_data, mismatch_data
        )

        # Build human-readable reasons
        reasons, flags = _build_reasons(
            score, overlap_data, mismatch_data, price_data, intent
        )

        data["final_score"] = round(score, 2)

    except Exception as exc:
        logger.exception("Intent module error: %s", exc)
        score = 50.0
        confidence = Confidence.LOW
        reasons = [f"Intent verification encountered an error: {exc}"]
        flags = []
        data = {"error": str(exc)}

    passed = score >= PASS_THRESHOLD

    return ModuleResult(
        module="intent",
        score=round(score, 2),
        confidence=confidence,
        passed=passed,
        reasons=reasons,
        flags=flags,
        data=data,
    )


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, split, and remove stop words."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return {w for w in text.split() if w and w not in _STOP_WORDS and len(w) > 1}


def _keyword_overlap_score(
    instruction_tokens: set[str], item_tokens: set[str]
) -> tuple[float, dict]:
    """
    Compute a recall-weighted overlap score in the 0–80 range.

    Primary signal: instruction recall — how many of the instruction's keywords
    appear in the item. This is the right signal for purchase intent verification:
    an item that satisfies all keywords in the instruction is a good match even
    if the item name also contains many additional specific terms (brand names,
    model numbers) that aren't in the instruction.

    Supplementary: item recall (fraction of item tokens shared with instruction).

    Returns (score, debug_data).
    """
    if not instruction_tokens or not item_tokens:
        return 40.0, {"instruction_recall": 0, "item_recall": 0,
                       "common": [], "note": "empty_token_set"}

    intersection = instruction_tokens & item_tokens

    # Instruction recall: what fraction of instruction requirements appear in item.
    instruction_recall = len(intersection) / len(instruction_tokens)
    # Item recall: fraction of item tokens shared with instruction (lower weight).
    item_recall = len(intersection) / len(item_tokens) if item_tokens else 0.0

    # Weighted combination: instruction recall 70% + item recall 30%
    combined = instruction_recall * 0.70 + item_recall * 0.30

    # Map combined [0, 1] → [10, 80] (remaining 20 pts come from category/price)
    score = 10.0 + combined * 70.0

    return score, {
        "instruction_recall": round(instruction_recall, 3),
        "item_recall": round(item_recall, 3),
        "common": sorted(intersection),
        "combined": round(combined, 3),
    }


def _category_mismatch_penalty(
    instruction_tokens: set[str], item_tokens: set[str]
) -> tuple[float, dict]:
    """
    Detect if the instruction and item belong to clearly different categories.

    Returns a penalty (0–40) and debug data.
    """
    instruction_categories = _classify_tokens(instruction_tokens)
    item_categories = _classify_tokens(item_tokens)

    data = {
        "instruction_categories": sorted(instruction_categories),
        "item_categories": sorted(item_categories),
    }

    if not instruction_categories or not item_categories:
        # No detectable categories — cannot penalize
        return 0.0, {**data, "penalty": 0, "reason": "unclassified"}

    overlap = instruction_categories & item_categories
    if overlap:
        return 0.0, {**data, "penalty": 0, "reason": "category_match"}

    # Mismatch — apply penalty proportional to how distinct the categories are
    penalty = 35.0
    return penalty, {**data, "penalty": penalty, "reason": "category_mismatch"}


def _classify_tokens(tokens: set[str]) -> set[str]:
    """Return the set of category names whose keywords appear in tokens."""
    matched: set[str] = set()
    for category, keywords in _CATEGORY_CLUSTERS.items():
        if tokens & keywords:
            matched.add(category)
    return matched


def _price_constraint_score(instruction: str, price: float) -> tuple[float, dict]:
    """
    Extract price constraints from the instruction and score accordingly.

    Looks for patterns like 'under $300', 'less than 50', 'max $200'.
    Returns a score delta (-10 to +10) and debug data.
    """
    patterns = [
        r"under\s*\$?([\d,]+(?:\.\d+)?)",
        r"less\s+than\s*\$?([\d,]+(?:\.\d+)?)",
        r"max(?:imum)?\s*\$?([\d,]+(?:\.\d+)?)",
        r"no\s+more\s+than\s*\$?([\d,]+(?:\.\d+)?)",
        r"budget\s+of\s*\$?([\d,]+(?:\.\d+)?)",
        r"\$?([\d,]+(?:\.\d+)?)\s+or\s+less",
    ]

    for pattern in patterns:
        match = re.search(pattern, instruction, re.IGNORECASE)
        if match:
            limit_str = match.group(1).replace(",", "")
            limit = float(limit_str)
            data = {"constraint_type": "max", "limit": limit, "price": price}
            if price <= limit:
                return 10.0, {**data, "satisfied": True}
            else:
                overage = price - limit
                return -10.0, {**data, "satisfied": False, "overage": overage}

    return 0.0, {"constraint_type": None}


def _assess_confidence(
    instruction_tokens: set[str], overlap_data: dict, mismatch_data: dict
) -> Confidence:
    """Assess confidence based on instruction richness and signal quality."""
    if len(instruction_tokens) >= 4 and overlap_data.get("instruction_recall", 0) > 0.1:
        return Confidence.HIGH
    if len(instruction_tokens) >= 2:
        return Confidence.MEDIUM
    return Confidence.LOW


def _build_reasons(
    score: float,
    overlap_data: dict,
    mismatch_data: dict,
    price_data: dict,
    intent: PurchaseIntent,
) -> tuple[list[str], list[str]]:
    """Build human-readable reason strings and flags from signal data."""
    reasons: list[str] = []
    flags: list[str] = []

    common = overlap_data.get("common", [])
    if common:
        reasons.append(
            f"Purchase item shares keywords with the original instruction: "
            f"{', '.join(repr(w) for w in common[:5])}."
        )
    else:
        reasons.append(
            "No keywords from the original instruction were found in the "
            "item name. Verify this is the correct item."
        )

    if mismatch_data.get("reason") == "category_mismatch":
        instr_cats = mismatch_data.get("instruction_categories", [])
        item_cats = mismatch_data.get("item_categories", [])
        reasons.append(
            f"Possible category mismatch — instruction suggests "
            f"{instr_cats} but item appears to be {item_cats}."
        )
        flags.append("INTENT_MISMATCH")

    if price_data.get("constraint_type") == "max":
        if not price_data.get("satisfied"):
            overage = price_data.get("overage", 0)
            reasons.append(
                f"Price ${intent.price:.2f} exceeds the instruction's stated "
                f"budget of ${price_data['limit']:.2f} by ${overage:.2f}."
            )
            flags.append("PRICE_EXCEEDS_INSTRUCTION_BUDGET")
        else:
            reasons.append(
                f"Price ${intent.price:.2f} is within the instruction's stated "
                f"limit of ${price_data['limit']:.2f}."
            )

    if score < 45:
        flags.append("INTENT_MISMATCH")

    return reasons, list(dict.fromkeys(flags))  # deduplicate while preserving order
