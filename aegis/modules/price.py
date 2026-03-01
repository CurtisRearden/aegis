"""
Price verification module.

Evaluates whether the proposed purchase price is reasonable relative to
market data. The module accepts agent-supplied market price context and
applies tiered heuristics to compute a score from 0–100.

Score interpretation:
  90–100  Price is at or below market; excellent deal.
  70–89   Price is within a reasonable margin above market.
  50–69   Price is notably above market; worth flagging.
  30–49   Price is significantly above market; strong concern.
   0–29   Price is extremely inflated; likely a scam or error.
"""

from __future__ import annotations

import logging

from ..models import Confidence, ModuleResult, PurchaseIntent

logger = logging.getLogger(__name__)

# Score boundaries (fraction of market price)
_PRICE_BANDS: list[tuple[float, float, float]] = [
    # (max_ratio, min_ratio, score)  — ratios are proposed/market
    (0.0, 0.5, 95.0),    # >50% below market  → great deal
    (0.5, 0.9, 90.0),    # 10–50% below       → very good
    (0.9, 1.05, 85.0),   # within 5% of market → good
    (1.05, 1.15, 75.0),  # 5–15% above         → acceptable
    (1.15, 1.30, 60.0),  # 15–30% above        → concerning
    (1.30, 1.50, 45.0),  # 30–50% above        → bad
    (1.50, 2.00, 25.0),  # 50–100% above       → very bad
    (2.00, float("inf"), 10.0),  # >100% above → likely fraud
]

PASS_THRESHOLD = 60.0


async def verify(intent: PurchaseIntent) -> ModuleResult:
    """
    Verify whether the proposed price is within an acceptable market range.

    Logic (in priority order):
    1. If a ``price_range`` (min, max) is supplied, the price is scored against
       that range directly.
    2. If a ``market_price`` is supplied, price/market_price ratio is computed
       and mapped through ``_PRICE_BANDS``.
    3. If neither is available, a LOW-confidence neutral score is returned.

    Args:
        intent: The PurchaseIntent submitted by the agent.

    Returns:
        ModuleResult with score, confidence, flags, and reasons.
    """
    reasons: list[str] = []
    flags: list[str] = []
    data: dict = {}

    try:
        # ---- Case 1: explicit price range provided -------------------------
        if intent.price_range is not None:
            low, high = intent.price_range
            score, confidence, reasons, flags, data = _score_against_range(
                intent.price, low, high
            )

        # ---- Case 2: single market price reference provided ----------------
        elif intent.market_price is not None:
            score, confidence, reasons, flags, data = _score_against_market_price(
                intent.price, intent.market_price
            )

        # ---- Case 3: no market data — return low-confidence neutral --------
        else:
            score = 50.0
            confidence = Confidence.LOW
            reasons = [
                "No market price data provided. "
                "Aegis cannot verify whether this price is reasonable. "
                "Consider supplying market_price or price_range."
            ]
            flags = []
            data = {"note": "no_market_data"}
            # When there is no data to verify against, we do not penalize —
            # pass the module but note low confidence.
            return ModuleResult(
                module="price",
                score=50.0,
                confidence=Confidence.LOW,
                passed=True,
                reasons=reasons,
                flags=[],
                data=data,
            )

    except Exception as exc:
        logger.exception("Price module error: %s", exc)
        score = 50.0
        confidence = Confidence.LOW
        reasons = [f"Price verification encountered an error: {exc}"]
        flags = []
        data = {"error": str(exc)}

    passed = score >= PASS_THRESHOLD

    return ModuleResult(
        module="price",
        score=round(score, 2),
        confidence=confidence,
        passed=passed,
        reasons=reasons,
        flags=flags,
        data=data,
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score_against_range(
    price: float, low: float, high: float
) -> tuple[float, Confidence, list[str], list[str], dict]:
    """Score price against a known acceptable range."""
    reasons: list[str] = []
    flags: list[str] = []
    data: dict = {"price": price, "range_low": low, "range_high": high}
    mid = (low + high) / 2

    if price <= low:
        score = 95.0
        reasons.append(
            f"Price ${price:.2f} is at or below the low end of market range "
            f"(${low:.2f}–${high:.2f})."
        )
        confidence = Confidence.HIGH
    elif price <= high:
        # Linear interpolation: full score at low end, 70 at high end
        fraction = (price - low) / (high - low)
        score = 95.0 - fraction * 25.0
        reasons.append(
            f"Price ${price:.2f} is within the expected market range "
            f"(${low:.2f}–${high:.2f})."
        )
        confidence = Confidence.HIGH
    else:
        # Above market range — compute ratio against the midpoint
        ratio = price / mid
        score = _ratio_to_score(ratio)
        overage_pct = (price - high) / high * 100
        reasons.append(
            f"Price ${price:.2f} is {overage_pct:.0f}% above the high end of the "
            f"expected range (${low:.2f}–${high:.2f})."
        )
        flags.append("PRICE_TOO_HIGH")
        confidence = Confidence.HIGH

    data["score"] = round(score, 2)
    return score, confidence, reasons, flags, data


def _score_against_market_price(
    price: float, market_price: float
) -> tuple[float, Confidence, list[str], list[str], dict]:
    """Score price against a single market price reference."""
    reasons: list[str] = []
    flags: list[str] = []
    ratio = price / market_price
    score = _ratio_to_score(ratio)
    diff_pct = (price - market_price) / market_price * 100
    data = {
        "price": price,
        "market_price": market_price,
        "ratio": round(ratio, 3),
        "diff_pct": round(diff_pct, 1),
        "score": round(score, 2),
    }

    if diff_pct < -20:
        # Suspiciously cheap — note before the general above/at-market check
        reasons.append(
            f"Price ${price:.2f} is {abs(diff_pct):.0f}% below the market reference "
            f"(${market_price:.2f}). Verify this is not a counterfeit or fraudulent listing."
        )
        flags.append("PRICE_SUSPICIOUSLY_LOW")
        confidence = Confidence.MEDIUM
    elif diff_pct < 0:
        reasons.append(
            f"Price ${price:.2f} is {abs(diff_pct):.0f}% below the market reference "
            f"(${market_price:.2f}) — a good deal."
        )
        confidence = Confidence.HIGH
    elif diff_pct <= 5:
        reasons.append(
            f"Price ${price:.2f} is within 5% of the market reference "
            f"(${market_price:.2f})."
        )
        confidence = Confidence.HIGH
    elif diff_pct <= 15:
        reasons.append(
            f"Price ${price:.2f} is {diff_pct:.0f}% above the market reference "
            f"(${market_price:.2f}), which is within an acceptable margin."
        )
        confidence = Confidence.HIGH
    elif diff_pct <= 30:
        reasons.append(
            f"Price ${price:.2f} is {diff_pct:.0f}% above the market reference "
            f"(${market_price:.2f}). Consider verifying before purchasing."
        )
        flags.append("PRICE_TOO_HIGH")
        confidence = Confidence.MEDIUM
    else:
        reasons.append(
            f"Price ${price:.2f} is {diff_pct:.0f}% above the market reference "
            f"(${market_price:.2f}). This is significantly above market rate."
        )
        flags.append("PRICE_TOO_HIGH")
        confidence = Confidence.HIGH

    return score, confidence, reasons, flags, data


def _ratio_to_score(ratio: float) -> float:
    """Map price/market ratio to a 0–100 score via the price bands table."""
    for low_r, high_r, score in _PRICE_BANDS:
        if low_r <= ratio < high_r:
            return score
    return 10.0
