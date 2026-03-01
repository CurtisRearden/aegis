"""
Purchase terms analysis module.

Evaluates the terms and conditions of a proposed purchase for consumer-hostile
clauses such as:
  - No refund policy
  - Auto-renewal subscriptions without clear disclosure
  - Hidden fees (booking fees, service charges, processing fees)
  - Cancellation fees
  - Very short refund windows

The agent is expected to populate intent.terms with structured data extracted
from the seller's terms page. If no terms data is provided, the module returns
a LOW-confidence neutral score with an advisory.

Score interpretation:
  85–100  Consumer-friendly terms; clear refund policy, no gotchas.
  65–84   Acceptable terms with minor concerns.
  45–64   Terms have notable consumer-hostile clauses worth reviewing.
  25–44   Terms have serious concerns (auto-renewal, large fees, no refunds).
   0–24   Terms are extremely problematic; likely subscription trap or hidden costs.
"""

from __future__ import annotations

import logging
from typing import Any

from ..models import Confidence, ModuleResult, PurchaseIntent

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 60.0

# Minimum acceptable refund window (days)
_MIN_REFUND_DAYS = 14

# Cancellation fee threshold as a fraction of purchase price above which we flag
_CANCELLATION_FEE_THRESHOLD = 0.20


async def verify(intent: PurchaseIntent) -> ModuleResult:
    """
    Analyze purchase terms for consumer-hostile clauses.

    Reads from intent.terms dict with these recognized keys:
      - has_refund_policy (bool): Whether any refund policy exists.
      - refund_window_days (int): Number of days in the refund window.
      - auto_renewal (bool): Whether the purchase auto-renews.
      - subscription (bool): Whether this is a subscription product.
      - hidden_fees (list[str]): List of discovered hidden/undisclosed fees.
      - cancellation_fee (float): Fee charged to cancel (absolute, same currency).

    Args:
        intent: The PurchaseIntent submitted by the agent.

    Returns:
        ModuleResult with score, confidence, flags, and reasons.
    """
    reasons: list[str] = []
    flags: list[str] = []
    data: dict[str, Any] = {}

    try:
        terms = intent.terms

        if not terms:
            return ModuleResult(
                module="terms",
                score=50.0,
                confidence=Confidence.LOW,
                passed=True,  # No data — don't penalize, but note the gap
                reasons=[
                    "No terms data was provided. Aegis could not verify refund "
                    "policy, auto-renewal clauses, or hidden fees. "
                    "Populate intent.terms for full terms verification."
                ],
                flags=[],
                data={"note": "no_terms_data"},
            )

        score = 100.0

        # ---- 1. Refund policy -----------------------------------------------
        has_refund = terms.get("has_refund_policy")
        refund_days = terms.get("refund_window_days")
        refund_score, refund_data, refund_reasons, refund_flags = _check_refund(
            has_refund, refund_days, intent.require_refund_policy
        )
        score += refund_score  # This is a delta (negative penalty or positive bonus)
        data["refund"] = refund_data
        reasons.extend(refund_reasons)
        flags.extend(refund_flags)

        # ---- 2. Auto-renewal ------------------------------------------------
        auto_renewal = terms.get("auto_renewal", False)
        subscription = terms.get("subscription", False)
        ar_score, ar_data, ar_reasons, ar_flags = _check_auto_renewal(
            auto_renewal, subscription, intent.allow_auto_renewal
        )
        score += ar_score
        data["auto_renewal"] = ar_data
        reasons.extend(ar_reasons)
        flags.extend(ar_flags)

        # ---- 3. Hidden fees -------------------------------------------------
        hidden_fees = terms.get("hidden_fees", [])
        fee_score, fee_data, fee_reasons, fee_flags = _check_hidden_fees(hidden_fees)
        score += fee_score
        data["hidden_fees"] = fee_data
        reasons.extend(fee_reasons)
        flags.extend(fee_flags)

        # ---- 4. Cancellation fee --------------------------------------------
        cancellation_fee = terms.get("cancellation_fee")
        canc_score, canc_data, canc_reasons, canc_flags = _check_cancellation_fee(
            cancellation_fee, intent.price
        )
        score += canc_score
        data["cancellation_fee"] = canc_data
        reasons.extend(canc_reasons)
        flags.extend(canc_flags)

        # Clamp
        score = max(0.0, min(100.0, score))

        # Positive summary if all checks passed cleanly
        if not reasons:
            reasons.append(
                "Purchase terms appear consumer-friendly: refund policy present, "
                "no auto-renewal, no hidden fees detected."
            )

        confidence = _assess_confidence(terms)

    except Exception as exc:
        logger.exception("Terms module error: %s", exc)
        score = 50.0
        confidence = Confidence.LOW
        reasons = [f"Terms verification encountered an error: {exc}"]
        flags = []
        data = {"error": str(exc)}

    passed = score >= PASS_THRESHOLD

    return ModuleResult(
        module="terms",
        score=round(score, 2),
        confidence=confidence,
        passed=passed,
        reasons=reasons,
        flags=flags,
        data=data,
    )


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def _check_refund(
    has_refund: bool | None, refund_days: int | None, require_refund: bool
) -> tuple[float, dict, list[str], list[str]]:
    """Evaluate refund policy. Returns (score_delta, data, reasons, flags)."""
    data: dict = {"has_refund_policy": has_refund, "refund_window_days": refund_days}
    reasons: list[str] = []
    flags: list[str] = []

    if has_refund is False:
        delta = -30.0
        flags.append("NO_REFUND_POLICY")
        if require_refund:
            delta = -50.0
            reasons.append(
                "No refund policy found. Your policy requires a refund policy — "
                "this purchase should be blocked."
            )
        else:
            reasons.append(
                "No refund policy found. You may be unable to return this item "
                "if there is an issue."
            )
        return delta, data, reasons, flags

    if has_refund is True:
        if refund_days is not None:
            if refund_days < _MIN_REFUND_DAYS:
                delta = -10.0
                reasons.append(
                    f"Refund window is only {refund_days} day(s), which is shorter "
                    f"than the recommended {_MIN_REFUND_DAYS} days."
                )
                flags.append("SHORT_REFUND_WINDOW")
            else:
                delta = 5.0
                reasons.append(
                    f"Refund policy present with a {refund_days}-day window."
                )
        else:
            delta = 0.0
            reasons.append("Refund policy present but window length is not specified.")
        return delta, data, reasons, flags

    # has_refund is None — no data
    return 0.0, {**data, "note": "not_specified"}, reasons, flags


def _check_auto_renewal(
    auto_renewal: bool,
    subscription: bool,
    allow_auto_renewal: bool,
) -> tuple[float, dict, list[str], list[str]]:
    """Evaluate auto-renewal / subscription terms."""
    data: dict = {"auto_renewal": auto_renewal, "subscription": subscription}
    reasons: list[str] = []
    flags: list[str] = []

    if auto_renewal or subscription:
        data["detected"] = True
        if not allow_auto_renewal:
            flags.append("AUTO_RENEWAL_DETECTED")
            if auto_renewal:
                reasons.append(
                    "Purchase includes auto-renewal. Your policy does not permit "
                    "auto-renewing purchases without explicit approval."
                )
            else:
                reasons.append(
                    "This is a subscription product. Confirm that recurring charges "
                    "are intended and within budget."
                )
            return -25.0, data, reasons, flags
        else:
            reasons.append(
                "Auto-renewal or subscription detected. Your policy permits this, "
                "but ensure the renewal amount is within budget."
            )
            return -5.0, data, reasons, flags

    data["detected"] = False
    return 0.0, data, reasons, flags


def _check_hidden_fees(hidden_fees: list) -> tuple[float, dict, list[str], list[str]]:
    """Evaluate disclosed hidden/surprise fees."""
    data: dict = {"fees_found": len(hidden_fees), "fees": hidden_fees}
    reasons: list[str] = []
    flags: list[str] = []

    if not hidden_fees:
        return 0.0, data, reasons, flags

    # Each hidden fee reduces score; cap at -40
    delta = max(-40.0, -len(hidden_fees) * 12.0)
    flags.append("HIDDEN_FEES_DETECTED")
    fee_list = ", ".join(f"'{f}'" for f in hidden_fees[:5])
    reasons.append(
        f"The following undisclosed fees were detected: {fee_list}. "
        "These may significantly increase the total cost."
    )
    return delta, data, reasons, flags


def _check_cancellation_fee(
    cancellation_fee: float | None, purchase_price: float
) -> tuple[float, dict, list[str], list[str]]:
    """Evaluate cancellation fee severity relative to purchase price."""
    data: dict = {"cancellation_fee": cancellation_fee}
    reasons: list[str] = []
    flags: list[str] = []

    if cancellation_fee is None or cancellation_fee == 0:
        return 0.0, data, reasons, flags

    fee_ratio = cancellation_fee / purchase_price
    data["fee_ratio"] = round(fee_ratio, 3)

    if fee_ratio >= _CANCELLATION_FEE_THRESHOLD:
        flags.append("HIGH_CANCELLATION_FEE")
        reasons.append(
            f"Cancellation fee of ${cancellation_fee:.2f} represents "
            f"{fee_ratio * 100:.0f}% of the purchase price. "
            "This significantly reduces the effective value of the refund policy."
        )
        return -15.0, data, reasons, flags

    reasons.append(
        f"Cancellation fee of ${cancellation_fee:.2f} "
        f"({fee_ratio * 100:.0f}% of purchase price)."
    )
    return -5.0, data, reasons, flags


def _assess_confidence(terms: dict) -> Confidence:
    """More fields filled in → higher confidence."""
    filled = sum(
        1
        for key in ["has_refund_policy", "auto_renewal", "hidden_fees", "subscription"]
        if terms.get(key) is not None
    )
    if filled >= 3:
        return Confidence.HIGH
    if filled >= 2:
        return Confidence.MEDIUM
    return Confidence.LOW
