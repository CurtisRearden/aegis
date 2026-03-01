"""
Authorization module.

Enforces budget limits, session spending caps, seller allowlists/blocklists,
and any other explicit user-defined permissions. This module is the hardest
policy gate — violations here almost always result in a BLOCK.

Score interpretation:
  90–100  Fully authorized with comfortable margin.
  70–89   Authorized, but approaching a limit.
  50–69   Soft constraint triggered (e.g. near limit, seller not on allowlist).
  20–49   Authorization concern (e.g. exceeds budget with some tolerance).
   0–19   Hard policy violation (blocked seller, hard budget exceeded).
"""

from __future__ import annotations

import logging

from ..models import Confidence, ModuleResult, PurchaseIntent

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 70.0

# Budget utilization thresholds
_WARN_THRESHOLD = 0.80   # Start warning at 80% of budget
_NEAR_THRESHOLD = 0.95   # Near-limit flag at 95% of budget


async def verify(intent: PurchaseIntent) -> ModuleResult:
    """
    Verify that the proposed purchase complies with user-defined authorization rules.

    Checks performed (in order):
    1. Blocked seller check — hard block if seller is on the blocklist.
    2. Seller allowlist check — flag if allowlist is set and seller is not on it.
    3. Per-transaction budget limit — hard block if price exceeds limit.
    4. Session cumulative budget — block if session would be exceeded.
    5. Session near-limit warning.

    Args:
        intent: The PurchaseIntent submitted by the agent.

    Returns:
        ModuleResult with score, confidence, flags, and reasons.
    """
    reasons: list[str] = []
    flags: list[str] = []
    data: dict = {}
    score = 100.0

    try:
        # ---- 1. Blocked seller -----------------------------------------------
        blocked, block_data = _check_blocked_seller(intent)
        data["blocked_seller"] = block_data
        if blocked:
            score = 0.0
            flags.append("SELLER_BLOCKED")
            reasons.append(
                f"Seller '{intent.seller}' is on the blocked sellers list "
                "and must not be used."
            )
            # Hard stop — no further checks needed
            return _build_result(score, Confidence.HIGH, passed=False,
                                  reasons=reasons, flags=flags, data=data)

        # ---- 2. Seller allowlist ---------------------------------------------
        allowlist_score, allowlist_data = _check_seller_allowlist(intent)
        data["allowlist"] = allowlist_data
        if allowlist_score < 100:
            score = min(score, allowlist_score)
            if allowlist_score < 60:
                flags.append("UNAUTHORIZED_SELLER")
                reasons.append(
                    f"Seller '{intent.seller}' is not on the approved sellers list. "
                    f"Approved sellers: {intent.allowed_sellers}."
                )
            else:
                reasons.append(
                    f"No seller allowlist is configured; seller '{intent.seller}' "
                    "has not been explicitly approved."
                )

        # ---- 3. Per-transaction budget limit ---------------------------------
        budget_score, budget_data = _check_budget_limit(intent)
        data["budget"] = budget_data
        if budget_score < 100:
            score = min(score, budget_score)
            if budget_score == 0:
                flags.append("OVER_BUDGET")
                reasons.append(
                    f"Purchase price ${intent.price:.2f} exceeds the authorized "
                    f"budget limit of ${intent.budget_limit:.2f}."
                )
            elif budget_score < 70:
                flags.append("NEAR_BUDGET_LIMIT")
                reasons.append(
                    f"Purchase price ${intent.price:.2f} uses "
                    f"{budget_data.get('utilization_pct', 0):.0f}% of the "
                    f"authorized budget (${intent.budget_limit:.2f})."
                )
            else:
                reasons.append(
                    f"Price ${intent.price:.2f} is within the authorized "
                    f"budget of ${intent.budget_limit:.2f} "
                    f"({budget_data.get('utilization_pct', 0):.0f}% utilized)."
                )

        # ---- 4. Session cumulative budget ------------------------------------
        session_score, session_data = _check_session_budget(intent)
        data["session"] = session_data
        if session_score < 100:
            score = min(score, session_score)
            if session_score == 0:
                flags.append("OVER_SESSION_BUDGET")
                reasons.append(
                    f"This purchase would bring session spend to "
                    f"${session_data.get('projected_spend', 0):.2f}, exceeding "
                    f"the session budget of ${intent.session_budget:.2f}."
                )
            elif session_score < 70:
                flags.append("NEAR_SESSION_BUDGET_LIMIT")
                reasons.append(
                    f"Session spend after this purchase would reach "
                    f"${session_data.get('projected_spend', 0):.2f} "
                    f"({session_data.get('utilization_pct', 0):.0f}% of session budget)."
                )

        # If no issues found, add a positive confirmation
        if not reasons:
            reasons.append(
                f"Purchase of ${intent.price:.2f} from '{intent.seller}' is "
                "authorized within all configured limits."
            )

        confidence = _assess_confidence(intent)

    except Exception as exc:
        logger.exception("Authorization module error: %s", exc)
        score = 50.0
        confidence = Confidence.LOW
        reasons = [f"Authorization verification encountered an error: {exc}"]
        flags = []
        data = {"error": str(exc)}

    passed = score >= PASS_THRESHOLD
    return _build_result(score, confidence, passed=passed,
                          reasons=reasons, flags=flags, data=data)


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def _check_blocked_seller(intent: PurchaseIntent) -> tuple[bool, dict]:
    """Return True if seller is on the blocklist."""
    seller_lower = intent.seller.lower()
    for blocked in intent.blocked_sellers:
        if _seller_matches(seller_lower, blocked.lower()):
            return True, {"matched": blocked}
    return False, {"matched": None}


def _check_seller_allowlist(intent: PurchaseIntent) -> tuple[float, dict]:
    """Return score based on allowlist membership. 100 if no list configured."""
    if not intent.allowed_sellers:
        return 100.0, {"allowlist_configured": False}

    seller_lower = intent.seller.lower()
    for allowed in intent.allowed_sellers:
        if _seller_matches(seller_lower, allowed.lower()):
            return 100.0, {"allowlist_configured": True, "matched": allowed}

    return 30.0, {
        "allowlist_configured": True,
        "matched": None,
        "allowed_sellers": intent.allowed_sellers,
    }


def _check_budget_limit(intent: PurchaseIntent) -> tuple[float, dict]:
    """Return score based on per-transaction budget compliance."""
    if intent.budget_limit is None:
        return 100.0, {"limit_configured": False}

    utilization = intent.price / intent.budget_limit
    utilization_pct = utilization * 100
    data = {
        "limit_configured": True,
        "budget_limit": intent.budget_limit,
        "price": intent.price,
        "utilization": round(utilization, 3),
        "utilization_pct": round(utilization_pct, 1),
    }

    if utilization > 1.0:
        return 0.0, data
    if utilization >= _NEAR_THRESHOLD:
        return 60.0, data
    if utilization >= _WARN_THRESHOLD:
        return 75.0, data
    return 100.0, data


def _check_session_budget(intent: PurchaseIntent) -> tuple[float, dict]:
    """Return score based on session cumulative spend."""
    if intent.session_budget is None:
        return 100.0, {"session_budget_configured": False}

    projected = intent.session_spend + intent.price
    utilization = projected / intent.session_budget
    utilization_pct = utilization * 100
    data = {
        "session_budget_configured": True,
        "session_budget": intent.session_budget,
        "current_spend": intent.session_spend,
        "projected_spend": round(projected, 2),
        "utilization": round(utilization, 3),
        "utilization_pct": round(utilization_pct, 1),
    }

    if utilization > 1.0:
        return 0.0, data
    if utilization >= _NEAR_THRESHOLD:
        return 60.0, data
    if utilization >= _WARN_THRESHOLD:
        return 80.0, data
    return 100.0, data


def _seller_matches(seller: str, reference: str) -> bool:
    """
    Flexible seller matching: exact match or domain containment.

    Handles cases like 'amazon' matching 'amazon.com' or vice versa.
    """
    return seller == reference or seller in reference or reference in seller


def _assess_confidence(intent: PurchaseIntent) -> Confidence:
    """High confidence when explicit policy data is available."""
    has_budget = intent.budget_limit is not None
    has_session = intent.session_budget is not None
    has_allowlist = bool(intent.allowed_sellers or intent.blocked_sellers)

    if has_budget or has_session or has_allowlist:
        return Confidence.HIGH
    return Confidence.MEDIUM


def _build_result(
    score: float,
    confidence: Confidence,
    passed: bool,
    reasons: list[str],
    flags: list[str],
    data: dict,
) -> ModuleResult:
    return ModuleResult(
        module="authorization",
        score=round(score, 2),
        confidence=confidence,
        passed=passed,
        reasons=reasons,
        flags=flags,
        data=data,
    )
