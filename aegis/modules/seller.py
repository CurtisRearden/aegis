"""
Seller legitimacy verification module.

Evaluates whether the seller appears to be a legitimate, trustworthy merchant
using heuristic signals derived from the seller identifier (typically a domain
name or marketplace username).

In production this module can be augmented with live data sources such as:
  - Domain WHOIS / age lookups
  - Google Safe Browsing API
  - Shopper review aggregators (Trustpilot, BBB)
  - Payment processor merchant verification

Without live data, the module uses structural heuristics and a curated list
of known-good and suspicious patterns to produce a best-effort score.

Score interpretation:
  85–100  Strong legitimacy signals; well-known or verified seller.
  65–84   Seller appears legitimate with no obvious red flags.
  45–64   Mixed signals; seller is unverified but not suspicious.
  25–44   Notable red flags; seller should be validated before purchase.
   0–24   Strong suspicious signals; likely scam or fraudulent seller.
"""

from __future__ import annotations

import logging
import re

from ..models import Confidence, ModuleResult, PurchaseIntent

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 60.0

# ---------------------------------------------------------------------------
# Curated seller reputation data
# ---------------------------------------------------------------------------

# Known-good major marketplaces and retailers (exact domain or name match)
_TRUSTED_SELLERS: frozenset[str] = frozenset(
    {
        "amazon.com", "amazon", "ebay.com", "ebay", "walmart.com", "walmart",
        "bestbuy.com", "bestbuy", "target.com", "target",
        "apple.com", "apple", "microsoft.com", "microsoft",
        "newegg.com", "newegg", "costco.com", "costco",
        "adorama.com", "adorama", "bhphotovideo.com", "bhphoto",
        "homedepot.com", "homedepot", "lowes.com", "lowes",
        "wayfair.com", "wayfair", "chewy.com", "chewy",
        "etsy.com", "etsy", "shopify.com",
        "nike.com", "nike", "adidas.com", "adidas",
        "booking.com", "expedia.com", "airbnb.com",
        "steam", "steampowered.com", "epicgames.com",
        "google.com", "google", "googleplay",
    }
)

# Suspicious TLDs commonly used in scam / phishing sites
_SUSPICIOUS_TLDS: frozenset[str] = frozenset(
    {
        ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".pw",
        ".top", ".click", ".download", ".stream", ".win",
        ".loan", ".review", ".science", ".work", ".party",
    }
)

# Red-flag patterns in domain / seller names
_SUSPICIOUS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(discount|cheap|deal|sale|promo|offer|free|win)\b", re.I),
    re.compile(r"\b(official|authorized|verified|genuine|authentic)\b", re.I),
    re.compile(r"\d{4,}"),            # Long numeric strings in domain
    re.compile(r"-{2,}"),             # Multiple consecutive hyphens
    re.compile(r"[a-z]+-[a-z]+-[a-z]+-"),  # Three+ hyphenated words (e.g. buy-cheap-phones-)
]

# Trusted TLDs (common legitimate domains)
_TRUSTED_TLDS: frozenset[str] = frozenset(
    {".com", ".net", ".org", ".edu", ".gov", ".co.uk", ".co", ".io", ".us"}
)


async def verify(intent: PurchaseIntent) -> ModuleResult:
    """
    Evaluate the legitimacy of the seller.

    Args:
        intent: The PurchaseIntent submitted by the agent.

    Returns:
        ModuleResult with score, confidence, flags, and reasons.
    """
    reasons: list[str] = []
    flags: list[str] = []
    data: dict = {}

    try:
        seller = intent.seller.strip().lower()
        domain = _extract_domain(seller)
        data["seller"] = seller
        data["domain"] = domain

        # ---- 1. Known trusted seller check ----------------------------------
        trusted, trust_data = _check_trusted(seller, domain)
        data["trusted_check"] = trust_data
        if trusted:
            score = 92.0
            confidence = Confidence.HIGH
            reasons.append(
                f"'{intent.seller}' is a recognized major marketplace or retailer."
            )
            return _build_result(score, confidence, passed=True,
                                  reasons=reasons, flags=flags, data=data)

        # ---- 2. Suspicious pattern detection --------------------------------
        suspicion_penalty, suspicion_data = _check_suspicious_patterns(seller, domain)
        data["suspicion_check"] = suspicion_data

        # ---- 3. TLD analysis ------------------------------------------------
        tld_score, tld_data = _check_tld(domain)
        data["tld_check"] = tld_data

        # ---- 4. Domain structure analysis -----------------------------------
        structure_score, structure_data = _check_domain_structure(domain)
        data["structure_check"] = structure_data

        # Aggregate
        base_score = 70.0  # Neutral starting point for unknown sellers
        score = base_score + tld_score + structure_score - suspicion_penalty
        score = max(0.0, min(100.0, score))

        # Determine confidence
        confidence = _assess_confidence(suspicion_data, tld_data, structure_data)

        # Build reasons
        reasons, flags = _build_reasons(
            score, suspicion_data, tld_data, structure_data, intent
        )

    except Exception as exc:
        logger.exception("Seller module error: %s", exc)
        score = 50.0
        confidence = Confidence.LOW
        reasons = [f"Seller verification encountered an error: {exc}"]
        flags = []
        data = {"error": str(exc)}

    passed = score >= PASS_THRESHOLD
    return _build_result(score, confidence, passed=passed,
                          reasons=reasons, flags=flags, data=data)


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def _extract_domain(seller: str) -> str:
    """Extract the domain from a URL or seller string."""
    # Strip protocol
    domain = re.sub(r"^https?://", "", seller)
    # Strip path
    domain = domain.split("/")[0]
    # Strip www.
    domain = re.sub(r"^www\.", "", domain)
    return domain


def _check_trusted(seller: str, domain: str) -> tuple[bool, dict]:
    """Return True if seller matches a known trusted seller."""
    seller_no_tld = re.sub(r"\.[a-z]{2,}$", "", domain)

    candidates = {seller, domain, seller_no_tld}
    for candidate in candidates:
        if candidate in _TRUSTED_SELLERS:
            return True, {"matched": candidate}

    return False, {"matched": None}


def _check_suspicious_patterns(seller: str, domain: str) -> tuple[float, dict]:
    """
    Check for red-flag patterns in seller/domain name.

    Returns a penalty (0–40) and debug data.
    """
    matched_patterns: list[str] = []
    for pattern in _SUSPICIOUS_PATTERNS:
        match = pattern.search(domain)
        if match:
            matched_patterns.append(match.group())

    penalty = min(40.0, len(matched_patterns) * 12.0)
    return penalty, {
        "matched_patterns": matched_patterns,
        "penalty": penalty,
    }


def _check_tld(domain: str) -> tuple[float, dict]:
    """
    Evaluate the TLD for legitimacy.

    Returns a score delta (-20 to +10) and debug data.
    """
    # Extract TLD
    tld = _extract_tld(domain)

    if not tld:
        return 0.0, {"tld": None, "verdict": "no_tld"}

    if tld in _SUSPICIOUS_TLDS:
        return -20.0, {"tld": tld, "verdict": "suspicious"}

    if tld in _TRUSTED_TLDS:
        return 5.0, {"tld": tld, "verdict": "trusted"}

    return 0.0, {"tld": tld, "verdict": "neutral"}


def _extract_tld(domain: str) -> str | None:
    """Extract the TLD including the dot (e.g. '.com')."""
    # Handle multi-part TLDs like .co.uk
    match = re.search(r"(\.[a-z]{2,3}\.[a-z]{2})$", domain)
    if match:
        return match.group()
    match = re.search(r"(\.[a-z]{2,})$", domain)
    if match:
        return match.group()
    return None


def _check_domain_structure(domain: str) -> tuple[float, dict]:
    """
    Analyze domain name structure for legitimacy signals.

    Returns a score delta (-15 to +5) and debug data.
    """
    data: dict = {}

    # Strip TLD for length analysis
    name = re.sub(r"(\.[a-z]{2,3}\.[a-z]{2}|\.[a-z]{2,})$", "", domain)
    data["name_without_tld"] = name

    # Excessively long domain names are suspicious
    if len(name) > 30:
        data["issue"] = "domain_too_long"
        return -15.0, data

    # Domain with digits mixed into the name (not at end) is slightly suspicious
    if re.search(r"[a-z]\d[a-z]", name):
        data["issue"] = "digits_in_name"
        return -8.0, data

    # Clean short domain is a positive signal
    if 3 <= len(name) <= 15 and re.match(r"^[a-z]+$", name):
        data["issue"] = None
        return 5.0, data

    data["issue"] = None
    return 0.0, data


def _assess_confidence(
    suspicion_data: dict, tld_data: dict, structure_data: dict
) -> Confidence:
    """Confidence is medium for all heuristic-only assessments."""
    if suspicion_data.get("matched_patterns") or tld_data.get("verdict") != "neutral":
        return Confidence.MEDIUM
    return Confidence.LOW


def _build_reasons(
    score: float,
    suspicion_data: dict,
    tld_data: dict,
    structure_data: dict,
    intent: PurchaseIntent,
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    flags: list[str] = []

    patterns = suspicion_data.get("matched_patterns", [])
    if patterns:
        reasons.append(
            f"Seller name contains suspicious keywords or patterns: "
            f"{patterns}. Verify seller legitimacy before purchasing."
        )
        flags.append("SUSPICIOUS_SELLER")

    tld_verdict = tld_data.get("verdict")
    if tld_verdict == "suspicious":
        reasons.append(
            f"Seller domain uses a TLD ({tld_data.get('tld')}) commonly "
            "associated with spam or scam sites."
        )
        flags.append("SUSPICIOUS_SELLER")

    if structure_data.get("issue") == "domain_too_long":
        reasons.append(
            "Seller domain name is unusually long, which is a common signal "
            "of phishing or counterfeit storefronts."
        )
        flags.append("SUSPICIOUS_SELLER")

    if score >= PASS_THRESHOLD and not reasons:
        reasons.append(
            f"Seller '{intent.seller}' passes basic legitimacy checks. "
            "No automated verification of business registration was performed."
        )

    if score < PASS_THRESHOLD and not flags:
        flags.append("SUSPICIOUS_SELLER")

    return reasons, list(dict.fromkeys(flags))


def _build_result(
    score: float,
    confidence: Confidence,
    passed: bool,
    reasons: list[str],
    flags: list[str],
    data: dict,
) -> ModuleResult:
    return ModuleResult(
        module="seller",
        score=round(score, 2),
        confidence=confidence,
        passed=passed,
        reasons=reasons,
        flags=flags,
        data=data,
    )
