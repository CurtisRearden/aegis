"""
Aegis MCP Server — Verify AI agent purchases from Claude and other MCP clients.

This server exposes Aegis purchase verification as MCP tools that any
MCP-compatible client (Claude Desktop, Cursor, etc.) can call.

Setup:
    pip install aegis-verify "mcp[cli]"

Run:
    python aegis_mcp_server.py

Configure in Claude Desktop (~/Library/Application Support/Claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "aegis": {
          "command": "python",
          "args": ["/absolute/path/to/aegis_mcp_server.py"]
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from aegis import verify_purchase, PolicyConfig

# Configure logging to stderr (NEVER stdout — it breaks MCP stdio transport)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("aegis-mcp")

# Initialize the MCP server
mcp = FastMCP("aegis")


# ─── Tools ───────────────────────────────────────────────────────────────────


@mcp.tool()
async def verify_agent_purchase(
    item: str,
    price: float,
    seller: str,
    original_instruction: str,
    market_price: float | None = None,
    budget_limit: float | None = None,
    currency: str = "USD",
    category: str | None = None,
    url: str | None = None,
    has_refund_policy: bool | None = None,
    refund_window_days: int | None = None,
    auto_renewal: bool | None = None,
    hidden_fees: list[str] | None = None,
    allow_auto_renewal: bool = False,
    require_refund_policy: bool = False,
    allowed_sellers: list[str] | None = None,
    blocked_sellers: list[str] | None = None,
) -> str:
    """Verify a proposed AI agent purchase before executing it.

    Runs all five Aegis verification modules (price, intent, authorization,
    seller, terms) and returns an approve/flag/block decision with detailed
    reasoning.

    Use this tool BEFORE any purchase is executed to catch overpaying,
    wrong items, scam sellers, hidden fees, and budget violations.

    Args:
        item: Name of the item or service being purchased.
        price: Proposed purchase price.
        seller: Seller domain or name (e.g. 'amazon.com').
        original_instruction: The original user request that triggered this purchase.
        market_price: Known market/reference price for comparison. Improves accuracy.
        budget_limit: Maximum allowed spend for this purchase.
        currency: ISO 4217 currency code (default: USD).
        category: Product category (e.g. 'electronics', 'travel', 'software').
        url: Direct URL of the product listing.
        has_refund_policy: Whether the seller offers refunds.
        refund_window_days: Number of days for refund eligibility.
        auto_renewal: Whether the purchase includes auto-renewal.
        hidden_fees: List of any detected hidden fees.
        allow_auto_renewal: Whether auto-renewal is permitted by user policy.
        require_refund_policy: Whether a refund policy is required by user policy.
        allowed_sellers: List of approved sellers. Empty means all allowed.
        blocked_sellers: List of blocked sellers.

    Returns:
        JSON string with decision (approve/flag/block), overall score,
        per-module breakdown, reasons, and flags.
    """
    logger.info("Verifying purchase: item=%r seller=%r price=%.2f", item, seller, price)

    # Build the purchase intent dict
    intent: dict[str, Any] = {
        "item": item,
        "price": price,
        "seller": seller,
        "original_instruction": original_instruction,
        "currency": currency,
        "allow_auto_renewal": allow_auto_renewal,
        "require_refund_policy": require_refund_policy,
    }

    # Add optional fields
    if market_price is not None:
        intent["market_price"] = market_price
    if budget_limit is not None:
        intent["budget_limit"] = budget_limit
    if category is not None:
        intent["category"] = category
    if url is not None:
        intent["url"] = url
    if allowed_sellers is not None:
        intent["allowed_sellers"] = allowed_sellers
    if blocked_sellers is not None:
        intent["blocked_sellers"] = blocked_sellers

    # Build terms dict from individual fields
    terms: dict[str, Any] = {}
    if has_refund_policy is not None:
        terms["has_refund_policy"] = has_refund_policy
    if refund_window_days is not None:
        terms["refund_window_days"] = refund_window_days
    if auto_renewal is not None:
        terms["auto_renewal"] = auto_renewal
    if hidden_fees is not None:
        terms["hidden_fees"] = hidden_fees
    if terms:
        intent["terms"] = terms

    # Run verification
    result = await verify_purchase(intent)

    # Format response
    response = {
        "decision": result.decision.value,
        "overall_score": round(result.overall_score, 1),
        "summary": result.summary(),
        "reasons": result.reasons,
        "flags": result.flags,
        "modules": {
            name: {
                "score": round(mod.score, 1),
                "passed": mod.passed,
                "confidence": mod.confidence.value,
                "reasons": mod.reasons,
                "flags": mod.flags,
            }
            for name, mod in result.modules.items()
        },
    }

    logger.info(
        "Verification complete: decision=%s score=%.1f",
        result.decision.value,
        result.overall_score,
    )

    return json.dumps(response, indent=2)


@mcp.tool()
async def quick_price_check(
    item: str,
    price: float,
    market_price: float,
    seller: str = "unknown",
) -> str:
    """Quick price check — is this price reasonable for this item?

    A lightweight check that focuses on price verification only.
    Use this for fast sanity checks when you just need to know if a price
    is fair.

    Args:
        item: Name of the item.
        price: Proposed price.
        market_price: Known market/reference price.
        seller: Seller name (optional, defaults to 'unknown').

    Returns:
        JSON string with price verdict and details.
    """
    logger.info("Quick price check: %r at $%.2f vs market $%.2f", item, price, market_price)

    result = await verify_purchase(
        {
            "item": item,
            "price": price,
            "seller": seller,
            "original_instruction": f"buy {item}",
            "market_price": market_price,
        },
        modules=["price"],
    )

    price_mod = result.modules["price"]
    diff_pct = (price - market_price) / market_price * 100

    response = {
        "item": item,
        "price": price,
        "market_price": market_price,
        "difference": f"{diff_pct:+.1f}%",
        "score": round(price_mod.score, 1),
        "verdict": (
            "good_deal" if diff_pct < -10
            else "fair_price" if diff_pct <= 15
            else "overpriced" if diff_pct <= 50
            else "significantly_overpriced"
        ),
        "reasons": price_mod.reasons,
        "flags": price_mod.flags,
    }

    return json.dumps(response, indent=2)


@mcp.tool()
async def check_seller(
    seller: str,
    item: str = "product",
    price: float = 100.00,
) -> str:
    """Check if a seller is legitimate and trustworthy.

    Runs the Aegis seller verification module to assess merchant
    legitimacy based on domain patterns, TLD analysis, and known
    suspicious indicators.

    Args:
        seller: Seller domain or name to check (e.g. 'amazon.com').
        item: Item being purchased (helps with context).
        price: Purchase price (helps with context).

    Returns:
        JSON string with seller trust assessment.
    """
    logger.info("Seller check: %r", seller)

    result = await verify_purchase(
        {
            "item": item,
            "price": price,
            "seller": seller,
            "original_instruction": f"buy {item}",
        },
        modules=["seller"],
    )

    seller_mod = result.modules["seller"]

    response = {
        "seller": seller,
        "score": round(seller_mod.score, 1),
        "trusted": seller_mod.passed,
        "confidence": seller_mod.confidence.value,
        "reasons": seller_mod.reasons,
        "flags": seller_mod.flags,
    }

    return json.dumps(response, indent=2)


# ─── Entry point ─────────────────────────────────────────────────────────────


def main():
    logger.info("Starting Aegis MCP server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
