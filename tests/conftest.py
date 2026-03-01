"""
Shared pytest fixtures and configuration for the Aegis test suite.
"""

import pytest

from aegis.models import PurchaseIntent


@pytest.fixture
def base_intent() -> PurchaseIntent:
    """A minimal valid PurchaseIntent for use across test modules."""
    return PurchaseIntent(
        item="Sony WH-1000XM5 Headphones",
        price=278.00,
        seller="amazon.com",
        original_instruction="buy the best noise-canceling headphones under $300",
    )


@pytest.fixture
def intent_with_market_data() -> PurchaseIntent:
    """PurchaseIntent with full market price and terms data."""
    return PurchaseIntent(
        item="Sony WH-1000XM5 Headphones",
        price=278.00,
        seller="amazon.com",
        original_instruction="buy the best noise-canceling headphones under $300",
        market_price=349.99,
        budget_limit=300.00,
        terms={
            "has_refund_policy": True,
            "refund_window_days": 30,
            "auto_renewal": False,
            "hidden_fees": [],
            "subscription": False,
        },
    )


@pytest.fixture
def intent_blocked_seller() -> PurchaseIntent:
    """PurchaseIntent that should be blocked due to seller on blocklist."""
    return PurchaseIntent(
        item="Laptop",
        price=999.00,
        seller="scam-store.xyz",
        original_instruction="buy a good laptop",
        blocked_sellers=["scam-store.xyz"],
    )


@pytest.fixture
def intent_over_budget() -> PurchaseIntent:
    """PurchaseIntent that exceeds the specified budget limit."""
    return PurchaseIntent(
        item="High-End Camera",
        price=2000.00,
        seller="amazon.com",
        original_instruction="buy a camera",
        budget_limit=500.00,
    )
