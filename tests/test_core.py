"""
Tests for aegis/core.py — verify_purchase() orchestration.

These are integration-level tests that exercise the full pipeline end-to-end.
"""

import asyncio

import pytest

from aegis import verify_purchase
from aegis.core import _coerce_intent, _resolve_modules
from aegis.models import Decision, PurchaseIntent, VerificationResult
from aegis.policy import PolicyConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_intent_dict():
    return {
        "item": "Sony WH-1000XM5 Headphones",
        "price": 278.00,
        "seller": "amazon.com",
        "original_instruction": "buy the best noise-canceling headphones under $300",
        "market_price": 349.99,
        "budget_limit": 300.00,
        "terms": {
            "has_refund_policy": True,
            "refund_window_days": 30,
            "auto_renewal": False,
            "hidden_fees": [],
        },
    }


@pytest.fixture
def blocked_intent_dict():
    return {
        "item": "Laptop",
        "price": 999.00,
        "seller": "scam-store.xyz",
        "original_instruction": "buy a laptop",
        "budget_limit": 500.0,  # price exceeds budget
        "blocked_sellers": ["scam-store.xyz"],
    }


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestVerifyPurchaseBasic:
    @pytest.mark.asyncio
    async def test_returns_verification_result(self, clean_intent_dict):
        result = await verify_purchase(clean_intent_dict)
        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_all_five_modules_run(self, clean_intent_dict):
        result = await verify_purchase(clean_intent_dict)
        assert "price" in result.modules
        assert "intent" in result.modules
        assert "authorization" in result.modules
        assert "seller" in result.modules
        assert "terms" in result.modules

    @pytest.mark.asyncio
    async def test_decision_is_valid_value(self, clean_intent_dict):
        result = await verify_purchase(clean_intent_dict)
        assert result.decision in (Decision.APPROVE, Decision.FLAG, Decision.BLOCK)

    @pytest.mark.asyncio
    async def test_overall_score_in_range(self, clean_intent_dict):
        result = await verify_purchase(clean_intent_dict)
        assert 0 <= result.overall_score <= 100

    @pytest.mark.asyncio
    async def test_intent_embedded_in_result(self, clean_intent_dict):
        result = await verify_purchase(clean_intent_dict)
        assert result.intent is not None
        assert result.intent.item == clean_intent_dict["item"]

    @pytest.mark.asyncio
    async def test_accepts_purchase_intent_object(self, clean_intent_dict):
        intent = PurchaseIntent(**clean_intent_dict)
        result = await verify_purchase(intent)
        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_verified_at_is_set(self, clean_intent_dict):
        result = await verify_purchase(clean_intent_dict)
        assert result.verified_at is not None


# ---------------------------------------------------------------------------
# Decision outcomes for known-good and known-bad scenarios
# ---------------------------------------------------------------------------


class TestDecisionOutcomes:
    @pytest.mark.asyncio
    async def test_legitimate_purchase_approves_or_flags(self, clean_intent_dict):
        result = await verify_purchase(clean_intent_dict)
        assert result.decision in (Decision.APPROVE, Decision.FLAG)

    @pytest.mark.asyncio
    async def test_blocked_seller_blocks(self, blocked_intent_dict):
        result = await verify_purchase(blocked_intent_dict)
        assert result.decision == Decision.BLOCK
        assert "SELLER_BLOCKED" in result.flags

    @pytest.mark.asyncio
    async def test_over_budget_blocks(self):
        result = await verify_purchase({
            "item": "Expensive Laptop",
            "price": 2000.0,
            "seller": "amazon.com",
            "original_instruction": "buy a laptop",
            "budget_limit": 500.0,
        })
        assert result.decision == Decision.BLOCK
        assert "OVER_BUDGET" in result.flags

    @pytest.mark.asyncio
    async def test_trusted_seller_good_price_tends_to_approve(self):
        result = await verify_purchase({
            "item": "Sony WH-1000XM5 Headphones",
            "price": 300.0,
            "seller": "amazon.com",
            "original_instruction": "buy noise-canceling headphones under $350",
            "market_price": 349.99,
            "budget_limit": 400.0,
            "terms": {
                "has_refund_policy": True,
                "refund_window_days": 30,
                "auto_renewal": False,
                "hidden_fees": [],
            },
        })
        assert result.decision in (Decision.APPROVE, Decision.FLAG)


# ---------------------------------------------------------------------------
# Module selection
# ---------------------------------------------------------------------------


class TestModuleSelection:
    @pytest.mark.asyncio
    async def test_subset_of_modules(self, clean_intent_dict):
        result = await verify_purchase(
            clean_intent_dict, modules=["price", "intent"]
        )
        assert "price" in result.modules
        assert "intent" in result.modules
        assert "authorization" not in result.modules
        assert "seller" not in result.modules
        assert "terms" not in result.modules

    @pytest.mark.asyncio
    async def test_single_module(self, clean_intent_dict):
        result = await verify_purchase(clean_intent_dict, modules=["seller"])
        assert set(result.modules.keys()) == {"seller"}

    @pytest.mark.asyncio
    async def test_unknown_module_raises(self, clean_intent_dict):
        with pytest.raises(ValueError, match="Unknown module"):
            await verify_purchase(clean_intent_dict, modules=["nonexistent"])


# ---------------------------------------------------------------------------
# Custom policy
# ---------------------------------------------------------------------------


class TestCustomPolicy:
    @pytest.mark.asyncio
    async def test_strict_policy_flags_good_purchase(self, clean_intent_dict):
        # Use a policy where ANYTHING below 100 is flagged — virtually impossible
        # to approve even for a good purchase (unless every module scores 100).
        # block_threshold at 60 means mid-range purchases are blocked.
        strict = PolicyConfig(approve_threshold=100.0, block_threshold=60.0)
        result = await verify_purchase(
            {
                "item": "Generic Product",
                "price": 100.0,
                "seller": "unknown-store.com",
                "original_instruction": "buy something",
            },
            policy=strict,
        )
        # Unknown seller + no market data + no terms → mid scores → flag or block
        assert result.decision in (Decision.FLAG, Decision.BLOCK)

    @pytest.mark.asyncio
    async def test_lenient_policy_approves(self):
        # Use a fully clean purchase that won't trigger any hard FLAG_FLAGS:
        # trusted seller, within budget, market price provided (well-priced),
        # and a clear intent match.
        lenient = PolicyConfig(
            approve_threshold=20.0,
            block_threshold=5.0,
            flag_flags=frozenset(),  # disable soft-flag overrides for this test
        )
        result = await verify_purchase(
            {
                "item": "Coffee Machine",
                "price": 50.0,
                "seller": "amazon.com",
                "original_instruction": "buy a coffee machine",
                "market_price": 60.0,
                "budget_limit": 100.0,
            },
            policy=lenient,
        )
        assert result.decision == Decision.APPROVE


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    @pytest.mark.asyncio
    async def test_timeout_produces_graceful_fallback(self, clean_intent_dict):
        # With a near-zero timeout, modules should produce fallback results
        result = await verify_purchase(clean_intent_dict, timeout=0.0001)
        assert isinstance(result, VerificationResult)
        # All modules should still appear — with timeout fallbacks
        assert len(result.modules) == 5


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------


class TestInputCoercion:
    def test_coerce_dict_to_intent(self):
        data = {
            "item": "Book",
            "price": 15.0,
            "seller": "amazon.com",
            "original_instruction": "buy a book",
        }
        intent = _coerce_intent(data)
        assert isinstance(intent, PurchaseIntent)
        assert intent.item == "Book"

    def test_coerce_intent_passthrough(self):
        intent = PurchaseIntent(
            item="Book",
            price=15.0,
            seller="amazon.com",
            original_instruction="buy a book",
        )
        result = _coerce_intent(intent)
        assert result is intent

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _coerce_intent("not a dict or intent")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Module resolution
# ---------------------------------------------------------------------------


class TestResolveModules:
    def test_none_returns_all_modules(self):
        modules = _resolve_modules(None)
        assert set(modules.keys()) == {"price", "intent", "authorization", "seller", "terms"}

    def test_subset_returned(self):
        modules = _resolve_modules(["price", "seller"])
        assert set(modules.keys()) == {"price", "seller"}

    def test_unknown_module_raises(self):
        with pytest.raises(ValueError):
            _resolve_modules(["bogus"])
