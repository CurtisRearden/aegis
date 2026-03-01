"""Tests for aegis/models.py — PurchaseIntent and VerificationResult."""

import pytest
from pydantic import ValidationError

from aegis.models import (
    Confidence,
    Decision,
    ModuleResult,
    PurchaseIntent,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# PurchaseIntent
# ---------------------------------------------------------------------------


class TestPurchaseIntent:
    def test_minimal_valid_intent(self):
        intent = PurchaseIntent(
            item="Laptop",
            price=999.99,
            seller="amazon.com",
            original_instruction="buy a laptop",
        )
        assert intent.item == "Laptop"
        assert intent.price == 999.99
        assert intent.seller == "amazon.com"
        assert intent.currency == "USD"
        assert intent.quantity == 1
        assert intent.budget_limit is None
        assert intent.allowed_sellers == []
        assert intent.blocked_sellers == []
        assert intent.terms == {}

    def test_full_intent(self):
        intent = PurchaseIntent(
            item="Sony WH-1000XM5",
            price=278.00,
            seller="amazon.com",
            original_instruction="best noise-canceling headphones under $300",
            market_price=349.99,
            price_range=(200.0, 400.0),
            budget_limit=300.0,
            session_spend=50.0,
            session_budget=500.0,
            allowed_sellers=["amazon.com", "bestbuy.com"],
            blocked_sellers=["scam-store.xyz"],
            category="electronics",
            currency="usd",  # should be uppercased
            terms={"has_refund_policy": True, "refund_window_days": 30},
        )
        assert intent.currency == "USD"
        assert intent.price_range == (200.0, 400.0)

    def test_currency_is_uppercased(self):
        intent = PurchaseIntent(
            item="Book",
            price=15.00,
            seller="amazon.com",
            original_instruction="buy a book",
            currency="gbp",
        )
        assert intent.currency == "GBP"

    def test_price_must_be_positive(self):
        with pytest.raises(ValidationError):
            PurchaseIntent(
                item="Item",
                price=0,
                seller="seller.com",
                original_instruction="buy something",
            )

    def test_price_negative_rejected(self):
        with pytest.raises(ValidationError):
            PurchaseIntent(
                item="Item",
                price=-10.0,
                seller="seller.com",
                original_instruction="buy something",
            )

    def test_price_range_validation_rejects_inverted(self):
        with pytest.raises(ValidationError):
            PurchaseIntent(
                item="Item",
                price=100.0,
                seller="seller.com",
                original_instruction="buy something",
                price_range=(500.0, 100.0),  # low > high — invalid
            )

    def test_price_range_equal_bounds_rejected(self):
        with pytest.raises(ValidationError):
            PurchaseIntent(
                item="Item",
                price=100.0,
                seller="seller.com",
                original_instruction="buy something",
                price_range=(100.0, 100.0),
            )

    def test_empty_item_rejected(self):
        with pytest.raises(ValidationError):
            PurchaseIntent(
                item="",
                price=10.0,
                seller="seller.com",
                original_instruction="buy something",
            )

    def test_quantity_must_be_at_least_one(self):
        with pytest.raises(ValidationError):
            PurchaseIntent(
                item="Item",
                price=10.0,
                seller="seller.com",
                original_instruction="buy something",
                quantity=0,
            )

    def test_session_spend_defaults_to_zero(self):
        intent = PurchaseIntent(
            item="Item",
            price=10.0,
            seller="seller.com",
            original_instruction="buy something",
        )
        assert intent.session_spend == 0.0


# ---------------------------------------------------------------------------
# ModuleResult
# ---------------------------------------------------------------------------


class TestModuleResult:
    def make_result(self, score: float = 75.0, passed: bool = True) -> ModuleResult:
        return ModuleResult(
            module="price",
            score=score,
            confidence=Confidence.HIGH,
            passed=passed,
            reasons=["Price is within range."],
            flags=[],
        )

    def test_valid_result(self):
        r = self.make_result()
        assert r.module == "price"
        assert r.score == 75.0
        assert r.passed is True

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ModuleResult(
                module="price",
                score=101.0,
                confidence=Confidence.HIGH,
                passed=True,
            )

    def test_score_negative_rejected(self):
        with pytest.raises(ValidationError):
            ModuleResult(
                module="price",
                score=-1.0,
                confidence=Confidence.HIGH,
                passed=True,
            )

    def test_defaults(self):
        r = ModuleResult(
            module="intent",
            score=80.0,
            confidence=Confidence.MEDIUM,
            passed=True,
        )
        assert r.reasons == []
        assert r.flags == []
        assert r.data == {}


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------


class TestVerificationResult:
    def _make_module_result(self, name: str, score: float) -> ModuleResult:
        return ModuleResult(
            module=name,
            score=score,
            confidence=Confidence.HIGH,
            passed=score >= 60,
        )

    def test_scores_property(self):
        modules = {
            "price": self._make_module_result("price", 80.0),
            "intent": self._make_module_result("intent", 90.0),
        }
        result = VerificationResult(
            decision=Decision.APPROVE,
            overall_score=85.0,
            modules=modules,
        )
        assert result.scores == {"price": 80.0, "intent": 90.0}

    def test_summary_string(self):
        modules = {
            "price": self._make_module_result("price", 80.0),
        }
        result = VerificationResult(
            decision=Decision.APPROVE,
            overall_score=80.0,
            modules=modules,
        )
        summary = result.summary()
        assert "APPROVE" in summary
        assert "80" in summary
        assert "price" in summary

    def test_all_decisions(self):
        for decision in [Decision.APPROVE, Decision.FLAG, Decision.BLOCK]:
            result = VerificationResult(
                decision=decision,
                overall_score=50.0,
                modules={},
            )
            assert result.decision == decision

    def test_verified_at_set(self):
        result = VerificationResult(
            decision=Decision.APPROVE,
            overall_score=90.0,
            modules={},
        )
        assert result.verified_at is not None
