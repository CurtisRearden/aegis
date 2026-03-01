"""Tests for aegis/modules/terms.py — Purchase terms analysis."""

import pytest

from aegis.models import Confidence, PurchaseIntent
from aegis.modules.terms import verify


def make_intent(terms: dict | None = None, **kwargs) -> PurchaseIntent:
    defaults = {
        "item": "Software License",
        "price": 99.0,
        "seller": "softwarevendor.com",
        "original_instruction": "buy a software license",
        "terms": terms or {},
    }
    defaults.update(kwargs)
    return PurchaseIntent(**defaults)


# ---------------------------------------------------------------------------
# No terms data
# ---------------------------------------------------------------------------


class TestNoTermsData:
    @pytest.mark.asyncio
    async def test_no_terms_returns_neutral(self):
        result = await verify(make_intent(terms={}))
        assert result.score == 50.0
        assert result.confidence == Confidence.LOW

    @pytest.mark.asyncio
    async def test_no_terms_does_not_fail(self):
        result = await verify(make_intent(terms={}))
        assert result.passed is True  # neutral — no data to penalize

    @pytest.mark.asyncio
    async def test_no_terms_includes_advisory(self):
        result = await verify(make_intent(terms={}))
        assert any("No terms data" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# Refund policy
# ---------------------------------------------------------------------------


class TestRefundPolicy:
    @pytest.mark.asyncio
    async def test_good_refund_policy_scores_high(self):
        result = await verify(
            make_intent(terms={"has_refund_policy": True, "refund_window_days": 30})
        )
        assert result.score > 85
        assert "NO_REFUND_POLICY" not in result.flags

    @pytest.mark.asyncio
    async def test_no_refund_policy_penalized(self):
        result = await verify(make_intent(terms={"has_refund_policy": False}))
        assert result.score < 80
        assert "NO_REFUND_POLICY" in result.flags

    @pytest.mark.asyncio
    async def test_no_refund_policy_with_require_policy_blocks(self):
        result = await verify(
            make_intent(
                terms={"has_refund_policy": False},
                require_refund_policy=True,
            )
        )
        assert result.score < 60
        assert "NO_REFUND_POLICY" in result.flags

    @pytest.mark.asyncio
    async def test_short_refund_window_flagged(self):
        result = await verify(
            make_intent(terms={"has_refund_policy": True, "refund_window_days": 3})
        )
        assert "SHORT_REFUND_WINDOW" in result.flags

    @pytest.mark.asyncio
    async def test_14_day_refund_not_flagged(self):
        result = await verify(
            make_intent(terms={"has_refund_policy": True, "refund_window_days": 14})
        )
        assert "SHORT_REFUND_WINDOW" not in result.flags


# ---------------------------------------------------------------------------
# Auto-renewal
# ---------------------------------------------------------------------------


class TestAutoRenewal:
    @pytest.mark.asyncio
    async def test_auto_renewal_flagged_by_default(self):
        result = await verify(
            make_intent(
                terms={"has_refund_policy": True, "auto_renewal": True},
                allow_auto_renewal=False,
            )
        )
        assert "AUTO_RENEWAL_DETECTED" in result.flags
        assert result.score < 85

    @pytest.mark.asyncio
    async def test_auto_renewal_allowed_by_policy(self):
        result = await verify(
            make_intent(
                terms={"has_refund_policy": True, "auto_renewal": True},
                allow_auto_renewal=True,
            )
        )
        assert "AUTO_RENEWAL_DETECTED" not in result.flags

    @pytest.mark.asyncio
    async def test_subscription_flagged_when_not_allowed(self):
        result = await verify(
            make_intent(
                terms={"has_refund_policy": True, "subscription": True},
                allow_auto_renewal=False,
            )
        )
        assert "AUTO_RENEWAL_DETECTED" in result.flags

    @pytest.mark.asyncio
    async def test_no_auto_renewal_no_penalty(self):
        result = await verify(
            make_intent(
                terms={
                    "has_refund_policy": True,
                    "refund_window_days": 30,
                    "auto_renewal": False,
                    "subscription": False,
                    "hidden_fees": [],
                }
            )
        )
        assert result.score > 85
        assert "AUTO_RENEWAL_DETECTED" not in result.flags


# ---------------------------------------------------------------------------
# Hidden fees
# ---------------------------------------------------------------------------


class TestHiddenFees:
    @pytest.mark.asyncio
    async def test_hidden_fees_penalize_score(self):
        result = await verify(
            make_intent(
                terms={
                    "has_refund_policy": True,
                    "hidden_fees": ["booking fee", "service charge"],
                }
            )
        )
        assert result.score < 90
        assert "HIDDEN_FEES_DETECTED" in result.flags

    @pytest.mark.asyncio
    async def test_many_hidden_fees_severe_penalty(self):
        result = await verify(
            make_intent(
                terms={
                    "has_refund_policy": True,
                    "hidden_fees": ["fee1", "fee2", "fee3", "fee4"],
                }
            )
        )
        assert result.score < 75

    @pytest.mark.asyncio
    async def test_no_hidden_fees_no_penalty(self):
        result = await verify(
            make_intent(
                terms={
                    "has_refund_policy": True,
                    "refund_window_days": 30,
                    "hidden_fees": [],
                }
            )
        )
        assert "HIDDEN_FEES_DETECTED" not in result.flags


# ---------------------------------------------------------------------------
# Cancellation fee
# ---------------------------------------------------------------------------


class TestCancellationFee:
    @pytest.mark.asyncio
    async def test_high_cancellation_fee_flagged(self):
        # $30 cancellation fee on $100 purchase = 30% > 20% threshold
        result = await verify(
            make_intent(
                price=100.0,
                terms={"has_refund_policy": True, "cancellation_fee": 30.0},
            )
        )
        assert "HIGH_CANCELLATION_FEE" in result.flags

    @pytest.mark.asyncio
    async def test_low_cancellation_fee_not_flagged(self):
        # $10 cancellation fee on $100 purchase = 10% < 20% threshold
        result = await verify(
            make_intent(
                price=100.0,
                terms={"has_refund_policy": True, "cancellation_fee": 10.0},
            )
        )
        assert "HIGH_CANCELLATION_FEE" not in result.flags

    @pytest.mark.asyncio
    async def test_no_cancellation_fee_no_flag(self):
        result = await verify(
            make_intent(terms={"has_refund_policy": True, "cancellation_fee": 0})
        )
        assert "HIGH_CANCELLATION_FEE" not in result.flags


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


class TestConfidence:
    @pytest.mark.asyncio
    async def test_full_terms_data_high_confidence(self):
        result = await verify(
            make_intent(
                terms={
                    "has_refund_policy": True,
                    "auto_renewal": False,
                    "hidden_fees": [],
                    "subscription": False,
                }
            )
        )
        assert result.confidence == Confidence.HIGH

    @pytest.mark.asyncio
    async def test_partial_terms_data_medium_confidence(self):
        result = await verify(
            make_intent(terms={"has_refund_policy": True, "auto_renewal": False})
        )
        assert result.confidence == Confidence.MEDIUM


# ---------------------------------------------------------------------------
# Module structure
# ---------------------------------------------------------------------------


class TestModuleStructure:
    @pytest.mark.asyncio
    async def test_module_name_is_terms(self):
        result = await verify(make_intent())
        assert result.module == "terms"
