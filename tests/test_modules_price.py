"""Tests for aegis/modules/price.py — Price verification."""

import pytest

from aegis.models import Confidence, PurchaseIntent
from aegis.modules.price import verify


def make_intent(**kwargs) -> PurchaseIntent:
    defaults = {
        "item": "Sony Headphones",
        "price": 300.0,
        "seller": "amazon.com",
        "original_instruction": "buy headphones",
    }
    defaults.update(kwargs)
    return PurchaseIntent(**defaults)


# ---------------------------------------------------------------------------
# No market data
# ---------------------------------------------------------------------------


class TestNoMarketData:
    @pytest.mark.asyncio
    async def test_returns_neutral_score_low_confidence(self):
        intent = make_intent(price=100.0)
        result = await verify(intent)
        assert result.module == "price"
        assert result.score == 50.0
        assert result.confidence == Confidence.LOW
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_reason_mentions_no_market_data(self):
        result = await verify(make_intent())
        assert any("No market price" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# Market price comparisons
# ---------------------------------------------------------------------------


class TestMarketPrice:
    @pytest.mark.asyncio
    async def test_at_market_price_scores_high(self):
        intent = make_intent(price=100.0, market_price=100.0)
        result = await verify(intent)
        assert result.score >= 80
        assert result.confidence == Confidence.HIGH

    @pytest.mark.asyncio
    async def test_below_market_price_scores_high(self):
        intent = make_intent(price=80.0, market_price=100.0)
        result = await verify(intent)
        assert result.score >= 85

    @pytest.mark.asyncio
    async def test_slightly_above_market_acceptable(self):
        intent = make_intent(price=110.0, market_price=100.0)
        result = await verify(intent)
        # 10% above should still pass
        assert result.score >= 70
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_30_percent_above_market_flags(self):
        intent = make_intent(price=130.0, market_price=100.0)
        result = await verify(intent)
        assert result.score < 70
        assert "PRICE_TOO_HIGH" in result.flags

    @pytest.mark.asyncio
    async def test_double_market_price_very_low_score(self):
        intent = make_intent(price=200.0, market_price=100.0)
        result = await verify(intent)
        assert result.score <= 25
        assert "PRICE_TOO_HIGH" in result.flags

    @pytest.mark.asyncio
    async def test_suspiciously_low_price_flagged(self):
        # >20% below market — possible counterfeit
        intent = make_intent(price=70.0, market_price=100.0)
        result = await verify(intent)
        assert "PRICE_SUSPICIOUSLY_LOW" in result.flags


# ---------------------------------------------------------------------------
# Price range comparisons
# ---------------------------------------------------------------------------


class TestPriceRange:
    @pytest.mark.asyncio
    async def test_price_within_range_high_score(self):
        intent = make_intent(price=150.0, price_range=(100.0, 200.0))
        result = await verify(intent)
        assert result.score >= 70
        assert result.passed is True
        assert result.confidence == Confidence.HIGH

    @pytest.mark.asyncio
    async def test_price_below_range_excellent(self):
        intent = make_intent(price=50.0, price_range=(100.0, 200.0))
        result = await verify(intent)
        assert result.score >= 90

    @pytest.mark.asyncio
    async def test_price_above_range_flagged(self):
        intent = make_intent(price=300.0, price_range=(100.0, 200.0))
        result = await verify(intent)
        assert result.score < 70
        assert "PRICE_TOO_HIGH" in result.flags

    @pytest.mark.asyncio
    async def test_price_at_range_boundary_passes(self):
        intent = make_intent(price=200.0, price_range=(100.0, 200.0))
        result = await verify(intent)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_price_range_takes_priority_over_market_price(self):
        # price_range should be used when both are present
        intent = make_intent(
            price=150.0,
            market_price=500.0,  # would fail if market_price were used
            price_range=(100.0, 200.0),  # price_range takes priority
        )
        result = await verify(intent)
        assert result.passed is True


# ---------------------------------------------------------------------------
# ModuleResult structure
# ---------------------------------------------------------------------------


class TestModuleResultStructure:
    @pytest.mark.asyncio
    async def test_result_has_required_fields(self):
        result = await verify(make_intent(price=100.0, market_price=100.0))
        assert result.module == "price"
        assert 0 <= result.score <= 100
        assert isinstance(result.reasons, list)
        assert isinstance(result.flags, list)
        assert isinstance(result.data, dict)
        assert result.passed is not None
