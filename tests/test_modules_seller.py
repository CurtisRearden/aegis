"""Tests for aegis/modules/seller.py — Seller legitimacy verification."""

import pytest

from aegis.models import Confidence, PurchaseIntent
from aegis.modules.seller import verify


def make_intent(seller: str = "amazon.com", **kwargs) -> PurchaseIntent:
    defaults = {
        "item": "Headphones",
        "price": 100.0,
        "seller": seller,
        "original_instruction": "buy headphones",
    }
    defaults.update(kwargs)
    return PurchaseIntent(**defaults)


# ---------------------------------------------------------------------------
# Trusted sellers
# ---------------------------------------------------------------------------


class TestTrustedSellers:
    @pytest.mark.asyncio
    async def test_amazon_scores_high(self):
        result = await verify(make_intent("amazon.com"))
        assert result.score >= 85
        assert result.confidence == Confidence.HIGH
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_bestbuy_scores_high(self):
        result = await verify(make_intent("bestbuy.com"))
        assert result.score >= 85

    @pytest.mark.asyncio
    async def test_apple_scores_high(self):
        result = await verify(make_intent("apple.com"))
        assert result.score >= 85

    @pytest.mark.asyncio
    async def test_ebay_scores_high(self):
        result = await verify(make_intent("ebay"))
        assert result.score >= 85

    @pytest.mark.asyncio
    async def test_walmart_scores_high(self):
        result = await verify(make_intent("walmart"))
        assert result.score >= 85


# ---------------------------------------------------------------------------
# Suspicious sellers
# ---------------------------------------------------------------------------


class TestSuspiciousSellers:
    @pytest.mark.asyncio
    async def test_suspicious_tld_scores_low(self):
        result = await verify(make_intent("headphones-deals.xyz"))
        assert result.score < 70
        assert "SUSPICIOUS_SELLER" in result.flags

    @pytest.mark.asyncio
    async def test_suspicious_tld_tk_flagged(self):
        result = await verify(make_intent("bestdeals.tk"))
        assert "SUSPICIOUS_SELLER" in result.flags

    @pytest.mark.asyncio
    async def test_suspicious_keyword_in_domain(self):
        result = await verify(make_intent("official-apple-discount.com"))
        assert result.score < 80

    @pytest.mark.asyncio
    async def test_long_domain_penalized(self):
        long_domain = "buy-cheap-authentic-branded-electronics-direct.com"
        result = await verify(make_intent(long_domain))
        assert result.score < 80

    @pytest.mark.asyncio
    async def test_scam_domain_fails(self):
        result = await verify(make_intent("buy-cheap-phones-now.xyz"))
        assert result.score < 60
        assert "SUSPICIOUS_SELLER" in result.flags


# ---------------------------------------------------------------------------
# Unknown / neutral sellers
# ---------------------------------------------------------------------------


class TestNeutralSellers:
    @pytest.mark.asyncio
    async def test_unknown_com_domain_neutral(self):
        result = await verify(make_intent("mynewstore.com"))
        # Should not fail outright — just unverified
        assert result.score > 30

    @pytest.mark.asyncio
    async def test_clean_short_domain_scores_well(self):
        result = await verify(make_intent("shopright.com"))
        assert result.score >= 55

    @pytest.mark.asyncio
    async def test_io_domain_neutral(self):
        result = await verify(make_intent("myshop.io"))
        assert result.score > 30


# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------


class TestUrlHandling:
    @pytest.mark.asyncio
    async def test_full_url_extracts_domain(self):
        # Seller field contains a full URL — module should strip protocol/path
        result = await verify(make_intent("https://www.amazon.com/dp/B09XS7JWHH"))
        assert result.score >= 85

    @pytest.mark.asyncio
    async def test_www_prefix_stripped(self):
        result = await verify(make_intent("www.amazon.com"))
        assert result.score >= 85


# ---------------------------------------------------------------------------
# ModuleResult structure
# ---------------------------------------------------------------------------


class TestModuleResultStructure:
    @pytest.mark.asyncio
    async def test_module_name_is_seller(self):
        result = await verify(make_intent())
        assert result.module == "seller"

    @pytest.mark.asyncio
    async def test_result_has_reasons(self):
        result = await verify(make_intent("amazon.com"))
        assert isinstance(result.reasons, list)
        assert len(result.reasons) > 0

    @pytest.mark.asyncio
    async def test_score_in_range(self):
        for seller in ["amazon.com", "scam.xyz", "mynewstore.com"]:
            result = await verify(make_intent(seller))
            assert 0 <= result.score <= 100
