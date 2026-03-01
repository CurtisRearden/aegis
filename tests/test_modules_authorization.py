"""Tests for aegis/modules/authorization.py — Budget and permission checks."""

import pytest

from aegis.models import PurchaseIntent
from aegis.modules.authorization import verify


def make_intent(**kwargs) -> PurchaseIntent:
    defaults = {
        "item": "Laptop",
        "price": 500.0,
        "seller": "amazon.com",
        "original_instruction": "buy a laptop",
    }
    defaults.update(kwargs)
    return PurchaseIntent(**defaults)


# ---------------------------------------------------------------------------
# No policy configured
# ---------------------------------------------------------------------------


class TestNoPolicyConfigured:
    @pytest.mark.asyncio
    async def test_no_limits_scores_100(self):
        result = await verify(make_intent())
        assert result.score == 100.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_no_limits_no_flags(self):
        result = await verify(make_intent())
        assert result.flags == []


# ---------------------------------------------------------------------------
# Budget limit checks
# ---------------------------------------------------------------------------


class TestBudgetLimit:
    @pytest.mark.asyncio
    async def test_price_within_budget_passes(self):
        result = await verify(make_intent(price=200.0, budget_limit=300.0))
        assert result.passed is True
        assert "OVER_BUDGET" not in result.flags

    @pytest.mark.asyncio
    async def test_price_exactly_at_budget_passes(self):
        result = await verify(make_intent(price=300.0, budget_limit=300.0))
        assert "OVER_BUDGET" not in result.flags

    @pytest.mark.asyncio
    async def test_price_over_budget_blocks(self):
        result = await verify(make_intent(price=500.0, budget_limit=300.0))
        assert result.score == 0.0
        assert "OVER_BUDGET" in result.flags
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_near_budget_warns(self):
        # 97% of budget — should trigger NEAR_BUDGET_LIMIT
        result = await verify(make_intent(price=291.0, budget_limit=300.0))
        assert "NEAR_BUDGET_LIMIT" in result.flags

    @pytest.mark.asyncio
    async def test_80_percent_budget_warns(self):
        # 83% of budget — should be flagged but not blocked
        result = await verify(make_intent(price=250.0, budget_limit=300.0))
        assert result.score < 100
        assert result.passed is True


# ---------------------------------------------------------------------------
# Session budget checks
# ---------------------------------------------------------------------------


class TestSessionBudget:
    @pytest.mark.asyncio
    async def test_within_session_budget_passes(self):
        result = await verify(
            make_intent(price=100.0, session_spend=50.0, session_budget=500.0)
        )
        assert result.passed is True
        assert "OVER_SESSION_BUDGET" not in result.flags

    @pytest.mark.asyncio
    async def test_session_budget_exceeded_blocks(self):
        # 50 already spent + 500 new = 550 > 500 limit
        result = await verify(
            make_intent(price=500.0, session_spend=50.0, session_budget=500.0)
        )
        assert "OVER_SESSION_BUDGET" in result.flags
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_near_session_budget_warns(self):
        # 450 + 100 = 550/500 = 110% — blocked
        # 450 + 40 = 490/500 = 98% — near limit warn
        result = await verify(
            make_intent(price=40.0, session_spend=450.0, session_budget=500.0)
        )
        assert "NEAR_SESSION_BUDGET_LIMIT" in result.flags


# ---------------------------------------------------------------------------
# Blocked sellers
# ---------------------------------------------------------------------------


class TestBlockedSellers:
    @pytest.mark.asyncio
    async def test_blocked_seller_returns_zero(self):
        result = await verify(
            make_intent(seller="scam-store.com", blocked_sellers=["scam-store.com"])
        )
        assert result.score == 0.0
        assert "SELLER_BLOCKED" in result.flags
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_blocked_seller_partial_match(self):
        # "scam-store" should match "scam-store.com"
        result = await verify(
            make_intent(seller="scam-store", blocked_sellers=["scam-store.com"])
        )
        assert "SELLER_BLOCKED" in result.flags

    @pytest.mark.asyncio
    async def test_non_blocked_seller_passes(self):
        result = await verify(
            make_intent(seller="amazon.com", blocked_sellers=["scam-store.com"])
        )
        assert "SELLER_BLOCKED" not in result.flags


# ---------------------------------------------------------------------------
# Seller allowlist
# ---------------------------------------------------------------------------


class TestSellerAllowlist:
    @pytest.mark.asyncio
    async def test_seller_on_allowlist_passes(self):
        result = await verify(
            make_intent(seller="amazon.com", allowed_sellers=["amazon.com", "bestbuy.com"])
        )
        assert "UNAUTHORIZED_SELLER" not in result.flags

    @pytest.mark.asyncio
    async def test_seller_not_on_allowlist_flagged(self):
        result = await verify(
            make_intent(seller="unknown-store.com", allowed_sellers=["amazon.com"])
        )
        assert "UNAUTHORIZED_SELLER" in result.flags

    @pytest.mark.asyncio
    async def test_empty_allowlist_no_restriction(self):
        result = await verify(make_intent(seller="any-store.com", allowed_sellers=[]))
        assert "UNAUTHORIZED_SELLER" not in result.flags

    @pytest.mark.asyncio
    async def test_blocked_takes_priority_over_allowlist(self):
        # Even if seller is on allowlist, block takes priority
        result = await verify(
            make_intent(
                seller="amazon.com",
                allowed_sellers=["amazon.com"],
                blocked_sellers=["amazon.com"],
            )
        )
        assert "SELLER_BLOCKED" in result.flags
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# Combined checks
# ---------------------------------------------------------------------------


class TestCombinedChecks:
    @pytest.mark.asyncio
    async def test_all_checks_pass_high_score(self):
        result = await verify(
            make_intent(
                price=100.0,
                budget_limit=200.0,
                session_spend=50.0,
                session_budget=500.0,
                seller="amazon.com",
                allowed_sellers=["amazon.com"],
            )
        )
        assert result.score == 100.0
        assert result.passed is True
        assert result.flags == []

    @pytest.mark.asyncio
    async def test_module_name_is_authorization(self):
        result = await verify(make_intent())
        assert result.module == "authorization"
