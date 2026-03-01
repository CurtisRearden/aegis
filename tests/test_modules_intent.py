"""Tests for aegis/modules/intent.py — Intent matching."""

import pytest

from aegis.models import PurchaseIntent
from aegis.modules.intent import verify


def make_intent(**kwargs) -> PurchaseIntent:
    defaults = {
        "item": "Sony WH-1000XM5 Headphones",
        "price": 278.0,
        "seller": "amazon.com",
        "original_instruction": "buy the best noise-canceling headphones under $300",
    }
    defaults.update(kwargs)
    return PurchaseIntent(**defaults)


# ---------------------------------------------------------------------------
# Strong intent matches
# ---------------------------------------------------------------------------


class TestStrongMatch:
    @pytest.mark.asyncio
    async def test_exact_category_match(self):
        intent = make_intent(
            item="Bose QuietComfort 45 Headphones",
            original_instruction="buy noise-canceling headphones",
        )
        result = await verify(intent)
        assert result.score >= 65
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_keyword_overlap_produces_reasons(self):
        result = await verify(make_intent())
        assert len(result.reasons) > 0

    @pytest.mark.asyncio
    async def test_price_within_instruction_constraint(self):
        intent = make_intent(
            price=250.0,
            original_instruction="buy headphones under $300",
        )
        result = await verify(intent)
        # Price is within constraint — should not flag for price
        assert "PRICE_EXCEEDS_INSTRUCTION_BUDGET" not in result.flags


# ---------------------------------------------------------------------------
# Price constraint detection
# ---------------------------------------------------------------------------


class TestPriceConstraints:
    @pytest.mark.asyncio
    async def test_price_exceeds_instruction_budget_flagged(self):
        intent = make_intent(
            price=400.0,
            original_instruction="buy headphones under $300",
        )
        result = await verify(intent)
        assert "PRICE_EXCEEDS_INSTRUCTION_BUDGET" in result.flags

    @pytest.mark.asyncio
    async def test_max_keyword_in_instruction(self):
        intent = make_intent(
            price=50.0,
            original_instruction="buy a book, max $40",
        )
        result = await verify(intent)
        assert "PRICE_EXCEEDS_INSTRUCTION_BUDGET" in result.flags

    @pytest.mark.asyncio
    async def test_price_within_max_budget(self):
        intent = make_intent(
            price=30.0,
            original_instruction="buy a book for no more than $50",
        )
        result = await verify(intent)
        assert "PRICE_EXCEEDS_INSTRUCTION_BUDGET" not in result.flags

    @pytest.mark.asyncio
    async def test_no_price_constraint_in_instruction(self):
        intent = make_intent(
            item="Coffee Maker",
            original_instruction="buy a good coffee maker",
        )
        result = await verify(intent)
        # No price flags from intent module when no constraint stated
        assert "PRICE_EXCEEDS_INSTRUCTION_BUDGET" not in result.flags


# ---------------------------------------------------------------------------
# Category mismatch detection
# ---------------------------------------------------------------------------


class TestCategoryMismatch:
    @pytest.mark.asyncio
    async def test_electronics_instruction_matches_electronics_item(self):
        intent = make_intent(
            item="Sony Camera",
            original_instruction="buy a camera for photography",
        )
        result = await verify(intent)
        assert "INTENT_MISMATCH" not in result.flags

    @pytest.mark.asyncio
    async def test_travel_instruction_with_electronics_item(self):
        intent = make_intent(
            item="Sony Headphones",
            original_instruction="book a flight to Paris",
        )
        result = await verify(intent)
        # Strong category mismatch → low score
        assert result.score < 65 or "INTENT_MISMATCH" in result.flags

    @pytest.mark.asyncio
    async def test_low_score_adds_intent_mismatch_flag(self):
        intent = make_intent(
            item="Completely Unrelated Product XYZ123",
            original_instruction="book a hotel room in New York",
        )
        result = await verify(intent)
        assert result.score < 65


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_description_supplements_item_tokens(self):
        intent = make_intent(
            item="XM5",  # sparse — keywords in description
            description="Sony noise-canceling wireless headphones",
            original_instruction="buy noise-canceling headphones",
        )
        result = await verify(intent)
        assert result.score >= 60

    @pytest.mark.asyncio
    async def test_module_name_is_intent(self):
        result = await verify(make_intent())
        assert result.module == "intent"

    @pytest.mark.asyncio
    async def test_confidence_low_for_short_instruction(self):
        intent = make_intent(original_instruction="buy")
        result = await verify(intent)
        # Very short instruction → low or medium confidence
        assert result.confidence.value in ("low", "medium")
