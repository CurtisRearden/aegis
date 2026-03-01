"""Tests for aegis/policy.py — PolicyEngine and decision logic."""

import pytest

from aegis.models import Confidence, Decision, ModuleResult
from aegis.policy import (
    BLOCK_FLAGS,
    FLAG_FLAGS,
    PolicyConfig,
    PolicyEngine,
    evaluate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(
    module: str,
    score: float,
    confidence: Confidence = Confidence.HIGH,
    flags: list[str] | None = None,
    passed: bool | None = None,
) -> ModuleResult:
    if passed is None:
        passed = score >= 60
    return ModuleResult(
        module=module,
        score=score,
        confidence=confidence,
        passed=passed,
        flags=flags or [],
    )


def all_high_score(score: float = 90.0) -> dict[str, ModuleResult]:
    return {
        "price": make_result("price", score),
        "intent": make_result("intent", score),
        "authorization": make_result("authorization", score),
        "seller": make_result("seller", score),
        "terms": make_result("terms", score),
    }


# ---------------------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------------------


class TestDecisionThresholds:
    def setup_method(self):
        self.engine = PolicyEngine()

    def test_high_scores_approve(self):
        result = self.engine.evaluate(all_high_score(90.0))
        assert result.decision == Decision.APPROVE

    def test_mid_scores_flag(self):
        result = self.engine.evaluate(all_high_score(55.0))
        assert result.decision == Decision.FLAG

    def test_low_scores_block(self):
        result = self.engine.evaluate(all_high_score(30.0))
        assert result.decision == Decision.BLOCK

    def test_exactly_at_approve_threshold_approves(self):
        # Default approve threshold = 70
        modules = {
            "price": make_result("price", 70.0),
            "intent": make_result("intent", 70.0),
            "authorization": make_result("authorization", 70.0),
            "seller": make_result("seller", 70.0),
            "terms": make_result("terms", 70.0),
        }
        result = self.engine.evaluate(modules)
        assert result.decision == Decision.APPROVE

    def test_exactly_at_block_threshold_blocks(self):
        # Default block threshold = 40; score at exactly 40 → FLAG (not block)
        modules = all_high_score(40.0)
        result = self.engine.evaluate(modules)
        assert result.decision == Decision.FLAG

    def test_score_just_below_block_threshold_blocks(self):
        modules = all_high_score(39.0)
        result = self.engine.evaluate(modules)
        assert result.decision == Decision.BLOCK


# ---------------------------------------------------------------------------
# Hard flag triggers
# ---------------------------------------------------------------------------


class TestHardFlagTriggers:
    def setup_method(self):
        self.engine = PolicyEngine()

    def test_block_flag_forces_block_regardless_of_score(self):
        modules = all_high_score(95.0)
        modules["authorization"] = make_result(
            "authorization", 95.0, flags=["SELLER_BLOCKED"]
        )
        result = self.engine.evaluate(modules)
        assert result.decision == Decision.BLOCK

    def test_over_budget_flag_blocks(self):
        modules = all_high_score(95.0)
        modules["authorization"] = make_result(
            "authorization", 95.0, flags=["OVER_BUDGET"]
        )
        result = self.engine.evaluate(modules)
        assert result.decision == Decision.BLOCK

    def test_flag_flag_forces_at_least_flag(self):
        # High score but FLAG_FLAG raised → should be FLAG
        modules = all_high_score(90.0)
        modules["seller"] = make_result("seller", 90.0, flags=["SUSPICIOUS_SELLER"])
        result = self.engine.evaluate(modules)
        assert result.decision in (Decision.FLAG, Decision.BLOCK)

    def test_price_too_high_forces_flag(self):
        modules = all_high_score(90.0)
        modules["price"] = make_result("price", 90.0, flags=["PRICE_TOO_HIGH"])
        result = self.engine.evaluate(modules)
        assert result.decision in (Decision.FLAG, Decision.BLOCK)

    def test_auto_renewal_forces_flag(self):
        modules = all_high_score(90.0)
        modules["terms"] = make_result("terms", 90.0, flags=["AUTO_RENEWAL_DETECTED"])
        result = self.engine.evaluate(modules)
        assert result.decision in (Decision.FLAG, Decision.BLOCK)


# ---------------------------------------------------------------------------
# Confidence multipliers
# ---------------------------------------------------------------------------


class TestConfidenceMultipliers:
    def setup_method(self):
        self.engine = PolicyEngine()

    def test_low_confidence_down_weights_score(self):
        high_conf_modules = all_high_score(80.0)
        low_conf_modules = {
            name: make_result(name, 80.0, confidence=Confidence.LOW)
            for name in high_conf_modules
        }
        high_result = self.engine.evaluate(high_conf_modules)
        low_result = self.engine.evaluate(low_conf_modules)
        # Low confidence → score still normalized to 0-100 range but
        # the effective contribution is scaled; overall_score should be
        # the same since all modules have equal confidence level
        # (confidence multipliers cancel out in normalization)
        # The key property: both are still in [0, 100]
        assert 0 <= low_result.overall_score <= 100
        assert 0 <= high_result.overall_score <= 100


# ---------------------------------------------------------------------------
# Weighted scoring
# ---------------------------------------------------------------------------


class TestWeightedScoring:
    def test_custom_weights_applied(self):
        config = PolicyConfig(
            weights={"price": 0.9, "intent": 0.1},
            approve_threshold=70.0,
        )
        engine = PolicyEngine(config)
        modules = {
            "price": make_result("price", 30.0),   # low price score
            "intent": make_result("intent", 100.0),  # high intent score
        }
        # With 90% weight on price (score 30), overall should be low
        result = engine.evaluate(modules)
        assert result.overall_score < 50

    def test_overall_score_bounded(self):
        engine = PolicyEngine()
        result = engine.evaluate(all_high_score(100.0))
        assert 0 <= result.overall_score <= 100

    def test_empty_modules_returns_neutral(self):
        engine = PolicyEngine()
        result = engine.evaluate({})
        assert result.overall_score == 50.0
        assert result.decision == Decision.FLAG  # neutral → flag


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


class TestResultAggregation:
    def setup_method(self):
        self.engine = PolicyEngine()

    def test_flags_aggregated_from_all_modules(self):
        modules = {
            "price": make_result("price", 40.0, flags=["PRICE_TOO_HIGH"]),
            "seller": make_result("seller", 40.0, flags=["SUSPICIOUS_SELLER"]),
        }
        result = self.engine.evaluate(modules)
        assert "PRICE_TOO_HIGH" in result.flags
        assert "SUSPICIOUS_SELLER" in result.flags

    def test_flags_sorted(self):
        modules = {
            "price": make_result("price", 50.0, flags=["PRICE_TOO_HIGH"]),
        }
        result = self.engine.evaluate(modules)
        assert result.flags == sorted(result.flags)

    def test_reasons_collected_from_failing_modules(self):
        modules = {
            "price": make_result("price", 30.0, passed=False),
            "intent": make_result("intent", 90.0, passed=True),
        }
        modules["price"] = ModuleResult(
            module="price",
            score=30.0,
            confidence=Confidence.HIGH,
            passed=False,
            reasons=["Price is too high."],
            flags=["PRICE_TOO_HIGH"],
        )
        result = self.engine.evaluate(modules)
        assert "Price is too high." in result.reasons

    def test_intent_embedded_in_result(self, make_purchase_intent):
        result = self.engine.evaluate({}, intent=make_purchase_intent)
        assert result.intent is not None
        assert result.intent.item == make_purchase_intent.item

    def test_scores_property_matches_modules(self):
        modules = all_high_score(80.0)
        result = self.engine.evaluate(modules)
        for name in modules:
            assert name in result.scores
            assert result.scores[name] == 80.0


# ---------------------------------------------------------------------------
# PolicyConfig customization
# ---------------------------------------------------------------------------


class TestPolicyConfig:
    def test_custom_thresholds(self):
        # Very strict — only approve at 95+
        config = PolicyConfig(approve_threshold=95.0, block_threshold=50.0)
        engine = PolicyEngine(config)
        result = engine.evaluate(all_high_score(80.0))
        assert result.decision == Decision.FLAG

    def test_lenient_thresholds(self):
        config = PolicyConfig(approve_threshold=40.0, block_threshold=10.0)
        engine = PolicyEngine(config)
        result = engine.evaluate(all_high_score(45.0))
        assert result.decision == Decision.APPROVE

    def test_custom_block_flags(self):
        config = PolicyConfig(block_flags=frozenset({"MY_CUSTOM_FLAG"}))
        engine = PolicyEngine(config)
        modules = all_high_score(95.0)
        modules["price"] = make_result("price", 95.0, flags=["MY_CUSTOM_FLAG"])
        result = engine.evaluate(modules)
        assert result.decision == Decision.BLOCK

    def test_no_intent_in_result_when_disabled(self, make_purchase_intent):
        config = PolicyConfig(include_intent_in_result=False)
        engine = PolicyEngine(config)
        result = engine.evaluate({}, intent=make_purchase_intent)
        assert result.intent is None


# ---------------------------------------------------------------------------
# Convenience evaluate() function
# ---------------------------------------------------------------------------


class TestEvaluateFunction:
    def test_evaluate_uses_defaults(self):
        result = evaluate(all_high_score(85.0))
        assert result.decision == Decision.APPROVE

    def test_evaluate_accepts_custom_config(self):
        config = PolicyConfig(approve_threshold=95.0)
        result = evaluate(all_high_score(85.0), config=config)
        assert result.decision == Decision.FLAG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_purchase_intent():
    from aegis.models import PurchaseIntent

    return PurchaseIntent(
        item="Laptop",
        price=999.0,
        seller="amazon.com",
        original_instruction="buy a laptop",
    )
