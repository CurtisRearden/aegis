"""
Policy engine for Aegis purchase verification.

Aggregates scores from all verification modules and produces a final
approve/flag/block decision using configurable weights and thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Confidence, Decision, ModuleResult, VerificationResult, PurchaseIntent


# ---------------------------------------------------------------------------
# Default module weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "price": 0.20,
    "intent": 0.25,
    "authorization": 0.25,
    "seller": 0.20,
    "terms": 0.10,
}

# ---------------------------------------------------------------------------
# Confidence multipliers — low-confidence scores are down-weighted
# ---------------------------------------------------------------------------

CONFIDENCE_MULTIPLIER: dict[Confidence, float] = {
    Confidence.HIGH: 1.0,
    Confidence.MEDIUM: 0.85,
    Confidence.LOW: 0.65,
}

# ---------------------------------------------------------------------------
# Default decision thresholds
# ---------------------------------------------------------------------------

DEFAULT_APPROVE_THRESHOLD: float = 70.0
DEFAULT_BLOCK_THRESHOLD: float = 40.0

# Flags that always force a BLOCK regardless of score
BLOCK_FLAGS: frozenset[str] = frozenset(
    {
        "SELLER_BLOCKED",
        "OVER_BUDGET",
        "OVER_SESSION_BUDGET",
        "UNAUTHORIZED_SELLER",
    }
)

# Flags that always force at least a FLAG regardless of score
FLAG_FLAGS: frozenset[str] = frozenset(
    {
        "PRICE_TOO_HIGH",
        "INTENT_MISMATCH",
        "AUTO_RENEWAL_DETECTED",
        "NO_REFUND_POLICY",
        "SUSPICIOUS_SELLER",
        "NEAR_BUDGET_LIMIT",
    }
)


@dataclass
class PolicyConfig:
    """
    Configurable parameters for the policy engine.

    Attributes:
        weights: Module weight mapping. Weights are normalized to sum to 1.0
            so adding/removing modules does not break the scale.
        approve_threshold: Minimum weighted score to issue APPROVE.
        block_threshold: Maximum weighted score before issuing BLOCK.
            Scores between block_threshold and approve_threshold yield FLAG.
        block_flags: Flag codes that unconditionally issue BLOCK.
        flag_flags: Flag codes that unconditionally issue at least FLAG.
        include_intent_in_result: Whether to embed the original PurchaseIntent
            in the returned VerificationResult.
    """

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    approve_threshold: float = DEFAULT_APPROVE_THRESHOLD
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD
    block_flags: frozenset[str] = field(default_factory=lambda: frozenset(BLOCK_FLAGS))
    flag_flags: frozenset[str] = field(default_factory=lambda: frozenset(FLAG_FLAGS))
    include_intent_in_result: bool = True


class PolicyEngine:
    """
    Aggregates ModuleResults into a final VerificationResult.

    The engine:
    1. Computes a confidence-adjusted weighted score for each module.
    2. Detects hard block / flag triggers from flag codes.
    3. Applies threshold logic to derive approve / flag / block.
    4. Collects all human-readable reasons and flag codes.
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        module_results: dict[str, ModuleResult],
        intent: PurchaseIntent | None = None,
    ) -> VerificationResult:
        """
        Evaluate module results and return a VerificationResult.

        Args:
            module_results: Dict mapping module name → ModuleResult.
            intent: Original PurchaseIntent (embedded in result if config allows).

        Returns:
            VerificationResult with decision, scores, reasons, and flags.
        """
        overall_score = self._compute_weighted_score(module_results)
        all_flags = self._collect_flags(module_results)
        all_reasons = self._collect_reasons(module_results)

        decision = self._decide(overall_score, all_flags)

        return VerificationResult(
            decision=decision,
            overall_score=round(overall_score, 2),
            modules=module_results,
            reasons=all_reasons,
            flags=sorted(all_flags),
            intent=intent if self.config.include_intent_in_result else None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_weighted_score(
        self, module_results: dict[str, ModuleResult]
    ) -> float:
        """
        Compute a confidence-adjusted weighted average score.

        Only modules present in both the results dict and the weights config
        are included. Weights are re-normalized after filtering so partial
        module sets (e.g. only 3 of 5 modules) still produce a valid 0–100
        score.
        """
        weights = self.config.weights
        active = {
            name: result
            for name, result in module_results.items()
            if name in weights
        }

        if not active:
            # No scorable modules — return neutral score that triggers FLAG
            return 50.0

        # Compute effective weights (normalized)
        raw_weight_sum = sum(weights[name] for name in active)
        normalized: dict[str, float] = {
            name: weights[name] / raw_weight_sum for name in active
        }

        # Compute effective weight for each module (normalized weight × confidence
        # multiplier). Low-confidence modules contribute proportionally less.
        effective_weights = {
            name: normalized[name] * CONFIDENCE_MULTIPLIER[result.confidence]
            for name, result in active.items()
        }
        total_effective_weight = sum(effective_weights.values())

        if total_effective_weight == 0:
            return 50.0

        # Weighted average: scores are already on 0–100 scale, so the result
        # is directly in 0–100 range after dividing by the effective weight sum.
        weighted_score = sum(
            result.score * effective_weights[name]
            for name, result in active.items()
        ) / total_effective_weight

        return max(0.0, min(100.0, weighted_score))

    def _collect_flags(self, module_results: dict[str, ModuleResult]) -> set[str]:
        flags: set[str] = set()
        for result in module_results.values():
            flags.update(result.flags)
        return flags

    def _collect_reasons(self, module_results: dict[str, ModuleResult]) -> list[str]:
        """Collect reasons from modules that did not fully pass."""
        reasons: list[str] = []
        for result in module_results.values():
            if not result.passed or result.flags:
                reasons.extend(result.reasons)
        return reasons

    def _decide(self, overall_score: float, flags: set[str]) -> Decision:
        """
        Determine final decision from score and flags.

        Priority order:
        1. Hard block flags → BLOCK
        2. Score below block threshold → BLOCK
        3. Hard flag codes → FLAG
        4. Score below approve threshold → FLAG
        5. Otherwise → APPROVE
        """
        # Hard block flags
        if flags & self.config.block_flags:
            return Decision.BLOCK

        # Score-based block
        if overall_score < self.config.block_threshold:
            return Decision.BLOCK

        # Soft flag triggers
        if flags & self.config.flag_flags:
            return Decision.FLAG

        # Score-based flag
        if overall_score < self.config.approve_threshold:
            return Decision.FLAG

        return Decision.APPROVE


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

_default_engine = PolicyEngine()


def evaluate(
    module_results: dict[str, ModuleResult],
    intent: PurchaseIntent | None = None,
    config: PolicyConfig | None = None,
) -> VerificationResult:
    """
    Convenience function: evaluate module results with optional custom config.

    Args:
        module_results: Dict mapping module name → ModuleResult.
        intent: Original PurchaseIntent for embedding in the result.
        config: Optional PolicyConfig; uses defaults if not provided.

    Returns:
        VerificationResult.
    """
    engine = PolicyEngine(config) if config else _default_engine
    return engine.evaluate(module_results, intent=intent)
