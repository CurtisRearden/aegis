"""
Pydantic models for Aegis purchase verification.

These models define the data contracts between AI agents, verification modules,
and the policy engine.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Decision(str, Enum):
    """Final verification decision returned to the calling agent."""

    APPROVE = "approve"
    FLAG = "flag"
    BLOCK = "block"


class Confidence(str, Enum):
    """Confidence level of a verification module's assessment."""

    HIGH = "high"      # Module had sufficient data to make a reliable assessment
    MEDIUM = "medium"  # Module made a reasonable assessment with some uncertainty
    LOW = "low"        # Module lacked data; result is a best-effort estimate


class PurchaseIntent(BaseModel):
    """
    Structured representation of a proposed purchase, submitted by an AI agent
    for verification before execution.

    All fields that the agent populates should reflect the actual transaction
    details. Optional fields (budget_limit, allowed_sellers, etc.) come from
    user-defined policy and are injected by the integration layer.
    """

    # --- Core transaction fields ---
    item: str = Field(
        ...,
        description="Human-readable name of the item or service being purchased.",
        min_length=1,
        max_length=500,
    )
    price: float = Field(
        ...,
        description="Proposed purchase price in the specified currency.",
        gt=0,
    )
    seller: str = Field(
        ...,
        description=(
            "Seller identifier — domain name (e.g. 'amazon.com'), "
            "marketplace username, or business name."
        ),
        min_length=1,
        max_length=300,
    )
    original_instruction: str = Field(
        ...,
        description=(
            "The original natural-language instruction given by the user to the agent "
            "(e.g. 'buy the best noise-canceling headphones under $300'). "
            "Used for intent matching."
        ),
        min_length=1,
    )

    # --- Optional enrichment ---
    url: str | None = Field(
        default=None,
        description="Direct URL of the product or checkout page.",
    )
    category: str | None = Field(
        default=None,
        description=(
            "Product category (e.g. 'electronics', 'travel', 'software'). "
            "Helps modules apply category-specific heuristics."
        ),
    )
    currency: str = Field(
        default="USD",
        description="ISO 4217 currency code.",
        min_length=3,
        max_length=3,
    )
    quantity: int = Field(
        default=1,
        description="Number of units being purchased.",
        ge=1,
    )
    description: str | None = Field(
        default=None,
        description="Additional description of the item (seller-provided listing text).",
        max_length=2000,
    )

    # --- Policy / authorization context ---
    budget_limit: float | None = Field(
        default=None,
        description=(
            "Maximum spend allowed for this transaction, set by the user's policy. "
            "If None, no hard budget limit is enforced."
        ),
        gt=0,
    )
    session_spend: float = Field(
        default=0.0,
        description=(
            "Total amount already spent in the current agent session. "
            "Used by the authorization module to enforce cumulative limits."
        ),
        ge=0,
    )
    session_budget: float | None = Field(
        default=None,
        description="Maximum cumulative spend allowed across the current session.",
        gt=0,
    )
    allowed_sellers: list[str] = Field(
        default_factory=list,
        description=(
            "Allowlist of approved seller identifiers. "
            "If non-empty, purchases from unlisted sellers are flagged."
        ),
    )
    blocked_sellers: list[str] = Field(
        default_factory=list,
        description="Blocklist of sellers that must never be used.",
    )
    require_refund_policy: bool = Field(
        default=False,
        description="If True, the terms module will block purchases without a clear refund policy.",
    )
    allow_auto_renewal: bool = Field(
        default=False,
        description="If False, purchases with auto-renewal clauses are flagged.",
    )

    # --- Seller-provided terms (agent-extracted) ---
    terms: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Agent-extracted terms and conditions. Recognized keys: "
            "'has_refund_policy' (bool), 'refund_window_days' (int), "
            "'auto_renewal' (bool), 'hidden_fees' (list[str]), "
            "'subscription' (bool), 'cancellation_fee' (float)."
        ),
    )

    # --- Market price context (agent-researched) ---
    market_price: float | None = Field(
        default=None,
        description=(
            "Agent-researched market/reference price for this item. "
            "If provided, the price module uses this as its primary benchmark."
        ),
        gt=0,
    )
    price_range: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Known acceptable price range (min, max) for this item. "
            "Overrides market_price if both are supplied."
        ),
    )

    # --- Metadata ---
    user_id: str | None = Field(
        default=None,
        description="Identifier of the user on whose behalf the agent is acting.",
    )
    agent_id: str | None = Field(
        default=None,
        description="Identifier of the calling agent.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional context passed through to audit logs.",
    )

    @field_validator("currency")
    @classmethod
    def currency_uppercase(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def price_range_valid(self) -> PurchaseIntent:
        if self.price_range is not None:
            low, high = self.price_range
            if low >= high:
                raise ValueError("price_range[0] must be less than price_range[1]")
        return self


class ModuleResult(BaseModel):
    """
    Output of a single verification module.

    Each module runs independently and returns a score, confidence level,
    and human-readable reasons so the policy engine and calling agents can
    understand the basis for the decision.
    """

    module: str = Field(..., description="Module name (e.g. 'price', 'intent').")
    score: float = Field(
        ...,
        description=(
            "Verification score from 0 to 100. "
            "100 = completely safe/verified. 0 = strong red flag."
        ),
        ge=0,
        le=100,
    )
    confidence: Confidence = Field(
        ...,
        description="How confident the module is in its score.",
    )
    passed: bool = Field(
        ...,
        description="True if the module's score meets its own passing threshold.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable explanations for the score and any flags raised.",
    )
    flags: list[str] = Field(
        default_factory=list,
        description=(
            "Short machine-readable flag codes (e.g. 'PRICE_TOO_HIGH', "
            "'NO_REFUND_POLICY') for downstream filtering."
        ),
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw data used or produced by the module (for audit logging).",
    )


class VerificationResult(BaseModel):
    """
    Aggregated output of the full Aegis verification pipeline.

    This is the object returned to the calling agent by verify_purchase().
    The agent should inspect `decision` before proceeding with any transaction.
    """

    decision: Decision = Field(
        ...,
        description=(
            "Final verification decision: "
            "'approve' (safe to proceed), "
            "'flag' (proceed with caution / surface to user), "
            "'block' (do not execute this purchase)."
        ),
    )
    overall_score: float = Field(
        ...,
        description=(
            "Weighted aggregate score across all modules (0–100). "
            "Higher is safer."
        ),
        ge=0,
        le=100,
    )
    modules: dict[str, ModuleResult] = Field(
        ...,
        description="Per-module results keyed by module name.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description=(
            "Consolidated list of human-readable reasons surfaced from all modules "
            "that contributed to the decision."
        ),
    )
    flags: list[str] = Field(
        default_factory=list,
        description="All machine-readable flag codes raised across all modules.",
    )
    verified_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when verification was completed.",
    )
    intent: PurchaseIntent | None = Field(
        default=None,
        description="The original PurchaseIntent that was verified.",
    )

    @property
    def scores(self) -> dict[str, float]:
        """Convenience accessor: {module_name: score} for all modules."""
        return {name: result.score for name, result in self.modules.items()}

    def summary(self) -> str:
        """One-line human-readable summary of the verification result."""
        score_parts = ", ".join(
            f"{name}={result.score:.0f}" for name, result in self.modules.items()
        )
        return (
            f"[{self.decision.value.upper()}] overall={self.overall_score:.0f} "
            f"({score_parts})"
        )
