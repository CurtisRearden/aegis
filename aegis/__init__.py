"""
Aegis — Independent verification layer for autonomous AI agent purchases.

Quick start::

    from aegis import verify_purchase

    result = await verify_purchase({
        "item": "Sony WH-1000XM5 Headphones",
        "price": 278.00,
        "seller": "amazon.com",
        "original_instruction": "best noise-canceling headphones under $300",
        "market_price": 349.99,
        "budget_limit": 300.00,
    })

    if result.decision == "approve":
        await execute_purchase()
    elif result.decision == "flag":
        print(f"Review needed: {result.reasons}")
    elif result.decision == "block":
        raise RuntimeError(f"Purchase blocked: {result.reasons}")
"""

from .core import verify_purchase
from .models import (
    Confidence,
    Decision,
    ModuleResult,
    PurchaseIntent,
    VerificationResult,
)
from .policy import PolicyConfig, PolicyEngine

__version__ = "0.1.0"

__all__ = [
    # Primary API
    "verify_purchase",
    # Models
    "PurchaseIntent",
    "VerificationResult",
    "ModuleResult",
    "Decision",
    "Confidence",
    # Policy
    "PolicyConfig",
    "PolicyEngine",
]
