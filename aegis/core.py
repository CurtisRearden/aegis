"""
Aegis core verification engine.

Provides the primary entry point: verify_purchase().

The function accepts a PurchaseIntent (or a plain dict that will be coerced
into one), fans out to all five verification modules concurrently using
asyncio.gather(), and feeds the results into the PolicyEngine to produce a
final VerificationResult.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .models import ModuleResult, PurchaseIntent, VerificationResult
from .policy import PolicyConfig, PolicyEngine
from .modules import authorization, intent, price, seller, terms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal module registry — add or remove modules here
# ---------------------------------------------------------------------------

_MODULES = {
    "price": price.verify,
    "intent": intent.verify,
    "authorization": authorization.verify,
    "seller": seller.verify,
    "terms": terms.verify,
}


async def verify_purchase(
    intent_data: PurchaseIntent | dict[str, Any],
    *,
    policy: PolicyConfig | None = None,
    modules: list[str] | None = None,
    timeout: float = 30.0,
) -> VerificationResult:
    """
    Verify a proposed purchase and return an approve / flag / block decision.

    This is the primary public API for Aegis. Call this function from your AI
    agent before executing any purchase.

    Args:
        intent_data:
            Either a PurchaseIntent object or a plain dict with the same fields.
            Minimum required keys: ``item``, ``price``, ``seller``,
            ``original_instruction``.

        policy:
            Optional PolicyConfig to customize weights and decision thresholds.
            If omitted, the default policy is used (see aegis/policy.py).

        modules:
            Optional list of module names to run. Defaults to all five:
            ``['price', 'intent', 'authorization', 'seller', 'terms']``.
            Pass a subset to run a lightweight check.

        timeout:
            Maximum seconds to wait for all module coroutines to complete.
            Modules that do not finish in time are replaced with a neutral
            50-score LOW-confidence result and a timeout flag.

    Returns:
        VerificationResult with decision, overall_score, per-module results,
        reasons, and flags.

    Example::

        result = await verify_purchase({
            "item": "Sony WH-1000XM5",
            "price": 278.00,
            "seller": "amazon.com",
            "original_instruction": "best noise-canceling headphones under $300",
            "market_price": 349.99,
            "budget_limit": 300.00,
        })

        if result.decision == "block":
            raise ValueError(f"Purchase blocked: {result.reasons}")
        elif result.decision == "flag":
            # Surface to user for approval
            ...
        else:
            await execute_purchase()
    """
    # Coerce dict to PurchaseIntent
    purchase_intent = _coerce_intent(intent_data)

    # Resolve module set
    active_modules = _resolve_modules(modules)

    logger.info(
        "Verifying purchase: item=%r seller=%r price=%.2f modules=%s",
        purchase_intent.item,
        purchase_intent.seller,
        purchase_intent.price,
        list(active_modules.keys()),
    )

    start = time.perf_counter()

    # Fan out to all modules concurrently
    module_results = await _run_modules(purchase_intent, active_modules, timeout)

    elapsed = time.perf_counter() - start
    logger.info("All modules completed in %.3fs", elapsed)

    # Aggregate through policy engine
    engine = PolicyEngine(policy)
    result = engine.evaluate(module_results, intent=purchase_intent)

    logger.info(
        "Verification complete: decision=%s overall_score=%.1f",
        result.decision.value,
        result.overall_score,
    )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_intent(data: PurchaseIntent | dict[str, Any]) -> PurchaseIntent:
    """Accept either a PurchaseIntent object or a raw dict."""
    if isinstance(data, PurchaseIntent):
        return data
    if isinstance(data, dict):
        return PurchaseIntent(**data)
    raise TypeError(
        f"intent_data must be a PurchaseIntent or dict, got {type(data).__name__}"
    )


def _resolve_modules(
    requested: list[str] | None,
) -> dict[str, Any]:
    """Return the subset of _MODULES matching the requested list."""
    if requested is None:
        return dict(_MODULES)

    unknown = set(requested) - set(_MODULES)
    if unknown:
        raise ValueError(f"Unknown module(s): {unknown}. Valid: {set(_MODULES)}")

    return {name: _MODULES[name] for name in requested}


async def _run_modules(
    purchase_intent: PurchaseIntent,
    active_modules: dict[str, Any],
    timeout: float,
) -> dict[str, ModuleResult]:
    """
    Run all modules concurrently and collect results.

    Modules that raise an exception or time out are replaced with a graceful
    fallback result so that a single bad module never crashes the pipeline.
    """
    coros = {
        name: _safe_run(name, fn, purchase_intent, timeout)
        for name, fn in active_modules.items()
    }

    # asyncio.gather preserves order of results
    names = list(coros.keys())
    results = await asyncio.gather(*coros.values())

    return dict(zip(names, results))


async def _safe_run(
    name: str,
    fn: Any,
    purchase_intent: PurchaseIntent,
    timeout: float,
) -> ModuleResult:
    """
    Run a single module with error and timeout handling.

    Returns a neutral LOW-confidence result on failure instead of propagating
    exceptions, so the pipeline can still produce a useful decision.
    """
    from .models import Confidence  # local import to avoid circular

    try:
        return await asyncio.wait_for(fn(purchase_intent), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("Module '%s' timed out after %.1fs", name, timeout)
        return ModuleResult(
            module=name,
            score=50.0,
            confidence=Confidence.LOW,
            passed=True,
            reasons=[f"Module '{name}' timed out and could not complete verification."],
            flags=["MODULE_TIMEOUT"],
            data={"timeout": timeout},
        )
    except Exception as exc:
        logger.exception("Module '%s' raised an unexpected error: %s", name, exc)
        return ModuleResult(
            module=name,
            score=50.0,
            confidence=Confidence.LOW,
            passed=True,
            reasons=[f"Module '{name}' encountered an error: {exc}"],
            flags=["MODULE_ERROR"],
            data={"error": str(exc)},
        )
