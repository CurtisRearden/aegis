"""
CrewAI integration for Aegis.

Provides a CrewAI Tool wrapper that agents can use to verify purchase intents
before executing transactions.

Usage::

    from aegis.integrations.crewai import AegisTool
    from crewai import Agent

    agent = Agent(
        role="Purchasing Agent",
        goal="Buy the best products for users",
        tools=[AegisTool()],
    )

    # Or with a custom policy:
    from aegis.policy import PolicyConfig
    tool = AegisTool(policy=PolicyConfig(block_threshold=50))
    agent = Agent(..., tools=[tool])

The tool accepts a JSON string describing the purchase and returns a
structured string that CrewAI agents can reason about.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from ..core import verify_purchase
from ..models import VerificationResult

if TYPE_CHECKING:
    from ..policy import PolicyConfig

logger = logging.getLogger(__name__)

_TOOL_DESCRIPTION = """\
Verify a proposed purchase before executing it. Always call this tool before
making any purchase on behalf of a user.

Input must be a JSON string with the following fields:
  Required:
    - item (str): Name of the item or service to purchase.
    - price (float): Proposed price.
    - seller (str): Seller domain or name (e.g. 'amazon.com').
    - original_instruction (str): The user's original purchase instruction.
  Optional:
    - market_price (float): Known market/reference price for comparison.
    - price_range (list[float, float]): Acceptable [min, max] price range.
    - budget_limit (float): Maximum allowed spend for this transaction.
    - session_spend (float): Amount already spent in this session.
    - session_budget (float): Maximum cumulative session spend.
    - allowed_sellers (list[str]): Approved seller identifiers.
    - blocked_sellers (list[str]): Blocked seller identifiers.
    - terms (dict): Purchase terms (has_refund_policy, auto_renewal, etc.).
    - category (str): Product category.
    - url (str): Product URL.

Returns a JSON string with:
  - decision: "approve" | "flag" | "block"
  - overall_score: 0-100
  - reasons: list of explanations
  - flags: list of machine-readable flag codes
  - scores: per-module scores dict
"""


def _format_result(result: VerificationResult) -> str:
    """Serialize a VerificationResult to a compact JSON string for the agent."""
    return json.dumps(
        {
            "decision": result.decision.value,
            "overall_score": result.overall_score,
            "reasons": result.reasons,
            "flags": result.flags,
            "scores": result.scores,
            "summary": result.summary(),
        },
        indent=2,
    )


def _run_verify_sync(intent_data: dict[str, Any], policy: PolicyConfig | None) -> str:
    """Run verify_purchase synchronously for use in CrewAI's sync tool runner."""
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run, verify_purchase(intent_data, policy=policy)
                    )
                    result = future.result(timeout=60)
            else:
                result = loop.run_until_complete(
                    verify_purchase(intent_data, policy=policy)
                )
        except RuntimeError:
            result = asyncio.run(verify_purchase(intent_data, policy=policy))
    except Exception as exc:
        logger.exception("AegisTool verification failed: %s", exc)
        return json.dumps({"error": str(exc), "decision": "flag"})

    return _format_result(result)


# ---------------------------------------------------------------------------
# CrewAI Tool class
# ---------------------------------------------------------------------------

try:
    from crewai.tools import BaseTool as CrewAIBaseTool
    from pydantic import BaseModel, Field as PydanticField

    class _AegisInput(BaseModel):
        """Input schema for AegisTool."""

        purchase_json: str = PydanticField(
            ...,
            description=(
                "JSON string describing the purchase intent. Must include: "
                "item, price, seller, original_instruction."
            ),
        )

    class AegisTool(CrewAIBaseTool):
        """
        CrewAI Tool that verifies a proposed purchase intent via Aegis.

        Integrates directly into a CrewAI agent's tool list. The agent passes
        purchase details as a JSON string; the tool returns a structured verdict
        the agent can use to decide whether to proceed.

        Attributes:
            policy: Optional PolicyConfig to customize decision thresholds.
        """

        name: str = "aegis_verify_purchase"
        description: str = _TOOL_DESCRIPTION
        args_schema: type[_AegisInput] = _AegisInput
        policy: Any = PydanticField(default=None, exclude=True)

        def __init__(self, policy: PolicyConfig | None = None, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            object.__setattr__(self, "policy", policy)

        def _run(self, purchase_json: str) -> str:
            """
            Execute purchase verification synchronously.

            Args:
                purchase_json: JSON string describing the purchase intent.

            Returns:
                JSON string with decision, score, reasons, and flags.
            """
            try:
                intent_data = json.loads(purchase_json)
            except json.JSONDecodeError as exc:
                return json.dumps(
                    {
                        "error": f"Invalid JSON input: {exc}",
                        "decision": "block",
                        "reasons": ["Purchase intent could not be parsed."],
                    }
                )
            return _run_verify_sync(intent_data, self.policy)

    # Alias for README compatibility
    CrewAIAuditTool = AegisTool

except ImportError:
    # CrewAI is not installed — stubs so import doesn't fail at module load time.

    class AegisTool:  # type: ignore[no-redef]
        """Stub: install crewai to use AegisTool."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "CrewAI is not installed. "
                "Install it with: pip install aegis-verify[crewai]"
            )

    CrewAIAuditTool = AegisTool  # type: ignore[misc]
