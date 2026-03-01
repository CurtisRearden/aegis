#!/usr/bin/env python3
"""
Aegis Demo — See AI purchase verification in action.

Run:
    pip install aegis-verify
    python demo.py

This demo simulates three real-world scenarios where an AI agent
attempts a purchase and Aegis intercepts it before money moves.
"""

import asyncio
from aegis import verify_purchase


# ─── Colors for terminal output ──────────────────────────────────────────────

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

DECISION_STYLE = {
    "approve": f"{GREEN}{BOLD}✅ APPROVED{RESET}",
    "flag": f"{YELLOW}{BOLD}⚠️  FLAGGED{RESET}",
    "block": f"{RED}{BOLD}🚫 BLOCKED{RESET}",
}


def print_header():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     █████╗ ███████╗ ██████╗ ██╗███████╗                      ║
║    ██╔══██╗██╔════╝██╔════╝ ██║██╔════╝                      ║
║    ███████║█████╗  ██║  ███╗██║███████╗                      ║
║    ██╔══██║██╔══╝  ██║   ██║██║╚════██║                      ║
║    ██║  ██║███████╗╚██████╔╝██║███████║                      ║
║    ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝╚══════╝                      ║
║                                                              ║
║    Independent Verification for AI Agent Purchases           ║
║    v0.1.0                                                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")


def print_scenario(number, title, description):
    print(f"\n{CYAN}{BOLD}{'━' * 62}{RESET}")
    print(f"{CYAN}{BOLD}  SCENARIO {number}: {title}{RESET}")
    print(f"{DIM}  {description}{RESET}")
    print(f"{CYAN}{BOLD}{'━' * 62}{RESET}\n")


def print_intent(intent):
    print(f"  {BOLD}Purchase Intent:{RESET}")
    print(f"    Item:         {intent['item']}")
    print(f"    Price:        ${intent['price']:.2f}")
    print(f"    Seller:       {intent['seller']}")
    print(f"    Instruction:  \"{intent['original_instruction']}\"")
    if "market_price" in intent:
        print(f"    Market Price: ${intent['market_price']:.2f}")
    if "budget_limit" in intent:
        print(f"    Budget Limit: ${intent['budget_limit']:.2f}")
    print()


def print_result(result):
    # Decision banner
    style = DECISION_STYLE.get(result.decision.value, result.decision.value)
    print(f"  {BOLD}Aegis Decision:{RESET}  {style}")
    print(f"  {BOLD}Overall Score:{RESET}   {result.overall_score:.0f}/100\n")

    # Module breakdown
    print(f"  {BOLD}Module Breakdown:{RESET}")
    for name, mod in result.modules.items():
        # Color the score
        if mod.score >= 70:
            color = GREEN
        elif mod.score >= 50:
            color = YELLOW
        else:
            color = RED

        icon = "✓" if mod.passed else "✗"
        print(
            f"    {icon} {name:<16} "
            f"{color}{mod.score:>5.0f}{RESET}/100  "
            f"{DIM}({mod.confidence.value} confidence){RESET}"
        )

    # Reasons
    if result.reasons:
        print(f"\n  {BOLD}Reasons:{RESET}")
        for reason in result.reasons:
            print(f"    • {reason}")

    # Flags
    if result.flags:
        print(f"\n  {BOLD}Flags:{RESET}  {', '.join(result.flags)}")

    print()


# ─── Scenarios ───────────────────────────────────────────────────────────────


async def scenario_1_approved():
    """A straightforward, well-priced purchase that should be approved."""

    print_scenario(
        1,
        "The Good Purchase",
        "Agent finds headphones matching the user's request, priced below market.",
    )

    intent = {
        "item": "Sony WH-1000XM5 Wireless Headphones",
        "price": 278.00,
        "seller": "amazon.com",
        "original_instruction": "Buy the best noise-canceling headphones under $300",
        "market_price": 349.99,
        "budget_limit": 300.00,
        "terms": {
            "has_refund_policy": True,
            "refund_window_days": 30,
            "auto_renewal": False,
            "hidden_fees": [],
        },
    }

    print_intent(intent)
    print(f"  {DIM}Running verification...{RESET}\n")

    result = await verify_purchase(intent)
    print_result(result)
    return result


async def scenario_2_flagged():
    """Agent finds the right item but from a questionable seller at a high price."""

    print_scenario(
        2,
        "The Sketchy Deal",
        "Agent finds a laptop but the price is above market and the seller is unknown.",
    )

    intent = {
        "item": "MacBook Pro 14-inch M3 Pro",
        "price": 2849.00,
        "seller": "super-electronics-deals-outlet.com",
        "original_instruction": "Get me a MacBook Pro for work, preferably the 14-inch M3 Pro",
        "market_price": 1999.00,
        "budget_limit": 3000.00,
        "terms": {
            "has_refund_policy": True,
            "refund_window_days": 7,
            "auto_renewal": False,
            "hidden_fees": ["restocking_fee"],
        },
    }

    print_intent(intent)
    print(f"  {DIM}Running verification...{RESET}\n")

    result = await verify_purchase(intent)
    print_result(result)
    return result


async def scenario_3_blocked():
    """Agent goes completely off-rails — wrong item, over budget, scam seller."""

    print_scenario(
        3,
        "The Rogue Agent",
        "Agent was asked for a book but tries to buy a $1,200 gadget from a scam site.",
    )

    intent = {
        "item": "Smart Home Robot Assistant Bundle",
        "price": 1199.99,
        "seller": "totally-not-a-scam-deals.xyz",
        "original_instruction": "Order the latest Python programming book",
        "market_price": 450.00,
        "budget_limit": 50.00,
        "terms": {
            "has_refund_policy": False,
            "refund_window_days": 0,
            "auto_renewal": True,
            "hidden_fees": ["monthly_service_fee", "activation_fee", "processing_fee"],
        },
    }

    print_intent(intent)
    print(f"  {DIM}Running verification...{RESET}\n")

    result = await verify_purchase(intent)
    print_result(result)
    return result


# ─── Main ────────────────────────────────────────────────────────────────────


async def main():
    print_header()

    results = []

    r1 = await scenario_1_approved()
    results.append(("The Good Purchase", r1))

    r2 = await scenario_2_flagged()
    results.append(("The Sketchy Deal", r2))

    r3 = await scenario_3_blocked()
    results.append(("The Rogue Agent", r3))

    # Summary
    print(f"\n{CYAN}{BOLD}{'═' * 62}{RESET}")
    print(f"{CYAN}{BOLD}  SUMMARY{RESET}")
    print(f"{CYAN}{BOLD}{'═' * 62}{RESET}\n")

    for title, r in results:
        style = DECISION_STYLE.get(r.decision.value, r.decision.value)
        print(f"  {title:<25} {style}  (score: {r.overall_score:.0f}/100)")

    print(f"\n{DIM}  Aegis verified 3 purchases in milliseconds —")
    print(f"  before any money moved.{RESET}\n")
    print(f"  {BOLD}Learn more:{RESET} https://github.com/CurtisRearden/aegis")
    print(f"  {BOLD}Install:{RESET}    pip install aegis-verify\n")


if __name__ == "__main__":
    asyncio.run(main())
