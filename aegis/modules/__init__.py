"""
Aegis verification modules.

Each module exposes a single async ``verify(intent: PurchaseIntent) -> ModuleResult``
coroutine. Modules are designed to run concurrently via asyncio.gather().
"""

from . import authorization, intent, price, seller, terms

__all__ = ["price", "intent", "authorization", "seller", "terms"]
