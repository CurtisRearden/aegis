"""
Aegis framework integrations.

Import the tool wrapper for your framework:

    from aegis.integrations import LangChainAuditTool, CrewAIAuditTool

Or import directly from the sub-module:

    from aegis.integrations.langchain import AegisTool
    from aegis.integrations.crewai import AegisTool
"""

from .crewai import AegisTool as CrewAIAuditTool
from .langchain import AegisTool as LangChainAuditTool

__all__ = ["LangChainAuditTool", "CrewAIAuditTool"]
