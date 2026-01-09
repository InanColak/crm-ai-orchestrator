"""
Tools Module
============
LangChain tools for agent use.
"""

from backend.tools.hubspot_tools import (
    # Read tools
    GetContactTool,
    SearchContactsTool,
    GetDealTool,
    SearchDealsTool,
    GetOwnersTool,
    # Write tools (HITL)
    PrepareCreateContactTool,
    PrepareUpdateContactTool,
    PrepareCreateDealTool,
    PrepareUpdateDealTool,
    PrepareCreateTaskTool,
    PrepareCreateNoteTool,
    # Collections
    get_hubspot_read_tools,
    get_hubspot_write_tools,
    get_all_hubspot_tools,
)

__all__ = [
    # Read tools
    "GetContactTool",
    "SearchContactsTool",
    "GetDealTool",
    "SearchDealsTool",
    "GetOwnersTool",
    # Write tools (HITL)
    "PrepareCreateContactTool",
    "PrepareUpdateContactTool",
    "PrepareCreateDealTool",
    "PrepareUpdateDealTool",
    "PrepareCreateTaskTool",
    "PrepareCreateNoteTool",
    # Collections
    "get_hubspot_read_tools",
    "get_hubspot_write_tools",
    "get_all_hubspot_tools",
]
