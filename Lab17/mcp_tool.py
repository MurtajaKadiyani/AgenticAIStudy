import asyncio

from crewai.tools import tool
from mcp_client import (
    execute_tool,
    connect,
    disconnect
)

async def call_mcp_current_time():

    client = await connect()

    try:

        result = await execute_tool(client,"current_time")

        return result.content[0].text

    finally:
        await disconnect(client)

@tool("MCP Current Time")
def mcp_current_time(reason: str) -> str:
     """Get the current date and time from MCP. Pass a short reason for needing the time."""

     # reason is unused: Groq rejects the schema crewai generates for a tool with no parameters
     return asyncio.run(call_mcp_current_time())