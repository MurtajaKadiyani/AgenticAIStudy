from openai import OpenAI
from dotenv import load_dotenv
import os
import asyncio

from mcp_client import (
    connect,
    disconnect,
    discover_tools
)

from planner import planner
from executor import execute_action
from loop import run_agent_loop
from answer import format_answer


async def main():
    # Step 1: Connect to the MCP Server.
    client = await connect()

     # Step 2: Discover all available tools.
    tools = await discover_tools(client)
    
    print()
    
    print("Available Tools")
    
    print("----------------")
    
    for tool in tools:
            
        print(tool.name)
    
    print()

    ### Step 3: Planner: Take the User Input (Question). Create Tool Descriptions for each tool. Then create a prompt by combining user input and Tool Description and Give this prompt to Ollama. ###

    user_request = input(
        "\nUser : "
    )

    state = await run_agent_loop(
        user_request,
        planner,
        execute_action,
        format_answer,
        tools,
        client
    )

    print("Final Answer:")
    print("------------")
    print(state["final_answer"])

    # Disconnect to the MCP Server.
    await disconnect(client)

if __name__ == "__main__":
    asyncio.run(main())