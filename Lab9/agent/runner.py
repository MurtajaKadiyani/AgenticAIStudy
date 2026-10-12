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

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)

def format_answer(state):
    prompt = f"""
    The user asked:
    
    User Request:
    {state["user_request"]}
    
    Actions performed:
    {state["actions"]}
    
    Observations:
    {state["observations"]}
    
    Answer the user naturally.

    Do not mention internal planning.
    Do not mention state.
    Do not mention tools.
    """

    response = client.chat.completions.create(
        model = os.getenv("MODEL"), # type: ignore
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip() # type: ignore

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

    print()

    print("Final Answer:")
    print(state["final_answer"])

    # Disconnect to the MCP Server.
    await disconnect(client)

if __name__ == "__main__":
    asyncio.run(main())