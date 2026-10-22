from openai import OpenAI
from dotenv import load_dotenv
import os
import asyncio

from mcp_manager import (
    create_tool_registry,
    close_servers
)

from planner import planner
from executor import execute_action
from loop import run_agent_loop

servers = [
    "time-mcp-server/server.py",
    # "weather-mcp-server/server.py"
    "http://127.0.0.1:8000/mcp"
    
]

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
    
    client, tool_registry = (
        await create_tool_registry(servers)
    )
    
    #client = await connect()
    
    #tools = await discover_tools(client)

    tools = [
        entry["tool"]
        for entry in tool_registry.values()
    ]
    
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
        tool_registry
    )

    print()

    print("Final Answer:")
    print(state["final_answer"])

    # Disconnect to the MCP Servers.
    await close_servers(client)

if __name__ == "__main__":
    asyncio.run(main())