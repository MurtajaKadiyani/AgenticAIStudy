import asyncio
from model import model
from prompt import prompt
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from tools import (generate_password, roll_dice, current_time)
from mcp_tools import client

# parser = StrOutputParser()

# chain = (prompt | model | parser)

async def main():

    tools =  await client.get_tools()

    agent = create_agent(
    model=model,
    tools=tools
    )


    result =  await agent.ainvoke(
        {
            "messages" : [
                {
                    "role": "user",
                    "content": "What is the current time?"
                }
            ]
        }
    )
    
    print("\nFINAL ANSWER")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())