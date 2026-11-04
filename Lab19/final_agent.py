import asyncio

from langchain.agents import create_agent
from langchain_core.tools import tool   
from model import model
from mcp_tools import client
from retriever import retriever

@tool
def search_knowledge(question: str) -> str:
    """Search the user's notes (local knowledge base) and return relevant passages.
    Use this for any question about the user's notes or documents."""

    documents = retriever.invoke(question)

    return "\n\n".join(
        document.page_content
        for document in documents
    )

async def main():
    mcp_tools = await client.get_tools()

    tools = mcp_tools + [
        search_knowledge
    ]

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant. When the user asks about their notes "
            "or documents, always call the search_knowledge tool first and answer "
            "only from what it returns."
        )
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is MCP and what is current time?"
                }
            ]
        }
    )

    print("\nFINAL ANSWER")
    print(result["messages"][-1].content)

if __name__ == "__main__": 
    asyncio.run(main())