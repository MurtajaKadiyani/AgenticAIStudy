import asyncio
import os

from mcp_client import (
    connect,
    disconnect,
    discover_tools
)

from planner import planner
from executor import execute_action
from loop import run_agent_loop
from answer import format_answer
from memory import (
    create_database,
    create_embedding,
    get_or_create_user,
    create_thread,
    list_threads,
    set_thread_title,
    get_fact_history,
    undo_replace
)

# The console has a single user. A web UI would take this
# from the logged-in account instead.
USER_NAME = os.getenv("AGENT_USER", "default")

HELP = (
    "Commands: /new (new chat), /threads (list chats), /open <id> (switch chat),\n"
    "          /facts (what I remember), /undo <fact id> (undo a fact replacement)"
)


def show_facts(user_id):

    history = get_fact_history(user_id)

    print("\nWhat I remember")
    print("---------------")

    for item in history:
        if item["superseded_by"] is None:
            print(f"#{item['id']}  {item['fact']}")

    replaced = [item for item in history if item["superseded_by"] is not None]

    if replaced:

        print("\nReplaced facts (undo with /undo <new id>)")
        print("-----------------------------------------")

        for item in replaced:
            print(f"#{item['id']}  {item['fact']}  ->  replaced by #{item['superseded_by']}")


def handle_command(command, user_id, thread_id):

    # Returns the thread to continue in.
    parts = command.split()

    if parts[0] == "/new":

        thread_id = create_thread(user_id)

        print(f"Started new chat #{thread_id}.")

    elif parts[0] == "/threads":

        for thread in list_threads(user_id):

            marker = "*" if thread["id"] == thread_id else " "

            print(
                f"{marker} #{thread['id']}  "
                f"{thread['title'] or '(empty)'}  "
                f"{thread['created_at']}"
            )

    elif parts[0] == "/open" and len(parts) == 2 and parts[1].isdigit():

        ids = {thread["id"] for thread in list_threads(user_id)}

        if int(parts[1]) in ids:

            thread_id = int(parts[1])

            print(f"Switched to chat #{thread_id}.")

        else:

            print("No such chat.")

    elif parts[0] == "/facts":

        show_facts(user_id)

    elif parts[0] == "/undo" and len(parts) == 2 and parts[1].isdigit():

        restored = undo_replace(user_id, int(parts[1]))

        if restored:
            print(f"Removed fact #{parts[1]} and restored {restored} older fact(s).")
        else:
            print(f"Fact #{parts[1]} didn't replace anything (or was replaced itself; undo the newer one first).")

    else:

        print(HELP)

    return thread_id


async def main():
    create_database()

    user_id = get_or_create_user(USER_NAME)

    # Each session starts a new chat, like opening a
    # new conversation in ChatGPT. Facts still carry over.
    thread_id = create_thread(user_id)
    # Step 1: Connect to the MCP Server.
    # Warm up the Ollama embedding model at the same time, so the
    # first question doesn't wait for the model to load.
    client, _ = await asyncio.gather(
        connect(),
        asyncio.to_thread(create_embedding, "warmup")
    )

     # Step 2: Discover all available tools.
    tools = await discover_tools(client)
    
    print()
    
    print("Available Tools")
    
    print("----------------")
    
    for tool in tools:
            
        print(tool.name)

    print()

    print(HELP)

    ### Step 3: Planner: Take the User Input (Question). Create Tool Descriptions for each tool. Then create a prompt by combining user input and Tool Description and Give this prompt to Ollama. ###

    # Keep the session open so startup cost is paid only once.
    while True:

        user_request = input(
            f"\n[chat #{thread_id}] User (type 'exit' to quit) : "
        ).strip()

        if user_request.lower() in ("exit", "quit"):
            break

        if not user_request:
            continue

        if user_request.startswith("/"):
            thread_id = handle_command(user_request, user_id, thread_id)
            continue

        # Name the chat after its first message.
        set_thread_title(thread_id, user_request[:50])

        # A failed LLM / embedding / network call should
        # end this turn only, not the whole session.
        try:
            state = await run_agent_loop(
                user_request,
                planner,
                execute_action,
                format_answer,
                tools,
                client,
                user_id,
                thread_id
            )
        except Exception as error:
            print(f"\nSomething went wrong: {error}")
            print("Please try again.")
            continue

        print("Final Answer:")
        print("------------")
        print(state["final_answer"])

    # Disconnect to the MCP Server.
    await disconnect(client)

if __name__ == "__main__":
    asyncio.run(main())