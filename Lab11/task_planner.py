import os

from dotenv import load_dotenv
from openai import OpenAI

from planner import build_tool_descriptions


load_dotenv()


client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)

def decompose_task(user_request, tools=None, history=None):

    tool_descriptions = (
        build_tool_descriptions(tools) if tools else "None"
    )

    prompt = f"""
    You are an AI task planner.

    Break the user's request into smaller tasks.

    Available Tools:
    {tool_descriptions}

    Rules:
    1. Create only necessary tasks.
    2. Put tasks in execution order.
    3. Each task should be a short action.
    4. Return one task per line.
    5. Do not number the tasks.
    6. Do not explain anything.
    7. Only create tasks that one of the available tools can perform.
       Anything that can be answered with knowledge or memory
       (explanations, notes, advice, study help) is NOT a task.

    CRITICAL RULE:

    Return NO tasks (or "NONE") if the user request belongs to any of the following categories:
    1. Greetings or casual conversation (e.g., "hi", "hello", "how are you?")
    2. Introductions or personal details (e.g., "my name is Murtaja", "I live in Pune")
    3. Memory questions about past interactions (e.g., "what is my name?", "what did we discuss earlier?")
    4. General knowledge/Q&A questions that can be answered directly without tools (e.g., "what is Machine Learning?", "explain gravity")
    5. Statements about what the user is learning or studying (e.g., "I am learning Newton's Law")
    6. Follow-up questions about the previous answer (e.g., "only 2?", "why?", "tell me more").
       Read them in the context of the recent conversation, not as new commands.

    Recent Conversation:
    {history or "None"}

    User Request:
    {user_request}

    """

    response = client.chat.completions.create(
            model=os.getenv("MODEL"), # type: ignore
            reasoning_effort="low",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI task planner."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    
    result = response.choices[0].message.content.strip() # type: ignore

    tasks = []

    for line in result.splitlines():

        line  = line.strip()

        if line:
            tasks.append(line)

    return tasks