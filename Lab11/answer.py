import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)

def format_answer(state):

    # One fact per line: the model skims a Python list
    # and drops items from it.
    memories = "\n".join(
        f"- {memory}" for memory in state["memories"]
    ) or "None"

    prompt = f"""
The user asked:

{state["user_request"]}

Recent conversation (oldest first):
{state["history"]}

What you know about the user:
{memories}

Tasks:
{state["tasks"]}

Actions:
{state["actions"]}

Observations:
{state["observations"]}

Intermediate Results:
{state["results"]}

Give the user a natural and concise answer.
Use relevant previous memories when they help answer the user's question.
When the user asks about themselves (name, topics, interests),
include EVERY relevant memory; concise means short wording,
not leaving facts out.
If the user's message is a follow-up, answer it in the context
of the recent conversation.

Respond in plain text only. Do not use Markdown formatting
such as **bold**, _italics_, or bullet points.

Do not mention:
- planner
- agent loop
- MCP
- internal state
- tools
- internal reasoning
"""

    response = client.chat.completions.create(
        model=os.getenv("MODEL"),  # type: ignore
        reasoning_effort="low",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()   # type: ignore
