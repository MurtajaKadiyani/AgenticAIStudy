from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)


def build_tool_descriptions(tools):

    descriptions = ""

    for tool in tools:

        descriptions += f"""
Tool:
{tool.name}

Description:
{tool.description}

----------------------------
"""

    return descriptions

def planner(state, tools):
    # Tools the user declined are removed so the LLM cannot pick them again.
    rejected_tools = {
        obs["action"]
        for obs in state["observations"]
        if str(obs["observation"]).startswith("REJECTED")
    }

    allowed_tools = [
        tool for tool in tools
        if tool.name not in rejected_tools
    ]

    tool_description = build_tool_descriptions(allowed_tools)

    ###Then create a prompt by combining user input and Tool Description.  and Give this prompt to Ollama. ###

    completed_tools = sorted(set(state["actions"]) - rejected_tools)

    prompt = f"""
You are an AI planner

Your job is to decide ONLY the next action.

User Request:
{state["user_request"]}

Tools already completed (each has already returned a result, listed once each):
{completed_tools}

Tools REJECTED by the user (never choose these; that part of the request is handled):
{sorted(rejected_tools)}

Full action history (may contain duplicates from past steps):
{state["actions"]}

Previous Observations:
{state["observations"]}

Available Tools:
{tool_description}

Rules:

1. Choose only ONE next action.
2. A tool in "Tools already completed" has already produced its result. Do NOT choose it again
   unless the user's request explicitly asks for that tool to run multiple times
   (e.g. "roll two dice").
3. Use the observations to decide what is still required.
4. If an observation says a tool was REJECTED by the user, never choose that tool again.
   Treat that part of the request as handled and move on.
5. As soon as every distinct part of the user's request has a matching completed or
   rejected tool, return FINISH immediately. Do not call any more tools.
6. Return ONLY the tool name or FINISH, with no other words or punctuation.

Next Action:
"""

    # Give this prompt to LLM
    response = client.chat.completions.create(
           model = os.getenv("PLANNER_MODEL"), # type: ignore
           temperature = 0,
           messages = [
                 {
                   "role": "user",
                   "content": prompt
                 }
           ]
       )

    # LLM Returns Tool Name
    action = response.choices[0].message.content.strip() # type: ignore

    # Safety net: if the LLM still picks a rejected tool, stop instead.
    if action in rejected_tools:
        return "FINISH"

    return action
