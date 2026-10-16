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
    tool_descriptions = build_tool_descriptions(tools)

    prompt = f"""
You are an AI planner

Your job is to select the next unfinished task
and the best tool to complete it.

User Request:
{state["user_request"]}

Tasks:
{state["tasks"]}

Pending Tasks:

{[
    item["task"]
    for item in state["tasks"]
    if item["status"] == "pending"
]}

Completed Actions:

{state["actions"]}

Observations:

{state["observations"]}

Intermediate Results:

{state["results"]}

Available Tools:

{tool_descriptions}

Rules:

1. Select only ONE pending task.

2. The TASK must be copied EXACTLY from the
   pending task list.

3. Do not change the task wording.

4. Do not select a completed task.

5. Select the best available tool for that task.

6. Use previous observations and results.

7. If an action failed, reconsider the task.

8. If there are no pending tasks,
   return ONLY the word:

FINISH

9. If tasks are still pending, return exactly
   two lines in this format:

TASK: <exact pending task>
TOOL: <tool_name>

10. Do not explain anything.

Next Decision:
"""

    # Give this prompt to LLM
    response = client.chat.completions.create(
           model = os.getenv("PLANNER_MODEL"), # type: ignore
           temperature = 0,
           messages=[
                       {
                           "role": "system",
                           "content": "You are an AI planner."
                       },
                       {
                           "role": "user",
                           "content": prompt
                       }
                   ]
               )
           
    return response.choices[0].message.content.strip() # type: ignore