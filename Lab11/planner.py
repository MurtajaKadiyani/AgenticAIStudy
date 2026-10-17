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

Parameters:
{(tool.input_schema or {}).get("properties") or "None"}

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

Relevant Previous Memories:
{state["memories"]}

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

6. Use previous observations, results and relevant memories.

7. If an action failed, reconsider the task.

8. If there are no pending tasks,
   return ONLY the word:

FINISH

9. If tasks are still pending, return exactly
   three lines in this format:

TASK: <exact pending task>
TOOL: <tool_name>
ARGS: <JSON object of tool parameters>

   Fill ARGS from the user's request using the tool's
   Parameters (e.g. ARGS: {{"length": 20}}).
   If nothing is specified, use ARGS: {{}}

10. Do not explain anything.

11. If no available tool fits the task, use:

TOOL: NONE

   Never pick an unrelated tool just to fill the TOOL line.

12. IF the request can be answered directly using conversational memory, standard knowledge, or if tasks list is empty/NONE, respond ONLY with: FINISH

Next Decision:
"""

    # Give this prompt to LLM
    response = client.chat.completions.create(
           model = os.getenv("PLANNER_MODEL"), # type: ignore
           temperature = 0,
           reasoning_effort = "low",
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