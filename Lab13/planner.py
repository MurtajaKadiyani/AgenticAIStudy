from openai import OpenAI
from dotenv import load_dotenv
import os
import json

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

Input Schema:
{tool.input_schema}

----------------------------
"""

    return descriptions

def planner(state, tools):
    tool_descriptions = (
            build_tool_descriptions(tools)
        )

    prompt = f"""
    You are an AI Planner.
    
    Your job is to decide ONLY the
    next action.
    
    User Request:
    
    {state["user_request"]}
    
    Completed Actions:
    
    {state["actions"]}
    
    Previous Observations:
    
    {state["observations"]}
    
    Available Tools:
    
    {tool_descriptions}
    
    Rules:
    
    1. Choose only ONE next action.
    
    2. Never repeat an action that has
       already been completed.
    
    3. Use the observations to decide
       what is still required.
    
    4. If the user's request has been
       completely satisfied, return FINISH.
    
    5. Return ONLY valid JSON.
    
    6. Do not explain your answer.
    
    Use exactly this format:
    
    {{
        "tool": "tool_name",
        "arguments": {{
            "argument_name": "argument_value"
        }}
    }}
    
    For FINISH, return:
    
    {{
        "tool": "FINISH",
        "arguments": {{}}
    }}
    
    Next Action:
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
           
    result = (
        response
        .choices[0]
        .message
        .content
        .strip() # type: ignore
    )

    return json.loads(result)
           