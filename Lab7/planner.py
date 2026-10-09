from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)

def choose_tool(user_request):
    planner_prompt = f"""
You are an AI planner. Do not call any function. Reply with plain text only.

Here are the available action labels:

1. get_current_time
   Use when the user asks for the current date or time.

2. roll_dice
   Use when the user asks to roll a dice.

3. generate_password
   Use when the user wants a secure password.

If no action is required, reply with the word:

none

Reply with ONLY the matching label text (or none). Do not call a function.

User Request:
{user_request}
"""

    response = client.chat.completions.create(
                model = os.getenv("PLANNER_MODEL"), # type: ignore
                messages = [
                    {
                        "role": "system",
                        "content": "You are an AI planner. Reply with plain text only, never call a function/tool."
                    },
                    {
                        "role": "user",
                        "content": planner_prompt
                    }
                 ]
             )

    return response.choices[0].message.content.strip() # type: ignore
