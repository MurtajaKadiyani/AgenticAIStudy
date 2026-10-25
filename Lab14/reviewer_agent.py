import json

from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()


client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)

def review(task,code):

    prompt = f"""
    You are a Reviewer Agent.

    Review the following solution.

    Task:
    {task}

    Solution:
    {code}

    Check:

    1. Correctness
    2. Python syntax
    3. Logic
    4. Missing issues
    5. Possible improvements

    Return ONLY valid JSON.

    Use exactly this format:

    {{
        "status": "approved",
        "feedback": "The solution is correct."
    }}

    OR:

    {{
        "status": "needs_revision",
        "feedback": "Explain what needs to be changed."
    }}

    Use "approved" only when the solution
    is acceptable.

    Use "needs_revision" when the solution
    requires a correction.

    """

    response = client.chat.completions.create(
            model=os.getenv("MODEL"), # type: ignore
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    
    result = (response.choices[0].message.content.strip()) # type: ignore

    # Models sometimes wrap JSON in ```json ... ``` or add text around it,
    # so keep only the part from the first { to the last }.
    start = result.find("{")
    end = result.rfind("}")

    if start != -1 and end != -1:
        result = result[start:end + 1]

    return json.loads(result)
    

