import json

from llm import call_llm

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

    response = call_llm(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    
    result = response.strip() # type: ignore

    # Models sometimes wrap JSON in ```json ... ``` or add text around it,
    # so keep only the part from the first { to the last }.
    start = result.find("{")
    end = result.rfind("}")

    if start != -1 and end != -1:
        result = result[start:end + 1]

    return json.loads(result)
    

