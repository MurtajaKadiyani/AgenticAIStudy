from crewai import Task 
from agents import (researcher,coder,reviewer)

research_task = Task(
    description=(
        "Hi. I am Murtaja Kadiyani.Remember that my name is Murtaja Kadiyani for future interactions."" Research autonomous AI agents. "
        "Explain the agent loop, tools, "
        "state and observations. "
        "Before completing the task, use "
        "the MCP Current Time tool and "
        "include the returned time."
    ),
    expected_output=(
        "A concise technical explanation "
        "including the current time "
        "obtained from the MCP tool."
    ),
    agent=researcher
)

coding_task = Task(
    description=(
        "Using the research provided by "
        "the researcher, create a simple "
        "Python example of an AI agent loop."
),
    expected_output=(
        "A complete Python code example "
        "with a short explanation."
    ),
    agent=coder,
    context=[research_task]

)

review_task = Task(
    description=(
        "Review the Python code produced "
        "by the developer. Identify "
        "errors and suggest improvements."
),
    expected_output=(
        "A code review containing "
        "identified problems and "
        "recommended improvements, in "
        "under 500 words. Do not rewrite "
        "the full code."
    ),
    agent=reviewer,
    context=[coding_task]

)

memory_task = Task(
    description=(
        "What is the user's name? Search "
        "memory for the user's name."
    ),
    expected_output=(
        "The user's name."
    ),
    agent=researcher
)