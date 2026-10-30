from crewai import Agent
from llm import llm
from mcp_tool import mcp_current_time

researcher = Agent(
    role="AI Researcher",
    goal=(
         "Research AI agent concepts "
         "and provide accurate information."
    ),
    backstory=(
        "You are an AI researcher "
        "specialized in AI agents. "
    ),
    llm=llm,
    tools=[mcp_current_time],
    verbose=True
)

coder = Agent(
    role="Python Developer",
    goal=(
         "Create clear and correct "
         "Python code based on research."
    ),
    backstory=(
        "You are an experienced Python "
        "developer who writes clean and "
        "practical code."
    ),
    llm=llm,
    verbose=True
)

reviewer = Agent(
    role="Code Reviewer",
    goal=(
        "Review generated code "
        "and identify problems."
    ),
    backstory=(
        "You are a senior Python reviewer "
        "focused on correctness and quality."
    ),
    llm=llm,
    verbose=True
)