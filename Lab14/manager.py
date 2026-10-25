from research_agent import research
from coding_agent import write_code
from reviewer_agent import review
from openai import OpenAI
from dotenv import load_dotenv
import os
from state import create_state

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)

MAX_REVISIONS = 3

def choose_worker(state):
    prompt=f"""
    You are the Manager Agent in a multi-agent system.

    Your job is to decide which worker should act NEXT.

    ========================
    ORIGINAL TASK
    ========================
    {state["task"]}

    ========================
    CURRENT STATE
    ========================

    Research:
    {state["research"]}

    Code:
    {state["code"]}

    Review:
    {state["review"]}

    Execution History:
    {state["history"]}

    Revision Count:
    {state["revision_count"]}

    Maximum Revisions:
    {MAX_REVISIONS}

    Available workers:

    research
    coding
    review
    finish

    ========================
    DECISION RULES
    ========================

    1. If Research is empty, choose research

    2. If research exists and code is empty, choose coding.

    3. If a review exists with status "needs_revision", choose coding.

    4. If code exists and there is no review, choose review.

    5. If the previous review requested revision and new code has been generated,choose review again.

    6. If the latest review status is "approved", choose finish.

    7. If the maximum revision count has been reached, choose finish.

    Return ONLY one word:

    research
    coding
    review
    finish

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

    return (
        response
        .choices[0]
        .message
        .content
        .strip() # type: ignore
        .lower()
        )


def run_manager(task):

    state = create_state(task)

    while True:
        worker = choose_worker(state)

        print(f"\nManager selected: {worker}")

        if worker == "research":
            state["research"] = research(state["task"])
            state["history"].append("research")
        elif worker == "coding":
            if(state["review"] and state["revision_count"] >= MAX_REVISIONS):
                print("Maximum revisions reached.")
                break
            feedback=""
            if state["review"]:
                feedback = state["review"].get("feedback","")
            state["revision_count"] +=1
            state["code"] = write_code(
                state["task"],
                state["research"],
                feedback
            )
            # Old review applied to the previous code; clear it so the
            # manager sends the new code to review.
            state["review"] = ""
            state["history"].append("coding")
        elif worker == "review":
            state["review"] = review(
                state["task"],
                state["code"]
            )
            state["history"].append("review")
        elif worker == "finish":
            break
        else:
            raise ValueError(
                f"Unknown Worker: {worker}"
            )

    return state