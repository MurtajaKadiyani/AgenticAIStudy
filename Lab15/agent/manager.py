from workers.research_agent import research
from workers.coding_agent import write_code
from workers.reviewer_agent import review
from llm import call_llm
from validation import validation_task
from logger import logger
from agent.state import create_state
from config import MAX_REVISIONS

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
    # Modified to call_llm
    response = call_llm(
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
    )

    return (
        response
        .strip() # type: ignore
        .lower()
        )


def run_manager(task):
    # Added validate_task here
    task = validation_task(task)
    state = create_state(task)

    # Added logger here
    logger.info(
    f"Starting manager for task: {task}"
    )

    while True:
        worker = choose_worker(state)

        # Logger added here
        logger.info(f"Manager selected: {worker}")

        print(f"\nManager selected: {worker}")

        if worker == "research":
            state["research"] = research(state["task"])
            state["history"].append("research")
        elif worker == "coding":
            if(state["review"] and state["revision_count"] >= MAX_REVISIONS):
                print("Maximum revisions reached.")
                # Added Logger
                logger.warning(f"Maximum steps reached: {MAX_REVISIONS}")
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
            # Added Logger here
            logger.info("Manager finished the workflow.")
            break
        else:
            raise ValueError(
                f"Unknown Worker: {worker}"
            )

    return state