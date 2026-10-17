# A tool that fails this many times on the same task
# is given up on, so the loop doesn't keep retrying it.
MAX_TOOL_RETRIES = 2

def create_state(user_request, user_id=None, thread_id=None, max_steps=5):
    return{
        "user_request": user_request,
        "user_id": user_id,
        "thread_id": thread_id,
        "tasks": [],
        "current_task": None,
        "current_step": 0,
        "max_steps": max_steps,
        "actions": [],
        "observations": [],
        "results": [],
        "memories": [],
        "history": [],
        "finished": False,
        "final_answer": ""
    }

def add_memory(state, memory):
    state["memories"].append(memory)

def add_task(state, task):
    state["tasks"].append({
        "task": task,
        "status": "pending",
        "failures": 0
    })

def find_pending_task(state, task):

    # Normalize the task received from the planner.
    # This makes the comparison tolerant of:
    # "Generate password"
    # "generate password"
    # "generate_password"

    normalized_task = (
        task.strip().lower().replace("_"," ")
    )

    pending = [
        item
        for item in state["tasks"]

        # Skip finished tasks so duplicate tasks
        # (e.g. two "roll_dice") each get completed.
        if item["status"] == "pending"
    ]

    for item in pending:

        normalized_existing_task = (
            item["task"].lower().replace("_"," ")
        )

        if normalized_existing_task == normalized_task:
            return item

    # The planner reworded the task: use the first
    # pending one so the loop doesn't repeat it forever.
    return pending[0] if pending else None

def complete_task(state, task):

    item = find_pending_task(state, task)

    if item is None:
        return False

    item["status"] = "completed"

    return True

def fail_task(state, task):

    # Count a failed attempt. Returns True once the task
    # has hit the retry limit and has been given up on.
    item = find_pending_task(state, task)

    if item is None:
        return False

    item["failures"] += 1

    if item["failures"] >= MAX_TOOL_RETRIES:

        item["status"] = "failed"

        return True

    return False

def add_action(state, action):
    state["actions"].append(action)

def next_step(state):
    state["current_step"] += 1


def record_observation(state, action, observations):
    state[ "observations"].append(
        {
            "step": state["current_step"],
            "action": action,
            "observations": observations
        }
    )

def add_result(state, result):
    state["results"].append(result)

def get_pending_task(state):

    return [
        item["task"]
        for item in state["tasks"]
        if item["status"] == "pending"

    ]

def all_tasks_completed(state):

    # "Done" includes tasks given up on after
    # too many failures.
    if not state["tasks"]:
        return False

    return all (
        item["status"] in ("completed", "failed")
        for item in state["tasks"]
    )

def can_continue(state):
    return (
        not state["finished"]
        and state["current_step"] < state["max_steps"]
    )

def finish(state, answer):
    state["final_answer"] = answer
    state["finished"] = True
