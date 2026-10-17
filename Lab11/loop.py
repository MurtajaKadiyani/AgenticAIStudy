import json
import re

from state import *
from task_planner import decompose_task
from memory import search_facts, save_message, get_recent_conversation
from memory_extractor import extract_memory, is_none, store_fact
from state import add_memory


def is_finish(decision):

    # Accept both "FINISH" and "TASK: FINISH".
    text = decision.strip().upper()

    return text == "FINISH" or text.startswith("TASK: FINISH")


def parse_decision(decision):

    task = None
    tool = None
    args_text = None

    # Handle the normal multi-line format
    for line in decision.splitlines():

        line = line.strip()

        if line.startswith("TASK:"):

            task = line[5:].strip()

        elif line.startswith("TOOL:"):

            tool = line[5:].strip()

        elif line.startswith("ARGS:"):

            args_text = line[5:].strip()

    # Handle the case where the LLM puts
    # TASK and TOOL on the same line.
    if task and "TOOL:" in task:

        parts = task.split(
            "TOOL:",
            1
        )

        task = parts[0].strip()

        if not tool:

            tool = parts[1].strip()

    # Same for TOOL and ARGS on one line.
    if tool and "ARGS:" in tool:

        parts = tool.split(
            "ARGS:",
            1
        )

        tool = parts[0].strip()

        if not args_text:

            args_text = parts[1].strip()

    try:
        args = json.loads(args_text) if args_text else {}
    except json.JSONDecodeError:
        args = {}

    if not isinstance(args, dict):
        args = {}

    return task, tool, args


def remember(state, answer):

    user_request = state["user_request"]

    # Never store generated passwords in conversation memory,
    # since recent conversation is fed back into later prompts.
    saved_answer = answer

    for item in state["observations"]:

        if item["action"] == "generate_password":

            # The model may escape characters when it writes
            # the password (e.g. "\0" becomes "\\0"), so allow
            # backslashes between characters, not just an
            # exact match.
            pattern = r"\\*".join(
                re.escape(char)
                for char in item["observations"]
            )

            saved_answer = re.sub(
                pattern,
                "[password hidden]",
                saved_answer
            )

    # Short-term memory: this thread's chat history.
    save_message(state["thread_id"], "user", user_request)
    save_message(state["thread_id"], "assistant", saved_answer)

    # When tools ran, the answer may contain tool output
    # (e.g. a generated password), so extract facts from
    # the user's message only.
    source_answer = "" if state["actions"] else answer

    fact = extract_memory(user_request, source_answer)

    # Long-term memory: added, merged into or replacing
    # the user's existing facts.
    if not is_none(fact):
        store_fact(state["user_id"], fact)


async def run_agent_loop(
        user_request,
        planner,
        execute_action,
        format_answer,
        tools,
        client,
        user_id,
        thread_id
):
    state = create_state(user_request, user_id, thread_id)

    # Facts come from all of this user's threads,
    # most relevant first.
    memories = search_facts(user_id, user_request)

    for memory in memories:
        add_memory(state, memory["fact"])

    # Recent conversation comes from this thread only.
    state["history"] = get_recent_conversation(thread_id)

   # --------------------------------
   # Step 1: Decompose the goal
   # --------------------------------

    planned_tasks = decompose_task(
        user_request,
        tools,
        state["history"]
    )

    tool_names = {tool.name for tool in tools}
   
    for task in planned_tasks:

        # "NONE" means the request needs no tools.
        if task.strip().upper() == "NONE":
            continue

        add_task(
            state,
            task
         )

    # No tool tasks: answer directly and skip the planner call.
    if not state["tasks"]:

        print("\nTask Plan")
        print("---------")
        print("- NONE (no tools needed)")

        print(
            "\n--- Step "
            f"{state['current_step'] + 1} ---"
        )

        print("Planner Decision:")
        print("TASK: FINISH")

        answer = format_answer(
            state
        )

        finish(
            state,
            answer
        )

        remember(state, answer)

        return state

    # One step per task, plus room for retries,
    # so larger requests can still finish.
    state["max_steps"] = max(
        state["max_steps"],
        len(state["tasks"]) + 3
    )

    print("\nTask Plan")
    print("---------")

    for item in state["tasks"]:
   
        print(
             "-",
            item["task"]
        )
   
    # --------------------------------
    # Step 2: Autonomous loop
    # --------------------------------
    
    while can_continue(state):
   
           # --------------------------------
           # Check whether all tasks are done
           # --------------------------------
   
           if all_tasks_completed(state):
   
               print(
                   "\n--- Step "
                   f"{state['current_step'] + 1} ---"
               )
   
               print(
                   "Planner Decision:"
               )
   
               print("TASK: FINISH")
   
               answer = format_answer(
                   state
               )
   
               finish(
                   state,
                   answer
               )

               remember(state, answer)

               break

           print(
               f"\n--- Step "
               f"{state['current_step'] + 1} ---"
           )
   
           # --------------------------------
           # Ask planner for next task
           # --------------------------------
   
           decision = planner(
               state,
               tools
           )
   
           print(
               "Planner Decision:"
           )
   
           print(decision)
   
           # --------------------------------
           # Check FINISH
           # --------------------------------
   
           if is_finish(decision):
   
               answer = format_answer(
                   state
               )
   
               finish(
                   state,
                   answer
               )

               remember(state, answer)
   
               break
   
           # --------------------------------
           # Extract task and tool
           # --------------------------------
   
           task, tool, args = parse_decision(
               decision
           )
   
           if not task or not tool:
   
               add_result(
                   state,
                   "Invalid planner decision."
               )
   
               next_step(state)
   
               continue
   
           state["current_task"] = task

           # --------------------------------
           # No matching tool: the answer step
           # handles this task from knowledge.
           # --------------------------------

           if tool not in tool_names:

               complete_task(
                   state,
                   task
               )

               next_step(state)

               print("No tool needed")

               continue

           # --------------------------------
           # Execute tool
           # --------------------------------
   
           observation = await execute_action(
               tool,
               state,
               client,
               args
           )
   
           print(
               "Observation:",
               observation
           )
   
           # --------------------------------
           # Update state
           # --------------------------------
   
           add_action(
               state,
               tool
           )

           next_step(state)

           record_observation(
               state,
               tool,
               observation
           )
   
           add_result(
               state,
               observation
           )
   
           # --------------------------------
           # Mark task complete
           # --------------------------------
   
           if not observation.lower().startswith(
               "error"
           ):
   
               complete_task(
                   state,
                   task
               )
   
               print(
                   "completed"
               )
   
           elif fail_task(state, task):

               add_result(
                   state,
                   f"The task '{task}' failed {MAX_TOOL_RETRIES} times "
                   "and was given up on. Tell the user it could not be done."
               )

               print("failed, giving up")

           else:

               add_result(
                   state,
                   "The previous action failed. "
                   "The planner must reconsider."
               )

    # Step limit reached before FINISH:
    # still answer with what was gathered.
    if not state["finished"]:

        answer = format_answer(
            state
        )

        finish(
            state,
            answer
        )

        remember(state, answer)

    return state
