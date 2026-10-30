from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver

from agent.planner import planner
from agent.executor import execute_action
from agent.state import AgentState
from agent.runner import format_answer


def create_planner_node(tools):

    def planner_node(state: AgentState):

        action = planner(state,tools)

        print("\nPlanner Selected:")
        print(action)

        return {
            "action": action,
            "finished": action == "FINISH"
        }

    return planner_node

def approval_node(state: AgentState):
    decision = interrupt(
        {
            "message": "Approve this tool call?",
            "action": state["action"]
        }
    )

    return {
        "approval": decision
    }

def create_tool_node(client):

    async def tool_node(state: AgentState):
        action = state["action"]

        observation = await execute_action(action,client)

        print("\nTool Observation:")
        print(observation)

        return{
            "observations": (
                state["observations"]
                +
                [
                  {
                    "step": state["current_step"] + 1,
                    "action": action,
                    "observation": observation
                  }
                ]
            ),
            "actions": (
                state["actions"] + [action]
            ),
            "current_step": (
                state["current_step"] + 1
            )
        }
    return tool_node

def rejected_node(state: AgentState):
    action = state["action"]

    print("\nTool Rejected:")
    print(action)

    return {
        "observations": (
            state["observations"]
            +
            [
              {
                "step": state["current_step"] + 1,
                "action": action,
                "observation": "REJECTED by user. This tool was NOT run."
              }
            ]
        ),
        "actions": (
            state["actions"] + [action]
        ),
        "current_step": (
            state["current_step"] + 1
        )
    }

def answer_node(state: AgentState):
    answer = format_answer(state)

    print("\nFinal Answer:")
    print(answer)

    return {
        "final_answer": answer,
        "finished": True
    }

def route_after_planner(state: AgentState):

    if state["action"] == "FINISH":
        return "finish"
    return "approval"

def route_after_approval(state: AgentState):
    if state["approval"] in ("yes", "y"):
        return "approved"
    return "rejected"

def route_after_tool(state: AgentState):
    if(state["current_step"]>=state["max_steps"]):
        return "stop"
    return "continue"

def build_graph(tools,client):

    builder = StateGraph(AgentState)

    builder.add_node(
        "planner",
        create_planner_node(tools)
    )

    builder.add_node(
        "approval",
        approval_node
    )

    builder.add_node(
        "tool",
        create_tool_node(client)
    )

    builder.add_node(
        "rejected",
        rejected_node
    )

    builder.add_node(
        "answer",
        answer_node
    )

    builder.add_edge(
        START,
        "planner"
    )

    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "approval": "approval",
            "finish": "answer"
        }
    )

    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "approved": "tool",
            "rejected": "rejected"
        }
    )

    builder.add_conditional_edges(
            "tool",
            route_after_tool,
            {
                "continue": "planner",
                "stop": "answer"
            }
        )

    builder.add_conditional_edges(
            "rejected",
            route_after_tool,
            {
                "continue": "planner",
                "stop": "answer"
            }
        )

    builder.add_edge(
        "answer",
        END
    )

    checkpointer = InMemorySaver()

    return builder.compile(checkpointer=checkpointer)
        