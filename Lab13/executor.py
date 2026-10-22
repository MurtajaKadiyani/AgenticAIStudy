from mcp_manager import run_tool

async def execute_action(action,state,tool_registry):

    tool_name = action["tool"]

    arguments = action.get(
        "arguments",
        {}
    )

    result = await run_tool(
        tool_registry,
        tool_name,
        arguments
    )

    if hasattr(result, "content"):

        if result.content:
            return result.content[0].text
    
    return str(result)