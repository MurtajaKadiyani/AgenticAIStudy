from mcp_client import execute_tool

async def execute_action(action, state, client, arguments=None):

    # fastmcp raises on tool errors; turn that into an
    # observation so the planner can reconsider instead
    # of the whole program crashing.
    try:
        result = await execute_tool(
            client,
            action,
            arguments
        )
    except Exception as error:
        return f"Error: {error}"

    if hasattr(result, "content"):

        if result.content:
            return result.content[0].text

    return str(result)
