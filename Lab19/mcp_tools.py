import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER_PATH = Path(__file__).parent / "server.py"

client = MultiServerMCPClient(
    {
        "time_server":{
            "transport": "stdio",
            # Use the same Python running main.py so fastmcp is available
            "command": sys.executable,
            "args":[
                str(SERVER_PATH)
            ]
        }
    }
)
