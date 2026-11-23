"""Connect to the 4 MCP servers and turn their tools into LangChain tools."""
import json
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

from config import ROOT

SERVERS = ["imaging", "patient", "knowledge", "triage"]


def server_config() -> dict:
    # Each server is started as a subprocess speaking MCP over stdin/stdout.
    return {name: {"command": sys.executable,
                   "args": [str(ROOT / "mcp_servers" / f"{name}_server.py")],
                   "transport": "stdio"}
            for name in SERVERS}


async def load_tools() -> dict:
    client = MultiServerMCPClient(server_config())
    tools = await client.get_tools()
    return {t.name: t for t in tools}


def parse_tool_output(out):
    """MCP tool results arrive as text content blocks; our servers always send one JSON dict."""
    if isinstance(out, tuple):                 # (content, artifact) in some adapter versions
        out = out[0]
    if isinstance(out, list):
        texts = [b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b)) for b in out]
        out = "\n".join(t for t in texts if t)
    if isinstance(out, str):
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return {"raw": out}
    return out


async def call(tools: dict, name: str, **kwargs) -> dict:
    return parse_tool_output(await tools[name].ainvoke(kwargs))
