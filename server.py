"""
Toolbox — Generic MCP tool aggregator.

Discovers tools from remote MCP servers at startup and exposes them
through a single OmniTool toolbox. No built-in tools — everything
comes from remote servers configured via environment variables.

Environment:
  Remote servers (repeat pattern for each):
    MCP_REMOTE_{NAME}_URL       Server URL (required — presence = server exists)
    MCP_REMOTE_{NAME}_PREFIX    Prefix for tool names from this server (optional)
    MCP_REMOTE_{NAME}_TOOLS     Comma-separated tool filter (optional)
"""
from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan
from omnitool import OmniTool
from remote import discover_remote_configs, make_refresh_callback


# ─── Remote Servers ─────────────────────────────────────

REMOTE_SERVERS = discover_remote_configs()


# ─── Lifespan ───────────────────────────────────────────

@lifespan
async def startup(server: FastMCP):
    """Connect to remote MCP servers, discover tools, create OmniTool."""
    if not REMOTE_SERVERS:
        print("[toolbox] Warning: No remote servers configured. Set MCP_REMOTE_*_URL env vars.")

    refresh_fn = make_refresh_callback(REMOTE_SERVERS)
    remote_defs = await refresh_fn()

    for td in remote_defs:
        print(f"[toolbox] Discovered: {td.name}")

    if not remote_defs:
        print("[toolbox] No remote tools discovered.")

    OmniTool(
        mcp=server,
        tools=[],
        tool_defs=remote_defs,
        refresh_callback=refresh_fn,
        n_primary=0,
        show_index=False,
    )
    yield {}


# ─── MCP Server ─────────────────────────────────────────

GUIDELINES = """\
All tools are accessed through the toolbox.

find_tools — discover what's available
  tool="find_tools" params={"query": "what you need"}   search by keyword
  tool="find_tools"                                      browse all

get_schema — get full parameter details before calling
  tool="get_schema" params={"tools": "name1, name2"}

Call a tool
  tool="tool_name" params={"key": "value", ...}

Always assume you don't know what tools exist or how to call them
UNLESS you already have the schema in context.
If you don't, look it up first."""

mcp = FastMCP("Toolbox", instructions=GUIDELINES, lifespan=startup)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
