"""
remote.py — Bridge between remote MCP servers and OmniTool.

Connects to remote MCP servers via fastmcp.Client, discovers their tools,
converts JSON Schema definitions to OmniTool's ToolDef/ParamDef format,
and creates per-call proxy callables for transparent tool forwarding.

Each proxy opens a fresh connection per call — no persistent sessions,
no stale state, resilient to remote server restarts.

Also provides env-var-based remote server discovery and refresh callbacks
for use by any hub/aggregator server.
"""
from __future__ import annotations
from typing import Any, Callable, List, Optional
from fastmcp import Client
from omnitool import ToolDef, ParamDef, _MISSING
import os


# ── JSON Schema → Compact Type ────────────────────────────────────────────────

def _json_type_to_str(prop: dict) -> str:
    """
    Convert a single JSON Schema property definition to a compact type string.
    Handles:
      - Primitive types: string, integer, number, boolean
      - Arrays with typed items
      - Objects
      - anyOf with null (Optional unwrapping)
      - Fallback to "any" for anything unrecognized
    """
    # anyOf pattern: [{"type": "string"}, {"type": "null"}] → unwrap to "str"
    if "anyOf" in prop:
        non_null = [t for t in prop["anyOf"] if t.get("type") != "null"]
        if len(non_null) == 1:
            return _json_type_to_str(non_null[0])
        return "any"

    schema_type = prop.get("type")
    if not schema_type:
        return "any"

    type_map = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "object": "dict",
    }
    if schema_type in type_map:
        return type_map[schema_type]
    if schema_type == "array":
        items = prop.get("items")
        if items:
            inner = _json_type_to_str(items)
            return f"list[{inner}]"
        return "list"
    return "any"


# ── JSON Schema → ParamDef List ───────────────────────────────────────────────

def schema_to_params(input_schema: dict) -> List[ParamDef]:
    """
    Convert a JSON Schema inputSchema to a list of ParamDef objects.
    Reads 'properties' and 'required' from the schema. Each property
    becomes a ParamDef with type, description, default, and optionality
    derived from the schema.
    """
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    params = []
    for name, prop in properties.items():
        # Determine if this is an Optional (anyOf with null)
        is_anyof_optional = False
        if "anyOf" in prop:
            null_types = [t for t in prop["anyOf"] if t.get("type") == "null"]
            if null_types:
                is_anyof_optional = True

        type_str = _json_type_to_str(prop)
        description = prop.get("description", "")
        has_default = "default" in prop
        default = prop.get("default", _MISSING)
        is_required = name in required
        optional = not is_required or is_anyof_optional or has_default

        params.append(ParamDef(
            name=name,
            type_str=type_str,
            description=description,
            default=default,
            optional=optional,
            has_default=has_default,
        ))
    return params


# ── Per-Call Proxy Callable ───────────────────────────────────────────────────

def _make_proxy(url: str, remote_name: str) -> Callable:
    """
    Create an async function that proxies a tool call to a remote MCP server.
    Each invocation opens a fresh connection (per-call model):
    no persistent sessions, resilient to remote restarts.
    The proxy closes over the URL and original remote tool name.
    """
    async def proxy(**kwargs: Any) -> str:
        try:
            async with Client(url) as client:
                result = await client.call_tool(remote_name, arguments=kwargs)
                texts = [
                    block.text
                    for block in result.content
                    if hasattr(block, "text")
                ]
                return "\n".join(texts) if texts else "No output."
        except Exception as e:
            return f"Remote tool error: {e}"

    # Preserve the remote tool name for introspection/debugging
    proxy.__name__ = remote_name
    proxy.__qualname__ = f"proxy<{remote_name}>"
    return proxy


# ── Tool Discovery ────────────────────────────────────────────────────────────

async def discover_tools(
    client: Client,
    url: str,
    tool_filter: Optional[List[str]] = None,
    prefix: Optional[str] = None,
) -> List[ToolDef]:
    """
    Discover tools from a connected remote MCP server and return ToolDefs.

    Args:
        client:       An already-connected fastmcp.Client (used for list_tools only).
        url:          The server URL — stored in proxy callables for per-call connections.
        tool_filter:  If set, only import tools with these names. None imports all.
        prefix:       If set, prepend "{prefix}-" to all tool names from this server.

    Returns:
        List of ToolDef objects ready to pass to OmniTool's tool_defs parameter.
    """
    tools = await client.list_tools()
    tool_defs = []
    for tool in tools:
        # Apply filter
        if tool_filter and tool.name not in tool_filter:
            continue

        # Build local name with optional prefix
        local_name = f"{prefix}-{tool.name}" if prefix else tool.name

        # Convert schema to params
        try:
            params = schema_to_params(tool.inputSchema or {})
        except Exception as e:
            print(f"[remote] Skipping {tool.name}: bad schema — {e}")
            continue

        # Create per-call proxy (stores URL, not client)
        func = _make_proxy(url, tool.name)

        tool_defs.append(ToolDef(
            name=local_name,
            func=func,
            docstring=tool.description or "",
            params=params,
        ))

    return tool_defs


# ── Env-Based Server Discovery ────────────────────────────────────────────────

def discover_remote_configs() -> list[dict]:
    """
    Build remote server configs from MCP_REMOTE_*_URL env vars.

    Convention:
        MCP_REMOTE_{NAME}_URL     — Server URL (required, presence = server exists)
        MCP_REMOTE_{NAME}_PREFIX  — Prefix for tool names from this server (optional)
        MCP_REMOTE_{NAME}_TOOLS   — Comma-separated tool name filter (optional)
    """
    servers = {}
    for key, value in os.environ.items():
        if not key.startswith("MCP_REMOTE_") or not value:
            continue
        rest = key[len("MCP_REMOTE_"):]
        if rest.endswith("_URL"):
            name = rest[:-4]
            servers.setdefault(name, {})["url"] = value
        elif rest.endswith("_PREFIX"):
            name = rest[:-7]
            servers.setdefault(name, {})["prefix"] = value
        elif rest.endswith("_TOOLS"):
            name = rest[:-6]
            servers.setdefault(name, {})["tools"] = [
                t.strip() for t in value.split(",") if t.strip()
            ]
    return [cfg for cfg in servers.values() if cfg.get("url")]


def make_refresh_callback(configs: list[dict]) -> Callable:
    """Create an async callback that re-discovers tools from all configured remotes."""
    async def refresh():
        remote_defs = []
        for cfg in configs:
            url = cfg.get("url")
            if not url:
                continue
            try:
                async with Client(url) as client:
                    defs = await discover_tools(
                        client, url,
                        tool_filter=cfg.get("tools"),
                        prefix=cfg.get("prefix"),
                    )
                    remote_defs.extend(defs)
            except Exception as e:
                print(f"[remote] {url} unavailable: {e}")
        return remote_defs
    return refresh
