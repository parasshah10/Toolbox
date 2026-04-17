"""
omnitool.py — Single-tool MCP gateway with lazy tool discovery via BM25 search.

Wraps any number of plain functions into a single FastMCP tool.
Primary tools are always available in the tool definition.
Secondary tools are discoverable via natural-language search.

Discovery is split into two actions:
  find_tools  — broad, cheap: returns tool names ranked by relevance
  get_schema  — precise, complete: returns full signatures and parameter details

Usage:
    from fastmcp import FastMCP
    from omnitool import OmniTool

    mcp = FastMCP("my-server")

    def create_file(path: str, content: str) -> str:
        ...

    def rename_file(old_path: str, new_path: str) -> str:
        ...

    OmniTool(
        mcp=mcp,
        tools=[create_file, rename_file, ...],
        n_primary=3,   # first 3 get full definitions baked in; rest go to search index
    )

Dependencies:
    Required : fastmcp
    Optional : rank-bm25   (pip install rank-bm25)  — better search; falls back gracefully
    Optional : pydantic                              — extracts Field() descriptions automatically
"""

from __future__ import annotations

import inspect
import json
import re
import difflib
import types as _types
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, get_type_hints

# ── Optional dependencies ──────────────────────────────────────────────────────

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

try:
    from pydantic.fields import FieldInfo as _FieldInfo
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False

# Sentinel for "no default value provided"
_MISSING = object()

# ── Configuration ──────────────────────────────────────────────────────────────

_MAX_GROUP_PREVIEW = 5   # max tool names shown per group in compressed listing
_DEFAULT_SEARCH_LIMIT = 10  # default results for find_tools search


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ParamDef:
    name: str
    type_str: str        # compact string: "str", "int", "list[str]", etc.
    description: str     # from Pydantic Field() or empty
    default: Any         # _MISSING if no default
    optional: bool       # True if Optional[X], or has a default
    has_default: bool    # True if a default value exists


@dataclass
class ToolDef:
    name: str
    func: Callable
    docstring: str
    params: List[ParamDef]

    def index_text(self) -> str:
        """All searchable text for this tool, flattened into one string."""
        parts = [
            self.name,
            self.name.replace('_', ' '),
            self.docstring or '',
        ]
        for p in self.params:
            parts.append(p.name.replace('_', ' '))
            if p.description:
                parts.append(p.description)
        return ' '.join(filter(None, parts))


# ── Main class ─────────────────────────────────────────────────────────────────

class OmniTool:
    """
    Registers a single FastMCP tool that routes to any wrapped function.
    Primary tools are fully described in the tool definition.
    Secondary tools are discovered via natural-language BM25 search.
    """

    def __init__(
        self,
        mcp: Any,
        tools: Optional[List[Callable]] = None,
        tool_defs: Optional[List[ToolDef]] = None,
        refresh_callback: Optional[Callable] = None,
        n_primary: int = 3,
        show_index: Union[bool, str] = False,
    ):
        """
        Args:
            mcp:              FastMCP instance to register the tool on.
            tools:            List of plain functions to wrap. Order = priority.
            tool_defs:        Pre-built ToolDef objects (e.g. from remote discovery).
                              Skips introspection — goes straight into the index.
            refresh_callback: Async function returning list[ToolDef]. If provided,
                              a 'refresh' tool is added that re-discovers remote
                              tools and rebuilds the index.
            n_primary:        How many functions (from the start of the list) get
                              full definitions baked into the tool description.
                              The rest go into the BM25 search index.
            show_index:       False = hide, True = auto-generate name list from
                              secondary tools, str = use verbatim text.
        """
        self.mcp = mcp
        self.n_primary = n_primary
        self.show_index = show_index

        # Store for refresh
        self._local_tools: List[Callable] = list(tools or [])
        self._refresh_callback = refresh_callback

        # Introspect local tools
        self._tool_defs: List[ToolDef] = [self._introspect(fn) for fn in self._local_tools]

        # Add pre-built remote tool defs
        self._tool_defs.extend(tool_defs or [])

        # Add refresh tool if callback provided
        if refresh_callback:
            self._tool_defs.append(ToolDef(
                name="refresh",
                func=self._do_refresh,
                docstring="Re-discover tools from remote servers. Use after remote servers restart or new tools are added.",
                params=[],
            ))

        # Build tool map with collision detection
        self._tool_map: Dict[str, ToolDef] = {}
        for td in self._tool_defs:
            if td.name in self._tool_map:
                print(f"[omnitool] Warning: Tool name '{td.name}' collision — later definition wins. Use prefix to resolve.")
            self._tool_map[td.name] = td

        # Split primary vs secondary
        self._primary: List[ToolDef] = self._tool_defs[:n_primary]
        self._secondary: List[ToolDef] = self._tool_defs[n_primary:]

        # Build BM25 index over secondary tools
        self._bm25: Any = None
        self._build_index()

        # Register single tool with FastMCP
        self._register()

    # ── Refresh ────────────────────────────────────────────────────────────────

    async def _do_refresh(self) -> str:
        """Re-run remote discovery and rebuild indexes."""
        if not self._refresh_callback:
            return "Refresh not available — no remote servers configured."
        try:
            new_remote_defs = await self._refresh_callback()
        except Exception as e:
            return f"Refresh failed: {e}"

        # Rebuild: re-introspect locals + new remotes + refresh tool
        self._tool_defs = [self._introspect(fn) for fn in self._local_tools]
        self._tool_defs.extend(new_remote_defs)
        self._tool_defs.append(ToolDef(
            name="refresh",
            func=self._do_refresh,
            docstring="Re-discover tools from remote servers. Use after remote servers restart or new tools are added.",
            params=[],
        ))

        # Rebuild all indexes
        self._tool_map = {td.name: td for td in self._tool_defs}
        self._primary = self._tool_defs[:self.n_primary]
        self._secondary = self._tool_defs[self.n_primary:]
        self._build_index()

        names = [td.name for td in self._tool_defs if td.name != "refresh"]
        return f"Refreshed. {len(names)} tool(s): {', '.join(names)}"

    # ── Introspection ──────────────────────────────────────────────────────────

    def _introspect(self, fn: Callable) -> ToolDef:
        """Extract full parameter and docstring metadata from a function."""
        name = fn.__name__
        docstring = (inspect.getdoc(fn) or '').strip()

        sig = inspect.signature(fn)
        try:
            hints = get_type_hints(fn, include_extras=True)
        except Exception:
            hints = {}

        params: List[ParamDef] = []
        for pname, param in sig.parameters.items():
            annotation = hints.get(pname, inspect.Parameter.empty)

            # Pull Field() description out of Annotated[T, Field(description="...")]
            description = ''
            if _HAS_PYDANTIC and hasattr(annotation, '__metadata__'):
                for meta in annotation.__metadata__:
                    if isinstance(meta, _FieldInfo) and meta.description:
                        description = meta.description
                        break

            # Strip Annotated wrapper to get the real type
            base_type = annotation
            if hasattr(annotation, '__args__') and hasattr(annotation, '__metadata__'):
                base_type = annotation.__args__[0]

            type_str = self._type_to_str(base_type)
            has_default = param.default is not inspect.Parameter.empty
            optional = has_default or self._is_optional(base_type)
            default = param.default if has_default else _MISSING

            params.append(ParamDef(
                name=pname,
                type_str=type_str,
                description=description,
                default=default,
                optional=optional,
                has_default=has_default,
            ))

        return ToolDef(name=name, func=fn, docstring=docstring, params=params)

    def _type_to_str(self, annotation: Any) -> str:
        """Convert a type annotation to a compact readable string."""
        if annotation is inspect.Parameter.empty or annotation is None:
            return 'any'

        origin = getattr(annotation, '__origin__', None)
        args = getattr(annotation, '__args__', ())

        # Union / Optional  (typing.Union and Python 3.10+ X | Y)
        if origin is typing.Union:
            non_none = [a for a in args if a is not type(None)]
            return self._type_to_str(non_none[0]) if len(non_none) == 1 else 'any'

        # Python 3.10+ union syntax: str | None  →  types.UnionType
        if isinstance(annotation, getattr(_types, 'UnionType', type(None))):
            non_none = [a for a in getattr(annotation, '__args__', ()) if a is not type(None)]
            return self._type_to_str(non_none[0]) if len(non_none) == 1 else 'any'

        # List
        if origin is list:
            return f'list[{self._type_to_str(args[0])}]' if args else 'list'

        # Dict
        if origin is dict:
            return 'dict'

        # Primitives
        _primitives = {str: 'str', int: 'int', float: 'float', bool: 'bool'}
        if annotation in _primitives:
            return _primitives[annotation]

        if hasattr(annotation, '__name__'):
            return annotation.__name__

        return 'any'

    def _is_optional(self, annotation: Any) -> bool:
        """Return True if the annotation is Optional[X] or X | None."""
        origin = getattr(annotation, '__origin__', None)
        args = getattr(annotation, '__args__', ())

        if origin is typing.Union and type(None) in args:
            return True

        # Python 3.10+ union syntax
        if isinstance(annotation, getattr(_types, 'UnionType', type(None))):
            return type(None) in getattr(annotation, '__args__', ())

        return False

    # ── Renderers ──────────────────────────────────────────────────────────────

    def _render_brief(self, td: ToolDef) -> str:
        """Just the tool name. No description, no parameters."""
        return td.name

    def _render_compact(self, td: ToolDef) -> str:
        """
        Compact DSL: signature + first docstring line + param descriptions.

        create_file(path: str, content: str, mode?: str="w")
          Creates a file at the given path with the given content.
          path: Destination file path.
          mode: Write mode. Default: "w".
        """
        param_parts = []
        for p in td.params:
            marker = '?' if p.optional else ''
            s = f'{p.name}{marker}: {p.type_str}'
            if p.has_default and p.default is not _MISSING and p.default is not None:
                if isinstance(p.default, bool):
                    s += f'={str(p.default).lower()}'
                elif isinstance(p.default, str):
                    s += f'="{p.default}"'
                else:
                    s += f'={p.default}'
            param_parts.append(s)

        lines = [f'{td.name}({", ".join(param_parts)})']

        # First line of docstring
        if td.docstring:
            first = td.docstring.split('\n')[0].strip()
            if first:
                lines.append(f'  {first}')

        # Param descriptions (only if non-empty)
        for p in td.params:
            if p.description:
                lines.append(f'  {p.name}: {p.description}')

        return '\n'.join(lines)

    def _render_full(self, td: ToolDef) -> str:
        """
        Full schema: signature + full docstring + all param descriptions.
        Used by get_schema for complete tool documentation.
        """
        param_parts = []
        for p in td.params:
            marker = '?' if p.optional else ''
            s = f'{p.name}{marker}: {p.type_str}'
            if p.has_default and p.default is not _MISSING and p.default is not None:
                if isinstance(p.default, bool):
                    s += f'={str(p.default).lower()}'
                elif isinstance(p.default, str):
                    s += f'="{p.default}"'
                else:
                    s += f'={p.default}'
            param_parts.append(s)

        lines = [f'{td.name}({", ".join(param_parts)})']

        # Full docstring
        if td.docstring:
            lines.append(f'  {td.docstring}')

        # Param descriptions
        for p in td.params:
            if p.description:
                lines.append(f'  {p.name}: {p.description}')

        return '\n'.join(lines)

    # ── Description Builder ────────────────────────────────────────────────────

    def _build_description(self) -> str:
        """
        Build the text registered as the FastMCP tool description.
        This is the most token-sensitive piece — kept tight on purpose.
        """
        lines = [
            'Call THIS tool: tool="name" params={...}',
            'find_tools to discover, get_schema for details, then call.',
        ]

        # Primary tools — always shown with full compact definitions
        for td in self._primary:
            lines.append(self._render_compact(td))

        # Show available tools hint
        if self.show_index:
            if isinstance(self.show_index, str):
                lines.append('TOOLS AVAILABLE: ' + self.show_index)
            elif self._secondary:
                lines.append('TOOLS AVAILABLE: ' + ', '.join(td.name for td in self._secondary))

        # Discovery tools — always available
        lines.append(
            'find_tools(query?: str, limit?: int=10) '
            'Names only. Empty/* = all grouped. Query = BM25 ranked.'
        )
        lines.append(
            'get_schema(tools: str) '
            'Full schema. Comma-separated names. '
            'Example: get_schema("tool1, tool2")'
        )

        return '\n'.join(lines)

    # ── BM25 Index ─────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """Split on whitespace and common delimiters; lowercase."""
        tokens = re.split(r'[\s_\-./,;:()\[\]"\']+', text.lower())
        return [t for t in tokens if t]

    def _build_index(self) -> None:
        """Build BM25 index over all secondary tools at init time."""
        if not self._secondary or not _HAS_BM25:
            return
        corpus = [self._tokenize(td.index_text()) for td in self._secondary]
        self._bm25 = _BM25Okapi(corpus)

    def _search(self, q: str, top_k: int = _DEFAULT_SEARCH_LIMIT) -> str:
        """Run BM25 (or fallback) search and return names-only results."""
        if not self._secondary:
            return (
                'No additional tools beyond the built-ins.\n'
                f'Primary tools: {", ".join(td.name for td in self._primary)}'
            )

        query_tokens = self._tokenize(q)

        if self._bm25 is not None:
            scores = self._bm25.get_scores(query_tokens)
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            results = [
                self._secondary[i]
                for i in indices[:top_k]
                if scores[i] > 0
            ]
        else:
            # Fallback: simple token overlap when rank_bm25 is not installed
            q_tokens = set(query_tokens)
            scored = []
            for td in self._secondary:
                doc_tokens = set(self._tokenize(td.index_text()))
                overlap = len(q_tokens & doc_tokens)
                if overlap > 0:
                    scored.append((overlap, td))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [td for _, td in scored[:top_k]]

        if not results:
            return (
                f'No tools found for "{q}".\n'
                'Try different keywords or a broader description.\n'
                'Tip: find_tools() to see all available tools.'
            )

        # Catalog size annotation
        total = len(self._secondary)
        if len(results) < total:
            header = f'Found {len(results)} of {total} tools for "{q}":'
        else:
            header = f'Found {len(results)} tools for "{q}":'

        lines = [header, '']
        for td in results:
            lines.append(f'  {self._render_brief(td)}')
        lines.append('')
        lines.append('get_schema("tool1, tool2, ...") for details')
        return '\n'.join(lines)

    # ── Forgiving JSON Parser ──────────────────────────────────────────────────

    def _clean_json(self, text: str) -> str:
        """Strip markdown fences and fix common LLM JSON mistakes."""
        text = text.strip()
        # Remove opening/closing markdown code fences
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return text

    def _parse_call(self, tool_param: str, params: Any) -> Tuple[str, dict]:
        """
        Normalize tool + params regardless of how the model packaged them.

        Handles:
          - JSON blob stuffed into the tool field
          - Alternate key names: "tool", "action", "tool_name", "name"
          - params passed as a JSON string instead of a dict
          - Markdown fences around JSON
          - Trailing commas in JSON
          - None params
        """
        # Case: model stuffed the whole call as a JSON blob into tool param
        stripped = tool_param.strip()
        if stripped.startswith(('{', '`')):
            cleaned = self._clean_json(stripped)
            try:
                data = json.loads(cleaned)
                tool_param = (
                    data.get('tool')
                    or data.get('action')
                    or data.get('tool_name')
                    or data.get('name')
                    or ''
                )
                params = (
                    data.get('params')
                    or data.get('parameters')
                    or data.get('args')
                    or {}
                )
            except json.JSONDecodeError:
                pass  # tool_param stays as-is, params stays as-is

        # Case: params is a JSON string instead of a dict
        if isinstance(params, str) and params.strip():
            cleaned = self._clean_json(params)
            try:
                params = json.loads(cleaned)
            except json.JSONDecodeError:
                params = {}

        if params is None:
            params = {}

        if not isinstance(params, dict):
            params = {}

        return tool_param.strip(), self._coerce_params(params)

    def _coerce_params(self, params: dict) -> dict:
        """Coerce string "true"/"false" to bool."""
        result = {}
        for k, v in params.items():
            if isinstance(v, str):
                lower = v.lower()
                if lower == 'true':
                    result[k] = True
                elif lower == 'false':
                    result[k] = False
                else:
                    result[k] = v
            else:
                result[k] = v
        return result

    # ── Dispatcher ─────────────────────────────────────────────────────────────

    async def _dispatch(self, tool_name: str, params: dict) -> Any:
        """Route a parsed call to the correct function or built-in."""

        # ── Built-in: find_tools ──────────────────────────────────────────

        if tool_name == 'find_tools':
            q = (
                params.get('q')
                or params.get('query')
                or params.get('search')
                or params.get('term')
                or ''
            )
            limit = params.get('limit') or params.get('n') or params.get('count') or _DEFAULT_SEARCH_LIMIT
            try:
                limit = max(1, min(int(limit), 50))
            except (TypeError, ValueError):
                limit = _DEFAULT_SEARCH_LIMIT

            # No query → compressed grouped listing (names only)
            if not q or q.strip() == '*':
                return self._list_all_compressed()

            q = str(q).strip()

            # Exact tool name match → confirm + redirect
            if q in self._tool_map:
                return (
                    f'Tool found: {q}\n'
                    f'get_schema("{q}") for full parameter details\n'
                    f'tool="{q}" params={{...}} to call'
                )

            # Prefix/group match → list all names in group
            prefix_matches = [
                td for td in self._secondary
                if td.name.startswith(q + '-') or td.name.startswith(q + '_')
            ]
            if len(prefix_matches) >= 3:
                names = [td.name for td in prefix_matches]
                lines = [f'{q} ({len(names)} tools):']
                for name in names:
                    lines.append(f'  {name}')
                lines.append('')
                lines.append(f'get_schema("{names[0]}, ...") for details')
                return '\n'.join(lines)

            # Natural language query → BM25 search (names only)
            return self._search(q, top_k=limit)

        # ── Built-in: get_schema ──────────────────────────────────────────

        if tool_name == 'get_schema':
            tools_param = (
                params.get('tools')
                or params.get('tool')
                or params.get('names')
                or params.get('name')
                or ''
            )

            # Parse comma-separated string or list
            if isinstance(tools_param, list):
                names = [str(n).strip() for n in tools_param if str(n).strip()]
            elif isinstance(tools_param, str):
                names = [n.strip() for n in tools_param.split(',') if n.strip()]
            else:
                names = []

            if not names:
                return (
                    'Provide tool name(s) to look up.\n'
                    'Example: get_schema("research, firecrawl-firecrawl_scrape")\n'
                    'Use find_tools() to discover available tools first.'
                )

            results = []
            found_count = 0
            for name in names:
                td = self._tool_map.get(name)
                if td:
                    results.append(self._render_full(td))
                    found_count += 1
                else:
                    close = difflib.get_close_matches(
                        name, list(self._tool_map.keys()), n=3, cutoff=0.5
                    )
                    if close:
                        suggestions = ', '.join(f'"{c}"' for c in close)
                        results.append(f'"{name}" not found. Did you mean: {suggestions}?')
                    else:
                        results.append(f'"{name}" not found. Use find_tools(query="...") to search.')

            # Build output
            header = ''
            if len(names) > 1:
                header = f'Schema for {len(names)} tool(s):\n\n'

            body = '\n\n'.join(results)
            footer = ''
            if found_count > 0:
                footer = '\n\nTo call: tool="<name>" params={...}'

            return (header + body + footer).strip()

        # ── Tool lookup and dispatch ──────────────────────────────────────

        td = self._tool_map.get(tool_name)

        if td is None:
            close = difflib.get_close_matches(
                tool_name, list(self._tool_map.keys()), n=1, cutoff=0.55
            )
            suggestion = f'\n  Did you mean: "{close[0]}"?' if close else ''
            primary_list = ', '.join(t.name for t in self._primary)
            return (
                f'Unknown tool "{tool_name}".{suggestion}\n'
                f'Primary tools: {primary_list}, find_tools, get_schema\n'
                f'Tip: find_tools(query="describe what you want to do")'
            )

        # Check required params
        missing = [
            p.name
            for p in td.params
            if not p.optional and not p.has_default and p.name not in params
        ]
        if missing:
            return (
                f'Missing required param(s) for "{tool_name}": {", ".join(missing)}\n'
                f'Schema:\n{self._render_compact(td)}'
            )

        # Call the wrapped function
        try:
            if inspect.iscoroutinefunction(td.func):
                return await td.func(**params)
            else:
                return td.func(**params)
        except TypeError as e:
            return (
                f'Parameter error calling "{tool_name}": {e}\n'
                f'Schema:\n{self._render_compact(td)}'
            )
        except Exception as e:
            return f'Error in "{tool_name}": {e}'

    # ── Compressed Listing ─────────────────────────────────────────────────────

    def _list_all_compressed(self) -> str:
        """Ultra-compressed listing: grouped tool names, no descriptions."""
        if not self._secondary:
            return 'No additional tools available.'

        # Group tools by prefix (prefix-toolname pattern via dash)
        grouped: Dict[str, List[str]] = {}
        ungrouped: List[str] = []

        for td in self._secondary:
            if '-' in td.name:
                prefix, short = td.name.split('-', 1)
                grouped.setdefault(prefix, []).append(td.name)
            else:
                ungrouped.append(td.name)

        lines = [f'All available tools ({len(self._secondary)}):', '']

        # Grouped tools
        for prefix, short_names in grouped.items():
            count = len(short_names)
            if count <= _MAX_GROUP_PREVIEW:
                names_str = ', '.join(short_names)
            else:
                names_str = ', '.join(short_names[:_MAX_GROUP_PREVIEW]) + ', ...'
            lines.append(f'{prefix} ({count}): {names_str}')

        # Ungrouped tools
        if ungrouped:
            if grouped:
                lines.append('')
            lines.append(', '.join(ungrouped))

        lines.append('')
        lines.append('get_schema("tool1, tool2, ...") for details')
        return '\n'.join(lines)

    # ── FastMCP Registration ───────────────────────────────────────────────────

    def _register(self) -> None:
        """Register the single gateway tool with the FastMCP instance."""
        description = self._build_description()
        omni = self  # explicit capture for the closure

        @self.mcp.tool(description=description)
        async def toolbox(tool: str, params: Optional[Dict[str, Any]] = None) -> Any:
            resolved = params if params is not None else {}
            tool_name, parsed_params = omni._parse_call(tool, resolved)

            if not tool_name:
                primary_list = ', '.join(td.name for td in omni._primary)
                return (
                    'No tool specified.\n'
                    f'Primary tools: {primary_list}, find_tools, get_schema\n'
                    'Example: tool="find_tools" params={"query": "what you need"}'
                )

            return await omni._dispatch(tool_name, parsed_params)
