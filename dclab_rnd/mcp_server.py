"""DCLab as an MCP server over HTTP, so any MCP client can drive the notebook.

The tools are the intern's toolbox: the evidence index, the column auditor, the
prediction contract, the five deterministic stages, the notebook export. Hugging
Face's Chat UI (and its ML Intern mode) connects to this endpoint through
``MCP_SERVERS``; so can Claude Desktop, Cursor or any other MCP client.

Served two ways:

- mounted at ``/mcp`` by the notebook server (``make notebook`` → ``http://127.0.0.1:8765/mcp``)
- standalone: ``python -m dclab_rnd.mcp_server --port 8777`` → ``http://127.0.0.1:8777/mcp``

The transport is streamable HTTP, stateless, JSON responses, so a plain POST of a
JSON-RPC message works too. Deterministic code still owns every split, metric and
selection rule; the client's model only proposes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dclab_rnd.agentic.catalog import ROOT
from dclab_rnd.intern.tools import Toolbox
from dclab_rnd.studio import ProjectStore

INSTRUCTIONS = (
    "DCLab notebook tools. Build a model from a table with evidence: write the prediction contract first "
    "(propose_contract → set_contract), run the stages in order (run_stage or run_all), read each record before "
    "the next step, cite the record IDs the notes give you (search_evidence / get_record), and never rerun the "
    "final stage to chase a score: it consumes the holdout once. Research evidence is not production approval."
)


def build_server(toolbox: Toolbox):
    """A low-level MCP server whose tool list is the toolbox's own JSON schemas."""
    from mcp.server.lowlevel import Server
    import mcp.types as types

    server = Server("dclab", instructions=INSTRUCTIONS)
    specs = {s["function"]["name"]: s["function"] for s in toolbox.schemas()}

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [types.Tool(name=name, description=spec["description"], inputSchema=spec["parameters"]) for name, spec in specs.items()]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        result = await asyncio.to_thread(toolbox.call, name, arguments or {})
        return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]

    return server


def session_manager(toolbox: Toolbox, *, json_response: bool = True):
    """One stateless streamable-HTTP manager; mount ``manager.handle_request`` and run ``manager.run()`` in the lifespan."""
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    return StreamableHTTPSessionManager(app=build_server(toolbox), json_response=json_response, stateless=True)


def available() -> bool:
    try:
        import mcp  # noqa: F401
    except ImportError:
        return False
    return True


def standalone_app(home: Path):
    """A minimal Starlette app serving only ``/mcp`` (for clients that do not need the notebook UI)."""
    from starlette.applications import Starlette
    from starlette.routing import Mount

    manager = session_manager(Toolbox(ProjectStore(home)))

    @asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    return Starlette(routes=[Mount("/mcp", app=manager.handle_request)], lifespan=lifespan)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=int(os.environ.get("DCLAB_MCP_PORT", "8777")))
    parser.add_argument("--home", type=Path, default=Path(os.environ.get("DCLAB_STUDIO_HOME") or ROOT / "agent_runs" / "projects"))
    args = parser.parse_args(argv)
    if not available():
        print("The MCP server needs the optional package: pip install mcp", file=sys.stderr)
        return 1
    import uvicorn

    print(f"DCLab MCP server on http://127.0.0.1:{args.port}/mcp (projects in {args.home})")
    uvicorn.run(standalone_app(args.home), host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
