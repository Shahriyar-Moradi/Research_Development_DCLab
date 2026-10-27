"""DCLab as an MCP server over HTTP, so any MCP client can drive the notebook.

The tools are the intern's toolbox: the evidence index, the column auditor, the
solution, the five deterministic stages, the notebook export. Hugging
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
from pathlib import Path
from typing import Any

from dclab_rnd.agentic.catalog import ROOT
from dclab_rnd.intern.tools import Toolbox

INSTRUCTIONS = (
    "DCLab notebook tools. Build a model from a table with evidence: write the solution first "
    "(propose_solution → set_solution), run the stages in order (run_stage or run_all), read each record before "
    "the next step, cite the record IDs the notes give you (search_evidence / get_record), and never rerun the "
    "final stage to chase a score: it consumes the holdout once. Research evidence is not production approval."
)


def build_server(toolbox: Toolbox):
    """A low-level MCP server whose tool list is the registry's project tools (the intern's), read at each request."""
    from mcp.server.lowlevel import Server
    import mcp.types as types

    server = Server("dclab", instructions=INSTRUCTIONS)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        specs = [s["function"] for s in toolbox.schemas() + toolbox.registry.schemas("chat")  # A7: DCLab's agents beside the project tools
                 + toolbox.registry.schemas("workspace")]  # A7.4: the workspace chat's read tools (30.21), which no route serves
        return [types.Tool(name=spec["name"], description=spec["description"], inputSchema=spec["parameters"]) for spec in specs]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        from . import experience
        from .agents import harness  # 21.1: each call of an MCP client is a run of the "mcp" agent, recorded even when it raises

        project = (arguments or {}).get("project_id")
        result: Any = None
        error: str | None = None
        run: str | None = None
        try:
            # A7: everything the call records, and the turns it starts, came through an external chat
            with experience.via("mcp"), experience.bind(project=project if isinstance(project, str) else None), harness.session("mcp") as current:
                run = current.run_id
                with current.timed(name, arguments or {}) as out:
                    tool = toolbox.registry.get(name)
                    if tool is not None and tool.scope == "chat":  # A7: one of DCLab's agents, in the request's workspace services
                        import inspect

                        from . import chat_agents, context

                        auth = chat_agents.AUTHORIZATION.set(_authorization(server))  # the caller's API token, for the operations it runs
                        try:
                            if inspect.iscoroutinefunction(tool.handler):  # the product's operations run on this loop, as the app's own
                                out["result"] = result = await toolbox.registry.acall(name, arguments or {}, context.services(), "chat")
                            else:  # an agent's turn holds a thread, as the routes' background turns do
                                out["result"] = result = await asyncio.to_thread(toolbox.registry.call, name, arguments or {}, context.services(), "chat")
                        finally:
                            chat_agents.AUTHORIZATION.reset(auth)
                    elif tool is not None and tool.scope == "workspace":  # A7.4: the workspace chat's tables, masked as for its own agent
                        from . import context, workspace_chat

                        # A fresh session a call: an MCP chat has no question boundary, so the shown-values budget holds per call.
                        session = workspace_chat.Session(context.services())
                        out["result"] = result = await asyncio.to_thread(toolbox.registry.call, name, arguments or {}, session, "workspace")
                    else:
                        out["result"] = result = await asyncio.to_thread(toolbox.call, name, arguments or {})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _keep_call(toolbox, run, name, arguments or {}, result, error)  # 32.7: kept whether it worked or not
        return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]

    return server


def _authorization(server: Any) -> str | None:
    """The MCP request's Authorization header (an API token when accounts are on), or None."""
    try:
        request = server.request_context.request
        return request.headers.get("authorization") if request is not None else None
    except (LookupError, AttributeError):
        return None


def _keep_call(toolbox: Toolbox, run: str | None, name: str, arguments: dict[str, Any], result: Any, error: str | None = None) -> None:
    """32.7: an MCP client's call kept whole as its own source, ``<workspace>/mcp/calls.jsonl`` (the store's keepers keep it,
    the keys that quote a table's cells removed). Never a reason for the call to fail."""
    from datetime import datetime, timezone

    try:
        from .corrections import _home

        home = _home()  # the workspace of the running services (its ledger's or its experience log's folder), on files or PostgreSQL
        if home is None:
            return
        (Path(home) / "mcp").mkdir(parents=True, exist_ok=True)
        line = json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="microseconds"), "run": run, "tool": name,
                           "arguments": arguments, "result": result, "error": error}, ensure_ascii=False, default=str)
        with open(Path(home) / "mcp" / "calls.jsonl", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


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


def standalone_app(workspace: Path | None = None):
    """``/mcp`` for a client that does not need the notebook's pages (DCLab Code, Chat UI), on its own port.

    A7.1: it is the product server's app on the workspace (``create_app``), so a call here runs in the workspace's
    whole services exactly as a call to the product server's ``/mcp``: the experience log, the ledger, the usage log
    and its caps, lessons, traces and the job worker; the same start-up checks (sign-in when accounts are on), the
    same request ids and visits, and the routes DCLab's own agents use (the Home agent's build is one). The workspace's
    ``.env`` is read first, as the server and the worker read it.
    """
    _load_env()
    from . import consent
    from .agentic.server import create_app

    consent.PROJECT_TTL = 0.0  # a second process beside the product server: a consent switched there is read here at once (as the worker does)
    return create_app(Path(workspace) if workspace is not None else None)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=False)  # as the server and the worker read it: all open the same workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=int(os.environ.get("DCLAB_MCP_PORT", "8777")))
    parser.add_argument("--home", type=Path, default=None,
                        help="the workspace folder (default: DCLAB_AGENT_HOME, else the default workspace), as the worker takes it")
    args = parser.parse_args(argv)
    if not available():
        print("The MCP server needs the optional package: pip install mcp", file=sys.stderr)
        return 1
    import uvicorn

    _load_env()  # before the workspace is chosen: DCLAB_AGENT_HOME may be set there
    app = standalone_app(args.home)
    print(f"DCLab MCP server on http://127.0.0.1:{args.port}/mcp (workspace {app.state.services.store.home})")
    # Idle connections kept 75 s, not uvicorn's 5 s: an MCP client that reuses one after a longer pause (a model thinking
    # between tool calls) met a socket closed under it, ECONNRESET on the first call (DCLab Code, 2026-10-27).
    uvicorn.run(app, host="127.0.0.1", port=args.port, timeout_keep_alive=75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
