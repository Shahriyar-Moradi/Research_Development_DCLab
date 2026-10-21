"""The `dclab` command: start the studio, choose a model, check the setup (the open edition's front door, 2026-10-21).

    dclab start [--port 8765] [--host 127.0.0.1] [--no-browser]   the studio (server and its job worker) in your browser
    dclab init                        choose your model: Ollama on this machine, any OpenAI-compatible URL, Claude, or none
    dclab doctor                      check Python, storage, notebook cells and the model (a completion and a tool call)
    dclab models status               which agents run on your model and which on their rules
    dclab models trust <role|all>     let the model serve tool-using agents without the doctor's probe
    dclab migrate                     create or upgrade the PostgreSQL schema (when DCLAB_DATABASE_URL is set)
    dclab users add|passwd|list ...   accounts (with PostgreSQL and DCLAB_AUTH=password)
    dclab worker                      a job worker of its own (for a server started with DCLAB_WORKER=external)
    dclab version

Settings come from the environment, then `./.env`, then `~/.config/dclabai/.env` (what `dclab init` writes, outside the
workspace folder so a workspace backup never holds a key). A key is never printed.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .extensions import PACKAGE

CONFIG_DIR = Path(os.environ.get("DCLAB_CONFIG_DIR") or Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "dclabai")
CONFIG_ENV = CONFIG_DIR / ".env"
OLLAMA = "http://localhost:11434"
TOOL_MODELS = ("qwen", "llama3.1", "llama3.2", "llama3.3", "mistral", "command-r", "firefunction", "hermes", "granite", "gpt-oss")


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is a dependency; without it the environment alone is read
        return
    load_dotenv(Path.cwd() / ".env", override=False)
    load_dotenv(CONFIG_ENV, override=False)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("dclabai")
    except Exception:  # noqa: BLE001 - run from a checkout
        from . import __version__

        return str(__version__)


# ---------------------------------------------------------------------- start


def start(host: str, port: int, browser: bool) -> int:
    import uvicorn

    if os.environ.get("DCLAB_DATABASE_URL"):
        migrate()
    url = f"http://{host}:{port}"
    if browser:
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()
    print(f"dclab {_version()}: the studio at {url} (Ctrl-C stops it). Workspace: {_home()}")
    uvicorn.run(f"{PACKAGE}.agentic.server:app", host=host, port=port, log_level="info")
    return 0


def _home() -> Path:
    from .settings import Settings

    return Path(Settings.load().agent_home)


def migrate() -> int:
    if not os.environ.get("DCLAB_DATABASE_URL"):
        print("No DCLAB_DATABASE_URL: the studio keeps its records in files; nothing to migrate.")
        return 0
    from .storage import db

    db.upgrade(os.environ["DCLAB_DATABASE_URL"])
    print("The PostgreSQL schema is up to date.")
    return 0


# ---------------------------------------------------------------------- init


def _get(url: str, timeout: float = 1.5) -> Optional[dict[str, Any]]:
    try:
        import httpx

        r = httpx.get(url, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:  # noqa: BLE001 - nothing there
        return None


def ollama_models() -> list[str]:
    tags = _get(f"{OLLAMA}/api/tags") or {}
    return [m.get("name") for m in tags.get("models") or [] if m.get("name")]


def _ask(question: str, default: str = "") -> str:
    answer = input(f"{question}{f' [{default}]' if default else ''}: ").strip()
    return answer or default


def _write_env(values: dict[str, str]) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    keep = []
    if CONFIG_ENV.exists():  # every line that is not a model setting stays
        model_keys = ("DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "OPENAI_API_KEY", "OPENAI_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL")
        keep = [line for line in CONFIG_ENV.read_text(encoding="utf-8").splitlines() if not line.split("=", 1)[0].strip() in model_keys]
    lines = keep + [f"{k}={v}" for k, v in values.items() if v]
    fd = os.open(CONFIG_ENV, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # private from the first byte, never readable for a moment
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")
    os.chmod(CONFIG_ENV, 0o600)  # a file that existed before keeps its old mode through os.open
    return CONFIG_ENV


def init() -> int:
    print("dclab init: choose the model the agents use (you can change it any time).")
    found = ollama_models()
    if found:
        print(f"Ollama is running here with {len(found)} model(s): {', '.join(found[:8])}")
    print("  1. Ollama on this machine (local, open models; no key)\n  2. Any OpenAI-compatible endpoint (URL, key, model)\n"
          "  3. Claude (an Anthropic key)\n  4. No model: every agent runs on its rules")
    choice = _ask("Your choice", "1" if found else "4")
    values: dict[str, str] = {}
    if choice == "1":
        suggested = next((m for m in found if m.split(":")[0].startswith(TOOL_MODELS)), found[0] if found else "qwen2.5:7b")
        if not found:
            print(f"Ollama is not running at {OLLAMA}: install it (https://ollama.com), then `ollama pull {suggested}`.")
        values = {"DCLAB_LLM_BASE_URL": f"{OLLAMA}/v1", "OPENAI_MODEL": _ask("Model (one that calls tools natively)", suggested)}
    elif choice == "2":
        values = {"DCLAB_LLM_BASE_URL": _ask("Base URL (ending in /v1)", "https://api.openai.com/v1"),
                  "OPENAI_API_KEY": getpass.getpass("API key (not shown): ").strip(), "OPENAI_MODEL": _ask("Model name")}
    elif choice == "3":
        values = {"DCLAB_LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": getpass.getpass("Anthropic key (not shown): ").strip(),
                  "ANTHROPIC_MODEL": _ask("Claude model name")}
    path = _write_env(values)
    print(f"Saved to {path} (only you can read it).")
    if values and _ask("Check it now with `dclab doctor`? (y/n)", "y").lower().startswith("y"):
        for key, value in values.items():
            os.environ[key] = value
        return doctor()
    return 0


# ---------------------------------------------------------------------- doctor

PROBE_TOOL = {"type": "function", "function": {"name": "record_answer", "description": "Record the answer to the question.",
                                               "parameters": {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}}}


def probe_model() -> dict[str, Any]:
    """A plain completion and one tool call through the gateway, the way the agents ask (a few tokens)."""
    from .models.gateway import for_workspace

    gateway = for_workspace(_home())  # the workspace's own gateway: its usage log and routing, as the server's
    client = gateway.client("home_orchestrator")
    if client is None:
        return {"configured": False}
    out: dict[str, Any] = {"configured": True, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    started = time.monotonic()
    try:
        reply = client.complete([{"role": "user", "content": "Say OK."}], max_tokens=10)
        out["completion"] = bool((reply.get("content") or "").strip())
        out["seconds"] = round(time.monotonic() - started, 2)
    except Exception as exc:  # noqa: BLE001 - said, with what to fix
        out.update(completion=False, error=f"{type(exc).__name__}: {str(exc)[:200]}")
        return out
    try:
        reply = client.complete([{"role": "user", "content": "What is 2 + 3? Record the answer with the tool."}], tools=[PROBE_TOOL], max_tokens=200)
        calls = reply.get("tool_calls") or []
        out["tools"] = bool(calls and calls[0].get("name") == "record_answer")
        if not out["tools"]:
            from .agents.runtime import writes_tool_calls

            out["tools_as_text"] = writes_tool_calls(reply.get("content") or "", ["record_answer"])
    except Exception as exc:  # noqa: BLE001
        out.update(tools=False, tools_error=f"{type(exc).__name__}: {str(exc)[:200]}")
    return out


def doctor() -> int:
    ok = True

    def line(good: bool, what: str, fix: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"  {'ok ' if good else 'no '} {what}" + (f"  ->  {fix}" if fix and not good else ""))
    print(f"dclab {_version()} on Python {sys.version.split()[0]}")
    line(sys.version_info >= (3, 10), "Python 3.10 or later", "install a newer Python")
    home = _home()
    line(True, f"workspace folder: {home}")
    if os.environ.get("DCLAB_DATABASE_URL"):
        try:
            import sqlalchemy as sa

            sa.create_engine(os.environ["DCLAB_DATABASE_URL"]).connect().close()
            line(True, "PostgreSQL reachable")
        except Exception as exc:  # noqa: BLE001
            line(False, f"PostgreSQL ({type(exc).__name__})", 'pip install "dclabai[postgres]" and check DCLAB_DATABASE_URL')
    else:
        line(True, "storage: local files (set DCLAB_DATABASE_URL for PostgreSQL)")
    try:
        import lightgbm  # noqa: F401  (the engine's gradient-boosted trees: it loads OpenMP)

        line(True, "the engine's libraries load")
    except OSError as exc:  # macOS without OpenMP: "Library not loaded: @rpath/libomp.dylib"
        line(False, f"LightGBM cannot load ({str(exc).splitlines()[0][:80]})",
             "brew install libomp" if sys.platform == "darwin" else "install your system's OpenMP runtime (libgomp)")
    except ImportError:
        line(False, "LightGBM is not installed", "pip install dclabai")
    try:
        import ipykernel  # noqa: F401
        import jupyter_client  # noqa: F401

        line(True, "notebook cells: a Jupyter kernel is available")
    except ImportError:  # an optional extra: the studio runs without it
        print('  --  notebook cells: off (optional: pip install "dclabai[notebook]")')
    probe = probe_model()
    if not probe.get("configured"):
        from .models import settings as model_settings

        tier = model_settings.public("standard")
        if tier.get("key_configured") and not tier.get("model_configured"):
            line(False, f"a model endpoint is set ({tier.get('endpoint')}) but no model name", "set OPENAI_MODEL (or run `dclab init`)")
        elif tier.get("key_configured") and not tier.get("sdk_installed"):
            line(False, f"the {tier.get('provider')} client library is missing",
                 'pip install "dclabai[anthropic]"' if tier.get("provider") == "anthropic" else "pip install openai")
        else:
            print("  --  no model configured: every agent runs on its rules (`dclab init` to choose one)")
    else:
        from . import gates

        gates.record_probe({"tools": bool(probe.get("completion") and probe.get("tools")), "completion": bool(probe.get("completion")),
                            "at": probe.get("at"), **gates.probed_as()}, home)  # every outcome: a failed probe never leaves an old pass
        line(bool(probe.get("completion")), f"the model answers ({probe.get('seconds', '?')} s)", probe.get("error") or "check the URL, key and model name")
        if not probe.get("completion"):
            print(f"  --  tool-using agents: {gates.why('built_chat', home)}")
            return 1
        tools = bool(probe.get("tools"))
        line(tools, "the model calls tools natively (the agents need it)",
             "choose a model with tool calling (for example qwen2.5 or llama3.1 on Ollama); "
             + ("it wrote the call as text instead" if probe.get("tools_as_text") else probe.get("tools_error") or "it answered without the tool"))
        print(f"  --  tool-using agents: {gates.why('built_chat', home)}")
    return 0 if ok else 1


# ---------------------------------------------------------------------- models


def models_status() -> int:
    from . import gates

    home = _home()
    for role in gates.ROLES:
        print(f"  {role:22} {gates.why(role, home)}")
    probe = (gates.read(home).get("probe") or {})
    if probe:
        print(f"  (the doctor's probe at {probe.get('at')}: tools {'yes' if probe.get('tools') else 'no'})")
    return 0


def models_trust(roles: list[str]) -> int:
    from . import gates

    unknown = [r for r in roles if r != "all" and r not in gates.ROLES]
    if unknown:
        print(f"Unknown role(s): {', '.join(unknown)}. One of: all, {', '.join(gates.ROLES)}")
        return 2
    gates.trust(roles, _home())
    return models_status()


# ---------------------------------------------------------------------- main


def main(argv: Optional[list[str]] = None) -> int:
    _load_env()
    ap = argparse.ArgumentParser(prog="dclab", description="dclabai: an agentic data-science studio")
    sub = ap.add_subparsers(dest="command")
    s = sub.add_parser("start", help="the studio in your browser")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=int(os.environ.get("DCLAB_PORT") or 8765))
    s.add_argument("--no-browser", action="store_true")
    sub.add_parser("serve", help="the studio, without opening a browser")
    sub.add_parser("init", help="choose your model")
    sub.add_parser("doctor", help="check the setup")
    m = sub.add_parser("models", help="which agents use your model")
    m.add_argument("action", choices=("status", "trust"))
    m.add_argument("roles", nargs="*")
    sub.add_parser("migrate", help="create or upgrade the PostgreSQL schema")
    sub.add_parser("worker", help="a job worker of its own")
    sub.add_parser("version")
    u = sub.add_parser("users", help="accounts (add, passwd, list, workspaces)")
    u.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    if args.command in (None, "start"):
        return start(getattr(args, "host", "127.0.0.1"), getattr(args, "port", 8765), not getattr(args, "no_browser", False))
    if args.command == "serve":
        return start("127.0.0.1", int(os.environ.get("DCLAB_PORT") or 8765), False)
    if args.command == "init":
        return init()
    if args.command == "doctor":
        return doctor()
    if args.command == "models":
        return models_status() if args.action == "status" else models_trust(args.roles or ["all"])
    if args.command == "migrate":
        return migrate()
    if args.command == "worker":
        from .worker import main as worker_main

        return worker_main([])
    if args.command == "version":
        print(_version())
        return 0
    if args.command == "users":
        from .accounts.__main__ import main as accounts_main

        return accounts_main(args.rest)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
