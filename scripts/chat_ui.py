#!/usr/bin/env python3
"""Run Hugging Face's Chat UI locally with the DCLab notebook attached as an MCP server.

    make chat-ui                 # plain Chat UI + DCLab tools (any OpenAI-compatible model)
    make chat-ui-intern          # the same with ML Intern mode compiled in (needs a Hugging Face OAuth app)
    python scripts/chat_ui.py --setup-only     # clone and write .env.local, do not start

What it does:
  1. clones https://github.com/huggingface/chat-ui into .chat-ui/ (gitignored), or updates it
  2. writes .chat-ui/.env.local from this repository's .env:
       OPENAI_BASE_URL / OPENAI_API_KEY   the model endpoint (Hugging Face router by default)
       MCP_SERVERS                        -> the DCLab notebook at http://127.0.0.1:8765/mcp
       ML_ASSISTANT_MODE / MODELS         when --ml-intern is given
       OPENID_CLIENT_ID / SECRET          when set in .env (sign-in; ML Intern needs it for Jobs)
  3. runs `npm install` once and `npm run dev`

Start the notebook first (`make notebook`) so the MCP endpoint is up. Chat UI needs Node.js 20+.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/huggingface/chat-ui"
DEFAULT_MODELS = [{"id": "zai-org/GLM-5.3-Flash", "provider": "together"}, {"id": "moonshotai/Kimi-K3", "provider": "baseten"}]
MANAGED = "# ---- written by scripts/chat_ui.py (edit .env in the DCLab repository instead) ----"


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is in requirements/agent.txt
        return
    load_dotenv(ROOT / ".env", override=False)


def env_local(args: argparse.Namespace) -> tuple[str, list[str]]:
    """The .env.local text and the warnings that go with it."""
    warnings = []
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://router.huggingface.co/v1"
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN") or ""
    if not key:
        warnings.append("OPENAI_API_KEY (or HF_TOKEN) is not set in the DCLab .env: Chat UI cannot list models until you add it.")
    lines = [MANAGED, f"OPENAI_BASE_URL={base_url}", f"OPENAI_API_KEY={key}", "",
             "PUBLIC_APP_NAME=DCLab Chat", 'PUBLIC_APP_DESCRIPTION="Hugging Face Chat UI with the DCLab notebook tools"', "",
             "# The DCLab notebook as an MCP server (start it with `make notebook`)",
             "MCP_SERVERS=" + json.dumps([{"name": "DCLab notebook", "url": f"{args.dclab_url.rstrip('/')}/mcp"}]),
             "MCP_ALLOW_INSECURE_URLS=true", ""]
    client_id, client_secret = os.environ.get("OPENID_CLIENT_ID", ""), os.environ.get("OPENID_CLIENT_SECRET", "")
    if client_id and client_secret:
        lines += ["# Sign in with Hugging Face", f"OPENID_CLIENT_ID={client_id}", f"OPENID_CLIENT_SECRET={client_secret}",
                  'OPENID_SCOPES="openid profile inference-api read-mcp read-billing contribute-repos jobs"', "MCP_FORWARD_HF_USER_TOKEN=true", ""]
    elif args.ml_intern:
        warnings.append("ML Intern needs sign-in to launch Hugging Face Jobs: create an OAuth app at https://huggingface.co/settings/applications/new "
                        "(redirect http://localhost:5173/login/callback, scopes openid profile inference-api read-mcp read-billing jobs) "
                        "and put OPENID_CLIENT_ID / OPENID_CLIENT_SECRET in the DCLab .env. Without it the DCLab tools still work; Hub tools run anonymously.")
    if args.ml_intern:
        models = os.environ.get("DCLAB_CHAT_UI_MODELS") or json.dumps(DEFAULT_MODELS)
        lines += ["# ML Intern mode (build-time flag; restart the dev server after changing it)", "ML_ASSISTANT_MODE=true", f"ML_ASSISTANT_MODELS={models}", ""]
    extra = os.environ.get("DCLAB_CHAT_UI_EXTRA_ENV", "")
    if extra:
        lines += ["# DCLAB_CHAT_UI_EXTRA_ENV", *extra.split("\\n"), ""]
    return "\n".join(lines) + "\n", warnings


def run(cmd: list[str], cwd: Path) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", type=Path, default=ROOT / ".chat-ui", help="where to clone Chat UI (default .chat-ui/)")
    parser.add_argument("--ml-intern", action="store_true", help="compile ML Intern mode in")
    parser.add_argument("--setup-only", action="store_true", help="clone and write .env.local, then stop")
    parser.add_argument("--no-update", action="store_true", help="do not git pull an existing clone")
    parser.add_argument("--port", type=int, default=5173)
    parser.add_argument("--dclab-url", default=os.environ.get("DCLAB_NOTEBOOK_URL", "http://127.0.0.1:8765"))
    args = parser.parse_args(argv)
    load_env()

    for tool in ("git", "node", "npm"):
        if not shutil.which(tool):
            print(f"{tool} is required (Chat UI needs Node.js 20 or later).", file=sys.stderr)
            return 1
    major = int(subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip().lstrip("v").split(".")[0] or 0)
    if major < 20:
        print(f"Node.js 20 or later is required (found {major}).", file=sys.stderr)
        return 1

    target = args.dir
    if not (target / "package.json").exists():
        run(["git", "clone", "--depth", "1", REPO, str(target)], ROOT)
    elif not args.no_update:
        try:
            run(["git", "pull", "--ff-only"], target)
        except subprocess.CalledProcessError:
            print("Could not update the clone (offline?); using what is there.")
    text, warnings = env_local(args)
    (target / ".env.local").write_text(text, encoding="utf-8")
    print(f"wrote {target / '.env.local'}")
    for warning in warnings:
        print("warning: " + warning)
    mode = "?mode=ml-intern" if args.ml_intern else ""
    print(f"\nStart the notebook in another terminal (`make notebook`), then open http://localhost:{args.port}/{mode}")
    if args.setup_only:
        return 0
    if not (target / "node_modules").exists():
        run(["npm", "install"], target)
    run(["npm", "run", "dev", "--", "--port", str(args.port)], target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
