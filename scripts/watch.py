"""DCLab Watch's two viewers on this Mac (B4 of the 2026-11-03 plan): MLflow always on, Langfuse when wanted.

    python scripts/watch.py mlflow-up|mlflow-down|langfuse-up|langfuse-down|status

- MLflow: ``uvx mlflow server`` on 127.0.0.1:5000, its database and artifacts in ``<workspace>/mlflow/`` (git-ignored with the
  workspace), its process id in ``<workspace>/mlflow/server.pid``. Light: one Python process.
- Langfuse: the official Docker Compose, cloned once into ``deploy/langfuse/upstream`` (git-ignored; the commit is recorded in
  ``deploy/langfuse/UPSTREAM_COMMIT``), with ``docker-compose.override.yml`` (127.0.0.1 only, its PostgreSQL on 5433, log
  rotation) and ``deploy/langfuse/.env.langfuse`` (secrets generated here once, mode 600, git-ignored; it also creates the
  local organisation, project, API keys and the owner's sign-in, all printed once by ``status``). Docker Desktop must run.

``status`` says what runs and prints the lines that switch DCLab's export on (``DCLAB_TRACE_EXPORT`` and the keys).
Nothing here is started by a test or by the server; the owner runs it.
"""

from __future__ import annotations

import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGFUSE = ROOT / "deploy" / "langfuse"
UPSTREAM = LANGFUSE / "upstream"
ENV = LANGFUSE / ".env.langfuse"
REPO = "https://github.com/langfuse/langfuse.git"


def home() -> Path:
    sys.path.insert(0, str(ROOT))
    from dclab_rnd.settings import Settings

    return Path(Settings.load().agent_home)


def up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:  # noqa: S310 - a local address
            return r.status < 500
    except Exception:  # noqa: BLE001
        return False


def mlflow_up() -> int:
    folder = home() / "mlflow"
    folder.mkdir(parents=True, exist_ok=True)
    if up("http://127.0.0.1:5000/health"):
        print("MLflow already runs at http://127.0.0.1:5000")
        return 0
    uvx = shutil.which("uvx") or str(Path.home() / ".local" / "bin" / "uvx")
    log = open(folder / "server.log", "a")
    proc = subprocess.Popen([uvx, "mlflow", "server", "--host", "127.0.0.1", "--port", "5000",
                             "--backend-store-uri", f"sqlite:///{folder / 'mlflow.db'}", "--default-artifact-root", str(folder / "artifacts")],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (folder / "server.pid").write_text(str(proc.pid))
    for _ in range(120):
        if up("http://127.0.0.1:5000/health"):
            print("MLflow runs at http://127.0.0.1:5000 (first start downloads it with uvx)")
            return 0
        time.sleep(2)
    print(f"MLflow did not answer in time: see {folder / 'server.log'}")
    return 1


def mlflow_down() -> int:
    pid_file = home() / "mlflow" / "server.pid"
    if not pid_file.exists():
        print("MLflow is not running (no pid file)")
        return 0
    try:
        os.killpg(int(pid_file.read_text()), signal.SIGTERM)
    except (ProcessLookupError, ValueError):
        pass
    pid_file.unlink(missing_ok=True)
    print("MLflow stopped")
    return 0


def _secrets() -> None:
    if ENV.exists():
        return
    pk, sk = f"pk-lf-{secrets.token_hex(8)}", f"sk-lf-{secrets.token_hex(16)}"
    values = {
        "NEXTAUTH_SECRET": secrets.token_hex(32), "SALT": secrets.token_hex(16), "ENCRYPTION_KEY": secrets.token_hex(32),
        "POSTGRES_PASSWORD": secrets.token_hex(16), "CLICKHOUSE_PASSWORD": secrets.token_hex(16), "REDIS_AUTH": secrets.token_hex(16),
        "MINIO_ROOT_PASSWORD": secrets.token_hex(16),
        "LANGFUSE_INIT_ORG_ID": "dclab", "LANGFUSE_INIT_ORG_NAME": "DCLab", "LANGFUSE_INIT_PROJECT_ID": "dclab-watch",
        "LANGFUSE_INIT_PROJECT_NAME": "DCLab Watch", "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": pk, "LANGFUSE_INIT_PROJECT_SECRET_KEY": sk,
        "LANGFUSE_INIT_USER_EMAIL": "owner@dclab.local", "LANGFUSE_INIT_USER_NAME": "Owner",
        "LANGFUSE_INIT_USER_PASSWORD": secrets.token_urlsafe(18),
    }
    ENV.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    os.chmod(ENV, 0o600)


def _compose(*args: str) -> list[str]:
    return ["docker", "compose", "--project-name", "dclab-langfuse", "--env-file", str(ENV), "-f", str(UPSTREAM / "docker-compose.yml"),
            "-f", str(LANGFUSE / "docker-compose.override.yml"), *args]


def langfuse_up() -> int:
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        print("Docker is not running: start Docker Desktop, then run this again")
        return 2
    if not UPSTREAM.exists():
        subprocess.run(["git", "clone", "--depth=1", REPO, str(UPSTREAM)], check=True)
        sha = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        (LANGFUSE / "UPSTREAM_COMMIT").write_text(sha + "\n")
    _secrets()
    code = subprocess.run(_compose("up", "-d")).returncode
    if code == 0:
        print("Langfuse starts at http://127.0.0.1:3000 (the first start pulls its images: several GB, a few minutes)")
    return code


def langfuse_down() -> int:
    if not UPSTREAM.exists():
        print("Langfuse was never set up here")
        return 0
    return subprocess.run(_compose("stop")).returncode


def status() -> int:
    print(f"MLflow   {'runs' if up('http://127.0.0.1:5000/health') else 'off'}  http://127.0.0.1:5000")
    print(f"Langfuse {'runs' if up('http://127.0.0.1:3000/api/public/health') else 'off'}  http://127.0.0.1:3000")
    targets = [t for t, ok in (("mlflow", up("http://127.0.0.1:5000/health")), ("langfuse", up("http://127.0.0.1:3000/api/public/health"))) if ok]
    print("\nTo send DCLab's traces (put these in .env, then restart the server):")
    print(f"DCLAB_TRACE_EXPORT={','.join(targets) or 'mlflow'}")
    if ENV.exists():
        env = dict(line.split("=", 1) for line in ENV.read_text().splitlines() if "=" in line)
        print(f"LANGFUSE_PUBLIC_KEY={env['LANGFUSE_INIT_PROJECT_PUBLIC_KEY']}")
        print(f"LANGFUSE_SECRET_KEY=<in {ENV.relative_to(ROOT)}: LANGFUSE_INIT_PROJECT_SECRET_KEY>")
        print(f"Langfuse sign-in: {env['LANGFUSE_INIT_USER_EMAIL']}, password in {ENV.relative_to(ROOT)} (LANGFUSE_INIT_USER_PASSWORD)")
    return 0


ACTIONS = {"mlflow-up": mlflow_up, "mlflow-down": mlflow_down, "langfuse-up": langfuse_up, "langfuse-down": langfuse_down, "status": status}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ACTIONS:
        print(__doc__)
        sys.exit(2)
    sys.exit(ACTIONS[sys.argv[1]]())
