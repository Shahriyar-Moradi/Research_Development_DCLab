"""The store on a schedule for this Mac (32.37), not only while the server's worker runs.

``python scripts/store_scheduled.py`` runs ``python -m dclab_rnd.store run`` once, at low priority, when all hold:
a whole run is due (the conductor's own rule: the last one an hour ago or more), no other store run is going, the
server's worker is not running (it runs the store itself), and at least 2 GB of disk is free; while it runs, it is stopped (SIGINT, so it records where it was) when free disk falls
under 2 GB. No live model is ever called (``DCLAB_NO_LIVE_MODELS=1``, as ``make capture``).

``--install`` writes ``~/Library/LaunchAgents/ai.dclab.store.plist`` (every 6 hours) and loads it; ``--uninstall``
unloads and removes it; ``--status`` says whether it is installed and what the last scheduled run did. The log is
``~/Library/Logs/dclab/store-scheduled.log``.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL = "ai.dclab.store"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LOG = Path.home() / "Library" / "Logs" / "dclab" / "store-scheduled.log"
EVERY_SECONDS = 6 * 3600
MIN_FREE = 2 * 1024 ** 3


def log(line: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now().isoformat(timespec='seconds')} {line}\n")


def _pgrep(pattern: str) -> bool:
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    return any(int(p) != os.getpid() for p in out if p.isdigit())


def other_run_going() -> bool:
    return _pgrep("dclab_rnd.store run")


def worker_running() -> bool:
    """The server's worker queues the store itself, at most once an hour, while it runs (conductor.sweep_due)."""
    return _pgrep("dclab_rnd.worker")


def decide(due: bool, free_bytes: int, running: bool, worker: bool = False) -> tuple[bool, str]:
    """Whether to start a run now, and why (said in the log either way)."""
    if running:
        return False, "another store run is going"
    if worker:
        return False, "the server's worker is running: it runs the store itself"
    if free_bytes < MIN_FREE:
        return False, f"only {free_bytes / 1024 ** 3:.1f} GB free (at least 2 GB needed)"
    if not due:
        return False, "the last whole run ended less than an hour ago"
    return True, "due"


def run_once() -> int:
    sys.path.insert(0, str(ROOT))
    from dclab_rnd.store import conductor

    go, why = decide(conductor.due(), shutil.disk_usage(Path.home()).free, other_run_going(), worker_running())
    if not go:
        log(f"skipped: {why}")
        return 0
    log("started")
    env = {**os.environ, "DCLAB_NO_LIVE_MODELS": "1"}
    python = ROOT / ".venv" / "bin" / "python"
    child = subprocess.Popen(["nice", "-n", "10", str(python if python.exists() else sys.executable), "-m", "dclab_rnd.store", "run"],
                             cwd=ROOT, env=env, stdout=open(LOG, "a"), stderr=subprocess.STDOUT)
    stopped = {"low_disk": False}

    def watch() -> None:
        while child.poll() is None:
            if shutil.disk_usage(Path.home()).free < MIN_FREE:
                stopped["low_disk"] = True
                child.send_signal(signal.SIGINT)
                return
            time.sleep(30)

    threading.Thread(target=watch, daemon=True).start()
    code = child.wait()
    log(f"ended: exit {code}" + (" (stopped: free disk under 2 GB)" if stopped["low_disk"] else ""))
    return code


def plist() -> dict:
    return {"Label": LABEL, "ProgramArguments": [sys.executable, str(Path(__file__).resolve())], "WorkingDirectory": str(ROOT),
            "StartInterval": EVERY_SECONDS, "RunAtLoad": False, "LowPriorityIO": True, "ProcessType": "Background",
            "StandardOutPath": str(LOG), "StandardErrorPath": str(LOG)}


def install() -> int:
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if PLIST.exists():
        subprocess.run(["launchctl", "unload", str(PLIST)], capture_output=True)
    with open(PLIST, "wb") as fh:
        plistlib.dump(plist(), fh)
    r = subprocess.run(["launchctl", "load", str(PLIST)], capture_output=True, text=True)
    print(f"installed {PLIST} (every {EVERY_SECONDS // 3600} hours)" if r.returncode == 0 else f"written {PLIST}, but launchctl said: {r.stderr.strip()}")
    return r.returncode


def uninstall() -> int:
    if PLIST.exists():
        subprocess.run(["launchctl", "unload", str(PLIST)], capture_output=True)
        PLIST.unlink()
        print(f"removed {PLIST}")
    else:
        print("not installed")
    return 0


def status() -> int:
    loaded = subprocess.run(["launchctl", "list", LABEL], capture_output=True).returncode == 0
    print(f"installed: {PLIST.exists()}, loaded: {loaded}, every {EVERY_SECONDS // 3600} hours")
    if LOG.exists():
        print("last lines of", LOG)
        print("".join(LOG.read_text(encoding="utf-8").splitlines(keepends=True)[-5:]), end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--install", action="store_true")
    g.add_argument("--uninstall", action="store_true")
    g.add_argument("--status", action="store_true")
    a = p.parse_args(argv)
    return install() if a.install else uninstall() if a.uninstall else status() if a.status else run_once()


if __name__ == "__main__":
    sys.exit(main())
