"""The store on a schedule for this Mac (32.37), not only while the server's worker runs.

``python scripts/store_scheduled.py`` runs ``python -m dclab_rnd.store run`` once, at low priority, when all hold:
a whole run is due (the conductor's own rule: the last one an hour ago or more), no other store run is going, the
server's worker is not running (it runs the store itself), and at least 2 GB of disk is free; while it runs, it is stopped (SIGINT, so it records where it was) when free disk falls
under 2 GB, and paused (SIGSTOP, everything it did kept) while the swap is over 6 GB, going on once it is under 5 GB; a run paused
for 3 hours is stopped. Each pause logs the memory the store run itself holds, so a night says whose memory filled the Mac. No live model is ever called (``DCLAB_NO_LIVE_MODELS=1``, as ``make capture``).

``--install`` writes ``~/Library/LaunchAgents/ai.dclab.store.plist`` (every night at 01:30) and loads it; ``--uninstall``
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
NIGHT = {"Hour": 1, "Minute": 30}  # the owner, 2026-11-03: every night, not every 6 hours
MIN_FREE = 2 * 1024 ** 3
MAX_SWAP = 6 * 1024 ** 3  # the owner, 2026-11-03: up to 6 GB of swap is fine; the 2 GB free-disk guard still stops a run first if the swap fills the disk
RESUME_SWAP = 5 * 1024 ** 3  # a paused run goes on once the swap is back under 5 GB (a gap, so it does not pause and go on every 30 s)
MAX_PAUSE = 3 * 3600  # paused this long, the swap is not someone else's passing job: the run is stopped as before


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


def swap_used() -> int:
    """Bytes of swap in use (macOS ``sysctl vm.swapusage``); 0 where it cannot be read."""
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    import re

    m = re.search(r"used = ([\d.]+)([MG])", out)
    return int(float(m.group(1)) * (1024 ** 3 if m.group(2) == "G" else 1024 ** 2)) if m else 0


def stop_child(child: subprocess.Popen) -> None:
    """Interrupt, then terminate, then kill: a run that swaps does not answer an interrupt for many minutes (2026-11-02: ten minutes of
    SIGINT with no effect, SIGTERM ended it in twenty seconds). The store writes through atomic replaces, so the last step is safe."""
    for sig, wait in ((signal.SIGINT, 30), (signal.SIGTERM, 30), (signal.SIGKILL, 0)):
        if child.poll() is not None:
            return
        child.send_signal(sig)
        deadline = time.time() + wait
        while time.time() < deadline and child.poll() is None:
            time.sleep(1)


def rss_of(pid: int) -> int:
    """Bytes the process holds in memory now (``ps -o rss``, kilobytes); 0 where it cannot be read."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 if out.isdigit() else 0


def guard(state: dict, swap: int, free: int, now: float) -> str:
    """What the watcher does now: "stop_disk" (free disk under 2 GB: always first), "pause" (swap over 6 GB: the run is
    stopped where it is, SIGSTOP, and keeps everything it did), "resume" (paused and the swap back under 5 GB),
    "stop_swap" (paused for 3 hours: the swap is not a passing job's), or "go". 2026-11-22: the nightly runs of 11-19 and
    11-22 were stopped while a 3-hour exam or the test gates held the memory; the same run alone fits (swap about 3 GB)."""
    if free < MIN_FREE:
        return "stop_disk"
    paused = state.get("paused_at")
    if paused is None:
        if swap > MAX_SWAP:
            state["paused_at"] = now
            return "pause"
        return "go"
    if now - paused >= MAX_PAUSE:
        return "stop_swap"
    if swap < RESUME_SWAP:
        state["paused_at"] = None
        state["resumed_from"] = paused
        return "resume"
    return "wait"


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
    stopped = {"low_disk": False, "swap": False, "pauses": 0, "paused_s": 0.0, "rss_at_pause": []}

    def go_on(state: dict) -> None:
        """A paused child is never left stopped: its own exit-with-the-parent guard is a thread inside it."""
        if state.get("paused_at") is not None and child.poll() is None:
            child.send_signal(signal.SIGCONT)
            state["paused_at"] = None

    watched = {"paused_at": None}

    def watch() -> None:
        state = watched
        try:
            while child.poll() is None:
                act = guard(state, swap_used(), shutil.disk_usage(Path.home()).free, time.time())
                if act in ("stop_disk", "stop_swap"):
                    stopped["low_disk" if act == "stop_disk" else "swap"] = True
                    if state["paused_at"] is not None:
                        stopped["paused_s"] += time.time() - state["paused_at"]
                    go_on(state)  # a stopped process answers no interrupt
                    stop_child(child)
                    return
                if act == "pause":
                    rss = rss_of(child.pid)
                    child.send_signal(signal.SIGSTOP)
                    stopped["pauses"] += 1
                    stopped["rss_at_pause"].append(rss)
                    log(f"paused: swap over 6 GB; the store run holds {rss / 1024 ** 3:.1f} GB, the rest is other work (goes on under 5 GB)")
                elif act == "resume":
                    child.send_signal(signal.SIGCONT)  # first: a failing log never leaves it stopped
                    stopped["paused_s"] += time.time() - (state.pop("resumed_from", None) or time.time())
                    log("resumed: swap back under 5 GB")
                time.sleep(30)
        finally:
            go_on(state)  # the watcher ending for any reason (an error reading the swap) never leaves the run stopped

    def on_term(signum, frame):  # noqa: ARG001 - launchd or a person stopping this script: the finally below runs
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, on_term)
    threading.Thread(target=watch, daemon=True).start()
    try:
        code = child.wait()
    finally:
        go_on(watched)  # Ctrl-C or SIGTERM here while paused: the child goes on and its own guard ends it
        signal.signal(signal.SIGTERM, previous)
    note = (" (stopped: free disk under 2 GB)" if stopped["low_disk"] else "") + \
        (" (stopped: swap stayed over 5 GB for 3 hours while paused)" if stopped["swap"] else "") + \
        (f" ({stopped['pauses']} pause(s) for other work's memory, {stopped['paused_s'] / 60:.0f} min paused; the run held "
         f"{max(stopped['rss_at_pause']) / 1024 ** 3:.1f} GB at most)" if stopped["pauses"] else "")
    log(f"ended: exit {code}" + note)
    return code


def plist() -> dict:
    return {"Label": LABEL, "ProgramArguments": [sys.executable, str(Path(__file__).resolve())], "WorkingDirectory": str(ROOT),
            "StartCalendarInterval": NIGHT, "RunAtLoad": False, "LowPriorityIO": True, "ProcessType": "Background",
            "StandardOutPath": str(LOG), "StandardErrorPath": str(LOG)}


def install() -> int:
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if PLIST.exists():
        subprocess.run(["launchctl", "unload", str(PLIST)], capture_output=True)
    with open(PLIST, "wb") as fh:
        plistlib.dump(plist(), fh)
    r = subprocess.run(["launchctl", "load", str(PLIST)], capture_output=True, text=True)
    print(f"installed {PLIST} (every night at 01:30)" if r.returncode == 0 else f"written {PLIST}, but launchctl said: {r.stderr.strip()}")
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
    print(f"installed: {PLIST.exists()}, loaded: {loaded}, every night at 01:30")
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
