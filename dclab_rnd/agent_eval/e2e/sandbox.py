"""A locked box for code a model wrote: macOS sandbox-exec with no network and no writes outside the run's folder, a
clean environment (no keys) and a time limit. Fails closed: without sandbox-exec, nothing runs."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SANDBOX = "/usr/bin/sandbox-exec"
TIMEOUT = 180
SHOWN = 6000  # characters of output returned to the model


def profile(folder: Path) -> str:
    here = os.path.realpath(folder)
    return ("(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n"
            f'(allow file-write* (subpath "{here}") (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr") (subpath "/dev/fd"))\n')


def run_python(code: str, folder: Path, step: int, timeout: int = TIMEOUT) -> dict[str, Any]:
    folder = Path(folder)
    if not Path(SANDBOX).is_file():
        return {"error": "no locked box on this machine (sandbox-exec is missing): nothing was run"}
    script = folder / f"step_{step:03d}.py"
    script.write_text(code, encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(folder), "TMPDIR": str(folder), "MPLCONFIGDIR": str(folder),
           "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}  # no key ever reaches the box
    start = time.monotonic()
    try:
        done = subprocess.run([SANDBOX, "-p", profile(folder), sys.executable, "-I", "-B", script.name], cwd=folder, env=env,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"stopped after {timeout} seconds", "seconds": timeout}
    return {"returncode": done.returncode, "stdout": done.stdout[-SHOWN:], "stderr": done.stderr[-SHOWN:],
            "seconds": round(time.monotonic() - start, 1)}
