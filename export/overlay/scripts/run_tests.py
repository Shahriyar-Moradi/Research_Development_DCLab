#!/usr/bin/env python3
"""Run the test suite: each tests/test_*.py file in its own process (with its own temporary folder), several at a time.

    python scripts/run_tests.py                 # every file, as many at a time as there are CPUs
    python scripts/run_tests.py test_studio     # only the files named (with or without .py)
    python scripts/run_tests.py --jobs 2

No model is ever called (DCLAB_NO_LIVE_MODELS=1). Tests that need PostgreSQL run when DCLAB_TEST_DATABASE_URL points at
a database that exists for tests (it is emptied), and are skipped otherwise. Exit status 1 when a file failed; a failing
file's output is printed whole.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(path: Path) -> tuple[Path, int, float, str]:
    started = time.time()
    with tempfile.TemporaryDirectory(prefix=f"{path.stem}-") as tmp:
        env = {**os.environ, "TMPDIR": tmp, "HOME": tmp, "DCLAB_NO_LIVE_MODELS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "DCLAB_DATABASE_URL"):
            env.pop(key, None)  # a developer's own settings never reach a test
        done = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "tests", "-p", path.name],
                              cwd=ROOT, env=env, capture_output=True, text=True)
    return path, done.returncode, time.time() - started, done.stdout + done.stderr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args(argv)
    files = sorted((ROOT / "tests").glob("test_*.py"))
    if args.names:
        wanted = {n.removesuffix(".py") for n in args.names}
        files = [f for f in files if f.stem in wanted]
        missing = wanted - {f.stem for f in files}
        if missing:
            print(f"no such test file: {', '.join(sorted(missing))}")
            return 2
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for path, code, seconds, output in pool.map(run, files):
            last = next((line for line in reversed(output.splitlines()) if line.startswith(("OK", "FAILED"))), "")
            print(f"  {'ok  ' if code == 0 else 'FAIL'} {path.name:<40} {seconds:6.1f}s  {last}", flush=True)
            if code != 0:
                failed.append((path, output))
    for path, output in failed:
        print(f"\n===== {path.name}\n{output}")
    print(f"\n{len(files) - len(failed)} of {len(files)} test files passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
