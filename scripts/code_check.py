"""Package 20.9's check: every kind the engine runs, exported as code, reproduces its holdout number in a clean
Python 3.12 (``make code-check``).

For each kind: its replica is built into a project, every stage run by the engine (no model asked), the code exported
(``dclab_rnd.studio.code``), then ``python -I pipeline.py`` run from an unrelated folder in a fresh ``uv venv`` that
holds only the pinned requirements the exports name (no dclab_rnd, no repository on the path). Prints one JSON line per
kind and a summary; exit 0 when every kind reproduced its number, 75 when it could not run here (no uv, no Python 3.12,
no network for the dependencies), 1 otherwise.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SKIP = 75
OFFLINE = re.compile(r"(?i)(failed to fetch|could not connect|dns error|network|connection (refused|reset))")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def engine_kinds() -> tuple[list[str], list[str]]:
    """The kinds the engine runs: those with a replica, and those without one (which this check cannot cover)."""
    from dclab_rnd import kinds

    run = [k for k in kinds.all_kinds() if getattr(k, "runs_in_engine", True)]
    return [k.key for k in run if k.case(0) is not None], [k.key for k in run if k.case(0) is None]


def export_all(keys: list[str], into: Path) -> dict[str, dict]:
    from dclab_rnd import kinds
    from dclab_rnd.expansion import runner
    from dclab_rnd.kinds import acceptance
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import code, engine

    out = {}
    for key in keys:
        store, pid = acceptance.build(kinds.get(key).case(0), into / "projects" / key)
        with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.run_all(store, pid)
        out[key] = code.export(store, pid, into / "code" / key)
    return out


def main(argv: list[str]) -> int:
    uv = shutil.which("uv")
    if uv is None:
        print(json.dumps({"status": "skipped", "reason": "uv is not installed"}))
        return SKIP
    keys, uncovered = (argv, []) if argv else engine_kinds()
    if uncovered:  # "every kind" must mean every kind: one without a replica is a failure of this check, said
        print(json.dumps({"status": "failed", "reason": "kinds the engine runs without a replica to check", "kinds": uncovered}))
        return 1
    with tempfile.TemporaryDirectory(prefix="dclab-code-check-") as tmp:
        tmp = Path(tmp)
        exported = export_all(keys, tmp)
        wanted = sorted({line.strip() for key in keys for line in (tmp / "code" / key / "requirements.txt").read_text().splitlines() if line.strip()})
        venv = _run([uv, "venv", "--python", "3.12", str(tmp / "venv")])
        if venv.returncode:
            text = venv.stdout + venv.stderr
            reason = "no Python 3.12 for uv" if "No interpreter found" in text else "uv venv failed"
            print(json.dumps({"status": "skipped" if reason.startswith("no") else "failed", "reason": reason, "log": text[-600:]}))
            return SKIP if reason.startswith("no") else 1
        python = tmp / "venv" / "bin" / "python"
        installed = _run([uv, "pip", "install", "--python", str(python), *wanted])
        if installed.returncode:
            text = installed.stdout + installed.stderr
            print(json.dumps({"status": "skipped" if OFFLINE.search(text) else "failed", "step": "uv pip install", "log": text[-1200:]}))
            return SKIP if OFFLINE.search(text) else 1
        (tmp / "elsewhere").mkdir()
        results, failed = [], 0
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
        for key in keys:
            started = time.perf_counter()
            run = _run([str(python), "-I", str(tmp / "code" / key / "pipeline.py")], cwd=tmp / "elsewhere", env=env, timeout=1800)
            line = (run.stdout.strip().splitlines() or ["{}"])[-1]
            try:
                got = json.loads(line)
            except json.JSONDecodeError:
                got = {}
            ok = run.returncode == 0 and got.get("same") is True
            failed += not ok
            results.append({"kind": key, "metric": exported[key]["metric"], "recorded": exported[key]["value"], "reproduced": got.get("value"),
                            "same": ok, "seconds": round(time.perf_counter() - started, 1), **({} if ok else {"log": (run.stdout + run.stderr)[-1500:]})})
            print(json.dumps(results[-1]), flush=True)
        version = _run([str(python), "-c", "import sys; print(sys.version.split()[0])"]).stdout.strip()
        print(json.dumps({"status": "passed" if not failed else "failed", "kinds": len(keys), "reproduced": len(keys) - failed,
                          "python": version, "requirements": wanted}))
        return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("DCLAB_NO_LIVE_MODELS", "1")
    sys.exit(main(sys.argv[1:]))
