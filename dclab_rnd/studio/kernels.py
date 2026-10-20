"""Running the project's notebook cells on the Notebook page (the owner, 2026-10-20).

One Jupyter kernel per project, started on the first run of a cell and shut down when it has been idle for
:data:`IDLE_SECONDS`. The kernel works in its own folder inside the project's, ``<project>/kernel/data/`` (deleted with
the project, and counted as a table by the store, never kept as its content), holding a copy of the
project's table with **the training rows only** (the engine's split, read back to the table's own rows), under the file
name the notebook reads: the notebook's cells run as they are, and the holdout is never in the kernel. The cell that
scores the holdout (``role: holdout`` in the exported notebook) is refused: the holdout is read once, by the workflow
(WF-09). A cell runs the source the server generates for that index, never code sent by the page.

The kernel's environment carries no key, token or database address of the server. Every run is kept whole (the code,
what it printed, its error) in the workspace's code runs, the store's raw material for training (32.21). Who may run
cells: any signed-in member for now (the owner's choice); a later step keeps them to the owner and ML engineers on a
machine of their own until a container sandbox exists (``docs/guides/ROADMAP.md``).
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

IDLE_SECONDS = 30 * 60
TIMEOUT_SECONDS = float(os.environ.get("DCLAB_CELL_TIMEOUT", "300") or 300)
MAX_TEXT = 20_000  # characters of one output sent to the page (the code run keeps it whole)
MAX_TRACEBACK = 30  # lines of a traceback sent to the page
CHART = "application/vnd.dclab.chart+json"  # a cell's chart, drawn by the page in the app's own chart style
MAX_SERIES, MAX_POINTS, MAX_BARS = 8, 500, 60

# What every kernel starts with: two helpers that show a chart as data (numbers only) instead of a picture; the page draws it
HELPERS = '''
def _dclab_show(spec):
    from IPython.display import display
    display({"application/vnd.dclab.chart+json": spec, "text/plain": spec.get("title") or "chart"}, raw=True)

def chart_line(series, title="", xlabel="", ylabel="", diag=False, dots=False):
    """A line chart in the app's style. series: {name: [(x, y), ...]}, or one list of (x, y) pairs."""
    if not isinstance(series, dict):
        series = {"": series}
    _dclab_show({"kind": "line", "title": str(title), "xlabel": str(xlabel), "ylabel": str(ylabel), "diag": bool(diag), "dots": bool(dots),
                 "series": [{"name": str(n), "pts": [[float(a), float(b)] for a, b in pts]} for n, pts in series.items()]})

def chart_bars(labels, values, title="", ylabel=""):
    """A bar chart in the app's style: one bar per label."""
    _dclab_show({"kind": "bars", "title": str(title), "ylabel": str(ylabel), "labels": [str(l) for l in labels], "values": [float(v) for v in values]})
'''


def _number(v: Any) -> Optional[float]:
    import math

    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def clean_chart(spec: Any) -> Optional[dict[str, Any]]:
    """A chart a cell showed, kept only when it is what the page draws: a line or bars, bounded, numbers and short texts.
    Anything else is dropped (the page never draws what a cell merely says is a chart)."""
    if not isinstance(spec, dict) or spec.get("kind") not in ("line", "bars"):
        return None
    text = lambda v, n=80: str(v if v is not None else "")[:n]  # noqa: E731
    out: dict[str, Any] = {"kind": spec["kind"], "title": text(spec.get("title")), "ylabel": text(spec.get("ylabel"))}
    if spec["kind"] == "bars":
        labels, values = spec.get("labels"), spec.get("values")
        if not isinstance(labels, list) or not isinstance(values, list) or not labels or len(labels) != len(values) or len(labels) > MAX_BARS:
            return None
        nums = [_number(v) for v in values]
        if any(n is None for n in nums):
            return None
        return {**out, "labels": [text(l, 40) for l in labels], "values": nums}
    series = spec.get("series")
    if not isinstance(series, list) or not series or len(series) > MAX_SERIES:
        return None
    cleaned = []
    for item in series:
        pts = item.get("pts") if isinstance(item, dict) else None
        if not isinstance(pts, list) or not pts or len(pts) > MAX_POINTS:
            return None
        pairs = [[_number(a), _number(b)] for a, b in (p if isinstance(p, (list, tuple)) and len(p) == 2 else (None, None) for p in pts)]
        if any(a is None or b is None for a, b in pairs):
            return None
        cleaned.append({"name": text(item.get("name"), 40), "pts": pairs})
    return {**out, "xlabel": text(spec.get("xlabel")), "diag": bool(spec.get("diag")), "dots": bool(spec.get("dots")), "series": cleaned}
SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|DATABASE_URL|CREDENTIAL|AUTH)", re.I)
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TABLES = (".csv", ".parquet", ".xlsx", ".json")


class CellRefused(Exception):
    """A cell that may not run here, with why (the page shows it)."""


def _env() -> dict[str, str]:
    """The kernel's environment: the server's, without any key, token, password or database address."""
    return {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}


def training_rows(store: Any, project: dict[str, Any]) -> tuple[Any, str]:
    """(the project's table restricted to the engine's training rows, its file name). Refused for a project whose rows
    come from a joined label file (the notebook joins it itself: the training rows are not rows of one file)."""
    import pandas as pd

    from . import data as studio_data
    from . import engine

    if project.get("labels"):
        raise CellRefused("This project joins a label file: run its notebook in your own Jupyter (download it); cells here need one table.")
    name = str((project.get("data") or {}).get("filename") or "")
    if not name.lower().endswith(TABLES):
        raise CellRefused(f"Cells here read a CSV, Parquet, Excel or JSON table; this project's is {Path(name).suffix or 'unknown'}.")
    prepared = engine.prepare(store, project)
    positions = getattr(prepared, "source_rows", None)
    if positions is None or len(positions) != len(prepared.bundle.X):  # rows built from the table (entities, windows) are not its rows
        raise CellRefused("This kind of project runs its own machine; its notebook runs in your own Jupyter.")
    frame = studio_data.load_table(studio_data.data_path(store, project))
    rows = sorted(int(i) for i in positions[prepared.bundle.train_idx])
    return frame.iloc[rows].reset_index(drop=True), name


def fingerprint(project: dict[str, Any]) -> str:
    """What the kernel's rows depend on: the table, the solution and the settings. A kernel started on another is restarted."""
    from . import engine

    try:
        pack = engine.pack_of(project)  # the pack picks the kind, and so the strata of the split
    except Exception:  # noqa: BLE001
        pack = None
    basis = {"sha256": (project.get("data") or {}).get("sha256"), "solution": project.get("solution"), "settings": project.get("settings"),
             "labels": project.get("labels"), "pack": str(pack)}
    return hashlib.sha256(json.dumps(basis, sort_keys=True, default=str).encode()).hexdigest()


def _for_page(out: dict[str, Any]) -> dict[str, Any]:
    """The run as the page draws it: long text and tracebacks cut (the code run keeps them whole)."""
    cut = []
    for o in out["outputs"]:
        if o["type"] == "text":
            o = {**o, "text": o["text"][:MAX_TEXT]}
        elif o["type"] == "error":
            o = {**o, "traceback": (o.get("traceback") or [])[-MAX_TRACEBACK:]}
        cut.append(o)
    return {**out, "outputs": cut}


def _write(frame: Any, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix == ".xlsx":
        frame.to_excel(path, index=False)
    elif suffix == ".json":
        frame.to_json(path, orient="records")
    else:
        frame.to_csv(path, index=False)


class Kernel:
    def __init__(self, project_id: str, folder: Path, fingerprint: str = ""):
        import sys

        from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
        from jupyter_client.manager import KernelManager

        class _ThisPython(KernelSpecManager):  # the server's own interpreter and libraries, never a kernel installed elsewhere
            def get_kernel_spec(self, kernel_name: str) -> KernelSpec:
                return KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], display_name="DCLab",
                                  language="python")

        self.project_id, self.folder, self.fingerprint = project_id, folder, fingerprint
        self.manager = KernelManager(kernel_name="dclab", kernel_spec_manager=_ThisPython())
        self.manager.start_kernel(cwd=str(folder), env=_env())
        self.client = self.manager.client()
        self.client.start_channels()
        self.client.wait_for_ready(timeout=60)
        self.used, self.count, self.lock = time.time(), 0, threading.Lock()
        self.execute(HELPERS, timeout=60)  # chart_line and chart_bars: not a run of the person's, so the kernel still counts as new
        self.count = 0

    def execute(self, code: str, timeout: float = TIMEOUT_SECONDS) -> dict[str, Any]:
        """Run ``code``; its outputs as the page draws them (text, a picture, an error), its count and its time."""
        with self.lock:
            started = time.time()
            msg_id = self.client.execute(code, store_history=True)
            outputs: list[dict[str, Any]] = []
            count, status = None, "ok"
            while True:
                left = timeout - (time.time() - started)
                if left <= 0:
                    try:
                        self.manager.interrupt_kernel()
                    except Exception:  # noqa: BLE001 - a kernel already gone
                        pass
                    outputs.append({"type": "error", "ename": "Timeout", "evalue": f"stopped after {int(timeout)} s", "traceback": []})
                    status = "error"
                    break
                try:
                    msg = self.client.get_iopub_msg(timeout=min(left, 5))
                except queue.Empty:  # no message yet: wait again, unless the kernel is gone
                    if self.manager.is_alive():
                        continue
                    msg = None
                except Exception:  # noqa: BLE001 - the channels were stopped (a restart)
                    msg = None
                if msg is None:
                    outputs.append({"type": "error", "ename": "KernelStopped", "evalue": "the kernel stopped (restarted, or it ran out of memory); run the cells again",
                                    "traceback": []})
                    status = "error"
                    break
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                kind, content = msg["msg_type"], msg["content"]
                if kind == "stream":
                    outputs.append({"type": "text", "name": content.get("name"), "text": content.get("text", "")})
                elif kind in ("execute_result", "display_data"):
                    bundle = content.get("data") or {}
                    if CHART in bundle:
                        spec = clean_chart(bundle[CHART])
                        if spec:
                            outputs.append({"type": "chart", "spec": spec})
                        else:
                            outputs.append({"type": "text", "name": "chart", "text": "(a chart this page cannot draw was dropped: use chart_line or chart_bars with numbers)"})
                    elif "image/png" in bundle:
                        outputs.append({"type": "image", "png": bundle["image/png"]})
                    elif "text/plain" in bundle:
                        outputs.append({"type": "text", "name": "result", "text": str(bundle["text/plain"])})
                    count = content.get("execution_count", count)
                elif kind == "error":
                    status = "error"
                    outputs.append({"type": "error", "ename": content.get("ename"), "evalue": content.get("evalue"),
                                    "traceback": [ANSI.sub("", line) for line in content.get("traceback") or []]})
                elif kind == "execute_input":
                    count = content.get("execution_count", count)
                elif kind == "status" and content.get("execution_state") == "idle":
                    break
            self.used, self.count = time.time(), (count or self.count + 1)
            return {"outputs": outputs, "status": status, "execution_count": count, "seconds": round(time.time() - started, 2)}

    def shutdown(self) -> None:
        try:
            self.client.stop_channels()
            self.manager.shutdown_kernel(now=True)
        except Exception:  # noqa: BLE001 - a kernel already gone
            pass


class Kernels:
    """The workspace's kernels, one per project."""

    def __init__(self, home: Path):
        self.home = Path(home)
        self._kernels: dict[str, Kernel] = {}
        self._lock = threading.Lock()

    @staticmethod
    def folder(store: Any, project_id: str) -> Path:
        """The kernel's folder, inside the project's own: its delete removes it, and the store reads a table there as one."""
        if not re.fullmatch(r"[0-9a-f]{12}", project_id or ""):
            raise CellRefused("not a project")
        return Path(store.data_dir(project_id)).parent / "kernel" / "data"

    def _kernel(self, store: Any, project: dict[str, Any]) -> Kernel:
        pid, now = project["id"], fingerprint(project)
        with self._lock:
            self._sweep()
            kernel = self._kernels.get(pid)
            if kernel is not None and kernel.manager.is_alive() and kernel.fingerprint == now:
                return kernel
            if kernel is not None:  # the table, the solution or the settings changed: its rows are not the training rows any more
                kernel.shutdown()
                self._kernels.pop(pid, None)
            folder = self.folder(store, pid)
            folder.mkdir(parents=True, exist_ok=True)
            frame, name = training_rows(store, project)
            for old in folder.iterdir():  # what a previous kernel left: the table is written again from the current split
                if old.is_file():
                    old.unlink()
            _write(frame, folder / name)
            kernel = self._kernels[pid] = Kernel(pid, folder, now)
            return kernel

    def _sweep(self) -> None:
        for pid, kernel in list(self._kernels.items()):
            if time.time() - kernel.used > IDLE_SECONDS or not kernel.manager.is_alive():
                kernel.shutdown()
                self._kernels.pop(pid, None)

    @staticmethod
    def _source(project: dict[str, Any], cells: list[dict[str, Any]], index: Any) -> str:
        from . import notebook_agent

        if isinstance(index, str):  # an agent's cell the person accepted
            found = next((c for c in project.get("notebook_agent_cells") or [] if c.get("id") == index), None)
            if found is None or found.get("status") != "accepted":
                raise CellRefused("only a cell you accepted runs")
            refused = notebook_agent.problems(found["code"])
            if refused:
                raise CellRefused("; ".join(refused))
            return found["code"]
        if not 0 <= index < len(cells) or cells[index]["cell_type"] != "code":
            raise CellRefused("not a code cell of this notebook")
        cell = cells[index]
        if (cell.get("metadata") or {}).get("dclab", {}).get("role") == "holdout":
            raise CellRefused("The holdout is read once, by the workflow (WF-09), so this cell does not run here. The kernel has the "
                              "training rows only; the reliability stage scores the holdout and records it.")
        return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])

    def _record(self, project: dict[str, Any], index: Any, source: str, out: dict[str, Any], by: Optional[str]) -> None:
        from .. import code_runs

        stdout = "".join(o["text"] for o in out["outputs"] if o["type"] == "text")
        error = "\n".join(f"{o['ename']}: {o['evalue']}\n" + "\n".join(o.get("traceback") or []) for o in out["outputs"] if o["type"] == "error")
        code_runs.record(f"notebook:{project['id']}:cell{index}", source, stdout, error, 0 if out["status"] == "ok" else 1, kind="notebook_cell",
                         project=project["id"], home=self.home, extra={"cell": index, "execution_count": out["execution_count"],
                                                                       "seconds": out["seconds"], "by": by,
                                                                       "pictures": [o["png"] for o in out["outputs"] if o["type"] == "image"],
                                                                       "charts": [o["spec"] for o in out["outputs"] if o["type"] == "chart"]})

    def run(self, store: Any, project: dict[str, Any], records: dict[str, Any], index: Any, by: Optional[str] = None) -> dict[str, Any]:
        """Run the notebook's code cell ``index`` (as the server generates it), or an agent's cell the person accepted (its id,
        ``a…``), in the project's kernel. An agent's cell on a kernel that has run nothing (new, restarted, swept when idle)
        runs after the notebook's cells up to the holdout, which make the variables it reads. Every run is kept whole."""
        from . import export

        cells = export.notebook(project, records)["cells"]
        source = self._source(project, cells, index)
        kernel = self._kernel(store, project)
        prepared = 0
        if isinstance(index, str) and kernel.count == 0:
            for i, cell in enumerate(cells):
                if cell["cell_type"] != "code":
                    continue
                if ((cell.get("metadata") or {}).get("dclab") or {}).get("role") == "holdout":
                    break
                code = self._source(project, cells, i)
                first = kernel.execute(code)
                self._record(project, i, code, first, by)
                prepared += 1
                if first["status"] != "ok":
                    return {"cell": index, "prepared": prepared, **_for_page(first), "status": "error",
                            "outputs": [{"type": "error", "ename": "NotebookCellFailed", "evalue": f"the notebook's cell {i} failed before yours could run",
                                         "traceback": []}] + _for_page(first)["outputs"]}
        out = kernel.execute(source)
        self._record(project, index, source, out, by)
        return {"cell": index, "prepared": prepared, **_for_page(out)}

    def restart(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            kernel = self._kernels.pop(project_id, None)
        if kernel is not None:
            kernel.shutdown()
        return {"restarted": kernel is not None}

    def status(self, project_id: str) -> dict[str, Any]:
        kernel = self._kernels.get(project_id)
        alive = bool(kernel and kernel.manager.is_alive())
        return {"running": alive, "count": kernel.count if alive else 0, "idle_seconds": round(time.time() - kernel.used) if alive else None,
                "timeout_seconds": TIMEOUT_SECONDS, "rows": "training rows only"}

    def shutdown_all(self) -> None:
        with self._lock:
            for kernel in self._kernels.values():
                kernel.shutdown()
            self._kernels.clear()


__all__ = ["Kernels", "Kernel", "CellRefused", "training_rows", "fingerprint", "IDLE_SECONDS", "TIMEOUT_SECONDS"]
