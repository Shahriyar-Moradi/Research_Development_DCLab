"""Running the project's notebook cells on the Notebook page (the owner, 2026-10-20).

One Jupyter kernel per project, started on the first run of a cell and shut down when it has been idle for
:data:`IDLE_SECONDS`. The kernel works in its own folder, ``<workspace>/kernels/<project>/``, holding a copy of the
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

import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

IDLE_SECONDS = 30 * 60
TIMEOUT_SECONDS = float(os.environ.get("DCLAB_CELL_TIMEOUT", "300") or 300)
MAX_TEXT = 20_000  # characters of one output kept for the page (the code run keeps it whole)
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
    if positions is None:
        raise CellRefused("This kind of project runs its own machine; its notebook runs in your own Jupyter.")
    frame = studio_data.load_table(studio_data.data_path(store, project))
    rows = sorted(int(i) for i in positions[prepared.bundle.train_idx])
    return frame.iloc[rows].reset_index(drop=True), name


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
    def __init__(self, project_id: str, folder: Path):
        import sys

        from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
        from jupyter_client.manager import KernelManager

        class _ThisPython(KernelSpecManager):  # the server's own interpreter and libraries, never a kernel installed elsewhere
            def get_kernel_spec(self, kernel_name: str) -> KernelSpec:
                return KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], display_name="DCLab",
                                  language="python")

        self.project_id, self.folder = project_id, folder
        self.manager = KernelManager(kernel_name="dclab", kernel_spec_manager=_ThisPython())
        self.manager.start_kernel(cwd=str(folder), env=_env())
        self.client = self.manager.client()
        self.client.start_channels()
        self.client.wait_for_ready(timeout=60)
        self.used, self.count, self.lock = time.time(), 0, threading.Lock()

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
                    self.manager.interrupt_kernel()
                    outputs.append({"type": "error", "ename": "Timeout", "evalue": f"stopped after {int(timeout)} s", "traceback": []})
                    status = "error"
                    break
                try:
                    msg = self.client.get_iopub_msg(timeout=min(left, 5))
                except Exception:  # noqa: BLE001 - no message yet: wait again until the time is up
                    continue
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                kind, content = msg["msg_type"], msg["content"]
                if kind == "stream":
                    outputs.append({"type": "text", "name": content.get("name"), "text": content.get("text", "")[:MAX_TEXT]})
                elif kind in ("execute_result", "display_data"):
                    bundle = content.get("data") or {}
                    if "image/png" in bundle:
                        outputs.append({"type": "image", "png": bundle["image/png"]})
                    elif "text/plain" in bundle:
                        outputs.append({"type": "text", "name": "result", "text": str(bundle["text/plain"])[:MAX_TEXT]})
                    count = content.get("execution_count", count)
                elif kind == "error":
                    status = "error"
                    outputs.append({"type": "error", "ename": content.get("ename"), "evalue": content.get("evalue"),
                                    "traceback": [ANSI.sub("", line) for line in content.get("traceback") or []][-30:]})
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

    def folder(self, project_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", project_id or ""):
            raise CellRefused("not a project")
        return self.home / "kernels" / project_id

    def _kernel(self, store: Any, project: dict[str, Any]) -> Kernel:
        pid = project["id"]
        with self._lock:
            self._sweep()
            kernel = self._kernels.get(pid)
            if kernel is not None and kernel.manager.is_alive():
                return kernel
            folder = self.folder(pid)
            folder.mkdir(parents=True, exist_ok=True)
            frame, name = training_rows(store, project)
            for old in folder.iterdir():  # what a previous kernel left: the table is written again from the current split
                if old.is_file():
                    old.unlink()
            _write(frame, folder / name)
            kernel = self._kernels[pid] = Kernel(pid, folder)
            return kernel

    def _sweep(self) -> None:
        for pid, kernel in list(self._kernels.items()):
            if time.time() - kernel.used > IDLE_SECONDS or not kernel.manager.is_alive():
                kernel.shutdown()
                self._kernels.pop(pid, None)

    def run(self, store: Any, project: dict[str, Any], records: dict[str, Any], index: int, by: Optional[str] = None) -> dict[str, Any]:
        """Run the notebook's code cell ``index`` (as the server generates it) in the project's kernel."""
        from .. import code_runs
        from . import export

        cells = export.notebook(project, records)["cells"]
        if not 0 <= index < len(cells) or cells[index]["cell_type"] != "code":
            raise CellRefused("not a code cell of this notebook")
        cell = cells[index]
        if (cell.get("metadata") or {}).get("dclab", {}).get("role") == "holdout":
            raise CellRefused("The holdout is read once, by the workflow (WF-09), so this cell does not run here. The kernel has the "
                              "training rows only; the reliability stage scores the holdout and records it.")
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        out = self._kernel(store, project).execute(source)
        stdout = "".join(o["text"] for o in out["outputs"] if o["type"] == "text")
        error = "\n".join(f"{o['ename']}: {o['evalue']}\n" + "\n".join(o.get("traceback") or []) for o in out["outputs"] if o["type"] == "error")
        code_runs.record(f"notebook:{project['id']}:cell{index}", source, stdout, error, 0 if out["status"] == "ok" else 1, kind="notebook_cell",
                         project=project["id"], home=self.home, extra={"cell": index, "execution_count": out["execution_count"],
                                                                       "seconds": out["seconds"], "by": by,
                                                                       "pictures": sum(1 for o in out["outputs"] if o["type"] == "image")})
        return {"cell": index, **out}

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


__all__ = ["Kernels", "Kernel", "CellRefused", "training_rows", "IDLE_SECONDS", "TIMEOUT_SECONDS"]
