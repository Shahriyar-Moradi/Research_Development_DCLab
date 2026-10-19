"""What the product calls before it removes anything (32.4): the store keeps it first (the owner, 2026-10-20: "keep data even
they remove project").

``before_removal(home, what, ident, by)`` runs the keepers on the workspace in ``home`` (only that one), so everything the
deletion or the retention sweep will remove is already in the store, then records the deletion itself under
``raw/deletions/`` (what, its id, who, when, how many entries the store holds of that workspace). It never raises: the
removal it precedes goes ahead whatever happens here, and a failure is printed and written to the deletion record.

``DCLAB_STORE_CAPTURE=0`` turns it off (the tests do, unless a test turns it on with its own ``DCLAB_STORE``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional


def enabled() -> bool:
    return os.environ.get("DCLAB_STORE_CAPTURE", "1") != "0"


def keep_before(home: Optional[Path], what: str, ident: Optional[str], getter: Any = None) -> None:
    """What a store's ``delete`` calls first: the document (``getter()``, a database row's document) and the workspace's files kept,
    then the deletion recorded. Never raises."""
    if not enabled() or home is None:
        return
    doc = None
    if getter is not None:
        try:
            doc = getter()
        except Exception:  # noqa: BLE001 - nothing to read: the files are kept all the same
            doc = None
    before_removal(home, what, ident, doc=doc)


def before_removal(home: Optional[Path], what: str, ident: Optional[str], by: str = "person", layout: Any = None,
                   doc: Any = None) -> Optional[dict[str, Any]]:
    """Keep the workspace's records, then record the removal (with ``doc``, the removed document itself when it lived in a
    database row); the deletion record, or None when capture is off or no home is known."""
    if not enabled() or home is None:
        return None
    from .conductor import conduct
    from .keepers import standard
    from .layout import Layout
    from .store import Store, now

    layout = layout or Layout()
    home = Path(home)
    from .keepers.workspaces import base_of, of

    base = base_of(home)  # the workspaces' common folder: the workspace has the same name here as when the conductor keeps it
    record: dict[str, Any] = {"what": what, "id": ident, "by": by, "at": now(), "workspace_folder": home.name, "kept_first": None, "error": None,
                              "doc": doc}
    try:
        run = conduct({"agent_runs": base, "only": home}, layout, standard.install({}))
        record["kept_first"] = {"run": run["id"], "kept": sum(s["kept"] for s in run["steps"]), "failed": run["failed"]}
    except Exception as exc:  # noqa: BLE001 - the removal goes ahead; the record says what could not be kept
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"store: could not keep {what} {ident} before its removal: {record['error']}", file=sys.stderr, flush=True)
    try:
        name = of(home).name
        Store(layout).keep("deletions", name, f"deletions/{what}-{ident or 'all'}-{record['at']}.json",
                           json.dumps(record, ensure_ascii=False, indent=1, default=str).encode(), consent="train")
    except Exception as exc:  # noqa: BLE001
        print(f"store: could not record the removal of {what} {ident}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    return record


__all__ = ["before_removal", "keep_before", "enabled"]
