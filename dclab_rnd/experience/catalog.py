"""The catalog of a workspace's datasets, and their lineage back to its sessions (package 19.13).

Every set ``make datasets`` built (``dclab_rnd.datasets``: ``<workspace>/datasets/<builder>/<name>/manifest.json``),
with its builder and version, when, on what consent, its rows and splits, its size on disk, and its sources counted by
kind; :func:`lineage` resolves a set's sources to the sessions (visits) they came from: an event to its visit; a
conversation, a run, a draft or a project to every visit that touched it; a decision to its conversation, run, draft
or project. The layout is ``docs/guides/EXPERIENCE_ARCHIVE.md``; ``make backup`` takes the folder with the rest of
the workspace, so a restored workspace lists the same catalog.

    python -m dclab_rnd.experience catalog [--lineage BUILDER/NAME]
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

FOLDER = "datasets"
LIMIT = 10_000_000


def _manifests(root: Path) -> list[Path]:
    """Every manifest inside the datasets folder; a folder linked from elsewhere is not the workspace's."""
    if not root.exists():
        return []
    inside = root.resolve()
    found = sorted(root.glob("*/*/manifest.json")) + sorted(root.glob("development/manifest.json"))
    return [p for p in found if inside in p.resolve().parents]


def catalog(home: Path) -> list[dict[str, Any]]:
    """Every dataset of the workspace, newest build first."""
    from dclab_core.datasets import DatasetManifest

    root = Path(home) / FOLDER
    items = []
    for path in _manifests(root):
        try:
            m = DatasetManifest.read(path.read_text(encoding="utf-8"))
            files = [f for f in path.parent.iterdir() if f.is_file()]
            size = sum(f.stat().st_size for f in files)
        except Exception:  # noqa: BLE001 - another version, a half-written one, or one being rebuilt: listed as unreadable
            items.append({"path": path.parent.relative_to(root).as_posix(), "error": "unreadable manifest"})
            continue
        items.append({"path": path.parent.relative_to(root).as_posix(), "builder": m.builder, "name": m.name, "version": m.version,
                      "built_at": m.built_at, "use": m.use, "consent": m.consent, "levels": m.levels, "rows": m.rows, "splits": m.splits, "bytes": size,
                      "sources": dict(Counter(s.kind for s in m.sources)), "files": sorted(f.name for f in files)})
    return sorted(items, key=lambda i: (i.get("built_at") or "", i["path"]), reverse=True)


def lineage(home: Path, path: str, log: Any, ledger: Any = None) -> dict[str, Any]:
    """The sessions one dataset came from: its sources resolved to visits through the session record."""
    from dclab_core.datasets import DatasetManifest

    root = (Path(home) / FOLDER).resolve()
    folder = (root / path).resolve()
    if root not in folder.parents or not (folder / "manifest.json").is_file():
        raise KeyError(path)
    try:
        m = DatasetManifest.read((folder / "manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - listed as unreadable by the catalog; no lineage to give
        raise KeyError(path) from exc
    wanted: dict[str, set[str]] = {}
    for s in m.sources:
        wanted.setdefault(s.kind, set()).add(s.id)
    if wanted.get("decision") and ledger is not None:  # a decision reaches its sessions through the narrowest thing it belonged to
        for record in ledger.records():
            if record.id in wanted["decision"]:
                kind, value = next(((k, v) for k, v in (("run", record.run), ("conversation", record.conversation), ("draft", record.draft),
                                                         ("project", record.project)) if v), (None, None))
                if kind:
                    wanted.setdefault(kind, set()).add(value)
    visits: Counter[str] = Counter()
    unresolved = set(wanted.get("event", set()))
    for event in log.read(limit=LIMIT) if log is not None else []:
        ids = event.ids
        hit = (event.id in wanted.get("event", ())) or (ids.visit in wanted.get("visit", ())) \
            or (ids.conversation and (ids.conversation in wanted.get("conversation", ()) or ids.conversation in wanted.get("draft", ()))) \
            or (ids.run and ids.run in wanted.get("run", ())) or (ids.project and ids.project in wanted.get("project", ()))
        if hit and ids.visit:
            visits[ids.visit] += 1
        unresolved.discard(event.id)
    return {"path": path, "builder": m.builder, "name": m.name, "built_at": m.built_at, "sources": dict(Counter(s.kind for s in m.sources)),
            "sessions": [{"visit": v, "events": n} for v, n in visits.most_common()], "events_not_found": len(unresolved)}


def lines(items: list[dict[str, Any]]) -> list[str]:
    out = []
    for i in items:
        if "error" in i:
            out.append(f"{i['path']:<48} {i['error']}")
            continue
        sources = ", ".join(f"{n} {k}" for k, n in sorted(i["sources"].items()))
        out.append(f"{i['path']:<48} {i['rows']:>7} rows {i['bytes'] / 1024:>9.1f} KB  v{i['version']}  {i['built_at']}  {sources}")
        out.append(f"{'':<48} consent: {i['consent']}")
        if i.get("levels"):
            out.append(f"{'':<48} rows by level: " + ", ".join(f"{n} {k}" for k, n in sorted(i["levels"].items())))
    return out


def find(home: Path, path: Optional[str]) -> Optional[dict[str, Any]]:
    return next((i for i in catalog(home) if i["path"] == path), None) if path else None


__all__ = ["catalog", "lineage", "lines", "FOLDER"]
