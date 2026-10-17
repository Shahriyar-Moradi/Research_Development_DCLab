"""The permanent record of everything captured for training DCLab's models (the owner, 2026-10-17): ``make archive``.

Everything DCLab captures that a training run will need is copied into ``training_records/`` beside this repository,
one dated snapshot at a time, organised by kind, every file with its SHA-256, nothing ever deleted by a rebuild:

- ``workspace/``: the development workspace's database (``pg_dump``) and folder (decision ledgers, experience logs,
  drafts, projects, live tests, twin runs, testers' records), through ``storage.backup`` (its own manifest of counts
  and hashes); the folder alone when no database answers;
- ``lake_bronze/``: the training lake's raw inputs, which no rebuild recreates (verified self-play, every other model's
  answers on every decision, the inputs' hashes);
- ``openml_profiles/``: the public tables' profiles (bands only; the data files were deleted after profiling);
- ``decision_sets/``: the generated decision sets (sealed partitions included, read-only here) and the workflow decisions
  from journeys.

A file unchanged since the last snapshot is a hard link to that snapshot's own copy (never to a live file, which may
grow), so a snapshot costs only what changed. A finished snapshot is made read-only. ``training_records/INDEX.json``
lists every snapshot with its counts and its manifest's hash; ``training_records/README.md`` says what is where.

    python -m dclab_rnd.archive [--to training_records] [--no-database]
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training_records"
DECISIONS = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions"
SOURCES = {  # name: folders copied as they are (the workspace goes through storage.backup)
    "lake_bronze": [ROOT / "lake" / "bronze"],
    "openml_profiles": [ROOT / "data" / "downloads" / "openml"],
    "decision_sets": [DECISIONS / "generated_v1", DECISIONS / "workflow_v1"],
}
README = """# DCLab's training records

Everything DCLab captured for training its own models, kept on this machine (the owner's choice, 2026-10-17), one dated
snapshot per `make archive`, never deleted by a rebuild. Each snapshot is read-only once written.

- `snapshots/<time>/workspace/`: the development workspace's database dump and folder (decision ledgers, experience logs,
  drafts, projects, live tests, twin runs, testers' records), with their own manifest.
- `snapshots/<time>/lake_bronze/`: the training lake's raw inputs: verified self-play, every other model's answers on every
  decision (`answers/<answerer>/<point>/<split>.jsonl`), the inputs' hashes.
- `snapshots/<time>/openml_profiles/`: public tables' profiles (bands only).
- `snapshots/<time>/decision_sets/`: the generated decision sets (their sealed partitions included: never train or tune on
  them) and the workflow decisions from journeys.
- `snapshots/<time>/MANIFEST.json`: every file with its size and SHA-256, counted by kind.
- `INDEX.json`: every snapshot, its counts and its manifest's hash.

What may train a model is said in each record (`use`, `origin`, `split`); Claude's and other hosted models' answers are
evaluation only until their terms say otherwise.
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _previous(out: Path) -> Optional[dict[str, Any]]:
    try:
        index = json.loads((out / "INDEX.json").read_text(encoding="utf-8"))
        last = index["snapshots"][-1]
        return {**last, "manifest": json.loads((out / "snapshots" / last["name"] / "MANIFEST.json").read_text(encoding="utf-8"))}
    except (OSError, ValueError, KeyError, IndexError):
        return None


def _place(src: Path, dest: Path, sha: str, before: dict[str, str], previous_root: Optional[Path], rel: str) -> str:
    """The file in the snapshot: a hard link to the last snapshot's own copy when it is the same, else a copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if previous_root is not None and before.get(rel) == sha:
        old = previous_root / rel
        try:
            os.link(old, dest)
            return "linked"
        except OSError:
            pass
    shutil.copy2(src, dest)
    return "copied"


def _freeze(folder: Path) -> None:
    for p in folder.rglob("*"):
        if p.is_file():
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def archive(out: Path = OUT, database: bool = True, sources: Optional[dict[str, list[Path]]] = None, home: Optional[Path] = None,
            now: Optional[str] = None) -> dict[str, Any]:
    out = Path(out)
    name = now or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap = out / "snapshots" / name
    if snap.exists():
        raise ValueError(f"{snap} exists: a snapshot is never written twice")
    snap.mkdir(parents=True)
    prev = _previous(out)
    before = {f["path"]: f["sha256"] for f in (prev or {}).get("manifest", {}).get("files", [])}
    previous_root = out / "snapshots" / prev["name"] if prev else None
    files, how, counts = [], {"linked": 0, "copied": 0}, {}
    for kind, folders in (sources or SOURCES).items():
        n = 0
        for folder in folders:
            folder = Path(folder)
            if not folder.exists():
                continue
            for src in sorted(p for p in folder.rglob("*") if p.is_file() and not p.name.endswith(".lock")):
                rel = f"{kind}/{folder.name}/{src.relative_to(folder)}"
                sha = _sha256(src)
                how[_place(src, snap / rel, sha, before, previous_root, rel)] += 1
                files.append({"path": rel, "bytes": src.stat().st_size, "sha256": sha})
                n += 1
        counts[kind] = n
    workspace = {"skipped": "not asked"}
    if database:
        workspace = _workspace(snap / "workspace", home)
        for p in sorted((snap / "workspace").rglob("*")) if (snap / "workspace").exists() else []:
            if p.is_file():
                files.append({"path": str(p.relative_to(snap)), "bytes": p.stat().st_size, "sha256": _sha256(p)})
        counts["workspace"] = sum(1 for f in files if f["path"].startswith("workspace/"))
    manifest = {"name": name, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "counts": counts, "placed": how,
                "workspace": workspace, "bytes": sum(f["bytes"] for f in files), "files": files,
                "rule": "Every file with its SHA-256; a file unchanged since the last snapshot is a hard link to that snapshot's copy."}
    (snap / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    _freeze(snap)
    index_path = out / "INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"snapshots": []}
    index["snapshots"].append({"name": name, "at": manifest["at"], "counts": counts, "bytes": manifest["bytes"], "placed": how,
                               "manifest_sha256": _sha256(snap / "MANIFEST.json")})
    index_path.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    return {"snapshot": str(snap), "counts": counts, "placed": how, "bytes": manifest["bytes"], "workspace": workspace}


def _workspace(folder: Path, home: Optional[Path]) -> dict[str, Any]:
    """The development workspace through storage.backup (its database and folder); its folder alone when no database answers."""
    import tarfile

    from .settings import Settings

    home = Path(home) if home is not None else Path(Settings().agent_home)
    url = os.environ.get("DCLAB_DATABASE_URL") or f"postgresql+psycopg://{os.environ.get('USER', '')}@/dclab_dev"
    try:
        from .storage.backup import backup

        made = backup(url, home, folder)  # a stamped folder under workspace/, with its own manifest of rows and hashes
        return {"database": "dumped", "folder": str(Path(made["folder"]).relative_to(folder.parent)), "rows": made.get("rows"),
                "files": made.get("files")}
    except Exception as e:  # noqa: BLE001 - no database here: the folder alone, said so
        folder.mkdir(parents=True, exist_ok=True)
        with tarfile.open(folder / "files.tar.gz", "w:gz") as tar:
            tar.add(home, arcname=home.name, filter=lambda t: None if t.name.endswith(".lock") else t)
        return {"database": f"not dumped: {type(e).__name__}: {str(e)[:200]}", "folder": "workspace/files.tar.gz"}


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.archive", description=__doc__.split("\n")[0])
    parser.add_argument("--to", type=Path, default=OUT)
    parser.add_argument("--no-database", action="store_true", help="leave the workspace's database and folder out")
    args = parser.parse_args(argv)
    got = archive(args.to, database=not args.no_database)
    print(json.dumps(got, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
