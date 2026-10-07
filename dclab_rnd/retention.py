"""Timed retention of raw uploads (package 12.5), the rule the Admin page's privacy list promises.

With ``DCLAB_RETENTION_RAW_DAYS=N`` (off when unset or 0), the raw file of every draft asset that was turned into a
cleaned table (status ready) more than N days ago is deleted, from the draft's folder and from file storage. What
stays: the cleaned table (``clean.parquet``, and the copy a project made of it), its SHA-256, the asset's record (name,
source, rows, columns, the raw file's SHA-256) and the audit. Each deletion is an audit entry of kind "retention".

A name uploaded twice holds the newest upload: that asset decides, and when it goes every asset of the name is
marked. Never deleted: a file still being processed, or one that failed (it has no cleaned table: the raw file is its
only copy, and a retry needs it); a file whose content is not the one its asset recorded (an upload being written);
a project's own table (the data it runs on, not an upload).

Each job worker sweeps once an hour on a thread of its own (never in its heartbeat). A file is moved aside in one
atomic step before it is checked and removed, so of several sweeps one deletes and audits it, and an upload that
landed under the name meanwhile is put back. A sweep stopped between marking and deleting is finished by the next.
``python -m dclab_rnd.retention [--days N] [--dry-run]`` sweeps now.
"""

from __future__ import annotations

import os
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SWEEP_SECONDS = 3600


def raw_days() -> int | None:
    raw = (os.environ.get("DCLAB_RETENTION_RAW_DAYS") or "").strip()
    try:
        days = int(raw) if raw else 0
    except ValueError:  # the server refuses to start with it (Settings.problems); the sweep stays off
        return None
    return days if days > 0 else None


def _when(value: Any) -> datetime | None:
    try:
        at = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return at if at.tzinfo else at.replace(tzinfo=timezone.utc)


def _retire(path: Path, expected: str) -> bool:
    """Delete ``path`` only when its content is ``expected``. The file is first moved aside (one atomic step, so of two
    sweeps only one gets it), then checked, then removed; an upload that landed under the same name meanwhile is put back."""
    from . import audit

    aside = path.with_name(f".retiring-{secrets.token_hex(4)}-{path.name}")
    try:
        os.replace(path, aside)
    except FileNotFoundError:
        return False
    if audit.sha256_of(aside) == expected:
        aside.unlink(missing_ok=True)
        return True
    if not path.exists():
        os.replace(aside, path)  # not ours: back where it was
    return False


def _sweep_draft(drafts: Any, draft_id: str, cutoff: datetime, days: int, dry_run: bool) -> list[dict[str, Any]]:
    from . import audit

    draft = drafts.get(draft_id)
    by_name: dict[str, list[dict[str, Any]]] = {}
    for a in draft.get("assets") or []:
        if a.get("filename"):
            by_name.setdefault(a["filename"], []).append(a)
    done = []
    for name, same in by_name.items():
        owner = same[-1]  # the file on disk is the newest upload of that name
        added = _when(owner.get("added"))
        if name == owner.get("clean_file", "clean.parquet") or owner.get("status") != "ready" or added is None or added > cutoff:
            continue  # still being processed, failed (a retry needs it; it has no cleaned table), or not old enough
        path = drafts.data_dir(draft_id) / name
        sha = audit.sha256_of(path)
        if sha is None or (owner.get("raw_sha256") and sha != owner["raw_sha256"]):
            continue  # gone already, or content that is not the newest asset's (an upload being written)
        resumed = all(a.get("raw_deleted") for a in same)  # marked by a sweep that stopped before the file went
        row = {"draft_id": draft_id, "asset": owner["id"], "filename": name, "sha256": sha, "added": owner.get("added")}
        if dry_run:
            done.append(row)
            continue
        if not resumed:
            claimed: list[bool] = []

            def mark(d, owner_id=owner["id"], name=name, sha=sha):
                now_same = [a for a in d["assets"] if a.get("filename") == name]
                if not now_same or now_same[-1]["id"] != owner_id or now_same[-1].get("status") != "ready":
                    return  # a newer upload of the name arrived: the file is its
                at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                for a in now_same:  # one file for all of them: the older ones' bytes were already replaced
                    if not a.get("raw_deleted"):
                        a.update(raw_deleted=at, raw_sha256=a.get("raw_sha256") or sha)
                        claimed.append(True)
            drafts.update(draft_id, mark)
            if not claimed:
                continue
        if not _retire(path, sha):
            continue  # another sweep took it, or a new upload: put back
        try:
            from .storage.files import files_for

            files = files_for(drafts)
            files.forget_one(files.key_of(path))  # its record and a bucket's copy; the local file is gone already
        except Exception:  # noqa: BLE001 — a file never recorded in storage
            pass
        audit.record_for(drafts, "retention", "system", "Retention rule", draft_id=draft_id, move="delete_raw_upload", status="allowed",
                         asset=owner["id"], filename=name, sha256=sha, days=days, **({"resumed": True} if resumed else {}))
        done.append(row)
    return done


def sweep(drafts: Any, days: int, now: datetime | None = None, dry_run: bool = False) -> list[dict[str, Any]]:
    """Delete the raw files older than ``days``; returns one row per file (draft, asset, file, sha256)."""
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    done: list[dict[str, Any]] = []
    for summary in drafts.list(limit=100_000):
        try:
            done += _sweep_draft(drafts, summary.get("id"), cutoff, days, dry_run)
        except (KeyError, FileNotFoundError):  # a draft deleted meanwhile: the others are still swept
            continue
    return done


def sweep_due(env: Any, state: dict[str, float], clock: float) -> list[dict[str, Any]]:
    """For the worker's retention thread: sweep when the rule is on and the last sweep was an hour ago."""
    days = raw_days()
    if days is None or clock - state.get("last", float("-inf")) < SWEEP_SECONDS:
        return []
    state["last"] = clock
    drafts = getattr(env, "drafts", None)
    return sweep(drafts, days) if drafts is not None else []


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.retention", description=__doc__.split("\n")[0])
    parser.add_argument("--home", type=Path, default=None, help="the workspace folder (default: DCLAB_AGENT_HOME)")
    parser.add_argument("--days", type=int, default=None, help="override DCLAB_RETENTION_RAW_DAYS")
    parser.add_argument("--dry-run", action="store_true", help="list what would be deleted, delete nothing")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv

    from .settings import ROOT, Settings

    load_dotenv(ROOT / ".env", override=False)
    days = args.days if args.days is not None else raw_days()
    if not days or days < 1:
        print("Retention is off: set DCLAB_RETENTION_RAW_DAYS (or --days) to a number of days.", file=sys.stderr)
        return 2
    from .agentic.pool import Pool

    pool = Pool(Settings.load(args.home))
    total = 0
    for services in [pool.default] + [pool.get(w) for w in (pool.workspace_ids()[1:] if pool.settings.auth != "none" else [])]:
        for row in sweep(services.drafts, days, dry_run=args.dry_run):
            total += 1
            print(f"{'would delete' if args.dry_run else 'deleted'} draft {row['draft_id']} {row['filename']} (added {row['added']})")
    print(f"{total} raw file{'s' if total != 1 else ''} {'older than' if args.dry_run else 'deleted, older than'} {days} days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
