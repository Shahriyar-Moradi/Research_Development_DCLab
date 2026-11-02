"""A sealed exam's vault, sealed and listed: ``python -m dclab_rnd.agent_eval.conversation_exam.vault seal|verify|all``.

A sealed exam keeps every conversation and every model call whole (with the model's thinking where it thought) in a folder of its
own, ``<workspace>/exam_vault/<run>_<split>/`` (37.15), outside everything the store turns into training sets. This writes the
vault's own ``MANIFEST.json`` when a run ends: what it holds (counts only), every file with its SHA-256, the commit and settings that
produced it, and its use: ``eval_only`` until the owner releases it. The permanent archive (``make archive``) keeps the vault whole
inside the workspace copy of each snapshot, checksummed; this manifest lets anyone check it is still what the run wrote.

The owner's labels are written into ``labels/`` after the run, so that folder is not part of the listing; everything else is, and
``verify`` reports a file that changed or went missing. No story is read or printed here: counts and hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

MANIFEST = "MANIFEST.json"
WHY = ("a sealed exam: written people played by a model and judged by a model, run once. Kept whole for evaluation and for the "
       "record; never part of a training set until the owner releases it (a released set can no longer test a model trained on it)")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def listing(folder: Path) -> list[dict[str, Any]]:
    """Every file of the vault but the manifest, lock files and the owner's labels, with its size and SHA-256."""
    out = []
    for path in sorted(Path(folder).rglob("*")):
        rel = path.relative_to(folder)
        if not path.is_file() or path.name in (MANIFEST,) or path.name.endswith(".lock") or rel.parts[0] == "labels":
            continue
        out.append({"path": str(rel), "bytes": path.stat().st_size, "sha256": _sha(path)})
    return out


def counts(folder: Path) -> dict[str, int]:
    """What the vault holds, as numbers: conversations, model calls kept whole, and how many of those carry the model's thinking."""
    folder = Path(folder)
    calls = whole = thinking = 0
    for path in folder.glob("model_exchanges/**/*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except ValueError:
                continue
            calls += 1
            whole += bool(doc.get("whole"))
            thinking += bool(doc.get("thinking"))
    return {"conversations": len(list(folder.glob("conversations/*/*/conversation.json"))), "model_calls": calls, "model_calls_whole": whole,
            "model_calls_with_thinking": thinking}


def head() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10, check=True).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def seal(folder: Path, run: str, split: str, info: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Write the vault's manifest once; a vault already sealed is verified and its manifest returned unchanged."""
    folder = Path(folder)
    path = folder / MANIFEST
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    files = listing(folder)
    manifest = {"run": run, "split": split, "sealed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "use": "eval_only",
                "released_to_training": None, "why": WHY, "counts": counts(folder), "bytes": sum(f["bytes"] for f in files),
                **(info or {}), "files": files}
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def verify(folder: Path) -> list[str]:
    """What no longer matches the manifest (empty when the vault is as it was sealed; the labels folder may have grown)."""
    folder = Path(folder)
    path = folder / MANIFEST
    if not path.exists():
        return [f"{folder.name} has no {MANIFEST}"]
    problems = []
    for entry in json.loads(path.read_text(encoding="utf-8"))["files"]:
        target = folder / entry["path"]
        if not target.is_file():
            problems.append(f"missing: {entry['path']}")
        elif _sha(target) != entry["sha256"]:
            problems.append(f"changed: {entry['path']}")
    return problems


def _split_of(folder: Path) -> tuple[str, str]:
    run, _, split = folder.name.partition("_")
    return run, split


def main(argv: Optional[list[str]] = None, root: Optional[Path] = None) -> int:
    from dclab_rnd.settings import Settings

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("seal", "verify", "all"), help="all: seal every vault that has no manifest yet, then verify them all")
    ap.add_argument("folder", nargs="?", help="a vault folder (seal, verify)")
    ap.add_argument("--commit", default=None, help="seal, all: the commit that produced the run (never guessed from today's HEAD)")
    ap.add_argument("--note", default=None, help="seal, all: a line for the manifest (spend, cap, effort, why it was sealed afterwards)")
    args = ap.parse_args(argv)
    base = Path(root) if root else Path(Settings().agent_home) / "exam_vault"
    targets = [Path(args.folder)] if args.folder else sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    if not targets:
        print("no vault", file=sys.stderr)
        return 2
    bad = 0
    for folder in targets:
        run, split = _split_of(folder)
        if args.action in ("seal", "all") and not (folder / MANIFEST).exists():
            m = seal(folder, run, split, {"sealed_after_the_run": True, "commit": args.commit, "note": args.note})
            print(f"sealed {folder.name}: {m['counts']} {m['bytes']:,} bytes")
        problems = verify(folder)
        bad += bool(problems)
        print(f"{folder.name}: " + ("as it was sealed" if not problems else "; ".join(problems[:5])))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
