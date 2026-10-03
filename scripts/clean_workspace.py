#!/usr/bin/env python3
"""Remove leftover clutter from a local checkout: caches and empty folders.

After files move, git keeps the old folders on disk whenever they still hold
untracked caches (``__pycache__``, ``.ipynb_checkpoints``, ``.DS_Store``). This
removes that clutter and the folders it leaves empty.

    python scripts/clean_workspace.py            # dry run: show what would be removed
    python scripts/clean_workspace.py --apply    # remove it
    make clean / make clean APPLY=1

Never removes: tracked files, `.git`, `.env`, virtual environments (`.venv*`),
`data/` (including `data/downloads/`), `agent_runs/` or training runs. Other
untracked files are only listed, so you decide what happens to them.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUNK_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
JUNK_FILES = {".DS_Store", "Thumbs.db"}
JUNK_SUFFIXES = (".pyc", ".pyo")
PROTECTED = {".git", ".venv", ".venv-agent", "data", "agent_runs", "node_modules", ".env"}


def protected(path: Path) -> bool:
    rel = path.relative_to(ROOT).parts
    return bool(rel) and (rel[0] in PROTECTED or rel[0].startswith(".venv") or "runs" in rel)


def tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return set(out.splitlines())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--apply", action="store_true", help="actually remove (default: dry run)")
    args = parser.parse_args(argv)
    files_in_git = tracked()
    junk: list[Path] = []
    for path in sorted(ROOT.rglob("*")):
        if protected(path) or any(part in JUNK_DIRS for part in path.relative_to(ROOT).parts[:-1]):
            continue
        if (path.is_dir() and path.name in JUNK_DIRS) or (path.is_file() and (path.name in JUNK_FILES or path.suffix in JUNK_SUFFIXES)):
            junk.append(path)
    for path in junk:
        print(("removing " if args.apply else "would remove ") + str(path.relative_to(ROOT)))
        if args.apply:
            shutil.rmtree(path) if path.is_dir() else path.unlink()

    # Empty folders, deepest first, so a chain of empty parents disappears in one run.
    empty = []
    for path in sorted((p for p in ROOT.rglob("*") if p.is_dir() and not protected(p)), key=lambda p: -len(p.parts)):
        if not path.exists():
            continue
        contents = [c for c in path.iterdir() if not (args.apply is False and c in junk)]
        remaining = [c for c in contents if c not in empty and not (c.is_dir() and c.name in JUNK_DIRS and not args.apply)]
        if not remaining:
            empty.append(path)
            print(("removing empty " if args.apply else "would remove empty ") + str(path.relative_to(ROOT)) + "/")
            if args.apply:
                path.rmdir()

    untracked = [p for p in sorted(ROOT.rglob("*")) if p.is_file() and not protected(p) and p not in junk
                 and not any(part in JUNK_DIRS for part in p.relative_to(ROOT).parts)
                 and p.relative_to(ROOT).as_posix() not in files_in_git
                 and not p.relative_to(ROOT).as_posix().startswith("research/llm-fine-tuning/experiments/sft/runs/")]
    ignored = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT, input="\n".join(
        p.relative_to(ROOT).as_posix() for p in untracked), capture_output=True, text=True).stdout.split()
    untracked = [p for p in untracked if p.relative_to(ROOT).as_posix() not in set(ignored)]
    if untracked:
        print("\nUntracked files kept for you to decide (commit, move or delete them yourself):")
        for p in untracked:
            print("  " + p.relative_to(ROOT).as_posix())
    print(f"\n{len(junk)} cache item(s) and {len(empty)} empty folder(s) "
          + ("removed." if args.apply else "would be removed. Run with --apply to remove them."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
