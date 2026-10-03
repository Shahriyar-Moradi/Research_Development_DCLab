#!/usr/bin/env python3
"""Create a new research track from research/_template.

    python scripts/new_research_track.py graph-neural-networks "Graph neural networks" --prefix GNN
    make new-track NAME=graph-neural-networks TITLE="Graph neural networks" PREFIX=GNN

Creates research/<name>/ with README.md (filled from the template), data.md and empty
experiments/, notebooks/, evaluation/ and reports/ folders. Never overwrites an existing track.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("name", help="folder name, lowercase-with-dashes, e.g. graph-neural-networks")
    parser.add_argument("title", help="human title, e.g. 'Graph neural networks'")
    parser.add_argument("--prefix", help="experiment ID prefix, e.g. GNN (default: initials of the name)")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", args.name):
        parser.error("name must be lowercase words separated by dashes")
    target = RESEARCH / args.name
    if target.exists() and any(p for p in target.iterdir() if p.name != "README.md"):
        parser.error(f"{target.relative_to(ROOT)} already has content; refusing to overwrite")
    prefix = (args.prefix or "".join(w[0] for w in args.name.split("-"))).upper()

    template = (RESEARCH / "_template" / "README.md").read_text(encoding="utf-8")
    readme = (template.replace("{Track name}", args.title)
              .replace("{TRACK}", prefix)
              .replace("{track}", args.name.replace("-", "_"))
              .replace("{YYYY-MM-DD}", date.today().isoformat())
              .replace("**Status:** planned · active · paused · concluded", "**Status:** active"))
    for folder in ("experiments", "notebooks", "evaluation", "reports"):
        (target / folder).mkdir(parents=True, exist_ok=True)
        (target / folder / ".gitkeep").touch()
    existing = target / "README.md"
    if existing.exists():
        # keep the planned-track notes below the fresh template
        readme += "\n---\n\n## Notes from the planned-track page\n\n" + existing.read_text(encoding="utf-8")
    existing.write_text(readme, encoding="utf-8")
    (target / "data.md").write_text(
        f"# Data for {args.title}\n\n| Dataset | Source URL | License | File | SHA-256 |\n|---|---|---|---|---|\n",
        encoding="utf-8",
    )
    print(f"Created {target.relative_to(ROOT)}/ (experiment prefix {prefix}). Add it to research/README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
