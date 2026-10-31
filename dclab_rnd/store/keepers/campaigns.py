"""The campaigns keeper (2026-10-31, the owner: every exam and test run is costly, so keep all of it for the seven models):
every campaign's result and label files under ``evidence/campaigns/<campaign>/``, kept in the store as
``campaigns/<campaign>/<path>``, each with its entry and hash, so a set builder can join a run's scores, verdicts and
per-case answers with the workspaces' records of the same run.

A practice (dev) run is kept whole. A sealed or test run is kept as its numbers only: ids, splits, short labels, counts,
scores and verdicts, never a text longer than a label (an exam's words stay out of every training set; its whole calls
live in the workspace's vault). Home's runs are the routing keeper's (``home_eval/``), so they are not kept twice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ..layout import ROOT
from . import Claim, Keeper, Source

CAMPAIGNS = ROOT / "evidence" / "campaigns"
PARTS = ("results", "labels")
SKIP = {("home_v1", "results")}  # kept by the routing keeper as home_eval/
SHORT = 40  # a sealed run keeps a string only as long as a label


def is_sealed(path: Path, doc: Any = None) -> bool:
    name = path.name.lower()
    split = str((doc or {}).get("split") or "").lower() if isinstance(doc, dict) else ""
    return "sealed" in name or "_test" in name or "-test" in name or "sealed" in split or split == "test"


def numbers_only(obj: Any, depth: int = 0) -> Any:
    """What a sealed run may keep: numbers, booleans and labels, never a sentence."""
    if depth > 8:
        return None
    if isinstance(obj, dict):
        out = {k: numbers_only(v, depth + 1) for k, v in obj.items()}
        return {k: v for k, v in out.items() if v is not None}
    if isinstance(obj, list):
        out = [numbers_only(v, depth + 1) for v in obj]
        return [v for v in out if v is not None]
    if isinstance(obj, str):
        return obj if len(obj) <= SHORT else None
    return obj


def _shape(raw: bytes) -> bytes:
    try:
        doc = json.loads(raw)
    except ValueError:
        return json.dumps({"unreadable": True, "shape_only": "a sealed run"}).encode()
    kept = numbers_only(doc)
    if isinstance(kept, dict):
        kept["shape_only"] = "a sealed or test run: its numbers and labels only, never its words"
    return (json.dumps(kept, ensure_ascii=False, indent=1) + "\n").encode()


def files(root: Path = CAMPAIGNS) -> list[Path]:
    out = []
    for campaign in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        for part in PARTS:
            if (campaign.name, part) in SKIP or not (campaign / part).is_dir():
                continue
            out += [p for p in sorted((campaign / part).rglob("*")) if p.is_file() and p.suffix in (".json", ".jsonl")]
        out += [p for p in sorted(campaign.glob("*.json")) if p.is_file()]  # a campaign's own manifest and benchmark
    return out


def sources(env: Any) -> Iterator[Source]:
    if isinstance(env, dict) and env.get("only"):
        return  # a workspace's removal is not the repository's
    root = Path(env["campaigns"]) if isinstance(env, dict) and env.get("campaigns") else CAMPAIGNS
    for path in files(root):
        rel = path.relative_to(root).as_posix()
        try:
            doc = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else None
        except (OSError, ValueError):
            doc = None
        if is_sealed(path, doc):
            yield Source("rnd", f"campaigns/{rel}", path, consent="train", transform=_shape,
                         note="a sealed or test run: its numbers and labels only")
        else:
            yield Source("rnd", f"campaigns/{rel}", path, consent="train", log=path.suffix == ".jsonl",
                         note="a campaign's run, whole")


KEEPER = Keeper("campaigns", (Claim("tests", "campaigns/"),), sources,
                ("m1_students", "m2_workflow_decisions", "m3_ml_decisions", "m4_workflow_llm", "m5_solver_llm", "m6_stepwise_decisions", "m7_judges"),
                "every campaign's result and label files: practice runs whole, sealed and test runs as numbers only")

__all__ = ["KEEPER", "sources", "files", "is_sealed", "numbers_only", "CAMPAIGNS"]
