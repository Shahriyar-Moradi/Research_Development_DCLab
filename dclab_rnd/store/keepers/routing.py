"""The routing keeper (P8, 2026-10-29): the repository's own sources of routing data, kept whole in the store so the routing
collector (``store/sets/routing.py``) can join them with the workspaces' records, each with its entry and hash.

- ``home_requests/<file>``: Home's written requests (``dclab_rnd/agent_eval/home/``): the dev set (``requests_dev.json``)
  and every written set of P6 (``intent_requests_*.json``, the checker's ``*_flagged.json`` too: kept, never trained on);
- ``home_eval/<file>``: every Home eval run (``evidence/campaigns/home_v1/results/HOME-*.json``).

An exam is kept as its shape only, as Home's sealed runs are today: a sealed request file is not kept at all (its words
are an exam's), and a sealed run keeps each request's id, language and every arm's route, never the fields it quoted from
the person's words. The workspaces' own routing decisions are kept by the ledger, drafts and exchange keepers already.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ..layout import ROOT
from . import Claim, Keeper, Source

HOME = ROOT / "dclab_rnd" / "agent_eval" / "home"
RESULTS = ROOT / "evidence" / "campaigns" / "home_v1" / "results"
SHAPE_KEYS = ("track", "tracks", "exit", "suggestion", "questions", "confirmation_asked")  # an arm's route, never what it quoted


def sealed(name: str) -> bool:
    return "sealed" in name


def request_files(home: Path = HOME) -> list[Path]:
    """The written request files a training set may read: never an exam's (a name with ``sealed``)."""
    files = [home / "requests_dev.json", *sorted(home.glob("intent_requests_*.json"))]
    return [p for p in files if p.is_file() and not sealed(p.name)]


def shape_only(raw: bytes) -> bytes:
    """A sealed run as its shape: each row's case id, language, trap and every arm's route; no field, no quote, no reply."""
    try:
        doc = json.loads(raw)
    except ValueError:
        return json.dumps({"unreadable": True}).encode()
    rows = []
    for row in doc.get("rows") or []:
        observed = row.get("observed") or {}
        rows.append({"case": row.get("case"), "lang": row.get("lang"), "trap": row.get("trap"),
                     "observed": {arm: {k: (v or {}).get(k) for k in SHAPE_KEYS} for arm, v in observed.items() if isinstance(v, dict)}})
    keep = {k: doc.get(k) for k in ("id", "campaign_id", "kind", "split", "at", "arms", "plan", "spent_eur", "status", "rule")}
    return (json.dumps({**keep, "rows": rows, "shape_only": "an exam's run: its routes kept, never the person's words or the fields "
                                                           "quoted from them"}, ensure_ascii=False, indent=1) + "\n").encode()


def sources(env: Any) -> Iterator[Source]:
    if isinstance(env, dict) and env.get("only"):
        return  # a workspace's removal is not the repository's
    home = Path(env["home_requests"]) if isinstance(env, dict) and env.get("home_requests") else HOME
    results = Path(env["home_results"]) if isinstance(env, dict) and env.get("home_results") else RESULTS
    for path in request_files(home):
        yield Source("rnd", f"home_requests/{path.name}", path, consent="train",
                     note="Home's written requests: their truth assigned by code, labels of the class written")
    for path in sorted(results.glob("HOME-*.json")) if results.is_dir() else []:
        if sealed(path.name):
            yield Source("rnd", f"home_eval/{path.name}", path, consent="train", transform=shape_only,
                         note="a sealed Home run: its shape only (each request's routes), never the person's words")
        else:
            yield Source("rnd", f"home_eval/{path.name}", path, consent="train", note="a dev Home run, whole")


KEEPER = Keeper("routing", (Claim("tests", "home_requests/"), Claim("tests", "home_eval/")), sources, ("m7_routing",),
                "Home's written requests and eval runs, the repository's routing data (sealed ones as shape only)")

__all__ = ["KEEPER", "sources", "shape_only", "request_files", "HOME", "RESULTS"]
