"""The ML-solver's benchmark (27.6): any answerer on verified invented problems, scored against their planted truth.

The cases are the self-play examples of the lake's dev or sealed split (never train) for the families whose truth a
replica plants: ``problem_type`` (the task), ``leakage_judgment`` (the planted leak and the identifier kept out; any
other column flagged is counted, not scored) and ``split_design`` (the time column, when the replica has one). The truth
is the replica's own facts (``kind.case(seed).facts``), never DCLab's answer, so the rules are measured too.

Answerers: ``rules`` (DCLab's verified answer, the reference), ``configured`` (the gateway's model for the purpose
``solver_answer``, today Claude, capped by ``--cap-eur``), ``tuned`` (the owner's fine-tuned model behind the tuned tier).
Each result is stored once as ``evidence/campaigns/solver_v1/results/SLV-NNN.json``; the sealed split is read once per
answerer and model (a second read is refused).

    python -m dclab_rnd.lake benchmark --answerer rules|configured|tuned [--split dev|sealed] [--limit 120] [--cap-eur 2] [--yes]
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import LAKE, ROOT, VERSION

CAMPAIGN = ROOT / "evidence" / "campaigns" / "solver_v1"
FAMILIES = ("problem_type", "leakage_judgment", "split_design")
RULE = ("A measurement on verified invented problems (replicas of each engine kind, solved by DCLab's engine and kept only "
        "when every acceptance decision passed), scored against the traps the replicas plant. It says how an answerer reads "
        "these kinds of problems, not how it does on real ones, and never that a model is ready for production.")
MANIFEST = {"campaign_id": "solver_v1",
            "objective": "Package 27.6: the ML-solver benchmark. An answerer (DCLab's rules, the configured model, the owner's tuned model) "
                         "answers the problem type, the leakage judgment and the split design of verified invented problems; each answer is "
                         "scored against the replica's planted facts, with Wilson intervals, cost and the share answered.",
            "rule": RULE, "command": "python -m dclab_rnd.lake benchmark --answerer rules|configured|tuned [--split dev|sealed]"}


def wilson(k: int, n: int, z: float = 1.96) -> list[float | None]:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def cases(split: str, lake: Path = LAKE, limit: int | None = None) -> list[dict[str, Any]]:
    out = []
    for family in FAMILIES:
        path = lake / "gold" / "solver" / family / VERSION / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
            row = json.loads(line)
            if row["provenance"]["origin"] == "verified_synthetic":
                out.append(row)
    out.sort(key=lambda r: r["example_id"])
    return out[:limit] if limit else out


_TRUTHS: dict[str, dict[str, Any]] = {}


def truth(row: dict[str, Any]) -> dict[str, Any]:
    """The replica's planted facts for a case, in the naming the case shows (masked names when it was masked)."""
    from .. import kinds
    from .selfplay import _columns

    _, kind_key, seed = row["provenance"]["group"].split(":")
    masked = "col_01" in row["messages"][1]["content"]
    key = f"{kind_key}:{seed}:{masked}"
    if key not in _TRUTHS:
        kind = next(k for k in kinds.all_kinds() if k.key == kind_key)
        case = kind.case(int(seed))
        _, alias = _columns(case.table, case.target, masked)
        f = case.facts
        name = lambda c: alias.get(c, c) if c else None  # noqa: E731
        _TRUTHS[key] = {"task": case.task, "leak": [name(c) for c in ([f["leak"]] if isinstance(f.get("leak"), str) else f.get("leaks") or []) if c],
                        "identifier": [name(c) for c in ([f["identifier"]] if isinstance(f.get("identifier"), str) else f.get("identifiers") or []) if c],
                        "time_column": name(f.get("time_column"))}
    return _TRUTHS[key]


def _json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)  # 27.9: a reasoning model's answer follows its thinking
    match = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(match.group(0)) if match else None
    except ValueError:
        return None


def score(row: dict[str, Any], answer: dict[str, Any] | None) -> dict[str, Any]:
    family, t = row["family"], truth(row)
    if answer is None:
        return {"family": family, "answered": False, "right": False}
    if family == "problem_type":
        return {"family": family, "answered": True, "right": str(answer.get("task")) == t["task"]}
    if family == "leakage_judgment":
        out = {str(f.get("column") if isinstance(f, dict) else f) for f in answer.get("forbidden") or []} | {str(c) for c in answer.get("identifiers") or []}
        planted = set(t["leak"]) | set(t["identifier"])
        return {"family": family, "answered": True, "right": planted <= out, "leak_caught": set(t["leak"]) <= out,
                "identifier_caught": set(t["identifier"]) <= out, "extra": len(out - planted)}
    if family == "split_design":
        if not t["time_column"]:
            return {"family": family, "answered": True, "right": not answer.get("time_column"), "no_time": True}
        return {"family": family, "answered": True, "right": str(answer.get("time_column")) == t["time_column"]}
    return {"family": family, "answered": True, "right": False}


def rules_answerer() -> tuple[Callable[[list[dict[str, str]]], dict[str, Any] | None], dict[str, Any]]:
    return (lambda messages: _json(messages[2]["content"])), {"name": "rules", "model": "DCLab engine (verified)"}


def model_answerer(tier: str, cap_eur: float | None) -> tuple[Callable[[list[dict[str, str]]], dict[str, Any] | None], dict[str, Any]]:
    from ..models import settings
    from ..models.gateway import Gateway

    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    gateway = Gateway()
    if tier == "tuned":
        from types import SimpleNamespace

        info = settings.public(settings.TUNED)
        if not info.get("model"):
            raise SystemExit("the tuned tier is not set (DCLAB_TIER_TUNED_BASE_URL and DCLAB_TIER_TUNED_MODEL)")
        doc = {"purposes": {"solver_answer": {"serve": settings.TUNED, "serve_model": info["model"], "serve_endpoint": info["endpoint"]}}}
        gateway.routing = SimpleNamespace(get=lambda: doc)
    served = gateway.route("solver_answer")[0]
    client = gateway.client("solver_answer")
    if client is None:
        raise SystemExit("no model serves solver_answer: configure one (or use --answerer rules)")
    if cap_eur is not None:
        client.run_limit_eur = cap_eur

    def ask(messages: list[dict[str, str]]) -> dict[str, Any] | None:
        try:
            out = client.complete(messages[:2], None, 1500)
        except Exception:  # noqa: BLE001 — a refused or failed request is an unanswered case, never a stop
            return None
        return _json(str(out.get("content") or ""))

    return ask, {"name": tier, "tier": served, "model": getattr(client, "model", None), "_client": client}


def run(answerer: str, split: str = "dev", limit: int | None = 120, cap_eur: float | None = None, store: bool = True,
        lake: Path = LAKE) -> dict[str, Any]:
    if split not in ("dev", "sealed"):
        raise ValueError("the benchmark reads dev or sealed, never train")
    ask, who = rules_answerer() if answerer == "rules" else model_answerer(answerer, cap_eur)
    if split == "sealed" and store:
        for path in sorted((CAMPAIGN / "results").glob("SLV-*.json")) if (CAMPAIGN / "results").is_dir() else []:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc.get("split") == "sealed" and doc.get("answerer", {}).get("model") == who.get("model") and doc.get("answerer", {}).get("name") == who["name"]:
                raise SystemExit(f"the sealed split was read for this answerer already ({path.name}): a sealed set is read once")
    client = who.pop("_client", None)
    rows = cases(split, lake, limit)
    scored = [score(r, ask(r["messages"])) for r in rows]
    spent = round(float(getattr(client, "spent_eur", 0.0) or 0.0), 4) if client is not None else 0.0
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in scored:
        by[s["family"]].append(s)
    families = {}
    for family, items in sorted(by.items()):
        n, k = len(items), sum(1 for s in items if s["right"])
        families[family] = {"n": n, "right": k, "accuracy": round(k / n, 4) if n else None, "interval": wilson(k, n),
                            "answered": sum(1 for s in items if s["answered"]),
                            **({"leak_caught": sum(1 for s in items if s.get("leak_caught")), "extra_flags": sum(s.get("extra", 0) for s in items)}
                               if family == "leakage_judgment" else {})}
    doc = {"campaign_id": "solver_v1", "kind": "solver_benchmark", "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "answerer": who, "split": split, "cases": len(rows), "problems": len({r["provenance"]["group"] for r in rows}), "families": families,
           "cap_eur": cap_eur, "spent_eur": spent, "status": "completed", "rule": RULE}
    if store and rows:
        results = CAMPAIGN / "results"
        results.mkdir(parents=True, exist_ok=True)
        if not (CAMPAIGN / "manifest.json").exists():
            (CAMPAIGN / "manifest.json").write_text(json.dumps(MANIFEST, indent=2) + "\n", encoding="utf-8")
        n = 1 + max([int(p.stem.split("-")[1]) for p in results.glob("SLV-*.json")] or [0])
        doc = {"id": f"SLV-{n:03d}", "experiment_id": f"SLV-{n:03d}", **doc}
        (results / f"SLV-{n:03d}.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc
