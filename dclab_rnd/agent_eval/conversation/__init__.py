"""The understanding conversation measured (package 20.2): written conversations, one per family, dev and sealed.

Each conversation is a problem sentence, the table's column names and the person's answers keyed by the card field
they answer (``conversations_dev.json``, ``conversations_sealed.json``; the sealed file's SHA-256 is pinned in
``sealed.sha256``: changed, it is refused). :func:`measure` plays each through ``dclab_core.conversation.converse``
with an arm's reader (``rules``: the rules alone; ``reader``: the conversation reader role, a model behind its
validator, with the rules as fallback) and scores it: the right card (the family, and the labels where the
conversation says them), ready, in at most five questions.

The gate (``GATE``, fixed before any run): the rules reach the right card on every sealed conversation; a model role
is on by default only when, on the sealed set, it reaches at least as many right cards as the rules and never one
the rules got right wrong. The sealed set is run once per arm (``python -m dclab_rnd.agent_eval.conversation``);
nothing is tuned on it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "evidence" / "campaigns" / "conversation_v1" / "results"
GATE = {
    "version": "conversation_gate_v1",
    "split": "sealed",
    "questions_at_most": 5,
    "rules": "every conversation reaches the right card, ready",
    "reader": "at least as many right cards as the rules, and none the rules got right turned wrong",
}
# v1 failed on the sealed set (CONV-002: "the time until each customer cancels" read as a regression); that set is
# spent. The rule was made general on dev, and a new sealed set, written before any run on it, is judged by v2: the
# same criteria, a new version, never the old set again.
GATE_V2 = {**GATE, "version": "conversation_gate_v2", "split": "sealed_v2"}
GATES = {"sealed": GATE, "sealed_v2": GATE_V2, "dev": GATE_V2}
SEALED = {"sealed": "sealed.sha256", "sealed_v2": "sealed_v2.sha256"}


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(split: str) -> list[dict[str, Any]]:
    path = HERE / f"conversations_{split}.json"
    if split in SEALED and fingerprint(path) != (HERE / SEALED[split]).read_text(encoding="utf-8").strip():
        raise ValueError("the sealed conversations changed since they were sealed: a sealed set is never edited")
    return json.loads(path.read_text(encoding="utf-8"))


def score(case: dict[str, Any], played: dict[str, Any]) -> dict[str, Any]:
    gold, got = case["gold"], played["summary"]
    right_family = got["family"] == gold["family"]
    right_labels = "labels" not in gold or got["labels"] == gold["labels"]
    right = right_family and right_labels and got["ready"] and len(played["asked"]) <= GATE["questions_at_most"]
    return {"id": case["id"], "family": case["family"], "right": bool(right), "asked": played["asked"], "questions": len(played["asked"]),
            "got": {"family": got["family"], "labels": got["labels"], "ready": got["ready"]},
            "why": None if right else ("family" if not right_family else "labels" if not right_labels else
                                       "not ready" if not got["ready"] else "too many questions")}


def measure(split: str, reader: Optional[Callable[..., dict[str, str]]] = None) -> list[dict[str, Any]]:
    from dclab_core.conversation import converse

    return [score(c, converse(c["problem"], c["answers"], c.get("columns"), c.get("data"), reader=reader, kinds=c.get("kinds")))
            for c in load(split)]


def verdict(rows: list[dict[str, Any]], split: str, baseline: Optional[list[dict[str, Any]]] = None) -> str:
    right = sum(r["right"] for r in rows)
    if baseline is None:  # the rules arm
        return ("pass" if right == len(rows) else "fail") + ("" if split in SEALED else " (dev: the gate is read on the sealed set)")
    base = {r["id"]: r["right"] for r in baseline}
    turned = [r["id"] for r in rows if base.get(r["id"]) and not r["right"]]
    ok = right >= sum(base.values()) and not turned
    return ("pass" if ok else "fail") + ("" if split in SEALED else " (dev: the gate is read on the sealed set)")


def sealed_runs(arm: str = "*", split: str = "sealed") -> list[Path]:
    """The stored runs of a sealed set, by any arm unless one is named: a set run once is spent for every arm (a
    later arm would be compared with rules changed since)."""
    return sorted(RESULTS.glob(f"CONV-*_{split}_{arm}.json")) if RESULTS.exists() else []


def store(rows: list[dict[str, Any]], split: str, arm: str, result: str, spent_eur: float = 0.0,
          roles: Optional[list[dict[str, Any]]] = None) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in RESULTS.glob("CONV-*.json")] or [0])
    ident = f"CONV-{n:03d}"
    doc = {"id": ident, "campaign_id": "conversation_v1", "kind": "understanding_conversation", "split": split, "arm": arm,
           "conversations": fingerprint(HERE / f"conversations_{split}.json"),  # 21.4: runs compare only on the same set
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "gate": GATES.get(split, GATE), "spent_eur": spent_eur, "rows": rows,
           "summary": {"conversations": len(rows), "right": sum(r["right"] for r in rows),
                       "questions_mean": round(sum(r["questions"] for r in rows) / max(len(rows), 1), 2)},
           "verdict": result, "status": "completed", "experiment_id": ident,
           # 21.4: which model answered for each role, at which prompt version, and what it cost
           "roles": roles or [{"role": "rules", "model": None, "tier": None, "prompt": None, "requests": 0, "cost_eur": 0.0}],
           "rule": f"Scores on {len(rows)} written conversations say how DCLab's understanding conversation handles these; they are "
                   "not a claim about any other conversation or about production readiness."}
    path = RESULTS / f"{ident}_{split}_{arm}.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def runs() -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("CONV-*.json"))] if RESULTS.exists() else []


def by_model(results: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]]:
    """Each role's score and cost by model (21.4): one row per run and role, newest last."""
    out = []
    for r in results if results is not None else runs():
        summary = r.get("summary") or {}
        for role in r.get("roles") or [{"role": "rules"}]:
            out.append({"run": r.get("id"), "split": r.get("split"), "conversations": r.get("conversations"), "arm": r.get("arm"),
                        "role": role.get("role"), "model": role.get("model"),
                        "tier": role.get("tier"), "prompt": role.get("prompt"), "right": summary.get("right"), "of": summary.get("conversations"),
                        "cost_eur": role.get("cost_eur"), "requests": role.get("requests")})
    return out


def measured(role: str, prompt: Optional[str], models: list[str], results: Optional[list[dict[str, Any]]] = None) -> dict[str, float]:
    """The share of right cards of each model for ``role`` at ``prompt``, on one set of conversations every model was run
    on (the same split and the same file, by its fingerprint): each model's mean over all its runs there, so a rerun
    cannot pick its luck. Models run on different conversations are not compared (empty)."""
    rows = [x for x in by_model(results) if x["role"] == role and x["model"] in models and x["prompt"] == prompt and x["of"]
            and x.get("conversations")]
    sets = sorted({(x["split"], x["conversations"]) for x in rows}, key=lambda k: (not str(k[0]).startswith("sealed"), str(k)))
    for split, which in sets:
        here = [x for x in rows if (x["split"], x["conversations"]) == (split, which)]
        shares: dict[str, list[float]] = {}
        for x in here:
            shares.setdefault(x["model"], []).append(x["right"] / x["of"])
        if set(models) <= set(shares):
            return {m: round(sum(shares[m]) / len(shares[m]), 4) for m in models}
    return {}


__all__ = ["GATE", "GATE_V2", "GATES", "load", "measure", "score", "verdict", "store", "sealed_runs", "runs", "by_model", "measured"]
