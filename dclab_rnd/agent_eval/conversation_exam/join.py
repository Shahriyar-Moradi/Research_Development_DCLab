"""Join the runs of one practice split into one record: ``python -m dclab_rnd.agent_eval.conversation_exam.join``.

A run can stop before it has played everyone (the cap, a full disk, a model account out of credit: CEX-020 and CEX-025). The
people it never played are played in a later run. This joins them, so the gate can be read once over everyone:

    python -m dclab_rnd.agent_eval.conversation_exam.join --split dev_route CEX-025 CEX-026

For every person and arm, the first run (in the order given) whose conversation was played and judged valid supplies it; a person
or arm with no such conversation refuses the join (nothing is guessed). The conversations are scored and summarised by the exam's
own ``score`` and ``summarize``, the gap and the gate preview as in ``measure``, and the record names the run each conversation
came from. A new result file is written (a result file is never edited); the benchmark page then marks the joined runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def pick(runs: Sequence[tuple[str, Mapping[str, Any]]], order: Sequence[str], arms: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The conversations of the joined record, person by person and arm by arm, and which run each came from.

    ``runs``: (run id, result document) in the order of preference; ``order``: the people, in the split's order."""
    found: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for run, doc in runs:
        for conv in doc.get("conversations") or []:
            key = (str(conv.get("arm")), str(conv.get("id")))
            if key in found:
                continue
            if (conv.get("judged") or {}).get("valid") and conv.get("transcript"):
                found[key] = (run, dict(conv))
    chosen, source, missing = [], {}, []
    for pid in order:
        for arm in arms:
            if (arm, pid) not in found:
                missing.append(f"{arm}/{pid}")
                continue
            run, conv = found[(arm, pid)]
            chosen.append(conv)
            source[f"{arm}/{pid}"] = run
    if missing:
        raise ValueError("no judged conversation in these runs for " + ", ".join(missing))
    return chosen, source


def main(argv: Optional[list[str]] = None) -> int:
    from . import RESULTS, _gate, load, next_id, score, summarize

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="the runs to join, in the order of preference (CEX-025 CEX-026)")
    ap.add_argument("--split", default="dev_route")
    ap.add_argument("--id", default=None, help="the joined record's id (default: the next one)")
    args = ap.parse_args(argv)
    docs = []
    for run in args.runs:
        path = RESULTS / f"{run}_{args.split}.json"
        if not path.exists():
            print(f"refused: no result {path.name}", file=sys.stderr)
            return 2
        docs.append((run, json.loads(path.read_text(encoding="utf-8"))))
    arms = list(docs[0][1]["arms"])
    people = {p["id"]: p for p in load(args.split)}
    played = {str(c["id"]) for _, d in docs for c in d.get("conversations") or []}
    order = [pid for pid in people if pid in played]
    try:
        chosen, source = pick(docs, order, arms)
    except ValueError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    scores: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    for conv in chosen:
        scores[conv["arm"]].append(score(conv, people[str(conv["id"])]))
    summary = {a: summarize(rows) for a, rows in scores.items()}
    result: dict[str, Any] = {"conversations": chosen, "scores": scores, "summary": summary, "incomplete": None}
    if "case" in scores:
        result["gap"] = {f"case_minus_{o}": _gate.gap(scores["case"], scores[o]) for o in ("plain", "home", "safe") if o in scores}
        result["gate_preview"] = _gate.verdict(scores, lead="case", agreement=None)  # a practice preview: the gate is read once on a sealed set
    spent = round(sum(sum((c.get("exam_spent_eur") or {}).values()) for c in chosen), 4)
    run_id = args.id or next_id()
    record = {"id": run_id, "split": args.split, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "arms": arms,
              "models": docs[0][1].get("models"), "spent_eur": spent, "cap_eur": None, "kept": docs[0][1].get("kept"),
              "joined": {"why": "a run stopped before it had played everyone; the people it never played were played in a later run, "
                                "and the first judged conversation of each person and arm is taken, once",
                         "runs": list(args.runs), "source": source},
              "gate_for_37_10": docs[0][1].get("gate_for_37_10"), **result}
    out = RESULTS / f"{run_id}_{args.split}.json"
    if out.exists():
        print(f"refused: {out.name} exists (a result file is never edited)", file=sys.stderr)
        return 2
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(out.name), "spent_eur": spent, "people": len(order), "from": sorted(set(source.values())),
                      "summary": summary, "gap": result.get("gap"), "gate_preview": result.get("gate_preview")}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
