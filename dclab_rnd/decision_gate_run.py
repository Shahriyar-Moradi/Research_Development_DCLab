"""The decision classifiers judged once on their sealed decisions (package 22.3): ``make decision-gate``.

For each point, the classifier built by ``make decision-models`` (its version checked against its committed card)
answers the point's sealed decisions once; ``dclab_core.decision_gate`` (fixed before the run) judges it beside the
rule. The sealed file is read only here, and only when its contents match the fingerprint committed in the generated
sets' ``MANIFEST.json``; a point whose sealed set was judged is refused under any gate version (a later gate needs a
new sealed partition, from a new generated set), and a gate version judges every point by the same thresholds. Each
verdict is an evidence record in ``evidence/campaigns/decision_models_v1/results/DMG-NNN_<point>.json``.

    python -m dclab_rnd.decision_gate_run [--points p1,p2] [--yes]
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "generated_v1"
MODELS = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "models_v1"
RESULTS = ROOT / "evidence" / "campaigns" / "decision_models_v1" / "results"


def sealed_fingerprint(rows: list[dict[str, Any]]) -> str:
    """The fingerprint ``make decisions-gen`` commits: what the sealed decisions hold, not only which they are."""
    content = sorted(json.dumps({k: x[k] for k in ("record", "state", "label")}, sort_keys=True, default=str) for x in rows)
    return hashlib.sha256("\n".join(content).encode()).hexdigest()


def records(results: Path = RESULTS) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(results).glob("DMG-*.json"))] if Path(results).exists() else []


def spent(point: str, fingerprint: str, results: Path = RESULTS) -> bool:
    """Whether this point's sealed set (by its fingerprint) was judged, under any gate version: a sealed set is read once."""
    return any(d.get("point") == point and d.get("sealed", {}).get("fingerprint") == fingerprint for d in records(results))


def committed_fingerprint(point: str, generated: Path = GENERATED) -> str:
    return json.loads((Path(generated) / "MANIFEST.json").read_text(encoding="utf-8"))["sets"][point]["sealed_fingerprint"]


def sealed(point: str, generated: Path = GENERATED) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in (Path(generated) / "decisions" / point / "sealed.jsonl").read_text(encoding="utf-8").splitlines()]
    if sealed_fingerprint(rows) != committed_fingerprint(point, generated):
        raise ValueError(f"{point}: the sealed decisions differ from the committed fingerprint (rebuild with make decisions-gen)")
    return rows


def judge(point: str, generated: Path = GENERATED, models: Path = MODELS, results: Path = RESULTS) -> Path:
    import joblib

    from dclab_core import decision_gate

    fingerprint = committed_fingerprint(point, generated)
    if spent(point, fingerprint, results):
        raise ValueError(f"{point}: its sealed set was judged; it is spent under every gate (a new gate needs a new sealed set)")
    for d in records(results):  # one gate version, one set of thresholds, whatever was judged first
        if d["gate"]["version"] == decision_gate.VERSION and d["gate"]["thresholds"] != decision_gate.THRESHOLDS:
            raise ValueError(f"{decision_gate.VERSION}'s thresholds differ from those it judged {d['point']} by: a new gate needs a new version")
    card = json.loads((Path(models) / f"{point}.json").read_text(encoding="utf-8"))
    model = joblib.load(Path(models) / f"{point}.joblib")
    if model.version != card["version"]:
        raise ValueError(f"{point}: the fitted model ({model.version}) is not the one its card describes ({card['version']})")
    rows = [r for r in sealed(point, generated) if r.get("label") and r["label"].get("value") is not None]
    answers, errors = [], 0
    for r in rows:
        try:
            got = model.answer(point, r["state"], list(r.get("options") or []))
            answers.append(None if got is None else got.value)
        except Exception:  # noqa: BLE001 - an error is counted, never skipped
            answers.append(None)
            errors += 1
    truth = [r["label"]["value"] for r in rows]
    result = decision_gate.verdict(truth, answers, [r.get("value_used") for r in rows], [r["group"] for r in rows], errors=errors)
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in results.glob("DMG-*.json")] or [0])
    ident = f"DMG-{n:03d}"
    doc = {"id": ident, "experiment_id": ident, "campaign_id": "decision_models_v1", "kind": "decision_model_gate",
           "status": "completed", "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "point": point,
           "model": {"name": card["name"], "version": card["version"], "family": card["family"], "strength": card["strength"],
                     "trained_set": card["trained_set"]},
           "gate": {"version": decision_gate.VERSION, "thresholds": decision_gate.THRESHOLDS},
           "sealed": {"decisions": len(rows), "fingerprint": fingerprint},
           "verdict": result, "result": "pass" if result["pass"] else "fail",
           "rule": f"One run of {card['name']} on {len(rows)} sealed decisions of generated tables, judged by a gate written "
                   "before it; evidence about these planted kinds, not about real tables nor production readiness."}
    path = results / f"{ident}_{point}.json"
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.decision_gate_run", description=__doc__.split("\n")[0])
    parser.add_argument("--points", default="")
    parser.add_argument("--yes", action="store_true", help="read the sealed sets (once: they are then spent for this gate)")
    args = parser.parse_args(argv)
    points = [p for p in args.points.split(",") if p] or sorted(p.stem for p in MODELS.glob("*.json"))
    if not args.yes:
        print(f"Would judge {', '.join(points)} once on their sealed decisions; add --yes to run (each set is then spent).")
        return 0
    for point in points:
        if spent(point, committed_fingerprint(point)):
            print(f"{point:<26} already judged: its sealed set is spent")
            continue
        doc = json.loads(judge(point).read_text(encoding="utf-8"))
        v = doc["verdict"]
        print(f"{point:<26} {doc['result']:<4}  " + "  ".join(f"{c['criterion'].split(',')[0]}={c['value']}" for c in v["criteria"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
