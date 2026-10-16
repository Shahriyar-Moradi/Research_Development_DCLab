"""25.8: where Claude, the software, or both are best, measured on planted truth (``make twin-eval``).

The measurement reads the workspace's blind twin runs on studied samples (each lane had to find the target and the leak
itself; ``compare.planted`` holds the R&D's own solution as the truth), turns every step of every run into the
twin contract's records (``steps.jsonl`` and ``expert_steps.jsonl``), and builds ``dclab_core.twin.effectiveness``:
per step, each arm's accuracy on the judged cases with its interval, the agreement, the cost and the mode of
``twin_rule_v1`` (``undecided`` below the rule's minimum of judged cases). The result is stored once per measurement
in ``evidence/campaigns/twin_v1/results/TWN-NNN.json`` with each run's holdout numbers and spend beside it: names,
counts and scores, never a row. ``--run`` starts the missing blind runs first, each under its own cap.

    python -m dclab_rnd.twin eval [--home DIR] [--run bank_marketing,hyperack] [--yes] [--cap-eur 1.5] [--no-store]
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "evidence" / "campaigns" / "twin_v1"
RESULTS = CAMPAIGN / "results"
SOURCE = "blind twin runs on studied samples, judged against the R&D's planted solution"
RULE = ("A measurement on these samples with these caps and this model: where each arm was right on the planted truth, "
        "and what it cost. It is not a claim about other problems or about production readiness; a step below the rule's "
        "minimum of judged cases stays undecided.")
MANIFEST = {
    "campaign_id": "twin_v1",
    "objective": "Phase 3 (25.8): the current system (rules, validator, standard plan, no model) and Claude as the agent with the same "
                 "tools and validator, on the same blind studied samples, step by step against the planted truth (the target and the "
                 "leak columns of the R&D's own solution), with each lane's holdout number, cost and time; and Claude's expert "
                 "record, version by version, against the system's answers. Read with dclab_core.twin.effectiveness (twin_rule_v1).",
    "rule": RULE,
    "command": "make twin-eval [ARGS='--run SAMPLE,SAMPLE --yes --cap-eur 1.5']",
}


def blind_runs(home: Path) -> list[Path]:
    """The workspace's finished blind runs on a studied sample, oldest first (only these are judged by a planted truth)."""
    out = []
    for path in sorted((Path(home) / "twin" / "runs").glob("*/run.json")):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if run.get("blind") and str(run.get("seed") or "").startswith("sample:") and (path.parent / "compare.json").exists():
            out.append(path.parent)
    return out


def records(run_dirs: list[Path]) -> list[Any]:
    """Every twin step of the runs, read back through the contract (a line that fails it is skipped, never guessed)."""
    from dclab_core.twin import TwinStep

    out = []
    for folder in run_dirs:
        for name in ("steps.jsonl", "expert_steps.jsonl"):
            path = folder / name
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        out.append(TwinStep.model_validate_json(line))
                    except ValueError:
                        continue
    return out


def run_summary(folder: Path) -> dict[str, Any]:
    """One run as the result keeps it: the sample, each lane's status, holdout score, seconds and spend, and who was right."""
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    cmp = json.loads((folder / "compare.json").read_text(encoding="utf-8"))
    model = cmp.get("model") or {}
    return {"run_id": run.get("run_id") or folder.name, "sample": str(run.get("seed") or "").split(":", 1)[-1], "model": run.get("model"),
            "status": cmp.get("status"), "holdout": cmp.get("holdout"), "seconds": cmp.get("seconds"),
            "eur": {lane: round(float((model.get(lane) or {}).get("eur") or 0.0), 4) for lane in ("workflow", "ai_first")},
            "requests": {lane: int((model.get(lane) or {}).get("requests") or 0) for lane in ("workflow", "ai_first")},
            "judged": [{k: j.get(k) for k in ("step", "winner", "differs")} for j in cmp.get("judged") or []],
            "design": [{k: v.get(k) for k in ("version", "checkpoint", "parsed", "problems")} for v in (cmp.get("design") or {}).get("versions") or []]
            if isinstance(cmp.get("design"), dict) else None}


def measure(home: Path) -> dict[str, Any]:
    from dclab_core.twin import effectiveness

    runs = blind_runs(home)
    steps = records(runs)
    matrix = effectiveness(steps, SOURCE)
    return {"runs": [run_summary(r) for r in runs], "matrix": matrix.model_dump(mode="json"), "steps": len(steps)}


def store(measured: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = CAMPAIGN / "manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps(MANIFEST, indent=2) + "\n", encoding="utf-8")
    n = 1 + max([int(p.stem.split("-")[1]) for p in RESULTS.glob("TWN-*.json")] or [0])
    ident = f"TWN-{n:03d}"
    spent = round(sum(sum(r["eur"].values()) for r in measured["runs"]), 4)
    doc = {"id": ident, "campaign_id": "twin_v1", "kind": "twin_effectiveness", "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source": SOURCE, "runs": measured["runs"], "matrix": measured["matrix"], "steps": measured["steps"], "spent_eur": spent,
           "summary": {"runs": len(measured["runs"]), "samples": sorted({r["sample"] for r in measured["runs"]}), "steps": measured["steps"],
                       "modes": {s["step"]: s.get("mode") for s in measured["matrix"]["steps"]}},
           "status": "completed", "experiment_id": ident, "rule": RULE}
    path = RESULTS / f"{ident}.json"
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def report(measured: dict[str, Any]) -> str:
    lines = [f"{len(measured['runs'])} blind runs, {measured['steps']} twin steps", ""]
    for r in measured["runs"]:
        h = r.get("holdout") or {}
        lines.append(f"  {r['sample']:<22} {h.get('metric') or '':<18} system {_f(h.get('workflow'))}  claude {_f(h.get('ai_first'))}  "
                     f"€{r['eur']['ai_first']:.2f}  " + ", ".join(f"{j['step']}:{j['winner']}" for j in r["judged"]))
    lines.append("")
    for s in measured["matrix"]["steps"]:
        lines.append(f"  {s['step']:<28} mode {s.get('mode') or '—':<18} judged {s.get('judged', 0)}")
    lines.append("")
    lines.append(RULE)
    return "\n".join(lines)


def _f(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"
