"""Verified synthetic solving (27.3): invented problems solved by DCLab's engine, kept only when verified.

Each engine kind brings a replica case per seed (``kind.case(seed)``: a table shaped like a real kind of problem, its
traps planted, the moment and the outcome stated). The engine solves it with no model asked anything, and the kind's
acceptance decisions check the solve (``kind.accept``): the planted leak and the identifier kept out, the right split,
the right metric, a holdout read once and above its baseline. A solve that passes every check becomes examples of the
ML-solver's task families; one that fails is logged with the decisions it missed, never kept.

Every example comes twice: with the table's own column names, and with the names masked (``col_01``…), so a model
learns to read the bands and the stated moment, not to recognise a replica's names. Origin ``verified_synthetic``;
the group is the kind and seed, so a problem never lands in two splits.

``python -m dclab_rnd.lake selfplay [--seeds 20] [--kinds binary,regression]`` writes ``lake/bronze/selfplay/``
(examples and the rejection log); ``make lake`` reads them into gold.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from dclab_core import lake as L

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "lake" / "bronze" / "selfplay"
SYSTEM = ("You are DCLab's machine-learning solver. You read a problem as DCLab describes it (the columns as bands and "
          "names, the outcome, the moment a prediction is made) and answer the one question asked, as JSON, using only "
          "what is given. Never use information unavailable at the prediction moment; never read a holdout twice.")


def _columns(table: Any, target: str | None, masked: bool) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The table's columns as DCLab states (bands, never values), and the name map when masked."""
    from ..ledger import column_state
    from ..studio.data import profile_table

    profile = profile_table(table)
    rows = int(len(table))
    names = [c.get("name") for c in profile.get("columns") or []]
    alias = {n: (f"col_{i + 1:02d}" if n != target else "outcome") for i, n in enumerate(names)} if masked else {n: n for n in names}
    out = []
    for c in profile.get("columns") or []:
        s = column_state(c, rows)
        if masked:
            s = {k: v for k, v in s.items() if k not in ("name_tokens",)}
        s["column"] = alias[c.get("name")]
        out.append(s)
    return out, alias


def _ask(question: str, given: dict[str, Any], answer: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({"question": question, **given}, ensure_ascii=False, sort_keys=True, default=str)},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, sort_keys=True, default=str)}]


def examples_of(kind: Any, seed: int, case: Any, run: Any, checks: list[dict[str, Any]], masked: bool) -> list[L.TrainingExample]:
    """The families one verified solve teaches, in one naming variant."""
    columns, alias = _columns(case.table, case.target, masked)
    rename = lambda c: alias.get(c, c) if c else c  # noqa: E731
    order = sorted((n for n in alias if n and alias[n] != n), key=len, reverse=True)

    def mask(text: Any) -> str:  # the masked variant hides the names inside free text too
        text = str(text or "")
        for n in order:
            text = text.replace(str(n), alias[n])
        return text

    sol = run.solution or {}
    data, models, features, final = (run.evidence(s) for s in ("data", "models", "features", "final"))
    target = rename(case.target)
    group = f"selfplay:{kind.key}:{seed}"
    prov = L.Provenance(source="selfplay", origin="verified_synthetic", record_ids=[group], consent="ours", group=group, split=L.split_of(group))  # type: ignore[arg-type]
    ok = L.Verification(kind="construction", passed=True, detail="; ".join(c["decision"] for c in checks)[:400])
    base = {"columns": columns, "outcome": target, "prediction_moment": case.moment, "rows": int(len(case.table))}
    out = []

    def add(family: str, question: str, given: dict[str, Any], answer: dict[str, Any]) -> None:
        msgs = _ask(question, given, answer)
        eid = "sp_" + L.fingerprint([family, msgs])[:24]
        out.append(L.TrainingExample(example_id=eid, family=family, messages=msgs, verification=ok, quality=0.8, provenance=prov))  # type: ignore[arg-type]

    add("problem_type", "What kind of machine-learning problem is this?", base,
        {"task": case.task, "kind": kind.key, "why": (getattr(kind, "title", "") or kind.key)})
    leak = run.evidence("leakage")
    signals = [{"column": rename(x.get("feature") or x.get("column")), **{k: x[k] for k in ("score", "metric", "auc", "value") if k in x}}
               for x in (leak.get("top_univariate_signals") or []) if isinstance(x, dict)][:8]
    add("leakage_judgment", "Which columns must stay out of the model, and why?",
        {**base, "train_only_signals": signals, "note": "each column alone against the outcome, measured on training rows only"},
        {"forbidden": [{"column": rename(f.get("column")), "reason": mask(f.get("reason"))[:300]} for f in sol.get("forbidden") or []],
         "identifiers": [rename(c) for c in sol.get("identifiers") or []]})
    add("split_design", "How should the rows be split so the evaluation is honest?", base,
        {"holdout": mask(data.get("holdout_split"))[:300],
         "time_column": rename(sol.get("time_column")), "group_column": rename(sol.get("group_column")),
         "cross_validation": mask(data.get("cv_protocol"))[:200]})
    if models.get("selected_model") and not masked:  # no names in these inputs: one variant, not two copies
        summary = {"task": case.task, "rows": int(len(case.table)), "usable_features": data.get("usable_feature_count"),
                   "target_summary": data.get("target_summary")}
        add("algorithm_choice", "Which feature recipe and model family should be fitted, measured inside the training folds?", summary,
            {"feature_recipe": features.get("selected_recipe"), "model": models.get("selected_model"),
             "ranking": [r.get("model") if isinstance(r, dict) else r for r in (models.get("ranking") or [])][:4],
             "selection_rule": mask(models.get("selection_rule"))[:200]})
    metrics = final.get("holdout_metrics") or {}
    ci = final.get("holdout_primary_metric_ci")
    if metrics and not masked:
        primary = next((m for m in ("roc_auc", "average_precision", "r2", "mae", "accuracy") if m in metrics), None)
        baseline = final.get("baselines") or {}
        chance = {"roc_auc": 0.5, "r2": 0.0, "average_precision": (baseline.get("majority_class") or {}).get("average_precision_random_ranking")}.get(primary)
        beats = (metrics[primary] > chance) if primary and isinstance(chance, (int, float)) else None
        add("read_results", "What do the holdout results say, with their uncertainty, and what may be claimed?",
            {"task": case.task, "holdout_metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))}, "primary_interval": ci,
             "baselines": baseline, "holdout_uses": final.get("holdout_uses_in_this_project")},
            {"primary_metric": primary, "value": metrics.get(primary) if primary else None, "interval": ci,
             "beats_baseline": beats, "holdout_read_once": final.get("holdout_uses_in_this_project") == 1,
             "claim": "benchmark evidence on synthetic data: says the pipeline behaves, not how it would do on real cases"})
    return out


def solve(kinds_keys: Iterable[str] | None = None, seeds: int = 20, log=print) -> dict[str, Any]:
    """Every kind's replica for each seed, solved and verified; examples and rejections written to bronze."""
    from .. import kinds
    from ..kinds import acceptance as A

    OUT.mkdir(parents=True, exist_ok=True)
    kept, rejected = [], []
    wanted = set(kinds_keys or [])
    for kind in kinds.all_kinds():
        if wanted and kind.key not in wanted:
            continue
        if not getattr(kind, "runs_in_engine", True):
            continue
        for seed in range(seeds):
            group = f"selfplay:{kind.key}:{seed}"
            started = time.perf_counter()
            try:
                case = kind.case(seed)
                if case is None or case.target is None:
                    break  # a kind without a replica case, or one with no outcome (anomaly: its own path)
                run = A.engine_run(case, kind.key)
                checks = kind.accept(run)
            except Exception as exc:  # noqa: BLE001 — a solve that breaks is a rejection with its reason, never a stop
                rejected.append({"group": group, "reason": f"{type(exc).__name__}: {str(exc)[:200]}"})
                continue
            missed = [c["decision"] for c in checks if not c["ok"]]
            if missed or not checks:
                rejected.append({"group": group, "missed": missed or ["no acceptance decisions"]})
                continue
            for masked in (False, True):
                kept += examples_of(kind, seed, case, run, checks, masked)
            log(f"{group}: verified ({len(checks)} decisions, {time.perf_counter() - started:.0f}s)")
    (OUT / "examples.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in kept), encoding="utf-8")
    (OUT / "rejected.json").write_text(json.dumps(rejected, indent=1) + "\n", encoding="utf-8")
    return {"examples": len(kept), "problems": len({e.provenance.group for e in kept}), "rejected": len(rejected)}


def examples() -> list[L.TrainingExample]:
    path = OUT / "examples.jsonl"
    if not path.is_file():
        return []
    return [L.TrainingExample.read(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
