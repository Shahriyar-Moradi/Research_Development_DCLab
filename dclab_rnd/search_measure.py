"""34.7's measurement (campaign ``search_v3``): the studio engine with a search against the same engine without one, both
run fresh on the same code, rows, splits and holdout.

For each dataset the bake-off's own project (34.5) is copied twice: one copy runs the engine as it is, the other with
``settings.search`` (the budget below). Each copy reads its holdout once, in its final stage. The two arms' holdout
predictions are compared on the same rows (the bake-off's paired bootstrap, over whole groups when the split keeps
them), and the search's selection-fold score is set against the holdout beside its search score.

What changed from ``search_v1`` (aborted before its first result, ``search_v1/ABORTED.json``): the arm without a search is
run again on the same code instead of read from the bake-off (whose engine was older); the code that makes the numbers
is pinned by hash and a resume on changed code is refused; each arm's captured predictions must reproduce its own
record's holdout number; the gates are the four below; the datasets add a time split, a group split with text and a
multiclass task. From ``search_v2`` (closed with 8 of 9 datasets, ``search_v2/CLOSED.json``): its code crashed the
search arm on the fraud table (a fold of the search rows without a fraud case); the fix skips the search there, and a
dataset where only the search arm fails is now a row that counts as clearly worse, never a silent gap.

    python -m dclab_rnd.search_measure plan
    python -m dclab_rnd.search_measure run --yes      # about 10 minutes a dataset on the 8 GB Air; resumable on the same code
    python -m dclab_rnd.search_measure report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "search_v3"
CAMPAIGN = ROOT / "evidence" / "campaigns" / CAMPAIGN_ID
WORK = ROOT / "agent_runs" / CAMPAIGN_ID
DATASETS = ("german_credit", "telco_churn", "online_shoppers", "wine_quality", "adult", "bank_marketing", "credit_card_fraud",
            "ecommerce_clothing_reviews", "letter_recognition")
BUDGET = {"minutes": 3.0, "trials": 24, "families": 2}
GATE = {
    "id": "search_gate_v3",
    "never_clearly_worse": "on at most 1 dataset the holdout difference (search minus no search, oriented so positive is better) has "
                           "its whole 95% paired interval below 0, or the search arm failed where the arm without did not",
    "failures": "a dataset where the arm without a search fails is left out of every gate and named; the verdict needs every "
                "dataset compared or failed",
    "rises_on_average": "the mean holdout difference over the datasets is above 0",
    "honest_selection": "over the datasets, the mean absolute gap between the selection fold's score and the holdout score is smaller "
                        "than the mean absolute gap between the search score and the holdout score",
    "bounded": "on every dataset the search's own minutes are at most 1.5 times the budget's minutes",
    "verdict": "pass when all four hold: the search may be recommended in a project's settings; otherwise it stays an opt-in, said "
               "with this result. Either way it is never on by default from this campaign alone",
    "reported_not_gated": "per dataset, the gain over the no-search choice on the selection fold with its interval, and whether its "
                          "sign agrees with the holdout difference; minutes of both arms",
}
CODE_FILES = ("dclab_rnd/studio/search.py", "dclab_rnd/studio/engine.py", "dclab_core/selection.py", "dclab_rnd/expansion/runner.py",
              "dclab_rnd/search_measure.py", "dclab_rnd/bakeoff.py")
ABORTED = {"search_v1": ("Stopped before its first result (2026-10-25): its review found the final stage re-tuning the searched "
                         "settings, entities shared across the selection fold on time splits with entities, no class check, the review's "
                         "change lost, an arm without a search read from the older bake-off engine, unpinned code and gates that "
                         "could not tell a gain. Replaced by search_v2.")}
CLOSED = {"search_v2": ("Ran on commit bf7fab50 (2026-10-25): 8 of 9 datasets compared; on credit_card_fraud the search arm "
                        "crashed (with the selection fold set aside, the earliest fold of the search rows held no fraud case, so no "
                        "model could be fitted there). A defect of the measured code, fixed in the next commit (the search is "
                        "skipped when a fold would miss a class); no verdict was written for search_v2 because one dataset failed. "
                        "Its eight results are kept as they are. Replaced by search_v3, whose gates count such a failure.")}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_sha256(root: Path = ROOT) -> str:
    from .campaign_capture import ENGINE_CODE, code_sha256 as whole

    return whole((*ENGINE_CODE, *CODE_FILES, "dclab_rnd/campaign_capture.py", "dclab_rnd/ledger", "dclab_rnd/experience"), root)  # every file behind the numbers


CODE_AT_START = code_sha256()  # the code this process loaded (an edit on disk later never changes what it runs)


def _git() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *CODE_FILES], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 - outside a checkout: the hash alone pins the code
        return {"commit": None, "code_files_dirty": None}
    return {"commit": commit, "code_files_dirty": bool(dirty)}


def manifest() -> dict[str, Any]:
    body = {"campaign": CAMPAIGN_ID, "package": "34.7", "datasets": list(DATASETS), "budget": BUDGET, "gate": GATE,
            "objective": ("Does the engine's opt-in search lift the holdout? The engine with a search against the same engine without "
                          "one, both run fresh on the same code, rows, splits and holdout, on nine of the bake-off's datasets."),
            "arms": {"without": "the engine as it is (no search setting)", "with": f"the same engine with settings.search = {BUDGET}"},
            "source": "each dataset's project, table, rows and split from evidence/campaigns/bakeoff_v1 (agent_runs/bakeoff_v1/<dataset>)",
            "statistic": "dclab_rnd.bakeoff.paired_bootstrap on the shared holdout rows (95%, whole groups when the split keeps them)"}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def abort_previous() -> None:
    for name, why in ABORTED.items():
        folder = ROOT / "evidence" / "campaigns" / name
        if folder.is_dir() and not (folder / "ABORTED.json").exists():
            (folder / "ABORTED.json").write_text(json.dumps({"campaign": name, "aborted": True, "why": why, "replaced_by": "search_v2",
                                                             "results": 0}, indent=2) + "\n")
        _ended_report(folder, name, "aborted before its first result", why)
    for name, why in CLOSED.items():
        folder = ROOT / "evidence" / "campaigns" / name
        if folder.is_dir() and not (folder / "CLOSED.json").exists():
            done = sorted(p.stem.split("_", 1)[1] for p in (folder / "results").glob("SRCH-0*.json")) if (folder / "results").is_dir() else []
            rows = [json.loads(p.read_text()) for p in sorted((folder / "results").glob("SRCH-0*.json"))] if done else []
            reading = {k: v for k, v in judge(rows).items() if k not in ("gate", "verdict")} if rows else None
            (folder / "CLOSED.json").write_text(json.dumps({"campaign": name, "closed": True, "why": why, "replaced_by": CAMPAIGN_ID,
                                                            "results": len(done), "datasets": done,
                                                            "reading_of_its_results": reading,
                                                            "reading_note": "the four gates on the datasets it compared (no verdict: one "
                                                                            "dataset failed and its gates had no rule for a failure)"},
                                                           indent=2) + "\n")
        _ended_report(folder, name, "closed with what it had", why)


def _ended_report(folder: Path, name: str, how: str, why: str) -> None:
    """The Lab page's title and objective for a campaign that ended without a verdict (generated, never edited)."""
    if folder.is_dir():  # written again each time: generated from the note above
        (folder / "CAMPAIGN_REPORT.md").write_text(
            f"# {name}: the engine's search against no search (package 34.7), {how}\n\n"
            f"Does the engine's opt-in search lift the holdout? {why}\n\n"
            "Generated by `python -m dclab_rnd.search_measure`; do not edit.\n")


def freeze() -> dict[str, Any]:
    """The manifest, written once before the first run; afterwards the gate and the code must be the same."""
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    path = CAMPAIGN / "manifest.json"
    want = manifest()
    if path.exists():
        held = json.loads(path.read_text())
        if held.get("sha256") != want["sha256"]:
            raise SystemExit("The gate, the datasets or the budget changed since the campaign was frozen: start a new campaign version")
        if held.get("code_sha256") != CODE_AT_START:
            raise SystemExit("The code that makes the numbers changed since the campaign was frozen: its results so far came from other "
                             "code, so a resume would mix two engines. Start a new campaign version.")
        return held
    held = {**want, "code_sha256": CODE_AT_START, "code_files": list(CODE_FILES), "git": _git(), "frozen_at": _now(),
            "python": sys.version.split()[0]}
    path.write_text(json.dumps(held, indent=2) + "\n")
    return held


def _arm(dataset: str, name: str, budget: dict[str, Any] | None) -> dict[str, Any]:
    """One arm on a fresh copy of the bake-off's project: the five stages, the final stage's one holdout fit captured and
    checked against the stage's own record."""
    from unittest import mock

    from dclab_rnd import bakeoff
    from dclab_rnd.expansion import runner as rn
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import engine

    source = bakeoff.WORK / dataset
    if not (source / "project.json").exists():
        raise ValueError(f"the bake-off has no project for {dataset}")
    work = WORK / dataset / name
    if work.exists():
        shutil.rmtree(work)  # this campaign's own working copy, made again from the bake-off's
    work.mkdir(parents=True)
    shutil.copytree(source / "workspace", work / "workspace")
    shutil.copy(source / "project.json", work / "project.json")
    store, pid = bakeoff._open(work)
    store.clear_stages(pid, "data")
    project = store.get(pid)
    project["holdout_uses"] = 0  # a fresh arm on a copy: it reads the holdout once
    settings = {k: v for k, v in (project.get("settings") or {}).items() if k != "search"}
    if budget:
        settings["search"] = budget
    project["settings"] = settings
    store.save(project)
    started = time.perf_counter()
    last: dict[str, Any] = {}
    real = rn.fit_and_predict

    def recording(*args: Any, **kwargs: Any) -> dict[str, Any]:
        out = real(*args, **kwargs)
        last.clear()
        last.update(out=out, rows=len(args[6]) if len(args) > 6 else len(kwargs["X_eval"]))
        return out

    from .campaign_capture import recorded

    with recorded(work / "workspace"), private_scope(True):  # the arm's ledger and session log kept; the rules decide, as in the bake-off
        for stage in engine.STAGE_KEYS[:-1]:
            engine.execute(store, pid, stage)
        with mock.patch.object(rn, "fit_and_predict", recording):  # the final stage's last fit is its one holdout fit
            engine.execute(store, pid, engine.STAGE_KEYS[-1])
    seconds = time.perf_counter() - started
    final = store.read_stage(pid, "final")
    models = store.read_stage(pid, "models") or {}
    prepared = engine.prepare(store, store.get(pid))
    metric, higher, fn = bakeoff._metric_fn(prepared)
    predictions = np.asarray(last["out"]["predictions"], dtype=float)
    if last.get("rows") != len(prepared.bundle.test_idx):
        raise RuntimeError(f"{dataset}/{name}: the final stage's last fit did not predict the holdout rows")
    recorded = final["evidence"]["holdout_metrics"][metric]
    got = fn(np.asarray(prepared.bundle.y_test), predictions)
    if recorded is None or not math.isclose(got, recorded, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(f"{dataset}/{name}: the captured predictions give {metric} {got}, the stage's record says {recorded}")
    if int(store.get(pid).get("holdout_uses") or 0) != 1:
        raise RuntimeError(f"{dataset}/{name}: the holdout was read {store.get(pid).get('holdout_uses')} times")
    return {"seconds": round(seconds, 1), "metric": metric, "higher_is_better": higher, "holdout": float(got), "fn": fn,
            "predictions": predictions, "y": np.asarray(prepared.bundle.y_test),
            "rows": np.asarray(prepared.source_rows)[np.asarray(prepared.bundle.test_idx)],
            "groups": prepared.bundle.groups("test") if prepared.bundle.spec.split_strategy in ("group", "time_group") else None,
            "split": prepared.bundle.spec.split_strategy, "model": final["evidence"].get("model"),
            "params": final["evidence"].get("selected_model_params"), "interval": final["evidence"].get("holdout_primary_metric_ci"),
            "search": (models.get("evidence") or {}).get("search")}


class ArmFailed(RuntimeError):
    def __init__(self, arm: str, error: BaseException):
        super().__init__(f"the {arm} arm failed: {type(error).__name__}: {str(error)[:300]}")
        self.arm, self.error = arm, error


def _failed(dataset: str, error: ArmFailed) -> dict[str, Any]:
    """A dataset whose arm failed is a row of its own (never a silent gap): which arm, the error's type and words."""
    row = {"id": f"SRCH-{DATASETS.index(dataset) + 1:03d}", "experiment_id": f"SRCH-{DATASETS.index(dataset) + 1:03d}", "status": "failed",
           "completed_at": _now(), "campaign_id": CAMPAIGN_ID, "dataset": dataset, "failed": error.arm,
           "error": f"{type(error.error).__name__}: {str(error.error)[:300]}", "code_sha256": CODE_AT_START, "git": _git(), "finished_at": _now()}
    out = CAMPAIGN / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{row['id']}_{dataset}.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def run_one(dataset: str) -> dict[str, Any]:
    from dclab_rnd import bakeoff

    try:
        without = _arm(dataset, "without", None)
    except Exception as exc:  # noqa: BLE001 - recorded as the row of a failed arm
        raise ArmFailed("without", exc) from exc
    try:
        searched = _arm(dataset, "with", BUDGET)
    except Exception as exc:  # noqa: BLE001
        raise ArmFailed("with", exc) from exc
    if not np.array_equal(without["rows"], searched["rows"]):
        raise RuntimeError(f"{dataset}: the two arms' holdout rows differ")
    fn, higher, y = without["fn"], without["higher_is_better"], without["y"]
    paired = bakeoff.paired_bootstrap(y, searched["predictions"], without["predictions"], fn, higher, groups=without["groups"])
    found = searched["search"] or {}
    chosen = found.get("selection") or {}
    sign = 1.0 if higher else -1.0
    gaps = None
    if chosen.get("selection_score") is not None:
        gaps = {"selection_minus_holdout": round(sign * (chosen["selection_score"] - searched["holdout"]), 6),
                "search_minus_holdout": round(sign * (chosen["search_score"] - searched["holdout"]), 6)}
    against = found.get("without_search") or {}
    agrees = None
    if against.get("gain") is not None and paired["point"] != 0:
        agrees = bool(np.sign(against["gain"]) == np.sign(paired["point"])) if against["gain"] != 0 else None
    result = {"id": f"SRCH-{DATASETS.index(dataset) + 1:03d}", "experiment_id": f"SRCH-{DATASETS.index(dataset) + 1:03d}", "status": "completed",
              "completed_at": _now(), "campaign_id": CAMPAIGN_ID, "dataset": dataset, "metric": without["metric"],
              "higher_is_better": higher, "split": without["split"], "holdout_rows": int(len(y)),
              "without": {k: without[k] for k in ("holdout", "seconds", "model", "params", "interval")},
              "with": {k: searched[k] for k in ("holdout", "seconds", "model", "params", "interval")},
              "difference": paired, "search": {k: found.get(k) for k in ("family", "params", "attempts", "minutes", "skipped", "failed")},
              "selection": chosen, "gaps": gaps, "selection_gain_over_no_search": against or None,
              "selection_gain_agrees_with_holdout": agrees, "code_sha256": CODE_AT_START, "git": _git(), "finished_at": _now()}
    out = CAMPAIGN / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{result['id']}_{dataset}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def judge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The four gates on the datasets' results (pure: the tests call it on made-up rows)."""
    every = rows
    search_failed = [r["dataset"] for r in every if r.get("failed") == "with"]
    left_out = [r["dataset"] for r in every if r.get("failed") == "without"]
    rows = [r for r in every if not r.get("failed")]
    diffs = [r["difference"]["point"] for r in rows]
    clearly_worse = [r["dataset"] for r in rows if r["difference"].get("high") is not None and r["difference"]["high"] < 0] + search_failed
    with_gaps = [r for r in rows if r.get("gaps")]
    selection_gap = float(np.mean([abs(r["gaps"]["selection_minus_holdout"]) for r in with_gaps])) if with_gaps else None
    search_gap = float(np.mean([abs(r["gaps"]["search_minus_holdout"]) for r in with_gaps])) if with_gaps else None
    over = [r["dataset"] for r in rows if (r["search"].get("minutes") or 0) > 1.5 * BUDGET["minutes"]]
    gates = {"never_clearly_worse": len(clearly_worse) <= 1,
             "rises_on_average": bool(diffs) and float(np.mean(diffs)) > 0,
             "honest_selection": selection_gap is not None and search_gap is not None and selection_gap < search_gap,
             "bounded": not over}
    complete = len(every) == len(DATASETS)
    return {"datasets": len(rows), "of": len(DATASETS), "gates": gates, "search_arm_failed": search_failed, "left_out_without_failed": left_out,
            "verdict": ("pass" if all(gates.values()) else "fail") if complete else f"not yet: {len(every)} of {len(DATASETS)} datasets",
            "mean_difference": round(float(np.mean(diffs)), 6) if diffs else None, "range": [min(diffs), max(diffs)] if diffs else None,
            "clearly_better": [r["dataset"] for r in rows if (r["difference"].get("low") or 0) > 0], "clearly_worse": clearly_worse,
            "mean_abs_gap": {"selection_fold": selection_gap, "search_score": search_gap}, "over_budget": over,
            "skipped": [r["dataset"] for r in rows if r["search"].get("skipped")],
            "selection_gain_agrees": [r["dataset"] for r in rows if r.get("selection_gain_agrees_with_holdout") is True],
            "selection_gain_disagrees": [r["dataset"] for r in rows if r.get("selection_gain_agrees_with_holdout") is False],
            "minutes": {"without": round(sum(r["without"]["seconds"] for r in rows) / 60, 1), "with": round(sum(r["with"]["seconds"] for r in rows) / 60, 1)},
            "gate": GATE}


def report() -> dict[str, Any]:
    folder = CAMPAIGN / "results"
    rows = [json.loads(p.read_text()) for p in sorted(folder.glob("SRCH-0*.json"))] if folder.exists() else []
    out = judge(rows)
    if len(rows) == len(DATASETS) and not (folder / "SRCH-100_decision.json").exists():  # every dataset in; written once
        (folder / "SRCH-100_decision.json").write_text(json.dumps({**out, "id": "SRCH-100", "experiment_id": "SRCH-100", "status": "completed",
                                                                   "completed_at": _now(), "campaign_id": CAMPAIGN_ID, "decided_at": _now()},
                                                                  indent=2) + "\n")  # an id as every result has (search_v3's verdict had none)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="34.7: the engine's search against no search, on the bake-off's datasets")
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps({**manifest(), "code_sha256": CODE_AT_START}, indent=2))
        return 0
    if args.command == "report":
        print(json.dumps(report(), indent=2))
        return 0
    if not args.yes:
        print(json.dumps(manifest(), indent=2))
        print("Add --yes to run it.")
        return 0
    abort_previous()
    freeze()
    done = {p.stem.split("_", 1)[1] for p in (CAMPAIGN / "results").glob("SRCH-0*.json")} if (CAMPAIGN / "results").exists() else set()
    for dataset in DATASETS:
        if dataset in done:
            continue
        print(f"[search] {dataset} {_now()}", flush=True)
        try:
            r = run_one(dataset)
            d = r["difference"]
            print(f"[search] {dataset}: without {r['without']['holdout']:.4f} with {r['with']['holdout']:.4f} difference {d['point']:+.4f} "
                  f"[{d['low']}, {d['high']}] gaps {r['gaps']} minutes {r['search']['minutes']}", flush=True)
        except ArmFailed as exc:  # a row of its own, counted by the gates
            _failed(dataset, exc)
            print(f"[search] {dataset} failed: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001 - a failure outside the arms (the comparison itself): said, never a row
            print(f"[search] {dataset} failed outside the arms: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    print(json.dumps(report(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
