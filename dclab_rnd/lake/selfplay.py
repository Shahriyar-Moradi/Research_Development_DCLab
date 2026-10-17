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
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

from dclab_core import lake as L

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "lake" / "bronze" / "selfplay"
PROMPT = "solver_answer"  # dclab_rnd/prompts/solver_answer.md: the same system text in training and when served (27.6)


def system() -> str:
    from .. import prompts

    return prompts.text(PROMPT).strip()


def _columns(table: Any, target: str | None, masked: bool, pixel_prefix: str | None = None) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The table's columns as DCLab states (bands, never values), and the name map when masked. An image table's pixel
    columns (26.8) are one state: how many, the image's side, never a pixel's value."""
    from ..ledger import column_state
    from ..studio.data import profile_table

    pixels = [c for c in table.columns if pixel_prefix and str(c).startswith(pixel_prefix)]
    if pixels:
        table = table.drop(columns=pixels)
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
    if pixels:
        side = int(round(len(pixels) ** 0.5))
        out.append({"column": f"{pixel_prefix}*", "kind": "pixels", "count": len(pixels),
                    "image": f"{side}x{side}" if side * side == len(pixels) else None, "dtype": "float"})
    return out, alias


class _Images:
    """26.8: the embedded kind's image case as a self-play kind of its own (``selfplay:<kind>_images:<seed>``), solved by
    the same engine and verified by the image checks."""

    runs_in_engine = True

    def __init__(self, base: Any):
        self.base, self.key, self.engine_key = base, f"{base.key}_images", base.key
        self.title = f"{getattr(base, 'title', base.key)}: images"

    def case(self, seed: int = 0) -> Any:
        return self.base.image_case(seed)

    def accept_engine(self, run: Any) -> list[dict[str, Any]]:
        return self.base.accept_image_engine(run)

    def own_machine(self) -> Any:
        return self.base.own_machine()


def _kinds() -> list[Any]:
    """Every engine kind, and an image variant beside a kind that has an image case."""
    from .. import kinds

    out = []
    for kind in kinds.all_kinds():
        out.append(kind)
        if hasattr(kind, "image_case") and hasattr(kind, "accept_image_engine"):
            out.append(_Images(kind))
    return out


# The answer each family asks for, stated in the question (live, 2026-10-16: a model asked cold answered right in substance
# under its own keys, and the benchmark could not read it): a trained model and a model asked cold see the same contract.
FORMATS: dict[str, dict[str, Any]] = {
    "problem_type": {"task": "binary | multiclass | regression (the base task the model learns)", "kind": "the kind of problem, e.g. churn, forecasting, rare_event, uplift, time_to_event", "why": "one sentence"},
    "leakage_judgment": {"forbidden": [{"column": "a column name as given", "reason": "why it is unknown at the moment or encodes the outcome"}],
                         "identifiers": ["a column name as given"]},
    "split_design": {"holdout": "how the holdout is drawn", "time_column": "a column name as given, or null", "group_column": "a column name as given, or null",
                     "cross_validation": "how the training rows are folded"},
    "algorithm_choice": {"feature_recipe": "a recipe name", "model": "a model family", "ranking": ["model families, best first"], "selection_rule": "one sentence"},
    "read_results": {"primary_metric": "a metric name", "value": "a number", "interval": "the interval given, or null", "beats_baseline": "true, false or null",
                     "holdout_read_once": "true or false", "claim": "what may be claimed, in one sentence"},
    "solution_design": {"task": "binary | multiclass | regression", "kind": "the kind of problem",
                        "workflow": [{"step": "split | leakage | features | models | final", "decision": "what is decided at that step"}]},
}


def _ask(question: str, given: dict[str, Any], answer: dict[str, Any], family: str | None = None) -> list[dict[str, str]]:
    asked = {"question": question, **given}
    if family in FORMATS:
        asked["answer_format"] = FORMATS[family]
        asked["answer_only"] = "Reply with this JSON object only, its keys exactly as written, and nothing else."
    return [{"role": "system", "content": system()},
            {"role": "user", "content": json.dumps(asked, ensure_ascii=False, sort_keys=True, default=str)},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, sort_keys=True, default=str)}]


def primary_of(metrics: dict[str, Any], interval: Any, reported: Any = None) -> str | None:
    """27.9: the primary metric is the one the engine reported, the one its interval was drawn for (a first match in a
    fixed order paired a rare event's ROC-AUC with its average precision's interval)."""
    if reported in metrics:
        return str(reported)
    point = interval.get("point") if isinstance(interval, dict) else None
    if isinstance(point, (int, float)):
        match = [m for m, v in metrics.items() if isinstance(v, (int, float)) and abs(float(v) - float(point)) < 1e-9]
        if match:
            return next((m for m in ("average_precision", "roc_auc", "r2", "mae", "accuracy") if m in match), match[0])
    return next((m for m in ("roc_auc", "average_precision", "r2", "mae", "accuracy") if m in metrics), None)


def beats_chance(primary: str | None, metrics: dict[str, Any], baseline: dict[str, Any]) -> bool | None:
    chance = {"roc_auc": 0.5, "r2": 0.0, "average_precision": (baseline.get("majority_class") or {}).get("average_precision_random_ranking")}.get(primary or "")
    return (metrics[primary] > chance) if primary and isinstance(chance, (int, float)) else None


def examples_of(kind: Any, seed: int, case: Any, run: Any, checks: list[dict[str, Any]], masked: bool) -> list[L.TrainingExample]:
    """The families one verified solve teaches, in one naming variant."""
    columns, alias = _columns(case.table, case.target, masked, ((case.solution or {}).get("kind_fields") or {}).get("pixel_prefix"))
    rename = lambda c: alias.get(c, c) if c else c  # noqa: E731
    order = sorted((n for n in alias if n and alias[n] != n), key=len, reverse=True)

    def mask(text: Any) -> str:  # the masked variant hides the names inside free text too
        text = str(text or "")
        for n in order:
            text = text.replace(str(n), alias[n])
        return text

    sol = dict(run.solution or {})
    # the embedded kind keeps its group (an author, a scene) as its source column: the split's group all the same
    sol["group_column"] = sol.get("group_column") or ((case.solution or {}).get("kind_fields") or {}).get("source_column")
    data, models, features, final = (run.evidence(s) for s in ("data", "models", "features", "final"))
    target = rename(case.target)
    group = f"selfplay:{kind.key}:{seed}"
    prov = L.Provenance(source="selfplay", origin="verified_synthetic", record_ids=[group], consent="ours", group=group, split=L.split_of(group))  # type: ignore[arg-type]
    ok = L.Verification(kind="construction", passed=True, detail="; ".join(c["decision"] for c in checks)[:400])
    base = {"columns": columns, "outcome": target, "prediction_moment": case.moment, "rows": int(len(case.table))}
    out = []

    def add(family: str, question: str, given: dict[str, Any], answer: dict[str, Any]) -> None:
        msgs = _ask(question, given, answer, family)
        eid = "sp_" + L.fingerprint([family, msgs])[:24]
        out.append(L.TrainingExample(example_id=eid, family=family, messages=msgs, verification=ok, quality=0.8, provenance=prov))  # type: ignore[arg-type]

    add("problem_type", "What kind of machine-learning problem is this?", base,
        {"task": case.task, "kind": getattr(kind, "engine_key", kind.key), "why": (getattr(kind, "title", "") or kind.key)})
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
    # 27.3: the whole solution workflow, step by step, as DCLab's verified run took it (the families above are its parts)
    workflow = [
        {"step": "split", "decision": {"holdout": mask(data.get("holdout_split"))[:300], "cross_validation": mask(data.get("cv_protocol"))[:200],
                                       "time_column": rename(sol.get("time_column")), "group_column": rename(sol.get("group_column"))}},
        {"step": "leakage", "decision": {"forbidden": [rename(f.get("column")) for f in sol.get("forbidden") or []],
                                         "identifiers": [rename(c) for c in sol.get("identifiers") or []]}},
        {"step": "features", "decision": {"recipe": features.get("selected_recipe")}},
        {"step": "models", "decision": {"model": models.get("selected_model"), "selection_rule": mask(models.get("selection_rule"))[:200]}},
        {"step": "final", "decision": {"metric": final.get("metric"), "holdout_read_once": final.get("holdout_uses_in_this_project") == 1}},
    ]
    add("solution_design", "Lay out the solution workflow for this problem, step by step, before anything is fitted.", base,
        {"task": case.task, "kind": getattr(kind, "engine_key", kind.key), "workflow": workflow})
    metrics = final.get("holdout_metrics") or {}
    ci = final.get("holdout_primary_metric_ci")
    if metrics and not masked:
        primary = primary_of(metrics, ci, final.get("metric"))
        baseline = final.get("baselines") or {}
        beats = beats_chance(primary, metrics, baseline)
        add("read_results", "What do the holdout results say, with their uncertainty, and what may be claimed?",
            {"task": case.task, "holdout_metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))}, "primary_interval": ci,
             "baselines": baseline, "holdout_uses": final.get("holdout_uses_in_this_project")},
            {"primary_metric": primary, "value": metrics.get(primary) if primary else None, "interval": ci,
             "beats_baseline": beats, "holdout_read_once": final.get("holdout_uses_in_this_project") == 1,
             "claim": "benchmark evidence on synthetic data: says the pipeline behaves, not how it would do on real cases"})
    return out


def screen_record(kind: Any, seed: int, case: Any, run: Any) -> dict[str, Any]:
    """26.6a: what a learned AutoML reads (the table's meta-features, from the data card's bands) and what it learns from
    (every family's cross-validated score in the screen, on training folds only). No value of the table."""
    from collections import Counter as _C

    columns, _ = _columns(case.table, case.target, False, ((case.solution or {}).get("kind_fields") or {}).get("pixel_prefix"))
    data, models = run.evidence("data"), run.evidence("models")
    return {"group": f"selfplay:{kind.key}:{seed}", "kind": kind.key, "task": case.task, "rows": int(len(case.table)),
            "columns": len(columns), "dtypes": dict(_C(c.get("dtype") for c in columns)),
            "cardinality": dict(_C(c.get("cardinality") for c in columns)), "nulls": dict(_C(c.get("nulls") for c in columns)),
            "usable_features": data.get("usable_feature_count"), "target_summary": data.get("target_summary"),
            "split": str(data.get("holdout_split") or "")[:120],
            "screen": [{k: r.get(k) for k in ("model", "mean", "std", "elapsed_seconds")} for r in models.get("ranking") or [] if isinstance(r, dict)],
            "selected": models.get("selected_model"), "recipe": run.evidence("features").get("selected_recipe")}


def _run_recorded(A: Any, case: Any, key: str, group: str) -> Any:
    """The engine's run with a decision ledger of its own (``bronze/selfplay/ledgers/<kind>/<seed>/``): every decision the
    engine makes on the problem (the split, the metric, each column's role and leak, the features, the model family, the
    threshold, the tuning), each with the replica's planted truth as its benchmark label where it has one (19.8)."""
    from .. import ledger

    _, kind, seed = group.split(":")
    folder = OUT / "ledgers" / kind / seed
    shutil.rmtree(folder, ignore_errors=True)  # a problem solved again is recorded again, whole
    book = ledger.FileLedger(folder, "selfplay")
    book.home = folder
    before = ledger._INSTALLED.get("ledger")
    ledger.install(book)
    try:
        return A.engine_run(case, key)
    finally:
        ledger.install(before)


def backfill(kinds_keys: Iterable[str] | None = None, log=print, limit: int | None = None) -> dict[str, Any]:
    """Problems verified before 27.3's widening, solved again to record their decision ledger and their solution workflow
    example (the other examples are already kept); resumable."""
    from ..kinds import acceptance as A

    tried = [json.loads(x) for x in (OUT / "done.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] if (OUT / "done.jsonl").is_file() else []
    marks = OUT / "backfill_done.jsonl"
    done = {json.loads(x)["group"] for x in marks.read_text(encoding="utf-8").splitlines() if x.strip()} if marks.is_file() else set()
    wanted, n = set(kinds_keys or []), 0
    by_key = {k.key: k for k in _kinds()}
    for t in tried:
        group = t["group"]
        _, key, seed = group.split(":")
        if not t.get("verified") or group in done or (wanted and key not in wanted) or key not in by_key:
            continue
        if limit is not None and n >= limit:
            break
        kind, started = by_key[key], time.perf_counter()
        try:
            case = kind.case(int(seed))
            run = _run_recorded(A, case, getattr(kind, "engine_key", key), group)
            checks = kind.accept_engine(run) if kind.own_machine() is not None else kind.accept(run)
            got = [e for masked in (False, True) for e in examples_of(kind, int(seed), case, run, checks, masked) if e.family == "solution_design"]
            if not [c for c in checks if not c["ok"]]:
                with (OUT / "examples.jsonl").open("a", encoding="utf-8") as f:
                    f.write("".join(e.model_dump_json() + "\n" for e in got))
            outcome = {"verified": not [c for c in checks if not c["ok"]], "examples": len(got)}
        except Exception as exc:  # noqa: BLE001 - a problem that breaks now is said, never a stop
            outcome = {"verified": False, "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}
        with marks.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"group": group, **outcome, "seconds": round(time.perf_counter() - started, 1)}) + "\n")
        n += 1
        log(f"{group}: recorded again ({outcome})")
    return {"recorded_again": n}


def solve(kinds_keys: Iterable[str] | None = None, seeds: int = 20, log=print) -> dict[str, Any]:
    """Every kind's replica for each seed, solved and verified; examples and rejections written to bronze."""
    from ..kinds import acceptance as A

    OUT.mkdir(parents=True, exist_ok=True)
    kept, rejected = [], []
    wanted = set(kinds_keys or [])
    done_path = OUT / "done.jsonl"  # one line per problem tried: a run that stops resumes where it was, nothing solved twice
    done = {json.loads(x)["group"] for x in done_path.read_text(encoding="utf-8").splitlines() if x.strip()} if done_path.is_file() else set()
    examples_path = OUT / "examples.jsonl"
    if not done:
        examples_path.write_text("", encoding="utf-8")

    def record(group: str, examples: list[L.TrainingExample], outcome: dict[str, Any]) -> None:
        with examples_path.open("a", encoding="utf-8") as f:
            f.write("".join(e.model_dump_json() + "\n" for e in examples))
        with done_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"group": group, **outcome}) + "\n")

    for kind in _kinds():
        if wanted and kind.key not in wanted:
            continue
        if not getattr(kind, "runs_in_engine", True):
            continue
        for seed in range(seeds):
            group = f"selfplay:{kind.key}:{seed}"
            if group in done:
                continue
            started = time.perf_counter()
            try:
                case = kind.case(seed)
                if case is None or case.target is None:
                    break  # a kind without a replica case, or one with no outcome (anomaly: its own path)
                run = _run_recorded(A, case, getattr(kind, "engine_key", kind.key), group)
                # a kind with its own machine checks the engine's run with accept_engine (its accept reads its own machine's result)
                checks = kind.accept_engine(run) if kind.own_machine() is not None else kind.accept(run)
            except Exception as exc:  # noqa: BLE001 — a solve that breaks is a rejection with its reason, never a stop
                rejected.append({"group": group, "reason": f"{type(exc).__name__}: {str(exc)[:200]}"})
                record(group, [], {"verified": False, "reason": rejected[-1]["reason"]})
                continue
            missed = [c["decision"] for c in checks if not c["ok"]]
            if missed or not checks:
                rejected.append({"group": group, "missed": missed or ["no acceptance decisions"]})
                record(group, [], {"verified": False, "missed": rejected[-1]["missed"]})
                continue
            got: list[L.TrainingExample] = []
            for masked in (False, True):
                got += examples_of(kind, seed, case, run, checks, masked)
            kept += got
            try:
                with (OUT / "screens.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(screen_record(kind, seed, case, run), default=str) + "\n")
            except Exception as exc:  # noqa: BLE001 — a screen that cannot be read costs itself, never the examples
                log(f"{group}: no screen record ({type(exc).__name__})")
            record(group, got, {"verified": True, "examples": len(got), "seconds": round(time.perf_counter() - started, 1)})
            log(f"{group}: verified ({len(checks)} decisions, {time.perf_counter() - started:.0f}s)")
    tried = [json.loads(x) for x in done_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    (OUT / "rejected.json").write_text(json.dumps([t for t in tried if not t.get("verified")], indent=1) + "\n", encoding="utf-8")
    return {"examples": len(examples()), "problems": sum(1 for t in tried if t.get("verified")), "rejected": sum(1 for t in tried if not t.get("verified"))}


def forget(kinds_keys: Iterable[str]) -> dict[str, int]:
    """A kind's solves taken out of the self-play files (its examples, screens, tries and ledgers), so ``solve`` does
    them again whole, when what a solve records has changed (26.8: the embedded kind's group column)."""
    wanted = set(kinds_keys)
    if not wanted:
        raise ValueError("name the kinds to solve again")
    mine = lambda group: str(group or "").split(":")[1:2] and str(group).split(":")[1] in wanted  # noqa: E731
    removed: dict[str, int] = {}
    for name, key in (("examples.jsonl", lambda d: d.get("provenance", {}).get("group")), ("screens.jsonl", lambda d: d.get("group")),
                      ("done.jsonl", lambda d: d.get("group")), ("backfill_done.jsonl", lambda d: d.get("group"))):
        path = OUT / name
        if not path.is_file():
            continue
        lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        kept = [x for x in lines if not mine(key(json.loads(x)))]
        removed[name] = len(lines) - len(kept)
        path.write_text("".join(x + "\n" for x in kept), encoding="utf-8")
    for k in wanted:
        shutil.rmtree(OUT / "ledgers" / k, ignore_errors=True)
    return removed


def examples() -> list[L.TrainingExample]:
    path = OUT / "examples.jsonl"
    if not path.is_file():
        return []
    return [L.TrainingExample.read(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def restate(path: Path | None = None) -> int:
    """Give every example already solved its family's answer format (the solves stay; only the question changes), and
    (27.9) a results example its primary metric as the engine measured its interval."""
    path = path or OUT / "examples.jsonl"
    rows = [L.TrainingExample.read(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []
    out = []
    for e in rows:
        asked = json.loads(e.messages[1]["content"])
        question = asked.pop("question")
        for k in ("answer_format", "answer_only"):
            asked.pop(k, None)
        answer = json.loads(e.messages[2]["content"])
        if e.family == "read_results":  # 27.9: the primary metric is the one its interval was drawn for
            metrics = asked.get("holdout_metrics") or {}
            primary = primary_of(metrics, asked.get("primary_interval"))
            if primary and primary != answer.get("primary_metric"):
                answer.update(primary_metric=primary, value=metrics.get(primary), beats_baseline=beats_chance(primary, metrics, asked.get("baselines") or {}))
        msgs = _ask(question, asked, answer, e.family)
        out.append(e.model_copy(update={"messages": msgs, "example_id": "sp_" + L.fingerprint([e.family, msgs])[:24]}))
    path.write_text("".join(e.model_dump_json() + "\n" for e in out), encoding="utf-8")
    return len(out)
