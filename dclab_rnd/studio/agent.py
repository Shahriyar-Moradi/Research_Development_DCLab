"""The agent's voice: evidence-cited notes on every stage, and answers to questions.

Notes are deterministic. Each one names the rule it applies and the measured
precedent it rests on, so "show proof" always points at a real record. When an
LLM is configured (``OPENAI_API_KEY`` plus the ``openai`` package), it adds one
plain-language critique per stage, constrained to the same record IDs; it never
decides anything.
"""

from __future__ import annotations

from typing import Any

from dclab_rnd import prompts
from dclab_rnd import cited, tools

STAGE_RULES = {
    "data": ["DCLAB-R01", "DCLAB-R02", "DCLAB-R03", "DCLAB-R12"],
    "leakage": ["DCLAB-R04", "DCLAB-R05", "DCLAB-R06", "DCLAB-R01"],
    "features": ["DCLAB-R07", "DCLAB-R08", "DCLAB-R09", "DCLAB-R11"],
    "models": ["DCLAB-R13", "DCLAB-R14", "DCLAB-R16"],
    "final": ["DCLAB-R15", "DCLAB-R16", "DCLAB-R17", "DCLAB-R18", "DCLAB-R22"],
}
STAGE_OF = {"data": "data_understanding", "leakage": "leakage_audit", "features": "feature_engineering",
            "models": "model_selection", "final": "optimization_reliability"}
TASK_FILTERS = {"binary_imbalanced": ["binary", "binary_imbalanced"], "text_tabular_binary": ["text_tabular_binary", "binary"],
                "multiclass": ["multiclass"], "timeseries_regression": ["timeseries_regression"]}


def _index():
    return tools._index()


def _exists(record_id: str) -> bool:
    return _index().get(record_id) is not None


def _note(severity: str, title: str, text: str, proof: list[str], action: str | None = None) -> dict[str, Any]:
    seen: list[str] = []
    for rid in proof:
        if rid and rid not in seen and _exists(rid):
            seen.append(rid)
    return {"severity": severity, "title": title, "text": text, "proof": seen, "action": action, "source": "evidence"}


def _precedents(stage: str, task_type: str, query: str, k: int = 2, record_type: str = "experiment") -> list[str]:
    hits = _index().search(query, k=k, type=record_type, stage=STAGE_OF.get(stage), task_type=TASK_FILTERS.get(task_type, task_type))
    if not hits and record_type == "experiment":
        hits = _index().search(query, k=k, type=record_type, stage=STAGE_OF.get(stage))
    return [h["record_id"] for h in hits]


def _leak_precedents(columns: list[str]) -> list[str]:
    wanted = {c.lower() for c in columns}
    out = []
    for record in _index().records:
        if record["type"] == "leakage_precedent" and wanted & {c.lower() for c in record["metadata"].get("columns", [])}:
            out.append(record["record_id"])
    return out


def _biggest_leaks(k: int = 3) -> list[str]:
    leaks = [r for r in _index().records if r["type"] == "leakage_precedent" and r["metadata"].get("apparent_lift") is not None]
    leaks.sort(key=lambda r: -abs(r["metadata"]["apparent_lift"]) if r["metadata"].get("task_type") != "timeseries_regression" else 0)
    return [r["record_id"] for r in leaks[:k]]


def _fmt(value: Any, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else str(value)


# ---------------------------------------------------------------------- notes per stage


def narrate(stage: str, record: dict[str, Any], solution: dict[str, Any], task_type: str, project_id: str | None = None) -> list[dict[str, Any]]:
    ev = record["evidence"]
    rules = STAGE_RULES[stage]
    notes: list[dict[str, Any]] = []
    metric = record.get("primary_metric", "")
    if stage == "data":
        notes.append(_note("info", "The holdout is locked before anything is learned",
                           f"{ev['holdout_rows']} rows ({ev['holdout_split']}) are sealed now and scored exactly once, in the final stage. "
                           f"Everything until then uses {ev['cv_protocol']}.", ["DCLAB-R02", "DCLAB-R17", "WF-03"]))
        risks = ev.get("risk_features", {})
        ids = [c for c, r in risks.items() if "identifier_like" in r]
        if ids:
            notes.append(_note("warning", f"{len(ids)} column(s) look like identifiers but are used as inputs",
                               f"{', '.join(ids[:6])} are nearly unique per row. A model can memorize them without learning anything that transfers. "
                               "If they are keys, declare them as identifiers in the solution.", ["DCLAB-R12", "DCLAB-R04"], "Edit the solution: mark them as identifiers"))
        shifted = [c for c, r in risks.items() if "train_holdout_feature_shift" in r]
        if shifted:
            notes.append(_note("warning", "Some inputs drift between training and holdout",
                               f"{', '.join(shifted[:6])} have PSI > 0.20 between the training rows and the sealed holdout. "
                               "Expect the holdout score to be lower than CV and treat it as the honest number.", ["DCLAB-R12", "DCLAB-R18"]))
        sparse = [c for c, r in risks.items() if "missingness" in r]
        if sparse:
            notes.append(_note("info", f"{len(sparse)} column(s) have more than 5% missing values",
                               "Imputation is fitted inside each training fold, so this is handled without leakage; the question is whether the values will also be missing in production.",
                               ["DCLAB-R03", "WF-09"]))
        ts = ev.get("target_summary", {})
        if task_type in ("binary_imbalanced", "text_tabular_binary") and ts.get("positive_rate_train", 0.5) < 0.1:
            notes.append(_note("warning", "Rare positives: accuracy would mislead",
                               f"Only {ts['positive_rate_train']:.1%} of training rows are positive. The campaigns scored such problems with average precision and recall at a precision target, not accuracy.",
                               ["DCLAB-R18", *_precedents("data", task_type, "imbalanced positive rate average precision")]))
        if task_type == "timeseries_regression" and not solution.get("time_column"):
            notes.append(_note("warning", "No time column declared for a regression target",
                               "If these rows are events over time, a random split lets the model see the future. The bike-sharing campaign measured how optimistic that is.",
                               ["DCLAB-R02", *_leak_precedents(["casual", "registered"]), *_precedents("leakage", task_type, "random split optimism time order")],
                               "Declare the time column if rows are ordered in time"))
        if ev.get("duplicate_feature_rows_train", 0) > 0:
            notes.append(_note("info", f"{ev['duplicate_feature_rows_train']} training rows repeat another row's inputs exactly",
                               "Identical rows can land on both sides of a random split and inflate scores slightly.", ["DCLAB-R02", *_precedents("leakage", "multiclass", "duplicate contamination")]))
    elif stage == "leakage":
        declared = ev.get("declared_leakage_features", [])
        lift = ev.get("apparent_lift", 0.0)
        if declared:
            precedents = _leak_precedents(declared) or _biggest_leaks(2)
            if lift > 0.02 or (task_type == "timeseries_regression" and lift > 0):
                notes.append(_note("high", f"The forbidden columns would have inflated the score by {_fmt(lift)}",
                                   f"With {', '.join(declared)} included, training-CV {metric} rises by {_fmt(lift)}. That is the signature of leakage: a score you would never see in production. "
                                   "The largest jumps ever recorded in the DCLab campaigns came from exactly this.", [*precedents, "DCLAB-R04", "DCLAB-R05"]))
            else:
                notes.append(_note("info", "The forbidden columns add little or nothing to the score",
                                   f"Excluding {', '.join(declared)} costs {_fmt(abs(lift))} {metric}. The exclusion still stands: the solution says they are unknown at the prediction moment, and that, not the score, decides.",
                                   [*precedents, "DCLAB-R05"]))
        else:
            notes.append(_note("info", "No column is forbidden yet", "Nothing in the solution is marked as unknown at the prediction moment. Review the flagged columns below before trusting the score.",
                               ["DCLAB-R01", "DCLAB-R04"]))
        flagged = [f for f in ev.get("heuristic_review_candidates", []) if "declared_post_outcome_or_contested" not in f["reasons"]]
        for finding in flagged[:6]:
            reasons = ", ".join(r.replace("_", " ") for r in finding["reasons"])
            severity = "high" if {"exact_target_proxy", "arithmetic_target_identity", "extreme_univariate_signal"} & set(finding["reasons"]) else "warning"
            notes.append(_note(severity, f"Review `{finding['feature']}`: {reasons}",
                               "A single column this predictive is either the real driver or something recorded after the outcome. Only the prediction moment can tell. "
                               "If it is not known at that moment, add it to the forbidden list and rerun the audit.",
                               [*_leak_precedents([finding["feature"]]), "DCLAB-R05", "DCLAB-R04"], "Confirm or forbid this column in the solution"))
        if "random_vs_time_cv" in ev:
            gap = ev["random_vs_time_cv"]["optimism_gap"]
            notes.append(_note("warning" if gap > 0 else "info", f"Random splits would look {_fmt(gap)} better than time-ordered ones",
                               "That gap is the optimism a random split would have hidden. The time-ordered protocol is kept for every later stage.",
                               ["DCLAB-R02", *_precedents("leakage", "timeseries_regression", "random KFold time ordered optimism")]))
        if not ev.get("detector_canary_passed", True):
            notes.append(_note("warning", "The leakage detector missed its own canary", "Treat the heuristic review list as incomplete and rely on the solution.", ["DCLAB-R05"]))
    elif stage == "features":
        selected, best = ev["selected_recipe"], ev["best_recipe"]
        if selected != best:
            notes.append(_note("info", f"`{selected}` chosen over `{best}` although `{best}` scored higher",
                               f"The best mean ({_fmt(ev['best_metric_mean'])}) and the selected one ({_fmt(ev['selected_metric_mean'])}) are within the tolerance, so the smaller recipe wins: fewer features, less production surface, same evidence.",
                               ["DCLAB-R09", "DCLAB-R11", *_precedents("features", task_type, "feature ladder smallest recipe within tolerance")]))
        else:
            notes.append(_note("info", f"`{selected}` is both the best and the smallest recipe within tolerance",
                               "Every transform was fitted inside the training folds, so this comparison is leakage-free by construction.",
                               ["DCLAB-R07", "DCLAB-R03", *_precedents("features", task_type, "feature engineering ablation recipes")]))
        gain = ev.get("selected_vs_reference_gain", 0.0)
        if abs(gain) < 0.003:
            notes.append(_note("info", "Feature engineering barely moved the score",
                               f"Versus the raw inputs the change is {_fmt(gain)}. In the 50-experiment campaign this was the common case: engineered features rarely beat a good raw baseline on tabular data.",
                               ["DCLAB-R08", *_precedents("features", "binary", "raw baseline ratios no reproducible lift")]))
    elif stage == "models":
        ranking = ev.get("ranking", [])
        leader = ranking[0] if ranking else {}
        notes.append(_note("info", f"{leader.get('model', '?')} leads, but a CV leader is a candidate, not a law",
                           "The campaigns found no universal best algorithm: the leader changed from dataset to dataset. Ranking penalizes fold spread so a lucky split cannot win.",
                           ["DCLAB-R13", "DCLAB-R14", *[r["record_id"] for r in _index().search("no universal best model family", k=1, type="finding")],
                            *_precedents("models", task_type, "algorithm screening model families identical folds")]))
        if len(ranking) > 1 and abs(ranking[0]["mean"] - ranking[1]["mean"]) <= ranking[0]["std"]:
            notes.append(_note("warning", f"{ranking[0]['model']} and {ranking[1]['model']} are within fold noise",
                               "Either could be the better model. Prefer the simpler or faster one unless the final holdout separates them; do not read the ranking as a verdict.",
                               ["DCLAB-R14", "DCLAB-R16"], "Consider choosing the simpler model below"))
    elif stage == "final":
        decision = ev["tuning_decision"]
        notes.append(_note("info", "Tuning " + ("helped by a small, real margin" if decision["accepted"] else "did not beat the defaults"),
                           f"The best candidate changed training-CV {metric} by {decision['gain']:+.4f} against a required {decision['required_gain']:.4f}. "
                           + ("Accepted." if decision["accepted"] else "The default configuration was kept; in the campaigns this happened on most datasets."),
                           ["DCLAB-R15", "DCLAB-R16", *_precedents("final", task_type, "optimization tuning margin default kept")]))
        gap = ev.get("cv_to_holdout_gap", 0.0)
        ci = ev.get("holdout_primary_metric_ci", {})
        notes.append(_note("warning" if gap < -0.02 else "info", f"Holdout {_fmt(ev['holdout_metrics'].get(metric))} vs CV mean {_fmt(ev['cv_selected_metric_mean'])}",
                           f"The 95% interval on the holdout is {_fmt(ci.get('low'))}–{_fmt(ci.get('high'))} ({ci.get('method')}). "
                           + ("The holdout is noticeably worse than CV: expect production to look like the holdout, not the CV." if gap < -0.02 else "CV and holdout agree within the interval."),
                           ["DCLAB-R17", "DCLAB-R18"]))
        notes.append(_note("high" if ev.get("holdout_uses_in_this_project", 1) > 1 else "info",
                           "The holdout has now been used " + ("once" if ev.get("holdout_uses_in_this_project", 1) == 1 else f"{ev['holdout_uses_in_this_project']} times"),
                           "Every extra look at the holdout turns it into another validation set and its score into an optimistic one. Change the model only with new data or a fresh split."
                           if ev.get("holdout_uses_in_this_project", 1) > 1 else
                           "This is the honest number. Changing the model after seeing it would make the next score optimistic.",
                           ["DCLAB-R17", "PIT-006", "DCLAB-R22"]))
        if "calibration_bins" in ev:
            worst = max(ev["calibration_bins"], key=lambda b: abs(b["predicted"] - b["observed"]), default=None)
            if worst and abs(worst["predicted"] - worst["observed"]) > 0.15:
                notes.append(_note("warning", "Probabilities are miscalibrated in at least one range",
                                   f"In the {worst['lower']:.1f}–{worst['upper']:.1f} bin the model predicts {worst['predicted']:.2f} but {worst['observed']:.2f} of rows are positive. Use thresholds from the table, not the raw probability.",
                                   ["DCLAB-R18", "WF-09"]))
        notes.append(_note("info", "Not production-approved", "Benchmark evidence proves the method, not deployment readiness: source lineage, monitoring and a fresh confirmation are still required.",
                           ["DCLAB-R22", "DCLAB-R21" if _exists("DCLAB-R21") else "DCLAB-R22"]))
    fillers = 0
    for rid in rules:
        if not any(rid in n["proof"] for n in notes) and fillers < 2:
            rule = _index().get(rid)
            if rule and len(notes) < 8:
                fillers += 1
                notes.append(_note("info", rule["title"], rule["text"].split(" Why: ")[1].split(" Failure signal:")[0] if " Why: " in rule["text"] else rule["text"][:300], [rid]))
    critique = llm_critique(stage, record, notes, project_id)
    if critique:
        notes.insert(0, critique)
    return notes


# ---------------------------------------------------------------------- questions


def answer(question: str, project: dict[str, Any], records: dict[str, dict[str, Any]], task_type: str | None) -> dict[str, Any]:
    """Evidence-first answer: project facts when the question is about this project, R&D records otherwise."""
    q = question.strip()
    lowered = q.lower()
    facts: list[str] = []
    proof: list[str] = []
    if any(w in lowered for w in ("leak", "forbid", "unavailable", "after the outcome")) and "leakage" in records:
        ev = records["leakage"]["evidence"]
        facts.append(f"In this project the forbidden columns are {ev['declared_leakage_features'] or 'none'}; their apparent lift is {_fmt(ev['apparent_lift'])} {records['leakage']['primary_metric']}.")
        flagged = [f["feature"] for f in ev["heuristic_review_candidates"] if "declared_post_outcome_or_contested" not in f["reasons"]]
        if flagged:
            facts.append(f"Still to review: {', '.join(flagged[:8])}.")
        proof += _leak_precedents(ev["declared_leakage_features"]) or _biggest_leaks(2)
    if any(w in lowered for w in ("model", "algorithm", "best", "which")) and "models" in records:
        ranking = records["models"]["evidence"]["ranking"]
        facts.append("Model ranking on identical folds: " + "; ".join(f"{r['model']} {_fmt(r['mean'])}±{_fmt(r['std'])}" for r in ranking) + ".")
    if any(w in lowered for w in ("feature", "recipe", "engineer")) and "features" in records:
        ev = records["features"]["evidence"]
        facts.append(f"Selected recipe `{ev['selected_recipe']}` ({ev['selected_recipe_description']}); best mean was `{ev['best_recipe']}`.")
    if any(w in lowered for w in ("holdout", "score", "final", "result", "good", "trust")) and "final" in records:
        ev = records["final"]["evidence"]
        ci = ev["holdout_primary_metric_ci"]
        facts.append(f"Holdout {records['final']['primary_metric']}: {_fmt(ev['holdout_metrics'][records['final']['primary_metric']])} (95% {_fmt(ci['low'])}–{_fmt(ci['high'])}), used {ev.get('holdout_uses_in_this_project', 1)} time(s); not production-approved.")
        proof += ["DCLAB-R17", "DCLAB-R22"]
    filters = {"task_type": TASK_FILTERS.get(task_type or "", None)} if task_type else {}
    hits = _index().search(q, k=4, **{k: v for k, v in filters.items() if v}) or _index().search(q, k=4)
    for h in hits:
        if h["record_id"] not in proof:
            proof.append(h["record_id"])
    evidence_text = [f"{h['record_id']}: {h['title']}" for h in hits]
    text = " ".join(facts) if facts else ""
    if evidence_text:
        text += (" " if text else "") + "Relevant evidence: " + " · ".join(evidence_text[:4]) + "."
    if not text:
        text = "I could not match that to the project or the evidence. Try naming a column, a stage or a rule."
    result = {"question": q, "answer": text, "proof": proof[:8], "source": "evidence"}
    llm = llm_answer(q, facts, hits, project.get("id"))
    if llm:
        result["llm_answer"] = llm
    return result


# ---------------------------------------------------------------------- optional LLM


def llm_available() -> bool:
    """A model serves the stage notes (through the model gateway; see dclab_rnd/models/settings.py)."""
    from ..models import installed

    return installed().available("stage_notes")


_POLICY = prompts.text("stage_critique")  # dclab_rnd/prompts/stage_critique.md (A4.3: prompts are versioned files)


def _chat(prompt: str, purpose: str, project_id: str | None = None) -> str | None:
    from ..models import installed

    client = installed().client(purpose, project_id=project_id)
    if client is None:
        return None
    try:
        reply = client.complete([{"role": "user", "content": prompt}], max_tokens=400)
        text, _ = cited.strip_overclaims((reply.get("content") or "").strip())  # a sentence that says "production-ready" is removed
        return text or None
    except Exception:  # noqa: BLE001 — the LLM is optional; the deterministic notes stand alone
        return None


def explain_stage(record: dict[str, Any], project_id: str | None = None) -> dict[str, Any] | None:
    """The stage's explanation: a model's sentences, each citing the record, with their numbers checked (``studio.explain``)."""
    from ..models import installed
    from . import explain

    return explain.explain(record, installed().client("stage_notes", project_id=project_id))


def llm_critique(stage: str, record: dict[str, Any], notes: list[dict[str, Any]], project_id: str | None = None) -> dict[str, Any] | None:
    ids = sorted({rid for n in notes for rid in n["proof"]})
    prompt = f"{_POLICY}\n\nStage: {record['title']}\nSummary: {record['setup_summary']}\nClaims: " + " | ".join(c["statement"] for c in record["claims"]) + f"\nAllowed record IDs: {ids}"
    text = _chat(prompt, "stage_notes", project_id)
    if not text:
        return None
    return {"severity": "info", "title": "LLM critique (advisory)", "text": text, "proof": [i for i in ids if i in text], "action": None, "source": "llm"}


def llm_answer(question: str, facts: list[str], hits: list[dict[str, Any]], project_id: str | None = None) -> str | None:
    context = "\n".join(f"[{h['record_id']}] {h['text'][:500]}" for h in hits)
    return _chat(f"{_POLICY}\n\nQuestion: {question}\nProject facts: {' '.join(facts) or 'none yet'}\nEvidence:\n{context}", "project_answer", project_id)
