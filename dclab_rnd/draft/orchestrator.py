"""The Home orchestrator (the owner, 2026-10-20): on a built project's chat, Home leads a team instead of answering alone.

The owner asked "improve the model for me", then "ok do it" and "ok": the old chat matched words, did not remember its own
offer, and answered the follow-ups with nonsense. Here every message is one turn of an orchestrator that reads the whole
conversation and the project's state, decides what is wanted, and hands each part to a specialist:

- the **Records reader** brings what the project's records hold, as tables the person sees in the chat (the models trained
  and their scores, the feature recipes, the data, the leakage audit, the stages);
- the **Evaluator** the held-back result and its measures (precision, recall, F1, the confusion matrix);
- the **Experimenter** improves the model practically (``studio.improve``, a background job on the training folds only) and
  adopts a way that cleared the margin;
- the **Notebook coder** writes a notebook cell for a plot or an analysis (``studio.notebook_agent``), run when the person
  says so.

Each specialist is a tool of the scope "home" in the one registry (``agents.default_registry``); a write passes the scope's
guard. With a model serving the purpose ``home_orchestrator``, the model plans the turn (``agents.runtime.run``, traced as
the agent ``home.project``) and writes the reply; its numbers must be numbers a tool returned, or the sentence is dropped.
Without a model, or when it fails, the rules plan it (``plan``, the agent ``home.project.rules``): a yes carries out the
pending offer, words name the specialists. A long task (the Experimenter's job) gets a task board in the chat that follows
the job until it ends and says a failure plainly. Every turn is kept whole in ``<workspace>/home_orchestrator/``.

What a model reads: the person's words (personal values redacted), the conversation's texts, and the tools' results, which are
the project's records (names, counts, scores, parameters): never a row or a cell value.
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from ..agents.registry import Registry, Tool

PURPOSE = "home_orchestrator"
AGENT = "home.project"
RULES_AGENT = "home.project.rules"
TEAM = {"reader": "Records reader", "evaluator": "Evaluator", "experimenter": "Experimenter", "coder": "Notebook coder"}
CAP_EUR = 0.25  # one turn's model spend, at most
TAIL = 12  # messages of the conversation the model reads
SHOWN_SENTENCES = 4  # a reply beside a table: at most this many sentences (the owner: "bring the table instead of the texts")
WATCH_SECONDS = 3 * 3600  # how long a job is followed on the board


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _r(value: Any, digits: int = 4) -> Any:
    return round(float(value), digits) if isinstance(value, (int, float)) and not isinstance(value, bool) else value


# ---------------------------------------------------------------------- the turn's context: what the tools read and where they show it


class Turn:
    """One message's work on one project: the records (read once), the cards shown, the tasks run, the results the reply may cite."""

    def __init__(self, services: Any, project_id: str, draft_id: str, agent: Any, text: str = "",
                 offer: Optional[Callable[[Optional[str], Optional[dict[str, Any]]], None]] = None):
        self.services, self.projects, self.project_id, self.draft_id, self.agent, self.text = services, services.projects, project_id, draft_id, agent, text
        self._offer = offer
        self.offered: Optional[dict[str, Any]] = None  # the offer the last turn made, read once at this turn's start (a later turn never sees it)
        self._records: Optional[dict[str, Any]] = None
        self.shown: dict[str, dict[str, Any]] = {}  # tool -> the card it posted this turn (one card a tool a turn)
        self.results: list[Any] = []
        self.tasks: list[dict[str, Any]] = []
        self.chips: list[str] = []

    def project(self) -> dict[str, Any]:
        return self.projects.get(self.project_id)

    def records(self) -> dict[str, Any]:
        if self._records is None:
            self._records = self.projects.records(self.project_id)
        return self._records

    def stage(self, name: str) -> Optional[dict[str, Any]]:
        return self.projects.read_stage(self.project_id, name)

    def agent_offer(self, tool: Optional[str], args: Optional[dict[str, Any]] = None) -> None:
        """What a yes to this turn would do next (a later "ok" carries it out); None withdraws the offer."""
        if self._offer is not None:
            self._offer(tool, args)

    def show(self, tool: str, card: dict[str, Any]) -> dict[str, Any]:
        """Post the card in the chat (once a tool a turn) and keep it as what the reply may cite."""
        card = {"agent": TEAM[card.get("agent", "reader")], "project": self.project_id, **{k: v for k, v in card.items() if k != "agent"}}
        if tool not in self.shown:
            self.agent.say(self.draft_id, "", kind="project_card", card=card)
            self.shown[tool] = card
        self.task(tool, card["agent"], card.get("title") or tool, "done")
        self.results.append(card)
        return card

    def task(self, tool: str, agent: str, title: str, status: str, **extra: Any) -> dict[str, Any]:
        found = next((t for t in self.tasks if t["tool"] == tool), None)
        if found is not None and found.get("job") and status == "failed":
            found = None  # a refused second call ("already queued") is its own task: the job's task stays as its job is
        if found is None:
            found = {"id": f"t{len(self.tasks) + 1}", "tool": tool, "agent": agent, "title": title, "status": status, **extra}
            self.tasks.append(found)
        else:
            found.update(status=status, title=title, **extra)
        return found


# ---------------------------------------------------------------------- the specialists' tools


def _stage_missing(name: str) -> dict[str, Any]:
    return {"error": f"The {name} stage has not run yet in this project, so there is nothing to show from it."}


def show_models(t: Turn, /, **_: Any) -> dict[str, Any]:
    """The models trained in the screen, with their scores on identical folds, the chosen one marked."""
    from ..studio import engine

    models = t.stage("models")
    if not models:
        return _stage_missing("models")
    ev, metric = models.get("evidence") or {}, models.get("primary_metric") or "score"
    chosen = engine.chosen(models, "selected_model")
    rows = [{"family": r["model"], "mean": _r(r.get("mean")), "spread": _r(r.get("std")), "adjusted": _r(r.get("adjusted_score")),
             "seconds": _r(r.get("elapsed_seconds"), 1), "rank": i, "chosen": r["model"] == chosen} for i, r in enumerate(ev.get("ranking") or [], 1)]
    facts = [["Metric", metric], ["Chosen", chosen], ["Features", ev.get("feature_recipe")], ["Folds", ev.get("cv_protocol")]]
    dummy = ev.get("dummy_cv") or {}
    if isinstance(dummy.get("mean"), (int, float)):
        facts.append(["The simplest model on the same folds", _r(dummy["mean"])])
    final = (t.stage("final") or {}).get("evidence") or {}
    if final.get("model"):
        facts.append(["Model scored on the held-back cases", final.get("model")])
    return {"card": t.show("show_models", {
        "agent": "reader", "title": "Models trained and compared", "page": "models",
        "subtitle": f"{len(rows)} model families on identical folds · ranked by mean − 0.25 × fold spread, then runtime",
        "columns": [{"key": "family", "label": "Family"}, {"key": "mean", "label": "Mean", "num": True}, {"key": "spread", "label": "Fold spread", "num": True},
                    {"key": "adjusted", "label": "Mean − k×spread", "num": True}, {"key": "seconds", "label": "Fit time (s)", "num": True, "digits": 1}, {"key": "rank", "label": "Rank", "num": True}],
        "rows": rows, "mark": "chosen", "facts": [f for f in facts if f[1] is not None]})}


def show_features(t: Turn, /, **_: Any) -> dict[str, Any]:
    """The feature recipes compared, the selected one marked."""
    feats = t.stage("features")
    if not feats:
        return _stage_missing("features")
    from ..expansion import runner as rn
    from ..studio import engine

    ev, metric = feats.get("evidence") or {}, feats.get("primary_metric") or "score"
    chosen = engine.chosen(feats, "selected_recipe")
    rows = [{"recipe": r.get("recipe"), "description": r.get("description"), "features": _r(r.get("feature_count_mean"), 1),
             "mean": _r(rn._mean(r, metric)), "spread": _r(rn._std(r, metric)), "chosen": r.get("recipe") == chosen} for r in ev.get("stage_results") or []]
    return {"card": t.show("show_features", {
        "agent": "reader", "title": "Feature recipes compared", "page": "models", "subtitle": f"each recipe fitted inside the training folds · {metric}",
        "columns": [{"key": "recipe", "label": "Recipe"}, {"key": "description", "label": "What it does"}, {"key": "features", "label": "Features", "num": True, "digits": 0},
                    {"key": "mean", "label": "Mean", "num": True}, {"key": "spread", "label": "Fold spread", "num": True}],
        "rows": rows, "mark": "chosen", "facts": [["Selected", chosen], ["Gain over the plain inputs", _r(ev.get("selected_vs_reference_gain"))]]})}


def show_data(t: Turn, /, **_: Any) -> dict[str, Any]:
    """The data as the project's first stage measured it: rows, the held-back split, the outcome, the risks found (never a row)."""
    data = t.stage("data")
    if not data:
        return _stage_missing("data")
    ev = data.get("evidence") or {}
    risks = ev.get("risk_features") or {}
    by: dict[str, list[str]] = {}
    for col, reasons in risks.items():
        for reason in reasons if isinstance(reasons, list) else [reasons]:
            by.setdefault(str(reason).replace("_", " "), []).append(str(col))
    ts = ev.get("target_summary") or {}
    facts = [["Training rows", ev.get("train_rows")], ["Held-back rows", ev.get("holdout_rows")], ["Held-back split", ev.get("holdout_split")],
             ["Folds", ev.get("cv_protocol")], ["Outcome", (t.project().get("solution") or {}).get("target")],
             ["Positive share (training)", _r(ts.get("positive_rate_train"))], ["Rows repeating another's inputs", ev.get("duplicate_feature_rows_train")]]
    rows = [{"risk": k, "columns": len(v), "examples": ", ".join(v[:6])} for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))]
    return {"card": t.show("show_data", {
        "agent": "reader", "title": "The data", "page": "project", "subtitle": "measured by the data stage; the held-back rows were sealed before anything was learned",
        "facts": [f for f in facts if f[1] is not None],
        "columns": [{"key": "risk", "label": "Risk found"}, {"key": "columns", "label": "Columns", "num": True}, {"key": "examples", "label": "Which"}], "rows": rows})}


def show_leakage(t: Turn, /, **_: Any) -> dict[str, Any]:
    """The leakage audit: the forbidden columns, what they would have added, the columns still to review."""
    leak = t.stage("leakage")
    if not leak:
        return _stage_missing("leakage")
    ev = leak.get("evidence") or {}
    rows = [{"column": f.get("feature"), "why": ", ".join(str(r).replace("_", " ") for r in f.get("reasons") or [])}
            for f in ev.get("heuristic_review_candidates") or [] if "declared_post_outcome_or_contested" not in (f.get("reasons") or [])][:20]
    return {"card": t.show("show_leakage", {
        "agent": "reader", "title": "Leakage audit", "page": "audit", "subtitle": "columns unknown at the prediction moment are never used",
        "facts": [["Forbidden columns", ", ".join(ev.get("declared_leakage_features") or []) or "none"],
                  [f"What they would have added ({leak.get('primary_metric')})", _r(ev.get("apparent_lift"))]],
        "columns": [{"key": "column", "label": "Column to review"}, {"key": "why", "label": "Why it was flagged"}], "rows": rows})}


def show_stages(t: Turn, /, **_: Any) -> dict[str, Any]:
    """Each stage's status, and what is running now."""
    p = t.project()
    rows = [{"stage": name, "status": (st or {}).get("status"), "finished": str((st or {}).get("finished") or "")[:16].replace("T", " "),
             "seconds": _r((st or {}).get("elapsed_seconds"), 1)} for name, st in (p.get("stages") or {}).items()]
    return {"card": t.show("show_stages", {
        "agent": "reader", "title": "The project's stages", "page": "project", "facts": [["Running now", p.get("running") or "nothing"]],
        "columns": [{"key": "stage", "label": "Stage"}, {"key": "status", "label": "Status"}, {"key": "finished", "label": "Finished"},
                    {"key": "seconds", "label": "Seconds", "num": True, "digits": 1}], "rows": rows})}


def show_result(t: Turn, /, **_: Any) -> dict[str, Any]:
    """The held-back result: the model, the score with its interval, the folds' score, the baselines, how often the held-back cases were read."""
    final = t.stage("final")
    if not final:
        return _stage_missing("final")
    ev, metric = final.get("evidence") or {}, final.get("primary_metric") or "score"
    ci = ev.get("holdout_primary_metric_ci") or {}
    rows = [{"baseline": name, metric: _r(b.get(metric))} for name, b in (ev.get("baselines") or {}).items() if isinstance(b, dict)]
    facts = [["Model", ev.get("model")], ["Metric", metric], ["Held-back score", _r((ev.get("holdout_metrics") or {}).get(metric))],
             ["95% interval", f"{_r(ci.get('low'))} to {_r(ci.get('high'))}" if ci.get("low") is not None else None],
             ["Training folds' score", _r(ev.get("cv_selected_metric_mean"))], ["Folds to held-back gap", _r(ev.get("cv_to_holdout_gap"))],
             ["Held-back rows", ev.get("holdout_rows")], ["Times the held-back cases were read", ev.get("holdout_uses_in_this_project", 1)],
             ["Approved for production", "no: benchmark evidence alone never approves a model"]]
    tuning = ev.get("tuning_decision") or {}
    if isinstance(tuning.get("gain"), (int, float)):
        facts.append(["Tuning", f"{'kept' if tuning.get('accepted') else 'not kept'}: {_r(tuning['gain'])} against a required {_r(tuning.get('required_gain'))}"])
    return {"card": t.show("show_result", {
        "agent": "evaluator", "title": "The held-back result", "page": "reliability", "facts": [f for f in facts if f[1] is not None],
        "columns": [{"key": "baseline", "label": "Baseline"}, {"key": metric, "label": metric, "num": True}] if rows else [], "rows": rows})}


KEY_MEASURES = ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "rmse", "mae", "r2", "macro_f1", "balanced_accuracy")
LABELS = {"f1": "F1", "roc_auc": "ROC AUC", "average_precision": "Average precision", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
          "balanced_accuracy": "Balanced accuracy", "mcc": "MCC", "rmse": "RMSE", "mae": "MAE", "r2": "R²", "macro_f1": "Macro F1", "log_loss": "Log loss",
          "brier": "Brier score", "specificity": "Specificity"}


def show_evaluation(t: Turn, /, rows: str = "holdout", **_: Any) -> dict[str, Any]:
    """The evaluation's measures (accuracy, precision, recall, F1, the curves' areas) and the confusion matrix, on the held-back cases
    (``rows`` "holdout") or the training folds' out-of-fold predictions ("validation")."""
    from dclab_core import model_evaluation as me

    final = t.stage("final")
    if not final:
        return _stage_missing("final")
    ev = final.get("evidence") or {}
    if rows == "validation":
        got = (ev.get("evaluation") or {}).get("validation")
        if not got:
            return {"error": "This project's final stage ran before the folds' evaluation was kept: run the final stage again to have it."}
    else:
        made = me.from_record(ev)
        got = made.model_dump(mode="json") if made is not None else None
        if not got:
            return {"error": "The final stage's record holds no evaluation this page can read."}
    metrics = {k: _r(v) for k, v in (got.get("metrics") or {}).items() if v is not None}
    conf = got.get("confusion") or {}
    matrix = None
    if conf.get("matrix"):
        matrix = {"labels": conf.get("labels"), "matrix": conf["matrix"]}
    elif all(k in conf for k in ("tp", "fp", "fn", "tn")):
        matrix = {"labels": ["negative", "positive"], "matrix": [[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]]}
    first = [k for k in KEY_MEASURES if k in metrics]
    facts = [[LABELS.get(k, k.replace("_", " ")), metrics[k]] for k in first]
    more = [[LABELS.get(k, k.replace("_", " ")), v] for k, v in metrics.items() if k not in first]
    if matrix and len(matrix["matrix"]) == 2:  # the totals a reply names ("flagged 13, 6 right"): actual and predicted positives, all cases
        (tn, fp), (fn, tp) = matrix["matrix"]
        facts += [["Flagged as positive", tp + fp], ["Actual positives", tp + fn], ["Actual negatives", tn + fp]]
    if got.get("cutoff") is not None:
        facts.append(["Cutoff", _r(got["cutoff"])])
    facts.append(["Cases", got.get("n")])
    return {"card": t.show("show_evaluation", {
        "agent": "evaluator", "title": "Evaluation on the " + ("held-back cases" if rows != "validation" else "training folds (out of fold)"), "page": "evaluation",
        "subtitle": "read from the final stage's record: the held-back cases are not read again", "facts": facts, "more": more, "matrix": matrix,
        "notes": list(got.get("notes") or [])[:4]})}


def list_projects(t: Turn, /, **_: Any) -> dict[str, Any]:
    """The workspace's projects: each one's name, how far its stages ran, its model and held-back score (a private project shows its name only)."""
    rows = []
    for p in t.projects.list(limit=30):
        private = bool((p.get("settings") or {}).get("private"))
        done = sum(1 for st in (p.get("stages") or {}).values() if (st or {}).get("status") == "completed")
        row = {"project": p.get("name"), "stages_done": f"{done} of {len(p.get('stages') or {})}", "model": None, "score": None, "this_chat": p.get("id") == t.project_id}
        if not private and (p.get("stages") or {}).get("final", {}).get("status") == "completed":
            final = t.projects.read_stage(p["id"], "final") or {}
            ev, metric = final.get("evidence") or {}, final.get("primary_metric")
            row.update(model=ev.get("model"), score=f"{metric} {_r((ev.get('holdout_metrics') or {}).get(metric))}" if metric else None)
        rows.append(row)
    return {"card": t.show("list_projects", {
        "agent": "reader", "title": "Your projects", "subtitle": "most recently changed first; the score is each one's held-back result",
        "columns": [{"key": "project", "label": "Project"}, {"key": "stages_done", "label": "Stages"}, {"key": "model", "label": "Model"}, {"key": "score", "label": "Held-back score"}],
        "rows": rows, "mark": "this_chat"})}


def show_notebook(t: Turn, /, **_: Any) -> dict[str, Any]:
    """The cells added to this project's notebook (by the Notebook coder or by the person): what each is for and its status."""
    cells = t.project().get("notebook_agent_cells") or []
    if not cells:
        return {"error": "No cell has been added to this project's notebook yet. The Notebook coder can write one (write_notebook_cell)."}
    rows = [{"cell": c.get("id"), "for": str(c.get("request") or "")[:120], "by": "you" if c.get("author") == "person" else "the Notebook coder", "status": c.get("status")}
            for c in cells[-20:]]
    return {"card": t.show("show_notebook", {
        "agent": "coder", "title": "The notebook's added cells", "page": "notebook",
        "columns": [{"key": "cell", "label": "Cell"}, {"key": "for", "label": "For"}, {"key": "by", "label": "Written by"}, {"key": "status", "label": "Status"}], "rows": rows})}


ACTIVITY_KEYS = ("stage", "model", "recipe", "family", "tried", "clear", "best", "state", "status", "seconds", "elapsed_seconds")


def show_activity(t: Turn, /, **_: Any) -> dict[str, Any]:
    """What happened in this project, most recent first: stages run, improvements tried, choices made."""
    rows = []
    for e in reversed(t.projects.activity(t.project_id, limit=25)):
        payload = e.get("payload") if isinstance(e.get("payload"), dict) else {}
        what = ", ".join(f"{k} {payload[k]}" for k in ACTIVITY_KEYS if isinstance(payload.get(k), (str, int, float)))  # known fields only: a payload can hold a person's words
        rows.append({"at": str(e.get("at") or "")[:16].replace("T", " "), "event": str(e.get("kind") or "").replace("_", " "), "what": str(what)[:160]})
    if not rows:
        return {"error": "Nothing is recorded in this project's activity yet."}
    return {"card": t.show("show_activity", {
        "agent": "reader", "title": "What happened in this project", "page": "project",
        "columns": [{"key": "at", "label": "When (UTC)"}, {"key": "event", "label": "Event"}, {"key": "what", "label": "What"}], "rows": rows})}


def improvement_ideas(t: Turn, /, **_: Any) -> dict[str, Any]:
    """What the project's own records say would help, each from a number in them, and the offer to try them."""
    from ..studio import agent as studio_agent

    if not t.stage("final"):
        return _stage_missing("final")
    lines, proof = studio_agent.improvements(t.records())
    t.agent_offer("improve_model")
    return {"card": t.show("improvement_ideas", {"agent": "experimenter", "title": "Ways this model could improve", "page": "reliability", "lines": lines,
                                                  "proof": proof[:8], "offer": "Say yes and the Experimenter tries them on the training folds now."})}


def show_improvements(t: Turn, /, **_: Any) -> dict[str, Any]:
    """The last improvement run: every way tried, its score on the training folds, its gain and whether it cleared the margin."""
    rec = t.project().get("improvements") or {}
    if not rec.get("candidates"):
        t.agent_offer("improve_model")
        return {"error": "No improvement run has finished in this project yet. Offer to run one (improve_model)."}
    rows = [{"way": c["label"], "aspect": c["aspect"], "mean": _r(c["mean"]), "gain": _r(c.get("gain")), "folds_better": c.get("folds_better"),
             "verdict": "clears the margin" if c["clears"] else "within the noise", "adoptable": bool(c["clears"] and c["adoptable"]), "id": c["id"]}
            for c in rec["candidates"]]
    cur = rec.get("current") or {}
    return {"card": t.show("show_improvements", {
        "agent": "experimenter", "title": "The last improvement run", "page": "reliability", "subtitle": f"{rec.get('metric')} on {rec.get('folds')} training folds · {rec.get('ran_at')}",
        "facts": [["Current model", f"{cur.get('family')} on {cur.get('recipe')}"], ["Its score on the folds", _r(cur.get("mean"))],
                  ["Best way that can be adopted", rec.get("best") or "none"], ["Adopted", rec.get("adopted") or "nothing yet"]],
        "columns": [{"key": "way", "label": "Way"}, {"key": "mean", "label": "Score on the folds", "num": True}, {"key": "gain", "label": "Gain", "num": True},
                    {"key": "folds_better", "label": "Folds better", "num": True}, {"key": "verdict", "label": "Verdict"}],
        "rows": rows, "note": rec.get("note")})}


def improve_model(t: Turn, /, **_: Any) -> dict[str, Any]:
    """Start the Experimenter: the improvement job on the training folds only (class weights, the next model families with their own
    settings, another feature recipe, a blend), each compared with the current model on the same folds. It reports in the chat."""
    if not all(t.stage(k) for k in ("features", "models", "final")):
        return {"error": "Run the project's stages up to the final one first: there is no model to improve yet."}
    p = t.project()
    if p.get("running"):
        return {"error": f"Something is running in this project ({p['running']}): wait for it to finish, or stop it."}
    going = _improving(t)
    if going is not None:
        return {"error": f"An improvement run is already {going['status']} for this project (job {going['id']}): its report comes here when it ends.",
                "running": going["id"]}
    submit = getattr(t.services, "submit", None)
    if submit is None:
        return {"error": "No background worker is available here to run the improvement."}
    try:
        job = submit("improve", f"{t.project_id}:improve", {"project_id": t.project_id, "draft_id": t.draft_id}, here=False)
    except Exception as exc:  # noqa: BLE001 - busy (409) or refused: told back as it is
        return {"error": str(getattr(exc, "detail", None) or exc)}
    job_id = (job or {}).get("id")
    t.agent_offer(None)
    t.task("improve_model", TEAM["experimenter"], "Trying ways to improve the model, on the training folds only", "running", job=job_id,
           detail="class weights · the next model families · another feature recipe · a blend")
    t.results.append({"started": True, "job": job_id})
    return {"started": True, "job": job_id, "ways": ["class weights", "the next two model families, two settings each", "another feature recipe", "a blend of the best"],
            "reports": "in this chat when it is done; the held-back cases are not touched"}


def _improving(t: Turn) -> Optional[dict[str, Any]]:
    """The project's improvement job when one is queued or running (a second "ok" must not start another)."""
    store = getattr(t.services, "job_store", None)
    latest = getattr(store, "latest", None)
    try:
        job = latest("improve", f"{t.project_id}:improve") if latest is not None else None
    except Exception:  # noqa: BLE001 - a job store that cannot answer: the submit's own check stands
        return None
    return job if job and job.get("status") in ("queued", "running") else None


def adopt_improvement(t: Turn, /, candidate: str = "", **_: Any) -> dict[str, Any]:
    """Adopt a way from the last improvement run that cleared the margin (the best one when none is named): the next final run uses it."""
    from ..studio import improve

    rec = t.project().get("improvements") or {}
    way = candidate or rec.get("best") or ""
    if not way:
        return {"error": "No way from the last improvement run cleared the margin and can be adopted here."}
    try:
        adopted = improve.adopt_candidate(t.projects, t.project_id, way)
    except (KeyError, ValueError) as exc:
        return {"error": str(exc.args[0] if exc.args else exc)}
    result = {"adopted": adopted["id"], "family": adopted["family"], "gain": _r(adopted.get("gain")),
              "next": "Run the final stage again with a written reason: it scores the held-back cases a second time, so that estimate is no longer a fresh one."}
    t.task("adopt_improvement", TEAM["experimenter"], f"Adopted {adopted['id']}", "done")
    t.results.append(result)
    return result


def write_notebook_cell(t: Turn, /, request: str = "", **_: Any) -> dict[str, Any]:
    """The Notebook coder writes a cell for a plot or an analysis (training rows only; checked before it is shown). It is shown in the chat
    with a button to run it; it runs only when the person says so."""
    from ..studio import notebook_agent

    request = (request or t.text).strip()
    p = t.project()
    if not p.get("solution"):
        return {"error": "Save the solution first: the notebook is generated from it."}
    gateway = getattr(t.services, "gateway", None)
    client = gateway.client("notebook_cell", project_id=t.project_id) if gateway is not None else None
    if client is None:
        return {"error": "No model is configured to write notebook cells here."}
    try:
        cell = notebook_agent.write(p, t.records(), request, client)
    except Exception as exc:  # noqa: BLE001 - the export does not support this kind, or the model failed: said
        return {"error": f"The cell could not be written ({type(exc).__name__})."}  # never the error's text: it can echo the request
    with notebook_agent.lock(t.project_id):
        cells = t.projects.get(t.project_id).get("notebook_agent_cells") or []
        t.projects.update(t.project_id, notebook_agent_cells=notebook_agent.keep(list(cells) + [cell]))
    t.agent.say(t.draft_id, "", kind="notebook_cell", cell={k: cell.get(k) for k in ("id", "request", "code", "explanation", "problems", "status")}, project=t.project_id)
    t.task("write_notebook_cell", TEAM["coder"], f"A cell for: {request[:80]}", "done" if cell["status"] == "proposed" else "failed")
    result = {"cell": cell["id"], "status": cell["status"], "explanation": cell.get("explanation"), "problems": cell.get("problems"),
              "runs": "when the person presses Run on it, on the training rows only"}
    t.results.append(result)
    return result


READS = {"list_projects": list_projects, "show_notebook": show_notebook, "show_activity": show_activity, "show_models": show_models, "show_features": show_features, "show_data": show_data, "show_leakage": show_leakage, "show_stages": show_stages,
         "show_result": show_result, "show_evaluation": show_evaluation, "improvement_ideas": improvement_ideas, "show_improvements": show_improvements}
WRITES = {"improve_model": ("improve", improve_model), "adopt_improvement": ("adopt_improvement", adopt_improvement),
          "write_notebook_cell": ("notebook_cell", write_notebook_cell)}
WHO = {"list_projects": "reader", "show_notebook": "coder", "show_activity": "reader", "show_models": "reader", "show_features": "reader", "show_data": "reader", "show_leakage": "reader", "show_stages": "reader",
       "show_result": "evaluator", "show_evaluation": "evaluator", "improvement_ideas": "experimenter", "show_improvements": "experimenter",
       "improve_model": "experimenter", "adopt_improvement": "experimenter", "write_notebook_cell": "coder"}


class HomeGuard:
    """Every write of the scope "home" runs for the chat's own project, never while it runs something else, and is audited."""

    def __call__(self, tool: Tool, turn: Any, arguments: dict[str, Any], proceed: Callable[[], Any]) -> Any:
        from .. import audit

        if not isinstance(turn, Turn):
            return {"error": f"{tool.name} runs only inside a project's chat."}
        try:
            project = turn.project()
        except Exception:  # noqa: BLE001 - the project was deleted meanwhile
            return {"error": "This chat's project no longer exists."}
        if project.get("running"):  # a job saves this project as it runs: a write now could lose its update or leave it marked running
            return {"error": f"Something is running in this project ({project['running']}): wait for it to finish, or stop it."}
        if tool.name == "adopt_improvement" and _improving(turn) is not None:
            return {"error": "An improvement run is still going: adopt from its report when it ends."}
        result = proceed()
        ok = isinstance(result, dict) and "error" not in result
        try:
            audit.record_for(turn.projects, "move", "agent", "home", move=tool.move, status="allowed" if ok else "refused", project_id=turn.project_id,
                             draft_id=turn.draft_id, message=(result or {}).get("error") if not ok else f"home orchestrator: {tool.name}")
        except Exception:  # noqa: BLE001 - the audit is written where it can be
            pass
        return result


def register(registry: Registry) -> None:
    empty = {"type": "object", "properties": {}}
    for name, fn in READS.items():
        params = {"type": "object", "properties": {"rows": {"type": "string", "enum": ["holdout", "validation"]}}} if name == "show_evaluation" else empty
        registry.register(Tool(name, f"[{TEAM[WHO[name]]}] " + " ".join((fn.__doc__ or "").split()), params, fn, scope="home", takes_context=True))
    for name, (move, fn) in WRITES.items():
        params = {"type": "object", "properties": {"candidate": {"type": "string"}}} if name == "adopt_improvement" else \
            {"type": "object", "properties": {"request": {"type": "string"}}, "required": ["request"]} if name == "write_notebook_cell" else empty
        registry.register(Tool(name, f"[{TEAM[WHO[name]]}] " + " ".join((fn.__doc__ or "").split()), params, fn, effect="write", scope="home", move=move,
                               takes_context=True))
    registry.guards["home"] = HomeGuard()


# ---------------------------------------------------------------------- the rules' plan (no model, or the model failed)


YES = re.compile(r"^\s*(?:ok(?:ay)?|yes|yeah|yep|yup|sure|fine|alright|go(?: ahead| on)?|do it|please(?: do)?|let'?s(?: do it| go)?|start|run it|proceed|"
                 r"try (?:it|them)|of course|perfect|great)(?:[\s,.!]+(?:ok(?:ay)?|yes|do it|go ahead|please|now|then|thanks|thank you|try them|run it))*[\s.!]*$", re.I)
NO = re.compile(r"^\s*(?:no|nope|not now|later|cancel|stop|don'?t)\b", re.I)
IMPROVE = re.compile(r"\b(?:improv\w*|better|boost\w*|enhanc\w*|tun(?:e|ing)|optimi[sz]\w*|increase|raise|higher score)\b", re.I)
DOING = re.compile(r"\b(?:for me|do it|go ahead|build|run|try|start|make|practically|actually|now|please|launch|go)\b|^\s*(?:ok(?:ay)?\s+)?(?:improve|build|run|try|tune|boost)\b", re.I)
ASKING = re.compile(r"\b(?:how|what|which|where|why|can (?:we|i|you)|could (?:we|i|you)|should|ideas?|ways?|suggest\w*|options?|aspects?)\b|\?", re.I)
ADOPT = re.compile(r"\b(?:adopt\w*|use (?:that|the best|it)|apply (?:it|that|the best))\b", re.I)
PLOT = re.compile(r"\b(?:plot\w*|chart\w*|graph\w*|visuali[sz]\w*|histogram|draw|notebook|cell|code|heat ?map|curve)\b", re.I)
WORDS = [  # a read named by its words, in this order
    ("list_projects", re.compile(r"\b(?:my projects|other projects|all (?:the )?projects|list (?:the |my )?projects|projects? (?:list|i have))\b", re.I)),
    ("show_notebook", re.compile(r"\b(?:notebook'?s? cells|cells? (?:added|written|so far)|what cells)\b", re.I)),
    ("show_activity", re.compile(r"\b(?:activity|history|what happened|log of|timeline)\b", re.I)),
    ("show_improvements", re.compile(r"\b(?:improvement run|last run|what did you try|ways? (?:you )?tried|the report|results? of the improvement)\b", re.I)),
    ("show_evaluation", re.compile(r"\b(?:evaluat\w*|precision|recall|f1|f-1|confusion|accuracy|auc|roc|true positives?|false (?:positives?|alarms?|negatives?)|metrics?)\b", re.I)),
    ("show_models", re.compile(r"\b(?:models?|algorithms?|famil(?:y|ies)|trained|traimed|train\w*|best(?: one)?|which one|winner|top one|leaderboard|lightgbm|xgboost|logistic|random forest|boosting)\b", re.I)),
    ("show_result", re.compile(r"\b(?:holdout|held[- ]back|final (?:score|result)|results?|score|how good|performance|trust|interval|baseline)\b", re.I)),
    ("show_features", re.compile(r"\b(?:features?|recipes?|engineer\w*|columns? used|inputs?)\b", re.I)),
    ("show_leakage", re.compile(r"\b(?:leak\w*|forbidden|after the outcome|unavailable)\b", re.I)),
    ("show_data", re.compile(r"\b(?:data(?:set)?|rows?|table|outcome|target|missing|drift|duplicates?)\b", re.I)),
    ("show_stages", re.compile(r"\b(?:stages?|status|progress|running|done yet|finished)\b", re.I)),
]
CHIPS = {"show_models": ["Show the evaluation", "Improve the model for me", "Plot each fold's score"],
         "show_evaluation": ["Show the held-back result", "How can it improve?", "Plot the confusion matrix"],
         "show_result": ["Show the evaluation", "Show the models trained", "How can it improve?"],
         "improvement_ideas": ["Yes, try them", "Show the models trained"],
         "show_improvements": ["Adopt the best way", "Show the models trained"],
         "improve_model": ["Show the models trained", "Show the evaluation"],
         "adopt_improvement": ["Show the held-back result"], "write_notebook_cell": ["Show the evaluation", "Show the models trained"],
         None: ["Show the models trained", "Show the evaluation", "How can it improve?", "Improve the model for me"]}


def plan(text: str, offer: Optional[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """The rules' reading of one message: the tools to call, in order. A yes carries out the pending offer; words name the rest."""
    said = text.strip()
    if offer and YES.match(said):
        return [(offer["tool"], dict(offer.get("args") or {}))]
    if NO.match(said):
        return []
    if ADOPT.search(said):
        return [("adopt_improvement", {})]
    if PLOT.search(said) and not dict(WORDS)["show_notebook"].search(said):  # "what cells were added" lists them; "plot ..." writes one
        return [("write_notebook_cell", {"request": said})]
    out: list[tuple[str, dict[str, Any]]] = []
    if IMPROVE.search(said):
        if DOING.search(said) and not ASKING.search(said):
            return [("improve_model", {})]
        if not dict(WORDS)["show_improvements"].search(said):  # "what did you try to improve?" is the last run, not the ideas
            out.append(("improvement_ideas", {}))
    for name, words in WORDS:
        if words.search(said) and name not in [n for n, _ in out] and not (out and name in ("show_models", "show_result") and IMPROVE.search(said)):
            out.append((name, {"rows": "validation"} if name == "show_evaluation" and re.search(r"\b(?:folds?|validation|cross)\b", said, re.I) else {}))
    return out[:4]


# ---------------------------------------------------------------------- numbers in a model's reply must be numbers a tool returned


NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?%?")


def _numbers(value: Any) -> list[float]:
    out: list[float] = []
    for m in NUMBER.findall(json.dumps(value, default=str)):
        try:
            out.append(float(m.rstrip("%").replace(",", "")))
        except ValueError:
            pass
    return out


def grounded(text: str, results: list[Any]) -> tuple[str, list[str]]:
    """The reply with each sentence that gives a number no tool returned dropped (a number may be rounded or a percentage of one)."""
    known = _numbers(results)
    lines_out, dropped = [], []
    for line in text.strip().split("\n"):  # the reply's lines (a list, a heading) are kept as lines
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if _ungrounded(sentence, known):
                dropped.append(sentence)
            else:
                kept.append(sentence)
        if kept or not line.strip():
            lines_out.append(" ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines_out)).strip(), dropped


LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def without_lists(text: str) -> tuple[str, list[str]]:
    """The reply without its list lines and Markdown headings (the table above already shows them), and the lines taken out."""
    kept, out = [], []
    for line in text.split("\n"):
        (out if LIST_LINE.match(line) or line.lstrip().startswith("#") else kept).append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip(), out


def _ungrounded(sentence: str, known: list[float]) -> bool:
    """True when the sentence gives a number no tool returned (a rounding, a percentage of one or a small count are fine)."""
    for raw in NUMBER.findall(sentence):
        pct, value = raw.endswith("%"), raw.rstrip("%").replace(",", "")
        try:
            v = float(value)
        except ValueError:
            continue
        if "." not in value and abs(v) <= 12:  # a small count ("two folds", "step 3", "top 5") is wording, not a measurement
            continue
        places = len(value.split(".")[1]) if "." in value else 0
        tol = 0.5 * 10 ** -places + 1e-9
        if not any(abs(k - v) <= tol or (pct and abs(k * 100 - v) <= tol) or abs(round(k, places) - v) <= 1e-9 for k in known):
            return True
    return False


# ---------------------------------------------------------------------- the orchestrator


class Orchestrator:
    """One turn of the Home orchestrator on a built project's chat."""

    def __init__(self, services: Any, agent: Any, gateway: Any = None):
        self.services, self.agent = services, agent
        self.gateway = gateway if gateway is not None else getattr(services, "gateway", None)

    # -- state the turn reads

    def offer(self, draft_id: str) -> Optional[dict[str, Any]]:
        return ((self.agent.store.get(draft_id).get("agent") or {}).get("offer")) or None

    def set_offer(self, draft_id: str, tool: Optional[str], args: Optional[dict[str, Any]] = None) -> None:
        value = {"tool": tool, "args": args or {}, "at": now()} if tool else None
        self.agent.store.update(draft_id, lambda d: d.setdefault("agent", {}).update(offer=value))

    def situation(self, turn: Turn) -> dict[str, Any]:
        """What the model is told of the project before it plans: stages, what runs, the chosen model and score, the last improvement, the offer."""
        p = turn.project()
        final = (turn.stage("final") or {})
        ev = final.get("evidence") or {}
        metric = final.get("primary_metric")
        rec = p.get("improvements") or {}
        return {"project": p.get("name"), "task": (p.get("solution") or {}).get("task"), "outcome": (p.get("solution") or {}).get("target"),
                "stages": {k: (v or {}).get("status") for k, v in (p.get("stages") or {}).items()}, "running": p.get("running"),
                "model": ev.get("model"), "metric": metric, "held_back_score": _r((ev.get("holdout_metrics") or {}).get(metric)) if metric else None,
                "last_improvement_run": {"ran_at": rec.get("ran_at"), "tried": len(rec.get("candidates") or []), "best": rec.get("best"), "adopted": rec.get("adopted")} if rec else None,
                "working_now": self.working(turn.draft_id), "pending_offer": (turn.offered or {}).get("tool"), "team": TEAM}

    def working(self, draft_id: str) -> list[dict[str, Any]]:
        """The board tasks of this chat still running (an improvement job under way), with their job's status now."""
        out = []
        for m in self.agent.store.get(draft_id).get("messages") or []:
            for x in m.get("tasks") or [] if m.get("kind") == "task_board" else []:
                if x.get("status") == "running":
                    job = self._job(x["job"]) if x.get("job") else None
                    out.append({"agent": x.get("agent"), "task": x.get("title"), "job": x.get("job"), "job_status": (job or {}).get("status"),
                                "progress": (job or {}).get("progress") or None})
        return out

    def history(self, draft_id: str) -> list[dict[str, Any]]:
        """The conversation's last messages as the model reads them: the person's words (redacted) and what DCLab said or showed."""
        from dclab_core import redact

        out = []
        for m in (self.agent.store.get(draft_id).get("messages") or [])[-TAIL - 1:-1]:
            if m.get("role") == "user":
                out.append({"role": "user", "content": redact.redact_text(str(m.get("text") or ""))[0]})
                continue
            said = str(m.get("text") or "")
            if m.get("kind") == "project_card":
                said = f"[showed the card: {(m.get('card') or {}).get('title')}]"
            elif m.get("kind") == "notebook_cell":
                said = f"[showed a notebook cell for: {(m.get('cell') or {}).get('request')}]"
            elif m.get("kind") == "task_board":
                said = "[task board: " + "; ".join(f"{x.get('agent')}: {x.get('title')} ({x.get('status')})" for x in m.get("tasks") or []) + "]"
            elif m.get("kind") == "improvement_report":
                said = "[improvement report] " + said
            if said:
                out.append({"role": "assistant", "content": said[:1500]})
        merged: list[dict[str, Any]] = []
        for m in out:  # a model API wants turns that alternate
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"] += "\n" + m["content"]
            else:
                merged.append(dict(m))
        while merged and merged[0]["role"] != "user":
            merged.pop(0)
        return merged

    # -- the turn

    def turn(self, draft_id: str, project_id: str, text: str) -> dict[str, Any]:
        """Answer one message: plan with the model (or the rules), run the specialists, reply, follow any job on a board; keep it all."""
        t = Turn(self.services, project_id, draft_id, self.agent, text, offer=lambda tool, args=None: self.set_offer(draft_id, tool, args))
        t.offered = self.offer(draft_id)  # an offer answers the next message only: "how can it improve?", "show the evaluation", "thanks" starts nothing
        if t.offered:
            self.set_offer(draft_id, None)  # a tool that offers again this turn sets it again
        self._refresh_boards(draft_id)
        record: dict[str, Any] = {"at": now(), "draft": draft_id, "project": project_id, "text": text}
        client = self.gateway.client(PURPOSE, project_id=project_id) if self.gateway is not None else None
        reply, path = None, "rules"
        if client is not None:
            try:
                reply = self.model_turn(t, client, record)
                path = "model"
            except Exception as exc:  # noqa: BLE001 - the model failed or its cap was reached: the rules carry this turn
                record["model_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                if t.tasks or t.shown:  # tools already ran this turn: say what they did; planning again would run a write twice
                    reply, path = self._rules_words(t), "model"
        if reply is None:
            reply = self.rules_turn(t, record)
        record.update(path=path, reply=reply, tasks=t.tasks, cards=[c.get("title") for c in t.shown.values()])
        self._finish(t, reply, path)
        self._keep(record)
        return record

    def model_turn(self, t: Turn, client: Any, record: dict[str, Any]) -> str:
        from dclab_core import redact

        from .. import prompts
        from ..agents import default_registry
        from ..agents.harness import tracer
        from ..agents.runtime import Policy, run as run_policy

        if getattr(client, "run_limit_eur", None) is None:
            client.run_limit_eur = CAP_EUR
        registry = default_registry()
        policy = Policy(PURPOSE, prompts.text("home_orchestrator"), tuple(registry.names("home")), max_steps=6, max_replies=5)
        state = json.dumps(self.situation(t), default=str, ensure_ascii=False)
        now_said = f"{redact.redact_text(t.text)[0]}\n\n[the project now: {state}]"
        past = self.history(t.draft_id)
        if past and past[-1]["role"] == "user":  # the turns alternate: an unanswered message joins this one
            past[-1] = {"role": "user", "content": past[-1]["content"] + "\n" + now_said}
            messages = [{"role": "system", "content": policy.system}, *past]
        else:
            messages = [{"role": "system", "content": policy.system}, *past, {"role": "user", "content": now_said}]
        out = run_policy(policy, client, registry, messages, context=t, contexts={"home": t}, trace=tracer(AGENT))
        record["steps"] = [{"n": s.n, "tool": s.tool, "arguments": s.arguments, "ok": s.ok, "result": s.result, "seconds": round(s.seconds, 3)} for s in out.steps]
        record["usage"], record["stopped"] = out.usage, out.stopped
        record["spent_eur"] = round(float(getattr(client, "spent_eur", 0.0) or 0.0), 6)
        said = (out.final or "").strip()
        for s in out.steps:
            if s.ok and isinstance(s.result, dict) and "error" not in s.result:
                t.results.append(s.result)
            elif isinstance(s.result, dict) and s.result.get("error"):
                t.results.append(s.result)  # an error's words may be repeated
                t.task(s.tool, TEAM[WHO.get(s.tool, "reader")], s.tool.replace("_", " "), "failed", detail=str(s.result["error"])[:200])
        kept, dropped = grounded(said, [t.results, self.situation(t)])
        if t.shown:  # the owner, 2026-10-20: "bring the table instead of the texts": a list that repeats a shown table is dropped
            kept, listed = without_lists(kept)
            dropped += listed
            sentences = re.split(r"(?<=[.!?])\s+", kept)
            if len(sentences) > SHOWN_SENTENCES:  # the table speaks; the words say what it means and the next step (the whole reply stays in the record)
                kept = " ".join(sentences[:SHOWN_SENTENCES - 1] + sentences[-1:])
        record["said"], record["dropped"] = said, dropped
        if not kept:
            return self._rules_words(t) if t.shown or t.tasks else "I could not answer that from this project's records. Here is what I can do:"
        return kept

    def rules_turn(self, t: Turn, record: dict[str, Any]) -> str:
        from ..agents import default_registry, harness

        registry = default_registry()
        steps = plan(t.text, t.offered)
        record["plan"] = [{"tool": n, "arguments": a} for n, a in steps]
        with harness.session(RULES_AGENT) as s:
            for name, args in steps:
                with s.timed(name, args) as out:
                    out["result"] = registry.call(name, args, context=t, scope="home")
                if isinstance(out["result"], dict) and out["result"].get("error"):
                    t.results.append(out["result"])
                    t.task(name, TEAM[WHO.get(name, "reader")], name.replace("_", " "), "failed", detail=str(out["result"]["error"])[:200])
        record["steps"] = [{"tool": n, "arguments": a} for n, a in steps]
        if not steps:
            if NO.match(t.text.strip()) and t.offered:
                return "All right, I won't start it. Ask for anything else about this project."
            return "Here is what I can do for this project. Pick one, or ask in your own words:"
        return self._rules_words(t)

    def _rules_words(self, t: Turn) -> str:
        """The reply the rules write: what each specialist did, said plainly (an error in its own words)."""
        parts = []
        for task in t.tasks:
            if task["status"] == "failed":
                parts.append(f"{task['agent']}: {task.get('detail') or 'it could not do that'}")
            elif task["tool"] == "improve_model":
                parts.append("The Experimenter is trying ways to improve the model now, on the training folds only (the held-back cases stay untouched): "
                             "class weights, the next model families with their own settings, another feature recipe and a blend. Each is compared with the "
                             "current model on the same folds and counts as a gain only if it beats the margin set in advance and the folds' noise. "
                             "The board below follows it, and the report comes here when it is done.")
            elif task["tool"] == "adopt_improvement":
                r = next((x for x in t.results if isinstance(x, dict) and x.get("adopted")), {})
                parts.append(f"Adopted {r.get('adopted')} ({r.get('family')}). {r.get('next')}")
            elif task["tool"] == "write_notebook_cell":
                parts.append("The Notebook coder wrote this cell: read it, then press Run to run it on the training rows only.")
            elif task["tool"] == "improvement_ideas":
                parts.append("These are the ways this project's own records point to. Say yes and the Experimenter tries them on the training folds now.")
            else:
                parts.append(f"{task['agent']}: here is {task['title'][0].lower() + task['title'][1:]}.")
        return "\n".join(parts) or "Done."

    def _finish(self, t: Turn, reply: str, path: str) -> None:
        last = t.tasks[-1]["tool"] if t.tasks else None
        chips = [c for c in CHIPS.get(last if last in CHIPS else None, CHIPS[None])]
        if t.shown.get("improvement_ideas") is None and (self.offer(t.draft_id) or {}).get("tool") == "improve_model" and "Yes, try them" not in chips:
            chips = ["Yes, try them", *chips]
        team = sorted({x["agent"] for x in t.tasks})
        self.agent.say(t.draft_id, reply, kind="project_answer", chips=chips[:4], team=team, path=path)
        jobs = [x for x in t.tasks if x.get("job")]
        if jobs:
            board = self.agent.say(t.draft_id, "", kind="task_board", tasks=[{k: v for k, v in x.items() if k != "tool"} for x in t.tasks])
            for x in jobs:
                self.watch(t.draft_id, board["id"], x["id"], x["job"])

    # -- the task board

    def _board(self, draft_id: str, board_id: str, task_id: str, **changes: Any) -> None:
        def change(d: dict[str, Any]) -> None:
            for m in d.get("messages") or []:
                if m.get("id") == board_id:
                    for x in m.get("tasks") or []:
                        if x.get("id") == task_id:
                            x.update(changes)
        self.agent.store.update(draft_id, change)
        board = next((m for m in self.agent.store.get(draft_id).get("messages") or [] if m.get("id") == board_id), None)
        if board is not None:
            self.agent.store.emit(draft_id, "board", {"id": board_id, "tasks": board.get("tasks")})

    def _job(self, job_id: str) -> Optional[dict[str, Any]]:
        store = getattr(self.services, "job_store", None)
        try:
            return store.get(job_id) if store is not None else None
        except Exception:  # noqa: BLE001 - an unknown job
            return None

    def settle(self, draft_id: str, board_id: str, task_id: str, job: dict[str, Any]) -> bool:
        """Bring one board task up to its job; True when the job has ended (its end said in the chat when it did not finish)."""
        board = next((m for m in self.agent.store.get(draft_id).get("messages") or [] if m.get("id") == board_id), {})
        task = next((x for x in board.get("tasks") or [] if x.get("id") == task_id), {})
        if task.get("status") in ("done", "failed"):  # settled already (the watcher and a turn's refresh may both look)
            return True
        status = job.get("status")
        if status in ("queued", "pending", "running", "claimed", None):
            prog = job.get("progress") or {}
            if prog.get("way"):
                self._board(draft_id, board_id, task_id, status="running", detail=f"step {prog.get('step')} of {prog.get('of')}: {prog.get('way')}")
            return False
        if status == "done":
            self._board(draft_id, board_id, task_id, status="done", detail="finished: its report is in this chat")
            return True
        why = str(job.get("error") or status)
        hint = (" The background worker is running older code that does not know this task: restart the app (make dev) and ask again."
                if "no handler" in why else "")
        self._board(draft_id, board_id, task_id, status="failed", detail=why[:300])
        self.agent.say(draft_id, f"The Experimenter's run did not finish ({status}): {why[:300]}.{hint}", kind="text", severity="warning")
        return True

    def watch(self, draft_id: str, board_id: str, task_id: str, job_id: str, every: float = 2.0, seconds: float = WATCH_SECONDS) -> threading.Thread:
        """Follow a job on the board until it ends, in a thread of this process (a restart is caught up by the next turn's refresh)."""
        def follow() -> None:
            ends = time.monotonic() + seconds
            while time.monotonic() < ends:
                job = self._job(job_id)
                if job is not None:
                    try:
                        if self.settle(draft_id, board_id, task_id, job):
                            return
                    except Exception:  # noqa: BLE001 - the draft was deleted meanwhile
                        return
                time.sleep(every)
        import contextvars

        # the request's context goes with the thread: the services are a proxy read from it (the workspace of this chat, not the default one)
        thread = threading.Thread(target=contextvars.copy_context().run, args=(follow,), name=f"home-board-{job_id}", daemon=True)
        thread.start()
        return thread

    def _refresh_boards(self, draft_id: str) -> None:
        """Before a turn: every board task still running is brought up to its job (a job that ended while nothing followed it)."""
        for m in self.agent.store.get(draft_id).get("messages") or []:
            if m.get("kind") != "task_board":
                continue
            for x in m.get("tasks") or []:
                if x.get("job") and x.get("status") == "running":
                    job = self._job(x["job"])
                    if job is not None:
                        self.settle(draft_id, m["id"], x["id"], job)
                    elif getattr(self.services, "job_store", None) is not None:  # a job the queue no longer knows never ends: said, not left running
                        self._board(draft_id, m["id"], x["id"], status="failed", detail="this job is no longer in the queue, so its end is unknown")

    def _keep(self, record: dict[str, Any]) -> None:
        """The turn kept whole (never cut), one line a turn, one file a day, under the workspace."""
        home = getattr(getattr(self.services, "store", None), "home", None) or getattr(self.services.projects, "home", None)
        if not home:
            return
        folder = Path(home) / "home_orchestrator"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


__all__ = ["Orchestrator", "Turn", "HomeGuard", "register", "plan", "grounded", "TEAM", "PURPOSE", "AGENT", "RULES_AGENT"]
