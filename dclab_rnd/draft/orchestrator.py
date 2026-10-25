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
TEAM = {"analyst": "Problem analyst", "planner": "Solution planner", "reader": "Records reader", "evaluator": "Evaluator", "experimenter": "Experimenter", "coder": "Notebook coder",
        "reporter": "Reporter", "data": "Data engineer"}
CAP_EUR = 0.25  # one turn's model spend, at most
TAIL = 12  # messages of the conversation the model reads
SHOWN_SENTENCES = 4  # a reply beside a table: at most this many sentences (the owner: "bring the table instead of the texts")
NUMBERS = {"one": 1, "a single": 1, "single": 1, "two": 2, "a couple of": 2, "three": 3, "four": 4, "five": 5}
LENGTH = re.compile(r"\b(?:in|with|using|within|to)\s+(?:just\s+|only\s+|at most\s+)?(one|a single|two|a couple of|three|four|five|[1-5])\s+(?:short\s+)?sentences?\b|"
                    r"\bin (one|a single) (?:line|sentence)\b|\b(one|two|three|four|five|[1-5])[- ]sentences?\b", re.I)


def asked_length(text: str) -> Optional[int]:
    """How many sentences the person asked for ("in two sentences", "in one line"), or None."""
    m = LENGTH.search(text or "")
    if not m:
        return None
    word = next(g for g in m.groups() if g).lower()
    return int(word) if word.isdigit() else NUMBERS.get(word)


def shorten(text: str, most: int) -> str:
    """At most ``most`` sentences: the first ones, and the last two together, since a reply ends with its next step and the
    request it leads to ("..., so I can plan that version next. Do you want me to?"; "Say \"plan it\""): a request never
    stands alone without the step it names (live, 2026-10-21)."""
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]
    if len(sentences) <= most:
        return text.strip()
    tail = sentences[-2:] if most >= 3 else sentences[-1:]
    return " ".join(sentences[:most - len(tail)] + tail)
WATCH_SECONDS = 3 * 3600  # how long a job is followed on the board


_WRITTEN_CALL = re.compile(r"```|\"(name|arguments)\"\s*:|\{\s*\w+\s*:|\[showed the table|\bcall the `?\w+`? (function|tool)\b", re.I)


def _not_an_answer(said: str, stopped: str) -> bool:
    """A reply that is a tool call written as text (a small model's habit, a known tool or not) or a word or two: nothing
    ran and nothing was said, so the rules answer instead (found with qwen2.5-coder:1.5b on Ollama, 2026-10-21)."""
    return stopped == "tools_as_text" or bool(_WRITTEN_CALL.search(said)) or len(re.findall(r"\w+", said)) < 3


class ModelUnhelpful(RuntimeError):
    """The model's turn ran nothing and said nothing grounded: the rules answer instead."""


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
        self.change_unclear = False  # the words change the problem but name nothing the rules can read

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


WAIT = re.compile(r"\b(?:first|not now|later|wait|not yet|hold on|don'?t|do not|instead|before (?:that|you|it))\b", re.I)


def _asked_to_improve(t: Turn) -> bool:
    """The person's own message starts the run: a yes to the offer, or a request to improve (not a question), never held back."""
    said = t.text.strip()
    if WAIT.search(said) or NO.match(said):
        return False
    if (t.offered or {}).get("tool") == "improve_model" and (YES.match(said) or YES_THEN.match(said)):
        return True
    return bool(IMPROVE.search(said) and DOING.search(said) and not ASKING.search(said))


def improve_model(t: Turn, /, **_: Any) -> dict[str, Any]:
    """Start the Experimenter: the improvement job on the training folds only (class weights, the next model families with their own
    settings, another feature recipe, a blend), each compared with the current model on the same folds. It reports in the chat."""
    if not all(t.stage(k) for k in ("features", "models", "final")):
        return {"error": "Run the project's stages up to the final one first: there is no model to improve yet."}
    if not _asked_to_improve(t):  # live with Claude, 2026-10-21: "yes, but show me the leakage first" started the run
        if (t.offered or {}).get("tool") == "improve_model" or WAIT.search(t.text):
            t.agent_offer("improve_model")  # still on offer: the next "yes" starts it (live, the recheck: a later yes found no offer)
        return {"error": "The person has not asked to start the improvement run now (a question, or a yes held back by \"first\", \"not now\", "
                         "\"wait\"): do what they asked, then offer it again."}
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


# ---------------------------------------------------------------------- the Problem analyst (step 1 of connecting understanding, 2026-10-21)
# The built project answers one problem. When the person changes it ("the outcome should be churn in 60 days", "we score at signup"),
# the analyst shows what would change and checks the new problem against the project's table (dclab_core.meet), then records the
# person's confirmation. Nothing here changes the built project: a confirmed problem becomes a new version through the Solution
# planner (step 2), as a new project with its own split record and its own holdout reading for the new problem.

CHANGE_FIELDS = {"outcome": "the outcome column", "prediction_moment": "the prediction moment", "horizon": "how far ahead it looks",
                 "unit": "what one prediction is for", "family": "the kind of problem", "time_column": "the column that orders the rows"}


def _model_serves() -> bool:
    """The built-chat journeys' serve rule (dclab_rnd.agent_eval.journeys.built.model_serves). ``DCLAB_BUILT_CHAT_SERVE``: "model" or
    "rules" overrides it (tests, a measurement's own runs); unset, the measurement decides. A rule that cannot be read: the rules serve."""
    import os

    forced = os.environ.get("DCLAB_BUILT_CHAT_SERVE", "").strip().lower()
    if forced in ("model", "rules"):
        return forced == "model"
    try:
        from ..extensions import optional

        built = optional("agent_eval.journeys.built")
        if built is None:  # the open edition: the person's switch or the doctor's probe (dclab_rnd.gates)
            from .. import gates

            return gates.passed("built_chat")
        return built.model_serves()
    except Exception:  # noqa: BLE001 - fail closed (the review of step 3)
        return False


def _draft(t: Turn) -> dict[str, Any]:
    return t.agent.store.get(t.draft_id)


def _data_card(t: Turn) -> Any:
    """The project's data card (its tables without their rows), else the draft's; None when neither was drawn."""
    from dclab_core.datacard import DataCard

    for doc in (((t.project().get("data") or {}).get("card")), _draft(t).get("data_card")):
        if isinstance(doc, dict) and doc.get("schema_version"):
            try:
                return DataCard.read({k: v for k, v in doc.items() if k != "fit"})
            except Exception:  # noqa: BLE001 - a card of an older shape: try the next
                continue
    return None


def _columns(t: Turn) -> list[str]:
    data = _data_card(t)
    return [c.name for tbl in (data.tables if data is not None else []) for c in tbl.columns]


def _problem_now(t: Turn) -> dict[str, Any]:
    """The problem the built project answers: from its saved solution, and the draft's card for what the solution does not hold."""
    from .card import card_of

    sol = t.project().get("solution") or {}
    card = card_of(_draft(t))
    u = _draft(t).get("understanding") or {}
    return {"outcome": sol.get("target"), "task": sol.get("task"), "family": card.family.value, "prediction_moment": sol.get("prediction_moment"),
            "horizon": card.horizon.value or u.get("horizon"), "unit": card.brief.unit.value, "time_column": sol.get("time_column"),
            "metric": sol.get("metric"), "forbidden": [f.get("column") for f in sol.get("forbidden") or [] if isinstance(f, dict)]}


def _problem_current(t: Turn) -> dict[str, Any]:
    """The problem as the conversation stands: as built, with the latest confirmed version's parts on top (a confirmed version
    keeps the whole problem it confirmed, so changes add up)."""
    versions = _draft(t).get("problem_versions") or []
    return dict(versions[-1].get("problem") or {}) if versions and versions[-1].get("problem") else _problem_now(t)


def show_problem(t: Turn, /, **_: Any) -> dict[str, Any]:
    """The problem as it stands: the outcome, the moment, the unit, the kind, the split's time column, the metric (as built, with any
    confirmed change on top, and what the built project still answers)."""
    now = _problem_current(t)
    built = _problem_now(t)
    facts = [[CHANGE_FIELDS.get(k, k.replace("_", " ")).capitalize(), v] for k, v in now.items() if k not in ("forbidden", "task") and v]
    if now.get("forbidden"):
        facts.append(["Never used (unknown at the moment)", ", ".join(str(c) for c in now["forbidden"][:12])])
    versions = _draft(t).get("problem_versions") or []
    if versions:
        differs = {k: built.get(k) for k in now if k != "forbidden" and now.get(k) != built.get(k)}
        facts.append([f"Confirmed version {len(versions) + 1}", "the built project still answers " + "; ".join(f"{CHANGE_FIELDS.get(k, k)}: {v or 'not set'}"
                                                                                                         for k, v in differs.items()) if differs else "the same problem"])
    return {"card": t.show("show_problem", {"agent": "analyst", "title": "The problem this project answers" if not versions else "The problem as it stands",
                                            "page": "solution", "facts": facts,
                                            "note": "To change it, say what is different (for example: the outcome is another column, or the prediction is made earlier)."})}


def _consequences(before: dict[str, Any], changes: dict[str, str]) -> list[str]:
    """What a change means for the built project, by rule."""
    out = []
    if "outcome" in changes or "family" in changes:
        out.append("A new outcome is a new problem: the split, the leakage audit, the features and the models are redone in a new project version, its "
                   "split. The current results answer the old outcome and stay as they are.")
    if "prediction_moment" in changes:
        out.append("What is known at the new moment differs: the leakage audit runs again and may forbid other columns (anything recorded "
                   "after the new moment).")
    if "horizon" in changes:
        out.append(f"The outcome's window changes (to {changes['horizon']}): the table must record the outcome over that window; a column "
                   "recorded for another window cannot be reused as it is.")
    if "unit" in changes:
        out.append("One prediction is for something else: the rows may need to be aggregated to the new unit first.")
    if "time_column" in changes:
        out.append(f"The rows are ordered by {changes['time_column']}: the held-back period changes, so every stage runs again.")
    return out


def propose_problem_change(t: Turn, /, outcome: str = "", prediction_moment: str = "", horizon: str = "", unit: str = "", family: str = "",
                           time_column: str = "", said: str = "", **_: Any) -> dict[str, Any]:
    """[the person changed the problem] What would change, field by field, checked against the project's table: the outcome column, the
    prediction moment, the horizon, the unit, the kind of problem or the time column. Nothing changes until the person confirms."""
    import copy
    import secrets

    from dclab_core.card import FAMILIES
    from dclab_core.meet import check, fit

    from .card import _stated, card_of

    before = _problem_current(t)
    asked = {"outcome": outcome, "prediction_moment": prediction_moment, "horizon": horizon, "unit": unit, "family": family, "time_column": time_column}
    changes = {k: str(v).strip()[:300] for k, v in asked.items() if str(v or "").strip() and str(v).strip() != str(before.get(k) or "")}
    if not changes:
        return {"error": "Nothing differs from the problem as built. Say which part changes: the outcome, the moment, the horizon or the unit."}
    if "family" in changes and changes["family"] not in FAMILIES:
        return {"error": f"{changes['family']!r} is not a kind of problem DCLab knows: one of {', '.join(FAMILIES)}."}
    data = _data_card(t)
    columns = [c.name for tbl in (data.tables if data is not None else []) for c in tbl.columns]
    for key in ("outcome", "time_column"):
        if key in changes and columns and changes[key] not in columns:
            near = [c for c in columns if changes[key].lower() in c.lower() or c.lower() in changes[key].lower()][:5]
            return {"error": f"{changes[key]!r} is not a column of this project's table" + (f"; did you mean {', '.join(near)}?" if near else ".")
                             + " Name the column that holds it.", "columns": columns[:60]}
    after_draft = copy.deepcopy(_draft(t))
    sol = after_draft.setdefault("solution", {}) or {}
    after_draft["solution"] = sol
    u = after_draft.setdefault("understanding", {})
    if "outcome" in changes:
        sol["target"], u["target"] = changes["outcome"], changes["outcome"]
        sol.pop("task", None)  # the new outcome's values say the task
    if "prediction_moment" in changes:
        sol["prediction_moment"] = u["prediction_moment"] = changes["prediction_moment"]
    if "time_column" in changes:
        sol["time_column"] = u["time_column"] = changes["time_column"]
    if "unit" in changes:
        u["unit"] = changes["unit"]
    if "family" in changes:
        u["family"], u["family_by"] = changes["family"], "person"
    after = card_of(after_draft)
    stated = {f: _stated(changes[k]) for k, f in (("outcome", "target_column"), ("time_column", "time_column")) if k in changes}
    if stated:  # what the person named is on the card checked, whether or not the draft's profile lists the column
        after = after.model_copy(update={"brief": after.brief.model_copy(update=stated)})
    mismatches, verdict = [], None
    if data is not None:
        checked = check(after, data)
        verdict = fit(after, data, checked)
        mismatches = [{"code": m.code, "blocking": m.blocking, "question": m.question} for m in checked.mismatches]
    rows = [{"field": CHANGE_FIELDS[k], "before": before.get(k) or "not set", "after": v} for k, v in changes.items()]
    earlier = _draft(t).get("problem_change") or {}
    replaces = {k: v for k, v in (earlier.get("changes") or {}).items() if k not in changes} if earlier.get("status") == "proposed" else {}
    problem = {**before, **{k: v for k, v in changes.items()}}
    change = {"id": "pc" + secrets.token_hex(4), "changes": changes, "before": {k: before.get(k) for k in changes}, "said": said[:600], "problem": problem,
              "replaces": replaces,
              "verdict": verdict.verdict if verdict is not None else "not_checked", "mismatches": mismatches,
              "consequences": _consequences(before, changes), "status": "proposed", "at": now()}
    t.agent.store.update(t.draft_id, lambda d: d.update(problem_change=change))
    blocking = [m["question"] for m in mismatches if m["blocking"]]
    if not blocking:  # a yes confirms it; a change that does not fit the table is answered first, never confirmed by a yes
        t.agent_offer("confirm_problem_change", {"change": change["id"]})
    if replaces:
        blocking_note = "; ".join(f"{CHANGE_FIELDS.get(k, k)} to {v}" for k, v in replaces.items())
    notes = ([f"This replaces the earlier proposal that was not confirmed ({blocking_note}): say it again to keep it."] if replaces else []) + \
            ([f"The new problem does not fit the table yet: {q}" for q in blocking] or
             ([f"Fit with the table: {verdict.verdict.replace('_', ' ')}"] if verdict is not None else ["The new problem could not be checked against the table."]))
    t.show("propose_problem_change", {
        "agent": "analyst", "title": "What would change", "page": "solution",
        "subtitle": "nothing changes in the built project until you confirm",
        "columns": [{"key": "field", "label": "Part of the problem"}, {"key": "before", "label": "Now"}, {"key": "after", "label": "New"}],
        "rows": rows, "lines": change["consequences"], "notes": notes + [m["question"] for m in mismatches if not m["blocking"]][:4],
        "offer": None if blocking else "Say yes to confirm this as the problem; the built project stays as it is."})
    result = {"proposed": change["id"], "changes": changes, "fit": change["verdict"], "blocking": blocking, "consequences": change["consequences"],
              "replaces_unconfirmed": replaces}
    t.results.append(result)
    return result


def confirm_problem_change(t: Turn, /, change: str = "", **_: Any) -> dict[str, Any]:
    """Record the person's confirmation of the proposed problem change: kept on the chat as a new version of the problem. The built
    project is not changed; a new version of it is built from the confirmed problem (the Solution planner's step)."""
    pending = _draft(t).get("problem_change") or {}
    if not pending or pending.get("status") != "proposed":  # one change is pending at a time: a yes confirms it (a model may name it in words)
        return {"error": "There is no proposed change to confirm. Say what is different about the problem first."}
    if (t.offered or {}).get("tool") != "confirm_problem_change":  # the person must have been asked, the turn before (the review, 2026-10-21)
        return {"error": "The person has not been asked to confirm this change: show it with propose_problem_change and wait for their yes."}
    if not (YES.match(t.text.strip()) or SAYS_YES.match(t.text)):  # asked is not agreed: Claude confirmed on "build a new version with 5 folds" (2026-10-21)
        return {"error": "The person did not say yes to this change: ask them to confirm it."}
    blocking = [m["question"] for m in pending.get("mismatches") or [] if m.get("blocking")]
    if blocking:
        return {"error": "The new problem does not fit the table yet, so it cannot be confirmed: " + blocking[0], "blocking": blocking}
    new_outcome = (pending.get("problem") or {}).get("outcome") or (pending.get("changes") or {}).get("outcome")
    kept_ops = [o for o in _base_ops(t) if new_outcome not in _reads(o)]  # a column read from the new outcome would carry its answer
    dropped_ops = [o for o in _base_ops(t) if new_outcome in _reads(o)]
    done = {**pending, "status": "confirmed", "confirmed_at": now(), "data_ops": kept_ops, "data_ops_dropped": dropped_ops}

    def keep(d: dict[str, Any]) -> None:
        current = d.get("problem_change") or {}
        if current.get("id") != pending["id"] or current.get("status") != "proposed":  # another proposal landed meanwhile: not this one's to confirm
            return
        d["problem_change"] = done
        d.setdefault("problem_versions", []).append(done)
    t.agent.store.update(t.draft_id, keep)
    t.task("confirm_problem_change", TEAM["analyst"], "The new problem, confirmed", "done")
    why = cannot_build(done)
    if why is None:  # a yes (or "build it") next asks the Solution planner for the new version's plan
        t.agent_offer("plan_new_version", {"version": done["id"]})
    result = {"confirmed": done["id"], "changes": done["changes"], "problem": done.get("problem"), "version": len(_draft(t).get("problem_versions") or []) + 1,
              "next": ("The built project stays as it is. Say yes (or build it) and the Solution planner plans the new version: a new project, its stages run again."
                       if why is None else f"The built project stays as it is. This version cannot be built from this table yet: {why}")
              + ("" if not dropped_ops else " Left out of this version, because they read the new outcome: "
                 + "; ".join(_describe_ops(dropped_ops)) + ".")}
    t.results.append(result)
    return result


# ---------------------------------------------------------------------- the Solution planner (step 2, 2026-10-21)
# A confirmed problem version becomes a new version of the project: the leakage audit and the solution proposal run again on the
# project's table for the new outcome and moment (the old outcome forbidden by default: it is usually recorded after the moment),
# the plan is shown, and on the person's yes a new chat for the version is built through the same path as "build it" on Home
# (quick_build: the proposal accepted, the project created, the agent running its stages; with the same table and time column its
# held-back rows are the same period as version 1's, read once for the new outcome). The built
# project stays as it was; the two are linked both ways.


def _latest_version(t: Turn) -> Optional[dict[str, Any]]:
    """The latest confirmed problem version this chat has not built yet."""
    versions = _draft(t).get("problem_versions") or []
    return versions[-1] if versions and not versions[-1].get("built") else None


def cannot_build(version: dict[str, Any]) -> Optional[str]:
    """Why a confirmed version cannot be built from this table by the planner, or None. A horizon or a unit changes what the outcome
    column means or what a row is: the table must hold that outcome, or be grouped to that unit, first."""
    ch = version.get("changes") or {}
    if "unit" in ch:
        return (f"one prediction is now for {ch['unit']}: the rows must first be grouped to that unit, which is not done from this chat; "
                "bring a table at that grain in a new chat")
    if "horizon" in ch and "outcome" not in ch:
        outcome = (version.get("problem") or {}).get("outcome")
        return (f"a horizon of {ch['horizon']} changes what the outcome means, and this table's outcome ({outcome}) records it for the old window: "
                "name the column that records it over the new window, or bring a table that has it")
    return None


def _forbidden_for(problem: dict[str, Any], old: dict[str, Any], proposal: dict[str, Any], versions: Any = (),
                   time_column: Optional[str] = None) -> list[dict[str, Any]]:
    """The new version's forbidden columns (the review, 2026-10-21: never fewer than before): every earlier outcome (version 1's and each
    confirmed version's; usually recorded after the moment; the person may allow one on the Solution page), every column version 1 forbade
    (the person or the leakage review ticked it; the audit run again may not flag it), and the audit's. Never the new outcome or the time
    column that orders the split."""
    keep_out = {problem.get("outcome"), time_column}
    out: list[dict[str, Any]] = []

    def add(column: Any, reason: str) -> None:
        if column and column not in keep_out and column not in {f["column"] for f in out}:
            out.append({"column": str(column), "reason": reason[:200]})
    for earlier in [old.get("target"), *[(v.get("problem") or {}).get("outcome") for v in versions or []]]:
        add(earlier, "an earlier outcome: usually recorded after the prediction moment (allow it on the Solution page if it is known in time)")
    for f in old.get("forbidden") or []:
        if isinstance(f, dict):
            add(f.get("column"), f"forbidden in version 1: {f.get('reason') or 'unknown at the prediction moment'}")
    for f in proposal.get("forbidden") or []:
        add(f.get("column"), str(f.get("reason") or "the leakage audit"))
    return out


def plan_new_version(t: Turn, /, **_: Any) -> dict[str, Any]:
    """[after a confirmed problem change] The plan for the project's new version: the leakage audit and the solution proposal run again on the
    project's table for the new outcome and moment, compared with the version built. Building it needs the person's yes."""
    from ..studio import data as studio_data
    from ..studio import solution as studio_solution

    version = _latest_version(t)
    if version is None:
        return {"error": "No confirmed problem change is waiting to be built. Say what is different about the problem first."}
    why = cannot_build(version)
    if why:
        return {"error": f"This version cannot be built from this table yet: {why}."}
    project = t.project()
    old = project.get("solution") or {}
    problem = dict(version.get("problem") or {})
    target = problem.get("outcome") or old.get("target")
    time_hint = problem.get("time_column") or old.get("time_column")  # the version's whole problem: a time column changed in an earlier version stays
    ops = list(version.get("data_ops") or [])
    try:
        frame = studio_data.load_table(studio_data.data_path(t.projects, project))
        if ops:  # the Data engineer's confirmed changes, on this version's copy of the table
            frame = _with_changes(t, frame, ops)
        profile = studio_data.profile_table(frame) if ops else (project.get("data") or {}).get("profile") or studio_data.profile_table(frame)
        from .card import card_of

        from ..kinds.propose import said_of

        proposal = studio_solution.propose(frame, profile, target, None, project.get("pack"), time_hint,
                                           card_of(_draft(t)).family.value,  # S9: the kind the card understood
                                           said_of(_draft(t).get("understanding") or {}), list(project.get("tables") or {}))
    except Exception as exc:  # noqa: BLE001 - the table cannot be read, or the outcome refused: said, nothing planned
        return {"error": f"The new version cannot be planned on this table: {str(getattr(exc, 'detail', None) or exc)[:300]}"}
    time_column = time_hint if time_hint in frame.columns else (proposal.get("time_candidates") or [None])[0]
    forbidden = _forbidden_for(problem, old, proposal, _draft(t).get("problem_versions") or [], time_column)
    number = len(_draft(t).get("problem_versions") or []) + 1
    planned = {"outcome": target, "task": proposal.get("task"), "kind": proposal.get("kind_title"), "metric": proposal.get("metric"),
               "prediction_moment": problem.get("prediction_moment") or old.get("prediction_moment"), "time_column": time_column,
               "forbidden": [f["column"] for f in forbidden], "identifiers": list(proposal.get("identifiers") or [])}
    before = {"outcome": old.get("target"), "task": old.get("task"), "metric": old.get("metric"), "prediction_moment": old.get("prediction_moment"),
              "time_column": old.get("time_column"), "forbidden": [f.get("column") for f in old.get("forbidden") or [] if isinstance(f, dict)]}
    label = {"outcome": "Outcome", "task": "Task", "metric": "Metric", "prediction_moment": "Prediction moment", "time_column": "Rows ordered by",
             "forbidden": "Never used (unknown at the moment)"}
    show = lambda v: ", ".join(map(str, v)) if isinstance(v, list) else v  # noqa: E731
    rows = [{"part": label[k], "v1": show(before.get(k)) or "none", "new": show(planned.get(k)) or "none", "changed": show(before.get(k)) != show(planned.get(k))}
            for k in label]
    t.agent.store.update(t.draft_id, lambda d: (d.get("problem_versions") or [{}])[-1].update(plan=planned, planned_at=now())
                         if (d.get("problem_versions") or [{}])[-1].get("id") == version["id"] else None)
    t.agent_offer("build_new_version", {"version": version["id"]})
    t.show("plan_new_version", {
        "agent": "planner", "title": f"Plan for version {number}", "page": "solution",
        "subtitle": "the leakage audit and the proposal run again on this project's table for the new problem",
        "columns": [{"key": "part", "label": "Part"}, {"key": "v1", "label": "Built (version 1)"}, {"key": "new", "label": f"Version {number}"}],
        "rows": rows, "mark": "changed",
        "lines": ["A new chat is opened for this version, on the same table; the agent runs its stages again. The held-back period is the same as version 1's "
                  "(same table, same time column), read once for the new outcome.",
                  *(["Data changes in this version: " + "; ".join(_describe_ops(ops))] if ops else []),
                  "The built project stays as it is; the two are linked."],
        "notes": [f"{f['column']}: {f['reason']}" for f in forbidden[:6]], "offer": "Say yes to build it."})
    result = {"planned": version["id"], "version": number, **planned, **({"data_changes": _describe_ops(ops)} if ops else {})}
    t.results.append(result)
    return result


def build_new_version(t: Turn, /, **_: Any) -> dict[str, Any]:
    """[after plan_new_version and the person's yes] Build the planned version: a new chat on the same table with the confirmed problem, built
    as "build it" builds (the proposal accepted, the agent running its stages). The built project is not changed."""
    import secrets
    import shutil

    from ..studio import data as studio_data

    if (t.offered or {}).get("tool") != "build_new_version":
        return {"error": "The person has not been asked to build this version: show its plan with plan_new_version and wait for their yes."}
    if not (YES.match(t.text.strip()) or VERSION_BUILD.search(t.text)):  # the review: an offer is not a yes; building spends and creates a project
        return {"error": "The person did not say yes to building this version: ask them."}
    version = _latest_version(t)
    if version is None or not version.get("plan"):
        return {"error": "No planned version is waiting to be built: plan it first."}
    builder = getattr(t.services, "build_from_chat", None)
    if builder is None:
        return {"error": "Building is not available here."}
    project, old_draft, plan = t.project(), _draft(t), version["plan"]
    number = len(old_draft.get("problem_versions") or []) + 1
    drafts = t.agent.store
    words = f"{old_draft.get('problem') or project.get('name')} (version {number}: the outcome is {plan['outcome']})"
    reuse = version.get("draft_tried")
    try:
        nd = drafts.get(reuse) if reuse else None
    except KeyError:
        nd = None
    nd = nd if nd is not None and not nd.get("project_id") else drafts.create(words[:2000])  # a retry after a refusal reuses its chat
    asset = next((a for a in old_draft.get("assets") or [] if a.get("id") == old_draft.get("active_asset")), {}) or {}
    table_file = drafts.data_dir(nd["id"]) / "clean.parquet"
    shutil.copyfile(studio_data.data_path(t.projects, project), table_file)
    ops, changed = list(version.get("data_ops") or []), None
    if ops:  # the confirmed data changes, on this version's copy only
        try:
            changed = _with_changes(t, studio_data.load_table(table_file), ops)
        except Exception as exc:  # noqa: BLE001 - a joined table changed or went away since the change was confirmed
            return {"error": f"Version {number}'s data changes cannot be made any more: {str(exc)[:200]}"}
        changed.to_parquet(table_file, index=False)
    forbidden = [{"column": c, "reason": "forbidden in the version's plan"} for c in plan.get("forbidden") or []]
    suggestion = {"target": plan["outcome"], "forbidden": forbidden, **({"prediction_moment": plan["prediction_moment"]} if plan.get("prediction_moment") else {}),
                  **({"time_column": plan["time_column"]} if plan.get("time_column") else {})}
    new_asset = {**{k: v for k, v in asset.items() if k not in ("suggestion", "raw_sha256", "spec_file", "spec_source")},
                 "id": "a" + secrets.token_hex(4), "kind": asset.get("kind") or "version", "status": "ready", "clean_file": "clean.parquet",
                 "name": asset.get("name") or str(project.get("name") or "table"), "suggestion": suggestion,
                 "version_of": {"project": t.project_id, "asset": asset.get("id")}}
    u = {k: v for k, v in (old_draft.get("understanding") or {}).items() if k not in ("judged_on",)}
    u.update(target=plan["outcome"], **({"prediction_moment": plan["prediction_moment"]} if plan.get("prediction_moment") else {}))
    problem = version.get("problem") or {}
    for key in ("unit", "horizon", "family"):
        if (version.get("changes") or {}).get(key):
            u[key] = version["changes"][key]
    intro_id = f"mv{version['id']}"
    intro = {"id": intro_id, "role": "agent", "kind": "text", "at": now(),
             "text": (f"This chat builds version {number} of {project.get('name')}: " + "; ".join([f"{CHANGE_FIELDS.get(k, k)} {v}" for k, v in (version.get('changes') or {}).items()]
                                                                                                   + _describe_ops(ops))
                      + f". Version 1 stays as it was. Asked from the chat of project {t.project_id}.")}

    data = project.get("data") or {}
    analysis = old_draft.get("analysis") if ((old_draft.get("analysis") or {}).get("columns")) else \
        {**(old_draft.get("analysis") or {}), "columns": [{"name": str(c)} for c in data.get("columns") or []], "profile": data.get("profile") or {}}
    data_card = old_draft.get("data_card")
    if changed is not None:  # the version's table has new columns: its analysis and data card are drawn again (names and bands, never a row)
        analysis = {**(old_draft.get("analysis") or {}), "columns": [{"name": str(c)} for c in changed.columns], "profile": studio_data.profile_table(changed)}
        data_card = studio_data.data_card({new_asset["name"]: changed}, new_asset["name"], plan["outcome"]) or data_card
    # the e2e check, 2026-10-21: a chat without its table's analysis left the build without the columns, so the outcome was "not named"

    def seed(d: dict[str, Any]) -> None:
        d["messages"] = [m for m in d.get("messages") or [] if m.get("id") != intro_id]  # a retry seeds the reused chat again, once
        d.update(assets=[new_asset], active_asset=new_asset["id"], analysis=analysis, data_card=data_card,
                 understanding=u, settings=old_draft.get("settings"), pack=old_draft.get("pack"), structure=old_draft.get("structure"),
                 version_of={"project": t.project_id, "draft": t.draft_id, "version": number, "change": version["id"], "problem": problem})
        d["messages"].append(intro)
    drafts.update(nd["id"], seed)
    done = builder(nd["id"])  # the same step as "build it" on Home: the proposal for the new outcome accepted, the project built, its agent started
    if done.get("refused"):
        drafts.update(t.draft_id, lambda d: [v.update(draft_tried=nd["id"]) for v in d.get("problem_versions") or [] if v.get("id") == version["id"]])
        return {"error": f"Version {number} could not be built: {done['refused']}", "draft": nd["id"]}
    new_pid = done.get("project_id")

    def link(d: dict[str, Any]) -> None:
        for v in d.get("problem_versions") or []:
            if v.get("id") == version["id"]:
                v.update(built=new_pid, draft=nd["id"], built_at=now())
    drafts.update(t.draft_id, link)
    try:  # the two versions linked both ways
        if not t.project().get("running"):  # a job saves version 1 as it runs: the link waits on the chat (the draft keeps it) rather than race it
            t.projects.update(t.project_id, versions=[*(t.project().get("versions") or []), {"project": new_pid, "version": number, "draft": nd["id"], "at": now()}])
        t.projects.update(new_pid, version_of={"project": t.project_id, "version": number, "change": version["id"]})  # a field, never the whole document
    except Exception:  # noqa: BLE001 - the link is written where it can be; the version stands either way
        pass
    t.agent.say(t.draft_id, f"Building version {number}: “{done.get('name') or 'the new project'}”, from the confirmed problem. "
                + ("Its agent is running the stages; the held-back period is read once for the new outcome." if done.get("started") else str(done.get("note") or "")),
                kind="build", project_id=new_pid, session_id=done.get("session_id"))
    t.task("build_new_version", TEAM["planner"], f"Version {number} built as a new project", "done")
    result = {"built": new_pid, "version": number, "draft": nd["id"], "started": bool(done.get("started")), "session": done.get("session_id")}
    t.results.append(result)
    return result


# ---------------------------------------------------------------------- project memory: the team remembers across chats


REMEMBER = re.compile(r"^\s*(?:please\s+)?(?:remember|keep in mind|note that|make a note)\b|\bfrom now on\b|\b(?:i|we)(?:'d| would)? (?:prefer|rather)\b|"
                      r"^\s*(?:please\s+)?(?:always|never)\b(?![\s-]*mind)", re.I)
EXPLICIT = re.compile(r"^\s*(?:please\s+)?(?:remember|keep in mind|make a note)\b", re.I)
PROBLEM_PARTS = re.compile(r"\b(?:outcome|target|label|horizon|moment|unit of prediction|predict(?:ed|s|ion)?)\b", re.I)
RECALL = re.compile(r"\bwhat (?:do|did) you (?:remember|know about (?:me|us|my|our))\b|\bwhat (?:did|have) (?:we|i) (?:decide|agree|chose|choose|settle)\w*|"
                    r"\b(?:our|my) (?:decisions|preferences)\b|\b(?:your|the team'?s?|(?:this )?project'?s?) memory\b|\bshow (?:me )?(?:the )?memory\b", re.I)
NOTE_LEAD = re.compile(r"^\s*(?:please\s+)?(?:remember|keep in mind|note|make a note)(?:\s+(?:that|this|of))?\s*[:,-]?\s*", re.I)


def asks_to_remember(said: str) -> bool:
    """The person asks the team to keep a decision or preference: never a question (unless it starts with "remember"), and
    never a change of what is predicted or when (that goes to the Problem analyst, checked against the table)."""
    if not REMEMBER.search(said) or PROBLEM_PARTS.search(said) or MOMENT.search(said):
        return False
    return bool(EXPLICIT.search(said)) or not ASKING.search(said)


def _new_note(kind: str, decision: str, reason: str, who: str, move: str) -> dict[str, Any]:
    """A note in studio.memory's shape (one line each, cut), not yet in any project."""
    from ..studio import memory as studio_memory

    return studio_memory.add({}, kind, decision, reason, who, move)


def _keep_notes(t: Turn, notes: list[dict[str, Any]]) -> None:
    """Into the project's memory. The store adds them to what is stored (storage/notes.py), so a job saving the whole
    project, another chat or a removal meanwhile never loses one."""
    if not notes:
        return
    t.projects.update(t.project_id, memory=notes)
    for n in notes:
        t.projects.log(t.project_id, "memory_noted", {"note": n["id"], "kind": n["kind"], "move": n["move"], "by": n["who"]})


def remember(t: Turn, /, **_: Any) -> dict[str, Any]:
    """Keep what the person just asked the team to remember (a decision or preference, for example "I prefer fewer false
    alarms" or "explain in plain words") in the project's memory: every chat on this project reads it from now on. Only when
    their message asks for it; their own words are kept, redacted, never yours and never a value from the table."""
    from dclab_core import redact

    if not asks_to_remember(t.text):
        return {"error": "The person did not ask to remember anything in this message: only their own request is kept."}
    words = redact.redact_text(NOTE_LEAD.sub("", t.text).strip())[0]
    if len(words.split()) < 2:
        return {"error": "Say what to remember, for example: remember that we prefer fewer false alarms."}
    _keep_notes(t, [_new_note("preference", words, "the person asked the team to remember it", "person", "remember")])
    t.task("remember", TEAM["reader"], "Kept in the project's memory", "done")
    result = {"remembered": words, "next": "Every chat on this project reads it from now on. Keeping it changes no setting by itself: "
                                            "ask for the change it implies (for example a cutoff that favours precision)."}
    t.results.append(result)
    return result


def show_memory(t: Turn, /, **_: Any) -> dict[str, Any]:
    """What the team remembers about this project: decisions and preferences from every chat and every person, latest last."""
    from ..studio import memory as studio_memory

    notes = studio_memory.active(t.project())
    if not notes:
        return {"error": "Nothing is remembered for this project yet. Say, for example, remember that we prefer fewer false alarms."}
    rows = [{"at": str(n.get("at") or "")[:16].replace("T", " "), "kind": n.get("kind"), "who": n.get("who"), "what": n.get("decision"), "why": n.get("reason")}
            for n in notes]
    return {"card": t.show("show_memory", {
        "agent": "reader", "title": "What the team remembers", "page": "project",
        "columns": [{"key": "at", "label": "When (UTC)"}, {"key": "kind", "label": "Kind"}, {"key": "who", "label": "Who"},
                    {"key": "what", "label": "What"}, {"key": "why", "label": "Why"}], "rows": rows})}


def _noted(tool: str, result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The note a successful chat write leaves in the project's memory (the person agreed to each of these)."""
    if tool == "confirm_problem_change":
        from dclab_core import redact

        changes = result.get("changes") if isinstance(result.get("changes"), dict) else {}
        said = redact.redact_text("; ".join(f"{k.replace('_', ' ')} {v}" for k, v in changes.items()))[0]  # a moment is the person's words
        return _new_note("decision", f"Version {result.get('version')} of the problem confirmed: {said or result.get('confirmed')}",
                         "the person confirmed it in the chat; this project still answers the problem it was built for", "person", "confirm_problem_change")
    if tool == "confirm_data_change":
        return _new_note("decision", f"Version {result.get('version')} confirmed with data changes: " + "; ".join(result.get("data_changes") or []),
                         "the person confirmed it in the chat; this project's table stays as it is", "person", "confirm_data_change")
    if tool == "adopt_improvement":
        return _new_note("decision", f"Adopted the improvement {result.get('adopted')} ({result.get('family')})", "it beat the current model on the training folds; the person adopted it",
                         "person", "adopt_improvement")
    if tool == "build_new_version":
        return _new_note("decision", f"Built version {result.get('version')} as a new project", "the person said yes to its plan", "person", "build_version")
    return None


# ---------------------------------------------------------------------- the Reporter: the report from the records, for a manager or the team


AUDIENCE = r"(?:my|the|our) (?:manager|boss|team|stakeholders?|director|client|lead|board)"
REPORT = re.compile(r"\b(?:write|make|draft|prepare|give|send|create|need|want|produce|download)\b[^.?!]{0,40}\breport\b|"
                    rf"\b(?:report|summary|write-?up)\b[^.?!]{{0,30}}\bfor {AUDIENCE}\b|\bsummari[sz]e\b[^.?!]{{0,40}}\bfor {AUDIENCE}\b|"
                    rf"\bwrite (?:it )?up\b[^.?!]{{0,40}}\bfor {AUDIENCE}\b|\bsummary of (?:the |this |my )?project\b|"
                    r"\b(?:full|final|solution|project'?s?) report\b", re.I)  # "the report" alone is the last improvement run's
NOT_REPORT = re.compile(r"\b(?:improvements?|experiments?|tried|last run|classification report|cells?)\b", re.I)


def asks_for_report(said: str) -> bool:
    """The person asks for the project's report: never a summary of one subject (the data, the models), a plot or a cell, or
    the improvement run's report; a subject named beside it is read by its own tool."""
    if not REPORT.search(said) or NOT_REPORT.search(said) or PLOT.search(said):
        return False
    named = [n for n, w in WORDS if n not in ("show_result", "show_evaluation", "show_improvements") and w.search(said)]  # "results", "the report": in every report
    return not named or bool(re.search(rf"\bfor {AUDIENCE}\b|\b(?:full|final|solution) report\b", said, re.I))
FOR_MANAGER = ("The problem as understood", "Results, with their intervals", "Limits, and what the data cannot show", "Next steps")
FOR_TEAM = FOR_MANAGER[:1] + ("What the data shows", "Decisions and reasons") + FOR_MANAGER[1:]


def _first_lines(body: str, n: int) -> list[str]:
    out = []
    for line in body.split("\n"):
        line = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line).replace("**", "").strip()
        if line and not line.startswith(("|", "#")):
            out.append(line if len(line) <= 400 else line[:399] + "…")
        if len(out) == n:
            break
    return out


def write_report(t: Turn, /, audience: str = "manager", **_: Any) -> dict[str, Any]:
    """The project's report from its records, every number traced to the record it came from: for a manager (the problem,
    the result with its interval, the limits, the next steps, a few lines each) or for the team (also the data and the
    decisions), with the full report to download. No model invents any of it: a model's summary, when one is on, is kept
    only where it cites the records."""
    from ..studio import memory as studio_memory, solution_report

    if not t.stage("final"):
        from ..studio import programme as studio_programme

        programme = studio_programme.stored(t.project())
        if programme is not None and programme.holdout is not None:  # a problem of several parts reports its programme: the download has it
            return {"card": t.show("write_report", {
                "agent": "reporter", "title": "The programme's report", "page": "evaluation",
                "lines": ["This project was solved as a programme of several parts: its report covers the composition and each part, every number from the records."],
                "download": {"route": "exportSolutionReport", "label": "Download the programme's report"}})}
        return _stage_missing("final")
    audience = "team" if str(audience).lower().startswith(("team", "tech", "data", "engineer")) else "manager"
    built = solution_report.latest(t.projects, t.project_id)
    report = built["report"]
    by_title = {sec.title: sec for sec in report.sections}
    lines, cites = [], []
    for title in FOR_TEAM if audience == "team" else FOR_MANAGER:
        sec = by_title.get(title)
        if sec is None:
            continue
        for line in _first_lines(sec.body, 4 if audience == "team" else 2):
            lines.append(f"{title}: {line}")
        cites += [c for c in (".".join(str(x).split(".")[:3]) for x in sec.cites) if c not in cites]  # deeper parts can name a table's values
    decided = [n["decision"] for n in studio_memory.active(t.project()) if n.get("who") == "person"][-3:]
    if decided:
        lines.append("What you decided: " + "; ".join(decided))
    return {"card": t.show("write_report", {
        "agent": "reporter", "title": "Report for your manager" if audience == "manager" else "Report for the team", "page": "evaluation",
        "lines": lines, "proof": cites[:8], "download": {"route": "exportSolutionReport", "label": "Download the full report"},
        "offer": "Every line comes from the project's records; the full report traces each number to the record it came from."}),
        "audience": audience, "sections": len(report.sections), "numbers_traced": len(report.numbers), "cached": built["cached"]}  # never the numbers' keys: they can name values


# ---------------------------------------------------------------------- the Data engineer: what is missing, a new column, a lookup join


DATA_GAPS = re.compile(r"\bwhat(?:'s| is| are)? (?:still )?missing (?:in|from) (?:the |my |this |our )?(?:data(?:set)?|table|columns?)\b|"
                       r"\bmissing (?:values?|data|cells?)\b|\bgaps? in (?:the|my|this|our) data\b|\bwhat data (?:is|are) missing\b|"
                       r"\b(?:incomplete|empty) (?:columns?|data|rows?|cells?)\b|\bnulls? (?:in|values?|counts?)\b", re.I)
ADD_COLUMN = re.compile(r"\b(?:add|create|make|derive|compute)\s+(?:a\s+|the\s+|one\s+)?(?:new\s+)?(?:column|feature|field)\s+"
                        r"[`'\"]?([A-Za-z_][A-Za-z0-9_]{0,59})[`'\"]?\s*(?:=|:|as|equal to|from)\s*(.+?)\s*[.!]?$", re.I)
ADD_WORDS = re.compile(r"\b(?:add|create|derive|compute)\s+(?:a\s+|the\s+)?new\s+(?:column|feature|field)\b|\b(?:add|create)\s+(?:a\s+|one\s+)?column\b", re.I)
NOT_DATA = re.compile(r"\b(?:importance|chart|plot|graph|histogram|heat ?map|figure|holdout|held[- ]back|team|us|me|meeting|call|note)\b", re.I)
FORMULA_LIKE = re.compile(r"[-+*/()<>=%]|\b(?:" + "|".join(("log1p", "abs", "sqrt", "round", "days_between", "year", "month", "weekday", "hour", "is_missing")) + r")\b")
JOIN = re.compile(r"\b(?:join|merge|bring in)\s+(?:the\s+|my\s+|this\s+)?(?:table\s+)?[`'\"]?([A-Za-z0-9_.-]{1,60}?)[`'\"]?"
                  r"(?:\s+table)?\s+(?:on|by|using)\s+(?:the\s+)?[`'\"]?([A-Za-z_][A-Za-z0-9_ ]{0,59}?)[`'\"]?(?:\s+column)?\s*[.!]?$", re.I)
JOIN_WORDS = re.compile(r"\b(?:join|merge)\b[^.?!]{0,40}\btables?\b", re.I)


def data_gaps(t: Turn, /, **_: Any) -> dict[str, Any]:
    """What is missing in the data: the columns with empty cells (their share, never a value), and where the problem asks for
    something the table does not hold (a date column for a horizon, the outcome itself), each with what would answer it."""
    card = _data_card(t)
    if card is None:
        return {"error": "This project's table has no data card yet: run the data stage first."}
    rows = []
    for table in card.tables:
        for c in table.columns:
            if c.null_rate > 0:
                rows.append({"column": c.name if len(card.tables) == 1 else f"{table.name}.{c.name}", "missing": round(100 * c.null_rate, 1),
                             "purpose": str(c.purpose or c.role).replace("_", " ")})
    rows.sort(key=lambda r: -r["missing"])
    lines = []
    try:
        from dclab_core import meet

        from .card import card_of

        checked = meet.check(card_of(_draft(t)), card)
        lines = [m.question for m in checked.mismatches][:6]
    except Exception:  # noqa: BLE001 - the comparison is advice: the columns stand without it
        pass
    if not rows and not lines:
        lines = ["No column has empty cells, and the table holds what the problem asks for."]
    total = sum(len(tb.columns) for tb in card.tables)
    return {"card": t.show("data_gaps", {
        "agent": "data", "title": "What is missing in the data", "page": "project",
        "subtitle": "from the table's data card: column names and shares, never a row",
        "facts": [["Columns", total], ["Columns with empty cells", len(rows)]],
        "columns": [{"key": "column", "label": "Column"}, {"key": "missing", "label": "Empty (%)", "num": True}, {"key": "purpose", "label": "Used as"}],
        "rows": rows[:40], "lines": lines,
        "offer": "To add a column: add column NAME = FORMULA (for example spend / tenure_months). To bring in a lookup table: join TABLE on KEY."})}


def _with_changes(t: Turn, frame: Any, ops: list[dict[str, Any]]) -> Any:
    """The table with a version's data changes, each joined table read again from where it was found."""
    from ..studio import derive

    sources = {o["table"]: o.get("source") or {} for o in ops if o.get("op") == "join"}
    return derive.apply(frame, ops, lambda name: _table_from(t, sources[name]))


def _describe_ops(ops: list[dict[str, Any]]) -> list[str]:
    from ..studio import derive

    return [derive.describe(o) for o in ops]


def _last_version(t: Turn) -> Optional[dict[str, Any]]:
    """The chat's last confirmed version, built or not: each new one starts from its problem and its data changes."""
    versions = _draft(t).get("problem_versions") or []
    return versions[-1] if versions else None


def _base_ops(t: Turn) -> list[dict[str, Any]]:
    """The data changes the last version holds (each new one adds to them), built or not."""
    return list((_last_version(t) or {}).get("data_ops") or [])


def _never_from(t: Turn) -> tuple[set[str], set[str]]:
    """The outcomes and forbidden columns a data change may never read: version 1's, every confirmed or proposed version's
    outcome, and every column a version's plan forbids (a column built from a later version's outcome carries its answer)."""
    sol = t.project().get("solution") or {}
    outcomes = {sol.get("target")}
    forbidden = {f.get("column") for f in sol.get("forbidden") or [] if isinstance(f, dict)}
    d = _draft(t)
    for v in [*(d.get("problem_versions") or []), *([d["problem_change"]] if (d.get("problem_change") or {}).get("status") == "proposed" else [])]:
        outcomes |= {(v.get("problem") or {}).get("outcome"), (v.get("changes") or {}).get("outcome")}
        forbidden |= set((v.get("plan") or {}).get("forbidden") or [])
        forbidden |= {f.get("column") if isinstance(f, dict) else f for f in (v.get("problem") or {}).get("forbidden") or []}
    outcomes.add(_problem_current(t).get("outcome"))
    return {o for o in outcomes if o}, {f for f in forbidden if f} - {o for o in outcomes if o}


def _reads(op: dict[str, Any]) -> set[str]:
    """Every column a data change reads or brings."""
    return set(op.get("reads") or []) | ({op["on"]} if op.get("on") else set()) | set(op.get("columns") or []) | set(op.get("source_columns") or []) \
        | ({op["name"]} if op.get("name") else set())  # a joined column renamed for a clash is still that column


def _version_frame(t: Turn) -> Any:
    from ..studio import data as studio_data, derive

    frame = studio_data.load_table(studio_data.data_path(t.projects, t.project()))
    return _with_changes(t, frame, _base_ops(t)) if _base_ops(t) else frame


def _join_table(t: Turn, name: str) -> tuple[Any, dict[str, Any]]:
    """Another table by name: one the project holds beside its table, else the table of another chat in this workspace."""
    import pandas as pd

    from ..studio import data as studio_data

    want = re.sub(r"\.(?:csv|parquet|xlsx?|json)$", "", str(name or "").strip().lower())
    project = t.project()
    for key in (project.get("tables") or {}):
        if key.lower() == want:
            return studio_data.load_tables(t.projects, project, [key])[key], {"kind": "project_table", "name": key}
    drafts = t.agent.store
    for d in drafts.list(limit=200):
        if d.get("id") == t.draft_id:
            continue
        for a in d.get("assets") or []:
            names = {str(a.get("name") or "").lower(), re.sub(r"\.\w+$", "", str(a.get("filename") or a.get("name") or "")).lower()}
            if want in names and a.get("status") == "ready" and a.get("clean_file"):
                path = drafts.data_dir(d["id"]) / a["clean_file"]
                if path.is_file():
                    return pd.read_parquet(path), {"kind": "chat", "draft": d["id"], "asset": a["id"], "file": a["clean_file"]}
    raise KeyError(name)


def _table_from(t: Turn, source: dict[str, Any]) -> Any:
    """A joined table again, from where the change found it."""
    import pandas as pd

    from ..studio import data as studio_data

    if source.get("kind") == "project_table":
        return studio_data.load_tables(t.projects, t.project(), [source["name"]])[source["name"]]
    return pd.read_parquet(t.agent.store.data_dir(source["draft"]) / source["file"])


def propose_data_change(t: Turn, /, change: str = "", name: str = "", formula: str = "", table: str = "", on: str = "", **_: Any) -> dict[str, Any]:
    """Propose a change to the project's data for a new version: add a column from a formula over the table's columns
    (change="add_column", name, formula such as "spend / tenure_months" or "days_between(signup, last_seen)"), or bring in a
    lookup table by a key (change="join", table, on). Checked first: never from the outcome, a forbidden column or a leak
    suspect; a joined table must have one row per key. Nothing changes until the person confirms; the built project never does."""
    import secrets

    from ..studio import derive

    outcomes, forbidden = _never_from(t)
    card = _data_card(t)
    suspects = [c.name for tb in (card.tables if card is not None else []) for c in tb.columns
                if c.purpose == "leak_suspect"]
    change = change or ("join" if table or on else "add_column" if name or formula else "")
    try:
        frame = _version_frame(t)
        if change == "add_column":
            if not (name and formula):
                return {"error": "Say the new column's name and its formula, for example: add column spend_per_month = spend / tenure_months. "
                                 "Formulas use column names, numbers, + - * / and " + ", ".join(derive.FUNCTIONS) + "."}
            op = derive.check_add(frame, name, formula, sorted(outcomes), sorted(forbidden), suspects)
        elif change == "join":
            if not (table and on):
                return {"error": "Say which table and which key, for example: join plans on plan_id. The table is one of this project's tables, "
                                 "or a table you added in another chat."}
            try:
                other, source = _join_table(t, table)
            except KeyError:
                return {"error": f"There is no table named {table!r} in this project or in another chat of this workspace: add it in a new chat first."}
            op = {**derive.check_join(frame, other, table, on, sorted(outcomes), sorted(forbidden), suspects), "source": source}
        else:
            return {"error": "Say what to change: add column NAME = FORMULA, or join TABLE on KEY."}
    except derive.ChangeRefused as exc:
        return {"error": f"The Data engineer cannot make that change: {exc}."}
    except Exception as exc:  # noqa: BLE001 - the table cannot be read: said, nothing proposed
        return {"error": f"The project's table cannot be read: {str(exc)[:200]}"}
    pending = {"id": "dc" + secrets.token_hex(4), "ops": [op], "status": "proposed", "at": now(), "said": t.text[:600]}
    t.agent.store.update(t.draft_id, lambda d: d.update(data_change=pending))
    t.agent_offer("confirm_data_change", {"change": pending["id"]})
    t.task("propose_data_change", TEAM["data"], "A change to the data, to confirm", "done")
    t.show("propose_data_change", {
        "agent": "data", "title": "What would change in the data", "page": "project",
        "lines": [derive.describe(op), *(["On top of the data changes already confirmed: " + "; ".join(derive.describe(o) for o in _base_ops(t))] if _base_ops(t) else []),
                  "It goes into a new version of the project: the leakage audit and every stage run again on the changed table, and the held-back "
                  "rows are read once, at the end. This project stays as it is."],
        "offer": "Say yes to confirm it."})
    result = {"proposed": pending["id"], "change": derive.describe(op)}
    t.results.append(result)
    return result


def confirm_data_change(t: Turn, /, **_: Any) -> dict[str, Any]:
    """[after propose_data_change and the person's yes] Keep the proposed data change as a new version of the project, to plan and
    build next (with any problem change confirmed before it). The built project is not changed."""
    from ..studio import derive

    pending = _draft(t).get("data_change") or {}
    if pending.get("status") != "proposed":
        return {"error": "There is no proposed data change to confirm."}
    if (t.offered or {}).get("tool") != "confirm_data_change":
        return {"error": "The person has not been asked to confirm this change: show it with propose_data_change and wait for their yes."}
    if not (YES.match(t.text.strip()) or SAYS_YES.match(t.text)):
        return {"error": "The person did not say yes to this change: ask them to confirm it."}
    base = _last_version(t) or {}
    version = {"id": pending["id"], "changes": dict(base.get("changes") or {}), "problem": base.get("problem") or _problem_now(t),
               "data_ops": _base_ops(t) + list(pending["ops"]), "said": pending.get("said"), "status": "confirmed", "confirmed_at": now(),
               "kind": "data"}

    def keep(d: dict[str, Any]) -> None:
        if (d.get("data_change") or {}).get("id") != pending["id"] or (d.get("data_change") or {}).get("status") != "proposed":
            return
        d["data_change"] = {**pending, "status": "confirmed"}
        versions = d.setdefault("problem_versions", [])
        if versions and base and not versions[-1].get("built") and versions[-1].get("id") == base.get("id"):
            versions[-1] = {**version, "replaces": base.get("id")}  # the unbuilt version takes the new change: one version, all its changes
        else:
            versions.append(version)
    t.agent.store.update(t.draft_id, keep)
    t.task("confirm_data_change", TEAM["data"], "The data change, confirmed", "done")
    t.agent_offer("plan_new_version", {"version": version["id"]})
    result = {"confirmed": version["id"], "data_changes": [derive.describe(o) for o in version["data_ops"]],
              "version": len(_draft(t).get("problem_versions") or []) + 1,
              "next": "The built project stays as it is. Say yes and the Solution planner plans the new version on the changed table."}
    t.results.append(result)
    return result


READS = {"show_problem": show_problem, "show_memory": show_memory, "write_report": write_report, "data_gaps": data_gaps, "list_projects": list_projects, "show_notebook": show_notebook, "show_activity": show_activity, "show_models": show_models, "show_features": show_features, "show_data": show_data, "show_leakage": show_leakage, "show_stages": show_stages,
         "show_result": show_result, "show_evaluation": show_evaluation, "improvement_ideas": improvement_ideas, "show_improvements": show_improvements}
WRITES = {"plan_new_version": ("plan_version", plan_new_version), "build_new_version": ("build_version", build_new_version),
          "propose_problem_change": ("propose_problem_change", propose_problem_change),
          "confirm_problem_change": ("confirm_problem_change", confirm_problem_change), "improve_model": ("improve", improve_model), "adopt_improvement": ("adopt_improvement", adopt_improvement),
          "write_notebook_cell": ("notebook_cell", write_notebook_cell), "remember": ("remember", remember),
          "propose_data_change": ("propose_data_change", propose_data_change), "confirm_data_change": ("confirm_data_change", confirm_data_change)}
WHO = {"data_gaps": "data", "propose_data_change": "data", "confirm_data_change": "data", "write_report": "reporter", "remember": "reader", "show_memory": "reader", "plan_new_version": "planner", "build_new_version": "planner", "show_problem": "analyst", "propose_problem_change": "analyst", "confirm_problem_change": "analyst", "list_projects": "reader", "show_notebook": "coder", "show_activity": "reader", "show_models": "reader", "show_features": "reader", "show_data": "reader", "show_leakage": "reader", "show_stages": "reader",
       "show_result": "evaluator", "show_evaluation": "evaluator", "improvement_ideas": "experimenter", "show_improvements": "experimenter",
       "improve_model": "experimenter", "adopt_improvement": "experimenter", "write_notebook_cell": "coder"}


HOME_MOVES = ("plan_version", "build_version", "propose_problem_change", "confirm_problem_change", "improve", "adopt_improvement",
              "notebook_cell", "remember", "propose_data_change", "confirm_data_change")  # every write the built chat's team may make (its guard checks each; the tests check each tool declares one)
ALONGSIDE_IMPROVE = ("notebook_cell",)  # an improvement run re-reads the project just before each save, so a cell written meanwhile stays
PROJECT_MOVES = ("improve", "adopt_improvement", "notebook_cell")  # the writes that change the project (the analyst's change the chat only)


def _remembered(t: Turn) -> list[str]:
    """The project's memory as the model reads it: one line a note, latest last (decisions and preferences from every chat)."""
    from ..studio import memory as studio_memory

    try:
        return studio_memory.summary(t.project())
    except Exception:  # noqa: BLE001 - the memory never stops a turn
        return []


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
        alongside = project.get("running") == "improve" and tool.move in ALONGSIDE_IMPROVE  # several tasks at once (the owner, 2026-10-21)
        if project.get("running") and tool.move in PROJECT_MOVES and not alongside:  # a job saves this project as it runs: a write now could lose its update or leave it marked running
            return {"error": f"Something is running in this project ({project['running']}): wait for it to finish, or stop it."}
        if tool.name == "adopt_improvement" and _improving(turn) is not None:
            return {"error": "An improvement run is still going: adopt from its report when it ends."}
        result = proceed()
        ok = isinstance(result, dict) and "error" not in result
        if ok:
            note = _noted(tool.name, result)
            if note is not None:
                try:
                    _keep_notes(turn, [note])
                except Exception:  # noqa: BLE001 - a note never undoes the write it describes
                    pass
        try:
            audit.record_for(turn.projects, "move", "agent", "home", move=tool.move, status="allowed" if ok else "refused", project_id=turn.project_id,
                             draft_id=turn.draft_id, message=(result or {}).get("error") if not ok else f"home orchestrator: {tool.name}")
        except Exception:  # noqa: BLE001 - the audit is written where it can be
            pass
        return result


def register(registry: Registry) -> None:
    empty = {"type": "object", "properties": {}}
    for name, fn in READS.items():
        params = {"type": "object", "properties": {"rows": {"type": "string", "enum": ["holdout", "validation"]}}} if name == "show_evaluation" else \
            {"type": "object", "properties": {"audience": {"type": "string", "enum": ["manager", "team"]}}} if name == "write_report" else empty
        registry.register(Tool(name, f"[{TEAM[WHO[name]]}] " + " ".join((fn.__doc__ or "").split()), params, fn, scope="home", takes_context=True))
    for name, (move, fn) in WRITES.items():
        S = {"type": "string"}
        params = {"type": "object", "properties": {"candidate": S}} if name == "adopt_improvement" else \
            {"type": "object", "properties": {"request": S}, "required": ["request"]} if name == "write_notebook_cell" else \
            {"type": "object", "properties": {"change": S}} if name in ("confirm_problem_change", "confirm_data_change") else \
            {"type": "object", "properties": {"change": {"type": "string", "enum": ["add_column", "join"]},
                                              "name": {**S, "description": "add_column: the new column's name"},
                                              "formula": {**S, "description": "add_column: over the table's columns, e.g. spend / tenure_months"},
                                              "table": {**S, "description": "join: the other table's name"}, "on": {**S, "description": "join: the key column"}},
             "required": ["change"]} if name == "propose_data_change" else \
            {"type": "object", "properties": {"version": S}} if name in ("plan_new_version", "build_new_version") else \
            {"type": "object", "properties": {"outcome": {**S, "description": "the new outcome column, exactly as named in the table"},
                                              "prediction_moment": {**S, "description": "when each prediction is made, and what is known then"},
                                              "horizon": {**S, "description": "how far ahead, e.g. 60 days"}, "unit": {**S, "description": "what one prediction is for"},
                                              "family": {**S, "description": "the kind of problem, when it changes"}, "time_column": S,
                                              "said": {**S, "description": "the person's words that changed it"}}} if name == "propose_problem_change" else empty
        registry.register(Tool(name, f"[{TEAM[WHO[name]]}] " + " ".join((fn.__doc__ or "").split()), params, fn, effect="write", scope="home", move=move,
                               takes_context=True))
    registry.guards["home"] = HomeGuard()


# ---------------------------------------------------------------------- the rules' plan (no model, or the model failed)


YES = re.compile(r"^\s*(?:ok(?:ay)?|yes|yeah|yep|yup|sure|fine|alright|go(?: ahead| on)?|do it|please(?: do)?|let'?s(?: do it| go)?|start|run it|proceed|"
                 r"try (?:it|them)|of course|perfect|great|confirm(?: it)?|build it)(?:[\s,.!]+(?:ok(?:ay)?|yes|do it|go ahead|please|now|then|thanks|thank you|try them|run it|confirm(?: it)?|build it))*[\s.!]*$", re.I)
NO = re.compile(r"^\s*(?:no|nope|not now|later|cancel|stop|don'?t)\b", re.I)
SAYS_YES = re.compile(r"^\s*(?:yes|yeah|yep|ok(?:ay)?|sure|confirm(?:ed)?|correct|right|agreed)\b", re.I)  # "yes, use weekly sales"
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
CHIPS = {"data_gaps": ["Show the data", "Write the report for my manager"], "propose_data_change": ["Yes, confirm it", "No, keep the data as it is"],
         "confirm_data_change": ["Yes, plan the new version", "Show the problem"],
         "write_report": ["Write the report for the team", "What do you remember?", "Show the held-back result"],
         "remember": ["What do you remember?", "Write the report for my manager"], "show_memory": ["Write the report for my manager", "Show the models trained"],
         "show_models": ["Show the evaluation", "Improve the model for me", "Plot each fold's score"],
         "show_evaluation": ["Show the held-back result", "How can it improve?", "Plot the confusion matrix"],
         "show_result": ["Show the evaluation", "Show the models trained", "How can it improve?"],
         "improvement_ideas": ["Yes, try them", "Show the models trained"],
         "propose_problem_change": ["Yes, confirm it", "No, keep the problem as built"], "confirm_problem_change": ["Build the new version", "Show the problem"],
         "plan_new_version": ["Yes, build it", "Show the problem"], "build_new_version": ["Show the problem", "Show the models trained"],
         "show_problem": ["Show the models trained", "Show the evaluation"],
         "show_improvements": ["Adopt the best way", "Show the models trained"],
         "improve_model": ["Show the models trained", "Show the evaluation"],
         "adopt_improvement": ["Show the held-back result"], "write_notebook_cell": ["Show the evaluation", "Show the models trained"],
         None: ["Show the models trained", "Show the evaluation", "How can it improve?", "Improve the model for me"]}


CHANGE_WORDS = re.compile(r"^\s*no\b|\b(?:actually|instead|rather|should (?:be|predict|look)|not (?:the )?right|wrong|change|changed|different|redefine|"
                          r"we (?:score|predict|decide) (?:at|when|before|on)|predict \w+ (?:at|when|before))\b", re.I)
PROBLEM_NOUNS = re.compile(r"\b(?:outcome|target|label|moment|horizon|problem|what (?:we|it) predicts?|unit of prediction)\b", re.I)
HORIZON = re.compile(r"\b(?:within|in the next|over the next|next)\s+(\d+)\s*(days?|weeks?|months?|years?)\b|\b(\d+)[- ](day|week|month|year)s?\b", re.I)
MOMENT = re.compile(r"\b(?:predict(?:ed|ion|ions)?|score[sd]?|decide[sd]?|decision)\s+(?:(?:it|them|each\s+\w+|is|are|gets?|will\s+be|should\s+be|must\s+be|"
                    r"made|done|scored|happens?)\s+)*(at|when|before|right after|on)\s+([^.;,?!]{4,120})", re.I)
ASK_PROBLEM = re.compile(r"\b(?:what (?:problem|are we predicting|is (?:the|our) (?:outcome|target|problem))|which (?:outcome|target)|the problem (?:we|this)|what does (?:it|this project) predict)\b", re.I)


OUTCOME_BEFORE = r"(?:outcome|target|label)\s+(?:column\s+)?(?:should\s+be|must\s+be|is|=|to\s+be|becomes?|would\s+be)\s+(?:the\s+)?(?:column\s+)?"
HORIZON_WITH = re.compile(r"\b(?:predict\w*|outcome|target|label|ahead|horizon|churn\w*|convert\w*|buy|leave|cancel\w*|happen\w*)\b", re.I)


def problem_change(said: str, columns: Any = ()) -> Optional[dict[str, str]]:
    """The rules' reading of a changed problem: {} when the words change the problem but name nothing they can read (ask which part,
    after every other reading had its turn), the parts they name (an outcome column of the table, a horizon, a moment), or None when
    the words do not change the problem. Strict on purpose (the review, 2026-10-21): an outcome only right after an outcome phrase
    ("the outcome should be X", "predict X"), a horizon only beside a word of the outcome ("predict churn within 60 days"), never
    "the 30-day features"."""
    if not CHANGE_WORDS.search(said):
        return None
    old = lambda c: re.search(rf"\b(?:not|instead of|rather than|than|from)\s+(?:the\s+)?{re.escape(str(c))}(?![\w])", said, re.I)  # noqa: E731 - "not X": the old one
    outcome = next((str(c) for c in sorted(columns or (), key=lambda c: -len(str(c))) if len(str(c)) >= 3 and not old(c)
                    and re.search(rf"(?:{OUTCOME_BEFORE}|\bpredict(?:ing)?\s+(?:the\s+)?){re.escape(str(c))}(?![\w])", said, re.I)), None)
    horizon = next((h for h in HORIZON.finditer(said) if not re.match(r"\s*(?:features?|columns?|averages?|counts?|windows?\s+of)", said[h.end():], re.I)), None)
    horizon = horizon if horizon is not None and HORIZON_WITH.search(said) else None
    moment = MOMENT.search(said)
    if not (outcome or horizon or moment or PROBLEM_NOUNS.search(said)):
        return None
    out: dict[str, str] = {}
    if outcome:
        out["outcome"] = outcome
    if horizon:
        n, unit = (horizon.group(1), horizon.group(2)) if horizon.group(1) else (horizon.group(3), horizon.group(4) + "s")
        out["horizon"] = f"{n} {unit.lower()}"
    if moment and not (horizon and horizon.group(0) in moment.group(0)):
        out["prediction_moment"] = f"{moment.group(1)} {moment.group(2).strip()}"
    return out


VERSION_BUILD = re.compile(r"\bbuild\s+(?:it|the\s+new\s+version|(?:the\s+)?version\s*\d*|(?:a|the)\s+new\s+one|the\s+new\s+project)\b|^\s*(?:ok(?:ay)?,?\s+|yes,?\s+|let'?s\s+)?build\b", re.I)


SPLIT = re.compile(r"\s*(?:,\s*(?:and\s+|then\s+)?|;\s*|\s+and then\s+|\s+then\s+|\s+and also\s+|\s+also\s+|\s+and\s+(?=(?:please\s+)?(?:show|plot|draw|chart|improve|"
                   r"write|make|tell|give|list|what|how|run|try|build|explain|compare)\b))\s*", re.I)
SPLITTABLE = ("write_notebook_cell", "improve_model", "improvement_ideas", "write_report", "data_gaps", "show_memory")  # each returns alone; a message may ask for more
MAX_TASKS = 3
YES_AND = ("improve_model", "confirm_problem_change", "confirm_data_change")
YES_THEN = re.compile(r"^\s*(?:yes|yeah|yep|ok(?:ay)?|sure|confirm(?:ed)?|correct|right|agreed)\b[\s,.!-]*(?:and|then|also)\b\s*", re.I)
HOLD = re.compile(r"\b(?:but|instead|first|not now|later|wait|don'?t|do not|no|not yet|hold)\b", re.I)  # the yes is not a yes  # offers a "yes, and ..." carries out (building needs a plain yes)
OFFERING = ("improvement_ideas", "propose_data_change")  # each asks for a yes next


def plan(text: str, offer: Optional[dict[str, Any]], columns: Any = (), version_waiting: bool = False) -> list[tuple[str, dict[str, Any]]]:
    """The rules' reading of one message, part by part when it asks for several things ("improve the model and plot each fold"):
    the tools to call, in order, at most MAX_TASKS. A write that needs the person's yes is never read from a part."""
    joined = YES_THEN.match(text.strip())
    if offer and offer.get("tool") in YES_AND and joined:  # "yes, and plot it": the offer, then the rest; never "yes, but ..."
        rest = text.strip()[joined.end():]
        tail = [(n, a) for n, a in plan(rest, None, columns, version_waiting)
                if n != offer["tool"] and (n in SPLITTABLE or n in READS or n in ("propose_data_change", "remember"))] \
            if len(rest.split()) >= 2 and not HOLD.search(rest) else []
        if tail:
            return [(offer["tool"], dict(offer.get("args") or {}))] + tail[:MAX_TASKS - 1]
    whole = _plan_one(text, offer, columns, version_waiting)
    if offer and YES.match(text.strip()) or not (len(whole) == 1 and whole[0][0] in SPLITTABLE):
        return whole
    parts = [x for x in SPLIT.split(text.strip()) if len(x.split()) >= 2]
    if len(parts) < 2:
        return whole
    out: list[tuple[str, dict[str, Any]]] = []
    for part in parts:
        for name, args in _plan_one(part, None, columns, version_waiting):
            offering = [n for n, _ in out if n in OFFERING]
            if name in ("improve_model", "improvement_ideas") and not re.search(r"\bmodel\b", part, re.I):
                continue  # "make it look better" beside a plot is about the plot, not a model run
            if (name in SPLITTABLE or name in READS or name == "propose_data_change") and name not in [n for n, _ in out] \
                    and not (name in OFFERING and offering):  # one offer a turn: a second would overwrite the first's question
                out.append((name, args))
    return out[:MAX_TASKS] if len({n for n, _ in out}) > 1 else whole


def _plan_one(text: str, offer: Optional[dict[str, Any]], columns: Any = (), version_waiting: bool = False) -> list[tuple[str, dict[str, Any]]]:
    """The rules' reading of one ask: the tools to call, in order. A yes carries out the pending offer; a changed problem goes to
    the Problem analyst; words name the rest."""
    said = text.strip()
    if offer and YES.match(said):
        return [(offer["tool"], dict(offer.get("args") or {}))]
    if offer and offer.get("tool") == "build_new_version" and VERSION_BUILD.search(said) and not NO.match(said):  # "build the new version" after its plan: build
        return [(offer["tool"], dict(offer.get("args") or {}))]
    if version_waiting and VERSION_BUILD.search(said) and not IMPROVE.search(said):  # "build it" with a confirmed version waiting: the planner plans it
        return [("plan_new_version", {})]
    changed = problem_change(said, columns)  # before a no: "no, the prediction is made at signup" corrects the problem
    if changed:
        return [("propose_problem_change", {**changed, "said": said[:600]})]
    if NO.match(said):  # "no" declines; after a bare "no", the rest is read ("no, show the models"), never a refusal ("no, don't improve it")
        rest = re.sub(r"^\s*(?:no|nope)\b[\s,.!-]*(?:thanks?|thank you)?[\s,.!-]*", "", said, flags=re.I)
        return _plan_one(rest, None, columns) if rest and rest != said and not NO.match(rest) else []
    if RECALL.search(said):
        return [("show_memory", {})]
    if asks_to_remember(said):
        return [("remember", {})]
    data_words = not NOT_DATA.search(said) and not IMPROVE.search(said)
    m = ADD_COLUMN.search(said)
    if m and data_words and FORMULA_LIKE.search(m.group(2)):
        return [("propose_data_change", {"change": "add_column", "name": m.group(1), "formula": m.group(2)})]
    if ADD_WORDS.search(said) and data_words:
        return [("propose_data_change", {"change": "add_column"})]
    m = JOIN.search(said)
    if m and data_words:
        return [("propose_data_change", {"change": "join", "table": m.group(1), "on": m.group(2).strip()})]
    if JOIN_WORDS.search(said) and data_words:
        return [("propose_data_change", {"change": "join"})]
    if DATA_GAPS.search(said) and not dict(WORDS)["show_leakage"].search(said):
        return [("data_gaps", {})]
    if asks_for_report(said):
        return [("write_report", {"audience": "team" if re.search(r"\b(?:team|technical|data scientists?|engineers?)\b", said, re.I) else "manager"})]
    if ASK_PROBLEM.search(said):
        return [("show_problem", {})]
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
    if not out and changed is not None:  # the problem changes but its words name nothing readable, and nothing else read them: ask which part
        return [("show_problem", {})]
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
                "problem": _problem_now(turn), "table_columns": _columns(turn)[:80],
                "proposed_problem_change": {k: (turn.agent.store.get(turn.draft_id).get("problem_change") or {}).get(k) for k in ("id", "changes", "status")},
                "confirmed_version_waiting": ({k: (_latest_version(turn) or {}).get(k) for k in ("id", "changes")} | {"planned": bool((_latest_version(turn) or {}).get("plan"))})
                if _latest_version(turn) else None,
                "working_now": self.working(turn.draft_id), "pending_offer": (turn.offered or {}).get("tool"), "team": TEAM,
                "remembered": _remembered(turn),
                **({"reply_length": f"the person asked for at most {asked_length(turn.text)} sentence(s): answer in that many"} if asked_length(turn.text) else {})}

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
        if client is not None and not _model_serves():  # step 3: the model serves only while the built-chat journeys say it should
            record["served_by"] = "rules: the model did not pass the built chat's sealed journeys"
            client = None
        reply, path = None, "rules"
        shadow = None
        if client is not None:  # step 3: the rules' reading of this turn, recorded beside the model's, never used (agreement, training labels)
            try:
                data = _data_card(t)
                shadow = [n for n, _ in plan(text, t.offered, [c.name for tbl in (data.tables if data is not None else []) for c in tbl.columns],
                                             _latest_version(t) is not None)]
            except Exception:  # noqa: BLE001 - a shadow never stops a turn
                shadow = None
        t.rules_plan = shadow or []  # what the rules would run for this message: a model that runs none of it did not answer
        if client is not None:
            try:
                reply = self.model_turn(t, client, record)
                path = "model"
            except Exception as exc:  # noqa: BLE001 - the model failed or its cap was reached: the rules carry this turn
                record["model_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                if t.tasks or t.shown:  # tools ran this turn (a refused cell was still written): say what they did; planning again would run a write twice
                    reply, path = self._rules_words(t), "model"
        if reply is None:
            reply = self.rules_turn(t, record)
        if (t.offered or {}).get("tool") == "confirm_problem_change":  # the offer lives one turn: an unconfirmed proposal it named expires
            offered_id = (t.offered.get("args") or {}).get("change")
            self.agent.store.update(draft_id, lambda d: (d.get("problem_change") or {}).update(status="expired", expired_at=now())
                                    if (d.get("problem_change") or {}).get("status") == "proposed" and (d.get("problem_change") or {}).get("id") == offered_id else None)
        record.update(path=path, reply=reply, tasks=t.tasks, cards=[c.get("title") for c in t.shown.values()])
        if path == "model" and shadow is not None:
            record["rules_shadow"] = shadow
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
        record["steps"] = []

        def kept(s: Any) -> None:  # each step kept as it runs: a model that fails after a tool still shows what it did (the review of step 3)
            record["steps"].append({"n": s.n, "tool": s.tool, "arguments": s.arguments, "ok": s.ok, "result": s.result, "seconds": round(s.seconds, 3)})
        out = run_policy(policy, client, registry, messages, context=t, contexts={"home": t}, trace=tracer(AGENT), on_step=kept)
        record["usage"], record["stopped"] = out.usage, out.stopped
        record["spent_eur"] = round(float(getattr(client, "spent_eur", 0.0) or 0.0), 6)
        said = (out.final or "").strip()
        for s in out.steps:
            if s.ok and isinstance(s.result, dict) and "error" not in s.result:
                t.results.append(s.result)
            elif isinstance(s.result, dict) and s.result.get("error"):
                t.results.append(s.result)  # an error's words may be repeated
                t.task(s.tool, TEAM[WHO.get(s.tool, "reader")], s.tool.replace("_", " "), "failed", detail=str(s.result["error"])[:200])
        worked = [s for s in out.steps if s.ok and not (isinstance(s.result, dict) and s.result.get("error"))]
        rules_would_act = bool(getattr(t, "rules_plan", None))  # the message asks for what a tool does, and the model ran none
        if not worked and not t.shown and (rules_would_act or not grounded(said, [t.results, self.situation(t)])[0] or _not_an_answer(said, out.stopped)):
            # every call refused and nothing grounded said (a small local model): nothing ran, so the rules carry the turn
            # and the refusals stay in the record, never in the reply
            t.tasks[:] = [x for x in t.tasks if x["status"] != "failed"]
            t.results[:] = [r for r in t.results if not (isinstance(r, dict) and r.get("error"))]
            raise ModelUnhelpful(f"{len(out.steps)} tool call(s), none accepted, and no grounded words")
        kept, dropped = grounded(said, [t.results, self.situation(t)])
        if t.shown:  # the owner, 2026-10-20: "bring the table instead of the texts": a list that repeats a shown table is dropped
            kept, listed = without_lists(kept)
            dropped += listed
            kept = shorten(kept, SHOWN_SENTENCES)  # the table speaks; the words say what it means and the next step (the whole reply stays in the record)
        most = asked_length(t.text)
        if most:  # "in two sentences": the person's length, kept (live, 2026-10-21: four came back)
            sentences = [x for x in re.split(r"(?<=[.!?])\s+", kept.strip()) if x]
            kept = " ".join(sentences[:most])
        record["said"], record["dropped"] = said, dropped
        if not kept:
            return self._rules_words(t) if t.shown or t.tasks else "I could not answer that from this project's records. Here is what I can do:"
        return kept

    def rules_turn(self, t: Turn, record: dict[str, Any]) -> str:
        from ..agents import default_registry, harness

        registry = default_registry()
        data = _data_card(t)
        steps = plan(t.text, t.offered, [c.name for tbl in (data.tables if data is not None else []) for c in tbl.columns],
                     _latest_version(t) is not None)  # only explicit build words reach the planner (VERSION_BUILD), which says why a version cannot be built
        t.change_unclear = problem_change(t.text, ()) == {} and steps == [("show_problem", {})]
        record["plan"] = [{"tool": n, "arguments": a} for n, a in steps]
        if record.get("steps"):  # the model ran tools before it failed: kept apart, the rules' steps are their own
            record["model_steps"] = record["steps"]
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
                if t.offered.get("tool") == "confirm_problem_change":  # the proposal is declined, kept as such (the person's answer is a label)
                    t.agent.store.update(t.draft_id, lambda d: (d.get("problem_change") or {}).update(status="declined", declined_at=now()))
                return "All right, nothing changes. Ask for anything else about this project."
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
            elif task["tool"] == "propose_data_change":
                r = next((x for x in t.results if isinstance(x, dict) and x.get("proposed", "").startswith("dc")), {})
                parts.append(f"The Data engineer checked the change: {r.get('change')}. Say yes to confirm it; this project stays as it is.")
            elif task["tool"] == "confirm_data_change":
                r = next((x for x in t.results if isinstance(x, dict) and x.get("data_changes")), {})
                parts.append(f"Confirmed as version {r.get('version')}: {'; '.join(r.get('data_changes') or [])}. {r.get('next')}")
            elif task["tool"] == "remember":
                r = next((x for x in t.results if isinstance(x, dict) and x.get("remembered")), {})
                parts.append(f"Kept in the project's memory: \"{r.get('remembered')}\". {r.get('next')}")
            elif task["tool"] == "write_notebook_cell":
                parts.append("The Notebook coder wrote this cell: read it, then press Run to run it on the training rows only.")
            elif task["tool"] == "propose_problem_change":
                r = next((x for x in t.results if isinstance(x, dict) and x.get("proposed")), {})
                parts.append("The Problem analyst compared the new problem with the one this project was built for: the table above shows what would change. "
                             + (f"It does not fit the table yet: {r['blocking'][0]}" if r.get("blocking") else
                                "Say yes to confirm it; the built project stays as it is."))
            elif task["tool"] == "confirm_problem_change":
                r = next((x for x in t.results if isinstance(x, dict) and x.get("confirmed")), {})
                parts.append(f"Confirmed as version {r.get('version')} of the problem. {r.get('next')}")
            elif task["tool"] == "plan_new_version":
                r = next((x for x in t.results if isinstance(x, dict) and x.get("planned")), {})
                parts.append(f"The Solution planner ran the leakage audit again for the new problem: the table above is the plan for version {r.get('version')}. "
                             "Say yes to build it; the built project stays as it is.")
            elif task["tool"] == "build_new_version":
                r = next((x for x in t.results if isinstance(x, dict) and x.get("built")), {})
                parts.append(f"Version {r.get('version')} is built as a new project with its own chat; open it from the card above.")
            elif task["tool"] == "show_problem" and t.change_unclear:
                parts.append("The Problem analyst: above is the problem this project answers. Which part changes: the outcome (name its column), "
                             "the moment the prediction is made, or how far ahead it looks?")
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
