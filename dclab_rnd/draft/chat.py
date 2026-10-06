"""The Home agent: understands the problem in a few questions and keeps the solution workflow current.

It is not a general chatbot. Its job on Home is narrow: find out what should be predicted, at which moment,
what each prediction changes and what an error costs; describe the data the user brought; and keep a
solution workflow for this problem. It asks at most ``MAX_QUESTIONS`` questions, one at a time.

Two modes, like the intern:
- with a model (``client`` given): the model reads a compact context (the problem, the answers so far, the
  data summary; never raw rows) and acts through a few tools. Code validates what it proposes.
- without a model, or when the model fails: a deterministic question script does the same job.

Every message and change is written to the draft and emitted as an event for the live Home page.
"""

from __future__ import annotations

import functools
import json
import re
import secrets
from dataclasses import dataclass
from typing import Any, Callable

from dclab_rnd.agents.registry import Registry, Tool
from dclab_rnd.agents.runtime import Policy, Step, run as run_policy
from dclab_rnd.agents.traces import Tracer

from . import pack as packs
from . import plan as plans
from . import workflow as wflow
from .store import DraftStore, now

MAX_QUESTIONS = 4
MODEL_REPLIES = 5  # model requests per user message
MODEL_STEPS = 40  # tool calls per user message (a reply may make several)
MODEL_SECONDS = 5 * 60
TOKEN_CHUNK = 48  # characters per streamed ``token`` event
PROFILE_PAGE = 40  # columns the model sees at a time, in its state and from get_profile
PROFILE_FIELDS = ("name", "kind", "missing_rate", "unique")  # what the model may know about a column: no values
FIELDS = ("target", "prediction_moment", "action", "costs", "data_plan")
BUILD_WORDS = re.compile(r"\b(let'?s build|build (it|the solution)|go ahead|start the project|ready to build)\b", re.I)
SIMULATE_WORDS = re.compile(r"\b(simulat\w*|synthetic|fake data|generate (some )?data)\b", re.I)

POLICY = """You are the DCLab Home agent. A user described a machine-learning problem and may have brought data.
Your job: understand the real problem (not only "build a model") in at most 4 short questions, one at a time,
and keep a solution workflow for it. Ask about: what exactly is predicted and for whom; the moment the prediction
is made and what is known then; what happens with each prediction and what a wrong one costs; and, if no data
was shared, whether they can upload a sample, connect a source, or want simulated data.
Rules: never invent numbers or claim a model will perform well; never ask for passwords or keys; the data summary
is descriptive only. Use tools: ask_user to ask (with 2-4 short options when natural), record to store an answer
you understood, set_pack when the problem clearly fits another domain pack, propose_workflow to replace the
workflow (every step tied to a block WF-01..WF-10, in order; keep WF-01, WF-03, WF-05, WF-09), request_data to
offer upload / connect / simulate, and simulate_data only after the user asked for simulated data (describe the
table they need; it is generated and labelled synthetic). get_profile and get_analysis read the prepared table's
column summaries and descriptive findings (aggregates only) when the state below does not show enough.
The state has a plan: ask only about a field whose status is unknown, plan.next first. When the user's words say
what a field is, record it with quote set to their exact words (an answer to the open question needs no quote).
Reply to the user in plain English, 1-3 sentences."""

def turn(method):
    """One agent turn at a time per draft. The data pipeline ("your data is ready") and the user's messages run in
    different threads; without this a message that lands between two steps of the other turn is read against a
    question that was just withdrawn, and the answer ends up in the notes. The store owns the lock (a thread lock
    for files, an advisory lock in a database), and it is re-entrant."""
    @functools.wraps(method)
    def locked(self, draft_id: str, *args: Any, **kwargs: Any):
        with self.store.turn(draft_id):
            return method(self, draft_id, *args, **kwargs)
    return locked


# Words of a problem sentence that name the setting, not the outcome ("customers", "next month"); compared as 5-letter stems.
GENERIC_STEMS = {w[:5] for w in (
    "predict which what will would could should model models build make want need using based data dataset team first before after "
    "their them they this that with from into each every next last month monthly week weekly year yearly daily today tomorrow "
    "customer customers client clients user users member members people person order orders store stores product products "
    "transaction transactions payment payments account accounts case cases time times").split()}
OPTION_COUNT = 3  # outcome columns offered as buttons, before "Something else" (a question shows at most four options)


def _stems(text: str) -> set[str]:
    words = re.findall(r"[a-z]{4,}", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text)).lower())
    return {w[:5] for w in words} - GENERIC_STEMS


def rank_targets(profile: dict[str, Any], problem: str) -> list[str]:
    """Outcome candidates, with the columns the problem sentence names first ("will cancel" finds `cancelled`).

    Only column names and the sentence are compared: no value is read and nothing is related to any column, so this
    is safe before the split. Among named columns, outcome-like names and few distinct values come first; the
    profile's own candidates follow in their order.
    """
    candidates = [str(c) for c in profile.get("target_candidates") or []]
    words = _stems(problem)
    eligible = [c for c in profile.get("columns") or [] if c.get("kind") != "text" and (c.get("unique") or 0) >= 2 and not c.get("id_like")]
    named = [c for c in eligible if _stems(c.get("name", "")) & words]
    named.sort(key=lambda c: (-int(bool(c.get("target_name_like"))), -int(bool(c.get("low_cardinality")))))
    first = [str(c["name"]) for c in named]
    return (first + [c for c in candidates if c not in first])[:max(8, len(candidates))]


def moment_options(pack: str | None) -> list[str]:
    if pack == "timeseries":
        return ["The day before", "A week ahead", "A month ahead"]
    if pack in ("imbalanced",):
        return ["At the moment of the transaction", "Before approving the case", "In a nightly review"]
    return ["Right before we act (e.g. before the call)", "At a fixed snapshot each month", "As soon as the event happens"]


class HomeAgent:
    def __init__(self, store: DraftStore, client=None, on_request: Callable[[str, str, dict[str, Any]], None] | None = None,
                 registry: Registry | None = None, traces: Any = None):
        self.store, self.client, self.traces = store, client, traces
        self.on_request = on_request or (lambda draft_id, what, args: None)
        if registry is None:
            from dclab_rnd.agents import default_registry
            registry = default_registry()
        self.registry = registry

    # ------------------------------------------------------------------ helpers
    def say(self, draft_id: str, text: str, kind: str = "text", message_id: str | None = None, **extra: Any) -> dict[str, Any]:
        msg = {"id": message_id or "m" + secrets.token_hex(5), "role": "agent", "kind": kind, "text": text, "at": now(), **extra}

        def add(d):
            d["messages"].append(msg)
        self.store.update(draft_id, add)
        self.store.emit(draft_id, "chat", msg)
        return msg

    def ask(self, draft_id: str, field: str, question: str, options: list[str] | None = None, why: str = "") -> dict[str, Any] | None:
        draft = self.store.get(draft_id)
        asked = [q for q in draft["questions"] if not q.get("superseded")]
        if len(asked) >= MAX_QUESTIONS or any(q.get("field") == field and not q.get("answered") for q in draft["questions"]):
            return None
        q = {"id": f"q{len(draft['questions']) + 1}", "field": field, "text": question, "options": (options or [])[:4], "why": why, "answered": None}

        def add(d):
            d["questions"].append(q)
        self.store.update(draft_id, add)
        return self.say(draft_id, question, kind="question", question=q)

    def pending(self, draft: dict[str, Any]) -> dict[str, Any] | None:
        return next((q for q in draft["questions"] if not q.get("answered")), None)

    def record(self, draft_id: str, field: str, value: str) -> None:
        def put(d):
            d.setdefault("understanding", {})[field] = value.strip()[:400]
        self.store.update(draft_id, put)

    def note(self, draft_id: str, field: str, value: str, status: str, source: str, quote: str | None) -> None:
        """Record a field with its plan entry: how it is known and the user's words that say it (package A3.1)."""
        def put(d):
            d.setdefault("understanding", {})[field] = value.strip()[:400]
            d["plan"] = {**plans.empty(), **(d.get("plan") or {}), field: plans.entry(status, source, quote)}
        self.store.update(draft_id, put)

    def update_plan(self, draft_id: str) -> dict[str, Any]:
        """Recompute the plan (what is known, what to ask next and why) and show it on the page."""
        draft = self.store.update(draft_id, lambda d: d.update(plan=plans.refresh(d)))
        self.store.emit(draft_id, "plan", {"plan": draft["plan"], "understanding": draft.get("understanding") or {}})
        return draft["plan"]

    def read_problem(self, draft_id: str) -> list[str]:
        """Fill the plan from the problem sentence: a model's reading when one is configured (each quote checked
        against the sentence), the keyword rules for what it leaves open. Returns the fields it filled."""
        draft = self.store.get(draft_id)
        filled: list[str] = []
        u = draft.get("understanding") or {}
        if self.client is not None:
            try:
                reply = self.client.complete([{"role": "system", "content": plans.PROPOSE}, {"role": "user", "content": draft["problem"]}], None)
                kept, dropped = plans.proposal(reply.get("content") or "", [draft["problem"]])
                if getattr(self.client, "output", None):
                    self.client.output(not dropped, "; ".join(dropped)[:200])
            except Exception:  # noqa: BLE001 — no model, no problem: the keyword rules still read the sentence
                kept = {}
            for field, item in kept.items():
                if not u.get(field):
                    self.note(draft_id, field, item["value"], item["status"], "model", item["quote"])
                    filled.append(field)
        for field, quote in plans.extract(draft["problem"]).items():
            if field not in filled and not u.get(field):
                self.note(draft_id, field, quote, "stated", "problem", quote)
                filled.append(field)
        return filled

    @turn
    def replan(self, draft_id: str) -> None:
        """The problem sentence changed: forget what was read from the old one and read the new one (no model)."""
        def forget(d):
            plan = d.get("plan") or {}
            for field in plans.FIELDS:
                if (plan.get(field) or {}).get("source") in ("problem", "model"):
                    (d.get("understanding") or {}).pop(field, None)
                    plan[field] = {"status": "unknown", "source": None, "quote": None}
        self.store.update(draft_id, forget)
        client, self.client = self.client, None
        try:
            self.read_problem(draft_id)
        finally:
            self.client = client
        self.update_plan(draft_id)
        self.next_turn(draft_id)

    def refresh_workflow(self, draft_id: str, proposed: dict[str, Any] | None = None) -> dict[str, Any]:
        draft = self.store.get(draft_id)
        key = (draft.get("pack") or {}).get("key") or "tabular"
        problems: list[str] = []
        if proposed:
            wf, problems = wflow.validate(proposed, key)
            if getattr(self.client, "output", None):  # the model proposed it: tell the gateway whether code accepted it
                self.client.output(not problems, "; ".join(problems)[:200])
        else:
            wf = None
        if wf is None:
            current = draft.get("workflow")
            wf = current if (current and current.get("source") == "model" and not proposed) else wflow.template(key, draft.get("understanding"))
        wf = wflow.with_states(wf, draft)
        wf["version"] = ((draft.get("workflow") or {}).get("version") or 0) + 1
        wf["title"] = wf.get("title") or title_for(draft, key)

        def put(d):
            d["workflow"] = wf
        self.store.update(draft_id, put)
        self.store.emit(draft_id, "workflow", {"workflow": wf, "rejected": problems})
        return wf

    def set_pack(self, draft_id: str, key: str, source: str, why: str) -> None:
        def put(d):
            d["pack"] = {"key": key, "source": source, "why": why}
        self.store.update(draft_id, put)
        self.store.emit(draft_id, "status", {"pack": {"key": key, "source": source, "why": why}})

    # ------------------------------------------------------------------ entry points
    @turn
    def start(self, draft_id: str) -> None:
        """A new draft with a problem sentence: restate it, pick the pack, draw the workflow, ask the first question."""
        draft = self.store.get(draft_id)
        chosen = (draft.get("pack") or {}).get("key") if (draft.get("pack") or {}).get("source") == "user" else None
        det = packs.detect(draft["problem"], draft.get("analysis"), chosen)
        self.set_pack(draft_id, det["key"], det["source"], det["why"])
        name = (packs.by_key(det["key"]) or {}).get("name", det["key"])
        has_data = bool(draft.get("assets"))
        why = det["why"].rstrip(".")
        filled = self.read_problem(draft_id)
        u = self.store.get(draft_id).get("understanding") or {}
        labels = {"target": "the outcome", "prediction_moment": "the prediction moment", "action": "what happens with each prediction"}
        known = [f"{labels[f]} (“{u[f]}”)" for f in plans.FIELDS if f in filled and f in labels]
        self.say(draft_id, f"I read the problem as: “{draft['problem']}”. I'll start from the {name} pack ({why[:1].lower() + why[1:]})."
                 + (f" Your sentence already gives {' and '.join(known)}; check them on the sheet." if known else "")
                 + (" Your data is being prepared now; I'll describe it as soon as it is ready." if has_data else
                    " You can upload or connect data at any time, or we can plan the solution first."), kind="text")
        self.update_plan(draft_id)
        self.refresh_workflow(draft_id)
        self.next_turn(draft_id, opening=True)

    @turn
    def data_ready(self, draft_id: str, asset: dict[str, Any]) -> None:
        """The table is analysed: describe it briefly, update the pack from data signals and the workflow, ask about the outcome."""
        draft = self.store.get(draft_id)
        analysis = draft.get("analysis") or {}
        s = analysis.get("summary") or {}
        steps = len(draft.get("cleaning_log") or [])
        label = "synthetic " if asset.get("synthetic") else ""
        text = (f"Your {label}data is ready: {s.get('rows', 0):,} rows and {s.get('columns', 0)} columns"
                + (f", after {steps} cleaning step{'s' if steps != 1 else ''}" if steps else "") + ". Here is a first look.")
        self.say(draft_id, text, kind="analysis", asset=asset.get("id"))
        if (draft.get("pack") or {}).get("source") != "user":
            det = packs.detect(draft["problem"], analysis)
            if det["key"] != (draft.get("pack") or {}).get("key") and det["source"] in ("problem", "data"):
                self.set_pack(draft_id, det["key"], det["source"], det["why"])
        self.refresh_workflow(draft_id)
        draft = self.store.get(draft_id)
        pending = self.pending(draft)
        candidates = ((draft.get("analysis") or {}).get("profile") or {}).get("target_candidates") or []
        if pending and pending["field"] == "target" and not pending.get("options") and candidates:
            # The outcome was asked before the data arrived; ask again with the columns to choose from.
            self.store.update(draft_id, lambda d: [q.update(answered="(asked again with the columns)", superseded=True) for q in d["questions"] if q["id"] == pending["id"]])
            pending = None
        if not pending:
            self.next_turn(draft_id)

    @turn
    def data_failed(self, draft_id: str, asset: dict[str, Any], error: str) -> None:
        self.say(draft_id, f"I could not use {asset.get('name', 'that file')}: {error}", kind="text", severity="warning")
        self.refresh_workflow(draft_id)

    @turn
    def reply(self, draft_id: str, text: str) -> None:
        """A user message: store it, then let the model (or the script) answer."""
        text = text.strip()[:2000]
        msg = {"id": "u" + secrets.token_hex(5), "role": "user", "kind": "text", "text": text, "at": now()}

        def add(d):
            d["messages"].append(msg)
        self.store.update(draft_id, add)
        self.store.emit(draft_id, "chat", msg)
        if self.client is not None:
            try:
                self.model_turn(draft_id, text=text)
                return
            except Exception as exc:  # noqa: BLE001 — provider errors: fall back to the script for this turn
                reason = str(exc).split(": ", 1)[-1][:120]
                if (self.store.get(draft_id).get("agent") or {}).get("model_note") != reason:  # say it once, not on every turn
                    self.store.update(draft_id, lambda d: d["agent"].update(model_note=reason))
                    self.store.emit(draft_id, "status", {"note": f"The model was unavailable ({reason}); DCLab continues with its standard questions."})
        self.script_turn(draft_id, text)

    def take_answer(self, draft_id: str, q: dict[str, Any], value: str) -> None:
        """The user's message answers the open question: record it, redraw the workflow, ask the next thing."""
        draft = self.store.get(draft_id)
        if q["field"] in plans.FIELDS:
            self.note(draft_id, q["field"], value, "stated", "answer", value)
        else:
            self.record(draft_id, q["field"], value)

        def answered(d):
            for item in d["questions"]:
                if item["id"] == q["id"]:
                    item["answered"] = value
        self.store.update(draft_id, answered)
        u = self.store.get(draft_id).get("understanding") or {}
        for field, quote in plans.extract(value).items():  # "…within 30 days, scored every Monday" also gives the moment
            if field != q["field"] and not u.get(field):
                self.note(draft_id, field, quote, "stated", "answer", quote)
        self.update_plan(draft_id)
        if q["field"] == "data_plan" and SIMULATE_WORDS.search(value):
            self.on_request(draft_id, "simulate", {"prompt": draft["problem"]})
        self.refresh_workflow(draft_id)
        self.next_turn(draft_id)

    # ------------------------------------------------------------------ the deterministic script
    def script_turn(self, draft_id: str, text: str) -> None:
        draft = self.store.get(draft_id)
        q = self.pending(draft)
        if q is not None:
            self.take_answer(draft_id, q, text)
            return
        if SIMULATE_WORDS.search(text):
            self.say(draft_id, "I'll simulate a dataset from this conversation. It will be labelled synthetic everywhere.")
            self.on_request(draft_id, "simulate", {"prompt": draft["problem"] + " " + text})
            return
        if BUILD_WORDS.search(text):
            self.summarize(draft_id)
            return
        self.record(draft_id, "notes", ((draft.get("understanding") or {}).get("notes", "") + " " + text).strip())
        self.say(draft_id, "Noted. I added it to the solution notes.")
        self.next_turn(draft_id)

    def next_turn(self, draft_id: str, opening: bool = False) -> None:
        """Ask the next missing thing, or summarise when everything needed is known."""
        draft = self.store.get(draft_id)
        if self.pending(draft):
            return
        self.update_plan(draft_id)
        question = self.next_question(self.store.get(draft_id), opening)
        if question is not None and self.ask(draft_id, *question):
            return
        if not opening:
            self.summarize(draft_id)

    def next_question(self, draft: dict[str, Any], opening: bool = False) -> tuple[str, str, list[str], str] | None:
        """The question the plan asks next (field, text, options, why), or None when nothing needs asking. Only an
        unknown field is asked, in the order of plans.FIELDS (the moment before the cost); an outcome known in words
        is asked again only for its column, once a table with outcome-like columns is there."""
        plan = plans.refresh(draft)
        key = (draft.get("pack") or {}).get("key")
        has_data = any(a.get("status") in ("ready", "queued", "structuring", "cleaning", "analysing") for a in draft["assets"])
        candidates = ((draft.get("analysis") or {}).get("profile") or {}).get("target_candidates") or []
        # a studied sample or a synthetic table knows its own outcome column: offer it first
        active = next((a for a in draft["assets"] if a.get("id") == draft.get("active_asset")), None) or {}
        known = (active.get("suggestion") or {}).get("target")
        columns = {c.get("name") for c in ((draft.get("analysis") or {}).get("columns") or [])}
        if known and (known in candidates or known in columns):
            candidates = [known] + [c for c in candidates if c != known]
        asked = {q["field"] for q in draft["questions"] if not q.get("superseded")}
        u = draft.get("understanding") or {}
        column_asked = any(q["field"] == "target" and q.get("options") and not q.get("superseded") for q in draft["questions"])
        for field in plans.FIELDS:
            if field == "target" and plan[field]["status"] != "unknown" and candidates and columns and u.get("target") not in columns and not column_asked:
                # the outcome is known in words, but the table needs its column: "cancel next month" is which one?
                return ("target", f"You want to predict “{u['target']}”. Which column records that outcome?",
                        candidates[:OPTION_COUNT] + ["Something else"], "The model learns from the column, so it must be the outcome you described.")
            if plan[field]["status"] != "unknown" or field in asked:
                continue
            if field == "target":
                if candidates:
                    return ("target", "Which column is the outcome the model should predict?", candidates[:OPTION_COUNT] + ["Something else"],
                            "The outcome decides the task, the metric and which columns could leak it.")
                if not has_data or opening:
                    return ("target", "What exactly should the model predict, and for whom? For example: will this customer leave in the next 30 days?",
                            [], "Everything else follows from the outcome.")
                continue  # a table without an outcome-like column: ask the rest first
            if field == "prediction_moment":
                return ("prediction_moment", "When is the prediction made, and what is already known at that moment?", moment_options(key),
                        plans.WHY["prediction_moment"])
            if field == "action":
                return ("action", "What happens with each prediction, and what does a wrong one cost, roughly?",
                        ["We contact or act on the top cases", "We block or review cases", "We plan capacity or stock", "It informs a person only"],
                        plans.WHY["action"])
            if field == "data_plan":
                return ("data_plan", "Can you share a sample of the data, connect a source, or should I simulate data from this conversation?",
                        ["Upload a sample", "Connect a source", "Simulate data", "No data yet, plan only"], plans.WHY["data_plan"])
        return None

    def answerable(self, draft: dict[str, Any]) -> dict[str, Any] | None:
        """The open question, if the user has written since it was asked (so a message can answer it). A question the
        model asked in this very reply has no answer yet, and recording one for it would put words in the user's mouth."""
        q = self.pending(draft)
        if q is None:
            return None
        messages = draft.get("messages") or []
        at = next((i for i, m in enumerate(messages) if m.get("kind") == "question" and (m.get("question") or {}).get("id") == q["id"]), None)
        return q if at is not None and any(m.get("role") == "user" for m in messages[at + 1:]) else None

    def summarize(self, draft_id: str) -> None:
        draft = self.store.get(draft_id)
        u = draft.get("understanding") or {}
        name = (packs.by_key((draft.get("pack") or {}).get("key")) or {}).get("name", "Tabular")
        line = lambda label, value: f"{label}: {str(value).strip().rstrip('.')}." if value else None  # noqa: E731 — answers are quoted as given
        parts = [line("Outcome", u.get("target")) or "The outcome is still open.", line("Prediction moment", u.get("prediction_moment")),
                 line("What happens with each prediction", u.get("action"))]
        ready = any(a.get("status") == "ready" for a in draft["assets"])
        text = ("Here is the solution I would build. " + " ".join(p for p in parts if p) + f" Pack: {name}. "
                + ("The data is ready, so we can build the solution now." if ready else
                   "We can build the solution plan now and add data in the next step."))
        last = next((m for m in reversed(draft["messages"]) if m["role"] == "agent"), None)
        if last and last.get("kind") == "summary" and last.get("text") == text:
            return  # two turns can end here (the data-ready turn and the reply that started a simulation): say it once
        self.say(draft_id, text, kind="summary", actions=["build"])

    # ------------------------------------------------------------------ the model
    def context(self, draft: dict[str, Any]) -> str:
        a = draft.get("analysis") or {}
        cols = [{k: c.get(k) for k in PROFILE_FIELDS} for c in (a.get("columns") or [])[:PROFILE_PAGE]]
        return json.dumps({
            "problem": draft["problem"], "pack": draft.get("pack"), "understanding": draft.get("understanding") or {},
            "plan": {**{f: ((draft.get("plan") or {}).get(f) or {}).get("status", "unknown") for f in plans.FIELDS}, "next": (draft.get("plan") or {}).get("next")},
            "questions_asked": [{k: q.get(k) for k in ("field", "text", "answered")} for q in draft["questions"]],
            "questions_left": MAX_QUESTIONS - len(draft["questions"]),
            "data": {"summary": a.get("summary"), "columns": cols, "highlights": [h.get("title") for h in (a.get("highlights") or [])][:8],
                     "target_candidates": (a.get("profile") or {}).get("target_candidates")} if a else None,
            "assets": [{k: x.get(k) for k in ("name", "kind", "status", "synthetic")} for x in draft["assets"]],
            "workflow": [{k: n.get(k) for k in ("wf", "label")} for n in ((draft.get("workflow") or {}).get("nodes") or [])],
        }, ensure_ascii=False, default=str)

    def policy(self, max_replies: int = MODEL_REPLIES) -> Policy:
        """The Home agent as a policy on the runtime: a few model replies per user message; a question waits for the
        user and a simulation ends the turn too (the data pipeline speaks next)."""
        return Policy("home", POLICY, tuple(self.registry.names("draft")), max_steps=MODEL_STEPS, max_seconds=MODEL_SECONDS,
                      stop_after=("ask_user",), stop_when=lambda step: isinstance(step.result, dict) and bool(step.result.get("simulating")),
                      max_replies=max_replies)

    def tracer(self, draft_id: str) -> Tracer | None:
        if self.traces is None:
            return None
        return Tracer.resume(self.traces, draft_id, "home", lambda: understood(self.store.get(draft_id)))

    def model_turn(self, draft_id: str, max_replies: int = MODEL_REPLIES, text: str = "") -> None:
        draft = self.store.get(draft_id)
        open_question = self.pending(draft)
        history = [{"role": "user" if m["role"] == "user" else "assistant", "content": m["text"]} for m in draft["messages"][-12:] if m.get("text")]
        messages = [{"role": "system", "content": POLICY}, {"role": "system", "content": "Current state: " + self.context(draft)}, *history]

        def on_step(step: Step) -> None:
            # A proposal the guard or the workflow check saw reports its own verdict; one the schema refused does not.
            own = step.tool == "propose_workflow" and isinstance(step.result, dict) and ("accepted" in step.result or step.ok)
            if not own and getattr(self.client, "output", None):
                self.client.output(step.ok, "" if step.ok else "the tool call was refused")

        model = _Model(self, draft_id)
        self.store.emit(draft_id, "status", {"thinking": True})
        try:
            out = run_policy(self.policy(max_replies), model, self.registry, messages, context=Turn(self, draft_id), on_step=on_step,
                             trace=self.tracer(draft_id))
        finally:
            self.store.emit(draft_id, "status", {"thinking": False})
        if out.stopped == "answered" and out.final:
            self.say(draft_id, out.final, message_id=model.live)  # the chat event replaces the live bubble
        self.store.update(draft_id, lambda d: d["agent"].update(mode="model", turns=d["agent"].get("turns", 0) + 1))
        if not out.steps and text:
            # A model that only talks (small models often never call a tool) must not stall the conversation:
            # code records the answer to the open question and asks the next thing, as the script would.
            still_open = self.pending(self.store.get(draft_id))
            if open_question and still_open and still_open["id"] == open_question["id"]:
                self.take_answer(draft_id, open_question, text)
            elif SIMULATE_WORDS.search(text) and not self.store.get(draft_id)["assets"]:
                self.on_request(draft_id, "simulate", {"prompt": draft["problem"] + " " + text})
            else:
                self.next_turn(draft_id)

    def complete(self, draft_id: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], str | None]:
        """One model step. With a streaming client the reply reaches the page as ``token`` events while it is written.

        Returns the reply and the id of the live bubble (None when nothing was streamed). Deltas are sent in chunks
        of ``TOKEN_CHUNK`` characters, not one by one: every event is a line of the draft's event log, which a page
        that reconnects replays. A reply that ends in tool calls, or fails, is never said, so its bubble is dropped.
        """
        stream = getattr(self.client, "stream", None)
        if stream is None:
            return self.client.complete(messages, tools if tools is not None else self.registry.schemas("draft")), None
        live, buffer = "m" + secrets.token_hex(5), []

        def flush(force: bool = False) -> None:
            if buffer and (force or sum(map(len, buffer)) >= TOKEN_CHUNK):
                self.store.emit(draft_id, "token", {"id": live, "delta": "".join(buffer)})
                buffer.clear()

        def on_text(delta: str) -> None:
            buffer.append(delta)
            flush()

        try:
            out = stream(messages, tools if tools is not None else self.registry.schemas("draft"), on_text=on_text)
            flush(True)
        except Exception:
            self.store.emit(draft_id, "token", {"id": live, "drop": True})
            raise
        if out.get("tool_calls") or not out.get("content"):
            self.store.emit(draft_id, "token", {"id": live, "drop": True})
            return out, None
        return out, live

    def run_tool(self, draft_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run one of the Home agent's tools on a draft, through the draft guard. The handlers are lenient (small models
        send strings for numbers, extra keys, and leave out what the handlers default): unknown keys are dropped and the
        rest is the handlers' to coerce and default. Used where code, not the model, makes the call; the model's calls go
        through the runtime, which checks them against the schemas."""
        tool = self.registry.get(name)
        if tool is None or tool.scope != "draft":
            return {"error": f"unknown tool {name}"}
        known = {k: v for k, v in args.items() if k in tool.parameters.get("properties", {})}
        return self.registry.call(tool.name, known, Turn(self, draft_id), scope="draft", check="none")


def title_for(draft: dict[str, Any], key: str) -> str:
    name = (packs.by_key(key) or {}).get("name", key)
    target = (draft.get("understanding") or {}).get("target")
    return f"{target} · {name}" if target else f"Solution workflow · {name}"


def understood(draft: dict[str, Any]) -> str:
    """The draft's state for a trace row: which of the fields are known (x) or open (-), in FIELDS order."""
    u = draft.get("understanding") or {}
    return "".join("x" if u.get(f) else "-" for f in FIELDS)


class _Model:
    """The gateway client as the runtime sees it for one Home turn: a streamed reply reaches the page as it is
    written, and the id of the last reply's live bubble is kept so the final answer replaces it."""

    def __init__(self, agent: HomeAgent, draft_id: str):
        self.agent, self.draft_id, self.live = agent, draft_id, None

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        out, self.live = self.agent.complete(self.draft_id, messages, tools)
        return out


# ---------------------------------------------------------------------- tools
@dataclass(frozen=True)
class Turn:
    """What a Home tool acts on: the agent (its store, client and request hook) and one draft."""
    agent: HomeAgent
    draft_id: str


def ask_user(turn: Turn, /, **args: Any) -> dict[str, Any]:
    field = args.get("field") if args.get("field") in FIELDS else "notes"
    msg = turn.agent.ask(turn.draft_id, field, str(args.get("question", ""))[:300], [str(o)[:60] for o in args.get("options") or []], str(args.get("why", ""))[:200])
    return {"asked": bool(msg), "note": None if msg else "Question limit reached or the same question is open."}


def record(turn: Turn, /, **args: Any) -> dict[str, Any]:
    agent, draft_id = turn.agent, turn.draft_id
    field, value = str(args.get("field", "notes")), str(args.get("value", ""))
    pending = agent.answerable(agent.store.get(draft_id))
    if pending and pending["field"] == field:  # the user's message answers the open question
        if field in plans.FIELDS:
            agent.note(draft_id, field, value, "stated", "answer", value)
        else:
            agent.record(draft_id, field, value)
        agent.store.update(draft_id, lambda d: [q.update(answered=value) for q in d["questions"] if q["id"] == pending["id"]])
    elif field in plans.FIELDS and args.get("quote"):  # the user said it elsewhere; the guard checked the quote
        agent.note(draft_id, field, value, args.get("status") or "stated", "model", str(args["quote"]))
    else:
        agent.record(draft_id, field, value)
    agent.update_plan(draft_id)
    agent.refresh_workflow(draft_id)
    return {"recorded": field}


def set_pack(turn: Turn, /, **args: Any) -> dict[str, Any]:
    agent, draft_id = turn.agent, turn.draft_id
    agent.set_pack(draft_id, args["key"], "agent", str(args.get("why", ""))[:200])
    agent.refresh_workflow(draft_id)
    return {"pack": args["key"]}


def propose_workflow(turn: Turn, /, **args: Any) -> dict[str, Any]:
    wf = turn.agent.refresh_workflow(turn.draft_id, proposed=args)
    return {"accepted": wf.get("source") == "model", "version": wf["version"]}


def request_data(turn: Turn, /, **args: Any) -> dict[str, Any]:
    turn.agent.say(turn.draft_id, str(args.get("text", ""))[:400] or "You can upload a sample, connect a source, or ask me to simulate data.",
                   kind="text", actions=["upload", "connect", "simulate"])
    return {"offered": True}


def get_profile(turn: Turn, /, **args: Any) -> dict[str, Any]:
    analysis = turn.agent.store.get(turn.draft_id).get("analysis") or {}
    columns = analysis.get("columns") or []
    if not columns:
        return {"error": "No table has been prepared yet."}
    wanted = {str(c) for c in args.get("columns") or []}
    try:
        offset = max(0, int(args.get("offset") or 0))
    except (TypeError, ValueError):
        offset = 0
    picked = [c for c in columns if c.get("name") in wanted] if wanted else columns[offset:offset + PROFILE_PAGE]
    profile = analysis.get("profile") or {}
    return {"total_columns": len(columns), "offset": 0 if wanted else offset,
            "columns": [{k: c.get(k) for k in PROFILE_FIELDS} for c in picked[:PROFILE_PAGE]],  # aggregates, never cell values
            **{k: profile.get(k) or [] for k in ("target_candidates", "time_candidates", "id_candidates", "text_candidates")}}


def get_analysis(turn: Turn, /, **args: Any) -> dict[str, Any]:
    analysis = turn.agent.store.get(turn.draft_id).get("analysis") or {}
    if not analysis.get("columns"):
        return {"error": "No table has been prepared yet."}
    # descriptive only: the analysis never relates a column to the outcome
    return {"summary": analysis.get("summary"), "correlations": (analysis.get("correlations") or [])[:10],
            "highlights": [{k: h.get(k) for k in ("severity", "title", "text", "columns")} for h in (analysis.get("highlights") or [])[:12]]}


def simulate_data(turn: Turn, /, **args: Any) -> dict[str, Any]:
    agent, draft_id = turn.agent, turn.draft_id
    draft = agent.store.get(draft_id)
    agent.say(draft_id, "I'll simulate a dataset from this conversation. It will be labelled synthetic everywhere.")
    try:
        rows = max(100, min(int(args.get("rows") or 5000), 200_000))
    except (TypeError, ValueError):
        rows = 5000
    agent.on_request(draft_id, "simulate", {"prompt": (draft["problem"] + " " + str(args.get("description", ""))[:2000]).strip(), "rows": rows})
    return {"simulating": True, "rows": rows}


# What each Home write does to a draft. The draft guard checks set_pack, propose_workflow and simulate; the others
# only add to the conversation or the understanding, which the user can always correct.
DRAFT_MOVES = {"ask_user": "ask", "record": "record", "set_pack": "set_pack", "propose_workflow": "propose_workflow",
               "request_data": "request_data", "simulate_data": "simulate"}
PREPARING = ("queued", "structuring", "cleaning", "analysing")


class DraftGuard:
    """The Home agent's writes change a draft, not a project, so they have their own checks (package A2.2):
    the pack must be on the list and the user's choice stands; a workflow must pass ``workflow.validate``; one
    simulation (or any data preparation) at a time."""

    def __call__(self, tool: Tool, turn: Turn, arguments: dict[str, Any], proceed: Callable[[], Any]) -> Any:
        agent, draft_id = turn.agent, turn.draft_id
        if tool.move == "record" and arguments.get("field") in plans.FIELDS:
            draft = agent.store.get(draft_id)
            pending = agent.answerable(draft)
            if not (pending and pending["field"] == arguments["field"]):
                if not arguments.get("quote"):
                    return {"error": f"The user has not answered a question about {arguments['field']}: give quote, their exact words that say it."}
                if not plans.quoted(arguments["quote"], plans.user_text(draft)):
                    return {"error": "The quote is not in the user's words; copy whole words exactly from the problem or their messages."}
                if not plans.supports(arguments["quote"], arguments.get("value")):
                    return {"error": "The value does not say what the quote says; record what the user's words mean."}
        if tool.move == "ask" and arguments.get("field") in plans.FIELDS:
            question = agent.next_question(agent.store.get(draft_id))
            if question is None or question[0] != arguments["field"]:
                nxt = f"ask about {question[0]} next" if question else "nothing in the plan is open; summarise instead"
                return {"error": f"The plan does not ask about {arguments['field']} now: {nxt}."}
        if tool.move == "set_pack":
            if arguments.get("key") not in packs.KEYS:
                return {"error": f"Unknown pack {str(arguments.get('key'))[:40]!r}. Packs: {', '.join(packs.KEYS)}."}
            if (agent.store.get(draft_id).get("pack") or {}).get("source") == "user":
                return {"error": "The user chose the pack; ask before changing it."}
        elif tool.move == "simulate":
            if any(a.get("status") in PREPARING for a in agent.store.get(draft_id)["assets"]):
                return {"error": "Data is already being prepared; wait for it."}
        elif tool.move == "propose_workflow":
            draft = agent.store.get(draft_id)
            _, problems = wflow.validate(arguments, (draft.get("pack") or {}).get("key") or "tabular")
            if problems:  # the current workflow stays; the page shows what was refused
                if getattr(agent.client, "output", None):
                    agent.client.output(False, "; ".join(problems)[:200])
                agent.store.emit(draft_id, "workflow", {"workflow": draft.get("workflow"), "rejected": problems})
                return {"error": "The workflow was not accepted: " + "; ".join(problems)[:400], "accepted": False, "problems": problems}
        return proceed()


def register(registry: Registry) -> None:
    """Register the Home agent's tools (scope "draft"). Each takes a ``Turn`` as its context; each write declares
    its draft move and passes the draft guard."""
    registry.guards["draft"] = DraftGuard()

    def add(name: str, description: str, properties: dict[str, Any], required: list[str], effect: str) -> None:
        registry.register(Tool(name, description, {"type": "object", "properties": properties, "required": required}, globals()[name],
                               effect=effect, scope="draft", move=DRAFT_MOVES.get(name) if effect == "write" else None, takes_context=True))

    S = {"type": "string"}
    add("ask_user", "Ask the user one short question.", {"question": S, "options": {"type": "array", "items": S},
        "field": {**S, "enum": list(FIELDS)}, "why": S}, ["question"], "write")  # no field: the answer goes to the notes
    add("record", "Store what the user told you. For target, prediction_moment, action or data_plan, outside an answer to the open "
        "question, give quote: the user's exact words that say it.",
        {"field": {**S, "enum": list(FIELDS) + ["unit", "horizon", "constraints", "notes"]}, "value": S, "quote": S,
         "status": {**S, "enum": ["stated", "inferred"]}}, ["field", "value"], "write")
    add("set_pack", "Switch the domain pack.", {"key": {**S, "enum": packs.KEYS}, "why": S}, ["key"], "write")
    add("propose_workflow", "Replace the solution workflow.", {"title": S, "nodes": {"type": "array", "items": {"type": "object", "properties": {
        "wf": S, "label": S, "detail": S}, "required": ["wf", "label"]}}}, ["nodes"], "write")
    add("request_data", "Offer the user ways to bring data.", {"text": S}, [], "write")  # no text: a standard offer
    add("get_profile", "Column summaries of the prepared table: name, kind, missing rate, unique count, and which columns look like outcomes, timestamps, identifiers or text. Aggregates only, never values. Name columns, or page through a wide table with offset.",
        {"columns": {"type": "array", "items": S}, "offset": {"type": "integer"}}, [], "read")
    add("get_analysis", "The descriptive analysis of the prepared table: size, column kinds, missing share, duplicate rows, findings worth a question, and the strongest feature-to-feature correlations. Says nothing about the outcome.",
        {}, [], "read")
    add("simulate_data", "Generate a synthetic table. Only after the user asked for simulated data.",
        {"description": {**S, "description": "The table to simulate: rows, columns, the outcome and its rate."}, "rows": {"type": "integer"}},
        ["description"], "write")
