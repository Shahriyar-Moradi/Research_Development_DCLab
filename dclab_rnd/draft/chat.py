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
import threading
from typing import Any, Callable

from . import pack as packs
from . import workflow as wflow
from .store import DraftStore, now

MAX_QUESTIONS = 4
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
Reply to the user in plain English, 1-3 sentences."""

TOOLS = [
    {"type": "function", "function": {"name": "ask_user", "description": "Ask the user one short question.", "parameters": {
        "type": "object", "properties": {"question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}},
                                         "field": {"type": "string", "enum": list(FIELDS)}, "why": {"type": "string"}},
        "required": ["question", "field"]}}},
    {"type": "function", "function": {"name": "record", "description": "Store an answer you understood.", "parameters": {
        "type": "object", "properties": {"field": {"type": "string", "enum": list(FIELDS) + ["unit", "horizon", "constraints", "notes"]},
                                         "value": {"type": "string"}}, "required": ["field", "value"]}}},
    {"type": "function", "function": {"name": "set_pack", "description": "Switch the domain pack.", "parameters": {
        "type": "object", "properties": {"key": {"type": "string", "enum": packs.KEYS}, "why": {"type": "string"}}, "required": ["key", "why"]}}},
    {"type": "function", "function": {"name": "propose_workflow", "description": "Replace the solution workflow.", "parameters": {
        "type": "object", "properties": {"title": {"type": "string"}, "nodes": {"type": "array", "items": {"type": "object", "properties": {
            "wf": {"type": "string"}, "label": {"type": "string"}, "detail": {"type": "string"}}, "required": ["wf", "label"]}}},
        "required": ["nodes"]}}},
    {"type": "function", "function": {"name": "request_data", "description": "Offer the user ways to bring data.", "parameters": {
        "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "get_profile", "description": "Column summaries of the prepared table: name, kind, missing rate, unique count, and which columns look like outcomes, timestamps, identifiers or text. Aggregates only, never values. Name columns, or page through a wide table with offset.", "parameters": {
        "type": "object", "properties": {"columns": {"type": "array", "items": {"type": "string"}}, "offset": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "get_analysis", "description": "The descriptive analysis of the prepared table: size, column kinds, missing share, duplicate rows, findings worth a question, and the strongest feature-to-feature correlations. Says nothing about the outcome.", "parameters": {
        "type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "simulate_data", "description": "Generate a synthetic table. Only after the user asked for simulated data.", "parameters": {
        "type": "object", "properties": {"description": {"type": "string", "description": "The table to simulate: rows, columns, the outcome and its rate."},
                                         "rows": {"type": "integer"}}, "required": ["description"]}}},
]


_TURNS: dict[tuple[str, str], threading.RLock] = {}
_TURNS_GUARD = threading.Lock()


def turn(method):
    """One agent turn at a time per draft. The data pipeline ("your data is ready") and the user's messages run in
    different threads; without this a message that lands between two steps of the other turn is read against a
    question that was just withdrawn, and the answer ends up in the notes. Re-entrant: a reply that starts a
    simulation runs the pipeline, and its data-ready turn, in the same thread."""
    @functools.wraps(method)
    def locked(self, draft_id: str, *args: Any, **kwargs: Any):
        key = (str(self.store.home), draft_id)
        with _TURNS_GUARD:
            lock = _TURNS.setdefault(key, threading.RLock())
        with lock:
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
    def __init__(self, store: DraftStore, client=None, on_request: Callable[[str, str, dict[str, Any]], None] | None = None):
        self.store, self.client = store, client
        self.on_request = on_request or (lambda draft_id, what, args: None)

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

    def refresh_workflow(self, draft_id: str, proposed: dict[str, Any] | None = None) -> dict[str, Any]:
        draft = self.store.get(draft_id)
        key = (draft.get("pack") or {}).get("key") or "tabular"
        problems: list[str] = []
        if proposed:
            wf, problems = wflow.validate(proposed, key)
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
        self.say(draft_id, f"I read the problem as: “{draft['problem']}”. I'll start from the {name} pack ({why[:1].lower() + why[1:]})."
                 + (" Your data is being prepared now; I'll describe it as soon as it is ready." if has_data else
                    " You can upload or connect data at any time, or we can plan the solution first."), kind="text")
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
                self.model_turn(draft_id)
                return
            except Exception as exc:  # noqa: BLE001 — provider errors: fall back to the script for this turn
                reason = str(exc).split(": ", 1)[-1][:120]
                if (self.store.get(draft_id).get("agent") or {}).get("model_note") != reason:  # say it once, not on every turn
                    self.store.update(draft_id, lambda d: d["agent"].update(model_note=reason))
                    self.store.emit(draft_id, "status", {"note": f"The model was unavailable ({reason}); DCLab continues with its standard questions."})
        self.script_turn(draft_id, text)

    # ------------------------------------------------------------------ the deterministic script
    def script_turn(self, draft_id: str, text: str) -> None:
        draft = self.store.get(draft_id)
        q = self.pending(draft)
        if q is not None:
            value = text
            self.record(draft_id, q["field"], value)

            def answered(d):
                for item in d["questions"]:
                    if item["id"] == q["id"]:
                        item["answered"] = value
            self.store.update(draft_id, answered)
            if q["field"] == "data_plan" and SIMULATE_WORDS.search(value):
                self.on_request(draft_id, "simulate", {"prompt": draft["problem"]})
            self.refresh_workflow(draft_id)
            self.next_turn(draft_id)
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
        u = draft.get("understanding") or {}
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
        if "target" not in u and "target" not in asked:
            if candidates:
                self.ask(draft_id, "target", "Which column is the outcome the model should predict?", candidates[:OPTION_COUNT] + ["Something else"],
                         "The outcome decides the task, the metric and which columns could leak it.")
                return
            if not has_data or opening:
                self.ask(draft_id, "target", "What exactly should the model predict, and for whom? For example: will this customer leave in the next 30 days?",
                         [], "Everything else follows from the outcome.")
                return
        if "prediction_moment" not in u and "prediction_moment" not in asked:
            self.ask(draft_id, "prediction_moment", "When is the prediction made, and what is already known at that moment?", moment_options(key),
                     "Anything written after this moment must stay out of the model; it is the main source of leakage.")
            return
        if "action" not in u and "action" not in asked:
            self.ask(draft_id, "action", "What happens with each prediction, and what does a wrong one cost, roughly?",
                     ["We contact or act on the top cases", "We block or review cases", "We plan capacity or stock", "It informs a person only"],
                     "The cost of errors sets the metric and the operating point.")
            return
        if not has_data and "data_plan" not in u and "data_plan" not in asked:
            self.ask(draft_id, "data_plan", "Can you share a sample of the data, connect a source, or should I simulate data from this conversation?",
                     ["Upload a sample", "Connect a source", "Simulate data", "No data yet, plan only"],
                     "A sample (even 1,000 rows) lets DCLab check the columns against the prediction moment.")
            return
        if not opening:
            self.summarize(draft_id)

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
            "questions_asked": [{k: q.get(k) for k in ("field", "text", "answered")} for q in draft["questions"]],
            "questions_left": MAX_QUESTIONS - len(draft["questions"]),
            "data": {"summary": a.get("summary"), "columns": cols, "highlights": [h.get("title") for h in (a.get("highlights") or [])][:8],
                     "target_candidates": (a.get("profile") or {}).get("target_candidates")} if a else None,
            "assets": [{k: x.get(k) for k in ("name", "kind", "status", "synthetic")} for x in draft["assets"]],
            "workflow": [{k: n.get(k) for k in ("wf", "label")} for n in ((draft.get("workflow") or {}).get("nodes") or [])],
        }, ensure_ascii=False, default=str)

    def model_turn(self, draft_id: str, max_steps: int = 5) -> None:
        draft = self.store.get(draft_id)
        history = [{"role": "user" if m["role"] == "user" else "assistant", "content": m["text"]} for m in draft["messages"][-12:] if m.get("text")]
        messages = [{"role": "system", "content": POLICY}, {"role": "system", "content": "Current state: " + self.context(draft)}, *history]
        self.store.emit(draft_id, "status", {"thinking": True})
        asked = False
        for _ in range(max_steps):
            out, live = self.complete(draft_id, messages)
            calls = out.get("tool_calls") or []
            messages.append(out["assistant_message"])
            if not calls:
                if out.get("content"):
                    self.say(draft_id, out["content"].strip(), message_id=live)  # the chat event replaces the live bubble
                break
            for call in calls:
                result = self.run_tool(draft_id, call["name"], call.get("arguments") or {})
                # a question waits for the user; a simulation ends the turn too (the data pipeline speaks next)
                asked = asked or call["name"] == "ask_user" or bool(result.get("simulating"))
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, default=str)[:4000]})
            if asked:
                break
        self.store.emit(draft_id, "status", {"thinking": False})
        self.store.update(draft_id, lambda d: d["agent"].update(mode="model", turns=d["agent"].get("turns", 0) + 1))

    def complete(self, draft_id: str, messages: list[dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
        """One model step. With a streaming client the reply reaches the page as ``token`` events while it is written.

        Returns the reply and the id of the live bubble (None when nothing was streamed). Deltas are sent in chunks
        of ``TOKEN_CHUNK`` characters, not one by one: every event is a line of the draft's event log, which a page
        that reconnects replays. A reply that ends in tool calls, or fails, is never said, so its bubble is dropped.
        """
        stream = getattr(self.client, "stream", None)
        if stream is None:
            return self.client.complete(messages, TOOLS), None
        live, buffer = "m" + secrets.token_hex(5), []

        def flush(force: bool = False) -> None:
            if buffer and (force or sum(map(len, buffer)) >= TOKEN_CHUNK):
                self.store.emit(draft_id, "token", {"id": live, "delta": "".join(buffer)})
                buffer.clear()

        def on_text(delta: str) -> None:
            buffer.append(delta)
            flush()

        try:
            out = stream(messages, TOOLS, on_text=on_text)
            flush(True)
        except Exception:
            self.store.emit(draft_id, "token", {"id": live, "drop": True})
            raise
        if out.get("tool_calls") or not out.get("content"):
            self.store.emit(draft_id, "token", {"id": live, "drop": True})
            return out, None
        return out, live

    def run_tool(self, draft_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "ask_user":
            field = args.get("field") if args.get("field") in FIELDS else "notes"
            msg = self.ask(draft_id, field, str(args.get("question", ""))[:300], [str(o)[:60] for o in args.get("options") or []], str(args.get("why", ""))[:200])
            return {"asked": bool(msg), "note": None if msg else "Question limit reached or the same question is open."}
        if name == "record":
            self.record(draft_id, str(args.get("field", "notes")), str(args.get("value", "")))
            pending = self.pending(self.store.get(draft_id))
            if pending and pending["field"] == args.get("field"):
                self.store.update(draft_id, lambda d: [q.update(answered=str(args.get("value", ""))) for q in d["questions"] if q["id"] == pending["id"]])
            self.refresh_workflow(draft_id)
            return {"recorded": args.get("field")}
        if name == "set_pack":
            if args.get("key") not in packs.KEYS:
                return {"error": "unknown pack"}
            if (self.store.get(draft_id).get("pack") or {}).get("source") == "user":
                return {"error": "The user chose the pack; ask before changing it."}
            self.set_pack(draft_id, args["key"], "agent", str(args.get("why", ""))[:200])
            self.refresh_workflow(draft_id)
            return {"pack": args["key"]}
        if name == "propose_workflow":
            wf = self.refresh_workflow(draft_id, proposed=args)
            return {"accepted": wf.get("source") == "model", "version": wf["version"]}
        if name == "request_data":
            self.say(draft_id, str(args.get("text", ""))[:400] or "You can upload a sample, connect a source, or ask me to simulate data.",
                     kind="text", actions=["upload", "connect", "simulate"])
            return {"offered": True}
        if name in ("get_profile", "get_analysis"):
            analysis = self.store.get(draft_id).get("analysis") or {}
            columns = analysis.get("columns") or []
            if not columns:
                return {"error": "No table has been prepared yet."}
            if name == "get_analysis":  # descriptive only: the analysis never relates a column to the outcome
                return {"summary": analysis.get("summary"), "correlations": (analysis.get("correlations") or [])[:10],
                        "highlights": [{k: h.get(k) for k in ("severity", "title", "text", "columns")} for h in (analysis.get("highlights") or [])[:12]]}
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
        if name == "simulate_data":
            draft = self.store.get(draft_id)
            if any(a.get("status") in ("queued", "structuring", "cleaning", "analysing") for a in draft["assets"]):
                return {"error": "Data is already being prepared; wait for it."}
            self.say(draft_id, "I'll simulate a dataset from this conversation. It will be labelled synthetic everywhere.")
            try:
                rows = max(100, min(int(args.get("rows") or 5000), 200_000))
            except (TypeError, ValueError):
                rows = 5000
            self.on_request(draft_id, "simulate", {"prompt": (draft["problem"] + " " + str(args.get("description", ""))[:2000]).strip(), "rows": rows})
            return {"simulating": True, "rows": rows}
        return {"error": f"unknown tool {name}"}


def title_for(draft: dict[str, Any], key: str) -> str:
    name = (packs.by_key(key) or {}).get("name", key)
    target = (draft.get("understanding") or {}).get("target")
    return f"{target} · {name}" if target else f"Solution workflow · {name}"
