"""The Home agent: understands the problem in a few questions and keeps the solution workflow current.

It first reads what kind of help the person needs (``intent.py``, package H2): a prediction, an analysis, data
preparation, a report, a business question or advice. Only the prediction track runs what follows; on the others Home
says plainly what DCLab can do for them now. The person can change the kind of help on the card at any time.

It is not a general chatbot. On the prediction track its job is narrow: find out what should be predicted, at which moment,
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

from dclab_rnd import kinds, prompts
from dclab_rnd.agents.registry import Registry, Tool
from dclab_rnd.agents.runtime import Policy, Step, run as run_policy
from dclab_rnd.agents.traces import Tracer

from . import brief as briefs
from . import intent as intents
from . import lang
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
FIELDS = ("target", "prediction_moment", "action", "costs", "data_plan", "entity", "labels", "budget")  # the last three: a pack's (15.6)
class _Words:
    """A word check in English and Persian (lang.BUILD, lang.SIMULATE), read on the normalised text (package H3)."""

    def __init__(self, pattern):
        self.pattern = pattern

    def search(self, text):
        return lang.has(self.pattern, text or "")


BUILD_WORDS = _Words(lang.BUILD)
SIMULATE_WORDS = _Words(lang.SIMULATE)

POLICY = prompts.text("home_agent")  # dclab_rnd/prompts/home_agent.md (A4.3: prompts are versioned files)

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
    kind = kinds.for_pack(pack)
    if kind is not None and kind.moments:  # the pack's kind words them (16.1)
        return list(kind.moments)
    return ["Right before we act (e.g. before the call)", "At a fixed snapshot each month", "As soon as the event happens"]


class HomeAgent:
    def __init__(self, store: DraftStore, client=None, on_request: Callable[[str, str, dict[str, Any]], None] | None = None,
                 registry: Registry | None = None, traces: Any = None, intent_client=None):
        self.store, self.client, self.traces, self.intent_client = store, client, traces, intent_client
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

    def ask(self, draft_id: str, field: str, question: str, options: list[str] | None = None, why: str = "",
            track: str | None = None) -> dict[str, Any] | None:
        draft = self.store.get(draft_id)
        asked = [q for q in draft["questions"] if not q.get("superseded") and q.get("field") != "intent" and not q.get("track")]  # the kind of help and a track's brief are not ML questions
        if len(asked) >= MAX_QUESTIONS or any(q.get("field") == field and not q.get("answered") for q in draft["questions"]):
            return None
        q = {"id": f"q{len(draft['questions']) + 1}", "field": field, "text": question, "options": (options or [])[:4], "why": why, "answered": None}
        if track:
            q["track"] = track  # a field of that track's brief (H4), answered by code

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
        for field, quote in self.read_brief(draft_id).items():  # package 17.2: dclab_core reads what the rules left open
            if field not in filled and not u.get(field):
                self.note(draft_id, field, quote, "stated", "problem", quote)
                filled.append(field)
        return filled

    def read_brief(self, draft_id: str) -> dict[str, str]:
        """The problem as dclab_core understands it (the library the main product adopts), kept on the draft as
        ``draft["brief"]``. Returns only what Home may take from it, in the user's own words (``plans.quoted``): a target
        the sentence binds to a column, and the action. The prediction moment stays with Home's own rules (``extract``),
        which are the ones tested for leakage; a cost or a review capacity is not taken as the action."""
        from dclab_core.understand import understand

        draft = self.store.get(draft_id)
        columns = [c.get("name") for c in ((draft.get("analysis") or {}).get("profile") or {}).get("columns") or [] if c.get("name")]
        from .. import roles

        # 18.9: the problem analyst fills only the fields the rules left open, when its gate passed and a model serves it
        # only when this agent runs with a model: replan and the pipeline call it with none, and must stay free
        advisor = roles.advisor("problem_analyst", draft_id=draft_id, cap_eur=roles.CALL_CAP_EUR) if self.client is not None else None
        brief = understand(draft.get("problem") or "", {"table": columns} if columns else None, advisor=advisor)
        self.store.update(draft_id, lambda d: d.update(brief=brief.model_dump(mode="json")))
        from .. import decisions

        decisions.record_brief(self.store, brief, draft_id=draft_id, columns=columns)  # A6.5: the contract fields, as samples
        texts = plans.user_text(draft)
        found: dict[str, str] = {}
        if brief.target_column.status == "stated" and plans.quoted(brief.target_column.quote, texts):
            found["target"] = brief.target_column.quote
        if brief.action.status == "stated" and plans.quoted(brief.action.quote, texts):
            found["action"] = brief.action.quote
        return found

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
        draft = self.store.get(draft_id)
        client, self.client = self.client, None  # an edited sentence is read again without a model, the track included
        try:
            if (draft.get("intent") or {}).get("source") not in ("user", None):  # the person's choice stands; a reading is redone
                reading = intents.read(draft["problem"])
                was = self.track(draft)
                if reading["status"] != "asked":
                    self.move_to(draft_id, reading, was)
                    if reading["track"] != was:
                        return  # the new track was opened (a prediction is read, planned and asked from the start)
            if self.track(self.store.get(draft_id)) != "prediction":
                if self.track(self.store.get(draft_id)) in briefs.TRACK_FIELDS:
                    self.refresh_brief(draft_id)
                return
            self.read_problem(draft_id)
        finally:
            self.client = client
        self.update_plan(draft_id)
        self.next_turn(draft_id)

    def refresh_workflow(self, draft_id: str, proposed: dict[str, Any] | None = None) -> dict[str, Any]:
        draft = self.store.get(draft_id)
        if self.track(draft) != "prediction":  # no ML workflow off the prediction track (H2): nothing drawn, nothing written
            return {"source": None, "version": (draft.get("workflow") or {}).get("version") or 0}
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

    # ------------------------------------------------------------------ the kind of help (H2)
    @staticmethod
    def track(draft: dict[str, Any]) -> str:
        """The draft's track: a draft read before H2 is a prediction; one whose kind of help is still being asked is
        "undecided", which runs none of the prediction path."""
        value = draft.get("intent")
        if not value:
            return "prediction"
        return "undecided" if value.get("status") == "asked" else value.get("track") or "prediction"

    def set_intent(self, draft_id: str, value: dict[str, Any], *, sample: bool = True) -> dict[str, Any]:
        """Keep the intent on the draft and show it on the card. A reading of the person's words (the rules or the model,
        always with its quote) is a decision sample; the default and the person's own choice are not."""
        if value.get("source") != "user":
            value = {**value, "rule_track": value.get("rule_track", value.get("track"))}
        draft = self.store.update(draft_id, lambda d: d.update(intent=value))
        self.store.emit(draft_id, "status", {"intent": value})
        if sample and value.get("source") in ("rules", "model") and value.get("quote"):
            from .. import decisions

            decisions.record_intent(self.store, value, draft_id=draft_id, has_data=bool(draft.get("assets")),
                                    words=len(str(draft.get("problem") or "").split()),
                                    actor="model" if value["source"] == "model" else "rule")
        return value

    def open_track(self, draft_id: str) -> None:
        """What Home says and does once the kind of help is known: the prediction path as it always was, or, on another
        track, a plain statement of what DCLab can do for it now (no pack, no ML workflow, no ML questions)."""
        draft = self.store.get(draft_id)
        track = self.track(draft)
        if track == "prediction":
            self.open_prediction(draft_id)
            return
        if track not in intents.NEXT:
            return
        has_data = bool(draft.get("assets"))
        offer = [] if track == "advice" or has_data else ["upload", "connect"]
        self.say(draft_id, f"I read this as {intents.LABELS[track].lower()}: “{draft['problem']}”. {intents.NEXT[track]}", kind="text",
                 actions=offer, track=track)
        self.refresh_brief(draft_id)
        self.ask_brief(draft_id)

    # ------------------------------------------------------------------ the other tracks' briefs (H4)
    def refresh_brief(self, draft_id: str) -> dict[str, Any]:
        """Read the brief of the draft's tracks again (the sentence, the answers, the defaults, the table's columns),
        keep it on the draft and show it. A field that names a column the table does not have is not taken."""
        draft = self.store.get(draft_id)
        found = briefs.refresh(draft)
        told = {tuple(m) for m in draft.get("brief_missing_told") or []}
        # a dropped column on a field with a default is said once (a field with no default says it in its question)
        tell = [m for m in found["missing"] if tuple(m) not in told and (briefs.field(self.track(draft), m[0]) or briefs.Field("", "", "")).default is not None]
        self.store.update(draft_id, lambda d: d.update(track_brief=found["briefs"], brief_missing=[list(m) for m in found["missing"]],
                                                       brief_missing_told=[list(m) for m in told | {tuple(m) for m in tell}]))
        self.store.emit(draft_id, "status", {"track_brief": found["briefs"]})
        for field_name, column in tell:
            entry = ((found["briefs"].get(self.track(draft)) or {}).get(field_name) or {})
            self.say(draft_id, f"I can't find a column named “{column}” in the table, so for the {field_name.replace('_', ' ')} I'll use "
                               f"{entry.get('value') or 'the default'} for now; tell me which column you meant.")
        return found

    def ask_brief(self, draft_id: str) -> None:
        """The one question the track's brief still needs, if any: a field with no safe default, within its budget."""
        draft = self.store.get(draft_id)
        track = self.track(draft)
        if track not in briefs.TRACK_FIELDS or self.pending(draft):
            return
        asked = sum(1 for q in draft["questions"] if q.get("track") == track and not q.get("superseded"))
        nxt = briefs.next_field(draft, track, asked)
        if nxt is None:
            return
        f, options = nxt
        column = next((m[1] for m in draft.get("brief_missing") or [] if isinstance(m, (list, tuple)) and m[0] == f.name), None)
        question = f"I can't find a column named “{column}” in the table. {f.question}" if column else f.question
        self.ask(draft_id, f.name, question, options, f.why, track=track)

    def choose_intent(self, draft_id: str, track: str) -> dict[str, Any]:
        """The person chose the kind of help (the card's Change control, the question's button, or a pack): it wins over
        any reading, is a label on this draft's reading, and moves Home to that track."""
        with self.store.turn(draft_id):
            draft = self.store.get(draft_id)
            before = draft.get("intent") or {}
            was = self.track(draft)
            value = intents.chosen(track, before)
            from .. import decisions

            decisions.label_intent(self.store, draft_id, track)
            self.move_to(draft_id, value, was, sample=False)
            return value

    @turn
    def pack_chosen(self, draft_id: str) -> None:
        """The person chose a pack: that is a prediction. Checked under the turn, so a reading still being made by start()
        (which holds the turn) is settled first and then changed, never missed."""
        if self.track(self.store.get(draft_id)) != "prediction":
            self.choose_intent(draft_id, "prediction")

    def move_to(self, draft_id: str, value: dict[str, Any], was: str, *, sample: bool = True) -> None:
        """Keep a new intent and, when the track changed, close what belonged to the old one (an open intent question,
        the ML questions and workflow when leaving a prediction) and open the new track."""
        track = value["track"]
        self.set_intent(draft_id, value, sample=sample)

        def settle(d):
            for q in d["questions"]:
                if not q.get("answered") and (q.get("field") == "intent" or (track != was and track != "prediction")
                                              or (q.get("track") and q.get("track") != track)):
                    q.update(answered="(the type of help changed)", superseded=True)
            if track != "prediction":
                d["workflow"] = None
        self.store.update(draft_id, settle)
        if track != was:
            self.open_track(draft_id)

    def set_pack(self, draft_id: str, key: str, source: str, why: str) -> None:
        def put(d):
            d["pack"] = {"key": key, "source": source, "why": why, "kind": packs.kind_for(key)}
        self.store.update(draft_id, put)
        self.store.emit(draft_id, "status", {"pack": {"key": key, "source": source, "why": why, "kind": packs.kind_for(key)}})

    # ------------------------------------------------------------------ entry points
    @turn
    def start(self, draft_id: str) -> None:
        """A new draft with a problem sentence: read the kind of help first (a person who chose a domain pack asked for a
        prediction), then open its track, or ask when the reading is a real tie."""
        draft = self.store.get(draft_id)
        if (draft.get("intent") or {}).get("source") == "user":  # chosen before Home read it (the wizard, an API client)
            reading = draft["intent"]
        elif (draft.get("pack") or {}).get("source") == "user":
            reading = self.set_intent(draft_id, intents.chosen("prediction"), sample=False)
        else:
            reading = self.set_intent(draft_id, intents.read(draft["problem"], self.intent_client, plans.user_text(draft)))
        if reading["status"] == "asked":  # an either-or question in the person's words, never a menu of DCLab's tracks
            ask = intents.contrast(reading)
            self.ask(draft_id, "intent", ask["question"], [text for _, text in ask["options"]] + ["Something else"], ask["why"])
            return
        self.open_track(draft_id)

    def open_prediction(self, draft_id: str) -> None:
        """The prediction track, as Home always ran it: restate the problem, pick the pack, draw the workflow, ask the first question."""
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
        prediction = self.track(self.store.get(draft_id)) == "prediction"
        if prediction:
            self.read_brief(draft_id)  # the brief again, now with the table's column names (entities, time, a bound target)
        draft = self.store.get(draft_id)
        analysis = draft.get("analysis") or {}
        s = analysis.get("summary") or {}
        steps = len(draft.get("cleaning_log") or [])
        label = "synthetic " if asset.get("synthetic") else ""
        text = (f"Your {label}data is ready: {s.get('rows', 0):,} rows and {s.get('columns', 0)} columns"
                + (f", after {steps} cleaning step{'s' if steps != 1 else ''}" if steps else "") + ". Here is a first look.")
        marked = analysis.get("labels") or {}
        if asset.get("kind") == "labels" and marked:  # 15.6: the second file is the table's labels
            text = (f"{asset.get('name', 'This file')} is a label file for your table: joined on {'+'.join(marked['key'])}, its "
                    f"{marked['label_rows']} rows mark {marked['rows_marked']} of {marked['rows']} rows ({marked['rows_marked'] / max(marked['rows'], 1):.2%})"
                    + (f"; {marked['not_found']} are not in the table" if marked.get("not_found") else "")
                    + (f"; refused keys: {', '.join('+'.join(r['key']) for r in marked.get('refused', [])[:2])}" if marked.get("refused") else "")
                    + ". The marked rows are the cases the model learns from.")
        self.say(draft_id, text, kind="analysis", asset=asset.get("id"))
        if not prediction:  # another track: the description, then the brief against the table's columns; no pack or workflow
            if self.track(self.store.get(draft_id)) in briefs.TRACK_FIELDS:
                self.refresh_brief(draft_id)
                self.ask_brief(draft_id)
            return
        if (draft.get("pack") or {}).get("source") != "user":
            det = packs.detect(draft["problem"], analysis)
            if det["key"] != (draft.get("pack") or {}).get("key") and det["source"] in ("problem", "data"):
                self.set_pack(draft_id, det["key"], det["source"], det["why"])
        self.refresh_workflow(draft_id)
        draft = self.store.get(draft_id)
        pending = self.pending(draft)
        candidates = ((draft.get("analysis") or {}).get("profile") or {}).get("target_candidates") or []
        if pending and pending["field"] == "target" and plans.label_asset(draft) is not None:
            # 15.6: the label file says what the cases are; the outcome question is withdrawn, not answered for the user
            self.store.update(draft_id, lambda d: [q.update(answered="(the label file defines the cases)", superseded=True) for q in d["questions"] if q["id"] == pending["id"]])
            pending = None
        elif pending and pending["field"] == "target" and not pending.get("options") and candidates:
            # The outcome was asked before the data arrived; ask again with the columns to choose from.
            self.store.update(draft_id, lambda d: [q.update(answered="(asked again with the columns)", superseded=True) for q in d["questions"] if q["id"] == pending["id"]])
            pending = None
        if candidates and not (draft.get("understanding") or {}).get("target"):
            # an outcome answered "not sure" before the data: the table's columns are a new chance to name it (H4)
            def again(d):
                for q in d["questions"]:
                    if q["field"] == "target" and not q.get("options") and q.get("answered") and lang.not_sure(q["answered"]):
                        q.update(superseded=True)
            self.store.update(draft_id, again)
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
        open_question = self.pending(self.store.get(draft_id))
        if open_question is not None and (open_question["field"] == "intent" or open_question.get("track")
                                          or lang.not_sure(text)):  # code reads these answers; a "not sure" never reaches a field
            self.take_answer(draft_id, open_question, text)
            return
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
        if q["field"] == "intent":
            before = draft.get("intent") or {}
            track = next((t for t, text in intents.contrast(before)["options"] if value.strip().casefold() == text.casefold()), None)
            track = track or intents.from_answer(value, {t: intents.LABELS[t] for t in intents.TRACKS})
            self.store.update(draft_id, lambda d: [x.update(answered=value) for x in d["questions"] if x["id"] == q["id"]])
            if track is None:  # "Something else", "I'm not sure": not a choice; start with the reading, which the card can change
                first = before.get("track") or "prediction"
                self.say(draft_id, f"I'll start with {intents.LABELS[first].lower()}. If that is not it, change the type of help above.")
                self.move_to(draft_id, {**before, "status": "inferred"}, "undecided", sample=False)
                return
            from .. import decisions

            decisions.label_intent(self.store, draft_id, track)
            self.move_to(draft_id, intents.chosen(track, before), "undecided", sample=False)
            return
        if q.get("track"):  # a field of a track's brief: code records it (a "not sure" takes the default)
            other = intents.read(value)
            if other["status"] == "stated" and other["track"] != q["track"] and other["track"] != "prediction" or \
                    (other["status"] == "stated" and other["track"] == "prediction" and intents.STRONG <= other["scores"]["prediction"]):
                # "Actually I'd rather predict…": not this field's value but another kind of help, in their words
                self.store.update(draft_id, lambda d: [x.update(answered="(the type of help changed)", superseded=True) for x in d["questions"] if x["id"] == q["id"]])
                self.move_to(draft_id, other, q["track"])
                return
            self.store.update(draft_id, lambda d: [x.update(answered=value) for x in d["questions"] if x["id"] == q["id"]])
            found = self.refresh_brief(draft_id)
            entry = (found["briefs"].get(q["track"]) or {}).get(q["field"]) or {}
            f = briefs.field(q["track"], q["field"])
            label = q["field"].replace("_", " ")
            if lang.not_sure(value):
                self.say(draft_id, f"No problem: I'll use {entry['value']} for the {label}; you can change it any time." if entry.get("value")
                         else f"No problem: the {label} stays open for now.")
            elif entry.get("source") == "answer":
                self.say(draft_id, f"Got it: {label}, “{entry['value']}”.")
            elif f is not None and any(m[0] == q["field"] for m in found["missing"]):
                column = next(m[1] for m in found["missing"] if m[0] == q["field"])
                self.say(draft_id, f"I can't find a column named “{column}” in the table, so the {label} stays open.")
            self.ask_brief(draft_id)
            return
        unsure = lang.not_sure(value) or (q["field"] == "target" and value.strip() == "Something else")
        if unsure:
            pass  # not an answer: what Home understood stays, the field stays as it was, and it is not asked again
        elif q["field"] in plans.ALL_FIELDS:
            self.note(draft_id, q["field"], value, "stated", "answer", value)
        else:
            self.record(draft_id, q["field"], value)

        def answered(d):
            for item in d["questions"]:
                if item["id"] == q["id"]:
                    item["answered"] = value
        self.store.update(draft_id, answered)
        u = self.store.get(draft_id).get("understanding") or {}
        for field, quote in ({} if unsure else plans.extract(value)).items():  # "…within 30 days, scored every Monday" also gives the moment
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
            if self.track(draft) != "prediction":
                self.say(draft_id, "Building a solution is for predictions. If you need a model, change the type of help above.")
                return
            self.summarize(draft_id)
            return
        self.record(draft_id, "notes", ((draft.get("understanding") or {}).get("notes", "") + " " + text).strip())
        self.say(draft_id, "Noted. I added it to the solution notes." if self.track(draft) == "prediction" else "Noted.")
        self.next_turn(draft_id)

    def next_turn(self, draft_id: str, opening: bool = False) -> None:
        """Ask the next missing thing, or summarise when everything needed is known."""
        draft = self.store.get(draft_id)
        if self.pending(draft) or self.track(draft) != "prediction":
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
        if self.track(draft) != "prediction":
            return None  # the outcome, moment and action belong to a prediction
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
        has_labels = plans.label_asset(draft) is not None  # the label file defines the target: no outcome column to ask for
        profile_columns = ((draft.get("analysis") or {}).get("profile") or {}).get("columns") or []
        for field in plans.fields_for(draft):
            if field == "target" and plan[field]["status"] != "unknown" and candidates and columns and u.get("target") not in columns \
                    and not column_asked and not has_labels:
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
                    return ("target", packs.opening_question(key), [], "Everything else follows from the outcome.")
                continue  # a table without an outcome-like column: ask the rest first
            if field == "prediction_moment":
                return ("prediction_moment", "When is the prediction made, and what is already known at that moment?", moment_options(key),
                        plans.WHY["prediction_moment"])
            if field == "action":
                return ("action", "What happens with each prediction, and what does a wrong one cost, roughly?",
                        ["We contact or act on the top cases", "We block or review cases", "We plan capacity or stock", "It informs a person only"],
                        plans.WHY["action"])
            if field == "entity":
                entities = [c["name"] for c in profile_columns if c.get("name_id_like") and not c.get("id_like")][:3]
                return ("entity", "Who could commit it: which person or thing do the cases belong to (a biker, a customer, a card)?",
                        entities + ["No single entity"], plans.WHY["entity"])
            if field == "labels":
                return ("labels", "Are the labelled cases every case there is, or a sample of the ones you found?",
                        ["A sample of the cases we found", "Every case is labelled"], plans.WHY["labels"])
            if field == "budget":
                return ("budget", "How many alerts can the team review, for example a day?",
                        ["About 50 a day", "About 200 a day", "About 1% of rows", "No fixed budget"], plans.WHY["budget"])
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
        if self.track(draft) != "prediction":
            return
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
            "intent": {k: (draft.get("intent") or {}).get(k) for k in ("track", "tracks", "status", "source")},
            "next_step": intents.NEXT.get(self.track(draft)),
            "track_brief": {k: {f: (e or {}).get("value") for f, e in v.items()} for k, v in (draft.get("track_brief") or {}).items()},
            "plan": {**{f: ((draft.get("plan") or {}).get(f) or {}).get("status", "unknown") for f in plans.fields_for(draft)}, "next": (draft.get("plan") or {}).get("next")},
            "questions_asked": [{k: q.get(k) for k in ("field", "text", "answered")} for q in draft["questions"]],
            "questions_left": MAX_QUESTIONS - len([q for q in draft["questions"] if q.get("field") != "intent" and not q.get("track")]),
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
        elif out.stopped == "tools_as_text" and model.live:
            self.store.emit(draft_id, "token", {"id": model.live, "drop": True})  # tool calls written as text are not a reply
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
    """The draft's state for a trace row: which of the first five fields are known (x) or open (-), in FIELDS order
    (five, as the stored trajectories have them; a pack's own fields are in the draft's plan)."""
    u = draft.get("understanding") or {}
    return "".join("x" if u.get(f) else "-" for f in FIELDS[:5])


class _Model:
    """The gateway client as the runtime sees it for one Home turn: a streamed reply reaches the page as it is
    written, and the id of the last reply's live bubble is kept so the final answer replaces it."""

    def __init__(self, agent: HomeAgent, draft_id: str):
        self.agent, self.draft_id, self.live = agent, draft_id, None

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if self.live:  # the reply streamed before (tool calls written as text, then a nudge) was not an answer: it goes
            self.agent.store.emit(self.draft_id, "token", {"id": self.live, "drop": True})
            self.live = None
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
        if field in plans.ALL_FIELDS:
            agent.note(draft_id, field, value, "stated", "answer", value)
        else:
            agent.record(draft_id, field, value)
        agent.store.update(draft_id, lambda d: [q.update(answered=value) for q in d["questions"] if q["id"] == pending["id"]])
    elif field in plans.ALL_FIELDS and args.get("quote"):  # the user said it elsewhere; the guard checked the quote
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


def set_intent(turn: Turn, /, **args: Any) -> dict[str, Any]:
    """The model's reading of the kind of help, backed by the person's words (the guard checked the quote)."""
    agent, draft_id = turn.agent, turn.draft_id
    draft = agent.store.get(draft_id)
    before = draft.get("intent") or {}
    track = args["track"]
    if track == agent.track(draft):
        return {"intent": track, "unchanged": True}  # nothing new: no sample, no move
    value = {**{k: before.get(k) for k in ("scores", "signals")}, "rules_version": intents.RULES_VERSION, "track": track,
             "tracks": [track] + [t for t in before.get("tracks") or [] if t != track], "status": "inferred", "source": "model",
             "quote": str(args["quote"]), "confidence": None, "alternatives": [t for t in before.get("tracks") or [] if t != track],
             "rule_track": before.get("rule_track", before.get("track"))}
    agent.move_to(draft_id, value, agent.track(draft))
    return {"intent": track}


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
               "request_data": "request_data", "simulate_data": "simulate", "set_intent": "set_intent"}
ML_MOVES = ("set_pack", "propose_workflow")  # the prediction track's moves: refused on the others (package H2)
PREPARING = ("queued", "structuring", "cleaning", "analysing")


REFUSED = {"record": "The answer was not backed by the user's own words", "ask": "Not the question the plan asks now",
           "set_pack": "Not a known pack, or the user chose the pack", "simulate": "Data was already being prepared",
           "propose_workflow": "The workflow did not pass the workflow checks",
           "set_intent": "Not backed by the user's words, or the user chose the type of help"}


class DraftGuard:
    """The Home agent's writes change a draft, not a project, so they have their own checks (package A2.2):
    the pack must be on the list and the user's choice stands; a workflow must pass ``workflow.validate``; one
    simulation (or any data preparation) at a time."""

    def __call__(self, tool: Tool, turn: Turn, arguments: dict[str, Any], proceed: Callable[[], Any]) -> Any:
        refusal = self.refusal(tool, turn, arguments)
        if refusal is None:
            return proceed()
        from .. import audit  # a refused move is audited like a project's (package 10.4), with a fixed reason: nothing the model wrote

        audit.record_for(turn.agent.store, "move", "agent", "Home agent", draft_id=turn.draft_id, move=tool.move, status="blocked",
                         message=REFUSED.get(tool.move, "The draft's checks refused this move"))
        return refusal

    def refusal(self, tool: Tool, turn: Turn, arguments: dict[str, Any]) -> dict[str, Any] | None:
        agent, draft_id = turn.agent, turn.draft_id
        if tool.move in ML_MOVES and agent.track(agent.store.get(draft_id)) != "prediction":
            return {"error": "This request is not a prediction, so there is no pack or ML workflow; if the user wants a model, "
                             "set the type of help with set_intent (their words) first."}
        if tool.move == "ask" and arguments.get("field") in FIELDS and agent.track(agent.store.get(draft_id)) != "prediction":
            return {"error": "This request is not a prediction: ask without a field (the answer goes to the notes)."}
        if tool.move == "set_intent":
            draft = agent.store.get(draft_id)
            if arguments.get("track") not in intents.TRACKS:
                return {"error": f"Unknown type of help. Types: {', '.join(intents.TRACKS)}."}
            if (draft.get("intent") or {}).get("source") == "user":
                return {"error": "The user chose the type of help; ask before changing it."}
            if not arguments.get("quote") or not plans.quoted(str(arguments["quote"]), plans.user_text(draft)):
                return {"error": "Give quote: whole words copied exactly from the user's problem or messages that say this."}
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
        return None


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
    add("set_intent", "Change the type of help the user needs, from their own words: prediction (a model), analysis (numbers "
        "about what happened), preparation (cleaning or combining data), reporting (a report or dashboard), business (a business "
        "question) or advice (an explanation). Give quote: their exact words that say it.",
        {"track": {**S, "enum": list(intents.TRACKS)}, "quote": S, "why": S}, ["track", "quote"], "write")
    add("request_data", "Offer the user ways to bring data.", {"text": S}, [], "write")  # no text: a standard offer
    add("get_profile", "Column summaries of the prepared table: name, kind, missing rate, unique count, and which columns look like outcomes, timestamps, identifiers or text. Aggregates only, never values. Name columns, or page through a wide table with offset.",
        {"columns": {"type": "array", "items": S}, "offset": {"type": "integer"}}, [], "read")
    add("get_analysis", "The descriptive analysis of the prepared table: size, column kinds, missing share, duplicate rows, findings worth a question, and the strongest feature-to-feature correlations. Says nothing about the outcome.",
        {}, [], "read")
    add("simulate_data", "Generate a synthetic table. Only after the user asked for simulated data.",
        {"description": {**S, "description": "The table to simulate: rows, columns, the outcome and its rate."}, "rows": {"type": "integer"}},
        ["description"], "write")
