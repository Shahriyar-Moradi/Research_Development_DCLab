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

import contextvars
import functools
import json
import re
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dclab_core.card import label_said
from dclab_rnd import kinds, prompts
from dclab_rnd.agents import harness
from dclab_rnd.agents.registry import Registry, Tool
from dclab_rnd.agents.runtime import Policy, Step, run as run_policy
from dclab_rnd.agents.traces import Tracer

from . import brief as briefs
from . import column_check as checks
from . import intent as intents
from . import lang
from . import memory as memories
from . import pack as packs
from . import plan as plans
from . import workflow as wflow
from .store import DraftStore, now

SHAPES = ("classification", "regression")  # an output's shape: the rules and the data may still name the family (20.10 v4)
NO_OUTCOME_COLUMN = ("ranking", "segmentation", "anomaly", "entity_tables", "streams")  # families that read no outcome column
def clause_of(text: str, word: str) -> str:
    """The clause of ``text`` that holds ``word``: from the delimiter before it to the one after."""
    at = text.find(word)
    if at < 0:
        return ""
    start = max(text.rfind(c, 0, at) for c in ",;.:") + 1
    ends = [i for i in (text.find(c, at) for c in ",;.:") if i >= 0]
    return text[start:min(ends) if ends else len(text)]


LATER = re.compile(r"\b(after|once|only|later|not known|unknown|leak\w*)\b", re.I)  # a column mentioned as known later
SEGMENTS_FIELD = "segments_route"
SEGMENTS_ROUTES = ("Describe groups I already have", "Find new groups in the data")
SEGMENTS_SAID = ("There are two ways to look at groups. If you already have them in a column (a plan, a region, a customer type), "
                 "upload the table and I'll show the numbers per group. If you want DCLab to find new groups of customers who "
                 "behave alike, that is a segmentation project: no outcome is needed, and DCLab checks how stable the groups are.")
COSTS_QUESTION = ("Roughly, what does acting on one case cost, and what is a case worth or a miss? For example: a call costs 5; "
                  "a lost customer costs 300.")


def _amounts_said(u: dict[str, Any]) -> bool:
    """34.4: whether the words on what is done and what it costs already give a decision's amounts (dclab_core.decide.read_spec)."""
    from dclab_core.decide import read_spec

    return any(read_spec(str(u.get(k) or "")) is not None for k in ("action", "costs"))  # each answer read on its own


MAX_QUESTIONS = 5  # 20.10: the conversation gate's five; at four a rare event's review budget was never asked
MODEL_REPLIES = 5  # model requests per user message
MODEL_STEPS = 40  # tool calls per user message (a reply may make several)
MODEL_SECONDS = 5 * 60
TOKEN_CHUNK = 48  # characters per streamed ``token`` event
PROFILE_PAGE = 40  # columns the model sees at a time, in its state and from get_profile
PROFILE_FIELDS = ("name", "kind", "missing_rate", "unique")  # what the model may know about a column: no values
FIELDS = ("target", "prediction_moment", "action", "costs", "data_plan", "entity", "labels", "budget")  # the last three: a pack's (15.6)
# the agent asking leave to build ("Shall I build it now?"), the one question a model may end its turn with while code asks the plan's next
LEAVE_TO_BUILD = re.compile(r"\b(shall i|should i|do you want me to|want me to|may i|can i|ready to|okay to|ok to)\b.{0,40}\b(build|start|go ahead|proceed)", re.I)
OPTIONAL = tuple(f for f in FIELDS if f not in plans.ALL_FIELDS)  # asked when useful, never blocking (30.13): no plan settles them
ANSWER_ONLY = ("entity",)  # recorded by a model only as the answer to its own question: who could commit it groups the folds
RULES_READ = ("unit",)  # what one prediction is for: the meet check blocks on it, so only the rules read it from the person's words
WORDS_KEPT = ("budget",)  # an answer the rules read a number from (a share of rows): kept in the person's words, never a model's paraphrase


def counted(draft: dict[str, Any]) -> list[dict[str, Any]]:
    """The questions that count against MAX_QUESTIONS: never a superseded one, never an optional field's (asked once at
    most: costs would otherwise take the place of a pack's last field)."""
    return [q for q in draft.get("questions") or [] if not q.get("superseded") and q.get("field") not in OPTIONAL
            and q.get("field") not in ("intent", checks.FIELD) and not q.get("brief")]  # H2: the kind of help is not one of the prediction's questions; H3.1: nor a column check  # what one prediction is for: the meet check blocks on it, so only the rules read it from the person's words
BUILD_WORDS = re.compile(r"\b(let'?s build|build (it|the solution)|go ahead|start the project|ready to build)\b", re.I)


class _Both:
    """H3: an English pattern beside its Persian one (lang.py), which is matched on the normalized text. Callers only ask
    whether it matched. ``whole``: the Persian pattern must be the whole message (an order to build: a sentence that only
    contains "شروع کنیم", "let's start", is not one; the review of H3)."""

    def __init__(self, english: "re.Pattern[str]", persian: "re.Pattern[str]", whole: bool = False) -> None:
        self.english, self.persian, self.whole = english, persian, whole

    def search(self, text: str) -> bool:
        norm = lang.normalize(text or "")
        return bool(self.english.search(text or "") or (self.persian.fullmatch(norm) if self.whole else self.persian.search(norm)))

    def match(self, text: str) -> bool:
        return bool(self.english.match(text or "") or self.persian.fullmatch(lang.normalize(text or "")))

    def fullmatch(self, text: str) -> bool:
        return bool(self.english.fullmatch(text or "") or self.persian.fullmatch(lang.normalize(text or "")))


BUILD_WORDS = _Both(BUILD_WORDS, lang.BUILD, whole=True)  # H3
# 25.13: a message that is an order to build now, read by its intent rather than an exact phrase (live, 2026-10-16: "build",
# "build now", "Yes, build now" and "ye s build now" were all missed): agreeing or polite words, then a build verb, then only
# words that name what to build or when. A sentence that describes the problem or asks ("we want to build a model that…",
# "should we build a model?", "start date is the signup day") is never taken for the order.
_ORDER_FILLERS = frozenset("""yes yeah yep yup ye ya y s ok okay k please pls sure so now then alright fine great good perfect cool
    right go ahead and let's lets let us just can could would you u kindly""".split())
_ORDER_VERBS = frozenset(("build", "train", "create", "make", "run", "start"))
_ORDER_OBJECTS = frozenset("""it the a an my this that model models solution project now please from chat conversation stages new
    building training right away then again pls for me one here""".split())


def build_order(text: str) -> bool:
    """Whether a message is, as a whole, an order to build (or train) now."""
    if "?" in text or "؟" in text:
        return False
    persian = lang.normalize(text)
    if lang.has_persian(text) and lang.BUILD.fullmatch(persian):  # H3: "بسازش", "باشه، برو جلو": the whole message
        return True
    words = re.findall(r"[a-z']+", text.lower())
    if not words or len(words) > 12:
        return False
    i = 0
    while i < len(words) and words[i] in _ORDER_FILLERS:
        i += 1
    if i >= len(words) or words[i] not in _ORDER_VERBS:
        return False
    rest = words[i + 1:]
    if words[i] in ("run", "start", "make", "create") and not rest:  # "start" alone answers something else
        return False
    return all(w in _ORDER_OBJECTS for w in rest)


ASSENT = re.compile(r"^\W*(yes|yeah|yep|yup|ye ?s|yse|ok(ay)?|sure|go( ahead| for it)?|do it|please do|proceed|start)(\W+(please|now|do it|go ahead))?\W*$", re.I)  # a bare yes
OFFER = re.compile(r"\b(build|train)\b[^.?!]*\?", re.I)  # DCLab asked whether to build ("Shall I build the model now?")


def asks_to_build(messages: list[dict[str, Any]]) -> bool:
    """The person's last message is an order to build, or a bare yes to DCLab's own offer to build just before it."""
    at = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), None)
    if at is None:
        return False
    said = messages[at].get("text") or ""
    if build_order(said):
        return True
    before = next((m for m in reversed(messages[:at]) if m.get("role") == "agent"), None) or {}
    offered = before.get("kind") == "summary" or bool(OFFER.search(before.get("text") or "")) or bool(BUILD_WORDS.search(before.get("text") or ""))
    return (bool(ASSENT.search(said)) or bool(lang.ASSENT.fullmatch(lang.normalize(said)))) and offered  # H3: a Persian yes


SIM_ROWS = 10_000  # the owner (2026-10-16): a simulated table is at most 10,000 rows
SIMULATE_WORDS = re.compile(r"\b(simulat\w*|synthetic|fake data|generate (some )?data)\b", re.I)
SIMULATE_WORDS = _Both(SIMULATE_WORDS, lang.SIMULATE)  # H3
CONFIRMED = re.compile(r"^\s*(yes|yeah|yep|correct|right|that'?s right|exactly|ok|okay|sure|confirmed?)\s*[.!]*\s*$", re.I)  # a bare yes
CONFIRMED = _Both(CONFIRMED, lang.CONFIRMED)  # H3

POLICY = prompts.text("home_agent")  # dclab_rnd/prompts/home_agent.md (A4.3: prompts are versioned files)

_SCRIPT: contextvars.ContextVar = contextvars.ContextVar("home_script", default=None)  # (draft id, harness run) of a scripted turn
_MODEL_TURN: contextvars.ContextVar = contextvars.ContextVar("home_model_turn", default=False)  # 36.3: a model turn is running (never one inside another)


_SHADOW: dict[str, Any] = {}


def _intent_shadow(words: str) -> dict[str, Any] | None:
    """P6's intent classifier answering in shadow (P8): its track and confidence, recorded beside the rules and the model and
    never used; None when the fitted model is not on this machine (it is not committed) or fails."""
    try:
        if "model" not in _SHADOW:
            import joblib

            from .. import intent_classifier as ic

            path = ic.MODELS / "request.intent.text_v2.joblib"
            _SHADOW["model"] = joblib.load(path) if path.exists() else None
        if _SHADOW["model"] is None or not words.strip():
            return None
        from .. import intent_classifier as ic

        track, confidence = ic.answer(_SHADOW["model"], [words])[0]
        return {"answerer": "classifier:intent_text_v2", "value": track, "confidence": confidence}
    except Exception:  # noqa: BLE001 - a shadow never stops Home
        return None


def _closed(history: list[dict[str, str]], request: str) -> tuple[list[dict[str, str]], str]:
    """The conversation as the model reads it, ending on the person's words, and what Home said after them.

    36.3, live (HOME-012): a first request handed to the model in the middle of Home's turn had only Home's own messages
    (the request is the draft's problem, not a message), so the request ended on the assistant's words and the API refused
    it (a 400 with thinking on). The person's request opens the conversation when none of their messages is in the window;
    Home's messages after the person's last one go to the state, so nothing is said twice. An empty list: nothing to answer."""
    if not any(m["role"] == "user" for m in history):
        if not request.strip():
            return [], ""
        history = [{"role": "user", "content": request.strip()}, *history]
    cut = max(i for i, m in enumerate(history) if m["role"] == "user") + 1
    return history[:cut], " ".join(m["content"] for m in history[cut:])


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
    from ..studio.data import proposable_target

    eligible = [c for c in profile.get("columns") or [] if proposable_target(c)]  # never a key, a constant or a row number
    named = [c for c in eligible if _stems(c.get("name", "")) & words]
    named.sort(key=lambda c: (-int(bool(c.get("target_name_like"))), -int(bool(c.get("low_cardinality")))))
    first = [str(c["name"]) for c in named]
    allowed = {str(c.get("name")) for c in eligible}
    described = {str(c.get("name")) for c in profile.get("columns") or []}  # a stored profile without its columns keeps its list
    return (first + [c for c in candidates if c not in first and (c in allowed or c not in described)])[:max(8, len(candidates))]


# S8 B3 (live, 2026-10-25): a person who does not know says so, and DCLab recorded "I don't know." as the answer (46
# prediction moments, 33 outcomes, 31 families of 589 written people; 4 projects built with it as their moment). An
# answer that is only that is never a value: the question is closed as declined, the field stays unknown, and only the
# moment, which every solution needs, is assumed, said as an assumption.
NOT_KNOWN = re.compile(r"\s*(i\s+(really\s+|honestly\s+)?(don'?t|do\s+not|dont)\s+know(\s+(yet|that|really|exactly|for\s+sure))?|((i'?m|i\s+am|im)\s+)?not\s+sure"
                       r"|no\s+idea|dunno|unknown|n/?a|skip)\s*[.!]*\s*", re.I)
ASSUMED_MOMENT = "At the time each row is recorded, using only what was known then (assumed: the person did not know; change it on the Solution page)"


def not_known(value: Any) -> bool:
    """Whether an answer says only that the person does not know."""
    return isinstance(value, str) and (bool(NOT_KNOWN.fullmatch(value)) or bool(lang.NOT_KNOWN.fullmatch(lang.normalize(value))))


def moment_options(pack: str | None) -> list[str]:
    if pack == "timeseries":
        return ["The day before", "A week ahead", "A month ahead"]
    kind = kinds.for_pack(pack)
    if kind is not None and kind.moments:  # the pack's kind words them (16.1)
        return list(kind.moments)
    return ["Right before we act (e.g. before the call)", "At a fixed snapshot each month", "As soon as the event happens"]


_TABLES: dict[tuple[str, int], Any] = {}
_TABLES_LOCK = threading.Lock()


def draft_table(store: Any, draft: dict[str, Any]) -> Any:
    """The draft's cleaned table (S9.3: the kind's fields are read from it), or None before it is ready; read once per
    file version, the last two kept."""
    asset = next((a for a in draft.get("assets") or [] if a.get("id") == draft.get("active_asset") and a.get("status") == "ready"), None)
    if asset is None:
        return None
    path = Path(store.data_dir(draft["id"])) / asset.get("clean_file", "clean.parquet")
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return None
    with _TABLES_LOCK:
        if key in _TABLES:
            return _TABLES[key]
    import pandas as pd

    try:
        frame = pd.read_parquet(path)
    except Exception:  # noqa: BLE001 - a table that cannot be read asks nothing of its kind
        return None
    with _TABLES_LOCK:  # two kept: a draft's table is read again on each turn, and memory is small
        while len(_TABLES) >= 2:
            _TABLES.pop(next(iter(_TABLES)))
        _TABLES[key] = frame
    return frame


class HomeAgent:
    def __init__(self, store: DraftStore, client=None, on_request: Callable[[str, str, dict[str, Any]], None] | None = None,
                 registry: Registry | None = None, traces: Any = None, reader: Callable[..., Any] | None = None,
                 second_opinion: bool = False, intent_client: Any = None, query_client: Any = None, business_client: Any = None):
        self.store, self.client, self.traces = store, client, traces
        self.intent_client = intent_client  # H2: the model reading the kind of help, only where the rules find no clear winner
        self.query_client = query_client  # H7: the model writing a query plan from column summaries, where the rules cannot
        self.business_client = business_client  # H11b: the model rewording a decision brief, kept only through cited.py
        self.second_opinion = second_opinion  # 28.6: the reader's family and the rules' judged before the card is concluded
        # 20.10 v4: the conversation reader (dclab_core.conversation.READER through a model), reading the problem sentence
        # and the card's answers; (field, text, columns) -> what the model read, each value quoting the person's words
        self.reader = reader
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

    def offer_synthetic(self, draft_id: str, prompt: str = "", rows: int = SIM_ROWS) -> dict[str, Any] | None:
        """Someone asked for simulated data: ask how (the owner, 2026-10-20), never start it unasked. One way is AI building the
        table from the conversation, the other is describing it in the chat and approving its design before it is created.
        The offer carries what the AI way would use; the page makes the choice a button (Home's Synthetic tab has both)."""
        last = (self.store.get(draft_id).get("messages") or [None])[-1]
        if last and last.get("kind") == "synthetic_way":
            return None  # asked already, and not answered: never twice in a row
        return self.say(draft_id, "How should I make the synthetic data? I can build the table with AI from this conversation, or you describe it "
                                  "here, see its design, and approve it before I create anything. It will be labelled synthetic everywhere.",
                        kind="synthetic_way", offer={"prompt": (prompt or "")[:2500], "rows": max(100, min(int(rows or SIM_ROWS), SIM_ROWS))},
                        actions=["simulate_ai", "simulate_describe"])

    def ask(self, draft_id: str, field: str, question: str, options: list[str] | None = None, why: str = "",
            source: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """``source="critic"`` (28.4): the understanding critic sends a settled field back to the person (the card it read
        is wrong or unsure there); it is asked past the cap and over a settled field, still never twice while open."""
        draft = self.store.get(draft_id)
        if field not in ("intent", SEGMENTS_FIELD) and self.track(draft) != "prediction":
            return None  # H2 (the review): off a prediction no question of the prediction path is asked, the critic's included
        asked = counted(draft)
        critic = source in ("critic", "second_opinion")  # 28.4, 28.6: a field the card settled is put to the person again
        over = len(asked) >= MAX_QUESTIONS and field not in (*self.CONTRADICTIONS.values(), checks.FIELD) and not critic  # 21.5: the words and the data disagreeing
        if over or any(q.get("field") == field and not q.get("answered") for q in draft["questions"]):  # is asked past the cap, once
            return None
        if self.settled(draft, field) and field not in self.CONTRADICTIONS.values() and not critic:
            return None  # the person said it already, or the table is here: asking again is what they found unintelligent
        if field in OPTIONAL and not critic and ((draft.get("understanding") or {}).get(field) or any(q.get("field") == field and q.get("answered") for q in draft["questions"])):
            return None  # live (HRN-022): an optional field no plan settles was asked four times in one walk; answered once, it is known
        q = {"id": f"q{len(draft['questions']) + 1}", "field": field, "text": question, "options": (options or [])[:7 if field == "intent" else 4], "why": why, "answered": None,
             **({"source": source} if source else {}), **(extra or {})}  # 28.6: the second opinion's verdict travels with its question
        replied = None if critic else self.answerable(draft)
        if replied is not None and (replied["field"] == field or replied.get("source") in ("critic", "second_opinion")):
            replied = None
        said = next((str(m.get("text") or "") for m in reversed(draft.get("messages") or []) if m.get("role") == "user"), "") if replied else ""
        if replied is not None:  # live (HRN-022): the question the person replied to stayed open behind the new one, and was put to them again
            self._replied(draft_id, replied, said)

        def add(d):
            d["questions"].append(q)
        self.store.update(draft_id, add)
        return self.say(draft_id, question, kind="question", question=q)

    def pending(self, draft: dict[str, Any]) -> dict[str, Any] | None:
        """The open question: not answered, and about something still unknown (live, 2026-10-16: a field the person had told
        in their own words, and the data once a table was ready, were asked again)."""
        for q in draft["questions"]:
            if q.get("answered") or (self.settled(draft, q.get("field")) and q.get("source") not in ("critic", "second_opinion")):  # 28.4: a critic's question re-opens its field
                continue
            return q
        return None

    @staticmethod
    def settled(draft: dict[str, Any], field: Any, value: Any = None) -> bool:
        """Whether a field is known well enough never to be asked again: the data once a table is on its way, the outcome
        as its column once a table with columns is there (in words before), any other field once recorded."""
        if field == "data_plan" and any(a.get("status") in ("ready", "queued", "structuring", "cleaning", "analysing") for a in draft.get("assets") or []):
            return True
        if field not in plans.ALL_FIELDS:
            return False
        known = value if value is not None else (draft.get("understanding") or {}).get(field)
        if not known:
            return False
        if field == "target":
            columns = {c.get("name") for c in ((draft.get("analysis") or {}).get("columns") or [])}
            return not columns or str(known) in columns or plans.label_asset(draft) is not None
        return True

    @classmethod
    def _close(cls, d: dict[str, Any], field: str, value: str) -> None:
        """A field recorded, however it came: the question about it is answered (never asked again)."""
        if cls.settled(d, field, value.strip()):
            for q in d.get("questions") or []:
                if q.get("field") == field and not q.get("answered"):
                    q["answered"] = value.strip()[:400]

    @staticmethod
    def _as_column(d: dict[str, Any], field: str, value: str) -> str:
        """The outcome as its column when the words name exactly one of the table's columns (live, 2026-10-17: a model
        recorded "The churned column (true = subscriber left…)", and the card waited for a column it had been told)."""
        value = value.strip()
        columns = [c.get("name") for c in ((d.get("analysis") or {}).get("columns") or []) if c.get("name")]
        if field != "target" or not columns or value in columns:
            return value
        def as_outcome(c: str) -> bool:  # 20.10's guard: named as the outcome, never one only mentioned or known later
            e = re.escape(str(c))
            q = "[`\"'\u201c\u201d\u2018\u2019]?"  # a column named in backticks or quotes (S9.4 live: 'The "urgent" column (0/1)')
            said = (f"`{c}`" in value or re.match(rf"\s*(the\s+)?((column|outcome|target|label|flag|field)\s+){{0,2}}{q}{e}{q}(?![\w])", value, re.I)
                    or re.search(rf"(?<![\w]){q}{e}{q}\s+(column|flag|field)\b", value, re.I))
            return bool(said) and not LATER.search(clause_of(value, str(c)))

        named = [c for c in columns if as_outcome(str(c))]
        return named[0] if len(named) == 1 else value

    @staticmethod
    def _not_known(d: dict[str, Any], field: str, value: str) -> None:
        """The person does not know: the question is closed as declined (never asked again), the field stays unknown;
        the prediction moment, which every solution needs, is assumed and said so."""
        for q in d.get("questions") or []:
            if q.get("field") == field and not q.get("answered"):
                q["answered"], q["declined"] = value.strip()[:400], True
        u = d.setdefault("understanding", {})
        u["not_known"] = sorted({*(u.get("not_known") or []), field})
        if field == "prediction_moment" and not u.get("prediction_moment"):
            u["prediction_moment"] = ASSUMED_MOMENT
            u["assumed"] = {**(u.get("assumed") or {}), "prediction_moment": ASSUMED_MOMENT}

    def record(self, draft_id: str, field: str, value: str) -> None:
        def put(d):
            if not_known(value):
                self._not_known(d, field, value)
                return
            d.setdefault("understanding", {})[field] = self._as_column(d, field, value)[:400]
            self._close(d, field, d["understanding"][field])
        self.store.update(draft_id, put)

    def note(self, draft_id: str, field: str, value: str, status: str, source: str, quote: str | None) -> None:
        """Record a field with its plan entry: how it is known and the user's words that say it (package A3.1)."""
        def put(d):
            if not_known(value):
                self._not_known(d, field, value)
                return
            d.setdefault("understanding", {})[field] = self._as_column(d, field, value)[:400]
            d["plan"] = {**plans.empty(), **(d.get("plan") or {}), field: plans.entry(status, source, quote)}
            self._close(d, field, d["understanding"][field])
        self.store.update(draft_id, put)

    def update_plan(self, draft_id: str) -> dict[str, Any]:
        """Recompute the plan (what is known, what to ask next and why) and show it on the page."""
        draft = self.store.update(draft_id, lambda d: d.update(plan=plans.refresh(d)))
        self.store.emit(draft_id, "plan", {"plan": draft["plan"], "understanding": draft.get("understanding") or {}})
        return draft["plan"]

    def model_read(self, field: str, text: str, columns: list[str] | None = None) -> dict[str, str]:
        """What the conversation reader read from one text that the rules could not, or nothing (no reader, a refusal,
        a failure: the rules' reading stands)."""
        if self.reader is None or not str(text or "").strip():
            return {}
        try:
            return dict(self.reader(field, str(text), list(columns or [])) or {})
        except Exception:  # noqa: BLE001 - a reader that fails leaves the rules' reading
            return {}

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
            from .. import cited

            said = set().union(*(cited.numbers(t, source=True) for t in plans.user_text(draft)), set())
            for field, item in kept.items():
                if not u.get(field):
                    # HOME-017: the model wrote "top 10%" for "the top decile", and the opening says the value back: a value
                    # with a number the person did not write is replaced by their own words (its quote)
                    # HOME-021: the model also wrote its own long sentences ("At the moment the customer puts the order in the
                    # cart" for "when customer put it in cart"); a value not in the person's words is said in their words too
                    # (only a value marked stated: an inferred one is the model's reading by name, shown as inferred)
                    reworded = item.get("status") == "stated" and not plans.quoted(str(item["value"]), plans.user_text(draft))
                    value = item["quote"] if reworded or cited.numbers(str(item["value"])) - said else item["value"]
                    self.note(draft_id, field, value, item["status"], "model", item["quote"])
                    filled.append(field)
        for field, quote in plans.extract(draft["problem"]).items():
            if field not in filled and not u.get(field):
                self.note(draft_id, field, quote, "stated", "problem", quote)
                filled.append(field)
        for field, quote in self.read_brief(draft_id).items():  # package 17.2: dclab_core reads what the rules left open
            if field not in filled and not u.get(field):
                self.note(draft_id, field, quote, "stated", "problem", quote)
                filled.append(field)
        read = self.model_read("problem", draft["problem"])  # 20.10 v4: the kind of result the sentence asks for
        if read.get("family") not in (None, *SHAPES) and not (self.store.get(draft_id).get("understanding") or {}).get("family"):
            self.record(draft_id, "family", read["family"])
            self.record(draft_id, "family_shape_only", "no")
            self.record(draft_id, "family_by", "reader")
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
        texts = plans.user_text(draft)
        found: dict[str, str] = {}
        if brief.target_column.status == "stated" and plans.quoted(brief.target_column.quote, texts):
            found["target"] = brief.target_column.quote
        if brief.action.status == "stated" and plans.quoted(brief.action.quote, texts):
            found["action"] = brief.action.quote
        return found

    def read_answers(self, draft_id: str) -> dict[str, Any]:
        """28.4, the problem analyst of the understanding team: the first sentence and every answer the person gave, read
        together with the table's columns (``dclab_core.understand``, as ``read_brief`` reads the sentence alone), for what
        is said in answer to another question. Only the action is taken from them ("..., to send each one to the review
        team", said with the review capacity), in the person's own words (``plans.quoted``), when it is still open; the
        outcome never is: by the analysis it is settled by the person or by the data (a file of known cases, the table's
        suggestion), and words that merely name a column are no outcome (the review of 28.4). No model is asked: the facts
        taken here are the person's stated words, and a model's are inferred. The record on the draft names the fields,
        never the words (they are in the understanding), and keeps every field the analyst ever filled."""
        from dclab_core.understand import understand

        draft = self.store.get(draft_id)
        texts = [t for t in plans.user_text(draft) if str(t or "").strip()]
        columns = [c.get("name") for c in ((draft.get("analysis") or {}).get("profile") or {}).get("columns") or [] if c.get("name")]
        brief = understand(". ".join(str(t).strip().rstrip(".") for t in texts), {"table": columns} if columns else None)
        filled: list[str] = []
        fact = brief.action
        if fact.status == "stated" and not (draft.get("understanding") or {}).get("action") and plans.quoted(fact.quote, texts):
            self.note(draft_id, "action", fact.quote, "stated", "analyst", fact.quote)
            filled.append("action")
        if filled:
            self.update_plan(draft_id)
        return self.analyst_read(draft_id, len(texts), filled)

    def analyst_read(self, draft_id: str, read: int, filled: list[str], error: str | None = None) -> dict[str, Any]:
        """The problem analyst's record on the draft: how many of the person's texts it has read, what it filled this time
        (``last``) and ever (``filled``), and an error when its reading failed (it still counts as read: the walk goes on)."""
        out: dict[str, Any] = {}

        def keep(d: dict[str, Any]) -> None:
            before = (d.get("coordinator") or {}).get("analyst") or {}
            out.update(filled=sorted(set(before.get("filled") or []) | set(filled)), last=list(filled), read=int(read), source="rules",
                       **({"error": error} if error else {}))
            d.setdefault("coordinator", {})["analyst"] = dict(out)
        self.store.update(draft_id, keep)
        return dict(out)

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
        client, self.client = self.client, None  # an edited sentence is read again without a model, the track included (H2)
        try:
            if (draft.get("intent") or {}).get("source") not in ("user", None):  # the person's choice stands; a reading is redone
                reading = intents.read(draft["problem"])
                was = self.track(draft)
                # only the rules' own reading moves the track: a default (nothing read) never replaces a model's reading (the
                # review of the H2 port: an edited sentence threw a model's "analysis" away and opened the ML path)
                if reading["status"] != "asked" and ((reading["source"] == "rules" and not reading.get("fallback"))
                                                     or (draft.get("intent") or {}).get("source") != "model"):
                    self.move_to(draft_id, reading, was)
                    if reading["track"] != was:
                        return  # the new track was opened (a prediction is read, planned and asked from the start)
            if self.track(self.store.get(draft_id)) != "prediction":
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
        from ..ledger import capture  # 19.7: the workflow drawn, the template's or a model's

        capture.workflow_drawn(draft_id, key, wf, draft.get("workflow"), bool(proposed), problems)
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
        always with its quote) is a decision in the ledger (request.intent), its id kept so the person's choice labels it;
        the default and the person's own choice are not readings."""
        if value.get("source") != "user":
            value = {**value, "rule_track": value.get("rule_track", value.get("track"))}
        if sample and value.get("source") in ("rules", "model") and (value.get("quote") or value.get("status") == "asked"):
            from ..ledger import capture

            draft = self.store.get(draft_id)
            words = str(draft.get("problem") or "")
            asked_model = value.get("source") == "model" or "model_failed" in value
            # P8: one routing record with every answerer: the rules, the model (with its exchange, so its thinking summary joins),
            # and P6's classifier in shadow (recorded, never used); the person's choice on the card labels it
            rid = capture.routing_read(
                draft_id, "request.intent", value.get("track"), "model_role" if value.get("source") == "model" else "rule",
                options=intents.TRACKS,
                state={"has_data": bool(draft.get("assets")), "words": len(words.split()), "signals": len(value.get("signals") or [])},
                rule={"answerer": f"rule:{intents.RULES_VERSION}", "value": value.get("rule_track")},
                model={"answerer": "home_intent", "value": value.get("model_track"), "confidence": value.get("confidence"),
                       "request": value.get("model_request"), "failed": value.get("model_failed")} if asked_model else None,
                own=_intent_shadow(words))
            if rid:
                value = {**value, "record": rid}
        self.store.update(draft_id, lambda d: d.update(intent=value))
        self.store.emit(draft_id, "status", {"intent": value})
        return value

    def route(self, draft_id: str) -> None:
        """H8: where the draft's kind of help leads (its exit; one already reached is kept) and, from the person's own words,
        another route they may mean, said once with its reason and a button; it never moves the draft by itself."""
        from . import exits
        from ..ledger import capture

        draft = self.store.get(draft_id)
        if (draft.get("intent") or {}).get("status") == "asked":
            return
        track = self.track(draft)
        door = exits.exit_for(track)
        idea = exits.suggest(str(draft.get("problem") or ""), track)
        changed = door is not None and (draft.get("exit") or {}).get("kind") != door["kind"]
        record = capture.request_exit(draft_id, track, door["kind"], (idea or {}).get("track")) if changed else None
        # P8's gap (2026-10-29): the suggestion is its own decision (a track, or none), labelled by what the person does with it
        hint = capture.suggestion_read(draft_id, track, (idea or {}).get("track"), (idea or {}).get("rule") or "") if changed else None

        def put(d):
            if door and (d.get("exit") or {}).get("kind") != door["kind"]:
                # a destination reached earlier stays reached when the person comes back to it (the review of H8)
                reached = (door["kind"] == "analysis" and d.get("answers")) or (door["kind"] == "ml_project" and d.get("project_id")) \
                    or (door["kind"] == "dataset" and (d.get("recipe") or {}).get("done")) \
                    or (door["kind"] == "decision_brief" and (d.get("business") or {}).get("result")) \
                    or (door["kind"] == "dashboard" and (d.get("dashboard") or {}).get("board"))
                d["exit"] = {**door, "at": now(), **({"record": record} if record else {}),
                             **({"status": "reached"} if reached else {})}
            d["route_suggestion"] = idea
            if hint:  # its record apart: route_suggestion is read as "a suggestion was offered"
                d["suggestion_record"] = {"record": hint, "on": track, "track": (idea or {}).get("track")}
        self.store.update(draft_id, put)
        offered = {(m.get("suggestion") or {}).get("track") for m in draft.get("messages") or [] if m.get("kind") == "route_suggestion"}
        if idea and idea["track"] not in offered:
            self.say(draft_id, f"Another way to read this: {intents.LABELS[idea['track']].lower()}. {idea['why']} "
                               f"(Your words: “{idea['quote']}”.)", kind="route_suggestion", suggestion=idea)

    def open_track(self, draft_id: str) -> None:
        """What Home says and does once the kind of help is known: the prediction path as it always was, or, on another
        track, a plain statement of what DCLab can do for it now (no pack, no ML workflow, no ML questions)."""
        self.route(draft_id)  # H8: the exit and a cross-route suggestion, before the track's own words
        draft = self.store.get(draft_id)
        track = self.track(draft)
        from ..ledger import capture

        if track == "prediction":
            # new groups chosen on the segments question come here as a prediction path of the segmentation kind
            self._tool(draft_id, track, "segmentation_project" if (draft.get("understanding") or {}).get("family") == "segmentation"
                       else capture.TOOL_OF_TRACK[track], "rule")
            self.open_prediction(draft_id)
            return
        if track not in intents.NEXT:
            return
        has_data = bool(draft.get("assets"))
        offer = [] if track == "advice" or has_data else ["upload", "connect"]
        need = (draft.get("intent") or {}).get("need")  # the model's one line, kept only with the person's own numbers
        if need:  # "find groups among your customers"; a model that still wrote "They want to ..." is said to the person
            need = re.sub(r"^(?:they|the (?:person|user|customer)|he|she)\s+(?:wants?|would like|needs?)\s+to\s+", "", need.strip(), flags=re.I)
        if need and re.search(r"[\u0600-\u06FF]", need):  # live: a Persian need inside an English sentence read badly; it is quoted
            opening = f"I understood: “{need.rstrip('.')}”."
        elif need:
            opening = f"I understood: you want to {need[:1].lower() + need[1:].rstrip('.')}."
        else:
            opening = f"I read this as {intents.LABELS[track].lower()}: “{draft['problem']}”."
        said_need = str((draft.get("intent") or {}).get("need") or "")
        segments = track == "analysis" and (set((draft.get("intent") or {}).get("signals") or []) & set(intents.SEGMENT_SIGNALS)
                                            or re.search(r"\b(?:segment\w*|cluster\w*|groups?\s+(?:of|among)\b|personas?)", said_need, re.I))
        if segments:  # 2026-10-29: groups have two ways, and only one of them is numbers per group
            self.say(draft_id, f"{opening} {SEGMENTS_SAID}", kind="text", actions=offer, track=track)
            self.ask(draft_id, SEGMENTS_FIELD, "Which way do you need?", list(SEGMENTS_ROUTES),
                     "Groups you already have are counted from a column; new groups are found by a segmentation project.")
            return
        self.say(draft_id, f"{opening} {intents.NEXT[track]}", kind="text", actions=offer, track=track)
        self._tool(draft_id, track, capture.TOOL_OF_TRACK[track], "rule")
        self.open_brief(draft_id)

    def _tool(self, draft_id: str, track: str, tool: str, used_by: str) -> None:
        """P8's gap (2026-10-29): the tool serving the kind of help, recorded once per track and tool (request.tool)."""
        from ..ledger import capture

        kept = self.store.get(draft_id).get("tool") or {}
        if kept.get("tool") == tool and (kept.get("track") == track or kept.get("by") == "person"):
            return  # recorded already (the person's choice of new groups stays theirs when the draft moves to its project path)
        rid = capture.tool_chosen(draft_id, track, tool, used_by, rule_tool=tool if used_by == "rule" else None)
        self.store.update(draft_id, lambda d: d.update(tool={"track": track, "tool": tool, "by": used_by, **({"record": rid} if rid else {})}))

    # ------------------------------------------------------------------ H4: a brief per kind of help
    def open_brief(self, draft_id: str) -> None:
        """H4: read each track's brief from the person's words (every track the request named), keep what was already known,
        record the safe defaults, say which named column the data does not have, then ask what matters (at most once each)."""
        draft = self.store.get(draft_id)
        track = self.track(draft)
        tracks = [t for t in [track, *((draft.get("intent") or {}).get("tracks") or [])] if t in briefs.TRACK_FIELDS]
        if not tracks:
            return
        columns = briefs.columns_of(draft)
        names = [c["name"] for c in columns] or None
        texts = " ".join(plans.user_text(draft))
        missing: dict[str, list[str]] = {}

        model_fields = (draft.get("intent") or {}).get("fields") or {}  # the model's reading of each kind's fields, quote-checked

        def put(d):
            kept = d.setdefault("track_brief", {})
            for t in dict.fromkeys(tracks):
                read = briefs.read(t, texts, names)
                missing.update(read["missing"])
                before = dict(kept.get(t) or {})
                for field in read["missing"]:  # the table has no such column: a reading of the rules is withdrawn, never a person's
                    if (before.get(field) or {}).get("source") == "rules":
                        before.pop(field)
                if names:  # the review: a model field kept earlier (its intent since changed by the person) is checked too
                    known_names = {n.lower() for n in names}
                    for field, entry in list(before.items()):
                        if (entry or {}).get("source") in ("rules", "model"):
                            absent = [w for w in briefs._SNAKE.findall(str(entry.get("value") or "").lower()) if w not in known_names]
                            if absent:
                                missing[field] = absent
                                before.pop(field)
                # 2026-10-29 (AI first): the model's fields come first, the rules fill only what it left; the rules' own reading is
                # kept beside it (for the routing data), and a field naming a column the table does not have is never kept
                for field, entry in (model_fields.get(t) or {}).items():
                    old = before.get(field) or {}
                    if old.get("asked") or old.get("source") not in (None, "default", "rules", "model"):
                        continue  # the person's own answer or edit stands, "Not sure" included (the review)
                    if names:
                        absent = [w for w in briefs._SNAKE.findall(str(entry.get("value") or "").lower()) if w not in {n.lower() for n in names}]
                        if absent:
                            missing[field] = absent
                            before.pop(field, None)
                            continue
                    before[field] = {**entry, "rule_value": (read["fields"].get(field) or {}).get("value")}
                kept[t] = briefs.fill(t, read["fields"], before)
        self.store.update(draft_id, put)
        self.store.emit(draft_id, "status", {"track_brief": self.store.get(draft_id).get("track_brief")})
        said = set((self.store.get(draft_id).get("brief_missing_said") or []))
        missing = {f: a for f, a in missing.items() if f"{f}:{','.join(a)}" not in said}  # said once per column, not with every file
        if missing:
            self.store.update(draft_id, lambda d: d.update(brief_missing_said=sorted(said | {f"{f}:{','.join(a)}" for f, a in missing.items()})))
        self.moved(draft_id, "read_brief", {"tracks": list(dict.fromkeys(tracks))},
                   {"fields": {t: sorted(k for k, v in ((self.store.get(draft_id).get("track_brief") or {}).get(t) or {}).items()
                                         if (v or {}).get("source") == "rules") for t in dict.fromkeys(tracks)},
                    "model_fields": {t: sorted(k for k, v in ((self.store.get(draft_id).get("track_brief") or {}).get(t) or {}).items()
                                               if (v or {}).get("source") == "model") for t in dict.fromkeys(tracks)}})
        for field, absent in missing.items():
            self.say(draft_id, f"Your data has no column named {', '.join(f'`{a}`' for a in absent)}, so I did not use it for the "
                               f"{field.replace('_', ' ')}. Which column did you mean?", kind="text")
        if self.ask_brief(draft_id) is None and self.answer_analysis(draft_id) is None and self.prepare_table(draft_id) is None \
                and self.answer_business(draft_id) is None:
            self.build_dashboard(draft_id)  # H10

    def answer_analysis(self, draft_id: str) -> dict[str, Any] | None:
        """H7: on the analysis track, with the table ready and nothing important left to ask, the question is answered from
        the person's own table: the rules' plan from the brief (or, where they cannot map the words to columns, the
        home_query model's plan from column summaries), run by the query engine, said in words code writes from the plan
        and the result (what was computed, then the numbers). Never a row, never a personal column; a refusal is said with
        its reason and ends the turn. The same question on the same table is answered once."""
        from .. import home_query
        from dclab_core import query as Q

        draft = self.store.get(draft_id)
        if self.track(draft) != "analysis" or self.pending(draft) is not None:
            return None
        if not any(a.get("status") == "ready" for a in draft.get("assets") or []):
            return None
        if not any(m.get("kind") == "analysis" and m.get("asset") == draft.get("active_asset") for m in draft.get("messages") or []):
            return None  # the table is described before it is answered from (a start turn may run after a fast upload)
        brief = (draft.get("track_brief") or {}).get("analysis") or {}
        columns = briefs.columns_of(draft)
        plan, why = home_query.plan_from_brief(brief, columns, draft.get("problem") or "")
        by = "rules"
        if plan is None and self.query_client is not None:
            plan, model_why = home_query.write_plan(self.query_client, draft.get("problem") or "", draft, brief)
            by = "model"
            if plan is None:  # a model's refusal is never shown as its own words: the rules' reason stands
                why = why or "the question could not be mapped to the table's columns"
            else:
                why = ""
        if plan is None:
            return self.say(draft_id, f"I can't answer from the table yet: {why}.", kind="text")
        return self.give_answer(draft_id, plan, by, why if by == "rules" else "", planner="home_query" if by == "model" else None)

    def give_answer(self, draft_id: str, plan: Any, by: str, note: str = "", edited_from: int | None = None,
                    planner: str | None = None) -> dict[str, Any]:
        """H7: one plan answered from the draft's table, the one way every answer is given (the rules' plan, the model's tool,
        a person's edit): never a personal column, the same plan on the same table once, run by the query engine, the words
        and the chart written by code, the answer kept with what it showed and the floor it ran under."""
        from .. import home_chart, home_query
        from dclab_core import query as Q

        draft = self.store.get(draft_id)
        marked = home_query.sensitive_columns(plan, briefs.columns_of(draft))
        if marked:
            return self.say(draft_id, f"That question uses {', '.join(marked)}, which holds personal data; I don't answer with it. "
                                      "Ask about another column.", kind="text")
        last = (draft.get("answers") or [None])[-1]
        if last and last.get("plan_hash") == Q.plan_hash(plan) and last.get("asset") == draft.get("active_asset"):
            return last  # the same question on the same table: answered already (the review of H7a: no duplicate answers)
        try:
            result = home_query.run(self.store, draft_id, plan)
        except Exception as error:  # noqa: BLE001 - a plan this table refuses is said, never a failed table
            return self.say(draft_id, f"That question could not be answered from this table: {str(error)[:300]}", kind="text")
        text = home_query.answer_text(plan, result, note)
        message_id = "m" + secrets.token_hex(5)
        exits_reached: list[tuple[Any, str]] = []
        answer = {"plan": plan.model_dump(mode="json"), "plan_hash": Q.plan_hash(plan), "asset": draft.get("active_asset"),
                  "result": result.model_dump(mode="json"), "shown": home_query.shown_numbers(result), "by": by, "at": now(),
                  "floor": home_query.floor(), "message": message_id, **({"edited_from": edited_from} if edited_from is not None else {})}

        def keep(d):
            d.setdefault("answers", []).append(answer)
            answer["n"] = len(d["answers"]) - 1
            first = (d.get("exit") or {}).get("status") != "reached"
            d["exit"] = {**(d.get("exit") or {}), "kind": "analysis", "track": "analysis", "status": "reached", "answers": len(d["answers"]),
                         "at": now(), **({"labelled": True} if first else {})}
            if first:
                exits_reached.append((d["exit"].get("record"), "analysis"))
        self.store.update(draft_id, keep)
        n = answer["n"]  # set inside the update: two answers at once never swap numbers (the review of H7c)
        from ..ledger import capture

        for record, kind in exits_reached:  # H8: where the request really left labels the exit read for it
            capture.exit_reached(record, kind)
        if edited_from is not None:  # 36.2: the person changed an answer; the changed one is the next result
            self.acted(draft_id, "edit")
        self.acted(draft_id, "answer")
        if by in ("rules", "model"):  # S4: how the question was computed, labelled later by the answer the person keeps
            # the agent's query_data tool plans without asking the rules; answer_analysis asks them first (planner home_query)
            rid = capture.query_planned(draft_id, plan, by, fields=len(((draft.get("track_brief") or {}).get("analysis") or {})),
                                        columns=len(briefs.columns_of(draft)), planner=planner, rules_asked=planner != "home_agent")
            if rid:
                self.store.update(draft_id, lambda d: d["answers"][n].update(record=rid))
        self.moved(draft_id, "answer_analysis", {"plan_by": by, **({"edited_from": edited_from} if edited_from is not None else {})},
                   {"rows": len(result.rows), "suppressed": result.suppressed_groups})
        return self.say(draft_id, text, kind="answer", message_id=message_id, model_text=home_query.model_text(plan),
                        answer={"n": n, "columns": result.columns, "rows": result.rows[:50], "suppressed_groups": result.suppressed_groups,
                                "chart": home_chart.spec(plan, result), "edits": home_query.edits(plan, briefs.columns_of(draft))})

    def build_dashboard(self, draft_id: str, force: bool = False) -> dict[str, Any] | None:
        """H10: on the reporting track, with the table described and nothing left to ask, the dashboard the person's words
        ask for (dclab_core.dashboard): one tile per KPI, each a query plan run by the same engine as an answer (the row floor
        holds), its chart chosen by code. A new data version (another table, a changed one) refreshes every tile and records
        the run; ``force`` refreshes on request. What the rules cannot read is said once, never guessed."""
        from .. import home_chart, home_query
        from dclab_core import dashboard as D
        from . import dashboard as boards

        draft = self.store.get(draft_id)
        if self.track(draft) != "reporting" or self.pending(draft) is not None:
            return None
        active = draft.get("active_asset")
        if not any(m.get("kind") == "analysis" and m.get("asset") == active for m in draft.get("messages") or []):
            return None
        kept = draft.get("dashboard") or {}
        told = (((draft.get("track_brief") or {}).get("reporting") or {}).get("kpis") or {})
        said = str(told.get("value")) if told.get("source") == "user" and told.get("value") else None  # their answer to "Which numbers?"
        if not kept.get("board") and kept.get("missing_for") == active and kept.get("said") == said:
            return None  # what it needs was said for this table, and nothing new was answered
        try:
            frame = home_query.table(self.store, draft)
        except ValueError as error:
            return self.say(draft_id, f"I can't build the dashboard yet: {error}.", kind="text")
        draft = self.store.get(draft_id)  # the table's hash, filled in by the read when it was missing (the review of H10a)
        version = next((a.get("clean_sha256") for a in draft.get("assets") or [] if a.get("id") == active), None)
        if kept.get("board") and kept.get("data_version") == version and not force:
            return None  # this data version is on the dashboard already
        if kept.get("board"):
            board = D.Dashboard.model_validate(kept["board"])
            unread = kept.get("unread") or []
        else:
            spec, unread = boards.read(str(draft.get("problem") or ""), frame, briefs.columns_of(draft), kpis_said=said)
            if spec is None:
                self.store.update(draft_id, lambda d: d.update(dashboard={"missing_for": active, "unread": unread, "said": said}))
                self.moved(draft_id, "build_dashboard", {"refused": True}, {"unread": len(unread)})
                return self.say(draft_id, "I can build a dashboard of numbers per group from your table. To do it here I need "
                                          + "; ".join(unread) + ".", kind="text")
            board = D.Dashboard(title=spec["title"], audience=spec["audience"], refresh=spec["refresh"], kpis=spec["kpis"],
                                floor=home_query.floor())
        refreshed = bool(kept.get("board"))
        from dclab_core import query as Q

        board, tiles = D.refresh(board, frame, data_version=version, at=now(), floor=home_query.floor(), chart=home_chart.spec)
        computed = [t for t in tiles if t.get("result")]
        record = {"board": board.model_dump(mode="json"), "tiles": tiles, "data_version": version, "unread": unread,
                  "shown": [n for t in computed for n in home_query.shown_numbers(Q_result(t["result"]))]}

        def keep(d):
            d["dashboard"] = record
            for k, t in zip(board.kpis, tiles):  # what each tile related, kept like an answer's (H8 carries it into a prediction)
                if t.get("result"):
                    d.setdefault("exposure", []).append({**Q.exposure(k.plan), "at": now(), "by": "dashboard", **home_query.table_identity(d),
                                                         **({"period": {"column": k.plan.period.column, "start": k.plan.period.start,
                                                                        "end": k.plan.period.end}} if k.plan.period else {})})
            if computed:  # a board none of whose tiles ran has not taken the person anywhere
                d["exit"] = {**(d.get("exit") or {}), "kind": "dashboard", "track": "reporting", "status": "reached", "at": now()}
        self.store.update(draft_id, keep)
        from ..ledger import capture

        if computed and (draft.get("exit") or {}).get("status") != "reached":
            capture.exit_reached((draft.get("exit") or {}).get("record"), "dashboard")
        if computed and not refreshed:  # 36.2: a new board is a result; a refresh recomputes the same one (no second result)
            self.acted(draft_id, "dashboard")
        self.moved(draft_id, "refresh_dashboard" if refreshed else "build_dashboard",  # names and hashes only, never a value
                   {"kpis": [{"op": k.plan.measures[0].op, "column": k.plan.measures[0].column, "groups": k.plan.group_by,
                              "filtered_by": [f.column for f in k.plan.filters], "version": k.version, "plan_hash": r.plan_hash}
                             for k, r in zip(board.kpis, board.runs[-1].tiles)], "refresh": board.refresh, "unread": len(unread)},
                   {"tiles": len(tiles), "errors": len(tiles) - len(computed), "data_version": (version or "")[:12]})
        if not computed:
            return self.say(draft_id, "None of the dashboard's tiles could be computed on this table: "
                            + "; ".join(sorted({t['error'] for t in tiles if t.get('error')})) + ". Upload the table it was built on, or "
                            "describe the numbers again.", kind="text")
        head = ("Your dashboard is refreshed on the new data" if refreshed else "Your dashboard is ready") + f": {board.title}."
        cut = [t["kpi"] for t in computed if len(t["result"]["rows"]) >= Q.MAX_LIMIT]
        if cut:
            unread = [*unread, f"{', '.join(cut)}: only the first {Q.MAX_LIMIT} rows of the tile are shown"]
        failed = [t["kpi"] for t in tiles if t.get("error")]
        if failed:
            unread = [*unread, f"not computed on this table: {', '.join(failed)}"]
        cadence = {"daily": "every day", "weekly": "every week", "monthly": "every month"}.get(board.refresh)
        if cadence:
            head += f" You asked for it {cadence}: it is refreshed here whenever the table changes, or with the Refresh button."
        if unread:
            head += " Not read from your words: " + "; ".join(unread) + "."
        return self.say(draft_id, head, kind="dashboard", dashboard={"title": board.title, "audience": board.audience, "refresh": board.refresh,
                                                                     "data_version": (version or "")[:12], "refreshed_at": board.refreshed_at,
                                                                     "tiles": [{"kpi": t["kpi"], "version": t["version"], "chart": t.get("chart"),
                                                                                "error": t.get("error")} for t in tiles]})

    def answer_business(self, draft_id: str, said: str | None = None) -> dict[str, Any] | None:
        """H11: on the business track, with the table described and nothing left to ask, where the number the person asks
        about changed between the two periods they named (dclab_core.change), said as a decision brief written by a template
        from the result alone, always with "association, not cause". What the rules cannot read is said once, never guessed."""
        from .. import home_query
        from . import business

        draft = self.store.get(draft_id)
        if self.track(draft) != "business" or self.pending(draft) is not None:
            return None
        active = draft.get("active_asset")
        if not any(m.get("kind") == "analysis" and m.get("asset") == active for m in draft.get("messages") or []):
            return None
        done = draft.get("business") or {}
        if done.get("asset") == active and not (said and done.get("missing")):
            return None  # answered (or said what it needs) for this table already; a reply to what it needs is read below
        try:
            frame = home_query.table(self.store, draft)
        except ValueError as error:
            return self.say(draft_id, f"I can't compare from the table yet: {error}.", kind="text")
        told = (draft.get("track_brief") or {}).get("business") or {}
        measure_words = str((told.get("success_measure") or {}).get("value") or "")
        text = f"{said}. {draft.get('problem') or ''}" if said else str(draft.get("problem") or "")  # the reply, with the question it answers
        spec, missing = business.read(text, frame, measure_words, briefs.columns_of(draft))
        try:
            result = business.compare(spec, frame) if spec else None
        except ValueError as error:
            result, missing = None, str(error)
        if result is None:
            self.store.update(draft_id, lambda d: d.update(business={"asset": active, "missing": missing or "a comparison the table holds"}))
            self.moved(draft_id, "answer_business", {"refused": True}, {"missing": (missing or "")[:200]})
            if said and done.get("missing") == missing:
                return None  # the reply did not give it either: the normal path reads the message
            answered = None if said else self._to_model(draft_id)
            if answered is not None:
                return answered  # 36.3: periods the rules could not read are the model's to read, never "say it again"
            return self.say(draft_id, f"I can show where a number changed between two periods. To do it here I need {missing or 'two periods with rows in both'}.",
                            kind="text")
        return self.give_brief(draft_id, spec, result)

    def give_brief(self, draft_id: str, spec: dict[str, Any], result: Any, by: str = "rules") -> dict[str, Any]:
        """H11: one comparison said as a decision brief: the template, or the home_business model's wording where every
        sentence passed cited.py; kept with the numbers it can show and how it was worded."""
        from . import business

        draft = self.store.get(draft_id)
        active = draft.get("active_asset")
        template = business.brief(result, spec["narrow"])
        text, worded = business.reword(self.business_client, result, template, spec["narrow"])
        record = {"asset": active, "result": result.model_dump(mode="json"), "narrow": spec["narrow"], "shown": business.shown(result),
                  "template": template, "worded": worded, "by": by, "at": now()}

        def keep(d):
            d["business"] = record
            d["exit"] = {**(d.get("exit") or {}), "kind": "decision_brief", "track": "business", "status": "reached", "at": now()}
        self.store.update(draft_id, keep)
        from ..ledger import capture

        if (draft.get("exit") or {}).get("status") != "reached":
            capture.exit_reached((draft.get("exit") or {}).get("record"), "decision_brief")
        self.acted(draft_id, "brief")  # 36.2
        self.moved(draft_id, "answer_business", {"by": result.by, "measure": result.measure, "periods": [result.a.label, result.b.label],
                                                  "narrowed_by": sorted(spec["narrow"]), "splits": spec["splits"], "planned_by": by,
                                                  "scores": {k: round(v, 4) for k, v in spec.get("scores", {}).items()}},
                   {"segments": len(result.segments), "worded_by": worded.get("by")})
        # the model's history reads what was compared, never the totals, the segments or the narrowing values
        return self.say(draft_id, text, kind="decision_brief",
                        model_text=f"(A decision brief is shown to the person: where {result.measure or 'the number of rows'} changed between "
                                   f"{result.a.label} and {result.b.label}, by {result.by}. Its numbers are shown to the person, not here; "
                                   "do not state or guess them.)")

    def _other_tables(self, draft: dict[str, Any]) -> dict[str, Any]:
        """H9: the draft's other prepared tables by name (a join's second table), each checked against its recorded hash."""
        import pandas as pd

        from ..audit import sha256_of

        out: dict[str, Any] = {}
        for a in draft.get("assets") or []:
            if a.get("id") == draft.get("active_asset") or a.get("status") != "ready" or not (a.get("kept_file") or a.get("clean_file")):
                continue
            path = Path(self.store.data_dir(draft["id"])) / (a.get("kept_file") or a["clean_file"])  # each table's own copy
            if path.is_file() and (not a.get("clean_sha256") or sha256_of(path) == a["clean_sha256"]):
                out[Path(str(a.get("name") or a["id"])).stem.lower()] = pd.read_parquet(path)
        return out

    def prepare_table(self, draft_id: str, said: str | None = None) -> dict[str, Any] | None:
        """H9: on the preparation track, with the table described and nothing left to ask, the steps the person's words ask
        for (dclab_core.recipe), each previewed and applied in order to a copy; a step that removes over 5% of the rows, or a
        join that changes the row count, waits for the person's yes ("not sure" skips it: rows are never dropped by default).
        The result is a new file beside the original (never written), ready to download in the format asked for."""
        from .. import home_query
        from dclab_core import recipe as R
        from . import prepare

        draft = self.store.get(draft_id)
        if self.track(draft) != "preparation" or self.pending(draft) is not None:
            return None
        active = draft.get("active_asset")
        if not any(m.get("kind") == "analysis" and m.get("asset") == active for m in draft.get("messages") or []):
            return None
        rec = dict(draft.get("recipe") or {})
        mine = rec.get("for") == active  # the recipe made for this upload (its main table may be another, for a join)
        follow_up = bool(said) and mine and rec.get("unread_said") and not rec.get("done")
        if mine and (rec.get("done") or rec.get("unread_said")) and not follow_up:
            return None
        made_for = active
        try:
            frame = home_query.table(self.store, draft)
        except ValueError as error:
            return self.say(draft_id, f"I can't prepare the table yet: {error}.", kind="text")
        tables = self._other_tables(draft)
        told = ((draft.get("track_brief") or {}).get("preparation") or {}).get("rules") or {}
        # the steps: a reply to Home's request for them, else the person's own answer to "What should I do to the data?",
        # else their first sentence
        text = said if follow_up else str(told.get("value")) if told.get("source") == "user" and told.get("value") else \
            str(draft.get("problem") or "")
        joins = prepare.JOIN.search(lang.normalize(text).lower())
        if joins and not tables and not rec.get("waiting_said"):  # a join needs its second table: wait for it, said once
            self.store.update(draft_id, lambda d: d.update(recipe={"waiting_said": True}))
            return self.say(draft_id, "Your request joins two tables: add the second one and I'll prepare them together.", kind="text",
                            actions=["upload", "connect"])
        if joins and not tables:
            return None
        if tables:  # the table a join adds to is the one where the key repeats, never just the last upload (the review of H9a)
            here = Path(str(next((a.get("name") for a in draft.get("assets") or [] if a.get("id") == active), active))).stem.lower()
            main = prepare.main_table(text, {here: frame, **tables})
            if main and main != here:
                tables = {**{k: v for k, v in tables.items() if k != main}, here: frame}
                frame = self._other_tables(draft)[main]
                active = next((a["id"] for a in draft.get("assets") or [] if Path(str(a.get("name") or "")).stem.lower() == main), active)
        if not mine or follow_up:
            steps, unread = prepare.read_steps(text, frame, tables)
            rec = {"for": made_for, "asset": active, "planned": [x.model_dump(mode="json") for x in steps], "unread": unread, "decisions": {},
                   "format": prepare.destination(text) if follow_up and prepare.destination(text) != "csv" else
                   prepare.destination(str(draft.get("problem") or "")), "source_sha256": next(
                       (a.get("clean_sha256") for a in draft.get("assets") or [] if a.get("id") == active), None)}
        if not rec["planned"]:
            rec["unread_said"] = True
            self.store.update(draft_id, lambda d: d.update(recipe=rec))
            if follow_up and self.client is not None:
                return None  # H9b: words the rules cannot read are the model's to read, through propose_step (the review of H9b)
            if not follow_up and self._to_model(draft_id):
                return None  # 36.3: a first request the rules cannot read is the model's too, never "say it again"
            words = "; ".join(f"“{w}”" for w in rec.get("unread") or [])
            return self.say(draft_id, "I could not read the steps from your words yet" + (f" ({words})" if words else "")
                            + ". Say them one at a time, for example: remove duplicate orders; keep only the paid orders.", kind="text")
        return self.run_recipe(draft_id, rec, frame, tables)

    def _frames_for(self, draft: dict[str, Any], asset_id: str | None) -> tuple[Any, dict[str, Any]]:
        """H9b: the recipe's main table (by its asset) and the draft's other tables by name, every one checked by its hash."""
        from .. import home_query

        active = draft.get("active_asset")
        here = Path(str(next((a.get("name") for a in draft.get("assets") or [] if a.get("id") == active), active))).stem.lower()
        frames = {here: home_query.table(self.store, draft), **self._other_tables(draft)}
        main = Path(str(next((a.get("name") for a in draft.get("assets") or [] if a.get("id") == (asset_id or active)), here))).stem.lower()
        frame = frames.pop(main, None)
        if frame is None:
            frame = frames.pop(here)
        return frame, frames

    def run_recipe(self, draft_id: str, rec: dict[str, Any], frame: Any, tables: dict[str, Any]) -> dict[str, Any] | None:
        """H9: every planned step in order, from the original table (never written), stopping at a guarded step that has no
        yes; the result written as its own file and said with each step as it ran."""
        from dclab_core import recipe as R
        from . import prepare

        draft = self.store.get(draft_id)
        active = rec.get("asset") or draft.get("active_asset")
        current, done = frame, []
        since_yes = len(frame)  # rows when the person last said yes (or the original): small steps that together remove many ask too
        for i, raw in enumerate(rec["planned"]):
            step = R.Step.model_validate(raw)
            decision = rec["decisions"].get(str(i))
            if decision == "skip":
                done.append({"kind": step.kind, "why": step.why, "skipped": True, "removed_share": 0.0, "confirmed": False})
                continue
            try:
                p, after = R.preview(step, current, tables)
            except (ValueError, TypeError) as error:  # a pandas type error is the step's, never the turn's (the review of H9b)
                done.append({"kind": step.kind, "why": step.why, "error": str(error)[:300], "removed_share": 0.0, "confirmed": False})
                continue
            together = max(0, since_yes - p.rows_after) if step.kind in ("filter", "dedupe", "drop_missing") else 0
            if not p.needs_confirmation and since_yes and together / since_yes > R.GUARD_SHARE:  # the review of 2026-10-28
                p = p.model_copy(update={"needs_confirmation": True, "reason": f"with the steps before it, it leaves {p.rows_after:,} of "
                                                                                  f"{since_yes:,} rows ({together / since_yes:.0%} removed)"})
            if p.needs_confirmation and decision != "yes":
                rec["pending"] = {"index": i, "needs_confirmation": True, "reason": p.reason, "kind": step.kind}
                rec.setdefault("asked", []).append({"index": i, "kind": step.kind, "reason": p.reason, "rows_before": p.rows_before,
                                                    "rows_after": p.rows_after, "removed_share": p.removed_share})  # kept: what was asked
                unread = f" (Not read from your words: {'; '.join(rec['unread'])}.)" if rec.get("unread") else ""
                q = {"id": f"q{len(draft['questions']) + 1}", "field": "confirm_step", "brief": None, "step": i,
                     "text": f"This step will {prepare.describe(step)}: {p.reason}.{unread} Apply it?", "options": ["Yes, apply it", "No, skip it"],
                     "why": "A step that removes many rows, or a join that changes the row count, is applied only with your yes.",
                     "answered": None,
                     "model_text": f"Apply this step: {prepare.describe(step)}? (It removes many rows or changes the row count.)"}
                self.store.update(draft_id, lambda d: (d.update(recipe=rec), d["questions"].append(q)))
                self.moved(draft_id, "confirm_step", {"kind": step.kind}, {"removed_share": p.removed_share})
                # the model reads the question without its counts: a filter it chose plus a count would tell it a row's value
                return self.say(draft_id, q["text"], kind="question", question=q,
                                model_text=f"(Asked the person whether to {prepare.describe(step)}; it removes many rows or changes the row count.)")
            current = after
            if p.needs_confirmation:  # asked and confirmed: what is removed from here on is counted from this table
                since_yes = p.rows_after
            done.append({"kind": step.kind, "why": step.why, **p.model_dump(mode="json"), "confirmed": decision == "yes"})
        name = prepare.prepared_name(active)
        path = Path(self.store.data_dir(draft_id)) / name
        current.to_parquet(path, index=False)  # a new file under a name an upload never takes: the original is never written
        from .pipeline import _record

        _record(self.store, path)  # in file storage too, so the download works where the worker ran elsewhere
        recipe = R.Recipe(source_sha256=rec.get("source_sha256"), steps=[R.Step.model_validate(r) for i, r in enumerate(rec["planned"])
                                                                        if rec["decisions"].get(str(i)) != "skip" and not done[i].get("error")])
        rec.update(done=True, pending=None, steps=done, recipe=recipe.model_dump(mode="json"), file=name, asset=active,
                   rows=int(len(current)), columns=int(current.shape[1]), at=now())

        def keep(d):
            d["recipe"] = rec
            d["exit"] = {**(d.get("exit") or {}), "kind": "dataset", "track": "preparation", "status": "reached", "at": now()}
        self.store.update(draft_id, keep)
        from ..ledger import capture

        capture.exit_reached((draft.get("exit") or {}).get("record"), "dataset")
        self.acted(draft_id, "table")  # 36.2
        self.moved(draft_id, "prepare_table",  # what the rules read (kinds and column names, never a value) and what came of it
                   {"steps": [{"kind": x["kind"], **({k: v for k, v in raw.items() if k in ("columns", "keys", "index", "spread", "values", "agg", "table")})}
                              for x, raw in zip(done, rec["planned"])], "unread": len(rec.get("unread") or [])},
                   {"rows": int(len(current)), "columns": int(current.shape[1]),
                    "applied": [not (x.get("skipped") or x.get("error")) for x in done]})
        described = [prepare.describe(R.Step.model_validate(raw)) for raw in rec["planned"]]
        lines = [f"{n}. {described[n - 1]}" + (" (skipped)" if x.get("skipped") else f" (not applied: {x['error']})" if x.get("error")
                 else f": {x['rows_before']:,} → {x['rows_after']:,} rows" + (f", added {', '.join(x['added'])}" if x.get("added") else ""))
                 for n, x in enumerate(done, 1)]
        unread = f"\n\nNot read from your words: {'; '.join(rec['unread'])}." if rec.get("unread") else ""
        applied = [f"{n}. {described[n - 1]}" + (" (skipped)" if x.get("skipped") else " (not applied)" if x.get("error") else "")
                   for n, x in enumerate(done, 1)]
        return self.say(draft_id, f"Your prepared table is ready: {len(current):,} rows and {current.shape[1]} columns. The original "
                                  "file is unchanged.\n\n" + "\n".join(lines) + unread, kind="prepared",
                        prepared={"format": rec["format"], "rows": int(len(current)), "columns": int(current.shape[1])},
                        model_text="(The prepared table is ready; the steps as run:\n" + "\n".join(applied) + ")")

    def change_recipe(self, draft_id: str, change: Callable[[dict[str, Any]], None], main: str | None = None) -> dict[str, Any] | None:
        """H9b: the recipe changed (a step added or the last one undone) and run again from the original table. ``main``: for
        a new recipe, the asset a join adds to (the table where its key repeats)."""
        from .. import home_query

        draft = self.store.get(draft_id)
        active = draft.get("active_asset")
        rec = dict(draft.get("recipe") or {})
        if rec.get("for") != active or "planned" not in rec:
            rec = {"for": active, "asset": main or active, "planned": [], "unread": [], "decisions": {}, "format": "csv",
                   "source_sha256": next((a.get("clean_sha256") for a in draft.get("assets") or [] if a.get("id") == active), None)}
            home_query.table(self.store, draft)  # the table's hash checked before anything is planned on it
        change(rec)
        rec.update(done=False, pending=None, unread_said=False)
        frame, tables = self._frames_for(draft, rec.get("asset"))
        if not rec["planned"]:
            self.store.update(draft_id, lambda d: d.update(recipe={**rec, "done": False}))
            return None
        return self.run_recipe(draft_id, rec, frame, tables)

    def confirm_step(self, draft_id: str, q: dict[str, Any], text: str) -> bool:
        """H9: the person's answer to a guarded step. Only a whole yes applies it ("ok no", "اعمال نکن" never do); a whole no
        or "not sure" skips it. Anything else is not an answer: the question closes undecided and the message is read as a
        message (False), so "actually I want a prediction" is never lost as a skip (the review of H9a)."""
        from . import prepare

        said = str(text or "").strip()
        yes = bool(prepare.YES.fullmatch(said)) and not prepare.NEGATION.search(said)
        no = bool(prepare.NO.fullmatch(said)) or not_known(said)
        decided = "yes" if yes else "skip" if no else None

        def put(d):
            for x in d["questions"]:
                if x["id"] == q["id"]:
                    x.update(answered=said[:400], **({} if decided else {"superseded": True, "declined": True}))
            r = d.setdefault("recipe", {})
            if decided:
                r.setdefault("decisions", {})[str(q.get("step"))] = decided
            r["pending"] = None
            if not decided:
                r["undecided_said"] = True
        self.store.update(draft_id, put)
        self.moved(draft_id, "confirm_step_answer", {"answer": decided or "other"}, {})
        if yes or prepare.NO.fullmatch(said):  # 36.2: a whole yes or a whole no; "not sure" also skips the step but is no verdict
            self.acted(draft_id, "step_yes" if yes else "step_no")
        from ..ledger import capture

        capture.recipe_step_answered(draft_id, (self.store.get(draft_id).get("recipe") or {}), q.get("step"), decided)
        if decided:
            rec = self.store.get(draft_id).get("recipe") or {}
            if rec.get("planned"):  # the recipe as it stands, on its own main table: never read again from the words (the review of H9b)
                frame, tables = self._frames_for(self.store.get(draft_id), rec.get("asset"))
                self.run_recipe(draft_id, rec, frame, tables)
            else:
                self.prepare_table(draft_id)
        return decided is not None

    @staticmethod
    def _answer_root(answers: list[dict[str, Any]], n: int) -> int | None:
        """The answer an edit started from: the first in its chain of edits (S4), by its number."""
        seen = set()
        while 0 <= n < len(answers) and answers[n].get("edited_from") is not None and n not in seen:
            seen.add(n)
            n = answers[n]["edited_from"]
        return n if 0 <= n < len(answers) else None

    def save_answer(self, draft_id: str, n: int, saved: bool = True, by: str = "person") -> dict[str, Any]:
        """H7c: the person keeps an answer (it goes into the notebook export) or lets it go. S4: the first answer a person
        keeps for a question labels that question's first plan with the kept answer's plan, once (the model's save_answer,
        on the person's word, is not that label)."""
        judge: list[tuple[str, Any]] = []

        def put(d):
            answers = d.get("answers") or []
            if not 0 <= n < len(answers):
                raise ValueError("No such answer")
            answers[n]["saved"] = bool(saved)
            root = self._answer_root(answers, n) if saved and by == "person" else None
            if root is not None and answers[root].get("record") and not answers[root].get("judged"):
                answers[root]["judged"] = True  # inside the update: two saves at once label once
                judge.append((answers[root]["record"], answers[n]["plan"]))
        self.store.update(draft_id, put)
        self.moved(draft_id, "save_answer", {"answer": n, "saved": bool(saved)}, {})
        if by == "person":  # 36.2: the person's own keep or let-go; a model's save on its own is not their verdict
            self.acted(draft_id, "save" if saved else "unsave")
        if judge:
            from ..ledger import capture

            capture.query_judged(*judge[0])
        return self.store.get(draft_id)["answers"][n]

    def edit_answer(self, draft_id: str, n: int, change: dict[str, Any]) -> dict[str, Any]:
        """H7c: an answer changed by the person's control (the operation, the split, the time grain), by code and never a
        model; the new plan is checked like any other and answered as a new answer that says which it changed."""
        from .. import home_query

        answers = self.store.get(draft_id).get("answers") or []
        if not 0 <= n < len(answers):
            raise ValueError("No such answer")
        plan = home_query.edit_plan(answers[n]["plan"], change, briefs.columns_of(self.store.get(draft_id)))
        return self.give_answer(draft_id, plan, "edit", edited_from=n)

    def ask_brief(self, draft_id: str) -> dict[str, Any] | None:
        """The next brief question of the current track: unknown, no safe default, not asked before, within its budget; with
        the table's columns (names only) as buttons and Not sure."""
        draft = self.store.get(draft_id)
        track = self.track(draft)
        if track not in briefs.TRACK_FIELDS or self.pending(draft) is not None:
            return None
        asked = sum(1 for q in draft["questions"] if q.get("brief") == track)
        f = briefs.asks(track, (draft.get("track_brief") or {}).get(track) or {}, asked, has_columns=bool(briefs.columns_of(draft)))
        if f is None:
            return None
        choices = [o for o in briefs.options(f, briefs.columns_of(draft)) if o != briefs.NOT_SURE][:3]  # at most four buttons
        q = {"id": f"q{len(draft['questions']) + 1}", "field": f.name, "brief": track, "text": f.question,
             "options": [*choices, briefs.NOT_SURE], "why": f.why, "answered": None}
        self.store.update(draft_id, lambda d: d["questions"].append(q))
        self.moved(draft_id, "ask_brief", {"track": track, "field": f.name})
        return self.say(draft_id, f.question, kind="question", question=q)

    @staticmethod
    def answers_brief(q: dict[str, Any], text: str) -> bool:
        """Whether a message answers the open brief question: "Not sure", one of its buttons, or a short reply that names no
        other kind of help and asks for no build or simulated data (the review of H4: "actually I want to predict churn"
        and "can you make synthetic data" were taken as the measure)."""
        said = str(text or "").strip()
        if not_known(said) or said.lower() in {str(o).lower() for o in q.get("options") or []}:
            return True
        if len(said.split()) > 12 or BUILD_WORDS.search(said) or SIMULATE_WORDS.search(said) or "?" in said or "؟" in said:
            return False
        scores = intents.rules(said)["scores"]
        return not any(score >= intents.STRONG for track, score in scores.items() if track != q.get("brief"))

    def brief_answer(self, draft_id: str, q: dict[str, Any], value: str) -> None:
        """H4: the person's answer to a brief question: their words, or the default when they do not know; nothing else in
        the brief changes (a "Not sure" never erases what was understood). Then the next question, or what DCLab does now."""
        track, field = q["brief"], q["field"]

        def put(d):
            kept = d.setdefault("track_brief", {})
            kept[track] = briefs.answered(track, kept.get(track) or {}, field, value, not_known(value))
            for item in d["questions"]:
                if item["id"] == q["id"]:
                    item["answered"] = value.strip()[:400]
        self.store.update(draft_id, put)
        self.store.emit(draft_id, "status", {"track_brief": self.store.get(draft_id).get("track_brief")})
        self.moved(draft_id, "brief_answer", {"track": track, "field": field})
        if self.ask_brief(draft_id) is None and self.answer_analysis(draft_id) is None and self.prepare_table(draft_id) is None \
                and self.answer_business(draft_id) is None and self.build_dashboard(draft_id) is None:
            self.say(draft_id, "Noted. " + intents.NEXT.get(track, ""), kind="text")

    @turn
    def set_brief(self, draft_id: str, track: str, field: str, value: str) -> dict[str, Any]:
        """H4: the person changes one field on the "DCLab understood" sheet: their words, source user."""
        if briefs.field(track, field) is None:
            raise ValueError(f"{field} is not a field of the {track} brief")
        value = str(value or "").strip()[:200]
        if not value:
            raise ValueError("write the value first")

        def put(d):
            kept = d.setdefault("track_brief", {})
            before = (kept.get(track) or {}).get(field) or {}
            entry = {"value": value, "status": "stated", "source": "user", "quote": value, "asked": True,
                     "corrected": {"value": before.get("value"), "source": before.get("source")} if before.get("value") else None}
            kept[track] = {**(kept.get(track) or briefs.fill(track, {})), field: entry}
            for q in d["questions"]:  # the sheet answers the open question on this field (the review of H4)
                if q.get("brief") == track and q.get("field") == field and not q.get("answered"):
                    q["answered"] = value
        self.store.update(draft_id, put)
        self.moved(draft_id, "set_brief", {"track": track, "field": field})  # a correction of a reading: kept for training
        msg = {"id": "u" + secrets.token_hex(5), "role": "user", "kind": "edit", "field": field, "text": value, "at": now()}
        self.store.update(draft_id, lambda d: d["messages"].append(msg))
        self.store.emit(draft_id, "chat", msg)
        self.store.emit(draft_id, "status", {"track_brief": self.store.get(draft_id).get("track_brief")})
        return self.store.get(draft_id)["track_brief"][track]

    def choose_intent(self, draft_id: str, track: str, *, by_person: bool = True) -> dict[str, Any]:
        """The person chose the kind of help (the card's Change control, the question's button, or a pack): it wins over
        any reading, labels this draft's own reading, and moves Home to that track. ``by_person=False``: Home moves the draft
        itself (new groups take the project path): the move labels nothing (the review, 2026-10-29)."""
        with self.store.turn(draft_id):
            draft = self.store.get(draft_id)
            before = draft.get("intent") or {}
            was = self.track(draft)
            value = intents.chosen(track, before)
            from ..ledger import capture

            hint = draft.get("suggestion_record") or {}
            if by_person:
                capture.intent_chosen(before.get("record"), track)
                if hint.get("track") == track and hint.get("on") == was:  # the person took the suggested route
                    capture.suggestion_taken(hint.get("record"), track)
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
                                              or (q.get("brief") and q.get("brief") != track)):
                    q.update(answered="(the type of help changed)", superseded=True)
            if track != was:  # a result made on the earlier track never judges this one (track_outcome reads from here)
                d["track_actions_from"] = int(d.get("actions_total", len(d.get("actions") or [])))
            if track != "prediction":
                d["workflow"] = None
                if (d.get("pack") or {}).get("source") != "user":
                    d["pack"] = None  # 2026-10-29: a pack set while the draft was read as a prediction showed on an analysis
        self.store.update(draft_id, settle)
        if track != was:
            self.open_track(draft_id)

    def set_pack(self, draft_id: str, key: str, source: str, why: str) -> None:
        def put(d):
            d["pack"] = {"key": key, "source": source, "why": why, "kind": packs.kind_for(key)}
        self.store.update(draft_id, put)
        self.store.emit(draft_id, "status", {"pack": {"key": key, "source": source, "why": why, "kind": packs.kind_for(key)}})
        from ..ledger import capture  # 19.7: the pack, detected or chosen

        capture.pack(draft_id, key, source)

    # ------------------------------------------------------------------ entry points
    @turn
    def start(self, draft_id: str) -> None:
        """A new draft with a problem sentence: read the kind of help first (a person who chose a domain pack asked for a
        prediction), then open its track, or ask when the reading is a real tie (H2)."""
        draft = self.store.get(draft_id)
        if (draft.get("intent") or {}).get("source") == "user":  # chosen before Home read it (the wizard, an API client)
            reading = draft["intent"]
        elif (draft.get("pack") or {}).get("source") == "user":
            reading = self.set_intent(draft_id, intents.chosen("prediction"), sample=False)
        else:
            reading = self.set_intent(draft_id, intents.read(draft["problem"], self.intent_client, plans.user_text(draft)))
        if reading["status"] == "asked":
            self.ask_intent(draft_id, reading)
            return
        self.open_track(draft_id)

    UNDERSTAND_TURNS = 2  # open questions before Home routes by the buttons alone

    def ask_intent(self, draft_id: str, reading: dict[str, Any]) -> None:
        """2026-10-29 (the owner: understand before routing): an unclear request gets one open question, the model's own
        when it kept the person's numbers, with every kind of help as a button (its reading first); two readings that
        clash get the either-or question. A prediction is never assumed."""
        if reading.get("ask") == "open":
            first = [t for t in reading.get("tracks") or [] if t in intents.TRACKS]
            order = first + [t for t in intents.TRACKS if t not in first]
            self.ask(draft_id, "intent", reading.get("question") or intents.OPEN_QUESTION, [intents.LABELS[t] for t in order],
                     "Answer in your own words or pick one; nothing is built until the kind of help is clear.")
            return
        options = [intents.LABELS[t] for t in reading["tracks"][:3]]
        self.ask(draft_id, "intent", "Which kind of help do you need?", options + ["Something else"],
                 "A prediction needs a model and a moment; an answer about what already happened needs neither.")

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
        # Live (2026-10-16): the opening quoted the person's whole sentence back and explained a default pack; it now
        # says what was understood from it, and why a pack only when something in the words chose it
        self.say(draft_id, (f"From your message I have {' and '.join(known)}; correct them on the sheet if I misread."
                            if known else ("Thanks. This kind of problem has no outcome to predict: first let's pin down which rows "
                                           "are grouped and what describes them." if (u.get("family") in NO_OUTCOME_COLUMN)
                                           else "Thanks. First let's pin down what the model should predict, and when."))
                 + (f" I'll start from the {name} pack ({why[:1].lower() + why[1:]})." if det["source"] != "default" else "")
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
        cases = analysis.get("cases_table") or {}
        if cases and asset.get("kind") != "labels":  # a list of known cases alone: say what it is and what is missing
            # names and counts only: this text goes into the model's history, where no cell value may go
            why = ([f"`{k}` is the same on every row" for k in cases["same_on_every_row"]]
                   + [f"`{k}` names the kind of case ({len(v)} kinds)" for k, v in cases["case_kinds"].items()]
                   + [f"`{cases['keys'][0]}` is different on every row"])
            text += (" This file looks like a list of known cases, not the table to learn from: " + "; ".join(why) + ". On its own"
                     " every row is a case, so there is nothing to tell the cases from. Add the table these cases belong to (with "
                     f"`{cases['keys'][0]}`) to this draft, and DCLab will join them: the cases become the labels.")
            for t in cases.get("tables") or []:
                text += (f" Another of your drafts has {t['asset']} ({t['rows']:,} rows), where {t['matched']} of these "
                         f"{t['of']} `{t['key']}` values appear: it is probably that table.")
                break
        self.say(draft_id, text, kind="analysis", asset=asset.get("id"))
        if not prediction:  # H2, another track: the description is what Home offers for it now; no pack, workflow or outcome question
            self.open_brief(draft_id)  # H4: the brief again, now with the table's columns (a name it does not have is said)
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
        if not pending:
            self.next_turn(draft_id)

    @turn
    def data_failed(self, draft_id: str, asset: dict[str, Any], error: str) -> None:
        self.say(draft_id, f"I could not use {asset.get('name', 'that file')}: {error}", kind="text", severity="warning", asset=asset.get("id"))
        self.refresh_workflow(draft_id)

    @turn
    def reply_built(self, draft_id: str, text: str, orchestrator: Any, project_id: str) -> None:
        """A message on the chat of a project that is built (the owner, 2026-10-20: "ok do it" and "ok" got nonsense, "build improvements"
        got advice again). The conversation that built the project is over; the person now asks about the project or asks for work on
        it, and the Home orchestrator answers (``draft.orchestrator``): it reads the conversation and the project's state, and its
        specialists show the project's records as tables, improve the model, or write a notebook cell. A word to build again, with
        nothing offered, is answered by saying it is built."""
        from . import orchestrator as orch

        text = text.strip()[:2000]
        msg = {"id": "u" + secrets.token_hex(5), "role": "user", "kind": "text", "text": text, "at": now()}
        self.store.update(draft_id, lambda d: d["messages"].append(msg))
        self.store.emit(draft_id, "chat", msg)
        offered = ((self.store.get(draft_id).get("agent") or {}).get("offer"))
        versions = self.store.get(draft_id).get("problem_versions") or []
        waiting = bool(versions and not versions[-1].get("built"))  # step 2: a confirmed version waits (one that cannot be built says why)
        if BUILD_WORDS.search(text) and not offered and not waiting and not orch.IMPROVE.search(text):
            self.say(draft_id, "This chat's project is already built. Ask about it here (its models, its evaluation, how to improve it), "
                               "open it from Projects, or start a new project from a new chat.", kind="project_answer", chips=orch.CHIPS[None])
            return
        try:
            orchestrator.turn(draft_id, project_id, text)
        except Exception as exc:  # noqa: BLE001 - the project could not be read just now: said, never a silence
            self.say(draft_id, f"I could not work on this project just now ({type(exc).__name__}). Open its pages (Models, Evaluation) or ask again.",
                     kind="text", severity="warning")

    @turn  # one turn at a time: a message waits for a data-ready turn instead of landing in its gap (lost on 2026-10-20, found by the gate)
    def reply(self, draft_id: str, text: str) -> None:
        """A user message: store it, then let the model (or the script) answer."""
        text = text.strip()[:2000]
        msg = {"id": "u" + secrets.token_hex(5), "role": "user", "kind": "text", "text": text, "at": now()}

        def add(d):
            d["messages"].append(msg)
        self.store.update(draft_id, add)
        self.store.emit(draft_id, "chat", msg)
        open_question = self.pending(self.store.get(draft_id))
        if open_question is not None and open_question.get("brief"):  # H4: a brief question is answered by code when it is an answer
            if self.answers_brief(open_question, text):
                self.brief_answer(draft_id, open_question, text)
                return
            self.store.update(draft_id, lambda d: [x.update(answered=text[:400], declined=True, superseded=True)
                                                   for x in d["questions"] if x["id"] == open_question["id"] and not x.get("answered")])
            # the person said something else (another kind of help, simulated data, a question): the normal path reads it
        if open_question is not None and open_question["field"] == "confirm_step":  # H9: a guarded step's yes or no, by code
            if self.confirm_step(draft_id, open_question, text):
                return
        biz = self.store.get(draft_id).get("business") or {}
        if self.track(self.store.get(draft_id)) == "business" and biz.get("missing"):  # H11: Home said what it needs: this may be it
            if self.answer_business(draft_id, said=text) is not None:
                return
        rec = self.store.get(draft_id).get("recipe") or {}
        if self.track(self.store.get(draft_id)) == "preparation" and rec.get("unread_said") and not rec.get("done"):
            if self.prepare_table(draft_id, said=text) is not None:  # H9: Home asked for the steps one at a time: this is them
                return
        if open_question is not None and open_question["field"] in ("intent", checks.FIELD):  # H2, H3.1: code reads these answers
            self.take_answer(draft_id, open_question, text)
            return
        if asks_to_build(self.store.get(draft_id)["messages"]) and self.track(self.store.get(draft_id)) == "prediction":
            # 25.13: "build it" (or yes to an offer) builds, as the button does; only a prediction builds (H2)
            done = self.on_request(draft_id, "build", {})
            if done is not None:
                self.built(draft_id, done)
                return
        waiting = self.pending(self.store.get(draft_id))
        if self.client is not None and waiting is not None and waiting.get("source") == "second_opinion":
            # 28.6 (the review): the model's turn reaches take_answer only when it made no tool step; the person's choice
            # between the two readings is read here first, by code, and the model then carries the conversation on
            self.take_second_opinion(draft_id, waiting, text, then_next=False)
        elif self.client is not None and waiting is not None and not_known(text):
            # live (the 45 again, 2026-10-26): "I don't know" to a model's question (the review budget) was left open by the
            # model's turn, and the walk waited on it for ever; code closes it as declined first (the field stays unknown, the
            # build assumes what it assumes and says so), and the model then carries the conversation on
            self.record(draft_id, waiting["field"], text)  # the plan's field: unknown, the question declined (never asked again)

            def closed(d: dict[str, Any]) -> None:  # a field outside the plan (a kind's, the repeating key's) is closed here too
                for item in d["questions"]:
                    if item["id"] == waiting["id"] and not item.get("answered"):
                        item["answered"], item["declined"] = text[:400], True
            self.store.update(draft_id, closed)
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

    @turn
    def edit(self, draft_id: str, field: str, value: str) -> dict[str, Any]:
        """25.5: the person corrects one field of the understanding on the page. Their words are a message of the
        conversation (kind ``edit``), and the field is recorded from them through the draft guard, as any record is."""
        if field not in plans.ALL_FIELDS:
            raise ValueError(f"{field} is not a field of the understanding; one of {', '.join(plans.ALL_FIELDS)}")
        value = value.strip()[:400]
        if not value:
            raise ValueError("write the value first")
        msg = {"id": "u" + secrets.token_hex(5), "role": "user", "kind": "edit", "field": field, "text": value, "at": now()}
        self.store.update(draft_id, lambda d: d["messages"].append(msg))
        self.store.emit(draft_id, "chat", msg)
        result = self.run_tool(draft_id, "record", {"field": field, "value": value, "quote": value, "status": "stated"})
        if "error" not in result:
            def answered(d):
                for q in d.get("questions") or []:
                    if q.get("field") == field and not q.get("answered"):
                        q["answered"] = True
            self.store.update(draft_id, answered)
            self.update_plan(draft_id)
            self.refresh_workflow(draft_id)
        return result

    def built(self, draft_id: str, done: dict[str, Any]) -> None:
        """25.13: what "build it" did, in the chat: the project and its running stages, or what is missing first."""
        self.moved(draft_id, "build", {}, {k: done.get(k) for k in ("project_id", "started", "refused")})
        if done.get("rule") == "column check" and self.ask_column_check(draft_id) is not None:
            return  # H3.1: a column kept out on statistics alone is the person's to decide; the build goes on after the answer
        if done.get("refused"):  # what is missing, with the step that clears it when there is one
            actions = ["confirm_map", "show_map"] if done.get("rule") == "problem map not confirmed" else \
                ["upload", "connect", "simulate"] if "no table" in done["refused"] else []
            self.say(draft_id, done["refused"], actions=actions, rule=done.get("rule"))
            return
        if done.get("project_id"):  # H5: what the person stated becomes the project's memory
            from .. import experience
            from . import exits

            carried = exits.conversion(self.store.get(draft_id), (self.store.get(draft_id).get("understanding") or {}).get("target"))
            if carried:  # H8: what the chat's answers already showed, said where the build is said
                self.say(draft_id, carried["note"], kind="text")

            from ..ledger import capture

            before = self.store.get(draft_id).get("exit") or {}
            self.store.update(draft_id, lambda d: d.update(exit={**(d.get("exit") or {}), "kind": "ml_project", "track": "prediction",
                                                                  "status": "reached", "project": str(done["project_id"]), "at": now()}))  # H8
            if before.get("status") != "reached":
                capture.exit_reached(before.get("record"), "ml_project")

            memories.hand_off(self.store, self.store.get(draft_id), str(done["project_id"]), log=experience.current_log())
        steps = [f"{i}. {line}" for i, line in enumerate(done.get("plan") or [], 1)]
        if done.get("started"):
            text = (f"Building now. The project “{done.get('name') or 'New project'}” is created, and DCLab's agent is carrying out the plan: "
                    "it runs each stage, checks it, and reports the holdout number with its interval.")
        else:
            text = done.get("note") or "The project is built."
        text += ("\n\nThe plan:\n" + "\n".join(steps)) if steps else ""
        self.say(draft_id, text, kind="build", project_id=done.get("project_id"), session_id=done.get("session_id"))

    def ask_column_check(self, draft_id: str) -> dict[str, Any] | None:
        """H3.1: ask about the column the build is waiting on, with its evidence and a recommendation."""
        pending = self.store.get(draft_id).get("column_check_pending") or {}
        item = pending.get("item")
        if not item:
            return None
        why = "A column that is only strong may be honest; whether it is known at the prediction moment is a fact about your business."

        def waiting_on_the_build(d):  # the person asked to build past an open question: the build's question is the one answered next
            for q in d["questions"]:
                if not q.get("answered") and q.get("field") != checks.FIELD:
                    q.update(answered="(the build was asked for without it)", superseded=True)
        self.store.update(draft_id, waiting_on_the_build)
        return self.ask(draft_id, checks.FIELD, checks.question(item, str(pending.get("moment") or "")), list(checks.OPTIONS), why=why,
                        extra={"column": item["column"]})

    def column_answer(self, draft_id: str, q: dict[str, Any], value: str) -> None:
        """H3.1: the person's answer about one column (known, written later, or left to DCLab), then the build goes on."""
        answer = "recommendation" if not_known(value) else checks.read_answer(value)
        column = q.get("column")
        pending = self.store.get(draft_id).get("column_check_pending") or {}
        if pending.get("item"):  # S4: the rule's recommendation against what the person said
            from ..ledger import capture

            capture.column_checked(draft_id, pending["item"], answer,
                                   ((self.store.get(draft_id).get("analysis") or {}).get("profile") or {}))

        def put(d):
            # the answer is about this outcome and this moment: a new version with another outcome asks again
            d.setdefault("understanding", {}).setdefault("column_checks", {})[column] = {
                "answer": answer, "said": value.strip()[:200], "target": pending.get("target"), "moment": pending.get("moment")}
            d.pop("column_check_pending", None)
            for item in d["questions"]:
                if item["id"] == q["id"]:
                    item["answered"] = value.strip()[:400]
        self.store.update(draft_id, put)
        self.moved(draft_id, "column_check", {"column": column}, {"answer": answer})
        done = self.on_request(draft_id, "build", {})
        if done is not None:
            self.built(draft_id, done)

    def take_answer(self, draft_id: str, q: dict[str, Any], value: str, then_next: bool = True) -> None:
        """The user's message answers the open question: record it, redraw the workflow, ask the next thing (``then_next``:
        not when a model has already replied)."""
        if q.get("source") == "second_opinion":
            self.take_second_opinion(draft_id, q, value)
            return
        if q["field"] == checks.FIELD:
            self.column_answer(draft_id, q, value)
            return
        draft = self.store.get(draft_id)
        if q["field"] == "intent":  # H2: the kind of help, from a button or the person's words
            before = draft.get("intent") or {}
            labels = {t: intents.LABELS[t] for t in intents.TRACKS}
            self.store.update(draft_id, lambda d: [x.update(answered=value) for x in d["questions"] if x["id"] == q["id"]])
            from ..ledger import capture

            button = next((t for t, label in labels.items() if value.strip().casefold() == label.casefold()), None)
            if button is not None:  # a button is the person's choice
                capture.intent_chosen(before.get("record"), button)
                self.move_to(draft_id, intents.chosen(button, before), "undecided", sample=False)
                return
            if self.intent_client is None:  # no model: the rules read the answer's own words, a strong phrase only (the review)
                alone = intents.read(value)
                if alone["status"] != "asked" and not alone.get("fallback"):
                    self.move_to(draft_id, alone, "undecided")
                    return
            # their own words (2026-10-29): read again, the model first, with everything they have said so far
            draft = self.store.get(draft_id)
            said = plans.user_text(draft)
            reading = intents.read("\n".join(t for t in said if t), self.intent_client, said)
            turns = sum(1 for x in draft["questions"] if x.get("field") == "intent" and x.get("answered"))
            if reading.get("ask") == "either" and reading.get("source") == "model" and reading.get("track"):
                # the person answered in their own words and the model read them: their words settle it over the rules' strong
                # reading of the first sentence (the review: "Neither, I want a weekly dashboard" was thrown away)
                reading = {**reading, "status": "inferred", "ask": None}
            if reading["status"] != "asked":
                self.move_to(draft_id, reading, "undecided")
                return
            same = set((reading.get("tracks") or [])[:2]) == set((before.get("tracks") or [])[:2])
            if before.get("ask") == "either" and before.get("track") and (reading.get("ask") == "open" or same):
                # "Something else", "I'm not sure" to two clear readings: start with the first, which the card can change
                first = before["track"]
                self.say(draft_id, f"I'll start with {intents.LABELS[first].lower()}. If that is not it, change the type of help above.")
                self.move_to(draft_id, {**before, "status": "inferred"}, "undecided", sample=False)
                return
            if turns > self.UNDERSTAND_TURNS:  # HOME-015: "not sure" to every question was asked seven times; Home stops asking
                best = next((t for t in [*(reading.get("tracks") or []), *(before.get("tracks") or [])] if t in intents.TRACKS), None)
                if best:
                    self.say(draft_id, f"I'll start with {intents.LABELS[best].lower()}. If that is not it, change the type of help above.")
                    self.move_to(draft_id, {**reading, "track": best, "tracks": [best], "status": "inferred", "ask": None}, "undecided")
                else:
                    self.say(draft_id, "Whenever you are ready, pick the kind of help above, or tell me what you would like to end up with.")
                return
            if reading.get("question") == intents.OPEN_QUESTION:  # the same words twice read as not listening
                reading = {**reading, "question": intents.OPEN_AGAIN}
            self.set_intent(draft_id, reading)
            if turns >= self.UNDERSTAND_TURNS:  # still unclear after two answers: the kinds of help as buttons alone
                reading = {**reading, "ask": "open", "question": "Which of these is closest to what you need?"}
            self.ask_intent(draft_id, reading)
            return
        if q["field"] == SEGMENTS_FIELD:  # 2026-10-29: groups the person has, or new groups found by a segmentation project
            self.store.update(draft_id, lambda d: [x.update(answered=value) for x in d["questions"] if x["id"] == q["id"]])
            said = value.strip().casefold()
            if said == SEGMENTS_ROUTES[0].casefold():
                new = False
            elif said == SEGMENTS_ROUTES[1].casefold():
                new = True
            else:  # their words: "already have", "describe", "existing" keep the groups they have; finding new ones is said plainly
                new = bool(re.search(r"\b(?:find|discover|create|build)\b[^.?!]{0,30}\b(?:new\s+)?(?:groups|segments|clusters)\b|^new\b", said)) \
                    and not re.search(r"\b(?:already|existing|describe|don'?t\s+need\s+new|no\s+new|not\s+new)\b", said)
            self._tool(draft_id, "analysis", "segmentation_project" if new else "query", "person")
            if new:
                self.record(draft_id, "family", "segmentation")
                self.record(draft_id, "family_by", "answer")
                self.choose_intent(draft_id, "prediction", by_person=False)  # the project path: a segmentation kind, which reads no outcome
            elif self.ask_brief(draft_id) is None:
                self.say(draft_id, intents.NEXT["analysis"], kind="text", track="analysis")
            return
        if q.get("field") == "entity_check":  # the repeating key's question: yes groups by it, no says there is none
            self.entity_answer(draft_id, q, value)
        critic = q.get("source") == "critic"
        if critic and (CONFIRMED.match(value) or q["field"] not in (*plans.ALL_FIELDS, *self.RECORDED_AS_SAID)):
            pass  # 28.4 (the review): "yes" to a critic's question keeps what is known; a family is read below, never its words
        elif q["field"] in plans.ALL_FIELDS:
            self.note(draft_id, q["field"], value, "stated", "answer", value)
        else:
            self.record(draft_id, q["field"], value)

        def answered(d):
            for item in d["questions"]:
                if item["id"] == q["id"]:
                    item["answered"] = value
        self.store.update(draft_id, answered)
        self.read_card_answer(draft_id, q["field"], value)
        u = self.store.get(draft_id).get("understanding") or {}
        for field, quote in plans.extract(value).items():  # "…within 30 days, scored every Monday" also gives the moment
            if field != q["field"] and not u.get(field):
                self.note(draft_id, field, quote, "stated", "answer", quote)
        self.update_plan(draft_id)
        if q["field"] == "data_plan" and SIMULATE_WORDS.search(value):
            self.offer_synthetic(draft_id, draft["problem"])
        self.refresh_workflow(draft_id)
        if then_next:
            self.next_turn(draft_id)

    def _replied(self, draft_id: str, q: dict[str, Any], said: str) -> None:
        """The open question the person replied to, closed as the model moves on: a plan field their words did not settle
        is superseded, so the plan asks it again (never left closed and unknown: an entity said and never recorded was
        built without); an optional one is closed on their words; a card's question (a contradiction, the family) is
        answered by them, read for the card as any answer is (the full live rerun: a contradiction closed unread)."""
        if q["field"] in plans.ALL_FIELDS:
            def supersede(d):
                for item in d["questions"]:
                    if item["id"] == q["id"] and not item.get("answered"):
                        item.update(answered=said.strip()[:400] or "(replied)", unsettled=True, superseded=True)
            self.store.update(draft_id, supersede)
        elif q["field"] in (*OPTIONAL, "notes"):
            self.close_optional(draft_id, q, said)
        elif said.strip():
            self.record(draft_id, q["field"], said)

            def answered(d):
                for item in d["questions"]:
                    if item["id"] == q["id"] and not item.get("answered"):
                        item["answered"] = said.strip()[:400]
            self.store.update(draft_id, answered)
            self.read_card_answer(draft_id, q["field"], said)

    def close_optional(self, draft_id: str, q: dict[str, Any], value: str) -> None:
        """A question about a field no plan settles, replied to in the person's words: closed. Their words are kept only
        when they say it (costs: a miss or a false alarm named; a note: added to the notes, never over them)."""
        from dclab_core.card import costs_said

        field, value = q["field"], value.strip()
        said = costs_said(value)
        keep = (said.miss.value or said.false_alarm.value or _amounts_said({"costs": value})) if field == "costs" else bool(value)
        if keep:
            known = str((self.store.get(draft_id).get("understanding") or {}).get("notes") or "")
            self.record(draft_id, field, (f"{known}; {value}" if field == "notes" and known else value))

        def answered(d):
            for item in d["questions"]:
                if item["id"] == q["id"]:
                    item.update(answered=value[:400] or "(replied)", **({} if keep else {"unsettled": True}))
        self.store.update(draft_id, answered)
        if keep:
            self.read_card_answer(draft_id, field, value)

    READ_BY_MODEL = ("target", "outcome", "family", "labels")  # the answers a reader is asked about: they settle the card
    CARD_FIELDS = {"target": "outcome", "outcome": "outcome", "labels": "labels", "family": "family", "prediction_moment": "prediction_moment",
                   "labels_check": "labels", "target_check": "outcome"}
    LABEL_PHRASES = {"column": "A column marks them", "file": "A file lists the known ones", "none": "Nothing is labelled"}
    CONTRADICTIONS = {"labels_contradict": "labels_check", "target_absent": "target_check", "target_refused": "target_check"}
    RECORDED_AS_SAID = ("entity", "labels")  # a critic's question on these keeps the person's words, as the Home agent's own does
    HOME_FIELDS = {"outcome": "target", "labels": "labels", "family": "family", "prediction_moment": "prediction_moment", "data": "data_plan"}

    def read_card_answer(self, draft_id: str, field: str, value: str) -> None:
        """20.2: what an answer says for the problem card, by rules (``dclab_core.conversation.read_answer``): "it is
        anomaly detection", typed whatever was asked, states the family."""
        from dclab_core.conversation import read_answer

        columns = [str(c.get("name")) for c in ((self.store.get(draft_id).get("analysis") or {}).get("profile") or {}).get("columns") or []]
        read = read_answer(self.CARD_FIELDS.get(field, field), value, columns or None)
        model = self.model_read(self.CARD_FIELDS.get(field, field), value, columns) if field in self.READ_BY_MODEL else {}
        before = self.store.get(draft_id).get("understanding") or {}
        rules_family = read.get("family")
        open_family = (not before.get("family") or before.get("family_shape_only") == "yes" or before.get("family_by") == "reader")
        if model.get("family") and (field == "family" or open_family) and (not read.get("family") or read.get("family_shape_only") == "yes"):
            # never over a family the person stated earlier: only when asked for it, or while the family is still open
            # a shape the rules read gives way to the family said; the reader's own shape (a yes or no: classification) stays a
            # shape, which the person's other words and the data may still refine
            read["family"] = model["family"]
            read["family_shape_only"] = "yes" if model["family"] in SHAPES else "no"
        for name in ("labels", "target_column"):
            if model.get(name) and not read.get(name):
                read[name] = model[name]
        if model.get("labels") and field == "labels" and not label_said(value):
            self.record(draft_id, "labels", self.LABEL_PHRASES[model["labels"]])  # 20.10 v4: the reader's reading of words the rules could not
        if read.get("family"):
            self.record(draft_id, "family", read["family"])
            self.record(draft_id, "family_by", "reader" if read["family"] != rules_family else "answer")
            self.record(draft_id, "family_shape_only", read.get("family_shape_only") or "no")  # 20.10 v2: a shape alone may be refined
        named = read.get("target_column")
        if field in ("target", "outcome") and named and value.strip() not in columns and \
                re.match(rf"\s*(the\s+)?((column|duration|outcome|target|label|flag|field)\s+){{0,2}}{re.escape(named)}\b", value, re.I) and \
                not LATER.search(re.split(r"[,;.]", value[value.find(named):] + ",")[0]):
            # 20.10: "days_subscribed, with cancelled saying whether they left" begins with its outcome; "a failure of the
            # machine" or "we only know it once final_refund_amount is paid" only mention a column
            self.record(draft_id, "target", named)
        elif field in ("target", "outcome") and (self.store.get(draft_id).get("understanding") or {}).get("family") in NO_OUTCOME_COLUMN:
            pass  # a ranking, groups, unusual rows: the reader's column is not an outcome these read (the Solution page refuses one)
        elif field in ("target", "outcome") and model.get("target_column") in columns and \
                model["target_column"] in (named, None if named else model["target_column"]) and \
                not LATER.search(clause_of(value, model["target_column"])):
            # 20.10 v4: the reader quoted the column as the outcome; one only mentioned as known later is never the outcome
            self.record(draft_id, "target", model["target_column"])
        if field in ("labels_check", "target_check"):  # the answer to a contradiction settles what it was about
            if read.get("labels"):
                self.record(draft_id, "labels", self.LABEL_PHRASES[read["labels"]])
                if read["labels"] in ("file", "column") and (self.store.get(draft_id).get("understanding") or {}).get("family") == "anomaly" \
                        and not read.get("family"):
                    self.record(draft_id, "family", "rare_event")  # 21.5: "anomaly", then "these are the cases we found": a rare event
            if read.get("target_column"):
                self.record(draft_id, "target", read["target_column"])

    # ------------------------------------------------------------------ 28.6: the second opinion on the family
    def _opinion(self, draft: dict[str, Any]) -> Any:
        """The judge on the two readings of the family: the conversation reader's (kept in the understanding when it set the
        family) and the card rules' from the words and the data alone."""
        from dclab_core import second_opinion as so

        from .card import card_of

        u = draft.get("understanding") or {}
        reader = u.get("family") if u.get("family_by") == "reader" else None
        rules = card_of({**draft, "understanding": {k: v for k, v in u.items() if k not in ("family", "family_by", "family_shape_only")}}).family.value
        return so.judge_family(reader, rules)

    def second_opinion_question(self, draft: dict[str, Any]) -> tuple[str, str, list[str], str, str, dict[str, Any]] | None:
        """The second opinion's question (field, text, options, why, source, and its verdict to keep on the question), once
        a draft: only where the reader and the rules both named a family and the two differ."""
        from dclab_core import second_opinion as so

        if any(q.get("source") == "second_opinion" or q.get("field") == "family" for q in draft.get("questions") or []):
            return None  # once a draft, and never after any family question (answered, declined or open): nothing is asked twice
        if (draft.get("solution") or {}).get("kind"):
            return None  # a kind the person stated on the Solution page is theirs: never put to them again (the review of 28.6)
        opinion = self._opinion(draft)
        if opinion.verdict != "ask":
            return None
        return ("family", opinion.question, [so.option_label(c) for c in opinion.choices], opinion.why, "second_opinion",
                {"opinion": json.loads(opinion.model_dump_json())})

    def take_second_opinion(self, draft_id: str, q: dict[str, Any], value: str, then_next: bool = True) -> None:
        """The person's answer to the second opinion, read by code (``second_opinion.read_choice``) between the choices they
        were shown: one named, the words of one meaning, or a family the rules read as stated (never a shape). Their choice
        is the family they stated; any other answer keeps the card. Either way recorded, with the verdict they were shown."""
        from dclab_core import second_opinion as so
        from dclab_core.conversation import read_answer

        draft = self.store.get(draft_id)
        opinion = so.Opinion.model_validate(q["opinion"]) if q.get("opinion") else self._opinion(draft)
        choices = so.choices_of(q.get("options") or []) or list(opinion.choices)  # what the person was shown (the review: never recomputed)
        columns = [str(c.get("name")) for c in ((draft.get("analysis") or {}).get("profile") or {}).get("columns") or []]
        read = read_answer("family", value, columns or None) if value.strip() else {}
        stated = read.get("family") if read.get("family") and read.get("family_shape_only") != "yes" else None
        choice = so.read_choice(value, choices, stated) if value.strip() else None

        def answered(d: dict[str, Any]) -> None:
            for item in d["questions"]:
                if item["id"] == q["id"]:
                    item["answered"] = value or "(no answer)"
                    if choice is None:
                        item["declined"] = True
        self.store.update(draft_id, answered)
        if choice is not None:
            self.record(draft_id, "family", choice)
            self.record(draft_id, "family_by", "person")
            self.record(draft_id, "family_shape_only", "no")  # a choice between two families is a family stated, never a shape
        from ..ledger import capture
        from .card import card_of

        stands = card_of(self.store.get(draft_id)).family.value  # the review of 28.6: on a decline the card that stands is recorded
        capture.second_opinion(draft_id, opinion.model_copy(update={"choices": choices}), choice, stands)
        self.update_plan(draft_id)
        self.refresh_workflow(draft_id)
        if then_next:
            self.next_turn(draft_id)

    def decline(self, draft_id: str, q: dict[str, Any]) -> None:
        """An advisory question the person did not answer (the second opinion): the card stays as it is, never asked again."""
        if q.get("source") == "second_opinion":
            self.take_second_opinion(draft_id, q, "")
        elif q.get("source") == "advisory":  # the repeating key unanswered: not grouped, the risk said in the summary

            def closed(d: dict[str, Any]) -> None:
                for item in d["questions"]:
                    if item["id"] == q["id"]:
                        item["answered"], item["declined"] = "(no answer)", True
            self.store.update(draft_id, closed)
            self.next_turn(draft_id)

    def card_question(self, draft: dict[str, Any]) -> tuple[str, str, list[str], str] | None:
        """20.2: when the plan has nothing left to ask but the problem card still blocks, its question that unblocks the
        most (a contradiction between the words and the data first), never one asked before."""
        from dclab_core.card import LABELS_OPTIONS
        from dclab_core.conversation import next_question, outcome_again

        from .card import card_of, meeting

        if not (draft.get("analysis") or {}).get("profile"):  # with no table the card's questions cannot be answered yet
            return None
        asked_home = {q.get("field") for q in draft["questions"]}
        met = meeting(draft, self.store)  # 20.4's mismatches: a contradiction is shown before anything else
        for m in (met.mismatches if met is not None else []):
            field = self.CONTRADICTIONS.get(m.code)
            if field and field not in asked_home:
                return (field, m.question, list(LABELS_OPTIONS) if field == "labels_check" else [], "The words and the data disagree.")
        card = card_of(draft)
        if card.ready:
            return None
        asked = tuple(self.CARD_FIELDS.get(q.get("field"), q.get("field")) for q in draft["questions"])
        q = next_question(card, asked=asked)
        if getattr(q, "field", None) not in ("labels", "family", "outcome"):
            # 28.7: an outcome answered with columns named but none settled is asked once more, with those columns
            columns = [str(c.get("name")) for c in (draft["analysis"]["profile"].get("columns") or [])]
            q = outcome_again(card, [str(x.get("answered")) for x in draft["questions"] if x.get("field") in ("target", "outcome")
                                     and isinstance(x.get("answered"), str)], columns, asked=tuple(asked_home))
            if q is not None:
                return (q.field, q.text, list(q.options), q.why)
        if q is None or getattr(q, "field", None) not in ("labels", "family", "outcome"):  # the plan asks the rest its own way
            return None
        return (self.HOME_FIELDS.get(q.field, q.field), q.text, list(q.options), q.why)

    def kind_proposal(self, draft: dict[str, Any]) -> tuple[Any, Any] | None:
        """S9.3: the kind the card's family means, its fields read from the draft's table and from what the person said
        to its questions; None when the family is a plain task's or the table is not ready."""
        from ..kinds.propose import FAMILY_KINDS, for_family, said_of
        from .card import card_of

        if not (draft.get("analysis") or {}).get("profile"):
            return None
        family = card_of(draft).family.value
        if str(family or "") not in FAMILY_KINDS:
            return None
        frame = draft_table(self.store, draft)
        if frame is None:
            return None
        u = draft.get("understanding") or {}
        target = u.get("target") if u.get("target") in frame.columns else None
        return for_family(family, frame, target, said_of(u), tables=()), frame  # a draft attaches no related table by name

    def kind_question(self, draft: dict[str, Any]) -> tuple[str, str, list[str], str, str | None, None] | None:
        """S9.3: the first field of the understood kind that only the person can give, asked once in the kind's words
        (whether an offer was random, which column names a series); None when the table shows them all."""
        from ..kinds.propose import options_for, question_for

        if self.track(draft) != "prediction":
            return None  # H2: a kind of model is a prediction's
        if any(x.get("source") == "second_opinion" and (x.get("declined") or not x.get("answered")) for x in draft.get("questions") or []):
            return None  # the family itself is in dispute (two readings, the person did not choose): its kind's questions would presume it
        got = self.kind_proposal(draft)
        if got is None or got[0] is None:
            return None
        p, frame = got
        q = question_for(p, {str(x.get("field")) for x in draft.get("questions") or []})
        u = draft.get("understanding") or {}
        if q is None:
            return None
        options = options_for(q, frame, p, u.get("target") if u.get("target") in frame.columns else None)
        # a field that only helps (a forecast's known-ahead columns) is advisory: no answer closes it, a walk goes on
        return (q.key, q.text, options, q.why, "advisory" if q.field in p.optional else None, None)

    ENTITY_NO = re.compile(r"\b(no|not|independent|never|separate)\b", re.I)
    ENTITY_YES = re.compile(r"^\W*(yes|yeah|yep|yup|sure|ok(ay)?|keep|group|one side)\b", re.I)

    def entity_question(self, draft: dict[str, Any]) -> tuple[str, str, list[str], str, None, dict[str, Any]] | None:
        """The owner, 2026-10-26: a key that repeats (biker_id: many rows for each biker) while no entity was said is
        asked, with its evidence, never grouped silently: if one biker's rows are on both sides of the split, the score
        is too good. Once at most; a row's own key (unique on every row) is never one."""
        u = draft.get("understanding") or {}
        if self.track(draft) != "prediction" or u.get("entity") or any(q.get("field") == "entity_check" for q in draft.get("questions") or []):
            return None  # H2: grouping the folds is a prediction's
        from ..kinds.propose import FAMILY_KINDS
        from .card import card_of

        if str(card_of(draft).family.value or "") in FAMILY_KINDS:  # a kind with its own structure (a forecast's series, a ranking's
            return None  # items, a stream's units) splits its own way: its keys are fields, never an entity to keep on one side
        profile = (draft.get("analysis") or {}).get("profile") or {}
        rows = int(profile.get("rows") or 0)
        found = [c for c in profile.get("columns") or [] if c.get("name_id_like") and not c.get("id_like") and c.get("name") != u.get("target")
                 and rows and 10 <= int(c.get("unique") or 0) <= rows // 2]
        if not found:
            return None
        c = max(found, key=lambda c: rows / int(c["unique"]))
        name, unique = str(c["name"]), int(c["unique"])
        text = (f"`{name}` repeats: {rows:,} rows for {unique:,} different values, about {rows // unique} rows each. If rows of one "
                f"`{name}` are both in the rows a model learns from and in the rows it is tested on, its score looks better than it "
                f"will be. Should each `{name}` stay on one side?")
        return ("entity_check", text, ["Yes, keep each one on one side", "No, its rows are independent", "I don't know"],
                "Grouping by who the rows belong to keeps the score honest: the folds and the holdout split by it.", "advisory", {"column": name})

    def entity_answer(self, draft_id: str, q: dict[str, Any], value: str) -> None:
        """Yes groups by the column (the entity said); no records that there is none; anything else changes nothing."""
        if not_known(value) or not q.get("column"):
            return
        if self.ENTITY_NO.search(value) and not self.ENTITY_YES.match(value):
            self.record(draft_id, "entity", "No single entity")
        elif self.ENTITY_YES.match(value):
            self.record(draft_id, "entity", str(q["column"]))

    def kind_said(self, draft: dict[str, Any]) -> str | None:
        """The summary's line on the kind: what it reads and assumes, or what it still needs and what runs meanwhile."""
        from .. import kinds

        got = self.kind_proposal(draft)
        if got is None or got[0] is None:
            return None
        from ..kinds.propose import still_needed

        p = got[0]
        title = kinds.get(p.kind).title
        assumed = "; ".join(str(v) for v in p.assumed.values())
        if p.complete:
            return f"Kind: {title}." + (f" {assumed[0].upper() + assumed[1:]}." if assumed else "")
        needed = still_needed(p) or [f"{', '.join(m.replace('_', ' ') for m in p.missing)}, on the Solution page"]
        return f"Kind suggested: {title}; it still needs {'; '.join(needed)}."

    # ------------------------------------------------------------------ the deterministic script
    def script_turn(self, draft_id: str, text: str) -> None:
        """The Home agent without a model: a run of the scripted agent in the harness (21.1), each move a step."""
        with harness.session("home.script", trace=harness.tracer("home.script", draft_id)) as current:
            token = _SCRIPT.set((draft_id, current))  # per thread and per draft: one agent object serves every draft
            try:
                self._script_turn(draft_id, text)
            finally:
                _SCRIPT.reset(token)

    def _script_turn(self, draft_id: str, text: str) -> None:
        draft = self.store.get(draft_id)
        q = self.pending(draft)
        if q is not None:
            self.moved(draft_id, "take_answer", {"field": q.get("field")})
            self.take_answer(draft_id, q, text)
            return
        if SIMULATE_WORDS.search(text):
            self.moved(draft_id, "simulate", {}, {"asked_how": True})
            self.offer_synthetic(draft_id, draft["problem"] + " " + text)
            return
        if BUILD_WORDS.search(text):
            if self.track(draft) != "prediction":  # H2: only a prediction builds a solution
                self.say(draft_id, "Building a solution is for predictions. If you need a model, change the type of help above.")
                return
            self.moved(draft_id, "summarize", {})
            self.summarize(draft_id)
            return
        self.record(draft_id, "notes", ((draft.get("understanding") or {}).get("notes", "") + " " + text).strip())
        self.moved(draft_id, "record", {"field": "notes"})
        self.say(draft_id, "Noted. I added it to the solution notes." if self.track(draft) == "prediction" else "Noted.")
        self.next_turn(draft_id)

    def moved(self, draft_id: str, tool: str, arguments: dict[str, Any], result: Any = None) -> None:
        """A move of the scripted agent (21.1): a step of its run, or of the draft's conversation between runs (the
        question asked when the data is ready). Field names only: what the person wrote is in the chat events."""
        result = {"ok": True} if result is None else result
        running = _SCRIPT.get()
        if running is not None and running[0] == draft_id:
            running[1].step(tool, arguments, result)
        else:
            harness.step("home.script", draft_id, tool, arguments, result)

    def acted(self, draft_id: str, name: str) -> dict[str, Any] | None:
        """36.2: a result produced (answer, table, brief, dashboard) or the person's act on it (save, export, notebook, a yes
        or no on a step, undo, edit, change) joins the draft's ordered actions, names only and capped. The verdict read from
        them (dclab_core.track_outcome) is kept on the draft and, when it changes, recorded through capture.track_outcome: an
        event, and the person's label on the draft's route while they keep or change the result."""
        from dclab_core import track_outcome as T

        if name not in T.PRODUCED and name not in T.DECISIVE:
            return None
        before: dict[str, Any] = {}

        def put(d):
            before.update(outcome=d.get("outcome"))
            d["actions_total"] = int(d.get("actions_total", len(d.get("actions") or []))) + 1
            d["actions"] = [*(d.get("actions") or []), name][-200:]
        self.store.update(draft_id, put)
        draft = self.store.get(draft_id)
        track = str((draft.get("intent") or {}).get("track") or "prediction")
        actions = list(draft.get("actions") or [])
        if draft.get("track_actions_from") is not None:  # the review (2026-10-29): only what was done on this track judges it
            since = int(draft.get("actions_total") or 0) - int(draft["track_actions_from"])
            actions = actions[-since:] if since > 0 else []
        outcome = T.read(track, actions)
        if outcome is None:
            return None
        now_ = outcome.model_dump(mode="json")
        self.store.update(draft_id, lambda d: d.update(outcome=now_))
        key = ("verdict", "by", "produced")
        if {k: now_.get(k) for k in key} == {k: (before.get("outcome") or {}).get(k) for k in key}:
            return now_  # the same verdict on the same results: recorded already
        from ..ledger import capture

        exit_ = draft.get("exit") or {}
        hint, tool = draft.get("suggestion_record") or {}, draft.get("tool") or {}
        more = []
        if hint.get("record") and hint.get("on") == outcome.track:  # kept their own track to a result: the suggestion was not needed
            more.append((hint["record"], "request.suggestion", "none"))
        if tool.get("record") and tool.get("by") == "rule" and tool.get("track") == outcome.track:
            more.append((tool["record"], "request.tool", tool.get("tool")))
        capture.track_outcome(draft_id, outcome, exit_record=exit_.get("record"), exit_kind=exit_.get("kind"),
                              intent_record=(draft.get("intent") or {}).get("record"), more=tuple(more))
        return now_

    def next_turn(self, draft_id: str, opening: bool = False) -> None:
        """Ask the next missing thing, or summarise when everything needed is known."""
        draft = self.store.get(draft_id)
        if self.pending(draft) or self.track(draft) != "prediction":  # H2: the questions below are a prediction's
            return
        self.update_plan(draft_id)
        question = self.next_question(self.store.get(draft_id), opening)
        if question is None and not opening:  # the plan is done; the problem card may still block (20.2)
            question = self.card_question(self.store.get(draft_id))
        if question is None and not opening and self.second_opinion:  # 28.6: two readings of the family that differ are asked
            question = self.second_opinion_question(self.store.get(draft_id))
        if question is None and not opening:  # S9.3: a field of the understood kind only the person can give
            question = self.kind_question(self.store.get(draft_id))
        if question is None and not opening:  # the owner, 2026-10-26: a repeating key, asked with its evidence
            question = self.entity_question(self.store.get(draft_id))
        asked = question is not None and self.ask(draft_id, *question)
        from ..ledger import capture  # 19.7: the question actually asked, or none

        capture.next_question(self.store.get(draft_id), question if asked else None)
        if asked:
            self.moved(draft_id, "ask_user", {"field": question[0]})
            return
        if not opening:
            self.moved(draft_id, "summarize", {})
            self.summarize(draft_id)

    def next_question(self, draft: dict[str, Any], opening: bool = False) -> tuple[str, str, list[str], str] | None:
        """The question the plan asks next (field, text, options, why), or None when nothing needs asking. Only an
        unknown field is asked, in the order of plans.FIELDS (the moment before the cost); an outcome known in words
        is asked again only for its column, once a table with outcome-like columns is there."""
        if self.track(draft) != "prediction":
            return None  # H2: the outcome, moment and action belong to a prediction
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
        outcome_raised = "target" in asked or bool(u.get("target"))  # the outcome comes first, the labels right after it
        if outcome_raised and "labels" not in asked and not u.get("labels") and not has_labels:  # 20.1: whatever the pack
            from .card import card_of
            from dclab_core.card import LABELS_OPTIONS, LABELS_QUESTION

            if set(card_of(draft).in_play) == {"rare_event", "anomaly"}:  # 20.10: another pair is asked as the family
                return ("labels", LABELS_QUESTION, list(LABELS_OPTIONS),
                        "Confirmed cases mean learning from them (rare events); none means finding the unusual rows (anomalies).")
        for field in plans.fields_for(draft):
            if field == "target" and plan[field]["status"] != "unknown" and candidates and columns and u.get("target") not in columns \
                    and not column_asked and not has_labels:
                # the outcome is known in words, but the table needs its column: "cancel next month" is which one?
                return ("target", f"You want to predict “{u['target']}”. Which column records that outcome?",
                        candidates[:OPTION_COUNT] + ["Something else"], "The model learns from the column, so it must be the outcome you described.")
            if plan[field]["status"] != "unknown" or field in asked:
                continue
            if field == "target" and u.get("family") == "segmentation":
                continue  # 2026-10-29: groups found in the data have no outcome to ask for (the moment and the action still are)
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
                # 20.1: "nothing is labelled" is an answer too: it makes the problem an anomaly search, not rare events
                return ("labels", "Are some cases already confirmed? If so, are they every case there is, or a sample of the ones you found?",
                        ["A sample of the cases we found", "Every case is labelled", "Nothing is labelled"], plans.WHY["labels"])
            if field == "budget":
                return ("budget", "How many alerts can the team review, for example a day?",
                        ["About 50 a day", "About 200 a day", "About 1% of rows", "No fixed budget"], plans.WHY["budget"])
            if field == "data_plan":
                return ("data_plan", "Can you share a sample of the data, connect a source, or should I simulate data from this conversation?",
                        ["Upload a sample", "Connect a source", "Simulate data", "No data yet, plan only"], plans.WHY["data_plan"])
        if (u.get("action") and (plan.get("action") or {}).get("source") == "problem" and "action" not in asked and "costs" not in asked
                and not u.get("costs") and len(counted(draft)) < MAX_QUESTIONS and not _amounts_said(u)):
            # 34.4: what is done came from the first sentence, so the question that asks what a wrong one costs was never put:
            # its amounts are asked once, last, never counted against the five
            return ("costs", COSTS_QUESTION, [], "With the amounts, the final stage chooses on how many rows acting pays, with its interval.")
        return None

    def replied_to(self, draft: dict[str, Any], field: str) -> dict[str, Any] | None:
        """An open question about ``field`` (not only the first open one) the person has written after."""
        messages = draft.get("messages") or []
        for q in draft.get("questions") or []:
            if q.get("field") != field or q.get("answered"):
                continue
            at = next((i for i, m in enumerate(messages) if m.get("kind") == "question" and (m.get("question") or {}).get("id") == q["id"]), None)
            if at is not None and any(m.get("role") == "user" for m in messages[at + 1:]):
                return q
        return None

    @staticmethod
    def edited(draft: dict[str, Any], field: str, value: Any) -> bool:
        """The person's own correction on the page: their last message an edit of this field with these words."""
        last = next((m for m in reversed(draft.get("messages") or []) if m.get("role") == "user"), None) or {}
        return last.get("kind") == "edit" and last.get("field") == field and str(last.get("text") or "").strip() == str(value or "").strip()

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
        if self.track(draft) != "prediction":  # H2: the solution summary is a prediction's
            return
        u = draft.get("understanding") or {}
        name = (packs.by_key((draft.get("pack") or {}).get("key")) or {}).get("name", "Tabular")
        line = lambda label, value: f"{label}: {str(value).strip().rstrip('.')}." if value else None  # noqa: E731 — answers are quoted as given
        parts = [line("Outcome", u.get("target")) or "The outcome is still open.", line("Prediction moment", u.get("prediction_moment")),
                 line("What happens with each prediction", u.get("action"))]
        parts.append(self.kind_said(draft))  # S9.3: the kind understood, its assumptions, or what it still needs
        checked = next((q for q in draft.get("questions") or [] if q.get("field") == "entity_check" and q.get("answered")), None)
        if checked and not u.get("entity"):  # asked and not settled: said as a risk, never grouped silently
            parts.append(f"Not grouped by `{checked.get('column')}`: if its rows are not independent, the score may look better than it will be.")
        ready = any(a.get("status") == "ready" for a in draft["assets"])
        text = ("Here is the solution I would build. " + " ".join(p for p in parts if p) + f" Pack: {name}. "
                + ("The data is ready, so we can build the solution now." if ready else
                   "We can build the solution plan now and add data in the next step."))
        last = next((m for m in reversed(draft["messages"]) if m["role"] == "agent"), None)
        if last and last.get("kind") == "summary" and last.get("text") == text:
            return  # two turns can end here (the data-ready turn and the reply that started a simulation): say it once
        from dclab_core.conversation import summary

        from .card import card_of

        self.say(draft_id, text, kind="summary", actions=["build"], card=summary(card_of(draft)))  # 20.2: the card to confirm

    # ------------------------------------------------------------------ the model
    def context(self, draft: dict[str, Any]) -> str:
        a = draft.get("analysis") or {}
        cols = [{k: c.get(k) for k in PROFILE_FIELDS} for c in (a.get("columns") or [])[:PROFILE_PAGE]]
        return json.dumps({
            "problem": draft["problem"], "pack": draft.get("pack"), "understanding": draft.get("understanding") or {},
            "intent": {**{k: (draft.get("intent") or {}).get(k) for k in ("tracks", "status", "source")}, "track": self.track(draft)},  # H2
            "next_step": intents.NEXT.get(self.track(draft)),
            "brief": (draft.get("track_brief") or {}).get(self.track(draft)),  # H4: what is known for this kind of help, with its sources
            "conversation": memories.summary(draft),  # H5: everything the conversation settled, however long it grew
            "plan": {**{f: ((draft.get("plan") or {}).get(f) or {}).get("status", "unknown") for f in plans.fields_for(draft)}, "next": (draft.get("plan") or {}).get("next")},
            "questions_asked": [{"field": q.get("field"), "text": q.get("model_text") or q.get("text"), "answered": q.get("answered")}
                                for q in draft["questions"]],  # H9b: a guard question as the model may read it, without counts
            "questions_left": MAX_QUESTIONS - len(counted(draft)),
            "data": {"summary": a.get("summary"), "columns": cols, "highlights": [h.get("title") for h in (a.get("highlights") or [])][:8],
                     "target_candidates": (a.get("profile") or {}).get("target_candidates")} if a else None,
            "assets": [{k: x.get(k) for k in ("name", "kind", "status", "synthetic")} for x in draft["assets"]],
            "workflow": [{k: n.get(k) for k in ("wf", "label")} for n in ((draft.get("workflow") or {}).get("nodes") or [])],
        }, ensure_ascii=False, default=str)

    def policy(self, max_replies: int = MODEL_REPLIES) -> Policy:
        """The Home agent as a policy on the runtime: a few model replies per user message; a question waits for the
        user and a simulation ends the turn too (the data pipeline speaks next)."""
        return Policy("home", POLICY, tuple(self.registry.names("draft")), max_steps=MODEL_STEPS, max_seconds=MODEL_SECONDS,
                      stop_after=("ask_user", "build_solution"), stop_when=lambda step: isinstance(step.result, dict) and bool(step.result.get("simulating") or step.result.get("asked_how")),
                      max_replies=max_replies)

    def tracer(self, draft_id: str) -> Tracer | None:
        # without a trace store the steps still reach the session's record (19.3): the tracer writes no row, only its event
        return Tracer.resume(self.traces, draft_id, "home", lambda: understood(self.store.get(draft_id)))

    def _to_model(self, draft_id: str) -> dict[str, Any] | None:
        """36.3: the person's first request, which the rules could not read, given to the model with its tools. Returns the
        model's newest message when its turn said something; None with no model, inside a model turn already, when the
        model fails, or when the turn said nothing, so the rules then say what they need (the review of 36.3)."""
        if self.client is None or _MODEL_TURN.get():
            return None
        before = len(self.store.get(draft_id).get("messages") or [])
        try:
            self.model_turn(draft_id)
        except Exception:  # noqa: BLE001 - the model unavailable: the rules' own words follow
            return None
        new = [m for m in (self.store.get(draft_id).get("messages") or [])[before:] if m.get("role") == "agent"]
        return new[-1] if new else None

    def model_turn(self, draft_id: str, max_replies: int = MODEL_REPLIES, text: str = "") -> None:
        draft = self.store.get(draft_id)
        open_question = self.pending(draft)
        history, since = _closed(memories.recent(draft), draft.get("problem") or "")  # H5: the last messages; the summary is in the state
        if not history:
            return  # nothing the person said to answer: the rules speak
        messages = [{"role": "system", "content": POLICY}, {"role": "system", "content": "Current state: " + self.context(draft)}, *history]
        if since:
            messages.insert(2, {"role": "system", "content": "What Home has said since the person's last message: " + since})

        def on_step(step: Step) -> None:
            # A proposal the guard or the workflow check saw reports its own verdict; one the schema refused does not.
            own = step.tool == "propose_workflow" and isinstance(step.result, dict) and ("accepted" in step.result or step.ok)
            if not own and getattr(self.client, "output", None):
                self.client.output(step.ok, "" if step.ok else "the tool call was refused")

        model = _Model(self, draft_id)
        self.store.emit(draft_id, "status", {"thinking": True})
        running = _MODEL_TURN.set(True)  # 36.3: the model's tools run here; none of them starts another model turn
        try:
            out = run_policy(self.policy(max_replies), model, self.registry, messages, context=Turn(self, draft_id), on_step=on_step,
                             trace=self.tracer(draft_id))
        finally:
            _MODEL_TURN.reset(running)
            self.store.emit(draft_id, "status", {"thinking": False})
        if out.stopped == "answered" and out.final:
            final = self.checked_reply(self.store.get(draft_id), out.final)
            self.say(draft_id, final, message_id=model.live, by="model" if final == out.final else "code")  # the chat event replaces the live bubble; by: a model wrote it (23.10 joins it to its request)
        elif out.stopped == "tools_as_text" and model.live:
            self.store.emit(draft_id, "token", {"id": model.live, "drop": True})  # tool calls written as text are not a reply
        self.store.update(draft_id, lambda d: d["agent"].update(mode="model", turns=d["agent"].get("turns", 0) + 1))
        after = self.store.get(draft_id)
        still = self.pending(after)
        acted = {step.tool for step in out.steps} & {"simulate_data", "build_solution"}  # their words were an order, not an answer
        if text and open_question and still and still["id"] == open_question["id"] and open_question["field"] in (*OPTIONAL, "notes") \
                and "?" not in text and not acted and open_question.get("source") not in ("critic", "second_opinion"):
            # live (HRN-022): a question about a field no plan settles (costs, a note), answered in the person's own words
            # the model did not record, stayed open and was put to them again and again, and the build waited on it: their
            # words are its answer (the card reads them), and the conversation goes on
            self.close_optional(draft_id, open_question, text)
            after = self.store.get(draft_id)
        elif text and open_question and still and still["id"] == open_question["id"] and open_question["field"] not in plans.ALL_FIELDS \
                and "?" not in text and not acted and open_question.get("source") not in ("critic", "second_opinion") \
                and after["messages"] and after["messages"][-1].get("role") != "user":
            # the full live rerun (JD-06): "negative" answered the card's "Which column is the outcome itself: negative?"; the
            # model said "Confirmed" and recorded nothing, and the walk asked it three more times: a card's question is read
            # from the person's words, as the rules read it (only what the rules parse is recorded)
            self.take_answer(draft_id, open_question, text, then_next=False)
            after = self.store.get(draft_id)
        if out.steps and text and after["messages"] and after["messages"][-1].get("role") == "user":
            # Live (2026-10-16): the model re-asked the open question (a no-op) and the turn ended in silence, three
            # messages in a row. The person always gets a reply: the open question again, or the plan's next step.
            still_open = self.pending(after)
            if still_open and "?" not in text and not acted and open_question and still_open["id"] == open_question["id"] \
                    and still_open.get("source") not in ("critic", "second_opinion"):
                # live (HRN-022, RG-02): "A sample of the cases we found" answered the labels question; the model's turn ended
                # without recording it, the question was put to them again, word for word: their words are its answer
                self.take_answer(draft_id, still_open, text)
            elif still_open:
                if "?" in text:  # the person asked something ("what is the plan?"): the plan so far first, then the question
                    self.summarize(draft_id)
                self.say(draft_id, still_open["text"], kind="question", question=still_open)
            else:
                self.next_turn(draft_id)
            return
        asks = re.findall(r"[^.?!]*\?", str(out.final or ""))
        if out.steps and out.stopped == "answered" and not self.pending(after) and not after.get("project_id") and not acted \
                and all(LEAVE_TO_BUILD.search(a) for a in asks):  # nothing asked in prose, or only leave to build
            # live (HRN-022): the model ended its turn (or offered to build) with one of the plan's own fields still open, and
            # a fraud walk was built without its entity, its folds never grouped: the plan's next open field is asked, as the
            # rules ask it (the model chose the order and the words before; the questions that change the solution are asked)
            question = self.next_question(after) or self.kind_question(after) or self.entity_question(after)  # S9.3; a repeating key
            if question is not None and self.ask(draft_id, *question):
                from ..ledger import capture

                capture.next_question(self.store.get(draft_id), question)
                self.moved(draft_id, "ask_user", {"field": question[0]})
            return
        if not out.steps and text:
            # A model that only talks (small models often never call a tool) must not stall the conversation:
            # code records the answer to the open question and asks the next thing, as the script would.
            still_open = self.pending(self.store.get(draft_id))
            if open_question and still_open and still_open["id"] == open_question["id"]:
                self.take_answer(draft_id, open_question, text)
            elif SIMULATE_WORDS.search(text) and not self.store.get(draft_id)["assets"]:
                self.offer_synthetic(draft_id, draft["problem"] + " " + text)
            else:
                self.next_turn(draft_id)

    def checked_reply(self, draft: dict[str, Any], text: str) -> str:
        """Every number in a model's reply must be in the person's words, the table's analysis or the product's own example
        questions (H2, HOME-003: "a churn rate of 2.5%"; HOME-013: a prediction's invented examples). Off a prediction the
        code's own line replaces the reply; elsewhere only the sentences whose numbers someone gave are kept."""
        track = self.track(draft)
        from .. import cited

        allowed = set().union(*(cited.numbers(t, source=True) for t in plans.user_text(draft)), set())
        allowed |= cited.numbers(json.dumps([draft.get(k) for k in ("analysis", "answers", "assets", "recipe", "business", "dashboard",
                                                                 "cleaning_log")], default=str), source=True)
        # the product's own example questions (a pack's opening, the cost question, the tracks' next steps) may be echoed
        allowed |= cited.numbers(" ".join([*packs.OPENING.values(), COSTS_QUESTION, *intents.NEXT.values()]), source=True)
        if not (cited.numbers(text) - allowed):
            return text
        if track in intents.NEXT:
            return f"I read this as {intents.LABELS[track].lower()}. {intents.NEXT[track]}"
        # HOME-013 (2026-10-29): on a prediction the model's own example numbers ("say 1000 customers") reached the person;
        # every track is checked now, and here only the sentences whose numbers someone gave are kept
        kept = [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s and not (cited.numbers(s) - allowed)]
        return " ".join(kept) if kept else "Tell me more about what you need, in your own words."

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
        # H1: the same schema check as the model's calls; code that sends a wrong type gets an error, never a silent coercion
        result = self.registry.call(tool.name, known, Turn(self, draft_id), scope="draft", check="schema")
        self.moved(draft_id, tool.name, known, result)  # 21.1: code's call is a step of the scripted agent
        return result


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
    if msg:
        return {"asked": True}
    known = (turn.agent.store.get(turn.draft_id).get("understanding") or {}).get(field)
    return {"asked": False, "note": (f"Not asked: the user already said {field}: \"{str(known)[:160]}\". Use it; ask about something still open, or answer them."
                                     if known else "Not asked: the question limit is reached, the same question is open, or a table is already here.")}


def record(turn: Turn, /, **args: Any) -> dict[str, Any]:
    agent, draft_id = turn.agent, turn.draft_id
    field, value = str(args.get("field", "notes")), str(args.get("value", ""))
    pending = agent.answerable(agent.store.get(draft_id))
    if not (pending and pending["field"] == field):  # the review: two questions asked in one reply, the second answered first
        pending = agent.replied_to(agent.store.get(draft_id), field) or pending
    if pending and pending["field"] == field:  # the user's message answers the open question
        if field in WORDS_KEPT:  # live (HRN-022): "About 80 a day" recorded as "About 80 bikers reviewed per daily audit": no budget read
            value = next((str(m.get("text") or "") for m in reversed(agent.store.get(draft_id)["messages"]) if m.get("role") == "user"), "") or value
        said = next((m.get("text") for m in reversed(agent.store.get(draft_id)["messages"]) if m.get("role") == "user"), "")
        if not_known(said):  # HOME-003: the person said they do not know, and the model recorded its own answer as theirs
            agent.record(draft_id, field, str(said))  # closed as declined, the field unknown (the moment assumed and said)
            return {"recorded": field, "not_known": True}
        if field in plans.ALL_FIELDS:
            agent.note(draft_id, field, value, "stated", "answer", str(said or value))  # the quote is the person's reply, never the model's value
        else:
            agent.record(draft_id, field, value)
        agent.store.update(draft_id, lambda d: [q.update(answered=value) for q in d["questions"] if q["id"] == pending["id"]])
        agent.read_card_answer(draft_id, field, str(said or ""))  # 20.2: the person's own words, read for the card
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
    """H2: the model's reading of the kind of help, backed by the person's words (the guard checked the quote)."""
    agent, draft_id = turn.agent, turn.draft_id
    draft = agent.store.get(draft_id)
    before = draft.get("intent") or {}
    track = args["track"]
    if track == agent.track(draft):
        return {"intent": track, "unchanged": True}  # nothing new: no reading recorded, no move
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


def Q_result(doc: dict[str, Any]) -> Any:
    from dclab_core import query as Q

    return Q.QueryResult.model_validate(doc)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def query_data(turn: Turn, /, **args: Any) -> dict[str, Any]:
    """H7c: the model's query plan for the person's question, checked and answered by code. The person gets the numbers,
    written by code; the model gets only that an answer was given (no table value reaches it)."""
    from .. import home_query
    from dclab_core import query as Q

    agent, draft_id = turn.agent, turn.draft_id
    try:
        plan = Q.QueryPlan.model_validate(args.get("plan") or {})
    except Exception as error:  # noqa: BLE001 - the model's malformed plan is its error to fix, never said to the person
        return {"error": f"Not a query plan ({type(error).__name__}): give group_by, measures and optional filters and period."}
    draft = agent.store.get(draft_id)
    try:
        problems = Q.validate(plan, home_query.table(agent.store, draft))
    except ValueError as error:
        return {"error": str(error)[:300]}
    marked = home_query.sensitive_columns(plan, briefs.columns_of(draft))
    if problems or marked:
        return {"error": "; ".join([*problems, *(f"{c} holds personal data" for c in marked)])[:600]}
    said = agent.give_answer(draft_id, plan, "model", planner="home_agent")
    # whether it was answered and its number only: a count of groups or of hidden small groups would let repeated plans
    # recover a hidden row's value (the review of H7c recovered one in forty calls)
    return {"answered": said.get("kind") == "answer" or "plan_hash" in said, "answer": (said.get("answer") or {}).get("n", said.get("n"))}


def propose_step(turn: Turn, /, **args: Any) -> dict[str, Any]:
    """H9b: one preparation step the model read from the person's words, added to the recipe and the recipe run again from
    the original table: applied, or asked (a guarded step waits for the person's yes). A step that does not run on the table
    as the earlier steps left it is taken back and its error returned. The model gets only which: never a row count, which
    repeated filters would turn into a row's value."""
    from dclab_core import recipe as R

    from . import prepare

    try:
        step = R.Step.model_validate({**(args.get("step") or {}), "why": str(args.get("why") or "proposed from the person's words")[:400]})
    except Exception as error:  # noqa: BLE001 - the model's malformed step is its error to fix, never said to the person
        return {"error": f"Not a step ({type(error).__name__}): {str(error)[:300]}"}
    agent, draft_id = turn.agent, turn.draft_id
    draft = agent.store.get(draft_id)
    main = None
    if step.kind == "join" and not (draft.get("recipe") or {}).get("planned"):  # the table a join adds to: where its key repeats
        frames = {}
        for a in draft.get("assets") or []:
            if a.get("status") == "ready":
                frames[Path(str(a.get("name") or a["id"])).stem.lower()] = a["id"]
        frame, tables = agent._frames_for(draft, None)
        here = next((k for k, v in frames.items() if v == draft.get("active_asset")), None)
        name = prepare.main_table(f"join on {step.keys[0]}", {**({here: frame} if here else {}), **tables})
        main = frames.get(name) if name else None
    index = len((draft.get("recipe") or {}).get("planned") or []) if (draft.get("recipe") or {}).get("for") == draft.get("active_asset") else 0
    said = agent.change_recipe(draft_id, lambda rec: rec["planned"].append(step.model_dump(mode="json")), main=main) or {}
    rec = agent.store.get(draft_id).get("recipe") or {}
    mine = (rec.get("steps") or [None] * (index + 1))[index:index + 1]
    if mine and mine[0] and mine[0].get("error"):  # it did not run where it stands: taken back, said to the model
        agent.change_recipe(draft_id, lambda r: r["planned"].pop())
        error = mine[0]["error"]
        if error.startswith("join: no table"):  # the names a join can use (the review of H9b)
            _, others = agent._frames_for(agent.store.get(draft_id), rec.get("asset"))
            error += f" (the other tables: {', '.join(sorted(others)) or 'none yet'})"
        return {"error": error}
    asked = said.get("kind") == "question" and (rec.get("pending") or {}).get("index") == index
    return {"applied": bool(rec.get("done")), "asked": asked}


def undo_step(turn: Turn, /, **args: Any) -> dict[str, Any]:
    """H9b: the last step taken back; the recipe run again from the original table."""
    if not (turn.agent.store.get(turn.draft_id).get("recipe") or {}).get("planned"):
        return {"undone": False, "steps": 0}

    def drop(rec: dict[str, Any]) -> None:
        if rec["planned"]:
            rec["decisions"].pop(str(len(rec["planned"]) - 1), None)
            rec["planned"].pop()
    turn.agent.change_recipe(turn.draft_id, drop)
    return {"undone": True, "steps": len((turn.agent.store.get(turn.draft_id).get("recipe") or {}).get("planned") or [])}


def export_data(turn: Turn, /, **args: Any) -> dict[str, Any]:
    """H9b: the prepared table offered in the format the person asked for (csv, excel or parquet)."""
    from . import prepare

    fmt = prepare.FORMATS.get(str(args.get("format") or "csv").lower())
    if fmt is None:
        return {"error": "Formats: csv, excel, parquet."}
    rec = turn.agent.store.get(turn.draft_id).get("recipe") or {}
    turn.agent.store.update(turn.draft_id, lambda d: d.setdefault("recipe", {}).update(format=fmt))
    turn.agent.say(turn.draft_id, f"Your prepared table ({rec.get('rows', 0):,} rows) is ready to download.", kind="prepared",
                   model_text=f"(The prepared table is offered to download as {fmt}.)",  # no row count: a filter the model chose never tells it a row's value
                   prepared={"format": fmt, "rows": rec.get("rows"), "columns": rec.get("columns")})
    return {"format": fmt}


def compare_periods(turn: Turn, /, **args: Any) -> dict[str, Any]:
    """H11b: the model's reading of a business question as two periods, a measure and (optionally) a grouping column, checked
    like the rules' reading (no identifier, no personal column, the row floor) and said to the person as a decision brief.
    The model gets only whether it was answered and the grouping used, never a number."""
    import pandas as pd

    from dclab_core import change as C
    from dclab_core import query as Q
    from dclab_core.times import is_text

    from .. import home_query
    from . import business

    agent, draft_id = turn.agent, turn.draft_id
    draft = agent.store.get(draft_id)
    try:
        frame = home_query.table(agent.store, draft)
        pd_check = [pd.Timestamp(str(args[k])) for k in ("a_start", "a_end", "b_start", "b_end")]
        # the labels the person reads are written by code from the dates, never by the model
        label = lambda s, e: f"{s:%B %Y}" if s.day == 1 and e == s + pd.offsets.MonthEnd(0) else f"{s:%Y-%m-%d} to {e:%Y-%m-%d}"  # noqa: E731
        a = C.Window(start=f"{pd_check[0]:%Y-%m-%d}", end=f"{pd_check[1]:%Y-%m-%d}", label=label(pd_check[0], pd_check[1]))
        b = C.Window(start=f"{pd_check[2]:%Y-%m-%d}", end=f"{pd_check[3]:%Y-%m-%d}", label=label(pd_check[2], pd_check[3]))
    except Exception as error:  # noqa: BLE001 - the model's malformed reading is its error to fix
        return {"error": f"Give a_start, a_end, b_start, b_end as dates ({type(error).__name__})"}
    if pd_check[0] > pd_check[1] or pd_check[2] > pd_check[3]:
        return {"error": "a period's start is after its end"}
    columns = briefs.columns_of(draft)
    personal = {str(c.get("name")) for c in columns if str(c.get("sensitivity") or "none") in home_query.SENSITIVE}
    usable = lambda c: c in frame.columns and c not in personal and not Q.ID_NAME.search(c)  # noqa: E731
    measure = args.get("measure") or None
    if measure is not None and not (usable(str(measure)) and str(measure) in frame.select_dtypes("number").columns):
        return {"error": "measure: a number column that is not an identifier or personal (or leave it out to count rows)"}
    dates = [d for d in business._date_columns(frame) if d not in personal]
    date = str(args.get("date_column") or (dates[0] if len(dates) == 1 else ""))
    if date not in dates:
        return {"error": f"date_column: one of {', '.join(dates) or 'none'}"}
    allowed = [str(c) for c in frame.columns if is_text(frame[c]) and usable(str(c)) and str(c) != date
               and 1 < frame[c].nunique() <= business.MAX_SPLIT]  # the same columns the rules may split by (the review of H11b)
    splits = [str(args["by"])] if args.get("by") else allowed
    if not splits or not all(c in allowed for c in splits):
        return {"error": f"by: one of {', '.join(allowed) or 'none'} (a text column of a few values, not an identifier or personal)"}
    spec = {"a": a, "b": b, "date_column": date, "measure": measure, "narrow": {}, "splits": splits}
    try:
        result = business.compare(spec, frame)
    except ValueError as error:
        return {"error": str(error)[:300]}
    if result is None:
        return {"error": "no grouping column gave a comparison"}
    agent.give_brief(draft_id, spec, result, by="model")
    return {"answered": True, "by": result.by}


def decision_brief(turn: Turn, /, **args: Any) -> dict[str, Any]:
    """H11b: the last comparison's brief said again (its words checked again), when the person asks for it."""
    from dclab_core import change as C

    done = turn.agent.store.get(turn.draft_id).get("business") or {}
    if not done.get("result"):
        return {"error": "No comparison yet: compare_periods first."}
    result = C.ChangeResult.model_validate(done["result"])
    turn.agent.give_brief(turn.draft_id, {"narrow": done.get("narrow") or {}, "splits": [result.by]}, result, by=done.get("by") or "rules")
    return {"said": True}


def make_chart(turn: Turn, /, **args: Any) -> dict[str, Any]:
    """H7c: the chart code chose for an answer (its kind and title; the numbers are the person's to see)."""
    for m in reversed(turn.agent.store.get(turn.draft_id).get("messages") or []):
        a = m.get("answer") or {}
        if m.get("kind") == "answer" and a.get("n") == _int(args.get("answer")):
            return {"kind": (a.get("chart") or {}).get("kind"), "title": (a.get("chart") or {}).get("title")}
    return {"error": "No such answer."}


def save_answer(turn: Turn, /, **args: Any) -> dict[str, Any]:
    """H7c: keep an answer for the notebook export, only when the person asked to keep it."""
    turn.agent.save_answer(turn.draft_id, _int(args.get("answer")), by="model")
    return {"saved": True}


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


def decide_point(turn: Turn, /, **args: Any) -> dict[str, Any]:
    """26.5: a typed decision answered by DCLab's decision models beside its rules, from the draft's own state (bands,
    counts and flags, never a value): a column's (its role, an identifier, a time, a group, a code, a leak), or the
    problem's (its family, whether it is ready to build, the next question). The answer to act on follows the
    workspace's routing (the rules' unless a model passed its sealed gate and was switched on); a disagreement is shown."""
    from .. import decide as D
    from ..ledger import column_state
    from .card import card_of

    point = str(args.get("point") or "")
    draft = turn.agent.store.get(turn.draft_id)
    analysis = draft.get("analysis") or {}
    try:
        if point in D.TECHNICAL:
            name = str(args.get("column") or "")
            column = next((c for c in analysis.get("columns") or [] if c.get("name") == name), None)
            if column is None:
                return {"error": f"No column {name!r} in the prepared table: name one get_profile lists."}
            state = column_state(column, (analysis.get("summary") or {}).get("rows"))
            target = (draft.get("understanding") or {}).get("target")
            if target in {c.get("name") for c in analysis.get("columns") or []} and target != name:
                state["target"] = target
            out = D.ask(point, {k: v for k, v in state.items() if v is not None})
        elif point in D.WORKFLOW:
            state, value = _problem_state(turn.agent, draft, point, card_of(draft))
            out = D.ask(point, state, rule_value=value, rule_name="rule:home")
        else:
            return {"error": f"{point!r} is not asked here: {', '.join(D.TECHNICAL + D.WORKFLOW)}"}
    except D.DecideError as e:
        return {"error": str(e)}
    return {k: out.get(k) for k in ("point", "question", "answer", "decided_by", "level", "review", "rule", "models", "disagree", "note", "refused_by_validator")}


def _problem_state(agent: HomeAgent, draft: dict[str, Any], point: str, card: Any) -> tuple[dict[str, Any], Any]:
    """The state the ledger records at a problem point (as ``ledger.capture`` writes it) and the rules' answer there."""
    u = draft.get("understanding") or {}
    if point == "workflow.next_question":
        from dclab_core.workflow_labels import QUESTION_LIMIT

        questions = [q for q in draft.get("questions") or [] if not q.get("superseded")]
        state = {"open_questions": sum(1 for f in ("target", "prediction_moment", "action") if not u.get(f)), "step": len(questions),
                 "steps_left": max(QUESTION_LIMIT - len(questions), 0), "labels": capture_labels(draft),
                 "outcome_known": bool(u.get("target")), "moment_known": bool(u.get("prediction_moment")),
                 "pack": capture_code((draft.get("pack") or {}).get("key"))}
        from dclab_core.workflow_labels import SAME

        if any(not q.get("answered") for q in questions):
            return state, None  # a question waits for its answer: the rules decide nothing yet (as the ledger records it)
        asked = agent.next_question(draft)
        return state, (capture_code(SAME.get(asked[0], asked[0])) or "none") if asked else "none"
    base = {"labels": card.labels.value if card.labels.value in ("column", "file", "none") else "unknown",
            "open_questions": len(card.questions or []), "outcome_known": bool(card.outcome)}
    if point == "problem.family":
        return base, card.family.value
    return {**base, "family": card.family.value}, bool(card.ready)


def capture_labels(draft: dict[str, Any]) -> str:
    from ..ledger.capture import _labels_band

    return _labels_band(draft)


def capture_code(value: Any) -> Any:
    from ..ledger.capture import _code

    return _code(value)


def get_analysis(turn: Turn, /, **args: Any) -> dict[str, Any]:
    analysis = turn.agent.store.get(turn.draft_id).get("analysis") or {}
    if not analysis.get("columns"):
        return {"error": "No table has been prepared yet."}
    # descriptive only: the analysis never relates a column to the outcome
    return {"summary": analysis.get("summary"), "correlations": (analysis.get("correlations") or [])[:10],
            "highlights": [{k: h.get(k) for k in ("severity", "title", "text", "columns")} for h in (analysis.get("highlights") or [])[:12]]}


def build_solution(turn: Turn, /, **args: Any) -> dict[str, Any]:
    """25.13: build the project and hand it to DCLab's agent, as the button and "build it" do; what refused it comes back."""
    done = turn.agent.on_request(turn.draft_id, "build", {})
    if done is None:
        return {"error": "Building is not available here; tell the user to press Let's build the solution."}
    turn.agent.built(turn.draft_id, done)
    return {k: done.get(k) for k in ("project_id", "started", "refused")}


def simulate_data(turn: Turn, /, **args: Any) -> dict[str, Any]:
    agent, draft_id = turn.agent, turn.draft_id
    draft = agent.store.get(draft_id)
    try:
        rows = max(100, min(int(args.get("rows") or SIM_ROWS), SIM_ROWS))
    except (TypeError, ValueError):
        rows = SIM_ROWS
    agent.offer_synthetic(draft_id, (draft["problem"] + " " + str(args.get("description", ""))[:2000]).strip(), rows)
    return {"asked_how": "with AI, or described by the person and approved first", "rows": rows}  # the person chooses; nothing was started


# What each Home write does to a draft. The draft guard checks set_pack, propose_workflow and simulate; the others
# only add to the conversation or the understanding, which the user can always correct.
DRAFT_MOVES = {"ask_user": "ask", "record": "record", "set_pack": "set_pack", "propose_workflow": "propose_workflow",
               "request_data": "request_data", "simulate_data": "simulate", "build_solution": "build", "set_intent": "set_intent",
               "query_data": "answer", "save_answer": "save_answer", "propose_step": "prepare", "undo_step": "prepare",
               "export_data": "export", "compare_periods": "brief", "decision_brief": "brief"}
ML_MOVES = ("set_pack", "propose_workflow", "build")  # H2: the prediction track's moves, refused on the others
PREPARING = ("queued", "structuring", "cleaning", "analysing")


REFUSED = {"record": "The answer was not backed by the user's own words", "ask": "Not the question the plan asks now",
           "set_pack": "Not a known pack, or the user chose the pack", "simulate": "Data was already being prepared",
           "propose_workflow": "The workflow did not pass the workflow checks", "build": "The user had not asked to build",
           "set_intent": "Not backed by the user's words, or the user chose the type of help",
           "answer": "Not an analysis question, or the table is not ready", "save_answer": "No such answer",
           "prepare": "Not a preparation request, or the table is not ready", "export": "No prepared table yet",
           "brief": "Not a business question, or the table is not ready"}


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
        if tool.move in ML_MOVES and agent.track(agent.store.get(draft_id)) != "prediction":  # H2
            return {"error": "This request is not a prediction, so there is no pack, ML workflow or build; if the user wants a model, "
                             "set the type of help with set_intent (their words) first."}
        if tool.move == "ask" and arguments.get("field") in FIELDS and agent.track(agent.store.get(draft_id)) != "prediction":
            return {"error": "This request is not a prediction: ask without a field (the answer goes to the notes)."}
        if tool.move in ("answer", "save_answer"):  # H7c: answers come from a described table, on the analysis track
            draft = agent.store.get(draft_id)
            if agent.track(draft) != "analysis":
                return {"error": "Answers from the table are for analysis questions; this request is another kind of help."}
            if tool.move == "answer" and not any(m.get("kind") == "analysis" and m.get("asset") == draft.get("active_asset")
                                                 for m in draft.get("messages") or []):
                return {"error": "The table is not prepared yet: wait for it, or offer the person ways to bring data."}
            if tool.move == "save_answer" and not 0 <= _int(arguments.get("answer")) < len(draft.get("answers") or []):
                return {"error": "No such answer."}
        if tool.move == "brief":  # H11b: a business question, on a described table, with no question open
            draft = agent.store.get(draft_id)
            if agent.track(draft) != "business":
                return {"error": "A comparison of periods is for business questions; this request is another kind of help."}
            if not any(m.get("kind") == "analysis" and m.get("asset") == draft.get("active_asset") for m in draft.get("messages") or []):
                return {"error": "The table is not prepared yet: wait for it, or offer the person ways to bring data."}
            if agent.pending(draft) is not None:
                return {"error": "A question is open: wait for the person's answer first."}
        if tool.move in ("prepare", "export"):  # H9b, the recipe guard: preparation only, on a described table, one step at a time
            draft = agent.store.get(draft_id)
            if agent.track(draft) != "preparation":
                return {"error": "Preparing a table is for preparation requests; this request is another kind of help."}
            if not any(m.get("kind") == "analysis" and m.get("asset") == draft.get("active_asset") for m in draft.get("messages") or []):
                return {"error": "The table is not prepared yet: wait for it, or offer the person ways to bring data."}
            if agent.pending(draft) is not None:
                return {"error": "A question is open: wait for the person's answer first."}
            if tool.move == "export" and not (draft.get("recipe") or {}).get("done"):
                return {"error": "No prepared table yet: propose the steps first."}
        if tool.move == "set_intent":
            draft = agent.store.get(draft_id)
            if not draft.get("intent"):  # a draft from before H2 is a prediction and stays one (the review: old drafts behave as before)
                return {"error": "This conversation started before Home read the type of help: it is a prediction; the person can change it on the card."}
            if arguments.get("track") not in intents.TRACKS:
                return {"error": f"Unknown type of help. Types: {', '.join(intents.TRACKS)}."}
            if (draft.get("intent") or {}).get("source") == "user":
                return {"error": "The user chose the type of help; ask before changing it."}
            if not arguments.get("quote") or not plans.quoted(str(arguments["quote"]), plans.user_text(draft)):
                return {"error": "Give quote: whole words copied exactly from the user's problem or messages that say this."}
        if tool.move == "record" and arguments.get("field") in RULES_READ:  # live (HRN-022): "unit: biker", from the answer "The biker"
            return {"error": f"{arguments['field']} is read by the rules from the person's own answers (an entity answered as \"one "
                             "customer\"), never written by a model: record what they said under its own field."}
        if tool.move == "record" and arguments.get("field") in ANSWER_ONLY:
            draft = agent.store.get(draft_id)
            if not (agent.replied_to(draft, arguments["field"]) or agent.edited(draft, arguments["field"], arguments.get("value"))):
                return {"error": f"{arguments['field']} is the person's to say, in answer to its own question (it decides how the folds "
                                 "are grouped): ask it with ask_user."}  # "the day's orders", grouped nothing
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
        if tool.move == "ask" and arguments.get("field") in plans.ALL_FIELDS:
            # The outcome comes first; after it, the plan's order is a suggestion: the model may ask any field still open
            # when it judges it more useful now (its questions about which anomalies mattered and what is done with a
            # flag were refused for the order once, and the outcome was never settled). The outcome may be asked again
            # while it is open; a field already known is not.
            draft = agent.store.get(draft_id)
            field = arguments["field"]
            if field in ((draft.get("understanding") or {}).get("not_known") or []) or any(
                    q.get("field") == field and q.get("declined") for q in draft.get("questions") or []):
                # live (the 45 again, 2026-10-26): the person said they do not know the entity, and the model asked it again in
                # other words, and again: the walk stopped for no progress; a field the person declined is never asked again
                return {"error": f"The person said they do not know {field}: do not ask it again. Go on with what is known, "
                                 "say what DCLab assumes, or summarise."}
            plan = plans.refresh(draft)
            outcome_raised = plan["target"]["status"] != "unknown" or any(q["field"] == "target" for q in draft.get("questions") or [])
            usable = field in plans.fields_for(draft) or field in plans.FIELDS
            if not usable or plan.get(field, {}).get("status") != "unknown" or (field != "target" and not outcome_raised):
                question = agent.next_question(draft)
                nxt = f"ask about {question[0]} next" if question else "nothing in the plan is open; summarise instead"
                return {"error": f"{field} is not open in this draft's plan: {nxt}."}
        if tool.move == "set_pack":
            if arguments.get("key") not in packs.KEYS:
                return {"error": f"Unknown pack {str(arguments.get('key'))[:40]!r}. Packs: {', '.join(packs.KEYS)}."}
            if (agent.store.get(draft_id).get("pack") or {}).get("source") == "user":
                return {"error": "The user chose the pack; ask before changing it."}
        elif tool.move == "build":  # 25.13: only on the person's word, never on the model's own initiative
            if not asks_to_build(agent.store.get(draft_id)["messages"]):  # a bare "yes" only to an offer to build
                return {"error": "The user's last message is not an order to build: end your reply with the question \"Shall I build it now?\" and wait."}
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
    I = {"type": "integer"}  # noqa: E741
    add("query_data", "Answer an analysis question from the prepared table with a query plan: group_by (at most 3 columns), "
        "measures [{name, op: count|count_distinct|sum|mean|median|min|max|rate|percentile, column, where, q}], optional filters "
        "[{column, op, value}] and period {column, grain, start, end}. Numbers per group only; identifiers and personal columns are "
        "refused. The person sees the answer and its chart, written by code; you get only whether it was answered.",
        {"plan": {"type": "object"}}, ["plan"], "write")
    add("make_chart", "The chart DCLab drew for an answer: its kind and title.", {"answer": I}, ["answer"], "read")
    add("propose_step", "Prepare the person's table: one step read from their words, applied to a copy (the original never "
        "changes). step: {kind: rename|drop_columns|filter|dedupe|to_date|drop_missing|sort|months_since|join|pivot, columns, to, "
        "filters [{column, op, value}], keys, table, how, index, spread, values, agg, at, desc}; why: their words. A step that "
        "removes over 5% of the rows, or a join that changes the row count, is asked to the person before it applies. You learn "
        "whether it applied or was asked, never the rows.", {"step": {"type": "object"}, "why": S}, ["step"], "write")
    add("compare_periods", "Answer a business question about a change between two periods: where the number changed, by group, "
        "said to the person as a decision brief (the data shows association, not cause). a_start/a_end: the baseline period, b_start/"
        "b_end: the period asked about (dates); measure: a number column to total (leave out to count rows); by: a grouping column "
        "(leave out to choose the one that explains most); date_column when the table has several. You learn only whether it was "
        "answered.", {"a_start": S, "a_end": S, "b_start": S, "b_end": S, "measure": S, "by": S,
                      "date_column": S}, ["a_start", "a_end", "b_start", "b_end"], "write")
    add("decision_brief", "Say the last comparison's decision brief again, when the person asks for it.", {}, [], "write")
    add("undo_step", "Take back the last preparation step (the table is made again from the original).", {}, [], "write")
    add("export_data", "Offer the prepared table for download in the format the person asked for.",
        {"format": {**S, "enum": ["csv", "excel", "parquet"]}}, ["format"], "write")
    add("save_answer", "Keep an answer for the person's notebook export. Only after they asked to keep or save it.",
        {"answer": I}, ["answer"], "write")
    add("get_profile", "Column summaries of the prepared table: name, kind, missing rate, unique count, and which columns look like outcomes, timestamps, identifiers or text. Aggregates only, never values. Name columns, or page through a wide table with offset.",
        {"columns": {"type": "array", "items": S}, "offset": {"type": "integer"}}, [], "read")
    add("get_analysis", "The descriptive analysis of the prepared table: size, column kinds, missing share, duplicate rows, findings worth a question, and the strongest feature-to-feature correlations. Says nothing about the outcome.",
        {}, [], "read")
    add("decide_point", "Ask DCLab's decision models a typed decision instead of reasoning it out: a column's (column.semantic_role, "
        "column.is_identifier, column.time, column.group, column.category_code, feature.leakage_suspect; give column) or the problem's "
        "(problem.family, problem.ready, workflow.next_question). Returns the answer to act on, the rules' and each model's answer, and any "
        "disagreement.", {"point": {**S, "enum": ["column.semantic_role", "column.is_identifier", "column.time", "column.group",
                                                   "column.category_code", "feature.leakage_suspect", "problem.family", "problem.ready",
                                                   "workflow.next_question"]}, "column": S}, ["point"], "read")
    add("build_solution", "Build the project from what was settled and start DCLab's agent, which plans, trains and checks the model. "
        "Only after the user asked to build, train or go ahead.", {}, [], "write")
    add("simulate_data", "Generate a synthetic table. Only after the user asked for simulated data.",
        {"description": {**S, "description": "The table to simulate: rows, columns, the outcome and its rate."}, "rows": {"type": "integer", "description": "at most 10,000"}},
        ["description"], "write")
