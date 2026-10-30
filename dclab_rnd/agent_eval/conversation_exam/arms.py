"""The arms of the conversation exam (37.0): each talks with the simulated person through one interface.

    turn = arm.open(person)   # the person's opening (and their table, when they share one)
    turn = arm.send(words)    # the person's next message
    turn = arm.pick(i)        # the person picks choice i of the last turn
    arm.outcome()             # {"route", "tool", "routed_by": "option" | "self" | None}
    arm.sources()             # texts the system computed, from which a number in a reply may come

A turn is {"text": what the assistant said, "choices": [{"title", "what", "track", "tool", "kind": "option" | "button"}]}.
Both arms stop by one rule, in ``play``: a route is chosen and the last message asks nothing (no choices, no question).

* ``home``: today's Home agent, built as the product builds it (``DraftWork.agent``: every client Home uses).
* ``plain``: Claude alone, with a plain prompt (``exam_plain``) and the same table view Home's model sees (column names,
  kinds and what DCLab noticed, never a value), plus two tools to offer options or start work, so its route can be read.
* ``case``: the Phase 13 lead agent; refused until package 37.3 builds it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

from dclab_rnd import prompts

PLAIN_PROMPT = "exam_plain"
TRACKS = ("prediction", "analysis", "preparation", "reporting", "business", "advice")
TOOLS = ("ml_project", "segmentation_project", "query", "recipe", "dashboard", "business_brief", "cited_answer")
MAX_CALLS = 5  # model requests in one plain turn: a table read, options or a start, then the message
MAX_TOKENS = 6000


def home_table_view(name: str) -> list[dict[str, Any]]:
    """The column view Home's understanding model gets for an exam table, through Home's own upload path."""
    from dclab_rnd.draft import pipeline
    from dclab_rnd.draft import understand as U
    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.draft.store import DraftStore

    from ..home import tables

    with tempfile.TemporaryDirectory() as folder:
        store = DraftStore(Path(folder) / "drafts")
        d = store.create("a table")
        agent = HomeAgent(store, None, on_request=lambda *a: None)
        filename = tables.write(name, store.data_dir(d["id"]))
        asset = pipeline.new_asset(store, d["id"], "upload", filename, filename)
        pipeline.run(store, d["id"], asset["id"], agent, None)
        return U.columns_view(store.get(d["id"]))


_VIEWS: dict[str, list[dict[str, Any]]] = {}


def table_view(name: str) -> list[dict[str, Any]]:
    """The view Claude alone gets: the same as Home's (computed once per table)."""
    if name not in _VIEWS:
        _VIEWS[name] = home_table_view(name)
    return _VIEWS[name]


PLAIN_TOOLS = [
    {"type": "function", "function": {"name": "profile_table", "description": "The shared table's columns: names, kinds and what DCLab noticed about each. No values.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "offer_options", "description": "Show the person 2 to 4 options to pick from.",
        "parameters": {"type": "object", "required": ["options"], "properties": {"options": {
            "type": "array", "minItems": 2, "maxItems": 4, "items": {
                "type": "object", "required": ["title", "what", "track", "tool"],
                "properties": {"title": {"type": "string"}, "what": {"type": "string"},
                               "track": {"type": "string", "enum": list(TRACKS)}, "tool": {"type": "string", "enum": list(TOOLS)}}}}}}}},
    {"type": "function", "function": {
        "name": "start_work", "description": "Begin one kind of work the person has clearly asked for.",
        "parameters": {"type": "object", "required": ["track", "tool"],
                       "properties": {"track": {"type": "string", "enum": list(TRACKS)}, "tool": {"type": "string", "enum": list(TOOLS)}}}}},
]


class PlainArm:
    name = "plain"

    def __init__(self, client: Any):
        self.client = client
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": prompts.text(PLAIN_PROMPT)}]
        self.upload: Optional[str] = None
        self.choices: list[dict[str, Any]] = []
        self.route: dict[str, Any] = {"route": None, "tool": None, "routed_by": None}
        self.computed: list[str] = []

    def open(self, person: Mapping[str, Any]) -> dict[str, Any]:
        self.upload = person.get("upload") or None
        shared = f"\n\n(I shared a table: {self.upload}.csv)" if self.upload else ""
        return self.send(str(person["opening"]) + shared)

    def send(self, words: str) -> dict[str, Any]:
        self.messages.append({"role": "user", "content": words})
        return self._turn()

    def pick(self, i: int) -> dict[str, Any]:
        """The person picks an option: that is the route, and Claude answers the pick, as Home answers its own."""
        choice = self.choices[i]
        self.route = {"route": choice.get("track"), "tool": choice.get("tool"), "routed_by": "option"}
        return self.send(f"I pick: {choice['title']}")

    def _turn(self) -> dict[str, Any]:
        said: list[str] = []
        self.choices = []
        tools = PLAIN_TOOLS if self.upload else PLAIN_TOOLS[1:]
        for _ in range(MAX_CALLS):
            out = self.client.complete(self.messages, tools, max_tokens=MAX_TOKENS)
            self.messages.append(out.get("assistant_message") or {"role": "assistant", "content": out.get("content") or ""})
            if str(out.get("content") or "").strip():
                said.append(str(out["content"]).strip())
            calls = out.get("tool_calls") or []
            if not calls:  # the turn ends on a message to the person, never on a tool call (CEX-001: an advice answer was cut)
                break
            for c in calls:
                args = c.get("arguments") or {}
                if c["name"] == "profile_table" and self.upload:
                    value: Any = {"table": f"{self.upload}.csv", "columns": table_view(self.upload)}
                    self.computed.append(json.dumps(value))
                elif c["name"] == "offer_options":
                    options = [o for o in args.get("options") or [] if isinstance(o, dict) and o.get("title")]
                    self.choices = [{"title": str(o["title"]), "what": str(o.get("what") or ""), "track": o.get("track"),
                                     "tool": o.get("tool"), "kind": "option"} for o in options[:4]]
                    value = {"shown": len(self.choices), "next": "the person sees them under your message"}
                elif c["name"] == "start_work":
                    self.route = {"route": args.get("track"), "tool": args.get("tool"), "routed_by": "self"}
                    value = {"started": True, "next": "tell the person"}
                else:
                    value = {"error": f"no tool {c['name']!r} here"}
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(value)})
        return {"text": "\n\n".join(said), "choices": list(self.choices)}

    def outcome(self) -> dict[str, Any]:
        return dict(self.route)

    def sources(self) -> list[str]:
        return list(self.computed)

    def notes(self) -> dict[str, Any]:
        return {}


COMPUTED_KEYS = ("analysis", "data_card", "cleaning_log", "structure", "dashboard", "query", "business")
CODE_KINDS = ("analysis", "answer", "table", "result", "query", "brief")  # messages code writes from computed facts


class HomeArm:
    name = "home"

    def __init__(self, folder: Path, models: Any = None):
        """``models``: a gateway (or a view of one) that gives every client Home uses; None is Home with no model."""
        from dclab_rnd.draft.store import DraftStore

        self.folder = Path(folder)
        self.store = DraftStore(self.folder / "drafts")
        self.models = models if models is not None else _NoModels()
        self.id: Optional[str] = None
        self.agent: Any = None
        self.seen = 0
        self.choices: list[dict[str, Any]] = []
        self.picked_option = False

    def open(self, person: Mapping[str, Any]) -> dict[str, Any]:
        from dclab_rnd.draft import pipeline
        from dclab_rnd.draft.work import DraftWork

        from ..home import tables

        d = self.store.create(str(person["opening"]))
        self.id = d["id"]
        self.agent = DraftWork(self.store, self.models).agent(self.id)  # the product's own wiring (the review of 37.0)
        self.agent.start(self.id)
        if person.get("upload"):
            filename = tables.write(str(person["upload"]), self.store.data_dir(self.id))
            asset = pipeline.new_asset(self.store, self.id, "upload", filename, filename)
            pipeline.run(self.store, self.id, asset["id"], self.agent, None)
        first = self.store.get(self.id)["messages"][:1]
        self.seen = 1 if first and first[0].get("role") == "user" else 0
        return self._turn()

    def transcript_messages(self) -> list[dict[str, Any]]:
        return list(self.store.get(self.id)["messages"])

    def _turn(self) -> dict[str, Any]:
        draft = self.store.get(self.id)
        new = draft["messages"][self.seen:]
        self.seen = len(draft["messages"])
        text = "\n\n".join(str(m.get("text") or "").strip() for m in new if m.get("role") != "user" and str(m.get("text") or "").strip())
        q = self.agent.pending(draft)
        choices = []
        if q:
            choices += [{"title": str(o.get("title") or ""), "what": str(o.get("what") or ""), "track": o.get("track"), "tool": o.get("tool"),
                         "kind": "option"} for o in q.get("offer") or [] if o.get("title")]
            titles = {c["title"] for c in choices}  # an offer's titles come back as its buttons too: shown once
            choices += [{"title": str(b), "kind": "button"} for b in q.get("options") or [] if str(b).strip() and str(b) not in titles]
            if not text.strip() and q.get("text"):
                text = str(q["text"])
        self.choices = choices
        return {"text": text, "choices": choices, "waiting": q is not None}

    def send(self, words: str) -> dict[str, Any]:
        self.agent.reply(self.id, words)
        return self._turn()

    def pick(self, i: int) -> dict[str, Any]:
        choice = self.choices[i]
        self.picked_option = self.picked_option or choice.get("kind") == "option"
        return self.send(choice["title"])

    def outcome(self) -> dict[str, Any]:
        from dclab_rnd.draft import understand as U

        draft = self.store.get(self.id)
        intent = draft.get("intent") or {}
        track = None if intent.get("status") in ("asked", None) else intent.get("track")
        if track is None:
            return {"route": None, "tool": None, "routed_by": None}
        family = (draft.get("understanding") or {}).get("family")
        tool = "segmentation_project" if track == "prediction" and family == "segmentation" else U.TOOL_OF_TRACK.get(track)
        return {"route": track, "tool": tool, "routed_by": "option" if self.picked_option else "self"}

    def sources(self) -> list[str]:
        """What code computed: the table's analysis and Home's computed outputs, never the model's own words (the review
        of 37.0: the whole draft let Home's model cite timestamps and its own card)."""
        draft = self.store.get(self.id)
        out = [json.dumps({k: draft[k] for k in COMPUTED_KEYS if draft.get(k)}, default=str)]
        out += [str(m.get("text") or "") for m in draft.get("messages") or [] if m.get("role") != "user" and m.get("kind") in CODE_KINDS]
        return out

    def notes(self) -> dict[str, Any]:
        """Where Home's model failed and the rules answered instead (recorded, so a fallback is never read as the model)."""
        draft = self.store.get(self.id)
        understand = draft.get("understand") or {}
        intent = draft.get("intent") or {}
        return {"understand_failed": bool(understand.get("failed")), "intent_by": intent.get("by") or intent.get("source")}


class _NoModels:
    """Home with no model: every client is None, as in the open edition."""

    def client(self, *args: Any, **kwargs: Any) -> None:
        return None

    def available(self, *args: Any, **kwargs: Any) -> bool:
        return False
