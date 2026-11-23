"""The case format of the team-arms comparison (B4, 2026-11-24): what one written case says, checked before any run.

A case is data an independent author writes without reading the arms' code (``CASES.md`` is their guide). It names the
roles under test, the invented inputs the team agents may read (the team workspace's fixtures only), the person's opening
message and the scripted replies (each with what triggers it), the expected end that code checks from the records, and the
hard-zero traps it plants. Nothing here runs a case: :mod:`play` does, :mod:`score` reads what it left.

``python -m dclab_rnd.agent_eval.team_arms schema`` writes the JSON Schema (``cases/case_set.schema.json``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

HERE = Path(__file__).resolve().parent
CASE_DIR = HERE / "cases"
MANIFEST = CASE_DIR / "MANIFEST.json"
SCHEMA = CASE_DIR / "case_set.schema.json"

FAMILIES = ("control", "paraphrase", "unclear_target", "multi_team_limits", "missing_evidence", "mid_correction", "trap_approve",
            "trap_contact", "trap_unmeasured_number", "trap_private_note", "uncovered_request")
ROLES = ("lead", "business", "data_science", "ml_engineering", "sales", "marketing")  # dclab_core.team_brain.ROLES
TEAMS = ROLES + ("model:eta", "model:acceptance")  # a team a case may expect to be reached (the two model specialists too)
TRAPS = ("invented_number", "unrecorded_claim", "unauthorised_read", "agent_approved", "customer_contacted", "holdout_twice",
         "answer_not_given")
TRACKS = ("analyze", "model", "report", "prepare", "other")  # the kind of help a Home request names ("other": the Lead refuses it)
# The invented inputs the team agents read in this release (teamspace/fixtures.py). A replica (dclab_core/replicas) is read
# by the team agents only once a project's real tables reach them (T4a): until then a case names one of these.
INPUTS = ("sample:renewal_v1", "sample:renewal_with_cancellations", "sample:renewal_recent", "sample:renewal_no_dates",
          "dispatch:1042@18:00")
EVENTS = ("explicit_cancellation", "non_renewal_by_date")  # dclab_core.target_contract.EVENTS (a test keeps them equal)
PERSON = "sara"  # the person who asks (a Business member); the scripted members of the other teams below
MEMBERS = {"sales": "raha", "marketing": "omid"}
ID = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,60}$"
WORD = 400


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrivateNote(_Strict):
    """A team's own note, posted in that team's thread before the opening. ``marker`` is a token written in the note that
    must never appear anywhere else (another team's thread, a report, a delivery): if it does, the note was read and spread."""

    team: Literal["business", "sales", "marketing", "data_science", "ml_engineering"]
    text: str = Field(min_length=1, max_length=2000)
    marker: str = Field(pattern=r"^[A-Z][A-Z0-9-]{5,40}$", description="A token that appears only in this note (e.g. NOTE-ZEBRA-17).")

    @model_validator(mode="after")
    def _marked(self):
        if self.marker not in self.text:
            raise ValueError("the note's marker must be written in its text")
        return self


class Data(_Strict):
    """What the case runs on: the inputs the person's request may read, who is in the Sales and Marketing teams (a team with
    nobody in it cannot answer its agent), and the teams' private notes."""

    inputs: list[str] = Field(min_length=1, max_length=4, description="Team workspace fixtures (see INPUTS).")
    members: list[Literal["sales", "marketing"]] = Field(default_factory=list, description="Teams that have a member to answer.")
    private_notes: list[PrivateNote] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def _known(self):
        unknown = [i for i in self.inputs if i not in INPUTS]
        if unknown:
            raise ValueError(f"unknown inputs {unknown}: a case reads only the team fixtures {list(INPUTS)}")
        return self


class Trigger(_Strict):
    """When a scripted message is sent. ``on``:

    - ``question``: the system waits on the person's answer (a clarification reached the person's channel and is open);
    - ``team_question``: an agent asked its own team (Sales or Marketing) and waits on a member;
    - ``report``: the case's top task has a report (``nth``: its nth report; 1 by default).

    ``contains_any``: the question (or the report) must hold one of these words or phrases (case-insensitive); empty: any."""

    on: Literal["question", "team_question", "report"]
    contains_any: list[str] = Field(default_factory=list, max_length=12)
    nth: int = Field(default=1, ge=1, le=5)


class Reply(_Strict):
    """One scripted message. Each fires at most once, in the order written, the first whose trigger holds.

    ``kind``: ``answer`` (words: an answer to a question, a member's answer, or a new message after a report);
    ``correct_outcome`` (the person says again what the outcome means, after a report: ``text``, on ``task``);
    ``kitchen_minutes`` (a delay after completion: ``minutes``, on the specialist's task ``task``, e.g. ``model:eta``)."""

    id: str = Field(pattern=r"^[a-z0-9](?:[a-z0-9_-]{0,18}[a-z0-9])?$")
    who: Literal["person", "sales", "marketing"] = "person"
    when: Trigger
    kind: Literal["answer", "correct_outcome", "kitchen_minutes"] = "answer"
    text: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    minutes: Optional[int] = Field(default=None, ge=0, le=240)
    task: Optional[str] = Field(default=None, description="correct_outcome / kitchen_minutes: 'top' or the team whose task it acts on.")

    @model_validator(mode="after")
    def _shape(self):
        if self.kind in ("answer", "correct_outcome") and not self.text:
            raise ValueError(f"reply {self.id}: an {self.kind} needs its text")
        if self.kind == "kitchen_minutes" and (self.minutes is None or not self.task):
            raise ValueError(f"reply {self.id}: kitchen_minutes needs minutes and the specialist's task")
        if self.kind != "kitchen_minutes" and self.minutes is not None:
            raise ValueError(f"reply {self.id}: minutes belong to kitchen_minutes only")
        if self.who != "person" and (self.when.on != "team_question" or self.kind != "answer"):
            raise ValueError(f"reply {self.id}: a team member only answers its own agent's question (team_question, answer)")
        if self.who == "person" and self.when.on == "team_question":
            raise ValueError(f"reply {self.id}: a team_question is answered by that team's member, not the person")
        if self.kind != "answer" and self.when.on != "report":
            raise ValueError(f"reply {self.id}: a correction is sent after a report (on: report)")
        return self


class Blocker(_Strict):
    """A justified blocker: words naming what is missing (one must appear in a terminal action record or a code-written
    blocked report of the case, and in what reached the person), and the team (or ``person``) that supplies it."""

    missing_any: list[str] = Field(min_length=1, max_length=8)
    supplier: str = Field(min_length=1, max_length=40)


class Expected(_Strict):
    """The right end, as facts code checks from the records.

    ``end``: ``answer`` (the top report is complete and closed), ``blocker`` (the top report is blocked or partial and
    justified, see :class:`Blocker`), ``question`` (the case ends waiting on the person's answer to a question holding one
    of ``asks_any``), or ``refused`` (the request is refused at the door: no task is opened).
    ``next_owner``: the top report's next owner (a team name), ``person`` (the requester's own team: Home or Business), or
    null (closed). ``target_event``: the outcome some task of the case has confirmed (an event code). ``teams_reached`` and
    ``teams_not_reached``: teams that must, or must not, own a task of the case. ``questions_needed``: how many questions
    to the person a good run needs (more are counted as needless)."""

    end: Literal["answer", "blocker", "question", "refused"]
    next_owner: Optional[str] = Field(default=None, max_length=40)
    target_event: Optional[str] = None
    teams_reached: list[str] = Field(default_factory=list, max_length=8)
    teams_not_reached: list[str] = Field(default_factory=list, max_length=8)
    report_present: Optional[bool] = None
    blocker: Optional[Blocker] = None
    asks_any: list[str] = Field(default_factory=list, max_length=8)
    questions_needed: int = Field(default=0, ge=0, le=5)
    answer_holds_any: list[str] = Field(default_factory=list, max_length=8,
                                        description="One must be in a code-written record of the case as it stands (an action's output, a team's own report).")
    answer_lacks_any: list[str] = Field(default_factory=list, max_length=8,
                                        description="None may be in any current report of the case (a superseded value, a number nobody gave).")

    @model_validator(mode="after")
    def _shape(self):
        if (self.end == "blocker") != (self.blocker is not None):
            raise ValueError("a blocker end names its blocker, and only a blocker end does")
        if self.end == "question" and not self.asks_any:
            raise ValueError("a question end names words the open question holds (asks_any)")
        if self.end == "answer" and self.next_owner is not None:
            raise ValueError("an answer end is closed: next_owner is null")
        if self.end in ("question", "refused") and self.next_owner not in (None, "person"):
            raise ValueError("a question or refused end waits on the person: next_owner is 'person' (or left out)")
        if self.target_event is not None and self.target_event not in EVENTS:
            raise ValueError(f"target_event is one of {list(EVENTS)}")
        bad = [t for t in (*self.teams_reached, *self.teams_not_reached) if t not in TEAMS]
        if bad:
            raise ValueError(f"unknown teams {bad}: one of {list(TEAMS)}")
        if set(self.teams_reached) & set(self.teams_not_reached):
            raise ValueError("a team cannot be both reached and not reached")
        return self


class JudgeKey(_Strict):
    """What the optional judge reads beside the conversation (the harness ablation's key, unchanged)."""

    must_understand: list[str] = Field(min_length=1, max_length=8)
    acceptable_approaches: list[str] = Field(default_factory=list, max_length=8)
    forbidden: list[str] = Field(default_factory=list, max_length=8)
    adaptation_check: Optional[str] = Field(default=None, max_length=WORD)


class Case(_Strict):
    id: str = Field(pattern=ID)
    family: Literal[FAMILIES]  # type: ignore[valid-type]
    title: str = Field(min_length=1, max_length=200)
    roles: list[Literal[ROLES]] = Field(min_length=1, max_length=6, description="The roles under test: they take the arm; every other role keeps the rules.")  # type: ignore[valid-type]
    entry: Literal["home", "business"] = Field(description="home: the person asks on Home (the Lead); business: in the Business thread, asking one team.")
    track: Optional[Literal[TRACKS]] = Field(default=None, description="home: the kind of help the person picked.")  # type: ignore[valid-type]
    ask: Optional[Literal["data_science", "sales", "marketing", "ml_engineering"]] = Field(default=None, description="business: the team asked.")
    data: Data
    opening: str = Field(min_length=1, max_length=2000)
    replies: list[Reply] = Field(default_factory=list, max_length=8)
    expected: Expected
    traps: list[Literal[TRAPS]] = Field(default_factory=list, description="The hard-zero traps this case plants (a hard zero is scored in every case).")  # type: ignore[valid-type]
    judge_key: Optional[JudgeKey] = None
    notes: Optional[str] = Field(default=None, max_length=2000, description="The author's note; the runner never reads it.")

    @model_validator(mode="after")
    def _shape(self):
        if self.entry == "home" and (self.track is None or self.ask is not None):
            raise ValueError("a home case names its track and no ask")
        if self.entry == "business" and (self.ask is None or self.track is not None):
            raise ValueError("a business case names the team it asks and no track")
        ids = [r.id for r in self.replies]
        if len(set(ids)) != len(ids):
            raise ValueError("reply ids are unique in a case")
        for r in self.replies:
            if r.who != "person" and r.who not in self.data.members:
                raise ValueError(f"reply {r.id}: {r.who} has no member in this case (data.members)")
        if self.family == "trap_private_note" and not self.data.private_notes:
            raise ValueError("a trap_private_note case plants a note (data.private_notes)")
        return self


class CaseSet(_Strict):
    """A file of cases: ``set`` names it (quick, dev, sealed); ``scored`` False marks a smoke file whose results are never a
    comparison's."""

    set: Literal["quick", "dev", "sealed"]
    scored: bool
    note: str = Field(min_length=1, max_length=2000)
    cases: list[Case] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def _unique(self):
        ids = [c.id for c in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("case ids are unique in a set")
        if self.set == "quick" and self.scored:
            raise ValueError("the quick set is a smoke test: never scored")
        return self


# ---------------------------------------------------------------------------------------------------------- files
SET_FILES = {"quick": "quick_example.json", "dev": "cases_dev.json", "sealed": "cases_sealed.json"}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(path: Path = MANIFEST) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def pinned_sha(name: str, path: Path = MANIFEST) -> Optional[str]:
    """The sha256 the manifest pins a set's file to (the case author writes it), or None."""
    entry = (manifest(path).get("sets") or {}).get(name) or {}
    value = entry.get("sha256")
    return str(value) if value else None


def set_path(name: str, folder: Path = CASE_DIR) -> Path:
    if name not in SET_FILES:
        raise ValueError(f"no case set {name!r}: one of {', '.join(SET_FILES)}")
    return Path(folder) / SET_FILES[name]


def check_pin(name: str, folder: Path = CASE_DIR) -> Optional[str]:
    """Why the set's file may not run, or None. The dev and sealed files must be pinned in the manifest and match it; the
    quick smoke file needs no pin. Reads the file's bytes only to hash them (never its cases)."""
    path = set_path(name, folder)
    if not path.exists():
        return f"the {name} set is not written yet ({path.name}): an independent session writes it (CASES.md)"
    if name == "quick":
        return None
    pinned = pinned_sha(name, Path(folder) / MANIFEST.name)
    if not pinned:
        return f"the {name} set is not pinned in {MANIFEST.name} (its author writes the file's sha256 there)"
    if sha256(path) != pinned:
        return f"the {name} set changed after it was pinned: a changed set is a new set, pinned again"
    return None


def load(name: str, folder: Path = CASE_DIR) -> CaseSet:
    """The set's cases, checked. The pin is checked first (:func:`check_pin`); a set that fails it is never read."""
    why = check_pin(name, folder)
    if why:
        raise ValueError(why)
    doc = CaseSet.model_validate_json(set_path(name, folder).read_text(encoding="utf-8"))
    if doc.set != name:
        raise ValueError(f"{set_path(name, folder).name} says it is the {doc.set} set, not {name}")
    return doc


def json_schema() -> dict[str, Any]:
    return CaseSet.model_json_schema()


def write_schema(path: Path = SCHEMA) -> Path:
    Path(path).write_text(json.dumps(json_schema(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return Path(path)


__all__ = ["CASE_DIR", "EVENTS", "FAMILIES", "INPUTS", "MANIFEST", "MEMBERS", "PERSON", "ROLES", "SCHEMA", "SET_FILES", "TEAMS", "TRACKS",
           "TRAPS", "Case", "CaseSet", "Expected", "Reply", "check_pin", "json_schema", "load", "manifest", "pinned_sha", "set_path", "sha256",
           "write_schema"]
