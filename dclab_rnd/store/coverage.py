"""36.1: the coverage report and its gate. How much of what every agent and every kind of help does reaches training
data with its thinking, as numbers, and the gaps by name.

Two parts:

- **The gate** (:data:`CAPTURE`, :data:`TRACK_CAPTURE`, :func:`capture_gaps`): every agent of the harness
  (``agents/harness.AGENTS``) and every kind of help Home reads (``draft/intent.TRACKS``) names its capture path (the
  events, ledger points, model purposes and store kinds it leaves) or a written reason why not. Each path is checked
  against its vocabulary, every agent leaves its ``run.recorded``, and every model purpose an agent sends is named.
  ``tests/test_store_coverage.py`` fails on any gap, so a new agent or track without a capture path fails the suite.
- **The report** (:func:`measure`, :func:`render`, :func:`write`, ``make coverage``): read-only counts from the store and
  this machine's workspace. Sources by agent, by track and by purpose (runs, model calls, kept exchanges: whole, with
  thinking, with redacted thinking, thought but lost; Persian and English); then one table per model type (m1 to m8) by
  agent and by track (examples, kept exchanges placed there, recorded thinking, written reasoning source by source, an
  explicit note, nothing at all, an outcome or label, Persian and English). Each set is read at its latest version, train
  and dev only; a sealed split is counted from its manifest, never read. The summary goes into the store's catalog
  (``CATALOG.json`` and its copy in this repository) as ``coverage``, beside the other keys, which stay.

How a row is attributed (never guessed beyond these rules):

- agent: a run by its ``run.recorded`` name; an exchange or a chat example by its purpose (``harness.purposes()``:
  an agent's own purpose, else ``(roles)`` or ``(model uses)``); a decision example by where its point is decided
  (``ledger.CAPTURED``'s file, matched to an agent's entry module, else ``(studio code)``); an episode by its process.
- track: a draft by its ``request.intent`` decision (the person's label first, then the value used); a draft with none,
  and a project, is a prediction (Home's own rule, ``HomeAgent.track``: a draft read before H2 is a prediction, and a
  project is built only from one); a row tied to neither is ``(no request)``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from dclab_core.store import MODEL_TYPES, REASONING_SOURCES

from .layout import DEFAULT, Layout
from .sets import NOTE_PREFIX, why_no_reasoning
from .store import now, write_json

# ------------------------------------------------------------------------------------------------------------ the gate


@dataclass(frozen=True)
class Capture:
    """What an agent or a track leaves for training: ``paths`` as ``event:<kind>``, ``point:<ledger point>``,
    ``purpose:<model purpose>`` or ``store:<raw kind>``; ``why_not`` when it leaves nothing by design; ``missing`` what it
    does not leave yet (listed among the gaps, not a failure)."""
    paths: tuple[str, ...]
    why_not: Optional[str] = None
    missing: Optional[str] = None


_RUN = ("event:run.recorded", "event:agent.step")  # the harness's own record of every run and its steps
CAPTURE: dict[str, Capture] = {
    "home": Capture((*_RUN, "event:chat.person", "event:chat.agent", "purpose:home_agent", "point:request.intent",
                     "point:problem.pack", "store:conversations")),
    "home.project": Capture((*_RUN, "event:chat.person", "event:chat.agent", "purpose:home_orchestrator", "store:conversations")),
    "home.project.rules": Capture((*_RUN, "event:chat.agent", "store:conversations")),
    "home.script": Capture((*_RUN, "event:chat.person", "event:chat.agent", "point:request.intent", "store:conversations")),
    "intern": Capture((*_RUN, "event:tool.call", "purpose:intern", "store:conversations", "store:actions")),
    "workspace_chat": Capture((*_RUN, "purpose:workspace_chat", "store:conversations")),
    "intern.standard": Capture((*_RUN, "event:tool.call", "store:actions")),
    "team": Capture((*_RUN, "event:team.handoff", "event:team.recorded", "store:workflows")),
    "campaign": Capture((*_RUN, "purpose:campaign", "store:actions")),
    "mcp": Capture((*_RUN, "event:tool.call", "store:actions")),
    "orchestra": Capture((*_RUN, "store:workflows")),
    "orchestra.walk": Capture((*_RUN, "store:workflows")),
    "orchestra.solution": Capture((*_RUN, "event:programme.built", "store:workflows")),
    "tutor": Capture((*_RUN, "store:students")),
    "conductor": Capture((*_RUN, "store:curation")),
    "coordinator": Capture((*_RUN, "event:draft.built", "store:workflows")),
    "learning": Capture((*_RUN, "store:actions")),
    "twin": Capture((*_RUN, "purpose:ai_first", "store:conversations")),
}

_ASKED = ("point:request.intent", "event:chat.person", "event:chat.agent")  # every track: the reading of the request and the chat
_JUDGED = "event:track.outcome"  # 36.2: what the person did with the result (kept, changed, refused, asked again)
TRACK_CAPTURE: dict[str, Capture] = {
    "prediction": Capture((*_ASKED, "event:draft.built", "event:stage.finished", "event:outcome.observed", "point:workflow.g4")),
    "analysis": Capture((*_ASKED, "event:data.analysed", _JUDGED)),  # 36.2: an answer saved, let go or edited
    "preparation": Capture((*_ASKED, "event:data.cleaned", "point:data.cleaning_step", _JUDGED)),  # 36.2: a yes or no on a step, the table taken
    "reporting": Capture((*_ASKED, _JUDGED)),  # 36.2: a thumb on the reply that delivered the dashboard
    "business": Capture((*_ASKED, _JUDGED)),  # 36.2: a thumb on the reply that delivered the brief
    "advice": Capture((*_ASKED, "purpose:evidence_answer", "point:evidence.records", _JUDGED)),  # 2026-10-30: Home's cited answer, a thumb on it
}


def _vocabulary() -> dict[str, set[str]]:
    from dclab_core import experience
    from dclab_core.ledger import POINTS
    from dclab_core.store import KINDS

    from ..agents import harness

    return {"event": set(experience.KINDS), "point": set(POINTS), "purpose": set(harness.purposes()), "store": set(KINDS)}


def _check(name: str, entry: Optional[Capture], vocab: dict[str, set[str]], need: Iterable[str]) -> list[str]:
    if entry is None:
        return [f"{name}: no capture path and no reason why not"]
    if not entry.paths:
        return [] if (entry.why_not or "").strip() else [f"{name}: no capture path and no reason why not"]
    out = []
    for path in entry.paths:
        prefix, _, value = path.partition(":")
        if prefix not in vocab:
            out.append(f"{name}: {path!r} is not a capture path (event:, point:, purpose:, store:)")
        elif value not in vocab[prefix]:
            out.append(f"{name}: {path!r} names no {prefix} that exists")
    out += [f"{name}: its capture does not name {p!r}" for p in need if p not in entry.paths]
    return out


def capture_gaps(agents: Optional[dict[str, Any]] = None, tracks: Optional[Iterable[str]] = None,
                 capture: Optional[dict[str, Capture]] = None, track_capture: Optional[dict[str, Capture]] = None) -> list[str]:
    """Every agent and track without a valid capture path, by name (empty: the gate passes). Scratch registries can be
    passed in to show a missing one fails."""
    from ..agents import harness
    from ..draft import intent

    agents = harness.AGENTS if agents is None else agents
    tracks = intent.TRACKS if tracks is None else tuple(tracks)
    capture = CAPTURE if capture is None else capture
    track_capture = TRACK_CAPTURE if track_capture is None else track_capture
    vocab = _vocabulary()
    vocab["purpose"] |= {p for a in agents.values() for p in a.purposes}  # a scratch agent's purpose is a real name here
    gaps: list[str] = []
    for name, agent in agents.items():
        entry = capture.get(name)
        need = () if entry is not None and entry.why_not and not entry.paths else ("event:run.recorded", *(f"purpose:{p}" for p in agent.purposes))
        gaps += [f"agent {g}" for g in _check(name, entry, vocab, need)]
    gaps += [f"agent {n}: in the capture table but not an agent of the harness" for n in capture if n not in agents]
    for track in tracks:
        gaps += [f"track {g}" for g in _check(track, track_capture.get(track), vocab, ("point:request.intent",))]
    gaps += [f"track {t}: in the capture table but not a kind of help Home reads" for t in track_capture if t not in tracks]
    return gaps


# ---------------------------------------------------------------------------------------------------------- the report
TRACE_ONLY = ("campaign",)  # agents whose runs may exist only as trace rows (their environment has no session log)
SOURCES = (*REASONING_SOURCES, *(s for s in ("replayed",) if s not in REASONING_SOURCES))
EXAMPLE_COLUMNS = ("examples", "exchanges", *SOURCES, "lost", "note", "no_model", "none", "outcome", "fa", "en")
SOURCE_COLUMNS = ("runs", "calls", "exchanges", "whole", "thinking", "thinking_redacted", "thought_lost", "fa", "en")
NO_REQUEST = "(no request)"
_PERSIAN = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile(r"[A-Za-z]{3,}")
# an episode's process -> the agent that walks it (the rest are the studio's own code: its graph, jobs and stages)
PROCESS_AGENT = {"team": "team", "walk": "orchestra.walk", "intern": "intern", "conductor": "conductor", "dclab_code": "(DCLab Code)"}


def _language(texts: Iterable[str]) -> Optional[str]:
    joined = " ".join(t for t in texts if isinstance(t, str))
    if _PERSIAN.search(joined):
        return "fa"
    return "en" if _LATIN.search(joined) else None


def _user_texts(messages: Any) -> list[str]:
    if not isinstance(messages, list):
        return []
    return [m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)]


def _gz_rows(base: Path, keep: Optional[str] = None) -> Iterator[dict[str, Any]]:
    """A kind's shaped records as plain dicts (no contract parsing: a count needs none); ``keep`` skips lines without it."""
    for path in sorted(base.rglob("*.jsonl.gz")) if base.exists() else []:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if keep is None or keep in line:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue


def _jsonl_rows(base: Path, keep: Optional[str] = None) -> Iterator[dict[str, Any]]:
    for path in sorted(base.rglob("*.jsonl")) if base.exists() else []:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if keep is None or keep in line:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue


def _home(home: Optional[Path]) -> Optional[Path]:
    if home is not None:
        return Path(home)
    try:
        from ..settings import Settings

        return Path(Settings.load().agent_home)
    except Exception:  # noqa: BLE001 — a report without the workspace still reads the store
        return None


class _Tracks:
    """Draft -> kind of help, project -> draft: the rules of the module's docstring."""

    def __init__(self) -> None:
        self.used: dict[str, str] = {}      # record id -> draft
        self.value: dict[str, str] = {}     # draft -> the value used
        self.label: dict[str, str] = {}     # draft -> the person's label (or the strongest one kept)
        self.project_draft: dict[str, str] = {}
        self.built: set[str] = set()        # drafts that became a project

    def decision(self, d: dict[str, Any]) -> None:
        if d.get("point") != "request.intent":
            return
        draft = d.get("draft") or (d.get("subject") or {}).get("key")
        if draft and d.get("value_used") is not None:
            self.used[str(d.get("id"))] = str(draft)
            self.value[str(draft)] = str(d["value_used"])

    def labelled(self, d: dict[str, Any]) -> None:
        draft = self.used.get(str(d.get("record")))
        if draft and d.get("value") is not None and d.get("source") == "user_decision_exposed":  # the person's own choice on the card
            self.label[draft] = str(d["value"])

    def of(self, draft: Any = None, project: Any = None) -> str:
        if not draft and project:
            draft = self.project_draft.get(str(project))
            if not draft:
                return "prediction"  # a project is built only from a prediction (Home's rule)
        if not draft:
            return NO_REQUEST
        d = str(draft)
        if d in self.label or d in self.value:
            return self.label.get(d) or self.value[d]
        return "prediction" if d in self.built else NO_REQUEST  # a draft read before H2 that was built; any other id is no request


def _point_agents() -> dict[str, str]:
    """A ledger point -> the agent whose entry module decides it (``ledger.CAPTURED``'s file), else ``(studio code)``."""
    from ..agents import harness
    from ..ledger import CAPTURED

    by_module: dict[str, list[str]] = {}
    for name, agent in harness.AGENTS.items():
        by_module.setdefault(agent.entry.split(":")[0], []).append(name)
    out = {}
    for point, where in CAPTURED.items():
        m = re.match(r"\s*([\w/]+)\.py", str(where))
        names = by_module.get("dclab_rnd." + m.group(1).replace("/", "."), []) if m else []
        if not names:
            out[point] = "(studio code)"
            continue
        parts = [n.split(".") for n in names]  # several agents in one module: their common name (home, home.script -> home)
        common = []
        for level in zip(*parts):
            if len(set(level)) > 1:
                break
            common.append(level[0])
        out[point] = ".".join(common) or names[0]
    return out


def _purpose_agent(purpose: Any) -> str:
    from ..agents import harness

    sender = harness.purposes().get(str(purpose or ""))
    if sender is None:
        return f"(other: {purpose})" if purpose else "(other)"
    return {"role": "(roles)", "model_use": "(model uses)"}.get(sender, sender)


def _example_agent(row: dict[str, Any], points: dict[str, str]) -> str:
    shape = str(row.get("schema_version") or "").split(".")[1:2]
    shape = shape[0] if shape else ""
    if shape in ("chat_example", "preference_pair"):
        task = str(row.get("task") or "")
        if task.startswith("selfplay"):
            return "(self-play)"
        if task.startswith("home_conversation"):
            return "home"
        return _purpose_agent(task.split(":")[0])
    if shape == "decision_example":
        return points.get(str(row.get("point")), "(studio code)")
    if shape == "episode":
        return PROCESS_AGENT.get(str(row.get("process")), "(studio code)")
    if shape == "solver_trajectory":
        return "(solver)"
    if shape == "retrieval_example":
        return "(evidence search)"
    if shape == "routing_example":  # P8: Home's routing decisions (and its written and eval requests)
        return "home"
    return "(studio code)"


def _reasoning(obj: Any, found: set[str], notes: list[str]) -> None:
    """Every labelled reasoning source and every note saying why there is none, anywhere in an example."""
    if isinstance(obj, dict):
        for key in ("reasoning_source", "thought_source", "thinking_source"):  # thinking_source: a routing reading's (P8)
            if obj.get(key):
                found.add(str(obj[key]))
        for key in ("no_reasoning", "no_thought", "no_thinking"):
            if obj.get(key):
                notes.append(str(obj[key]))
        for value in obj.values():
            if isinstance(value, (dict, list)):
                _reasoning(value, found, notes)
    elif isinstance(obj, list):
        for value in obj:
            _reasoning(value, found, notes)


def _outcome(row: dict[str, Any]) -> bool:
    for key in ("label", "outcome", "preferred", "score", "passed"):
        if row.get(key) not in (None, "", [], {}):
            return True
    if row.get("decided_by") or (row.get("check") or {}).get("outcome") or row.get("used"):
        return True
    return False


def _example_language(row: dict[str, Any]) -> Optional[str]:
    texts = _user_texts(row.get("messages")) + _user_texts(row.get("prompt"))
    if isinstance(row.get("query"), str):
        texts.append(row["query"])
    return _language(texts) if texts else None


def _latest_sets(layout: Layout, model_type: str) -> list[Path]:
    """Each set's latest version folder (``.../<set>/vN``), the highest N."""
    base = layout.sets(model_type)
    best: dict[Path, tuple[int, Path]] = {}
    for manifest in base.rglob("MANIFEST.json") if base.exists() else []:
        folder = manifest.parent
        m = re.fullmatch(r"v(\d+)", folder.name)
        if not m:
            continue
        n = int(m.group(1))
        if folder.parent not in best or n > best[folder.parent][0]:
            best[folder.parent] = (n, folder)
    return sorted(v[1] for v in best.values())


def _hash(doc: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _choices_of(layout: Layout, records: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    """The latest rule curator's choice per (record, model type), for these records only (``sets.standing`` reads all)."""
    from .store import Store

    store = Store(layout)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in sorted(store.entries("curation"), key=lambda e: (e.captured_at, e.version)):
        for line in store.read(entry).decode("utf-8").splitlines():
            if not any(r in line for r in records) if len(records) < 64 else '"record"' not in line:
                continue
            d = json.loads(line)
            if d.get("record") in records and str(d.get("curator", "")).startswith("rule:") and d.get("model_type"):
                out[(d["record"], d["model_type"])] = d
    return out


def _zero(columns: tuple[str, ...]) -> dict[str, int]:
    return {c: 0 for c in columns}


def _bump(table: dict[str, dict[str, int]], row: str, columns: tuple[str, ...], **adds: int) -> None:
    cell = table.setdefault(row, _zero(columns))
    for key, n in adds.items():
        if n:
            cell[key] = cell.get(key, 0) + n


def measure(layout: Optional[Layout] = None, home: Optional[Path] = None) -> dict[str, Any]:
    """The coverage of this store and workspace: read-only, no model, no network."""
    from ..agents import harness
    from ..draft import intent

    layout = layout or Layout()
    home = _home(home)
    tracks = _Tracks()
    track_rows = (*intent.TRACKS, NO_REQUEST)

    # the kinds of help: request.intent decisions and their labels, in the store and the workspace's ledger
    for r in _gz_rows(layout.records_of("decisions"), keep='"request.intent"'):
        tracks.decision(r.get("body") or {})
    if home is not None:
        for d in _jsonl_rows(home / "ledger" / "decisions", keep='"request.intent"'):
            tracks.decision(d)
    for r in _gz_rows(layout.records_of("decisions"), keep='"request.intent"'):
        tracks.labelled(r.get("body") or {})
    if home is not None and tracks.used:
        for d in _jsonl_rows(home / "ledger" / "labels"):
            tracks.labelled(d)

    # runs (and which draft each project was built from)
    by_agent: dict[str, dict[str, int]] = {a: _zero(SOURCE_COLUMNS) for a in harness.AGENTS}
    by_track: dict[str, dict[str, int]] = {t: _zero(SOURCE_COLUMNS) for t in track_rows}
    by_purpose: dict[str, dict[str, int]] = {}
    runs: dict[str, tuple[str, Any, Any]] = {}
    traced: set[str] = set()  # the review of 2026-10-30: a session's record names its trace's run id as its parent
    for r in _gz_rows(layout.records_of("sessions")):
        b = r.get("body") or {}
        ids, payload = b.get("ids") or {}, b.get("payload") or {}
        if b.get("kind") == "draft.built" and ids.get("project") and ids.get("conversation"):
            tracks.project_draft[str(ids["project"])] = str(ids["conversation"])
            tracks.built.add(str(ids["conversation"]))
        elif b.get("kind") == "run.recorded" and payload.get("run_kind") == "agent" and payload.get("run"):
            runs[str(payload["run"])] = (str(payload.get("name") or "(unnamed)"), ids.get("conversation"), ids.get("project"))
            traced.update(str(x) for x in (payload.get("parent"), ids.get("run")) if x)  # the trace's own run id
    # 2026-10-30: an agent whose environment cannot write session events (the campaign's, with no numpy) still leaves its
    # trace rows (agent_steps.jsonl, kind actions): one run per run id, counted where no run.recorded names it
    for r in _gz_rows(layout.records_of("actions"), keep='"run_id"'):
        b = r.get("body") or {}
        agent, run_id = b.get("agent"), b.get("run_id")
        if agent in harness.AGENTS and run_id and str(run_id) not in runs and str(run_id) not in traced and harness.AGENTS[agent].runs == "session" \
                and str(agent) in TRACE_ONLY:
            runs[str(run_id)] = (str(agent), None, None)
    purposes = harness.purposes()
    for name, conversation, project in runs.values():
        name = name if name in harness.AGENTS else str(purposes.get(name, name))  # the orchestrator records its runs by its purpose
        _bump(by_agent, name, SOURCE_COLUMNS, runs=1)
        # a Home or intern run's conversation is its draft; other agents' runs have no request of their own
        draft = conversation if name.split(".")[0] in ("home", "intern") else None
        _bump(by_track, tracks.of(draft, project), SOURCE_COLUMNS, runs=1)

    # model calls (every request, its purpose only) and the kept exchanges (whole or shape)
    calls: set[str] = set()
    for r in _gz_rows(layout.records_of("model_requests")):
        b, prov = r.get("body") or {}, r.get("provenance") or {}
        source = str(prov.get("source") or "")
        if not b.get("purpose") or not (source.startswith("model_requests.jsonl") or source.startswith("db/model_request")):
            continue
        # one request once: a log kept again whole (a new entry version) repeats its lines under new record ids
        body = {k: v for k, v in b.items() if k != "_context"}
        key = str(prov.get("workspace") or "") + ":" + (str(b["id"]) if b.get("id") else _hash(body))
        if key in calls:
            continue
        calls.add(key)
        _bump(by_agent, _purpose_agent(b["purpose"]), SOURCE_COLUMNS, calls=1)
        _bump(by_purpose, str(b["purpose"]), SOURCE_COLUMNS, calls=1)
        _bump(by_track, tracks.of(b.get("draft"), b.get("project")), SOURCE_COLUMNS, calls=1)
    exchanges: dict[str, dict[str, Any]] = {}
    store_ids: dict[str, str] = {}  # exchange -> its store record id (for where the curators placed it)
    for r in _gz_rows(layout.records_of("model_requests")):
        b, source = r.get("body") or {}, str((r.get("provenance") or {}).get("source") or "")
        if source.startswith("model_exchanges") and (b.get("model_request") or r.get("id")):
            key = str(b.get("model_request") or r.get("id"))
            exchanges.setdefault(key, b)
            store_ids.setdefault(key, str(r.get("id")))
    if home is not None:
        for d in _jsonl_rows(home / "model_exchanges"):
            if d.get("model_request"):
                exchanges.setdefault(str(d["model_request"]), d)
    # 2026-10-30: a twin lane keeps Claude's thinking in its own transcript (transcripts/thinking.jsonl, one row per reply
    # that thought), filed as "other" in the store; counted here as kept replies with their thinking, by purpose
    seen_thoughts: set[str] = set()
    for r in _gz_rows(layout.records_of("other"), "transcripts/thinking.jsonl"):
        b, prov = r.get("body") or {}, r.get("provenance") or {}
        if not str(prov.get("source") or "").endswith("transcripts/thinking.jsonl") or not (b.get("thinking") or b.get("redacted")):
            continue
        key = f"{prov.get('workspace')}:{b.get('request') or b.get('n')}"
        if key in seen_thoughts:
            continue
        seen_thoughts.add(key)
        adds = {"exchanges": 1, "thinking": int(bool(b.get("thinking"))), "thinking_redacted": int(not b.get("thinking") and bool(b.get("redacted")))}
        _bump(by_agent, _purpose_agent(b.get("purpose")), SOURCE_COLUMNS, **adds)
        _bump(by_purpose, str(b.get("purpose") or "(none)"), SOURCE_COLUMNS, **adds)
    exchange_agent: dict[str, str] = {}
    exchange_track: dict[str, str] = {}
    for key, d in exchanges.items():
        thinking, redacted = bool(d.get("thinking")), bool(d.get("thinking_redacted"))
        lang = _language(_user_texts(d.get("messages"))) if d.get("whole") else None
        adds = {"exchanges": 1, "whole": int(bool(d.get("whole"))), "thinking": int(thinking), "thinking_redacted": int(redacted),
                "thought_lost": int(bool(d.get("thought")) and not thinking and not redacted and not d.get("exam")), "fa": int(lang == "fa"), "en": int(lang == "en")}
        agent, track = _purpose_agent(d.get("purpose")), tracks.of(d.get("draft"), d.get("project"))
        exchange_agent[key], exchange_track[key] = agent, track
        _bump(by_agent, agent, SOURCE_COLUMNS, **adds)
        _bump(by_track, track, SOURCE_COLUMNS, **adds)
        _bump(by_purpose, str(d.get("purpose") or "(none)"), SOURCE_COLUMNS, **adds)

    # where the curators placed each kept exchange
    placed: dict[str, list[str]] = {}
    record_exchange = {rid: key for key, rid in store_ids.items()}
    try:
        choices = _choices_of(layout, set(record_exchange))  # not every curation choice: on this store they take gigabytes
    except Exception:  # noqa: BLE001 — a store without curation still reports its sets
        choices = {}
    for (rid, model_type), choice in choices.items():
        if rid in record_exchange and choice.get("quality") != "out" and not choice.get("refused"):
            placed.setdefault(model_type, []).append(record_exchange[rid])

    # the sets: one table per model type
    points = _point_agents()
    model_types: dict[str, Any] = {}
    for model_type in MODEL_TYPES:
        a_table: dict[str, dict[str, int]] = {a: _zero(EXAMPLE_COLUMNS) for a in harness.AGENTS}
        t_table: dict[str, dict[str, int]] = {t: _zero(EXAMPLE_COLUMNS) for t in track_rows}
        total, sealed, sets = _zero(EXAMPLE_COLUMNS), 0, []
        for folder in _latest_sets(layout, model_type):
            try:
                manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
            except ValueError:
                manifest = {}
            sealed += int((manifest.get("counts") or {}).get("sealed") or 0)
            sets.append(layout.relative(folder) if hasattr(layout, "relative") else str(folder))
            for split in ("train", "dev"):  # never the sealed split
                path = folder / f"{split}.jsonl"
                if not path.exists():
                    continue
                with path.open(encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        found: set[str] = set()
                        notes: list[str] = []
                        _reasoning(row, found, notes)
                        why = why_no_reasoning(row) if not found and not notes else None
                        if why and why.startswith(NOTE_PREFIX):  # a model's work with a stated reason: a note, not "no model"
                            notes.append(why)
                            why = None
                        no_model = bool(why)  # no model made it: nothing to keep
                        lang = _example_language(row)
                        adds = {"examples": 1, **{s: int(s in found) for s in SOURCES},
                                "lost": int(not found and any("lost" in n.lower() for n in notes)),
                                "note": int(not found and bool(notes)), "no_model": int(no_model),
                                "none": int(not found and not notes and not no_model),
                                "outcome": int(_outcome(row)), "fa": int(lang == "fa"), "en": int(lang == "en")}
                        prov = row.get("provenance") or {}
                        _bump(a_table, _example_agent(row, points), EXAMPLE_COLUMNS, **adds)
                        track = tracks.of(prov.get("draft"), prov.get("project"))
                        if track == NO_REQUEST and (row.get("check") or {}).get("track") in intent.TRACKS:
                            track = row["check"]["track"]  # 2026-10-30: a Home conversation says its own kind of help
                        _bump(t_table, track, EXAMPLE_COLUMNS, **adds)
                        for k, n in adds.items():
                            total[k] += n
        for key in placed.get(model_type, []):
            _bump(a_table, exchange_agent.get(key, "(other)"), EXAMPLE_COLUMNS, exchanges=1)
            _bump(t_table, exchange_track.get(key, NO_REQUEST), EXAMPLE_COLUMNS, exchanges=1)
            total["exchanges"] += 1
        model_types[model_type] = {**total, "sealed_not_read": sealed, "sets": len(sets), "by_agent": a_table, "by_track": t_table}

    from .sets import routing as routing_sets

    doc = {"schema_version": 1, "built_at": now(), "sources": {"by_agent": by_agent, "by_track": by_track,
                                                                 "by_purpose": dict(sorted(by_purpose.items()))},
           "model_types": model_types, "routing": routing_sets.counts(layout)}
    doc["gaps"] = gaps(doc)
    return doc


def gaps(doc: dict[str, Any]) -> list[str]:
    """What is missing, by name: the gate's gaps, then agents, tracks and model types without the data they need."""
    from ..agents import harness
    from ..draft import intent

    out = list(capture_gaps())
    src = doc["sources"]
    for name, agent in harness.AGENTS.items():
        row = src["by_agent"].get(name, {})
        if not row.get("runs"):
            out.append(f"agent {name}: no run recorded")
        if agent.purposes and not row.get("thinking") and not row.get("thinking_redacted"):
            out.append(f"agent {name}: sends {', '.join(agent.purposes)} but no kept exchange holds its thinking")
    for track in intent.TRACKS:
        row = src["by_track"].get(track, {})
        examples = sum(m["by_track"].get(track, {}).get("examples", 0) for m in doc["model_types"].values())
        if not row.get("runs") and not examples:
            out.append(f"track {track}: no run and no training example")
        elif not examples:
            out.append(f"track {track}: {row.get('runs', 0)} runs but no training example")
        if not row.get("fa") and not sum(m["by_track"].get(track, {}).get("fa", 0) for m in doc["model_types"].values()):
            out.append(f"track {track}: nothing in Persian")
        if TRACK_CAPTURE.get(track) and TRACK_CAPTURE[track].missing:
            out.append(f"track {track}: {TRACK_CAPTURE[track].missing}")
    routing = doc.get("routing") or {}
    for point in ("request.intent",):  # P8: what the routing model still lacks, by name
        row = routing.get(point) or {}
        if not (row.get("label") or {}).get("real"):
            out.append(f"m7_routing {point}: no real label yet (a person's confirm or change on the card, on a real request)")
        if not row.get("own_model"):
            out.append(f"m7_routing {point}: no own model's shadow answer yet")
        missing = row.get("model_readings", 0) - row.get("thinking", 0) - row.get("explained", 0)  # explained: an exam, a copy, no thinking made
        if missing > 0:
            out.append(f"m7_routing {point}: {missing} of {row['model_readings']} model readings without their thinking summary and "
                       "without a reason it cannot exist")
    for model_type, m in doc["model_types"].items():
        if not m["examples"]:
            out.append(f"{model_type}: no example")
            continue
        if not m.get("recorded") and (m.get("none") or m.get("lost")):  # a model made rows (missing, or lost before capture)
            out.append(f"{model_type}: no example with recorded thinking")
        if m.get("none"):  # made by a model, or by someone the record does not name: its thinking is the gap
            out.append(f"{model_type}: {m['none']} of {m['examples']} examples were made by a model or by someone the record does not name, "
                       "and carry no reasoning and no note")
    return out


# ---------------------------------------------------------------------------------------------------------- printing
_SHORT = {"examples": "ex", "exchanges": "xch", "recorded": "rec", "from_record": "from_rec", "teacher": "teach", "visible": "vis",
          "replayed": "replay", "outcome": "outc", "thinking_redacted": "redacted", "thought_lost": "lost"}


def _table(title: str, rows: dict[str, dict[str, int]], columns: tuple[str, ...], skip_empty: bool = False) -> list[str]:
    heads = [_SHORT.get(c, c) for c in columns]
    names = [n for n, r in rows.items() if not skip_empty or any(r.get(c) for c in columns)]
    width = max([len(title), *(len(n) for n in names)] or [10])
    lines = [f"{title:<{width}}  " + "  ".join(f"{h:>8}" for h in heads)]
    lines += [f"{n:<{width}}  " + "  ".join(f"{rows[n].get(c, 0):>8}" for c in columns) for n in names]
    return lines


def render(doc: dict[str, Any]) -> str:
    """The tables as text: sources by agent and track, then each model type by agent and by track, then the gaps."""
    src = doc["sources"]
    out = ["# Coverage (36.1)", "", "## Sources: runs, model calls and kept exchanges", ""]
    out += _table("agent", src["by_agent"], SOURCE_COLUMNS) + [""]
    out += _table("kind of help", src["by_track"], SOURCE_COLUMNS) + [""]
    out += _table("purpose", src["by_purpose"], SOURCE_COLUMNS) + [""]
    out += ["Columns: xch = exchanges kept, whole = with their text, thinking = with the model's thinking, redacted = its thinking "
            "kept redacted, lost = it thought but only the flag was kept, fa/en = the person's language.", ""]
    for model_type, m in doc["model_types"].items():
        out += [f"## {model_type}: {m['examples']} examples in {m['sets']} sets (train and dev; {m['sealed_not_read']} sealed rows "
                f"counted, not read), {m['exchanges']} kept exchanges placed here", ""]
        if not m["examples"] and not m["exchanges"]:
            continue
        out += _table("agent", m["by_agent"], EXAMPLE_COLUMNS) + [""]
        out += _table("kind of help", m["by_track"], EXAMPLE_COLUMNS) + [""]
    out += render_routing(doc.get("routing") or {})
    out += ["Columns: ex = examples, rec..replay = examples with written reasoning from that source (rec = recorded thinking), "
            "lost/note = no reasoning but a note why (lost: the note says it was lost), no_model = no model made it (a rule, a classifier, a "
            "person, code): nothing to keep, none = no reasoning and no note, "
            "outc = with an outcome or label.", "", f"## Gaps ({len(doc['gaps'])})", ""]
    out += [f"- {g}" for g in doc["gaps"]] or ["- none"]
    return "\n".join(out)


ROUTING_COLUMNS = ("rows", "real", "written", "silver", "none", "text", "thinking", "model_readings", "explained", "label_thinking", "own_model", "fa", "en", "mixed",
                   "sealed_not_read")


def render_routing(routing: dict[str, Any]) -> list[str]:
    """P8: the routing model's data (m7_routing) per routing point: rows by label class (real, written, silver, none), with
    words, model readings and their thinking summaries, our own model's shadow answers, by language; and rows by origin."""
    out = ["## m7_routing: the routing model's data, per routing point (train and dev read; sealed counted, not read)", ""]
    if not routing:
        return out + ["- no set yet (python -m dclab_rnd.store run, or python -m dclab_rnd.store.sets.routing)", ""]
    table = {p: {**{c: int(v.get(c, 0) or 0) for c in ROUTING_COLUMNS if c not in LABELS}, **{c: int((v.get("label") or {}).get(c, 0)) for c in LABELS}}
             for p, v in sorted(routing.items())}
    origins = sorted({o for v in routing.values() for o in (v.get("origin") or {})})
    out += _table("routing point", table, ROUTING_COLUMNS) + [""]
    out += _table("origin", {p: {o: int((v.get("origin") or {}).get(o, 0)) for o in origins} for p, v in sorted(routing.items())},
                  tuple(origins)) + [""]
    out += ["Columns: real = a person's label on a real request, written = a writer's, an eval's or a script's truth, silver = Claude's "
            "reading of an unconfirmed real request (apart, weighted lower), text = with the person's words (train consent), thinking = "
            "model readings with their API thinking summary, label_thinking = Claude's silver labels carrying its API thinking summary, "
            "own_model = with our own model's shadow answer.", ""]
    return out


LABELS = ("real", "written", "silver", "none")


def summary(doc: dict[str, Any]) -> dict[str, Any]:
    """What the catalog keeps: per model type its totals and per agent and track its non-zero rows, the sources, the gaps."""
    def nonzero(table: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        return {k: {c: n for c, n in v.items() if n} for k, v in table.items() if any(v.values())}

    return {"schema_version": 1, "built_at": doc.get("built_at"),
            "sources": {k: nonzero(v) for k, v in doc["sources"].items()},
            "model_types": {m: {**{k: v for k, v in t.items() if k not in ("by_agent", "by_track")},
                                "by_agent": nonzero(t["by_agent"]), "by_track": nonzero(t["by_track"])}
                            for m, t in doc["model_types"].items()},
            "routing": doc.get("routing") or {}, "gaps": doc["gaps"]}


def write(layout: Optional[Layout] = None, doc: Optional[dict[str, Any]] = None, copy: Optional[Path] = None) -> dict[str, Any]:
    """The summary into the store's ``CATALOG.json`` as ``coverage`` (the other keys stay), and into the repository's copy for the
    owner's own store (or where ``copy`` says), without its time."""
    from .catalog import COPY

    layout = layout or Layout()
    doc = doc or measure(layout)
    kept = summary(doc)
    for path, value in ((layout.catalog, kept),):
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        write_json(path, {**current, "coverage": value})
    if copy is None and layout.root.resolve() == DEFAULT.resolve():
        copy = COPY
    if copy is not None:
        current = json.loads(Path(copy).read_text(encoding="utf-8")) if Path(copy).exists() else {}
        write_json(Path(copy), {**current, "coverage": {k: v for k, v in kept.items() if k != "built_at"}})
    return kept


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.store.coverage", description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", help="the store's root (default: DCLAB_STORE or training_records/store)")
    parser.add_argument("--workspace", help="the workspace folder (default: the settings' DCLAB_AGENT_HOME)")
    parser.add_argument("--json", action="store_true", help="print the whole report as JSON")
    parser.add_argument("--no-write", action="store_true", help="print only; leave the catalog as it is")
    args = parser.parse_args(argv)
    layout = Layout(args.store) if args.store else Layout()
    doc = measure(layout, home=Path(args.workspace) if args.workspace else None)
    print(json.dumps(doc, indent=1, ensure_ascii=False) if args.json else render(doc))
    if not args.no_write:
        write(layout, doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Capture", "CAPTURE", "TRACK_CAPTURE", "capture_gaps", "measure", "gaps", "render", "summary", "write", "main"]
