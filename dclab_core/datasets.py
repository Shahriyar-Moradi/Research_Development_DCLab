"""Training and evaluation sets from real use (package 19.12): builders over the decision ledger and the session record.

Every builder reads what a consented export already let through (``dclab_rnd.consent``: redacted text, no cells, no
prompts, no person) and returns its rows with a :class:`DatasetManifest`: the builder and its version, when, on what
consent, every source by id, the counts, and how it was split. The builders:

* :func:`decisions`: one set per decision point: the state, the options, every answer, the value used, and each
  label kept apart by its source (benchmark, measured, a person's correction, blind), with the label that counts
  (``ledger.effective_label``). Rows are grouped (by dataset, else project, else draft) and whole groups go to train,
  dev or a **sealed** partition kept in its own file, never used for tuning.
* :func:`conversations`: Home conversations whose problem the person accepted (the draft was built), as chat
  examples of understanding a problem.
* :func:`preferences`: a proposal and the person's choice in its place, as preference pairs.
* :func:`trajectories`: the steps of an agent run (tool, verdict, state), never a tool's arguments.
* :func:`workflows`: a problem card and a data card, the solution workflow drawn for them, the solution and the
  pipeline the project ran.
* :func:`failures`: the sessions a failure detector flagged (``signals``), as candidate benchmark cases; only those a
  reviewer accepted are cases, and they go to a new benchmark version, never into a sealed set.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any, ClassVar, Literal, Optional

from pydantic import Field, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

VERSION = "1"
Builder = Literal["decisions", "conversations", "preferences", "trajectories", "workflows", "failures", "development"]
SourceKind = Literal["decision", "label", "event", "visit", "conversation", "run", "project", "draft", "package", "flag"]
SPLITS = (("sealed", 0.2), ("dev", 0.4))  # a group's hash below 0.2: sealed; below 0.4: dev; the rest train
SALT = "dclab-datasets-v1"  # one salt for every set: a project is sealed (or not) everywhere at once
LABEL_SOURCES = ("benchmark", "measured", "user_decision_exposed", "blind", "model")  # model: a second opinion, never the effective label


class Source(Contract):
    kind: SourceKind
    id: str = Field(min_length=1, max_length=120)


class DatasetManifest(Versioned):
    """What a dataset holds and where every row came from."""

    schema_version: Literal["dclab_core.dataset_manifest.v2"] = _version("dataset_manifest", 2)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.dataset_manifest.v1"})  # v2 added levels
    builder: Builder
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=20)
    built_at: str
    workspace: Optional[str] = Field(None, max_length=64)
    use: Optional[Literal["improve", "train"]] = Field(None, description="The consent it was built under; None for the owner's own records.")
    consent: str = Field(max_length=300, description="The basis, in words.")
    rows: int = Field(ge=0)
    splits: dict[str, int] = Field(default_factory=dict)
    counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    sources: list[Source] = Field(default_factory=list, max_length=2_000_000)
    policy: str = Field(max_length=600, description="How rows were grouped and split.")
    limits: list[str] = Field(default_factory=list, max_length=20)
    levels: dict[str, int] = Field(default_factory=dict, description="v2 (23.1): rows by the consent level their records were written under.")

    @model_validator(mode="before")
    @classmethod
    def _read_v1(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("schema_version") == "dclab_core.dataset_manifest.v1":
            data = {**data, "schema_version": "dclab_core.dataset_manifest.v2"}
        return data


CONTRACTS["dataset_manifest"] = DatasetManifest


def lowest(levels: Iterable[Any]) -> str:
    """The lowest consent level among records (``operate`` when any is unknown): what a row built from them is stamped."""
    order = {"operate": 0, "improve": 1, "train": 2}
    found = [order.get(str(x), 0) for x in levels]
    return "operate" if not found else {v: k for k, v in order.items()}[min(found)]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def split_of(group: str, salt: str = SALT) -> str:
    """A whole group's split, from its hash: the same group always lands in the same split."""
    h = int(hashlib.sha256(f"{salt}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return next((name for name, bound in SPLITS if h < bound), "train")


def group_of(project: Optional[str] = None, draft: Optional[str] = None, dataset: Optional[str] = None,
             drafts: Optional[dict[str, str]] = None, fallback: str = "") -> str:
    """The group a row belongs to, the same in every set: its dataset, else its project, else the project its draft
    became, else its draft. A table's draft and its project are one group, so one table never sits in two splits."""
    if dataset:
        return f"dataset:{dataset}"
    project = project or (drafts or {}).get(draft or "")
    if project:
        return f"project:{project}"
    return f"draft:{draft}" if draft else fallback


def sources_of(rows: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(kind, ident) for row in rows for kind, ident in row.get("sources") or []]


def refit(m: DatasetManifest, rows: list[dict[str, Any]], note: Optional[str] = None) -> DatasetManifest:
    """The manifest of the rows kept, after a host refused some: rows, splits and sources recounted from them."""
    fresh = manifest(m.builder, m.name, rows, sources_of(rows), use=m.use, consent=m.consent, policy=m.policy, workspace=m.workspace,
                     counts=m.counts, limits=[*m.limits, note] if note else m.limits)
    return fresh.model_copy(update={"built_at": m.built_at})


def manifest(builder: str, name: str, rows: list[dict[str, Any]], sources: Iterable[tuple[str, str]], *, use: Optional[str], consent: str,
             policy: str, workspace: Optional[str] = None, counts: Optional[dict[str, dict[str, int]]] = None,
             limits: Iterable[str] = ()) -> DatasetManifest:
    seen, out = set(), []
    for kind, ident in sources:
        if ident and (kind, ident) not in seen:
            seen.add((kind, ident))
            out.append(Source(kind=kind, id=str(ident)[:120]))
    return DatasetManifest(builder=builder, name=name[:120], version=VERSION, built_at=now(), workspace=workspace, use=use, consent=consent[:300],
                           rows=len(rows), splits=dict(Counter(r.get("split", "train") for r in rows)), counts=counts or {}, sources=out,
                           policy=policy[:600], limits=list(limits)[:20], levels=dict(Counter(str(r.get("consent") or "unknown") for r in rows)))


# ------------------------------------------------------------------------------------------------ (a) decisions

def _group(doc: dict[str, Any], drafts: Optional[dict[str, str]] = None) -> str:
    return group_of(doc.get("project"), doc.get("draft") or doc.get("conversation"), doc.get("dataset"), drafts, f"record:{doc.get('id')}")


def decisions(docs: Iterable[dict[str, Any]], point: str, *, workspace: Optional[str] = None, consent: str = "train",
              drafts: Optional[dict[str, str]] = None) -> tuple[list[dict[str, Any]], DatasetManifest]:
    """The decision set of one point from exported decision records (``consent.export_decisions``: each with its
    ``labels``). Each label source in its own column; ``label`` the one that counts; whole groups split."""
    from dclab_core import ledger

    rows = []
    by_source: Counter[str] = Counter()
    for doc in docs:
        if doc.get("point") != point:
            continue
        labels = [ledger.DecisionLabel.model_validate(x) for x in doc.get("labels") or []]
        record = ledger.DecisionRecord.model_validate({k: v for k, v in doc.items() if k != "labels"})
        current = ledger.current_labels(labels)
        apart: dict[str, Any] = {s: None for s in LABEL_SOURCES}
        for source in LABEL_SOURCES:
            mine = [x for x in current if x.source == source]
            if mine:
                last = max(mine, key=lambda x: x.at)
                apart[source] = ({"value": last.value, "weight": last.weight} if source == "blind" else
                                 {"value": last.value, "weight": last.weight, "by": last.by} if source == "model" else last.value)
                by_source[source] += 1
        effective = ledger.effective_label(record, labels)
        group = _group(doc, drafts)
        rows.append({"record": record.id, "point": point, "subject": doc.get("subject"), "state": record.state, "options": record.options,
                     "consent": record.consent,
                     "answers": [{"by": a.by, "answerer": a.answerer, "value": a.value, "confidence": a.confidence} for a in record.answers],
                     "value_used": record.value_used, "used": record.used, "labels": apart,
                     "label": {"value": effective.value, "source": effective.source} if effective is not None else None,
                     "group": group, "split": split_of(group), "sources": [("decision", record.id)] + [("label", x.id) for x in labels]})
    by_source["unlabelled"] = sum(1 for r in rows if r["label"] is None)
    return rows, manifest("decisions", f"decisions.{point}", rows, sources_of(rows), use="train", consent=consent, workspace=workspace,
                          counts={"labels_by_source": dict(by_source)},
                          policy="Grouped by dataset, else project (a draft with the project it became), else draft; whole groups to "
                                 "sealed (20%), dev (20%) or train by the group's hash, the same in every set. The sealed partition is a "
                                 "file of its own, never used for tuning.",
                          limits=["Labels are kept apart by source; `label` follows ledger.effective_label (blind, benchmark, measured, "
                                  "then a person's correction)."])


# ------------------------------------------------------------------------------------------------ (b) conversations

def _by_conversation(events: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        c = (e.get("ids") or {}).get("conversation")
        if c:
            out[c].append(e)
    for c in out:
        out[c].sort(key=lambda e: (e.get("at") or "", e.get("seq") or 0, e.get("id") or ""))
    return out


SYSTEM = ("You are DCLab's Home agent. Understand the person's problem before anything is built: ask one question at a time, "
          "about what the data and the words leave open, and say why it matters.")


def conversations(events: Iterable[dict[str, Any]], *, workspace: Optional[str] = None, consent: str = "train",
                  drafts: Optional[dict[str, str]] = None) -> tuple[list[dict[str, Any]], DatasetManifest]:
    """Chat examples from Home conversations that ended in a build (the person accepted the problem as understood)."""
    rows = []
    for conversation, items in _by_conversation(events).items():
        kinds = [e["kind"] for e in items]
        if "draft.created" not in kinds or "draft.built" not in kinds:
            continue
        messages = [{"role": "system", "content": SYSTEM}]
        for e in items:
            p = e.get("payload") or {}
            if e["kind"] == "draft.created" and p.get("problem"):
                messages.append({"role": "user", "content": str(p["problem"])})
            elif e["kind"] == "chat.person" and p.get("text") and p.get("surface", "home") == "home":
                messages.append({"role": "user", "content": str(p["text"])})
            elif e["kind"] == "chat.agent" and p.get("text") and p.get("surface", "home") == "home":
                messages.append({"role": "assistant", "content": str(p["text"])})
            elif e["kind"] == "draft.built":
                break
        merged: list[dict[str, str]] = []
        for m in messages:  # two turns in a row from one side are one turn
            if merged and merged[-1]["role"] == m["role"] and m["role"] != "system":
                merged[-1] = {"role": m["role"], "content": merged[-1]["content"] + "\n" + m["content"]}
            else:
                merged.append(m)
        if not any(m["role"] == "assistant" for m in merged):
            continue
        while merged and merged[-1]["role"] != "assistant":
            merged.pop()
        project = next(((e.get("ids") or {}).get("project") for e in items if (e.get("ids") or {}).get("project")), None)
        group = group_of(project, conversation, None, drafts, f"conversation:{conversation}")
        rows.append({"conversation": conversation, "messages": merged, "group": group, "split": split_of(group), "consent": lowest(e.get("consent") for e in items),
                     "sources": [("conversation", conversation)] + [("event", e.get("id")) for e in items]})
    return rows, manifest("conversations", "conversations.problem_understanding", rows, sources_of(rows), use="train", consent=consent, workspace=workspace,
                          policy="One example per built draft; grouped with the project it became, split as every set.",
                          limits=["Text is as the export redacted it.", "A conversation not built is left out: its understanding was not accepted."])


# ------------------------------------------------------------------------------------------------ (c) preferences

def preferences(docs: Iterable[dict[str, Any]], *, workspace: Optional[str] = None, consent: str = "train",
                drafts: Optional[dict[str, str]] = None) -> tuple[list[dict[str, Any]], DatasetManifest]:
    """A rule's or a model's answer and the person's different choice that was used instead, as a preference pair."""
    from dclab_core import ledger

    rows = []
    for doc in docs:
        answers = doc.get("answers") or []
        person = next((a for a in answers if a.get("by") == "person"), None)
        proposed = next((a for a in answers if a.get("by") != "person" and a.get("value") is not None), None)
        if person is None or proposed is None or person.get("value") == proposed.get("value") or doc.get("value_used") != person.get("value"):
            continue
        point = ledger.POINTS.get(doc.get("point"))
        group = _group(doc, drafts)
        rows.append({"record": doc.get("id"), "point": doc.get("point"), "question": point.about if point is not None else doc.get("point"), "consent": doc.get("consent"),
                     "state": doc.get("state"), "options": doc.get("options"), "chosen": person.get("value"),
                     "rejected": proposed.get("value"), "rejected_by": proposed.get("by"), "reason": person.get("reason"),
                     "group": group, "split": split_of(group), "sources": [("decision", doc.get("id"))]})
    return rows, manifest("preferences", "preferences.corrections", rows, sources_of(rows), use="train", consent=consent, workspace=workspace,
                          counts={"by_point": dict(Counter(r["point"] for r in rows))},
                          policy="Grouped as the decision sets; whole groups split by their hash.",
                          limits=["A correction says the person preferred it, not that it was right: the decision sets hold the labels."])


# ------------------------------------------------------------------------------------------------ (d) trajectories

def trajectories(events: Iterable[dict[str, Any]], *, workspace: Optional[str] = None, consent: str = "train",
                 drafts: Optional[dict[str, str]] = None) -> tuple[list[dict[str, Any]], DatasetManifest]:
    """The steps of each agent run: its state, the tool it chose, the verdict; never a tool's arguments or result."""
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        run = (e.get("ids") or {}).get("run")
        if run and e.get("kind") in ("agent.step", "tool.call"):
            runs[run].append(e)
    rows = []
    for run, items in runs.items():
        items.sort(key=lambda e: (e.get("at") or "", e.get("seq") or 0))
        steps = []
        for e in items:
            p = e.get("payload") or {}
            if e["kind"] == "agent.step":
                steps.append({"agent": p.get("agent"), "n": p.get("n"), "state": p.get("state"), "tool": p.get("tool"), "verdict": p.get("verdict")})
            else:
                steps.append({"tool": p.get("tool"), "argument_names": p.get("argument_names"), "verdict": p.get("verdict"), "code": p.get("code")})
        if len(steps) < 2:
            continue
        ids = next((e.get("ids") or {} for e in items if (e.get("ids") or {}).get("project") or (e.get("ids") or {}).get("conversation")), {})
        group = group_of(ids.get("project"), ids.get("conversation"), None, drafts, f"run:{run}")
        rows.append({"run": run, "agent": next((s.get("agent") for s in steps if s.get("agent")), None), "steps": steps, "consent": lowest(e.get("consent") for e in items),
                     "group": group, "split": split_of(group), "sources": [("run", run)] + [("event", e.get("id")) for e in items]})
    return rows, manifest("trajectories", "trajectories.agent_runs", rows, sources_of(rows), use="train", consent=consent, workspace=workspace,
                          policy="One example per run with two steps or more; grouped by its project, split as every set.",
                          limits=["A tool's arguments and results are cell values: they never leave, only their names."])


# ------------------------------------------------------------------------------------------------ (e) workflows

def workflows(items: Iterable[dict[str, Any]], *, workspace: Optional[str] = None, consent: str = "train") -> tuple[list[dict[str, Any]], DatasetManifest]:
    """Problem card and data card, the workflow drawn, the solution saved and the stages the project ran, one example
    per built project: ``items`` as the host assembles them (``draft``, ``project``, ``problem_card``, ``data_card``,
    ``workflow``, ``solution``, ``pipeline``)."""
    rows = []
    for item in items:
        if not item.get("problem_card") or not item.get("solution"):
            continue
        group = group_of(item.get("project"), item.get("draft"), None, None, "workflow")
        runs = [("run", stage["run"]) for stage in item.get("pipeline") or [] if stage.get("run")]
        rows.append({**{k: item.get(k) for k in ("draft", "project", "problem_card", "data_card", "workflow", "solution", "forbidden", "pipeline")},
                     "group": group, "split": split_of(group), "sources": [("draft", item.get("draft")), ("project", item.get("project"))] + runs})
    return rows, manifest("workflows", "workflows.problem_to_pipeline", rows, sources_of(rows), use="train", consent=consent, workspace=workspace,
                          policy="One example per built project; grouped by the project, split as every set.",
                          limits=["The code of a pipeline is not in it yet: the stages and their choices are."])


# ------------------------------------------------------------------------------------------------ (f) failures

def failures(sessions: Iterable[dict[str, Any]], reviews: Optional[dict[str, dict[str, Any]]] = None, *, workspace: Optional[str] = None,
             consent: str = "improve") -> tuple[list[dict[str, Any]], DatasetManifest]:
    """Flagged sessions as candidate benchmark cases (``sessions``: visit, flags, problem, family, outcome, data_card, and
    the person's answers by question field).
    A candidate a reviewer accepted (``reviews[id]["verdict"] == "accept"``) is a case; never into a sealed set."""
    reviews = reviews or {}
    rows = []
    for s in sessions:
        flags = s.get("flags") or []
        if not flags:
            continue
        ident = "FC-" + hashlib.sha256(f"{s.get('visit')}:{','.join(sorted(f['detector'] for f in flags))}".encode()).hexdigest()[:12]
        review = reviews.get(ident) or {}
        rows.append({"id": ident, "visit": s.get("visit"), "detectors": sorted({f["detector"] for f in flags}),
                     "flags": [{"detector": f["detector"], "severity": f.get("severity"), "message": f.get("message")} for f in flags],
                     "problem": s.get("problem"), "family": s.get("family"), "outcome": s.get("outcome"), "data_card": s.get("data_card"),
                     "answers": dict(s.get("answers") or {}),  # 21.5: the person's answers by question, to replay the session
                     "review": review.get("verdict") or "pending", "reviewed_by": review.get("by"), "split": "candidate",
                     "sources": [("visit", s.get("visit"))] + [("flag", f"{s.get('visit')}:{f['detector']}") for f in flags]
                     + ([("draft", s["draft"])] if s.get("draft") and s.get("data_card") else [])})
    accepted = sum(1 for r in rows if r["review"] == "accept")
    return rows, manifest("failures", "failures.benchmark_candidates", rows, sources_of(rows), use="improve", consent=consent, workspace=workspace,
                          counts={"review": dict(Counter(r["review"] for r in rows)), "by_detector": dict(Counter(d for r in rows for d in r["detectors"]))},
                          policy=f"Every flagged session a candidate; {accepted} accepted by a reviewer go to a new benchmark version "
                                 "as dev cases, never into a sealed set.",
                          limits=["A candidate is not a case until a reviewer accepts it."])


__all__ = ["group_of", "sources_of", "refit", "SALT", "DatasetManifest", "Source", "decisions", "conversations", "preferences", "trajectories", "workflows", "failures", "manifest",
           "split_of", "LABEL_SOURCES", "VERSION"]
