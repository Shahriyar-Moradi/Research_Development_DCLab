"""The step-by-step, judge and retrieval sets (32.13), from the shaped records only.

**M6** (:class:`dclab_core.store.Episode`): each walk from a draft to a report (its WorkflowRecord's steps), each project's
graph moves, each job's states, as episodes of (state, options, action, outcome) in time order, the episode's outcome from
its record, a gap in the numbering said.

**M7** (:class:`dclab_core.store.JudgedExample`): a person's correction (what DCLab proposed against what the person made it,
the person's preferred), a fix (the failing run against the run that passed, judged by the test), and G4's verdicts (the
result judged by its measurement). Each says who made the candidate and who judged it; a model judging its own answer with
no other evidence never enters (``candidate_by`` equal to the judge).

**M8** (:class:`dclab_core.store.RetrievalExample`): each search an agent made through a tool (the query, the passages
offered), with whether the answer that used them passed when the record says.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Optional

from dclab_core.datasets import split_of
from dclab_core.store import Episode, EpisodeStep, JudgedExample, Message, PreferencePair, RetrievalExample, StoreRecord

from ..layout import Layout
from . import write_set

SEARCH_TOOLS = ("find_examples", "search_evidence", "find_pitfalls", "evidence_search")


def _code(value: Any) -> str:
    return re.sub(r"[^a-z0-9_.\-]+", "_", str(value or "none").lower()).strip("_.-")[:80] or "none"


def _episode(eid: str, process: str, steps: list[tuple[str, str, Optional[str]]], first: StoreRecord, others: list[StoreRecord], outcome: Optional[str],
             key: str) -> Optional[Episode]:
    if not steps:
        return None
    built = [EpisodeStep(n=i, state={"step": i, "move": _code(action.split(":")[0])}, action=action[:200], outcome=(out or None) and str(out)[:200], at=at)
             for i, (at, action, out) in enumerate(steps)]
    more = list({r.provenance.entry: r.provenance for r in others if r.provenance.entry != first.provenance.entry}.values())
    return Episode(id=eid, model_type="m6_stepwise_decisions", split=first.split or split_of(key), provenance=first.provenance, more=more,
                   process=process, steps=built, outcome=(outcome or None) and str(outcome)[:200])


def episodes(layout: Layout, choices: Optional[dict] = None) -> list[Episode]:
    from .. import shapers
    from . import allowed, standing

    MODEL, choices = "m6_stepwise_decisions", choices if choices is not None else standing(layout)

    out: list[Episode] = []
    moves: dict[str, list[StoreRecord]] = defaultdict(list)
    for r in (x for x in shapers.records(layout, "workflows") if allowed(x, MODEL, choices)):
        if r.type == "walk" and isinstance(r.body.get("steps"), list):
            steps = [(str(s.get("at")), f"{s.get('kind')}:{s.get('ref')}", (s.get("detail") or {}).get("status")) for s in r.body["steps"] if isinstance(s, dict)]
            ep = _episode(f"m6:walk:{r.id}", "walk", sorted(steps), r, [], r.body.get("outcome"), str(r.body.get("project") or r.body.get("draft") or r.id))
            if ep:
                out.append(ep)
        elif r.type == "graph_move" and r.body.get("project"):
            moves[str(r.body["project"])].append(r)
    for project, rs in sorted(moves.items()):
        rs.sort(key=lambda r: (r.at, r.id))
        steps = [(r.at, f"{r.body.get('move')}:{r.body.get('from')}>{r.body.get('to')}", r.body.get("status")) for r in rs]
        ep = _episode(f"m6:graph:{project}", "graph", steps, rs[0], rs[1:], rs[-1].body.get("to"), f"project:{project}")
        if ep:
            out.append(ep)
    jobs: dict[str, list[StoreRecord]] = defaultdict(list)
    for r in (x for x in shapers.records(layout, "tasks") if allowed(x, MODEL, choices)):
        if r.type == "job_state" and r.body.get("job"):
            jobs[str(r.body["job"])].append(r)
    for job, rs in sorted(jobs.items()):
        rs.sort(key=lambda r: (r.at, r.id))
        steps = [(r.at, f"{(r.body.get('state') or {}).get('status')}:{r.body.get('kind')}", (r.body.get("state") or {}).get("error")) for r in rs]
        ep = _episode(f"m6:job:{job}", "job", steps, rs[0], rs[1:], (rs[-1].body.get("state") or {}).get("status"), f"job:{job}")
        if ep:
            out.append(ep)
    return out


def judged(layout: Layout, choices: Optional[dict] = None) -> list[JudgedExample]:
    from .. import shapers
    from . import allowed, standing

    MODEL, choices = "m7_judges", choices if choices is not None else standing(layout)

    out: list[JudgedExample] = []
    for r in (x for x in shapers.records(layout, "decisions") if allowed(x, MODEL, choices)):
        b = r.body
        if r.type == "correction" and b.get("before") is not None and b.get("after") is not None:
            out.append(JudgedExample(id=f"m7:correction:{r.id}", model_type="m7_judges", split=r.split or split_of(str(b.get("group") or r.id)),
                                     provenance=r.provenance, judged=_code(b.get("what") or "correction"), candidate=b.get("before"), other=b.get("after"),
                                     preferred="other", candidate_by="dclab", judge="person", evidence="person"))
        elif r.type == "decision" and b.get("point") == "workflow.g4":
            out.append(JudgedExample(id=f"m7:g4:{r.id}", model_type="m7_judges", split=r.split or split_of(str(b.get("group") or r.id)), provenance=r.provenance,
                                     judged="result", candidate=b.get("state"), score=1.0 if b.get("value_used") in (True, "passed", "pass") else 0.0,
                                     candidate_by="dclab", judge="g4", evidence="measured"))
    for r in (x for x in shapers.records(layout, "workflows") if allowed(x, MODEL, choices)):  # 32.19: a move the validator refused is a failure, judged; an allowed one passed
        b = r.body
        if r.type == "graph_move" and b.get("status") in ("allowed", "blocked", "refused"):
            out.append(JudgedExample(id=f"m7:move:{r.id}", model_type="m7_judges", split=r.split or split_of(f"project:{b.get('project') or r.id}"),
                                     provenance=r.provenance, judged="step", candidate={"move": b.get("move"), "from": b.get("from"), "to": b.get("to")},
                                     score=1.0 if b.get("status") == "allowed" else 0.0, candidate_by=str(b.get("actor") or "agent"), judge="validator",
                                     evidence="verdict"))
    for r in (x for x in shapers.records(layout, "fixes") if allowed(x, MODEL, choices)):
        b = r.body
        if b.get("status") in ("proposed", "accepted") and b.get("passing_run"):
            out.append(JudgedExample(id=f"m7:fix:{r.id}", model_type="m7_judges", split=r.split or split_of(str(b.get("test") or r.id)), provenance=r.provenance,
                                     judged="code", candidate={"run": b.get("failing_run"), "commit": b.get("failing_commit"), "output": b.get("failing_output")},
                                     other={"run": b.get("passing_run"), "commit": b.get("passing_commit"), "commits": b.get("commits")}, preferred="other",
                                     candidate_by="development", judge="tests", evidence="test", weight=1.0 if b.get("status") == "accepted" else 0.5))
    return [j for j in out if j.candidate_by != j.judge]  # a candidate judged by its own maker with no other evidence: never


def preferences(layout: Layout, choices: Optional[dict] = None) -> list[PreferencePair]:
    """32.19: each correction as a preference pair: what the person made it (chosen) over what DCLab proposed (rejected)."""
    from .. import shapers
    from . import allowed, standing

    MODEL, choices = "m7_judges", choices if choices is not None else standing(layout)

    out = []
    for r in (x for x in shapers.records(layout, "decisions") if allowed(x, MODEL, choices)):
        b = r.body
        if r.type != "correction" or b.get("before") is None or b.get("after") is None:
            continue
        ask = Message(role="user", content=f"Propose the {b.get('what') or 'answer'} for this case.")
        out.append(PreferencePair(id=f"m7:preference:{r.id}", model_type="m7_judges", split=r.split or split_of(str(b.get("group") or r.id)),
                                  provenance=r.provenance, task=f"correction:{_code(b.get('what'))}", prompt=[ask],
                                  chosen=Message(role="assistant", content=json.dumps(b["after"], ensure_ascii=False, default=str),
                                                 no_reasoning="the person's own choice: their reason, when given, is in the record"),
                                  rejected=Message(role="assistant", content=json.dumps(b["before"], ensure_ascii=False, default=str),
                                                   no_reasoning="DCLab's proposal"),
                                  decided_by="person"))
    return out


def retrievals(layout: Layout, choices: Optional[dict] = None) -> list[RetrievalExample]:
    from .. import shapers
    from . import allowed, standing

    MODEL, choices = "m8_retrieval", choices if choices is not None else standing(layout)

    out: list[RetrievalExample] = []
    kept = [x for x in shapers.records(layout, "actions") if allowed(x, MODEL, choices)]
    by_run: dict[str, list[StoreRecord]] = defaultdict(list)  # 32.23: each search with the answer that followed it in its run
    for r in kept:
        if r.provenance.source.startswith("searches/") and r.body.get("what") in ("search", "answer"):
            by_run[str(r.body.get("run") or r.body.get("conversation") or r.id)].append(r)
    for run, rs in sorted(by_run.items()):
        rs.sort(key=lambda r: (r.at, r.id))
        for i, r in enumerate(rs):
            if r.body.get("what") != "search" or not r.body.get("query"):
                continue
            answer = next((a for a in rs[i + 1:] if a.body.get("what") == "answer"), None)
            offered = [str(o) for o in r.body.get("offered") or [] if o]
            cited = [c for c in (answer.body.get("cited") if answer else []) or [] if c in offered]
            out.append(RetrievalExample(id=f"m8:{r.id}", model_type="m8_retrieval", split=r.split or split_of(run), provenance=r.provenance,
                                        corpus=str(r.body.get("corpus") or "evidence")[:80], query=str(r.body["query"])[:2000], offered=offered,
                                        used=cited, cited=cited, passed=answer.body.get("passed") if answer else None))
    for r in kept:
        b = r.body
        tool = str(b.get("tool") or "")
        if tool not in SEARCH_TOOLS:
            continue
        args = b.get("arguments") or {}
        result = b.get("result") or {}
        hits = result.get("results") if isinstance(result, dict) else result if isinstance(result, list) else []
        offered = [str(h.get("record_id") or h.get("id") or h.get("title") or "")[:200] for h in hits or [] if isinstance(h, dict)]
        query = str(args.get("query") or args.get("question") or "")
        if not query:
            continue
        out.append(RetrievalExample(id=f"m8:{r.id}", model_type="m8_retrieval", split=r.split or split_of(str(b.get("run") or r.id)), provenance=r.provenance,
                                    corpus={"search_evidence": "evidence", "evidence_search": "evidence", "find_examples": "examples"}.get(tool, "pitfalls"),
                                    query=query[:2000], offered=[o for o in offered if o],
                                    passed=b.get("answer_passed") if isinstance(b.get("answer_passed"), bool) else None))  # the answer's verdict, when recorded
    return out


def build(layout: Optional[Layout] = None) -> dict[str, Any]:
    from . import standing

    layout = layout or Layout()
    choices = standing(layout)
    report: dict[str, Any] = {"sets": {}}
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for e in episodes(layout, choices):
        groups[("m6_stepwise_decisions", e.process)].append(e)
    for j in judged(layout, choices):
        groups[("m7_judges", j.judged if j.judged in ("code", "result") else "correction")].append(j)
    for p in preferences(layout, choices):
        groups[("m7_judges", "preferences")].append(p)
    for x in retrievals(layout, choices):
        groups[("m8_retrieval", x.corpus)].append(x)
    abouts = {"m6_stepwise_decisions": "Processes decided step by step: each step's state, action and outcome, the episode's outcome from its record.",
              "m7_judges": "Candidates and what judged them: a person's correction, a test's verdict, a measurement; never a maker judging itself.",
              "m8_retrieval": "Searches: the query and the passages offered (and used, when the record says)."}
    for (model_type, folder), examples in sorted(groups.items()):
        sources = {e.provenance.entry: {"entry": e.provenance.entry, "sha256": e.provenance.sha256} for e in examples}
        m = write_set(layout, model_type, folder, examples, sources=sources.values(), about=abouts[model_type])
        report["sets"][f"{model_type}/{folder}"] = m.counts
    return report


__all__ = ["build", "episodes", "judged", "preferences", "retrievals"]
