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


def _episode(eid: str, process: str, steps: list[tuple], first: StoreRecord, others: list[StoreRecord], outcome: Optional[str],
             key: str) -> Optional[Episode]:
    """``steps``: (at, action, outcome) or (at, action, outcome, why); ``why`` is the step's reasoning written by code from its own
    record (from_record), never a model's and never the outcome restated."""
    if not steps:
        return None
    built = []
    for i, step in enumerate(steps):
        at, action, out = step[:3]
        why = step[3] if len(step) > 3 and step[3] else None
        built.append(EpisodeStep(n=i, state={"step": i, "move": _code(action.split(":")[0])}, action=action[:200],
                                 outcome=(out or None) and str(out)[:200], at=at,
                                 **({"reasoning": str(why)[:2000], "reasoning_source": "from_record"} if why else {})))
    more = list({r.provenance.entry: r.provenance for r in others if r.provenance.entry != first.provenance.entry}.values())
    return Episode(id=eid, model_type="m6_stepwise_decisions", split=first.split or split_of(key), provenance=first.provenance, more=more,
                   process=process, steps=built, outcome=(outcome or None) and str(outcome)[:200])


def _move_why(previous: dict[str, Any]) -> Optional[str]:
    """What the agent knew before a move: the previous move, the validator's verdict on it and the checks it failed (the
    reason it moved on or tried again). Never the move's own verdict, which is its outcome."""
    if not previous.get("move"):
        return None
    checks = ", ".join(map(str, previous.get("failed_checks") or []))
    rules = ", ".join(map(str, previous.get("rules") or []))
    return (f"Before this move: {previous.get('move')} from {previous.get('from')} to {previous.get('to')} was "
            f"{previous.get('status')} by the workflow validator" + (f", failing {checks}" if checks else "")
            + (f" (rules {rules})" if rules else "") + ".")


def episodes(layout: Layout, choices: Optional[dict] = None) -> list[Episode]:
    """M6: each process decided step by step, from each source's current records (a walk kept again is one episode, a rewritten
    log read once), per workspace: every walk; every project's graph moves (file or PostgreSQL rows) and its stage runs in the
    order they started; every team's walk node by node (its board's state, open questions and blockers); every job's states."""
    from ..service import DONE, _current, _latest
    from . import allowed, standing

    MODEL, choices = "m6_stepwise_decisions", choices if choices is not None else standing(layout)

    out: list[Episode] = []
    moves: dict[tuple[str, str], list[tuple[StoreRecord, dict[str, Any]]]] = defaultdict(list)
    boards: dict[tuple[str, str], list[StoreRecord]] = defaultdict(list)
    for r in (x for x in _current(layout, "workflows") if allowed(x, MODEL, choices)):
        ws = r.provenance.workspace
        if r.type == "walk" and isinstance(r.body.get("steps"), list):
            steps = [(str(s.get("at")), f"{s.get('kind')}:{s.get('ref')}", (s.get("detail") or {}).get("status")) for s in r.body["steps"] if isinstance(s, dict)]
            ep = _episode(f"m6:walk:{r.id}", "walk", sorted(steps), r, [], r.body.get("outcome"), str(r.body.get("project") or r.body.get("draft") or r.id))
            if ep:
                out.append(ep)
        elif r.type == "graph_move":
            move = r.body.get("entry") if isinstance(r.body.get("entry"), dict) else r.body  # a PostgreSQL row holds it in ``entry``
            parts = r.provenance.source.split("/")
            pid = r.body.get("project") or (parts[1] if parts[0] == "projects" and len(parts) > 1 else None)
            if pid:
                moves[(ws, str(pid))].append((r, move))
        elif r.provenance.source.endswith("orchestra/boards.jsonl") and r.body.get("walk"):
            boards[(ws, str(r.body["walk"]))].append(r)
    for (ws, project), rs in sorted(moves.items()):
        rs.sort(key=lambda x: (str(x[1].get("at") or x[0].at), x[0].id))
        # each move's reasoning is what was known before it: the previous move's verdict and the checks it failed (why the
        # agent moved on, or tried again), never this move's own verdict, which is its outcome (the review of C2)
        steps = [(str(m.get("at") or r.at), f"{m.get('move')}:{m.get('from')}>{m.get('to')}", m.get("status"),
                  _move_why(rs[i - 1][1]) if i else None) for i, (r, m) in enumerate(rs)]
        ep = _episode(f"m6:graph:{ws}:{project}", "graph", steps, rs[0][0], [r for r, _ in rs[1:]], rs[-1][1].get("to"), f"project:{project}")
        if ep:
            out.append(ep)
    for (ws, pid), p in sorted(_latest(layout).items()):  # a project's stages in the order they ran, each with its status
        records = p.get("records") or {}
        ran = sorted((str(b.get("started_at") or b.get("started") or ""), stage) for stage, b in p["stages"].items() if b.get("status"))
        ran = [(at, stage) for at, stage in ran if at and stage in records and allowed(records[stage], MODEL, choices)]
        if not ran:
            continue
        steps = []
        for at, stage in ran:
            b = p["stages"][stage]
            chosen = (b.get("evidence") or {}).get("selected_model") if stage == "models" else None
            rule = (b.get("evidence") or {}).get("selection_rule")
            steps.append((at, f"{stage}:{_code(chosen)}" if chosen else stage, b.get("status"),
                          f"The stage {stage} chose by its rule: {str(rule)[:600]}" if rule else None))
        last = p["stages"][ran[-1][1]]
        ep = _episode(f"m6:stages:{ws}:{pid}", "stages", steps, records[ran[0][1]], [records[s] for _, s in ran[1:]],
                      "completed" if all(p["stages"][s].get("status") in DONE for _, s in ran) else last.get("status"), f"project:{pid}")
        if ep:
            out.append(ep)
    for (ws, walk), rs in sorted(boards.items()):  # a team's walk node by node
        rs.sort(key=lambda r: (r.at, r.id))
        steps = []
        before = None  # the board as the previous node left it: what the team knew when it chose this node
        for r in rs:
            board = r.body.get("board") if isinstance(r.body.get("board"), dict) else {}
            open_q, blockers = len(board.get("open_questions") or []), len(board.get("blockers") or [])
            why = None
            if before is not None:
                why = (f"The board before this step: {before.get('state')}, with {len(before.get('open_questions') or [])} questions open "
                       f"and {len(before.get('blockers') or [])} blockers.")
            steps.append((r.at, f"{r.body.get('node')}:{r.body.get('what')}", f"{board.get('state')}; open {open_q}; blockers {blockers}", why))
            before = board
        thread = str(((rs[0].body.get("board") or {}).get("thread")) or walk)
        ep = _episode(f"m6:team:{ws}:{walk}", f"team/{_code(rs[0].body.get('team'))}", steps, rs[0], rs[1:],
                      str((rs[-1].body.get("board") or {}).get("state") or ""), thread)
        if ep:
            out.append(ep)
    jobs: dict[tuple[str, str], list[StoreRecord]] = defaultdict(list)
    for r in (x for x in _current(layout, "tasks") if allowed(x, MODEL, choices)):
        if r.type == "job_state" and r.body.get("job"):
            jobs[(r.provenance.workspace, str(r.body["job"]))].append(r)
    for (ws, job), rs in sorted(jobs.items()):
        rs.sort(key=lambda r: (r.at, r.id))
        steps = [(r.at, f"{(r.body.get('state') or {}).get('status')}:{r.body.get('kind')}", (r.body.get("state") or {}).get("error")) for r in rs]
        ep = _episode(f"m6:job:{ws}:{job}", "job", steps, rs[0], rs[1:], (rs[-1].body.get("state") or {}).get("status"), f"job:{job}")
        if ep:
            out.append(ep)
    return out


def _scripted(scripted: Optional[frozenset[str]]) -> frozenset[str]:
    from ..keepers.workspaces import scripted_names

    return scripted_names() if scripted is None else scripted


def judged(layout: Layout, choices: Optional[dict] = None, scripted: Optional[frozenset[str]] = None) -> list[JudgedExample]:
    from ..service import _current
    from . import allowed, standing

    MODEL, choices = "m7_judges", choices if choices is not None else standing(layout)
    scripted = _scripted(scripted)  # S8: a scripted person's correction is no human's judgement

    out: list[JudgedExample] = []
    for r in (x for x in _current(layout, "decisions") if allowed(x, MODEL, choices)):
        b = r.body
        if r.type == "correction" and b.get("before") is not None and b.get("after") is not None and r.provenance.workspace not in scripted:
            out.append(JudgedExample(id=f"m7:correction:{r.id}", model_type="m7_judges", split=r.split or split_of(str(b.get("group") or r.id)),
                                     provenance=r.provenance, judged=_code(b.get("what") or "correction"), candidate=b.get("before"), other=b.get("after"),
                                     preferred="other", candidate_by="dclab", judge="person", evidence="person"))
        elif r.type == "decision" and b.get("point") == "workflow.g4":
            out.append(JudgedExample(id=f"m7:g4:{r.id}", model_type="m7_judges", split=r.split or split_of(str(b.get("group") or r.id)), provenance=r.provenance,
                                     judged="result", candidate=b.get("state"), score=1.0 if b.get("value_used") in (True, "passed", "pass") else 0.0,
                                     candidate_by="dclab", judge="g4", evidence="measured"))
    for r in (x for x in _current(layout, "workflows") if allowed(x, MODEL, choices)):  # 32.19: a move the validator refused is a failure, judged; an allowed one passed
        b = r.body.get("entry") if isinstance(r.body.get("entry"), dict) else r.body  # a PostgreSQL row holds the move in ``entry``
        if r.type == "graph_move" and b.get("status") in ("allowed", "blocked", "refused"):
            out.append(JudgedExample(id=f"m7:move:{r.id}", model_type="m7_judges", split=r.split or split_of(f"project:{r.body.get('project') or r.id}"),
                                     provenance=r.provenance, judged="step", candidate={"move": b.get("move"), "from": b.get("from"), "to": b.get("to")},
                                     score=1.0 if b.get("status") == "allowed" else 0.0, candidate_by=str(b.get("actor") or "agent"), judge="validator",
                                     evidence="verdict"))
    for r in (x for x in _current(layout, "fixes") if allowed(x, MODEL, choices)):
        b = r.body
        if b.get("status") in ("proposed", "accepted") and b.get("passing_run"):
            out.append(JudgedExample(id=f"m7:fix:{r.id}", model_type="m7_judges", split=r.split or split_of(str(b.get("test") or r.id)), provenance=r.provenance,
                                     judged="code", candidate={"run": b.get("failing_run"), "commit": b.get("failing_commit"), "output": b.get("failing_output")},
                                     other={"run": b.get("passing_run"), "commit": b.get("passing_commit"), "commits": b.get("commits")}, preferred="other",
                                     candidate_by="development", judge="tests", evidence="test", weight=1.0 if b.get("status") == "accepted" else 0.5))
    return [j for j in out if j.candidate_by != j.judge]  # a candidate judged by its own maker with no other evidence: never


def preferences(layout: Layout, choices: Optional[dict] = None, scripted: Optional[frozenset[str]] = None) -> list[PreferencePair]:
    """32.19: each correction as a preference pair: what the person made it (chosen) over what DCLab proposed (rejected);
    never a scripted person's (S8)."""
    from ..service import _current
    from . import allowed, standing

    MODEL, choices = "m7_judges", choices if choices is not None else standing(layout)
    scripted = _scripted(scripted)

    out = []
    for r in (x for x in _current(layout, "decisions") if allowed(x, MODEL, choices)):
        b = r.body
        if r.type != "correction" or b.get("before") is None or b.get("after") is None or r.provenance.workspace in scripted:
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
    from ..service import _current
    from . import allowed, standing

    MODEL, choices = "m8_retrieval", choices if choices is not None else standing(layout)

    out: list[RetrievalExample] = []
    kept = [x for x in _current(layout, "actions") if allowed(x, MODEL, choices)]
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
