"""The first divergence of the lead agent in one kept conversation: the earliest recorded moment where DCLab's code
changed, refused or set what the model was doing. Read from what a run already keeps beside each conversation (the
experience log's ``tool.call`` and ``case.changed`` events and the ledger's ``workflow.stop``), never from the model's words.

Each reason is tied to one place in the code, and found by a fragment of the message that place returns
(``FRAGMENTS``; ``tests/test_harness_ablation.py`` fails when a fragment is no longer in the source, so a reworded
message is noticed here). A refusal of a move that decides what the person gets and matches no fragment is kept as
``other_refusal`` with its words, so it is never read as "no gate fired".

    python -m dclab_rnd.agent_eval.harness_ablation.first_divergence agent_runs/live_tests/<date>-conversation-exam/CEX-033
    python -m dclab_rnd.agent_eval.harness_ablation.first_divergence --recorded-dev      # every dev exam run, plain against case

Not readable from the kept events, so never claimed here: a tool result cut at the runtime's character limit, a reply
rewritten by the number check, and evidence missing from the context (``NOT_DETECTABLE``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

ROOT = Path(__file__).resolve().parents[3]
EXAM_RESULTS = ROOT / "evidence" / "campaigns" / "conversation_exam_v1" / "results"

CASE_AGENT = "dclab_rnd/agents/case_agent.py"
# (reason, the tool whose refusal it is or None for any, a fragment of the refusal, the file that returns it or None when a
# library writes the message). Read in this order: the first fragment found names the reason.
FRAGMENTS: tuple[tuple[str, Optional[str], str, Optional[str]], ...] = (
    # the table tools (found in the first live run, HAB-002: a filtered sum and a date's range could not be asked)
    ("query_cannot_express", None, "a condition (where) belongs to a count or a share", None),  # refused until dclab-core 0.159.0: kept so older runs are still read
    ("query_cannot_express", None, "needs a numeric column", "dclab_core/query.py"),
    ("query_cap", None, "the table has been asked", CASE_AGENT),
    ("query_cap", None, "questions of the table were refused this turn", CASE_AGENT),
    ("bad_arguments", None, "were not valid JSON", "dclab_rnd/agents/registry.py"),
    ("reply_cut", None, "cut at its length limit", "dclab_rnd/agents/runtime.py"),
    # the build's own rules (stage 2 of 2026-11-15 gave each its own words; the first is the one message older runs hold)
    ("build_needs_yes", "case_request_build", "a build needs the person's yes", None),
    ("build_needs_yes", "case_request_build", "holds no yes to building", CASE_AGENT),
    ("build_needs_yes", "case_request_build", "which was not about building", CASE_AGENT),
    ("build_pick_not_yes", "case_request_build", "a pick is not a yes", CASE_AGENT),
    ("build_needs_start", "case_request_build", "a build starts a project", CASE_AGENT),
    ("outcome_not_named", "case_request_build", "has not named", CASE_AGENT),
    ("start_refused_not_ready", "case_start", "the case is not ready", CASE_AGENT),
    ("propose_refused_not_ready", "case_propose", "not yet: ", CASE_AGENT),
    ("route_overruled", None, "the person's need may call for", CASE_AGENT),
    ("options_dropped", "case_propose", "fewer than two options passed the checks", CASE_AGENT),
    ("route_change_refused", "case_start", "the work has started as", CASE_AGENT),
    ("guard_refused", None, "the work has started: tell the person", CASE_AGENT),
    ("closed_list", "case_start", "DCLab cannot run", CASE_AGENT),
    ("closed_list", None, "DCLab has no tool", CASE_AGENT),
    ("closed_list", "case_start", "is not one of", None),  # the registry's schema check: a kind of work or a tool outside the list
    ("closed_list", "case_propose", "is not one of", None),
    ("bad_arguments", None, "Unknown argument(s) for", "dclab_rnd/agents/registry.py"),  # after the closed list: an "is not one of" is that
    ("bad_arguments", None, "Invalid argument for", "dclab_rnd/agents/registry.py"),
    ("advice_refused", "case_start", "the person's need points to data work", CASE_AGENT),
    ("no_data_redirect", "case_start", "has no data yet or prefers not to upload", CASE_AGENT),
    ("chip_bound", None, "the person picked", CASE_AGENT),
    ("needs_table", "case_start", "the table it runs on", CASE_AGENT),
)
# the moves whose refusal changes what the person gets: one that matches no fragment is kept as ``other_refusal`` with its
# words, never read as "no gate fired"
GATED = ("case_start", "case_propose", "case_answer", "case_dashboard", "case_query", "case_request_build")
# reasons read from the shape of the events, not from a message
SHAPE_REASONS = ("route_set_by_code", "options_dropped", "options_trimmed", "other_refusal", "budget_exhausted", "work_done_no_route")
# what the conversation shows when no code intervened before the route
NO_GATE = ("picked_wrong_option", "model_error", "arm_error", "no_route", "no_gate")
NOT_DETECTABLE = ("result_truncated", "reply_rewritten", "context_missing")
REASONS = tuple(dict.fromkeys([*(f[0] for f in FRAGMENTS), *SHAPE_REASONS, *NO_GATE]))
# a refusal that protects the data or the person's choice, against one that asks for the product's shape
SAFETY = frozenset({"needs_table", "chip_bound", "guard_refused", "route_change_refused", "build_needs_yes", "build_pick_not_yes", "outcome_not_named"})
SHOWS = ("case_answer", "case_dashboard")  # the tools whose side effect sets the route in code (case_agent._worked)
_DROPPED = re.compile(r'"refused":\s*\{\s*"')
_TRIMMED = re.compile(r'"refused":\s*\{\s*"[^"]*\(kept\)"')  # since dclab-core 0.159.0: an option kept without the sentence holding a number nobody gave


def events(folder: Path) -> list[dict[str, Any]]:
    """Every experience event kept beside one conversation, in the order they happened."""
    out: list[dict[str, Any]] = []
    for path in sorted(Path(folder).glob("experience/**/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return sorted(out, key=lambda e: (str(e.get("at") or ""), int(e.get("seq") or 0)))


def stops(folder: Path) -> list[dict[str, Any]]:
    """The lead agent's recorded stops (the ledger's ``workflow.stop``): the steps a turn had left when it ended."""
    out = []
    for path in sorted(Path(folder).glob("ledger/decisions/**/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("point") == "workflow.stop" and (record.get("state") or {}).get("agent") == "case":
                    out.append(record)
    return out


def _out_of_budget(stop: Mapping[str, Any]) -> bool:
    """A recorded stop that a budget made (``ledger.capture.stopped``): no step left, or ended by rule at the reply limit."""
    state = stop.get("state") or {}
    by_budget = any(a.get("answerer") == "rule:budget" for a in stop.get("answers") or [])
    return state.get("steps_left") == 0 or by_budget


def moves(evs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The lead agent's tool calls and the changes to the case file's route, options and choice, in order."""
    out: list[dict[str, Any]] = []
    for e in evs:
        payload = e.get("payload") or {}
        if e.get("kind") == "tool.call" and str(payload.get("tool") or "").startswith("case_"):
            out.append({"at": e.get("at"), "seq": e.get("seq"), "what": "tool", "tool": payload["tool"], "verdict": payload.get("verdict"),
                        "arguments": payload.get("arguments") or {}, "result": str(payload.get("result") or "")})
        elif e.get("kind") == "case.changed" and payload.get("field") in ("route", "options", "chosen"):
            out.append({"at": e.get("at"), "seq": e.get("seq"), "what": "case", "field": payload["field"], "by": payload.get("by"),
                        "after": str(payload.get("after") or "")})
    return out


def blocked_runs(ms: Iterable[Mapping[str, Any]]) -> list[list[str]]:
    """The dead ends of one conversation: runs of refused moves with no allowed one between them, three in a row or two of
    different tools (the model asked for one thing, was refused, asked for another, was refused). Each as its tools in order.
    A table question refused for its shape is no dead end (the model asks again in another shape): only the moves that
    decide what the person gets count."""
    deciding = ("case_start", "case_propose", "case_answer", "case_dashboard", "case_request_build", "case_ask")
    runs, current = [], []
    for move in ms:
        if move.get("what") != "tool" or move.get("tool") not in deciding:
            continue
        if move.get("verdict") == "ok":
            if len(current) >= 3 or len(set(current)) >= 2:
                runs.append(current)
            current = []
        else:
            current = [*current, str(move["tool"])]
    if len(current) >= 3 or len(set(current)) >= 2:
        runs.append(current)
    return runs


def counters(folder: Path, conv: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """What code can count in one kept lead-agent conversation without a judge (the gate's zero checks): dead ends, the
    turns that ended on code's own fallback line, and the model's answers that came back empty or cut."""
    from dclab_rnd.agents import case_agent

    folder = Path(folder)
    fallback = (case_agent.FALLBACK, case_agent.FALLBACK_NUMBERS)
    said = [str(t.get("text") or "") for t in (conv or {}).get("transcript") or [] if t.get("role") == "assistant"]
    empty = cut = calls = 0
    for path in sorted(folder.parents[1].glob("model_exchanges/**/*.jsonl")):  # the run keeps its exchanges once, each with its conversation's id
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("purpose") != "case_agent" or not str(row.get("draft") or "").endswith(f":case-{folder.name}") or row.get("outcome") != "ok":
                continue
            calls += 1
            empty += not str(row.get("reply") or "").strip() and not row.get("tool_calls")
            cut += row.get("finish_reason") == "length"
    return {"blocked_runs": blocked_runs(moves(events(folder))), "fallback_lines": sum(any(s.startswith(f) for f in fallback) for s in said),
            "model_calls": calls, "empty_answers": int(empty), "cut_replies": int(cut)}


def reason_of(move: Mapping[str, Any], following: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """The reason code of one move, or None when DCLab's code did nothing to it."""
    if move["what"] == "case":
        # the route appears in the case file as the agent's, and the tool call recorded right after it is a figure shown,
        # not a start: code set it (case_agent._worked)
        if move["field"] == "route" and move.get("by") == "agent" and following and following.get("what") == "tool" and following.get("tool") in SHOWS:
            return "route_set_by_code"
        return None
    result = move.get("result") or ""
    if move.get("verdict") == "ok":
        if move["tool"] == "case_propose" and _TRIMMED.search(result):
            return "options_trimmed"
        return "options_dropped" if move["tool"] == "case_propose" and _DROPPED.search(result) else None
    for reason, tool, fragment, _ in FRAGMENTS:
        if (tool is None or tool == move["tool"]) and fragment in result:
            return reason
    return "other_refusal" if move["tool"] in GATED else None


def _wanted(move: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """What the model asked for in a refused or changed move: the route of a start, the routes of the options."""
    args = move.get("arguments") or {}
    if move.get("tool") == "case_start":
        return {"track": args.get("track"), "tool": args.get("tool")}
    if move.get("tool") == "case_propose":
        return {"tracks": sorted({str(o.get("track")) for o in args.get("options") or [] if isinstance(o, dict)})}
    return None


def _right(wanted: Optional[Mapping[str, Any]], person: Optional[Mapping[str, Any]]) -> Optional[bool]:
    """Whether what the model asked for held the person's right route (None when there is no written route to compare)."""
    if not wanted or not person or not person.get("route"):
        return None
    good = {person["route"], *(person.get("also_right") or [])}
    if "track" in wanted:
        return wanted["track"] in good
    return bool(good & set(wanted.get("tracks") or []))


def read(folder: Path, person: Optional[Mapping[str, Any]] = None, conv: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """One conversation's record: every intervention of code in order, the first one, and what the model wanted there."""
    folder = Path(folder)
    if conv is None and (folder / "conversation.json").exists():
        conv = json.loads((folder / "conversation.json").read_text(encoding="utf-8"))
    person = person or (conv or {}).get("person")
    ms = moves(events(folder))
    found: list[dict[str, Any]] = []
    routed_at: Optional[int] = None
    for i, move in enumerate(ms):
        reason = reason_of(move, ms[i + 1] if i + 1 < len(ms) else None)
        if move["what"] == "case" and move["field"] == "route" and routed_at is None:
            routed_at = i
        if reason:
            wanted = _wanted(move)
            found.append({"reason": reason, "at": move.get("at"), "seq": move.get("seq"), "tool": move.get("tool"), "wanted": wanted,
                          "wanted_right": _right(wanted, person), "safety": reason in SAFETY, "before_route": routed_at is None or routed_at == i,
                          "evidence": (move.get("result") or move.get("after") or "")[:300]})
    if any(_out_of_budget(s) for s in stops(folder)):
        found.append({"reason": "budget_exhausted", "at": None, "seq": None, "tool": None, "wanted": None, "wanted_right": None, "safety": False,
                      "before_route": None, "evidence": "a turn of the lead agent was ended by its budget (steps, replies or time)"})
    route = (conv or {}).get("route")
    showed = any(m["what"] == "tool" and m["tool"] in SHOWS and m["verdict"] == "ok" for m in ms)
    if conv is not None and route is None and showed:
        found.append({"reason": "work_done_no_route", "at": None, "seq": None, "tool": None, "wanted": None, "wanted_right": None, "safety": False,
                      "before_route": None, "evidence": "figures were shown, and no route was recorded for the conversation"})
    gates = [f for f in found if f["before_route"] in (True, None)]
    first = gates[0] if gates else None
    if first is None and conv is not None:  # no code was seen to intervene before the route: the conversation's own outcome
        wrong = _right({"track": route}, person) is False  # only a route that is not the person's is called an error
        if conv.get("ended") == "error":
            reason = "arm_error"
        elif route is None:
            reason = "no_route"
        elif wrong and conv.get("routed_by") == "option":
            reason = "picked_wrong_option"
        elif wrong:
            reason = "model_error"
        else:
            reason = "no_gate"
        first = {"reason": reason, "at": None, "seq": None, "tool": None, "wanted": None, "wanted_right": None, "safety": False,
                 "before_route": None, "evidence": f"ended {conv.get('ended')}, route {route}/{conv.get('tool')} by {conv.get('routed_by')}"}
    return {"id": (person or {}).get("id") or (conv or {}).get("id"), "folder": str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder),
            "events": len(ms), "first": first, "all": found, "counters": counters(folder, conv), "route": route, "tool": (conv or {}).get("tool"), "routed_by": (conv or {}).get("routed_by"),
            "not_detectable": list(NOT_DETECTABLE)}


def lost(result: Mapping[str, Any]) -> list[str]:
    """The people of one exam result for whom the plain arm reached the right route and the lead agent did not."""
    scores = result.get("scores") or {}
    if not isinstance(scores, dict) or "plain" not in scores or "case" not in scores:
        return []
    plain = {r["id"]: r for r in scores["plain"]}
    return [r["id"] for r in scores["case"] if r["id"] in plain and plain[r["id"]].get("route_right") and not r.get("route_right")]


def won(result: Mapping[str, Any]) -> list[str]:
    """The reverse: the lead agent right, the plain arm wrong (reported beside ``lost``, so one side is never read alone)."""
    scores = result.get("scores") or {}
    if not isinstance(scores, dict) or "plain" not in scores or "case" not in scores:
        return []
    plain = {r["id"]: r for r in scores["plain"]}
    return [r["id"] for r in scores["case"] if r["id"] in plain and r.get("route_right") and not plain[r["id"]].get("route_right")]


def recorded_dev(results: Path = EXAM_RESULTS) -> dict[str, Any]:
    """Every kept dev exam run that compared plain and case (never a sealed one): the pairs lost and won, and the first
    divergence of each lost pair. Costs nothing: it reads what the runs kept."""
    runs, rows = [], []
    for path in sorted(Path(results).glob("CEX-*_dev*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "sealed" in str(doc.get("split") or ""):
            continue
        scores = doc.get("scores") or {}
        if not (isinstance(scores, dict) and scores.get("case") and scores.get("plain")):
            continue  # only a run that played both arms is a comparison
        ids, wins = lost(doc), won(doc)
        runs.append({"run": doc.get("id"), "split": doc.get("split"), "at": doc.get("at"), "model": (doc.get("models") or {}).get("arms"),
                     "people": len(scores["case"]), "plain_right_case_wrong": ids, "case_right_plain_wrong": wins})
        kept = ROOT / str(doc.get("kept") or "")
        for pid in ids:
            folder = kept / "case" / pid
            if not (folder / "conversation.json").exists():
                rows.append({"run": doc.get("id"), "id": pid, "first": None, "all": [], "missing": "no kept conversation"})
                continue
            # an older run kept no case.changed event: there a route set by code cannot be seen, only refused calls
            rows.append({"run": doc.get("id"), **read(folder), "case_changes_kept": any(e.get("kind") == "case.changed" for e in events(folder))})
    counts: dict[str, int] = {}
    for row in rows:
        reason = (row.get("first") or {}).get("reason") or "not_kept"
        counts[reason] = counts.get(reason, 0) + 1
    return {"runs": runs, "lost_pairs": rows, "first_reason_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "lost": sum(len(r["plain_right_case_wrong"]) for r in runs), "won": sum(len(r["case_right_plain_wrong"]) for r in runs),
            "lost_people": sorted({i for r in runs for i in r["plain_right_case_wrong"]}), "won_people": sorted({i for r in runs for i in r["case_right_plain_wrong"]}),
            "what_is_visible": "refused or changed calls of the lead agent's tools, and (where case.changed was kept) the route's changes; "
                               "a lost pair with no such event is not proof that no code shaped it",
            "not_detectable": list(NOT_DETECTABLE)}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="The first divergence of the lead agent in kept conversations.")
    parser.add_argument("run", nargs="?", help="a kept run folder (holding case/<person>/)")
    parser.add_argument("--recorded-dev", action="store_true", help="every kept dev exam run")
    args = parser.parse_args(argv)
    if args.recorded_dev:
        print(json.dumps(recorded_dev(), indent=1, ensure_ascii=False))
        return 0
    if not args.run:
        parser.error("name a kept run folder, or pass --recorded-dev")
    folders = sorted(p for p in (Path(args.run) / "case").glob("*") if p.is_dir())
    if not folders:
        print(f"no case conversations under {args.run}", file=sys.stderr)
        return 2
    print(json.dumps([read(f.resolve()) for f in folders], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
