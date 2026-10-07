"""The user study's harness (package 14.6): time each task and record what happened, without keeping a cell value.

Everything recorded is an enumerated value or a number: a pseudonymous participant (P1, P2, …), a task (T1, T2, …), the
condition (``with`` or ``without`` DCLab), a kind of event and its small payload. There is no free-text field, so a
cell value, a name or a column cannot end up in the file by accident. The events are an append-only JSON-lines file
(``evaluation/study/events.jsonl``). ``analyze`` reads it and reports, per condition: tasks completed (with a Wilson
interval), time to complete, time spent correcting the assistant, how many cited records were correct, and severe false
alarms. With three to five people the numbers are descriptive: no significance test is run or implied.

    python -m dclab_rnd.agent_eval.study start --participant P1 --task T1 --condition with
    python -m dclab_rnd.agent_eval.study correction --participant P1 --task T1 --seconds 40
    python -m dclab_rnd.agent_eval.study citation --participant P1 --task T1 --correct yes
    python -m dclab_rnd.agent_eval.study severe-false-alarm --participant P1 --task T1
    python -m dclab_rnd.agent_eval.study finish --participant P1 --task T1 --completed yes
    python -m dclab_rnd.agent_eval.study analyze --output evidence/campaigns/benchmark_v2/study/STU-001.json
    python -m dclab_rnd.agent_eval.study delete --participant P1      # at the participant's request: their events are removed
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run import wilson

EVENTS = Path(__file__).resolve().parents[2] / "evaluation" / "study" / "events.jsonl"
CONDITIONS = ("with", "without")
KINDS = ("start", "finish", "correction", "citation", "severe_false_alarm")
PARTICIPANT = re.compile(r"^P\d{1,3}$")
TASK = re.compile(r"^T\d{1,2}$")
MAX_SECONDS = 4 * 3600


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record(kind: str, participant: str, task: str, path: Path = EVENTS, condition: str | None = None, **payload: Any) -> dict[str, Any]:
    """Append one event. A value outside the allowed forms is refused, so the file holds nothing but the study's own fields."""
    if kind not in KINDS:
        raise ValueError(f"unknown event {kind!r}")
    if not PARTICIPANT.match(participant or ""):
        raise ValueError("a participant is a pseudonym such as P1 (no name)")
    if not TASK.match(task or ""):
        raise ValueError("a task is T1, T2, …")
    allowed = {"finish": {"completed"}, "correction": {"seconds"}, "citation": {"correct"}}.get(kind, set())
    if set(payload) - allowed:
        raise ValueError(f"{kind} records only {', '.join(sorted(allowed)) or 'its kind'}: {', '.join(sorted(set(payload) - allowed))} has no place in the file")
    event: dict[str, Any] = {"kind": kind, "participant": participant, "task": task, "at": _now()}
    existing = read(path)
    started = [e for e in existing if e["kind"] == "start" and (e["participant"], e["task"]) == (participant, task)]
    if kind == "start":
        if condition not in CONDITIONS:
            raise ValueError("the condition is 'with' or 'without'")
        if started:
            raise ValueError(f"{participant} {task} was already started")
        event["condition"] = condition
    else:
        if not started:
            raise ValueError(f"{participant} {task} has not started")
        if any(e["kind"] == "finish" and (e["participant"], e["task"]) == (participant, task) for e in existing):
            raise ValueError(f"{participant} {task} has finished: nothing more is recorded for it")
        event["condition"] = started[0]["condition"]
        if kind in ("correction", "citation", "severe_false_alarm") and event["condition"] != "with":
            raise ValueError(f"{kind} is about the assistant's output: it is recorded only in the 'with' condition")
    if kind == "finish":
        if payload.get("completed") not in (True, False):
            raise ValueError("finish needs completed yes or no")
        event["completed"] = payload["completed"]
        event["seconds"] = round((datetime.fromisoformat(event["at"]) - datetime.fromisoformat(started[0]["at"])).total_seconds())
    elif kind == "correction":
        seconds = payload.get("seconds")
        if not isinstance(seconds, int) or isinstance(seconds, bool) or not 0 <= seconds <= MAX_SECONDS:
            raise ValueError("a correction records the whole seconds spent (0 to 14400), nothing else")
        event["seconds"] = seconds
    elif kind == "citation":
        if payload.get("correct") not in (True, False):
            raise ValueError("a citation is correct yes or no")
        event["correct"] = payload["correct"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    return event


def delete_participant(participant: str, path: Path = EVENTS) -> int:
    """Remove every event of one participant (their request to have their sessions deleted); returns how many were removed.
    The file is rewritten whole, then moved into place."""
    if not PARTICIPANT.match(participant or ""):
        raise ValueError("a participant is a pseudonym such as P1")
    events = read(path)
    kept = [e for e in events if e["participant"] != participant]
    part = Path(path).with_name(Path(path).name + ".part")
    part.write_text("".join(json.dumps(e) + "\n" for e in kept), encoding="utf-8")
    part.replace(path)
    return len(events) - len(kept)


def read(path: Path = EVENTS) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze(path: Path = EVENTS) -> dict[str, Any]:
    events = read(path)
    out: dict[str, Any] = {}
    for condition in CONDITIONS:
        tasks = {(e["participant"], e["task"]) for e in events if e["kind"] == "start" and e["condition"] == condition}
        finished = {(e["participant"], e["task"]): e for e in events if e["kind"] == "finish" and e["condition"] == condition}
        done = [k for k, e in finished.items() if e["completed"]]
        times = [finished[k]["seconds"] for k in done]
        corrections = [e["seconds"] for e in events if e["kind"] == "correction" and e["condition"] == condition]
        cites = [e["correct"] for e in events if e["kind"] == "citation" and e["condition"] == condition]
        out[condition] = {
            "tasks_started": len(tasks), "tasks_finished": len(finished), "tasks_completed": [len(done), len(tasks)],
            "completed_ci95": wilson(len(done), len(tasks)),
            "seconds_to_complete_median": statistics.median(times) if times else None, "seconds_to_complete": sorted(times),
            "correction_seconds_total": sum(corrections), "corrections": len(corrections),
            "citations_correct": [sum(cites), len(cites)], "citations_correct_ci95": wilson(sum(cites), len(cites)),
            "severe_false_alarms": sum(e["kind"] == "severe_false_alarm" and e["condition"] == condition for e in events),
            "participants": sorted({k[0] for k in tasks}),
        }
    return {"kind": "user_study", "completed_at": _now(), "events": len(events), "by_condition": out,
            "limitations": ["Three to five people: the numbers are descriptive. No significance test is run, and none is implied.",
                            "Tasks differ in difficulty; the protocol counterbalances the order, and a difference smaller than that spread is not a finding.",
                            "Nothing here says the assistant is ready for production."]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.study", description="The user study's harness (14.6).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "finish", "correction", "citation", "severe-false-alarm"):
        p = sub.add_parser(name)
        p.add_argument("--participant", required=True)
        p.add_argument("--task", required=True)
        p.add_argument("--events", type=Path, default=EVENTS)
        if name == "start":
            p.add_argument("--condition", choices=CONDITIONS, required=True)
        if name == "finish":
            p.add_argument("--completed", choices=("yes", "no"), required=True)
        if name == "correction":
            p.add_argument("--seconds", type=int, required=True)
        if name == "citation":
            p.add_argument("--correct", choices=("yes", "no"), required=True)
    d = sub.add_parser("delete", help="remove every event of one participant, at their request")
    d.add_argument("--participant", required=True)
    d.add_argument("--events", type=Path, default=EVENTS)
    a = sub.add_parser("analyze")
    a.add_argument("--events", type=Path, default=EVENTS)
    a.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "delete":
            print(f"removed {delete_participant(args.participant, args.events)} events of {args.participant}")
            return 0
        if args.command == "analyze":
            report = analyze(args.events)
            print(json.dumps(report["by_condition"], indent=2))
            if args.output:
                if args.output.exists():
                    parser.error(f"{args.output} exists; results are never overwritten")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as handle:
                    json.dump(report, handle, indent=2)
                    handle.write("\n")
            return 0
        kind = args.command.replace("-", "_")
        payload: dict[str, Any] = {}
        if kind == "finish":
            payload["completed"] = args.completed == "yes"
        if kind == "correction":
            payload["seconds"] = args.seconds
        if kind == "citation":
            payload["correct"] = args.correct == "yes"
        event = record(kind, args.participant, args.task, args.events, getattr(args, "condition", None), **payload)
        print(json.dumps(event))
        return 0
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
