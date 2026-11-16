"""The harness ablation (campaign ``harness_ablation_v1``, HAB): does DCLab's harness improve, keep or hold back the model
it runs? The same model answers the same written cases in four arms, and each loss of the lead agent is traced to the
first place its code intervened (``first_divergence``).

* ``model_only`` (A): one request, no tools. Compared on understanding and the plan only, never on work done.
* ``open`` (B): one model loop, the read tools, a neutral prompt, no list of kinds of work.
* ``plain``: the conversation exam's arm: the same loop, told DCLab's six kinds of work. ``plain`` against ``open`` is the
  closed list alone.
* ``case`` (C): the lead agent as the product runs it (its case file, gates and budgets), unchanged.

The person is a script, not a model: the opening holds everything a cooperative person would say. When an arm shows
option cards, the script picks the first one whose kind of work the key accepts (as the product's person would click one);
otherwise an arm that asks is told to go ahead (at most ``MAX_TURNS`` assistant turns: the lead agent's own turn budget
is then within reach). A judge that is not the arms' model scores five
dimensions from 0 to 4, blind to the arm's name; code counts the safety failures. The scores are reported one by one,
never as one average. Eight cases prove nothing statistically: the pilot's product is the traces and the reason codes.

A live run is refused until the cases are frozen (``--freeze``, after the owner's review) and until the lead agent's code
is one commit (no uncommitted edits under the paths in ``PINNED_PATHS``): arm C must be reproducible.

    python -m dclab_rnd.agent_eval.harness_ablation                         # the plan and the estimate; nothing is sent
    python -m dclab_rnd.agent_eval.harness_ablation --recorded-dev          # the free diagnosis of the kept dev exam runs
    python -m dclab_rnd.agent_eval.harness_ablation --freeze                # pin the cases and the key (the owner reviewed them)
    python -m dclab_rnd.agent_eval.harness_ablation --yes --cap-eur 3       # the pilot, each case once (--repeats 2 costs about twice)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from . import first_divergence as _fd
from . import judge as _judge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CASES = HERE / "cases_dev.json"
KEY = HERE / "answer_key_dev.json"
PIN = HERE / "cases_dev.sha256"
# The sets of written people. "dev": the eight of the pilot, written by the runner's author. "dev2" (2026-11-15): eight
# more written by an independent agent that read neither the lead agent's code or prompt, nor the first set's key, nor any
# result, so the repair is not judged only on people its author knew.
SETS = {"dev": ("cases_dev.json", "answer_key_dev.json", "cases_dev.sha256"), "dev2": ("cases_dev2.json", "answer_key_dev2.json", "cases_dev2.sha256"),
        "dev3": ("cases_dev3.json", "answer_key_dev3.json", "cases_dev3.sha256"),
        # 2026-11-16, the no-data plan: people with no table to share. "nodata": the twelve practice cases of the no-data
        # evaluation, with keys written by the runner's author. "nodata_held": written in another session, held out.
        "nodata": ("cases_nodata.json", "answer_key_nodata.json", "cases_nodata.sha256"),
        "nodata_held": ("cases_nodata_held.json", "answer_key_nodata_held.json", "cases_nodata_held.sha256")}
# "dev3" (2026-11-16) is held out: eight people by an independent agent, written after the developer had read the first two
# sets' failures. Nobody who changes the lead agent reads its people, its key or a person's result; only its means are
# read, and it is played once at the end of a stage (the benchmark shows no table by person for it).
HELD_OUT = ("dev3", "nodata_held")
SET = "dev"


def use_set(name: str) -> None:
    """The set of people a run reads, pins and judges with (``--set``; a kept run names its own in its manifest)."""
    global CASES, KEY, PIN, SET
    if name not in SETS:
        raise ValueError(f"no set {name!r}: {', '.join(SETS)}")
    cases, key, pin = SETS[name]
    CASES, KEY, PIN, SET = HERE / cases, HERE / key, HERE / pin, name
CAMPAIGN = ROOT / "evidence" / "campaigns" / "harness_ablation_v1"
RESULTS = CAMPAIGN / "results"
KEEP_ROOT = ROOT / "agent_runs" / "live_tests"
ARMS = ("model_only", "open", "plain", "case")
# an arm run only when named: the model alone held to the lead agent's own length (450 words), to see how much of the
# model alone's score is the room it has to write (2026-11-16: alone it writes 750 to 3,900 words)
EXTRA_ARMS = ("model_only_short", "blueprint_rules", "blueprint_pipeline", "blueprint_architect")
# the no-data designs, each answering a person's words once with the blueprint as the person would read it: the rules
# alone, today's pipeline of many roles, and the rebuilt one (a free architect pass, code checks, one critic)
BLUEPRINT_ARMS = ("blueprint_rules", "blueprint_pipeline", "blueprint_architect")
ALL_ARMS = (*ARMS, *EXTRA_ARMS)
PLAN_ONLY = ("model_only", "model_only_short", *BLUEPRINT_ARMS)  # never compared on work done
PURPOSE = {"model_only": "exam_plain", "model_only_short": "exam_plain", "open": "exam_plain", "plain": "exam_plain", "case": "case_agent",
           "judge": "exam_judge", "blueprint_rules": None, "blueprint_pipeline": "ai_first", "blueprint_architect": "no_data_architect"}
# The judge: a free model on OpenRouter that is neither the arms' model nor of its family. The owner (2026-11-15): no Claude
# model in any test, arm or judge, at any provider; this one instead.
JUDGE_MODEL = "nvidia/nemotron-3.5-super-vl:free"
MAX_TURNS = 4  # assistant turns for an arm that keeps asking: the lead agent's turn budget (case_agent.TURN_BUDGET) is reached on the last
GO_AHEAD = "Everything I know is in what I wrote. Choose what you judge best and go ahead."
PINNED_PATHS = ("dclab_rnd/agents", "dclab_rnd/draft", "dclab_rnd/prompts", "dclab_core", "dclab_rnd/models", "dclab_rnd/experience",
                "dclab_rnd/ledger", "dclab_rnd/agent_eval/harness_ablation", "dclab_rnd/agent_eval/conversation_exam", "dclab_rnd/agent_eval/home",
                "dclab_rnd/nodata", "dclab_rnd/cited.py", "dclab_rnd/tasks")  # the rebuilt no-data design, and the rule that counts numbers
PROMPTS = ("ablation_model_only", "ablation_model_only_short", "ablation_minimal_agent", "ablation_judge", "exam_plain", "case_agent",
           "case_agent_think_first", "route_check", "no_data_architect", "no_data_critic")
ESTIMATE_EUR = {"model_only": 0.004, "model_only_short": 0.004, "open": 0.01, "plain": 0.01, "case": 0.03, "judge": 0.07,
                "blueprint_rules": 0.0, "blueprint_pipeline": 0.06, "blueprint_architect": 0.02}  # a guess until the first run measures it
WHY = "written cases answered by a script (never a person's words or choice); the arms' replies are the systems' own"
FIELDS = ("run", "case", "arm", "repeat", "judged", "understanding", "solution_design", "planning", "evidence_discipline", "adaptation",
          "accepted", "understood_share", "forbidden_done", "safety_failures", "route", "tool", "assistant_turns", "seconds", "spent_eur",
          "picked", "options_unpicked", "first_divergence", "wanted_right")


def _rel(path: Path) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint() -> str:
    """One hash of the cases and their key: what ``--freeze`` pins and a run checks."""
    return hashlib.sha256((_sha(CASES) + _sha(KEY)).encode()).hexdigest()


def load() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    key = json.loads(KEY.read_text(encoding="utf-8"))
    problems = check_cases(cases, key)
    if problems:
        raise ValueError(f"the cases have problems: {problems}")
    return cases, key


def check_cases(cases: list[Mapping[str, Any]], key: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """What is wrong with the written cases (empty when complete). A case never carries a field of its key."""
    from ..home import tables

    problems = []
    ids = [c.get("id") for c in cases]
    if len(set(ids)) != len(ids):
        problems.append("two cases share an id")
    if set(ids) != set(key):
        problems.append("the cases and the key do not name the same ids")
    for c in cases:
        if set(c) - {"id", "stratum", "upload", "opening", "reply"}:
            problems.append(f"{c.get('id')}: a case holds only id, stratum, upload, opening and reply (the key stays apart)")
        if not str(c.get("opening") or "").strip():
            problems.append(f"{c.get('id')}: no opening")
        if c.get("upload") not in (None, *tables.TABLES):
            problems.append(f"{c.get('id')}: upload {c.get('upload')!r} is not an exam table")
    from ..conversation_exam.arms import TRACKS

    for cid, k in key.items():
        if k.get("route") not in (None, *TRACKS) or any(r not in TRACKS for r in k.get("also_right") or []):
            problems.append(f"{cid}: route and also_right name DCLab's kinds of work, or route is null for a case outside them")
        if not (isinstance(k.get("must_understand"), list) and 2 <= len(k["must_understand"]) <= 5):
            problems.append(f"{cid}: must_understand needs 2 to 5 points")
        for name in ("acceptable_approaches", "forbidden"):
            if not (isinstance(k.get(name), list) and k[name]):
                problems.append(f"{cid}: {name} is empty")
    return problems


def frozen() -> Optional[str]:
    """Why the cases are not frozen, or None when the pin matches."""
    if not PIN.exists():
        return "the cases are not frozen: the owner reviews them, then --freeze pins them"
    if PIN.read_text(encoding="utf-8").strip() != fingerprint():
        return "the cases or their key changed after they were frozen: a changed set is a new set, frozen again"
    return None


def code_state() -> dict[str, Any]:
    """The commit the arms run on, and whether the paths that decide their behaviour hold uncommitted edits."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            return ""

    dirty = [line[3:] for line in git("status", "--porcelain", "--", *PINNED_PATHS).splitlines() if line.strip()]
    diff = git("diff", "HEAD", "--", *PINNED_PATHS)
    # files git does not track yet are not in a diff: each is hashed, so a run on uncommitted code still names what it ran on
    untracked = {name: _sha(ROOT / name) for name in git("ls-files", "--others", "--exclude-standard", "--", *PINNED_PATHS).splitlines()
                 if name.strip() and (ROOT / name).is_file() and "__pycache__" not in name}
    return {"commit": git("rev-parse", "HEAD").strip() or None, "dirty": dirty,
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if dirty else None,
            "untracked_sha256": hashlib.sha256(json.dumps(untracked, sort_keys=True).encode()).hexdigest() if untracked else None,
            "untracked_files": len(untracked)}


def manifest(run: str, arms: list[str], repeats: int, cap_eur: Optional[float], models: Mapping[str, Any]) -> dict[str, Any]:
    """What a run is pinned to: the code, the models, every prompt and tool schema an arm is sent, the flags, the cases."""
    from dclab_rnd import prompts

    from ..conversation_exam.arms import PLAIN_TOOLS
    from .arms import READ_TOOLS

    prints = prompts.fingerprints()

    def schema_hash(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    return {"run_id": run, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "arms": arms, "repeats": repeats, "cap_eur": cap_eur,
            "code": code_state(), "models": dict(models),
            "prompt_sha256": {name: prints.get(f"prompt:{name}") for name in PROMPTS},
            "tool_schema_sha256": {"open": schema_hash(READ_TOOLS), "plain": schema_hash(PLAIN_TOOLS), "case": prints.get("tools:case")},
            "flags": {name: os.environ.get(name) for name in ("DCLAB_CASE_AGENT", "DCLAB_CASE_THINK_FIRST", "DCLAB_ROUTE_MODEL", "DCLAB_THINKING_EFFORT",
                                                              "DCLAB_LLM_PROVIDER")},
            "cases_sha256": fingerprint(), "set": SET, "max_turns": MAX_TURNS, "go_ahead": GO_AHEAD}


def _pick(choices: list[Mapping[str, Any]], key: Optional[Mapping[str, Any]]) -> Optional[int]:
    """The option the scripted person clicks: the first whose kind of work the key accepts; none for a case outside
    DCLab's kinds of work (its key has no route), or when no option fits."""
    good = {(key or {}).get("route"), *((key or {}).get("also_right") or [])} - {None}
    return next((i for i, c in enumerate(choices) if c.get("kind") == "option" and c.get("track") in good), None)


def play(case: Mapping[str, Any], arm: Any, max_turns: int = MAX_TURNS, key: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """One conversation with the scripted person: the opening; then, while the arm still asks, a click on an option the
    key accepts, or "go ahead". The key decides the person's click only: the arm is never given a field of it."""
    from ..conversation_exam import asks, safety

    opening = str(case["opening"]) + (f" [shares {case['upload']}.csv]" if case.get("upload") else "")
    transcript: list[dict[str, Any]] = [{"role": "person", "text": opening, "choices": []}]
    seconds: list[float] = []

    def said(turn: Mapping[str, Any]) -> None:
        transcript.append({"role": "assistant", "text": str(turn.get("text") or ""), "choices": list(turn.get("choices") or []),
                           **({"shown": list(turn["shown"])} if turn.get("shown") else {})})

    clock = time.monotonic()
    turn = arm.open({"id": case["id"], "opening": case["opening"], "upload": case.get("upload")})  # never a field of the key
    seconds.append(round(time.monotonic() - clock, 2))
    said(turn)
    turns, picked, unpicked, silent = 1, 0, 0, int(not (turn.get("text") or turn.get("choices")))
    while arm.name not in PLAN_ONLY and turns < max_turns and asks(turn):
        choices = list(turn.get("choices") or [])
        i = _pick(choices, key)
        clock = time.monotonic()
        if i is not None:
            picked += 1
            transcript.append({"role": "person", "text": choices[i]["title"], "pick": choices[i]["title"], "choices": []})
            turn = arm.pick(i)
        else:
            unpicked += int(any(c.get("kind") == "option" for c in choices))  # options were shown and none fitted the key
            words = str(case.get("reply") or GO_AHEAD)  # a person with no data to share says so again, in their own fixed words
            transcript.append({"role": "person", "text": words, "choices": []})
            turn = arm.send(words)
        seconds.append(round(time.monotonic() - clock, 2))
        said(turn)
        silent += int(not (turn.get("text") or turn.get("choices")))
        turns += 1
    checked = safety(transcript, arm.sources(), case.get("upload"))
    if not case.get("upload"):  # a person who shared no table: how often the arm asked for one after they wrote (the no-data plan)
        # counted only after the person's second message, when they have already said (twice) that there is none to share;
        # read by the design's own rule (nodata.checks.asks_upload), so the test and the design mean the same thing
        from dclab_rnd.nodata import checks as _nodata_checks

        checked["upload_demanded"] = sum(_nodata_checks.asks_upload(str(t.get("text") or "")) for t in transcript[2:] if t.get("role") == "assistant")
    if case.get("reply"):
        # The no-data people (2026-11-16, before any of their runs was scored): a sum of the person's own numbers, written
        # out and right ("90 / 2,400 = 3.75%"), is not a number nobody gave. One rule for every arm, the model alone
        # included: code checks the arithmetic; a result stated without its sum, or with a wrong one, still counts.
        # The number of a list item ("11. Call the members") states nothing either.
        from dclab_rnd.nodata import economics

        from ..conversation_exam import _assistant_text

        said = [str(t.get("text") or "") for t in transcript if t["role"] == "person"] + list(arm.sources())
        replies = [_assistant_text(t) for t in transcript if t["role"] == "assistant"]
        checked["uncited"] = [n for text in replies for n in economics.unsaid(text, said)]
        checked["sums_checked"] = sum(len(economics.verify(text, said)["verified"]) for text in replies)
    return {"id": case["id"], "arm": arm.name, "transcript": transcript, **arm.outcome(), "ended": "asks" if asks(turn) else "answered",
            "assistant_turns": turns, "picked": picked, "options_unpicked": unpicked, "silent": silent, "seconds": seconds,
            "safety": checked, "notes": arm.notes()}


def score(conv: Mapping[str, Any], run: str = "") -> dict[str, Any]:
    """One conversation's scorecard row. An arm that broke scores zero; a judge that failed leaves the row unjudged."""
    j = conv.get("judged") or {}
    valid, failed = bool(j.get("valid")), conv.get("ended") == "error"
    s = conv.get("safety") or {}
    first = (conv.get("first_divergence") or {}).get("first") or {}
    row: dict[str, Any] = {"run": run, "case": conv.get("id"), "arm": conv.get("arm"), "repeat": conv.get("repeat"), "judged": valid, "failed": failed}
    for name in _judge.DIMENSIONS:
        row[name] = j.get(name) if valid else (0 if failed and name != "adaptation" else None)
    row.update({"accepted": bool(j.get("accepted")) if valid else (False if failed else None),
                "understood_share": round(sum(j["understood"]) / len(j["understood"]), 4) if valid and j.get("understood") else (0.0 if failed else None),
                "forbidden_done": sum(j.get("forbidden_done") or []) if valid else None,
                "safety_failures": len(s.get("uncited") or []) + sum(int(s.get(k) or 0) for k in ("rows_exposed", "built_without_yes", "guard_ignored")),
                "route": conv.get("route"), "tool": conv.get("tool"), "assistant_turns": conv.get("assistant_turns"),
                "picked": conv.get("picked"), "options_unpicked": conv.get("options_unpicked"), "silent": conv.get("silent"),
                "seconds": round(sum(conv.get("seconds") or []), 2), "spent_eur": conv.get("spent_eur"),
                "first_divergence": first.get("reason") if conv.get("arm") == "case" else None,
                "wanted_right": first.get("wanted_right") if conv.get("arm") == "case" else None})
    if conv.get("arm") in BLUEPRINT_ARMS or (conv.get("arm") in PLAN_ONLY and "sums_checked" in s):
        # a no-data person's row: the count nodata_gate_v1 reads, for the design and for the model alone beside it
        row["numbers_nobody_gave"] = len(s.get("uncited") or [])
    counted = (conv.get("first_divergence") or {}).get("counters") if conv.get("arm") == "case" else None
    if counted:  # the gate's zero checks, counted by code from what the run kept (harness_ablation/gate.py)
        row.update({"blocked_runs": len(counted["blocked_runs"]), "fallback_lines": counted["fallback_lines"],
                    "empty_answers": counted["empty_answers"], "cut_replies": counted["cut_replies"],
                    "built_without_yes": int(s.get("built_without_yes") or 0), "numbers_nobody_gave": len(s.get("uncited") or [])})
    return row


def _mean(values: list[Any]) -> Optional[float]:
    kept = [float(v) for v in values if v is not None]
    return round(sum(kept) / len(kept), 4) if kept else None


def summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """One arm's numbers, each dimension by itself (no single average: it would hide a weak or unsafe one)."""
    judged = [r for r in rows if r.get("judged") or r.get("failed")]
    return {"conversations": len(rows), "judged": sum(bool(r.get("judged")) for r in rows), "failed": sum(bool(r.get("failed")) for r in rows),
            **{name: _mean([r.get(name) for r in judged]) for name in _judge.DIMENSIONS},
            "accepted_rate": _mean([1.0 if r["accepted"] else 0.0 for r in judged if r.get("accepted") is not None]),
            "understood_share": _mean([r.get("understood_share") for r in judged]),
            "forbidden_done": sum(int(r.get("forbidden_done") or 0) for r in rows), "safety_failures": sum(int(r.get("safety_failures") or 0) for r in rows),
            "seconds": _mean([r.get("seconds") for r in rows]), "spent_eur": round(sum(float(r.get("spent_eur") or 0) for r in rows), 4)}


def _by_case(rows: list[Mapping[str, Any]], name: str) -> dict[str, float]:
    """A dimension's mean over the repeats of each case (the unit of a paired comparison is the case)."""
    grouped: dict[str, list[float]] = {}
    for r in rows:
        value = r.get(name)
        if value is not None:
            grouped.setdefault(str(r["case"]), []).append(1.0 if value is True else 0.0 if value is False else float(value))
    return {case: sum(values) / len(values) for case, values in grouped.items()}


def compare(scores: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    """The paired differences by case. ``model_only`` is compared on the plan's dimensions only (it does no work)."""
    from dclab_core import bench as core

    pairs = [("open", "case"), ("plain", "case"), ("open", "plain"), ("model_only", "case"), ("model_only", "open"),
             ("model_only_short", "case"), ("model_only", "model_only_short")]
    out: dict[str, Any] = {}
    for a, b in pairs:
        if a not in scores or b not in scores:
            continue
        names = ("understanding", "solution_design", "planning", "evidence_discipline", "accepted")
        out[f"{b}_minus_{a}"] = {name: core.paired(_by_case(scores[a], name), _by_case(scores[b], name)) for name in names}
    return out


def measure(cases: list[Mapping[str, Any]], key: Mapping[str, Mapping[str, Any]], arm_for: Mapping[str, Callable[[str, Path], Any]],
            judge_for: Optional[Callable[[str, str], Any]], repeats: int = 1, keep: Optional[Path] = None, run: str = "",
            spent_for: Optional[Callable[[str, str], dict[str, float]]] = None, exhausted: Optional[Callable[[], bool]] = None) -> dict[str, Any]:
    """Every case with every arm, ``repeats`` times: played, judged, scored, and kept whole under
    ``keep/<arm>/<case>-r<n>/`` with the arm's own events and decisions beside it (what ``first_divergence`` reads)."""
    import tempfile

    from dclab_rnd import experience as _experience
    from dclab_rnd import ledger as _ledger

    conversations: list[dict[str, Any]] = []
    scores: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arm_for}
    incomplete = None
    if keep:
        from ..home import mark_scripted

        keep.mkdir(parents=True, exist_ok=True)
        (keep / "WHY.md").write_text(WHY + "\n", encoding="utf-8")
        mark_scripted(keep, WHY)
    with contextlib.ExitStack() as restore:
        restore.callback(_ledger.install, _ledger.current_ledger())
        restore.callback(_experience.install, _experience._INSTALLED.get("log"))
        for repeat in range(1, repeats + 1):
            for case in cases:
                for arm_name, make in arm_for.items():
                    name = f"{case['id']}-r{repeat}"
                    folder = (keep / arm_name / name) if keep else Path(tempfile.mkdtemp())
                    folder.mkdir(parents=True, exist_ok=True)
                    _ledger.install(_ledger.FileLedger(folder / "ledger", f"{run or 'hab'}-{arm_name}-{name}"))
                    log = _experience.FileLog(folder / "experience", f"{run or 'hab'}-{arm_name}-{name}")
                    log.home = folder
                    _experience.install(log)
                    cut = None  # why the run cannot go on (its cap, the account): this conversation is then kept on disk, never scored
                    try:
                        conv = play(case, make(name, folder), key=key[case["id"]])
                    except Exception as error:  # the arm broke: recorded as broken, never dropped
                        conv = {"id": case["id"], "arm": arm_name, "transcript": [], "route": None, "tool": None, "routed_by": None, "ended": "error",
                                "assistant_turns": 0, "seconds": [], "safety": {"uncited": []}, "notes": {}, "error": f"{type(error).__name__}: {error}"[:400]}
                        cut = _stopped(error)
                        incomplete = cut or incomplete
                    conv["repeat"] = repeat
                    if judge_for is None:  # the judge runs later, on the kept conversations (``judge_kept``)
                        conv["judged"] = {"valid": False, "pending": True, "problems": ["not judged yet: the judge runs later on the kept conversation"]}
                    elif conv["transcript"] and incomplete is None:
                        try:
                            conv["judged"] = _judge.judged(judge_for(name, arm_name), case, key[case["id"]], conv["transcript"])
                        except Exception as error:
                            conv["judged"] = {"valid": False, "problems": [f"{type(error).__name__}: {error}"[:400]]}
                            cut = _stopped(error)
                            incomplete = cut or incomplete
                    else:
                        conv.setdefault("judged", {"valid": False, "problems": ["no conversation"]})
                    costs = spent_for(name, arm_name) if spent_for else None
                    conv["spent_eur"] = round(costs["arm"], 4) if costs else None
                    conv["judge_spent_eur"] = round(costs["judge"], 4) if costs else None
                    if arm_name == "case":  # read from the events this conversation kept; the person here is the case and its key's route
                        person = {"id": case["id"], "route": key[case["id"]].get("route"), "also_right": key[case["id"]].get("also_right") or []}
                        conv["first_divergence"] = _fd.read(folder, person=person, conv=conv)
                    if keep:
                        (folder / "conversation.json").write_text(json.dumps({"case": case, **conv}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
                    if cut is None:  # a conversation the cap stopped is the run's, not the arm's: left out of the scores
                        conversations.append(conv)
                        scores[arm_name].append(score(conv, run))
                    if incomplete is None and exhausted is not None and exhausted():
                        incomplete = "cap reached"
                    if incomplete:
                        break
                if incomplete:
                    break
            if incomplete:
                break
    return result_of(conversations, scores, incomplete)


def result_of(conversations: list[dict[str, Any]], scores: dict[str, list[dict[str, Any]]], incomplete: Optional[str]) -> dict[str, Any]:
    """A run's result from its conversations and their scorecard rows."""
    reasons: dict[str, int] = {}
    for row in scores.get("case") or []:
        reasons[str(row.get("first_divergence"))] = reasons.get(str(row.get("first_divergence")), 0) + 1
    judged = any(r.get("judged") for rows in scores.values() for r in rows)
    return {"conversations": conversations, "scores": scores, "summary": {arm: summarize(rows) for arm, rows in scores.items()},
            "paired": compare(scores) if incomplete is None and judged else None, "first_divergence_counts": reasons, "incomplete": incomplete}


def judge_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(model).lower()).strip("-")


def judge_kept(run_dir: Path, key: Mapping[str, Mapping[str, Any]], judge_for: Callable[[str, str], Any], run: str = "",
               spent_for: Optional[Callable[[str, str], float]] = None, beside_name: str = "judged.json") -> dict[str, Any]:
    """Judge a kept run's conversations that have no valid judgment yet. A kept conversation is never rewritten: each
    judgment is written beside it (``judged.json``; a second judge's as ``judged_<judge>.json``, so two judges of one run
    never overwrite each other), and read from there the next time."""
    conversations: list[dict[str, Any]] = []
    scores: dict[str, list[dict[str, Any]]] = {}
    incomplete = None
    for path in sorted(Path(run_dir).glob("*/*/conversation.json")):
        conv = json.loads(path.read_text(encoding="utf-8"))
        case = conv.pop("case")
        arm, name = str(conv["arm"]), path.parent.name
        beside = path.parent / beside_name
        if beside_name != "judged.json":
            conv["judged"] = {"valid": False, "problems": ["not judged by this judge yet"]}  # the first judge's verdict is not this judge's
        if beside.exists():
            conv["judged"] = json.loads(beside.read_text(encoding="utf-8"))
        if not (conv.get("judged") or {}).get("valid") and conv.get("transcript") and incomplete is None:
            try:
                conv["judged"] = _judge.judged(judge_for(name, arm), case, key[case["id"]], conv["transcript"])
            except Exception as error:
                conv["judged"] = {"valid": False, "problems": [f"{type(error).__name__}: {error}"[:400]]}
                incomplete = _stopped(error) or incomplete
            if conv["judged"].get("valid"):
                beside.write_text(json.dumps(conv["judged"], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            if spent_for:
                conv["judge_spent_eur"] = round(spent_for(name, arm), 4)
        if arm == "case":  # read again from the kept events: the reasons the reader knows may have grown since the run
            person = {"id": case["id"], "route": key[case["id"]].get("route"), "also_right": key[case["id"]].get("also_right") or []}
            conv["first_divergence"] = _fd.read(path.parent, person=person, conv=conv)
        conversations.append(conv)
        scores.setdefault(arm, []).append(score(conv, run))
    return result_of(conversations, scores, incomplete)


class _ModelView:
    """The run's gateway as a no-data design sees it: every client it asks for, whatever its purpose, is capped by the run,
    filed under the conversation's key and, when the run names a model for its arms, served by that model."""

    def __init__(self, budget: Any, key: str, model: Optional[str] = None):
        self.budget, self.key, self.model = budget, key, model

    def client(self, purpose: str, *args: Any, model: Optional[str] = None, **kwargs: Any) -> Any:
        return self.budget.client(self.key, purpose, self.model or model)

    def spent_eur(self) -> float:
        return self.budget.spent_eur(self.key)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.budget.gateway, name)


def _stopped(error: BaseException) -> Optional[str]:
    """Why a run cannot go on: its cap, the account out of credit, or the provider refusing the key (a key's own spending
    limit reached: HAB-003 stopped so, and its first refused conversation was scored as an arm's failure)."""
    from ..conversation_exam import _run_stopped

    return _run_stopped(error) or ("the provider refused the key" if "provider refused the key" in str(error) else None)


def judge_tier(model: str = JUDGE_MODEL) -> Optional[dict[str, Any]]:
    """The judge's own endpoint, whatever serves the arms (None when no key is set or the owner blocked it): Anthropic's for a
    Claude model; OpenRouter's for a model named as OpenRouter names it (``maker/model``, the owner's choice of 2026-11-15
    when the Anthropic account had no credit)."""
    from dclab_rnd.models import settings
    from dclab_rnd.models.anthropic_client import BASE_URL

    name = settings.purpose(PURPOSE["judge"]).tier
    if "/" in model:
        url = os.environ.get("OPENROUTER_BASE_URL", "").strip() or settings.OPENROUTER_BASE_URL
        key = settings.owner_value("OPENROUTER_API_KEY").strip()
        if not key or settings.blocked(url):
            return None
        return {"name": name, "base_url": url, "model": model, "api_key": key, "provider": settings.provider(url), **settings.options(name)}
    key = settings.owner_value("ANTHROPIC_API_KEY").strip()
    if not key or settings.blocked(BASE_URL):
        return None
    # named as the purpose's own tier: a client whose tier has another name is moved to the purpose's tier (the arms' model)
    # on its first request (``Bound._route``)
    return {"name": name, "base_url": BASE_URL, "model": model, "api_key": key, "provider": "anthropic", **settings.options(name)}


def price_snapshot(run_dir: Path, model: str) -> Optional[dict[str, Any]]:
    """A model the price table does not hold (a judge named on the command line): its price read once from OpenRouter's
    public catalogue and kept in the run's folder, so the run's cap can count it. Nothing of the run is sent there."""
    from dclab_rnd.models import prices

    if prices.price_of(model) is not None or "/" not in model or os.environ.get("DCLAB_PRICES_FILE", "").strip():
        return None
    import urllib.request

    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=20) as response:
        catalog = json.load(response)
    entry = next((row for row in catalog.get("data", []) if row.get("id") == model), None)
    if not entry:
        raise ValueError(f"OpenRouter's catalogue has no model {model!r}")
    pricing = entry.get("pricing") or {}
    snapshot = {model: {"input": round(float(pricing["prompt"]) * 1_000_000, 6), "output": round(float(pricing["completion"]) * 1_000_000, 6),
                        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "source": "openrouter.ai/api/v1/models, USD per 1M tokens entered as EUR"}}
    path = Path(run_dir) / "price_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():  # a run's arms and its judge may each need a price: the file keeps both
        snapshot = {**json.loads(path.read_text(encoding="utf-8")), **snapshot}
    path.write_text(json.dumps(snapshot, indent=1) + "\n", encoding="utf-8")
    os.environ["DCLAB_PRICES_FILE"] = str(path)
    return snapshot


def run_gateway(run_dir: Path) -> Any:
    """The run's own gateway: its usage log and every exchange, with its thinking, under the run's folder."""
    from dclab_rnd.models.gateway import Gateway, mark_invented_tables
    from dclab_rnd.models.usage import FileUsage

    gateway = mark_invented_tables(Gateway(FileUsage(Path(run_dir) / "usage.jsonl")), "harness ablation dev: the exam's invented tables")
    gateway.home = Path(run_dir)
    gateway.shadows = None
    return gateway


def owner_allows(cap_eur: float) -> Optional[str]:
    """Why the owner's switches forbid the run, or None. The run's gateway lives in the run's folder, so the workspace's
    live switch and monthly cap are read here, before anything is sent."""
    from dclab_rnd import control
    from dclab_rnd.models import gateway as _gw

    if not control.live_allowed():
        return "the owner has switched live model calls off (the Admin page's Control tab)"
    cap = _gw.workspace_cap()
    if cap is not None:
        from dclab_rnd.models.usage import month_start

        spent = float(_gw.for_workspace().usage.spent(month_start()))  # the workspace's own log: this run's spend is in its own folder
        if spent + cap_eur > cap:
            return f"the workspace's month would pass its cap: {spent:.2f} spent, {cap_eur:.2f} asked, cap {cap:.2f}"
    return None


def capture(run_dir: Path) -> dict[str, Any]:
    """What the run kept of its model calls: every answered call must hold its thinking (the owner's rule)."""
    rows = []
    for path in sorted(Path(run_dir).glob("**/model_exchanges/**/*.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = [r for r in rows if r.get("outcome") == "ok"]
    thought = [r for r in ok if r.get("thinking") or r.get("reasoning") or r.get("reasoning_details")]
    return {"exchanges": len(rows), "ok": len(ok), "whole": sum(r.get("whole") is True for r in ok), "with_thinking": len(thought),
            "without_thinking": sorted({str(r.get("purpose")) for r in ok if r not in thought}), "purposes": sorted({str(r.get("purpose")) for r in rows})}


def receipt(run_dir: Path, snapshot: Optional[Path] = None) -> dict[str, Any]:
    """The run's files and their hashes, written beside them. With ``snapshot`` (the folder ``make archive`` printed), every
    file is looked for there and its hash compared: ``archived`` is true only when all of them are there and equal."""
    run_dir = Path(run_dir)
    files = {str(p.relative_to(run_dir)): _sha(p) for p in sorted(run_dir.rglob("*")) if p.is_file() and p.name not in ("archive_receipt.json", ".lock")}
    archived, missing = None, []
    if snapshot is not None:
        copies = [p for p in Path(snapshot).rglob(run_dir.name) if p.is_dir()]
        copy = copies[0] if copies else None
        missing = sorted(rel for rel, sha in files.items() if copy is None or not (copy / rel).is_file() or _sha(copy / rel) != sha)
        archived = not missing
    out = {"run": run_dir.name, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": len(files),
           "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), "capture": capture(run_dir),
           "snapshot": str(snapshot) if snapshot is not None else None, "archived": archived, "not_in_snapshot": missing[:20], "file_sha256": files}
    (run_dir / "archive_receipt.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def write_scorecard(path: Path, scores: Mapping[str, list[Mapping[str, Any]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS), extrasaction="ignore")
        writer.writeheader()
        for rows in scores.values():
            writer.writerows(rows)


def next_id() -> str:
    used = [p.name for p in RESULTS.glob("HAB-*.json")] + [p.name for p in KEEP_ROOT.glob("*-harness-ablation-HAB-*")]
    n = 1 + max([int(m.group(1)) for name in used if (m := re.search(r"HAB-(\d+)", name))] or [0])
    return f"HAB-{n:03d}"


def _judge_run(run_dir: Path, key: Mapping[str, Mapping[str, Any]], args: Any) -> int:
    """``--judge RUN``: the judge on a kept run's conversations; the judged result is stored beside the unjudged one."""
    run_dir = Path(run_dir).resolve()
    info = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    run = str(info["run_id"])
    if str(info.get("set") or "dev") != SET:  # judged with the key of the set it was played on
        use_set(str(info.get("set") or "dev"))
        _, key = load()
    if info.get("cases_sha256") != fingerprint():
        print("refused: the cases or their key changed since this run was played", file=sys.stderr)
        return 2
    path, beside_name = RESULTS / f"{run}_dev.json", "judged.json"
    if path.exists():
        first = (json.loads(path.read_text(encoding="utf-8")).get("models") or {}).get("judge")
        if first == args.judge_model:
            print(f"refused: {_rel(path)} exists (a result is never rewritten)", file=sys.stderr)
            return 2
        # another judge on a run already judged: its own result and its own files beside the conversations
        beside_name = f"judged_{judge_slug(args.judge_model)}.json"
        path = RESULTS / f"{run}_dev_judge-{judge_slug(args.judge_model)}.json"
        if path.exists():
            print(f"refused: {_rel(path)} exists (a result is never rewritten)", file=sys.stderr)
            return 2
    todo = [p for p in run_dir.glob("*/*/conversation.json") if not (p.parent / beside_name).exists()]
    estimate = round(len(todo) * ESTIMATE_EUR["judge"], 2)
    print(json.dumps({"run": run, "to_judge": len(todo), "judge": args.judge_model, "estimate_eur": estimate}, indent=1))
    if not args.yes:
        print("nothing was sent: add --yes and --cap-eur")
        return 0
    if args.cap_eur is None or args.cap_eur <= 0:
        print("refused: --cap-eur above zero is required", file=sys.stderr)
        return 2
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    from dclab_rnd.models import settings as _settings
    from dclab_rnd.models.gateway import Bound

    from ..conversation_exam import EXAM, Budget, _Capped, practice_effort

    tier = judge_tier(args.judge_model)
    why = owner_allows(args.cap_eur) if tier is not None else "no key serves the judge's endpoint, or the provider is blocked"
    if "claude" in str(args.judge_model).lower():
        why = "the owner's rule of 2026-11-15: no Claude model in any test, arm or judge, at any provider"
    if args.judge_model == (info.get("models") or {}).get("arms"):
        why = "the judge must not be the model it judges"
    if why:
        print(f"refused: {why}", file=sys.stderr)
        return 2
    price_snapshot(run_dir, args.judge_model)
    gateway = run_gateway(run_dir)
    practice_effort()
    budget = Budget(gateway, args.cap_eur)
    tag = f"{EXAM}{run}:"

    def judge_client(name: str, arm: str) -> Any:
        bound = Bound(gateway, PURPOSE["judge"], _settings.purpose(PURPOSE["judge"]), tier, None, f"{tag}judge:{arm}-{name}", fixed=True)
        budget.clients.append((f"{tag}judge:{arm}-{name}", bound))
        return _Capped(bound, budget)

    result = judge_kept(run_dir, key, judge_client, run=run, spent_for=lambda name, arm: budget.spent_eur(f"{tag}judge:{arm}-{name}"),
                        beside_name=beside_name)
    left = sum(1 for c in result["conversations"] if any(t.get("role") == "assistant" and t.get("text") for t in c.get("transcript") or [])
               and not (c.get("judged") or {}).get("valid"))
    if left:  # the valid judgments are kept beside their conversations: the same command asks only for the ones still missing
        print(f"{left} conversations are not judged yet ({result['incomplete'] or 'the judge gave no valid answer'}): no result stored; "
              f"run the same command again. Spent €{budget.spent_eur():.4f}", file=sys.stderr)
        return 2
    write_scorecard(run_dir / ("scorecard_judged.csv" if beside_name == "judged.json" else f"scorecard_{beside_name[:-5]}.csv"), result["scores"])
    record = {"id": run, "split": "dev", "set": SET, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "arms": info["arms"], "repeats": info["repeats"],
              "models": {**info["models"], "judge": args.judge_model}, "spent_eur": round(budget.spent_eur(), 4), "cap_eur": args.cap_eur,
              "judge_only": True, "kept": _rel(run_dir), "manifest": info, "capture": capture(run_dir), "label": info.get("label"),
              "note": "a pilot of eight cases: traces and reason codes, no claim that one arm is better; judged after the run, from the kept conversations", **result}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if result["incomplete"]:
        print(f"INCOMPLETE ({result['incomplete']})", file=sys.stderr)
    print(json.dumps({"summary": result["summary"], "first_divergence_counts": result["first_divergence_counts"]}, indent=1))
    print(f"spent €{budget.spent_eur():.4f}; stored {_rel(path)}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--only", default=None, help="a comma list of case ids")
    parser.add_argument("--set", default="dev", choices=sorted(SETS), help="the set of written people: dev (the pilot's eight) or dev2 (eight by an independent author)")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--cap-eur", type=float, default=None)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--label", default=None, help="a name for the state of the code this run measures (for example after-three-fixes): runs "
                                                      "of one model and one judge are read as one run only when their labels are the same")
    parser.add_argument("--arm-model", default=None, help="another model for every arm, named as the arms' endpoint names it "
                                                          "(for example anthropic/claude-sonnet-5.5 on OpenRouter): the same cases on another model")
    parser.add_argument("--freeze", action="store_true", help="pin the cases and their key (after the owner's review)")
    parser.add_argument("--no-judge", action="store_true", help="play the arms and keep the conversations and their first divergence; judge them later with --judge")
    parser.add_argument("--judge", default=None, help="a kept run folder: judge its conversations that are not judged yet, and store the judged result")
    parser.add_argument("--allow-dirty", action="store_true", help="the owner's choice: run on uncommitted code; the manifest keeps the commit, "
                                                                   "the diff's hash and the untracked files' hash, and the result says so")
    parser.add_argument("--recorded-dev", action="store_true", help="the free diagnosis: every kept dev exam run, plain against case")
    parser.add_argument("--receipt", default=None, help="a kept run folder: write its archive receipt (after make archive)")
    parser.add_argument("--snapshot", default=None, help="with --receipt: the snapshot folder make archive printed; the run's files are checked there")
    args = parser.parse_args(argv)
    if args.recorded_dev:
        found = _fd.recorded_dev()
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / "HAB-000_recorded_dev.json"
        if path.exists():
            print(f"refused: {path.relative_to(ROOT)} exists (a result is never rewritten)", file=sys.stderr)
            return 2
        path.write_text(json.dumps({"id": "HAB-000", "split": "dev", "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    "what": "the kept dev conversation-exam runs read again: no model call", "spent_eur": 0.0, **found},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({k: found[k] for k in ("lost", "won", "first_reason_counts")}, indent=1))
        print(f"stored {path.relative_to(ROOT)}")
        return 0
    if args.receipt:
        got = receipt(Path(args.receipt), Path(args.snapshot) if args.snapshot else None)
        print(json.dumps({k: v for k, v in got.items() if k != "file_sha256"}, indent=1))
        return 0 if got["archived"] is not False else 2
    if args.set != SET:  # the default set is left as the module holds it
        use_set(args.set)
    cases, key = load()
    if args.freeze:
        PIN.write_text(fingerprint() + "\n", encoding="utf-8")
        print(f"frozen: {len(cases)} cases, {PIN.relative_to(ROOT)}")
        return 0
    if args.judge:
        return _judge_run(Path(args.judge), key, args)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if any(a not in ALL_ARMS for a in arms):
        parser.error(f"--arms takes {', '.join(ALL_ARMS)}")
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    conversations = len(cases) * len(arms) * args.repeats
    estimate = round(len(cases) * args.repeats * sum(ESTIMATE_EUR[a] + (0 if args.no_judge else ESTIMATE_EUR["judge"]) for a in arms), 2)
    state = code_state()
    blockers = [b for b in (frozen(), "the lead agent's code has uncommitted edits: commit them, so arm C is one commit (or --allow-dirty, the owner's choice)"
                            if state["dirty"] and not args.allow_dirty else None) if b]
    print(json.dumps({"cases": len(cases), "arms": arms, "repeats": args.repeats, "conversations": conversations, "estimate_eur": estimate,
                      "judge": "later (--judge)" if args.no_judge else args.judge_model, "commit": state["commit"], "uncommitted": state["dirty"][:8], "blockers": blockers}, indent=1))
    if not args.yes:
        print("nothing was sent: add --yes and --cap-eur")
        return 0
    if blockers:
        print("refused: " + "; ".join(blockers), file=sys.stderr)
        return 2
    if args.cap_eur is None or args.cap_eur <= 0:
        print("refused: --cap-eur above zero is required", file=sys.stderr)
        return 2
    if args.cap_eur < estimate:
        print(f"note: the cap (€{args.cap_eur:.2f}) is under the estimate (€{estimate:.2f}): the run stops at the cap, marked incomplete", file=sys.stderr)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    from dclab_rnd.agents import case_agent as _case_agent
    from dclab_rnd.models import settings as _settings
    from dclab_rnd.models.gateway import Bound

    from ..conversation_exam import EXAM, Budget, _Capped, practice_effort
    from ..conversation_exam.arms import CaseArm, PlainArm
    from .arms import BlueprintArm, ModelOnlyArm, ModelOnlyShortArm, OpenArm

    tier = None if args.no_judge else judge_tier(args.judge_model)
    if tier is None and not args.no_judge:
        print("refused: no key serves the judge's endpoint (a model that is not the arms'), or the provider is blocked", file=sys.stderr)
        return 2
    why = owner_allows(args.cap_eur)
    if why:
        print(f"refused: {why}", file=sys.stderr)
        return 2
    run = next_id()
    run_dir = KEEP_ROOT / f"{datetime.now(timezone.utc):%Y-%m-%d}-harness-ablation-{run}"
    if "claude" in str(args.arm_model or "").lower() or "claude" in str(args.judge_model or "").lower():
        print("refused: the owner's rule of 2026-11-15: no Claude model in any test, arm or judge, at any provider", file=sys.stderr)
        return 2
    if args.arm_model:
        price_snapshot(run_dir, args.arm_model)  # before the gateway reads its prices: the run's cap must be able to count it
    gateway = run_gateway(run_dir)
    unserved = sorted({PURPOSE[a] for a in arms if PURPOSE[a] and not gateway.available(PURPOSE[a])})
    if unserved:  # no model serves an arm (no key, or DCLAB_NO_LIVE_MODELS): a run of failures is never stored as a result
        shutil.rmtree(run_dir, ignore_errors=True)
        print(f"refused: no model serves {', '.join(unserved)}", file=sys.stderr)
        return 2
    practice_effort()
    budget = Budget(gateway, args.cap_eur)
    standard = args.arm_model or str((_settings.tier("standard") or {}).get("model") or "")
    route_model = args.arm_model or _case_agent.route_model()  # one model for everything an arm asks, when another model is named
    tag = f"{EXAM}{run}:"

    def client(key_: str, purpose: str, model: Optional[str] = None) -> Any:
        return budget.client(f"{tag}{key_}", purpose, model or args.arm_model)

    def judge_client(name: str, arm: str) -> Any:
        bound = Bound(gateway, PURPOSE["judge"], _settings.purpose(PURPOSE["judge"]), tier, None, f"{tag}judge:{arm}-{name}", fixed=True)
        budget.clients.append((f"{tag}judge:{arm}-{name}", bound))
        return _Capped(bound, budget)

    arm_for = {"model_only": lambda name, folder: ModelOnlyArm(client(f"model_only-{name}", PURPOSE["model_only"])),
               "model_only_short": lambda name, folder: ModelOnlyShortArm(client(f"model_only_short-{name}", PURPOSE["model_only_short"])),
               "blueprint_rules": lambda name, folder: BlueprintArm("blueprint_rules", None, name),
               "blueprint_pipeline": lambda name, folder: BlueprintArm("blueprint_pipeline", _ModelView(budget, f"{tag}blueprint_pipeline-{name}", args.arm_model), name),
               "blueprint_architect": lambda name, folder: BlueprintArm("blueprint_architect", _ModelView(budget, f"{tag}blueprint_architect-{name}", args.arm_model), name),
               "open": lambda name, folder: OpenArm(client(f"open-{name}", PURPOSE["open"])),
               "plain": lambda name, folder: PlainArm(client(f"plain-{name}", PURPOSE["plain"])),
               "case": lambda name, folder: CaseArm(client(f"case-{name}", PURPOSE["case"]), client(f"case-{name}", "case_specialist"), folder,
                                                    client(f"case-{name}", "route_check", route_model))}
    info = manifest(run, arms, args.repeats, args.cap_eur, {"arms": standard, "judge": None if args.no_judge else args.judge_model,
                                                            "route_check": route_model})
    info["label"] = args.label
    (run_dir / "manifest.json").write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    result = measure(cases, key, {a: arm_for[a] for a in arms}, None if args.no_judge else judge_client, repeats=args.repeats, keep=run_dir, run=run,
                     exhausted=lambda: budget.spent_eur() >= 0.98 * budget.cap_eur,
                     spent_for=lambda name, arm: {"arm": budget.spent_eur(f"{tag}{arm}-{name}"), "judge": budget.spent_eur(f"{tag}judge:{arm}-{name}")})
    write_scorecard(run_dir / "scorecard.csv", result["scores"])
    kept = capture(run_dir)
    record = {"id": run, "split": "dev", "set": SET, "at": info["at"], "arms": arms, "repeats": args.repeats, "models": info["models"],
              "spent_eur": round(budget.spent_eur(), 4), "cap_eur": args.cap_eur, "kept": _rel(run_dir), "manifest": info,
              "capture": kept, "label": args.label, "note": "a pilot of eight cases: traces and reason codes, no claim that one arm is better",
              "code": "uncommitted edits (the owner's choice): see manifest.code" if state["dirty"] else "one commit", **result}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / (f"{run}_dev_unjudged.json" if args.no_judge else f"{run}_dev.json")
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if result["incomplete"]:
        print(f"INCOMPLETE ({result['incomplete']})", file=sys.stderr)
    print(json.dumps({"summary": result["summary"], "first_divergence_counts": result["first_divergence_counts"], "capture": kept}, indent=1))
    if kept["without_thinking"]:
        print(f"WARNING: answered calls with no thinking kept, by purpose: {kept['without_thinking']}", file=sys.stderr)
    print(f"spent €{budget.spent_eur():.4f}; stored {_rel(path)}; kept {_rel(run_dir)}\n"
          f"next: make archive, then python -m dclab_rnd.agent_eval.harness_ablation --receipt {_rel(run_dir)} --snapshot <the snapshot it printed>")
    return 0
