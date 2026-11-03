"""Agent versions and their timeline (B5 of the 2026-11-03 plan): what each agent ran with, when it changed, and how it scored.

An agent's version is what its model is told: its prompt file (``dclab_rnd/prompts/<name>.md``) and the tool schemas of its
scope, each as the short content hash ``prompts.fingerprints()`` gives (the same hashes the prompt gate checks). This module
puts three records side by side:

- **the history** of each prompt from git: every commit that changed it, with the hash the file had after that commit;
- **the checks**: results in ``evidence/campaigns`` that record each hash (``prompt_hashes``). These are the scripted suites
  (``make agent-eval``, the prompt gate): they record every prompt's hash but send no prompt to a model, so they say a version
  was checked, never that it was measured;
- **the live results**: the suites that do send a prompt to a model (``SUITE_PROMPTS``: the lead-agent exam, Home, the journeys)
  record no hash, so each of their results is placed in the version that was committed before its date ("linked by date": a
  run on uncommitted edits is placed in the version before them);
- **the current** hash in the working tree, and whether a live result has run on it yet (``measured_now``).

So a prompt change shows its date, its commit and the scores before and after it; a current version no suite has run on is
named, because a prompt is code and it was measured only if a result says so.

    python -m dclab_rnd.bench versions [--agent case_agent]
    GET /api/admin/bench/versions
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from dclab_core import bench as core

ROOT = Path(__file__).resolve().parents[2]
# the suites that send these prompts to a model (read from the code that loads them: agents/case_agent.py, the exam's arms,
# draft/chat.py, intent.py, understand.py, orchestrator.py, business.py, plan.py); a suite not here measures no prompt
# the conversation exam names its arms: a prompt is measured by a result only when an arm that sends it ran there
ARM_PROMPTS: dict[str, tuple[str, ...]] = {"case": ("case_agent", "route_check"), "plain": ("exam_plain",)}
EVERY_ARM = ("exam_person", "exam_judge")  # the written person and the judge take part in every arm's conversations
SUITE_PROMPTS: dict[str, tuple[str, ...]] = {
    "conversation_exam_v1": ("case_agent", "route_check", "exam_plain", "exam_person", "exam_judge"),
    "home_v1": ("home_agent", "home_intent", "home_understand", "home_orchestrator", "home_business", "home_plan", "home_query"),
    "journeys_v1": ("home_agent", "home_intent", "home_understand", "home_orchestrator"),
    "journeys_built_v1": ("home_agent", "home_orchestrator"),
}
PROMPTS = "dclab_rnd/prompts"
CAMPAIGNS = ROOT / "evidence" / "campaigns"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _blob_hash(commit: str, path: str) -> Optional[str]:
    try:
        data = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return hashlib.sha256(data).hexdigest()[:16] if data else None


@lru_cache(maxsize=4)
def history(head: str) -> dict[str, list[dict[str, Any]]]:
    """Every prompt's versions from git, oldest first: the commit, its date and subject, and the file's hash after it (a deletion
    has no hash). ``head`` keys the cache: a new commit reads it again."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    log = _git("log", "--reverse", "--format=@@%H|%cI|%s", "--name-only", "--", PROMPTS)  # the commit date: when it reached the branch
    commit = None
    for line in log.splitlines():
        if line.startswith("@@"):
            sha, date, subject = (line[2:].split("|", 2) + ["", ""])[:3]
            commit = {"commit": sha[:12], "date": date, "subject": subject[:160]}
        elif line.endswith(".md") and line.startswith(PROMPTS) and commit:
            name = Path(line).stem
            digest = _blob_hash(commit["commit"], line)
            if not out[name] or out[name][-1]["hash"] != digest:
                out[name].append({**commit, "hash": digest})
    return dict(out)


def results_by_hash(campaigns: Path = CAMPAIGNS) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """For each fingerprint key (``prompt:case_agent``, ``tools:project``), the results that ran on each of its hashes."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(campaigns).glob("*/results/*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        prints = doc.get("prompt_hashes") or doc.get("fingerprints")
        if not isinstance(prints, dict):
            continue
        passed, words = core.verdict_of(doc)
        row = {"suite": path.parent.parent.name, "ref": path.stem, "at": str(doc.get("at") or doc.get("completed_at") or ""),
               "split": core.split_of(doc, path.stem), "passed": passed, "verdict": words[:120]}
        for key, value in prints.items():
            if isinstance(value, str) and (key.startswith("prompt:") or key.startswith("tools:")):
                out[key][value].append(row)
    return {k: dict(v) for k, v in out.items()}


def live_results(campaigns: Path = CAMPAIGNS) -> dict[str, list[dict[str, Any]]]:
    """For each prompt, the results of the suites that send it (``SUITE_PROMPTS``), oldest first, with what they spent."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for suite, names in SUITE_PROMPTS.items():
        for path in sorted((Path(campaigns) / suite / "results").glob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(doc, dict):
                continue
            at = str(doc.get("at") or doc.get("completed_at") or "")
            if not at:
                continue  # a result without a date cannot be placed in a version
            passed, words = core.verdict_of(doc)
            spent = doc.get("spent_eur")
            row = {"suite": suite, "ref": path.stem, "at": at, "split": core.split_of(doc, path.stem), "passed": passed,
                   "verdict": words[:120], "spent_eur": spent if isinstance(spent, (int, float)) else None}
            for name in _sent(suite, doc, names):
                out[name].append(row)
    return {k: sorted(v, key=lambda r: (r["at"], r["ref"])) for k, v in out.items()}


def _sent(suite: str, doc: dict[str, Any], names: tuple[str, ...]) -> tuple[str, ...]:
    """The prompts a result really sent: for the exam, those of the arms that ran there (a relabelling or a pair check without
    arms sent none); for the other suites, the mapped prompts when the result ran conversations (it has scores or arms)."""
    arms = doc.get("arms") or (list(doc["scores"]) if isinstance(doc.get("scores"), dict) else None)
    if not arms:
        return ()
    if suite != "conversation_exam_v1":
        return names
    sent = {p for a in arms for p in ARM_PROMPTS.get(str(a), ())} | set(EVERY_ARM)
    return tuple(n for n in names if n in sent)


def _when(text: str) -> str:
    """A date as UTC text, so a commit's date and a result's date compare."""
    from datetime import datetime, timezone

    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat()


def current() -> dict[str, str]:
    from ..prompts import fingerprints

    return fingerprints()


def timeline(agent: Optional[str] = None, campaigns: Path = CAMPAIGNS, now: Optional[dict[str, str]] = None) -> list[dict[str, Any]]:
    """One entry per prompt (or the one named): its versions oldest first, each with the results that ran on it and their
    passes and failures; the current hash, whether it is a committed version, and whether any result ran on it."""
    head = _git("rev-parse", "HEAD").strip()
    past = history(head)
    runs = results_by_hash(campaigns)
    live = live_results(campaigns)
    now = now if now is not None else current()
    names = sorted({k.split(":", 1)[1] for k in now if k.startswith("prompt:")} | set(past))
    out = []
    for name in names:
        if agent and name != agent:
            continue
        key = f"prompt:{name}"
        on = runs.get(key, {})
        versions = []
        hist = past.get(name, [])
        for i, v in enumerate(hist):
            start, end = _when(v["date"]), _when(hist[i + 1]["date"]) if i + 1 < len(hist) else None
            checks = sorted(on.get(v["hash"], []), key=lambda r: (r["at"], r["ref"]))
            runs_here = [r for r in live.get(name, []) if _when(r["at"]) >= start and (end is None or _when(r["at"]) < end)]
            versions.append({**v, "checks": len(checks), "checks_passed": sum(r["passed"] is True for r in checks),
                             "live": runs_here, "passed": sum(r["passed"] is True for r in runs_here),
                             "failed": sum(r["passed"] is False for r in runs_here),
                             "spent_eur": round(sum(r["spent_eur"] or 0 for r in runs_here), 2)})
        here = now.get(key)
        committed = bool(versions) and versions[-1]["hash"] == here
        measured = committed and bool(versions[-1]["live"])  # a live run on the version in force now
        last_pass = next((v for v in reversed(versions) if v["passed"]), None)
        out.append({"agent": name, "current": here, "committed": committed, "measured_now": measured,
                    "live_suites": [s for s, names in SUITE_PROMPTS.items() if name in names],
                    "versions": versions, "changes": len(versions),
                    "since_last_pass": None if last_pass is None else sum(1 for v in versions if _when(v["date"]) > _when(last_pass["date"])),
                    "last_pass": None if last_pass is None else {"hash": last_pass["hash"], "date": last_pass["date"]}})
    return out


__all__ = ["history", "results_by_hash", "live_results", "current", "timeline", "SUITE_PROMPTS"]
