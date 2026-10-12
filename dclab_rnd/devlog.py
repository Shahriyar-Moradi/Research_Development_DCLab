"""Development trajectories: how DCLab itself is built and fixed, as SFT examples for the owner's model.

The owner asked (2026-10-12) that every package's process be kept for training: the request, the plan, the decisions
and why, each finding (a reviewer's, a failing test's, a measurement's) with its failure scenario, the cause, the fix
and the test that proves it, and each measurement against its gate. One record per package lives in
``research/llm-fine-tuning/experiments/dev_trajectories/records/`` (written when the package is built, checked by
:class:`Trajectory`); ``python -m dclab_rnd.devlog`` (``make sft-dev``) turns them into examples:

- ``plan_package``: the request → the plan and the decisions with their reasons;
- ``review_change``: the package and what changed → the findings, each with the scenario that shows it;
- ``fix_finding``: one finding and its scenario → the cause, the smallest fix and the test that proves it;
- ``judge_measurement``: a gate fixed before the run and the numbers → the verdict and what follows;
- ``retrospective``: the request and the outcome → the lessons the next package should keep.

Whole packages are held out for validation (by the hash of the package id, about one in five), so validation measures
work the model has not seen. The gates of corpus v3 apply: no absolute path, no long hex hash, no e-mail address; a
record never holds a row of anyone's data (the challenge data in particular: names, decisions and aggregates only).
The output is ``research/llm-fine-tuning/experiments/sft/out_dev/`` (chat and alpaca formats, and a manifest), built
from the committed records only, so it is committed and checked for freshness like v3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "research/llm-fine-tuning/experiments/dev_trajectories/records"
OUT = ROOT / "research/llm-fine-tuning/experiments/sft/out_dev"
SYSTEM = ("You are the senior engineer who builds DCLab, a system that makes machine-learning work leakage-safe and honestly "
          "evaluated. Work package by package: state the plan and the decision behind each choice; review a change by naming "
          "concrete defects, each with the scenario that shows it; fix a defect at its cause with the smallest change and a "
          "test that fails without it; judge a measurement only against the gate fixed before it ran, and never rewrite a "
          "failed gate after the fact. Never claim production readiness from benchmark evidence alone.")
VAL_SHARE = 0.2
REFUSED = [(re.compile(r"(/Users/|/home/|[A-Z]:\\\\)"), "an absolute path"), (re.compile(r"\b[0-9a-f]{16,}\b"), "a long hex hash"),
           (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "an e-mail address")]


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: str = Field(min_length=3)
    why: str = Field(min_length=3)


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    source: Literal["reviewer", "test", "measurement", "owner", "self"]
    defect: str = Field(min_length=10)
    scenario: str = Field(min_length=10)
    cause: str = ""
    fix: str = Field(min_length=10)
    test: str = ""
    severity: Literal["behaviour", "latent", "gap", "flaky", "privacy", "leakage"] = "behaviour"
    impact: str = ""  # what this defect would have done, specifically (who gets what wrong result, and when)
    alternatives: list[str] = Field(default_factory=list)  # fixes considered and set aside, each with why
    open: str = ""  # what the fix leaves open, when something does


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    gate: str = Field(min_length=10)
    fixed_before_run: bool = True
    result: str = Field(min_length=5)
    verdict: Literal["pass", "fail", "not scored"]
    cost_eur: float | None = None
    next: str = ""
    against_gate: str = ""  # each number set against its threshold, using only the numbers of the gate and the result
    limits: list[str] = Field(default_factory=list)  # what this measurement cannot show, specifically


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    package: str
    title: str
    date: str
    request: str = Field(min_length=10)
    spec: str = ""
    plan: list[str] = Field(min_length=1)
    decisions: list[Item] = Field(default_factory=list)
    change: str = ""  # what the change did, in words: the input of a review
    findings: list[Finding] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    outcome: str = Field(min_length=10)
    lessons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)  # what could go wrong in this package, as it looked before building it
    commits: list[str] = Field(default_factory=list)


def load(directory: Path = RECORDS) -> list[Trajectory]:
    return [Trajectory(**json.loads(p.read_text(encoding="utf-8"))) for p in sorted(directory.glob("*.json"))]


def _split(package: str) -> str:
    return "val" if int(hashlib.sha256(package.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < VAL_SHARE else "train"


# Each task has its own instruction and a fixed answer structure (the shape of corpus v3's grounded traces, made
# stricter: the answer reasons from the record, never restates the prompt, and invents no number).
TASK_SYSTEM = {
    "plan_package": SYSTEM + " Task: plan a package. Answer with exactly these parts: (1) What the package is for, in one or two "
                             "sentences; (2) Plan, as ordered steps; (3) Decisions, each with the reason that makes it right here; "
                             "(4) Risks to check before calling it done; (5) How you will know it is done.",
    "review_change": SYSTEM + " Task: review a change before it is committed. Answer with exactly these parts: (1) Findings, most "
                              "severe first, each with its severity, the concrete scenario (inputs, then the wrong result), why it "
                              "matters, and the fix you would ask for; (2) Verdict.",
    "fix_finding": SYSTEM + " Task: fix a defect. Answer with exactly these parts: (1) Root cause; (2) Why it matters; (3) The "
                            "smallest fix; (4) The test that proves it (it fails before the fix and passes after); (5) What stays open.",
    "judge_measurement": SYSTEM + " Task: judge a measurement. Answer with exactly these parts: (1) Observed evidence, each point "
                                  "from the numbers given; (2) Against the gate; (3) Decision; (4) Uncertainty and limits; (5) The "
                                  "smallest falsifiable next step. Use only numbers that were given; never round, convert or compute one.",
    "retrospective": SYSTEM + " Task: a retrospective. Answer with exactly these parts: (1) What happened, in one or two sentences; "
                              "(2) Lessons the next package should keep, each one a rule you could check.",
}
WHY = {  # why a finding of each kind matters, in DCLab's own rules
    "leakage": "It lets information that is not known at the prediction moment reach the model (rule 1): every score after it is "
               "optimistic, and a decision built on it would fail when the information is not there.",
    "behaviour": "It changes what DCLab does for a person without anyone deciding it: a result, a refusal or an artefact differs "
                 "from what the person was promised, silently.",
    "latent": "Nothing fails on today's paths, but the first caller to take the other path gets a wrong result without any error.",
    "gap": "The package claims something the code does not do yet; left as it is, a record, a page or a report would overstate "
           "what DCLab does.",
    "flaky": "A test that fails by chance teaches people to ignore red results, and a real failure then hides among the false ones.",
    "privacy": "Data, a key or a request could leave the workspace or reach a model against the owner's rules; that cannot be undone.",
}
FIX_ONLY = re.compile(r"^(?:not fixed|no fix|left open|recorded, not fixed)", re.I)
NOT_EVIDENCE = ("A single run on seeded or replica data is evidence about the method on these cases, not a reason to rely on a "
                "model in use.")


def _numbered(lines: list[str]) -> str:
    return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))


def _context(t: Trajectory) -> str:
    return f"Package {t.package}: {t.title} ({t.date})." + (f"\n\nWhat the package must deliver:\n{t.spec}" if t.spec else "")


def _plan_answer(t: Trajectory) -> str:
    parts = [f"(1) What the package is for\n{t.request}", f"(2) Plan\n{_numbered(t.plan)}"]
    if t.decisions:
        parts.append("(3) Decisions\n" + "\n".join(f"- {d.decision} Because: {d.why}" for d in t.decisions))
    risks = t.risks or [f.defect for f in t.findings if f.severity in ("leakage", "privacy", "behaviour")][:4]
    if risks:
        parts.append("(4) Risks to check before calling it done\n" + "\n".join(f"- {r}" for r in risks))
    gates = [m for m in t.measurements if m.fixed_before_run]
    done = [f"- {m.name}: {m.gate}" for m in gates[:3]]
    if done:
        parts.append("(5) How you will know it is done\n" + "\n".join(done))
    return "\n\n".join(parts)


def _finding_lines(f: Finding) -> str:
    return (f"- {f.id} [{f.severity}] {f.defect}\n  Scenario: {f.scenario}\n  Why it matters: {f.impact or WHY[f.severity]}\n"
            f"  Fix to ask for: {f.fix}")


def _fix_answer(f: Finding) -> str:
    parts = [f"(1) Root cause\n{f.cause or 'Not recorded beyond the scenario: find it before changing code.'}",
             f"(2) Why it matters\n{f.impact + ' ' if f.impact else ''}{WHY[f.severity]}",
             f"(3) The smallest fix\n{f.fix}" + ("\nSet aside:\n" + "\n".join(f"- {a}" for a in f.alternatives) if f.alternatives else "")]
    if f.test:
        parts.append(f"(4) The test that proves it\n{f.test}")
    else:
        parts.append("(4) The test that proves it\nNo dedicated test was recorded for this fix: write one that reproduces the scenario "
                     "above and fails before the change.")
    parts.append("(5) What stays open\n" + (f.open or ("The fix as recorded leaves the defect itself open; it is a stated limit, not a "
                                                       "solved problem." if FIX_ONLY.search(f.fix) else "Nothing beyond the scenario, once the test passes.")))
    return "\n\n".join(parts)


def _judge_answer(m: Measurement) -> str:
    against = {"pass": "The numbers meet the gate as it was written.", "fail": "The numbers do not meet the gate as it was written.",
               "not scored": "There is no pass mark here: it is a measurement to read, not a gate to pass."}[m.verdict]
    if m.against_gate:
        against = f"{m.against_gate} {against}"
    if not m.fixed_before_run:
        against += " The gate was written after the run, so this describes the run rather than testing it."
    decision = {"pass": "Accept the result as evidence for this package's claim, within the limits below.",
                "fail": "Do not switch on or claim what the gate guarded; keep the failed gate on record as it is, and write any "
                        "new gate before its own run.",
                "not scored": "Record it, and use it to choose the next step; it decides nothing on its own."}[m.verdict]
    limits = [*m.limits, NOT_EVIDENCE]
    if m.cost_eur:
        limits.append(f"It cost EUR {m.cost_eur:g}; a repeat on new seeds or cases would say how stable it is.")
    parts = [f"(1) Observed evidence\n- {m.result}", f"(2) Against the gate\n{against}", f"(3) Decision\n{decision}",
             "(4) Uncertainty and limits\n" + "\n".join(f"- {line}" for line in limits),
             f"(5) The smallest falsifiable next step\n{m.next or 'Repeat it once on cases it has not seen, with the gate fixed first.'}"]
    return "\n\n".join(parts)


def examples(t: Trajectory) -> list[dict]:
    """The SFT examples of one trajectory: (task, system, prompt, answer)."""
    head = _context(t)
    out = [("plan_package", f"{head}\n\nThe request: {t.request}\n\nPlan the work.", _plan_answer(t))]
    reviewed = [f for f in t.findings if f.source == "reviewer"]
    if t.change and reviewed:
        order = ("leakage", "privacy", "behaviour", "latent", "gap", "flaky")
        ranked = sorted(reviewed, key=lambda f: order.index(f.severity))
        answer = ("(1) Findings\n" + "\n".join(_finding_lines(f) for f in ranked)
                  + f"\n\n(2) Verdict\nFAIL: {len(ranked)} finding{'s' if len(ranked) != 1 else ''} to fix before this change is committed.")
        out.append(("review_change", f"{head}\n\nThe change: {t.change}\n\nReview it.", answer))
    for f in t.findings:
        out.append(("fix_finding", f"{head}\n\nThe change: {t.change or t.request}\n\nA finding ({f.source}, {f.severity}): {f.defect}\n"
                                   f"How it shows: {f.scenario}\n\nDiagnose and fix it.", _fix_answer(f)))
    for m in t.measurements:
        fixed = "fixed before the run" if m.fixed_before_run else "written after the run"
        cost = f" Cost: EUR {m.cost_eur:g}." if m.cost_eur else ""
        out.append(("judge_measurement", f"{head}\n\nMeasurement: {m.name}.\nThe gate ({fixed}): {m.gate}\nThe numbers: {m.result}{cost}"
                                         "\n\nJudge it.", _judge_answer(m)))
    if t.lessons:
        out.append(("retrospective", f"{head}\n\nThe request: {t.request}\n\nThe outcome: {t.outcome}\n\nLook back on it.",
                    f"(1) What happened\n{t.outcome}\n\n(2) Lessons\n" + "\n".join(f"- {lesson}" for lesson in t.lessons)))
    return [{"task": task, "system": TASK_SYSTEM[task], "prompt": prompt, "answer": answer, "package": t.package}
            for task, prompt, answer in out]


def ungrounded(example: dict) -> set[str]:
    """Numbers in a judged answer that its prompt does not give (only the evidence-reading tasks must be grounded)."""
    if example["task"] not in ("judge_measurement",):
        return set()
    from dclab_rnd.cited import numbers

    stated = numbers(example["prompt"] + "\n" + example["system"], source=True)
    evidence = example["answer"].split("(5) The smallest falsifiable next step")[0]  # the next step plans; the parts before report
    return numbers(evidence) - stated


def check(text: str) -> str | None:
    """Why a text cannot go into the corpus, or None."""
    for pattern, what in REFUSED:
        if pattern.search(text):
            return what
    return None


def build(records: list[Trajectory] | None = None) -> dict:
    records = load() if records is None else records
    splits: dict[str, list[dict]] = {"train": [], "val": []}
    refused, seen = [], set()
    for t in records:
        for ex in examples(t):
            why = check(ex["prompt"] + "\n" + ex["answer"]) or ("a number the prompt does not give" if ungrounded(ex) else None)
            if why:
                refused.append({"package": t.package, "task": ex["task"], "why": why})
                continue
            if ex["prompt"] in seen:
                continue
            seen.add(ex["prompt"])
            splits[_split(t.package)].append(ex)
    by_task: dict[str, int] = {}
    for rows in splits.values():
        for ex in rows:
            by_task[ex["task"]] = by_task.get(ex["task"], 0) + 1
    manifest = {"schema_version": 2, "source": "research/llm-fine-tuning/experiments/dev_trajectories/records", "system_prompts": TASK_SYSTEM,
                "packages": sorted(t.package for t in records), "total": sum(len(v) for v in splits.values()),
                "train": len(splits["train"]), "val": len(splits["val"]), "by_task": dict(sorted(by_task.items())),
                "val_packages": sorted({ex["package"] for ex in splits["val"]}), "refused": refused,
                "validation_policy": f"whole packages held out by the hash of their id (about {VAL_SHARE:.0%})"}
    return {"splits": splits, "manifest": manifest}


def write(built: dict, out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in built["splits"].items():
        with open(out / f"{split}.chat.jsonl", "w", encoding="utf-8") as chat, open(out / f"{split}.alpaca.jsonl", "w", encoding="utf-8") as alpaca:
            for ex in rows:
                meta = {"task": ex["task"], "package": ex["package"]}
                chat.write(json.dumps({"messages": [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["prompt"]},
                                                    {"role": "assistant", "content": ex["answer"]}], "metadata": meta}, ensure_ascii=False) + "\n")
                alpaca.write(json.dumps({"instruction": ex["system"], "input": ex["prompt"], "output": ex["answer"], "metadata": meta},
                                        ensure_ascii=False) + "\n")
    (out / "MANIFEST.json").write_text(json.dumps(built["manifest"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the development-trajectory SFT examples")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    built = build()
    write(built, args.out)
    m = built["manifest"]
    print(f"{m['total']} examples from {len(m['packages'])} packages ({m['train']} train, {m['val']} val); by task {m['by_task']}"
          + (f"; refused {len(m['refused'])}" if m["refused"] else ""))
    return 1 if m["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
