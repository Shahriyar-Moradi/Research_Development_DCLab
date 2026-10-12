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


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    gate: str = Field(min_length=10)
    fixed_before_run: bool = True
    result: str = Field(min_length=5)
    verdict: Literal["pass", "fail", "not scored"]
    cost_eur: float | None = None
    next: str = ""


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
    commits: list[str] = Field(default_factory=list)


def load(directory: Path = RECORDS) -> list[Trajectory]:
    return [Trajectory(**json.loads(p.read_text(encoding="utf-8"))) for p in sorted(directory.glob("*.json"))]


def _split(package: str) -> str:
    return "val" if int(hashlib.sha256(package.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < VAL_SHARE else "train"


def _numbered(lines: list[str]) -> str:
    return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))


def examples(t: Trajectory) -> list[dict]:
    """The SFT examples of one trajectory: (task, prompt, answer)."""
    head = f"Package {t.package}: {t.title}."
    spec = f"\n\nWhat the package must deliver:\n{t.spec}" if t.spec else ""
    out = []
    plan = f"Plan:\n{_numbered(t.plan)}"
    if t.decisions:
        plan += "\n\nDecisions:\n" + "\n".join(f"- {d.decision} Why: {d.why}" for d in t.decisions)
    out.append(("plan_package", f"{head}\n\nThe request: {t.request}{spec}\n\nPlan the work and state the decisions with their reasons.", plan))
    reviewed = [f for f in t.findings if f.source == "reviewer"]
    if t.change and reviewed:
        answer = "\n\n".join(f"{f.id} ({f.severity}). {f.defect}\nScenario: {f.scenario}" for f in reviewed)
        out.append(("review_change", f"{head}{spec}\n\nThe change: {t.change}\n\nReview it: name each concrete defect and the scenario "
                                     "that shows it.", answer))
    for f in t.findings:
        answer = (f"Cause: {f.cause}\n\n" if f.cause else "") + f"Fix: {f.fix}" + (f"\n\nTest: {f.test}" if f.test else "")
        out.append(("fix_finding", f"{head}\n\nA finding ({f.source}): {f.defect}\nScenario: {f.scenario}\n\nFind the cause and fix it "
                                   "with the smallest change, and say which test proves it.", answer))
    for m in t.measurements:
        answer = f"Verdict: {m.verdict}. {m.result}" + (f"\n\nNext: {m.next}" if m.next else "")
        fixed = "fixed before the run" if m.fixed_before_run else "written after the run (not a valid gate)"
        out.append(("judge_measurement", f"{head}\n\nMeasurement: {m.name}. The gate ({fixed}): {m.gate}\n\nThe numbers: {m.result}\n\n"
                                         "Judge it against the gate and say what follows.", answer))
    if t.lessons:
        out.append(("retrospective", f"{head}\n\nThe request: {t.request}\n\nThe outcome: {t.outcome}\n\nWhat should the next package "
                                     "keep from this one?", "\n".join(f"- {lesson}" for lesson in t.lessons)))
    return [{"task": task, "prompt": prompt, "answer": answer, "package": t.package} for task, prompt, answer in out]


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
            why = check(ex["prompt"] + "\n" + ex["answer"])
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
    manifest = {"schema_version": 1, "source": "research/llm-fine-tuning/experiments/dev_trajectories/records", "system_prompt": SYSTEM,
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
                chat.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["prompt"]},
                                                    {"role": "assistant", "content": ex["answer"]}], "metadata": meta}, ensure_ascii=False) + "\n")
                alpaca.write(json.dumps({"instruction": SYSTEM, "input": ex["prompt"], "output": ex["answer"], "metadata": meta},
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
