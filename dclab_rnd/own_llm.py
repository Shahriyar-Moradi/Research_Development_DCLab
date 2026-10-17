"""A company's own language model's data (package 27.10): its sessions as training files, kept in its own workspace.

The 26.7 path (a company's own decision models) extended to sessions: a workspace's own conversations, agent runs, built
projects' workflows and labelled decisions become examples in the lake's gold shapes, cleaned (``dclab_core.clean``),
with a verified reasoning trace where one can be written, and written as the chat files a LoRA run reads, inside
``<workspace>/own_models/llm/``. Nothing leaves the folder, so DCLab's own ``train`` consent does not apply; a table's
values never enter (the sets hold states, bands and the person's words, never cells); a move Claude made stays out of
training (Anthropic's terms bind the company too) and goes to the evaluation file; the sealed share (whole projects, by
hash) is in its own file, for the company's own measurement, never trained on. The training run itself is the company's
(the recipe in ``research/llm-fine-tuning/experiments/lake/TRAINING.md``).

    python -m dclab_rnd.own_llm export [--home DIR]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

FOLDER = Path("own_models") / "llm"


def examples(events: Iterable[dict[str, Any]], records: Iterable[dict[str, Any]], workflow_items: Iterable[dict[str, Any]],
             workspace: str = "own") -> tuple[list[Any], Counter]:
    """The workspace's sessions and decisions as training examples (the lake's shapes and mapping), with why any was left out."""
    from dclab_core import datasets as builders
    from dclab_core.ledger import POINTS

    from .lake import Run, clean_examples, decision_samples, reason, session_examples
    from .lake.train import _decision_messages

    run, events, records = Run(), list(events), list(records)
    basis = "own: the company's own workspace, its data kept in its folder"
    other: dict[str, list[dict[str, Any]]] = {}
    for builder, (rows, _) in (("conversations", builders.conversations(events, workspace=workspace, consent=basis)),
                               ("trajectories", builders.trajectories(events, workspace=workspace, consent=basis)),
                               ("workflows", builders.workflows(workflow_items, workspace=workspace, consent=basis))):
        other[f"{builder}/own"] = [{"row": r, "from_sealed": r.get("split") == "sealed", "source": f"own:{builder}"} for r in rows]
    found = session_examples(other, run)
    found, _ = reason(found)
    kept, why = clean_examples(found, set())
    from dclab_core import lake as L

    decisions = []
    for point in sorted({d.get("point") for d in records if d.get("point") in POINTS}):
        rows, _ = builders.decisions(records, point, workspace=workspace, consent=basis)
        decisions += decision_samples([r for r in rows if r.get("split") != "sealed"], f"own:decisions/{point}", "train", False, run)
        decisions += decision_samples([r for r in rows if r.get("split") == "sealed"], f"own:decisions/{point}", "train", True, run)
    for d in decisions:  # a decision sample as the chat row a served decision model is asked (26.4's shape)
        kept.append(L.TrainingExample(example_id="od_" + d.sample_id[-24:], family="typed_decision",
                                      messages=_decision_messages({"state_text": d.state_text, "label": d.label}),
                                      verification=L.Verification(kind="label_match", passed=True, detail=f"label from {d.provenance.origin}"),
                                      quality=0.7, provenance=d.provenance))
    for source, reasons in run.dropped.items():
        why.update({f"{source}: {k}": v for k, v in reasons.items()})
    return kept, why


def export(home: Path, events: Optional[list[dict[str, Any]]] = None, records: Optional[list[dict[str, Any]]] = None,
           workflow_items: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    """The training files of the workspace's own language model, written inside its folder; returns the manifest."""
    from dclab_core.reasoning import think

    from dclab_core import lake as L

    if events is None or records is None or workflow_items is None:
        events, records, workflow_items = _read(home)
    kept, why = examples(events, records, workflow_items)
    out = Path(home) / FOLDER
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, list[str]] = {"train": [], "val": [], "sealed": [], "eval_only": []}
    for e in kept:
        split = e.provenance.split
        into = "val" if split == "dev" else split if split in files else "train"
        messages = e.messages
        if e.rationale and messages and messages[-1].get("role") == "assistant":
            messages = [*messages[:-1], {**messages[-1], "content": think(e.rationale.splitlines()) + "\n" + messages[-1]["content"]}]
        if into == "train" and e.provenance.origin not in L.TRAINABLE:
            into = "eval_only"
        files[into].append(json.dumps({"messages": messages, "metadata": {"family": e.family, "origin": e.provenance.origin,
                                                                          "group": e.provenance.group}}, ensure_ascii=False))
    for name, lines in files.items():
        (out / f"{name}.chat.jsonl").write_text("".join(x + "\n" for x in lines), encoding="utf-8")
    manifest = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "counts": {k: len(v) for k, v in files.items()},
                "by_family": dict(Counter(e.family for e in kept)), "by_origin": dict(Counter(e.provenance.origin for e in kept)),
                "left_out": dict(why),
                "rule": "Kept in this workspace's folder: train.chat.jsonl may train its own model, val.chat.jsonl validates it, sealed.chat.jsonl "
                        "measures it once, eval_only.chat.jsonl (a model's moves whose terms forbid training, Claude's) only compares. "
                        "No table value is in any file."}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def _read(home: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """The workspace's own events, decision records and built projects' workflows (through its services)."""
    import os

    from . import consent
    from .agentic.services import Services
    from .datasets import _workflow_items
    from .settings import Settings

    os.environ["DCLAB_AGENT_HOME"] = str(home)
    services = Services(Settings())
    events = [e.model_dump(mode="json") if hasattr(e, "model_dump") else dict(e) for e in services.experience.read(limit=10**7)]
    labels: dict[str, list[dict[str, Any]]] = {}
    for x in services.ledger.labels():
        labels.setdefault(x.record, []).append(x.model_dump(mode="json"))
    records = [{**r.model_dump(mode="json"), "labels": labels.get(r.id, [])} for r in services.ledger.records()]
    projects = {p["id"]: p for p in services.projects.list()}
    items = _workflow_items(services, projects)  # every project of its own (the company's data, kept here)
    _ = consent  # the company's own folder: DCLab's train switch is not what governs it
    return events, records, items


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    from .settings import Settings

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.own_llm", description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("export",))
    parser.add_argument("--home", default=None)
    args = parser.parse_args(argv)
    home = Path(args.home) if args.home else Path(Settings().agent_home)
    print(json.dumps(export(home), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
