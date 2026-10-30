"""Tier 3, the training sets (32.11 to 32.13): one folder per model type, each set with its train, dev and sealed splits,
its ``MANIFEST.json`` (:class:`dclab_core.store.SetManifest`) and its ``CARD.md``.

``write_set`` is the one writer: the split is each example's own (from its record's sealed key, one rule for every set,
``dclab_core.datasets.split_of``); a set's sealed partition is written once and never rewritten (rows that arrive sealed
later wait in ``sealed.pending.jsonl`` for the next version); every example parses as its contract before it is written.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable, Optional

from dclab_core.store import EXAMPLE_SHAPES, SetManifest

from ..layout import Layout
from ..store import now, write_json

SPLIT_RULE = ("dclab_core.datasets.split_of(the record's sealed key), salt dclab-datasets-v1: sealed below 0.2, dev below 0.4, else train; "
              "a public table (openml:<id>): the lake's split")


def standing(layout: Layout) -> dict[tuple[str, str], dict[str, Any]]:
    """Every rule curator's latest choice per (record, model type), refused ones included (read once per build)."""
    from ..store import Store

    store = Store(layout)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in sorted(store.entries("curation"), key=lambda e: (e.captured_at, e.version)):
        for line in store.read(entry).decode("utf-8").splitlines():
            d = json.loads(line)
            if str(d.get("curator", "")).startswith("rule:") and d.get("record") and d.get("model_type"):
                out[(d["record"], d["model_type"])] = d
    return out


def allowed(record: Any, model_type: str, choices: dict[tuple[str, str], dict[str, Any]]) -> bool:
    """A record may enter a model type's set: its consent allows training, and no curator kept it out or had its placement refused."""
    if record.provenance.consent != "train":
        return False
    choice = choices.get((record.id, model_type))
    return choice is None or (choice.get("quality") != "out" and not choice.get("refused") and choice.get("used") is not False)


def write_set(layout: Layout, model_type: str, folder: str, examples: Iterable[Any], *, version: str = "v1", sources: Iterable[dict[str, Any]] = (),
              about: str = "", decontaminated: Iterable[str] = ("the store's own sealed partition",), commit: Optional[str] = None,
              consent: str = "train") -> SetManifest:
    """Write one set; returns its manifest."""
    examples = list(examples)
    shape = examples[0].schema_version if examples else f"dclab_core.{EXAMPLE_SHAPES[model_type][0]}.v1"
    base = layout.sets(model_type, *folder.split("/"), version)
    base.mkdir(parents=True, exist_ok=True)
    sealed, pending = base / "sealed.jsonl", base / "sealed.pending.jsonl"
    sealed_lines = sealed.read_text(encoding="utf-8").splitlines() if sealed.exists() else []
    sealed_ids = {json.loads(line)["id"] for line in sealed_lines if line.strip()}
    by_split: dict[str, list[str]] = {"train": [], "dev": [], "sealed": []}
    from ..exams import guard  # 36.4: a frozen exam's row is never a training row

    exams, held_out = guard(layout), 0
    for ex in sorted(examples, key=lambda e: e.id):
        if ex.model_type != model_type:
            raise ValueError(f"{ex.id} is an example of {ex.model_type}, not {model_type}")
        split = "sealed" if ex.id in sealed_ids else ex.split  # once sealed, always sealed: never in train or dev again
        if split != "sealed" and exams and exams.why(ex):
            held_out += 1  # kept in its exam, out of train and dev
            continue
        by_split[split].append(ex.model_dump_json() + "\n")
    if exams:
        decontaminated = [*decontaminated, "the frozen exams (36.4)"]
    for split in ("train", "dev"):
        (base / f"{split}.jsonl").write_text("".join(by_split[split]), encoding="utf-8")
    later = [line for line in by_split["sealed"] if json.loads(line)["id"] not in sealed_ids]
    if not sealed_lines and later:
        sealed.write_text("".join(later), encoding="utf-8")  # written once, on the first build that has sealed rows
        later = []
    pending.write_text("".join(later), encoding="utf-8")
    waiting = len(later)
    sealed_count = len(sealed.read_text(encoding="utf-8").splitlines()) if sealed.exists() else 0
    labels = Counter(getattr(ex, "label_source", None) or getattr(ex, "decided_by", None) or getattr(ex, "evidence", None) or "none" for ex in examples)
    weights: dict[str, float] = {}
    for ex in examples:
        key = getattr(ex, "label_source", None) or "none"
        weights.setdefault(key, ex.weight)
    manifest = SetManifest(model_type=model_type, name=folder, version=version, shape=shape, sources=sorted(sources, key=lambda s: str(s.get("entry"))),
                           split_rule=SPLIT_RULE, counts={"train": len(by_split["train"]), "dev": len(by_split["dev"]), "sealed": sealed_count,
                                                          "sealed_pending": waiting, **({"exam_held_out": held_out} if held_out else {})},
                           label_sources=dict(labels), weights=weights, consent=consent, decontaminated=list(decontaminated), commit=commit,
                           built_at=now())
    write_json(base / "MANIFEST.json", json.loads(manifest.model_dump_json()))
    write_json(base / "REASONING.json", reasoning_counts(examples))  # the owner (2026-10-28): the gap closing, per source
    (base / "CARD.md").write_text(card(manifest, about), encoding="utf-8")
    return manifest


LOST_PREFIX = "lost before"

# Who answers without thinking (2026-10-30, the owner: the gap list counted ~460,000 rule and classifier rows as "no reasoning
# and no note"; the reasoning backfill is cancelled, so a row no model made needs only the note saying so)
NO_MODEL_ANSWERERS = ("rule", "classifier", "person", "student")
NO_MODEL_JUDGES = ("validator", "g4", "tests", "person", "rule", "measured")
NO_MODEL_PROCESSES = ("job", "walk", "stages")  # the job queue, the record's walk, the engine's stages: code's steps


def why_no_reasoning(row: dict[str, Any]) -> Optional[str]:
    """Why an example with no reasoning needs none: no model made it. ``None`` when a model made it (its thinking is a
    gap) or the record does not say who did."""
    shape = str(row.get("schema_version") or "").split(".")[1:2]
    shape = shape[0] if shape else ""
    if shape == "decision_example":
        by = {str(a.get("by") or "").split(":")[0] for a in row.get("answers") or [] if isinstance(a, dict)}
        if by and by <= set(NO_MODEL_ANSWERERS):
            return f"no model answered ({', '.join(sorted(by))}): there was no thinking to keep"
        return None
    if shape == "judged_example":
        judge = str(row.get("judge") or "").split(":")[0]
        if judge in NO_MODEL_JUDGES or row.get("evidence") in ("measured", "test", "person"):
            return f"judged by {judge or row.get('evidence')}, not a model: there was no thinking to keep"
        return None
    if shape == "episode":
        return "the steps of code (a job, a walk, the stages): no model acted" if row.get("process") in NO_MODEL_PROCESSES else None
    if shape == "retrieval_example":
        return "a search by code: no model chose the passages"
    if shape == "routing_example" and not any(isinstance(r, dict) and r.get("by") == "model" for r in row.get("readings") or []):
        return "no model read this request (the rules, a script or a person): there was no thinking to keep"
    if shape == "experiment_record":
        return "a measured experiment: its numbers come from code, not a model"
    return None


def reasoning_counts(examples: Iterable[Any]) -> dict[str, int]:
    """How many answer turns (an assistant message, a trajectory's step, a pair's answer) carry reasoning, by its source:
    recorded, from_record, teacher, visible, replayed; ``lost`` (a note: lost before the rule), ``none`` (another stated
    reason), ``no_model`` (no model made it: :func:`why_no_reasoning`), ``missing`` (the field is there, with no reasoning and no note: a decision row or a step no source explains yet),
    and ``no_field`` for examples whose shape has no reasoning."""
    out: Counter = Counter()
    for ex in examples:
        turns = [m for m in getattr(ex, "messages", None) or [] if getattr(m, "role", None) == "assistant"]
        turns += [s for s in getattr(ex, "steps", None) or []]
        turns += [x for x in (getattr(ex, "chosen", None), getattr(ex, "rejected", None)) if x is not None]
        no_model = (lambda ex=ex: why_no_reasoning(ex.model_dump(mode="json")) if hasattr(ex, "model_dump") else None)  # only when asked
        if not turns and hasattr(ex, "reasoning_source"):  # a decision row: its reasoning, why none (v4), no model (nothing to keep), or missing
            said = str(getattr(ex, "no_reasoning", None) or "")
            out[getattr(ex, "reasoning_source", None) or ("lost" if said.startswith(LOST_PREFIX) else "none" if said else
                                                          "no_model" if no_model() else "missing")] += 1
            continue
        if not turns:
            out["no_field"] += 1
            continue
        if not any(hasattr(t, f) for t in turns for f in ("reasoning_source", "thought_source")):
            out["no_field"] += 1  # a shape with no reasoning (an episode's steps)
            continue
        for t in turns:
            source = getattr(t, "reasoning_source", None) or getattr(t, "thought_source", None)
            note = str(getattr(t, "no_reasoning", None) or getattr(t, "no_thought", None) or "")
            out[source or ("lost" if note.startswith(LOST_PREFIX) else "none" if note else "no_model" if no_model() else "missing")] += 1
    return dict(out)


def card(m: SetManifest, about: str) -> str:
    counts = ", ".join(f"{k} {v}" for k, v in m.counts.items())
    labels = ", ".join(f"{k} {v}" for k, v in sorted(m.label_sources.items())) or "none"
    return (f"# {m.model_type} / {m.name} ({m.version})\n\n{about}\n\n- Shape: `{m.shape}`\n- Rows: {counts}\n- Labels by source: {labels}\n"
            f"- Split: {m.split_rule}\n- Decontaminated against: {', '.join(m.decontaminated)}\n- Built: {m.built_at}\n\n"
            "The sealed split is written once and is never read to tune anything. A teacher's or a rule's label is silver: its weight is "
            "below a confirmed label's, and it is kept apart by its source.\n")


__all__ = ["write_set", "card", "SPLIT_RULE", "why_no_reasoning", "reasoning_counts"]
