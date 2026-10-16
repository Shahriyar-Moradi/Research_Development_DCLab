"""The training lake (26.1): every source DCLab's own models learn from, in three layers, through one command.

``make lake`` (``python -m dclab_rnd.lake build``) reads:

- the generated decision benchmarks (``decisions/generated_v1``: invented tables, labels true by construction, their
  sealed partitions kept sealed);
- OpenML's public tables (26.2, ``data/downloads/openml``, fetched by ``python -m dclab_rnd.lake openml``);
- verified synthetic solving (27.3, ``lake/bronze/selfplay``, by ``python -m dclab_rnd.lake selfplay``);
- the R&D's evidence-grounded SFT corpus (v3) and DCLab's build history (evaluation only: written in Claude Code sessions);
- each workspace's sets **as ``make datasets`` wrote them under the workspace's train consent** (its gates: the workspace's
  switch, each project's and draft's own consent, a project gone leaves nothing). The lake reads a workspace only while its
  train switch is on and its sets are not marked stale. Claude's answers (the twin sets of 25.7) arrive this way and go to
  ``gold/eval/``: evaluation only (the owner, 2026-10-16), never a training target.

and writes ``lake/`` (not committed: it holds rows), rebuilt whole each time so nothing withdrawn lingers:

- ``bronze/INPUTS.json``: every file read, with its SHA-256;
- ``silver/<source>.jsonl``: each source's rows as read (a workspace set's rows wrapped with its provenance);
- ``gold/decisions/<point>/v1/``, ``gold/solver/<family>/v1/`` (training shapes) and ``gold/eval/<set>/v1/`` (evaluation
  only), each split in its own file with its ``manifest.json`` (``GoldManifest``): deduplicated, decontaminated (a training
  row identical to an evaluation row is dropped), every group in one split, checked by ``gold_problems``.

``research/llm-fine-tuning/experiments/lake/MANIFEST.json`` (committed) holds the gold manifests' counts, origins and drops,
never a row nor a path of this machine.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from dclab_core import lake as L

ROOT = Path(__file__).resolve().parents[2]
LAKE = ROOT / "lake"
MANIFEST = ROOT / "research" / "llm-fine-tuning" / "experiments" / "lake" / "MANIFEST.json"
GENERATED = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "generated_v1" / "decisions"
SFT_V3 = ROOT / "research" / "llm-fine-tuning" / "experiments" / "sft" / "out_v3"
SFT_DEV = ROOT / "research" / "llm-fine-tuning" / "experiments" / "sft" / "out_dev"
VERSION = "v1"

# a decision row's label source, as the ledger names it, to the lake's origin
LABEL_ORIGIN = {"benchmark": "construction", "measured": "measured_outcome", "user_decision_exposed": "human_exposed", "blind": "human_blind"}
# the ledger's subject kinds to the sample's
SUBJECT = {"column": "column", "dataset": "dataset", "project": "problem", "draft": "problem", "conversation": "problem",
           "workflow": "step", "proposal": "step", "step": "step", "problem": "problem"}
DECISION_BUILDERS = ("decisions", "outcomes")  # workspace sets in the ledger's decision shape
EVAL_BUILDERS = ("twin",)  # Claude's answers beside DCLab: evaluation only


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def _rel(path: Path) -> str:
    """A path as the committed manifest may show it: under the repository, or the workspace-relative tail."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return "/".join(path.parts[-4:])


class Run:
    """One build: the inputs read and, per source, what was dropped and why."""

    def __init__(self) -> None:
        self.inputs: list[dict[str, str]] = []
        self.dropped: dict[str, Counter] = defaultdict(Counter)
        self.skipped: list[dict[str, str]] = []

    def read(self, path: Path) -> None:
        self.inputs.append({"path": _rel(path), "sha256": sha256(path)})

    def drop(self, source: str, why: str, n: int = 1) -> None:
        self.dropped[source][why] += n


# ---------------------------------------------------------------------------------------------------- decisions

def _state_text(point: str, state: dict[str, Any], options: list[Any]) -> str:
    from dclab_core.decision_export import request

    try:
        return json.dumps(request(point, state, options), ensure_ascii=False, sort_keys=True)
    except KeyError:  # a point the export does not know yet: its state and options as they are
        return json.dumps({"point": point, "state": state, "options": [str(o) for o in options]}, ensure_ascii=False, sort_keys=True)


def decision_samples(rows: Iterable[dict[str, Any]], source: str, consent: str, sealed: bool, run: Run) -> list[L.DecisionSample]:
    """Ledger-shaped decision rows (the generated sets, a workspace's decision and outcome sets) as decision samples. A row
    with no label, or a label only a model gave, is dropped and counted under its source."""
    out = []
    for r in rows:
        label = r.get("label") or {}
        origin = LABEL_ORIGIN.get(str(label.get("source")))
        if label.get("value") is None or origin is None:
            run.drop(source, "no usable label")
            continue
        group = str(r.get("group") or f"record:{r.get('record')}")
        split = "sealed" if sealed else ("dev" if r.get("split") in ("dev", "val") else "train" if r.get("split") == "train" else L.split_of(group))
        if split == "sealed" and not sealed:  # only a source's own sealed partition is sealed: its other rows never are
            split = "dev"
        point, state, options = str(r.get("point")), dict(r.get("state") or {}), list(r.get("options") or [])
        rule = next((a.get("value") for a in r.get("answers") or [] if a.get("by") == "rule"), None)
        subject = r.get("subject") or {}
        out.append(L.DecisionSample(
            sample_id=str(r.get("record")), decision_point=point, subject_kind=SUBJECT.get(str(subject.get("kind")), "column"),  # type: ignore[arg-type]
            subject_ref=str(subject.get("key") or state.get("column") or "")[:200], features=state, state_text=_state_text(point, state, options),
            options=[str(o) for o in options], label=label["value"], rule_answer=rule,
            provenance=L.Provenance(source=source, origin=origin, record_ids=[str(x[1]) for x in r.get("sources") or []][:20], consent=consent,  # type: ignore[arg-type]
                                    group=group, split=split, from_sealed=sealed)))  # type: ignore[arg-type]
    return out


def generated_decisions(run: Run) -> list[L.DecisionSample]:
    out: list[L.DecisionSample] = []
    for folder in sorted(GENERATED.glob("*")) if GENERATED.is_dir() else []:
        for name, sealed in (("rows.jsonl", False), ("sealed.jsonl", True)):
            path = folder / name
            if path.is_file():
                run.read(path)
                out += decision_samples(read_jsonl(path), "decisions/generated_v1", "ours", sealed, run)
    return out


# ---------------------------------------------------------------------------------------------------- workspaces

def workspace_sets(home: Path, run: Run) -> tuple[list[L.DecisionSample], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """What a workspace's consent allows today: its decision samples, its evaluation records and its other sets' rows (silver
    only until their mapping, 27.2). Nothing while its train switch is off or its sets wait for a rebuild."""
    from .. import consent
    from ..datasets import FOLDER, STALE

    root = home / FOLDER
    if not root.is_dir() or not any(p.is_dir() and p.parent.name != "development" for p in root.glob("*/*")):
        run.skipped.append({"workspace": home.name, "why": "no sets built (make datasets writes what its consent allows)"})
        return [], [], {}
    if not consent.workspace_consent(home).get("train"):
        run.skipped.append({"workspace": home.name, "why": "its train consent is off"})
        return [], [], {}
    if (root / STALE).exists():
        run.skipped.append({"workspace": home.name, "why": "its sets are marked stale: run make datasets first"})
        return [], [], {}
    decisions: list[L.DecisionSample] = []
    evaluation: list[dict[str, Any]] = []
    other: dict[str, list[dict[str, Any]]] = {}
    for folder in sorted(p for p in root.glob("*/*") if p.is_dir()):
        builder, name = folder.parent.name, folder.name
        source = f"workspace:{home.name}/{builder}/{name}"
        try:
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            run.drop(source, "a set without its manifest")
            continue
        if manifest.get("use") != "train":  # consent to improve DCLab is not consent to train its models
            run.drop(source, f"built for {manifest.get('use')!r}, not train")
            continue
        for file, sealed in (("rows.jsonl", False), ("sealed.jsonl", True)):
            path = folder / file
            if not path.is_file():
                continue
            run.read(path)
            rows = list(read_jsonl(path))
            if builder in DECISION_BUILDERS:
                decisions += decision_samples(rows, source, "train", sealed, run)
            elif builder in EVAL_BUILDERS:
                evaluation += [{"set": name, "row": r, "group": str(r.get("group") or source), "from_sealed": sealed, "source": source} for r in rows]
            else:
                other.setdefault(f"{builder}/{name}", []).extend({"row": r, "from_sealed": sealed, "source": source} for r in rows)
    return decisions, evaluation, other


def eval_records(items: list[dict[str, Any]]) -> dict[str, list[L.TrainingExample]]:
    """The twin sets' rows (Claude beside DCLab) as evaluation-only records, one set per twin set: the row is kept whole as
    the assistant turn of a two-turn record, so the gold rules and the manifest apply to it like any other."""
    out: dict[str, list[L.TrainingExample]] = defaultdict(list)
    for i, item in enumerate(items):
        row = item["row"]
        msgs = [{"role": "user", "content": json.dumps({k: row.get(k) for k in ("point", "step", "field", "subject", "state", "input", "options") if row.get(k) is not None},
                                                         ensure_ascii=False, default=str)[:12000]},
                {"role": "assistant", "content": json.dumps(row, ensure_ascii=False, default=str)[:20000]}]
        out[item["set"]].append(L.TrainingExample(
            example_id="tw_" + L.fingerprint([item["set"], row])[:24], family="solution_design" if item["set"] == "twin.expert" else "typed_decision",
            messages=msgs, verification=L.Verification(kind="none", passed=False, detail="Claude beside DCLab: compared against, not verified"),
            provenance=L.Provenance(source=item["source"], origin="teacher_claude", record_ids=[str(s[1]) for s in row.get("sources") or []][:20],  # type: ignore[arg-type]
                                    consent="train", group=item["group"], split="eval_only")))
    return out


# ---------------------------------------------------------------------------------------------------- corpora

def chat_corpus(folder: Path, family: str, origin: str, source: str, run: Run, verification: L.Verification) -> list[L.TrainingExample]:
    """An SFT corpus already split by its builder (whole datasets or packages held out): train stays train, val is dev; an
    evaluation-only origin goes to eval_only whatever the file."""
    out = []
    for name, split in (("train.chat.jsonl", "train"), ("val.chat.jsonl", "dev")):
        path = folder / name
        if not path.is_file():
            continue
        run.read(path)
        for row in read_jsonl(path):
            messages = [{"role": str(m.get("role")), "content": str(m.get("content") or "")} for m in row.get("messages") or [] if m.get("content")]
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                run.drop(source, "a conversation that does not end in its answer")
                continue
            meta = row.get("metadata") or {}
            group = f"{source}:{meta.get('dataset') or meta.get('package') or meta.get('group') or L.fingerprint(messages[:2])[:16]}"
            out.append(L.TrainingExample(example_id="ex_" + L.fingerprint([family, messages])[:24], family=family, messages=messages,  # type: ignore[arg-type]
                                         verification=verification,
                                         provenance=L.Provenance(source=source, origin=origin, consent="ours", group=group,  # type: ignore[arg-type]
                                                                 split="eval_only" if origin in L.EVAL_ONLY else split)))  # type: ignore[arg-type]
    return out


# ---------------------------------------------------------------------------------------------------- gold

def decontaminate(rows: list[Any], key: Callable[[Any], str]) -> tuple[list[Any], Counter]:
    """Exact duplicates removed; a training row identical (by ``key``, without its group) to a dev or sealed row dropped, and
    a dev row identical to a sealed row too: nothing of an evaluation set is learned or tuned on."""
    by_split: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_split[r.provenance.split].add(key(r))
    held = {"train": by_split["dev"] | by_split["sealed"], "dev": by_split["sealed"]}
    seen: set[str] = set()
    out, why = [], Counter()
    for r in rows:
        k, split = key(r), r.provenance.split
        if k in held.get(split, ()):
            why[f"{split} row identical to an evaluation row"] += 1
            continue
        if (k, split, r.provenance.group) in seen:
            why["duplicate"] += 1
            continue
        seen.add((k, split, r.provenance.group))
        out.append(r)
    return out, why


def write_set(folder: Path, kind: str, name: str, rows: list[Any], inputs: list[dict[str, str]], dropped: Counter) -> dict[str, Any]:
    problems = L.gold_problems(rows)
    manifest = L.manifest_of(name, VERSION, kind, rows, inputs, dropped, now(), commit())
    if problems:  # nothing is written: the previous build's files were removed with the whole gold folder
        return {"name": name, "written": False, "problems": problems[:10], "manifest": json.loads(manifest.model_dump_json())}
    folder.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[Any]] = defaultdict(list)
    for r in rows:
        by_split[r.provenance.split].append(r)
    for split, items in by_split.items():
        (folder / f"{split}.jsonl").write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")
    (folder / "manifest.json").write_text(manifest.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return {"name": name, "written": True, "manifest": json.loads(manifest.model_dump_json())}


def workspaces(extra: Iterable[Path] = ()) -> list[Path]:
    """The workspaces the lake reads: the development one, every kept live test, every collected tester, and any given."""
    homes = [ROOT / "agent_runs"]
    homes += sorted((ROOT / "agent_runs" / "live_tests").glob("*/workspace"))
    homes += sorted(p for p in (ROOT / "agent_runs" / "testers").glob("*") if p.is_dir()) if (ROOT / "agent_runs" / "testers").is_dir() else []
    homes += [Path(p) for p in extra]
    seen, out = set(), []
    for h in homes:
        r = h.resolve()
        if r.is_dir() and r not in seen:
            seen.add(r)
            out.append(h)
    return out


def _merged(run: Run, sources: Iterable[str], extra: Counter | None = None) -> Counter:
    total = Counter()
    for s in set(sources):
        total.update(run.dropped.get(s, Counter()))
    total.update(extra or Counter())
    return total


def build(lake: Path = LAKE, homes: Iterable[Path] | None = None, manifest_path: Path | None = MANIFEST) -> dict[str, Any]:
    """Bronze, silver and gold from every source, the lake rebuilt whole; returns (and writes, when a path is given) the
    committed manifest."""
    from . import openml, selfplay

    homes = list(homes) if homes is not None else workspaces()
    run = Run()
    for layer in ("silver", "gold"):  # rebuilt whole: a withdrawn row or a refused set never lingers
        shutil.rmtree(lake / layer, ignore_errors=True)
    decisions = generated_decisions(run)
    public = openml.samples(run.dropped["openml"])
    decisions += public
    evaluation: list[dict[str, Any]] = []
    others: dict[str, list[dict[str, Any]]] = {}
    for home in homes:
        d, e, o = workspace_sets(home, run)
        decisions += d
        evaluation += e
        for k, v in o.items():
            others.setdefault(k, []).extend(v)
    verified = L.Verification(kind="label_match", passed=True, detail="the builder's own checked corpus")
    examples = chat_corpus(SFT_V3, "evidence_answer", "research_record", "sft/out_v3", run, verified)
    examples += chat_corpus(SFT_DEV, "engineering", "dev_record", "sft/out_dev", run, verified)
    examples += selfplay.examples()
    teacher = eval_records(evaluation)
    # silver: each source's rows as read, with provenance
    silver = lake / "silver"
    silver.mkdir(parents=True, exist_ok=True)
    by_source: dict[str, list[str]] = defaultdict(list)
    for r in [*decisions, *examples, *(x for v in teacher.values() for x in v)]:
        by_source[r.provenance.source].append(r.model_dump_json())
    for name, rows in others.items():
        by_source[f"workspace_sets/{name}"] += [json.dumps(r, ensure_ascii=False, default=str) for r in rows]
    for source, items in by_source.items():
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in source)[:120]
        (silver / f"{safe}.jsonl").write_text("".join(i + "\n" for i in items), encoding="utf-8")
    (lake / "bronze").mkdir(parents=True, exist_ok=True)
    (lake / "bronze" / "INPUTS.json").write_text(json.dumps({"built_at": now(), "inputs": run.inputs, "skipped": run.skipped}, indent=1) + "\n", encoding="utf-8")
    sets = []
    by_point: dict[str, list[L.DecisionSample]] = defaultdict(list)
    for d in decisions:
        by_point[d.decision_point].append(d)
    for point, rows in sorted(by_point.items()):
        kept, why = decontaminate(rows, lambda r: L.fingerprint([r.decision_point, r.features, r.label]))
        sources = {r.provenance.source for r in rows}
        sets.append(write_set(lake / "gold" / "decisions" / point / VERSION, "decision_sample", f"decisions/{point}", kept,
                              [i for i in run.inputs if point in i["path"] or "datasets" in i["path"] or "openml" in i["path"]],
                              _merged(run, sources, why)))
    by_family: dict[str, list[L.TrainingExample]] = defaultdict(list)
    for e in examples:
        by_family[e.family].append(e)
    for family, rows in sorted(by_family.items()):
        kept, why = decontaminate(rows, lambda r: L.fingerprint(r.messages))
        sets.append(write_set(lake / "gold" / "solver" / family / VERSION, "training_example", f"solver/{family}", kept,
                              [i for i in run.inputs if "sft" in i["path"] or "selfplay" in i["path"]], _merged(run, {r.provenance.source for r in rows}, why)))
    for name, rows in sorted(teacher.items()):
        kept, why = decontaminate(rows, lambda r: L.fingerprint(r.messages))
        sets.append(write_set(lake / "gold" / "eval" / name / VERSION, "eval_record", f"eval/{name}", kept,
                              [i for i in run.inputs if "twin" in i["path"]], why))
    summary = {"name": "DCLab training lake", "version": VERSION, "builder_commit": commit(),
               "workspaces_read": len(homes) - len(run.skipped), "workspaces_skipped": [s["why"] for s in run.skipped], "inputs": len(run.inputs),
               "silver_only": {k: len(v) for k, v in sorted(others.items())},
               "dropped_by_source": {k: dict(v) for k, v in sorted(run.dropped.items()) if v},
               "sets": [{"name": s["name"], "written": s["written"], **({"problems": s["problems"]} if not s["written"] else {}),
                         **{k: s["manifest"][k] for k in ("rows", "by_split", "by_origin", "by_family", "groups", "dropped")}} for s in sets],
               "rule": "Counts only: the rows are in lake/ (not committed). Claude's answers and DCLab's build history are evaluation only "
                       "(the owner, 2026-10-16). Counts depend on the workspaces whose train consent is on."}
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
