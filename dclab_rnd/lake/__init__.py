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


def _origin_of(models: dict[str, Any]) -> str | None:
    """27.2: who wrote a session row's moves: Claude (evaluation only), DCLab's own rules (no model asked), or another model
    (None: left out until its terms are known)."""
    names = [str(m).lower() for m in (models or {}).values() if m]
    if any("claude" in m for m in names):
        return "teacher_claude"
    return None if names else "rule_prod"


def session_examples(other: dict[str, list[dict[str, Any]]], run: Run) -> list[L.TrainingExample]:
    """27.2: a workspace's session sets as training examples, each marked by who made its moves: conversations that ended in a
    build (``understand_problem``), each allowed agent step with the state it was taken in (``agent_step``), and each built
    project's problem to its workflow and pipeline (``solution_design``)."""
    from . import selfplay

    out: list[L.TrainingExample] = []

    def add(family: str, messages: list[dict[str, str]], item: dict[str, Any], origin: str | None, verification: L.Verification,
            group: str) -> None:
        if origin is None:
            run.drop(item["source"], "written by a model whose terms are not known")
            return
        split = "eval_only" if origin in L.EVAL_ONLY else ("sealed" if item.get("from_sealed") else item["row"].get("split") or L.split_of(group))
        out.append(L.TrainingExample(example_id="ss_" + L.fingerprint([family, messages])[:24], family=family, messages=messages,  # type: ignore[arg-type]
                                     verification=verification, quality=0.6,
                                     provenance=L.Provenance(source=item["source"], origin=origin, record_ids=[str(x[1]) for x in item["row"].get("sources") or []][:20],  # type: ignore[arg-type]
                                                             consent="train", group=group, split=split, from_sealed=bool(item.get("from_sealed")))))  # type: ignore[arg-type]

    for key, items in sorted(other.items()):
        builder = key.split("/", 1)[0]
        for item in items:
            r = item["row"]
            group = str(r.get("group") or item["source"])
            if builder == "conversations" and len(r.get("messages") or []) >= 2:
                add("understand_problem", list(r["messages"])[-60:], item, _origin_of(r.get("models") or {}),
                    L.Verification(kind="person", passed=True, detail="the person built the project the conversation understood"), group)
            elif builder == "trajectories":
                origin, seen = _origin_of(r.get("models") or {}), []
                for step in r.get("steps") or []:
                    if not step.get("state") or not step.get("tool"):
                        continue
                    if step.get("verdict") not in (None, "allowed", "ok"):
                        seen.append(step["tool"])
                        continue  # a refused move is not taught
                    asked = {"agent": r.get("agent"), "state": step["state"], "steps_before": seen[-8:]}
                    messages = [{"role": "system", "content": selfplay.system()},
                                {"role": "user", "content": json.dumps(asked, ensure_ascii=False, sort_keys=True, default=str)},
                                {"role": "assistant", "content": json.dumps({"tool": step["tool"]}, ensure_ascii=False)}]
                    add("agent_step", messages, item, origin, L.Verification(kind="none", passed=True, detail="the validator allowed the move"), group)
                    seen.append(step["tool"])
            elif builder == "workflows" and r.get("problem_card") and r.get("solution"):
                wf = r.get("workflow") or {}
                origin = "teacher_claude" if str(wf.get("source") or "") == "model" else "rule_prod"
                asked = {"question": "Lay out the solution workflow for this problem, step by step.", "problem_card": r["problem_card"],
                         "data_card": r.get("data_card")}
                answer = {"workflow": wf, "solution": {k: (r["solution"] or {}).get(k) for k in ("target", "task", "kind", "metric", "time_column",
                                                                                                "group_column", "identifiers")},
                          "forbidden": r.get("forbidden"), "pipeline": r.get("pipeline")}
                messages = [{"role": "system", "content": selfplay.system()},
                            {"role": "user", "content": json.dumps(asked, ensure_ascii=False, sort_keys=True, default=str)},
                            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, sort_keys=True, default=str)}]
                add("solution_design", messages, item, origin,
                    L.Verification(kind="executed", passed=bool(r.get("pipeline")), detail="the project ran its pipeline"), group)
            else:
                run.drop(item["source"], f"{builder}: no mapping yet")
    return out


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
    seen: set[tuple[str, str]] = set()
    out, why = [], Counter()
    for r in rows:
        k, split = key(r), r.provenance.split
        if k in held.get(split, ()):
            why[f"{split} row identical to an evaluation row"] += 1
            continue
        if (k, split) in seen:  # one copy per split, whatever its group: two seeds can make the very same problem in bands
            why["duplicate"] += 1
            continue
        seen.add((k, split))
        out.append(r)
    return out, why


def eval_seeds() -> set[int]:
    """27.5: the replicas' seeds an evaluation reads (benchmark v4's dev and sealed seeds, the journeys' dev and sealed
    seeds, the workflow models' sealed journeys): a self-play example of those tables is never a training row."""
    from ..agent_eval import benchmark_v4
    from ..agent_eval.journeys import GATES
    from ..agent_eval.journeys.programme import GATE as PROGRAMME
    from ..workflow_models import GATE as WORKFLOW

    return {0, *(s for v in benchmark_v4.REPLICA_SEEDS.values() for s in v), *(g["seed"] for g in GATES.values()), int(WORKFLOW["seed"]),
            int(PROGRAMME["seed"])}


def truth_problems(r: L.TrainingExample, text: str) -> list[str]:
    """27.9: a self-play trace must reach the replica's planted truth (its leak and identifier, its time column), the facts
    the replica was built with, not DCLab's answer."""
    if r.family not in ("leakage_judgment", "split_design"):
        return []
    from .benchmark import truth

    try:
        t = truth(r.model_dump(mode="json"))
    except Exception:  # noqa: BLE001 - a replica that cannot be rebuilt: no trace is kept without its truth
        return ["the replica's truth could not be read"]
    if r.family == "leakage_judgment":
        return [f"the planted column `{c}` is not in the trace" for c in [*t["leak"], *t["identifier"]] if f"`{c}`" not in text]
    if t["time_column"] and f"ordered in time by `{t['time_column']}`" not in text:
        return [f"the planted time column `{t['time_column']}` is not the trace's"]
    return []


def reason(rows: list[L.TrainingExample]) -> tuple[list[L.TrainingExample], Counter]:
    """27.9: each example's verified reasoning (``dclab_core.reasoning``) kept as its rationale, the `<think>` its export
    writes; a trace that is not grounded in the example's own numbers and columns is not kept (the example stays, with none);
    a self-play example whose answer misses the replica's planted leak, identifier or time column is dropped (the engine may
    have handled the trap its own way, but the answer as written would teach that nothing is there)."""
    from dclab_core import reasoning

    out, why = [], Counter()
    for r in rows:
        if r.rationale:
            out.append(r)
            continue
        steps, problems = reasoning.of_messages(r.family, r.messages)
        if steps and not problems and r.provenance.source == "selfplay":
            missed = truth_problems(r, "\n".join(steps))
            if missed:  # the answer itself misses what the replica planted: it would teach a wrong answer, so it goes
                why["answer misses the replica's planted truth (example dropped)"] += 1
                continue
        if steps and not problems:
            out.append(r.model_copy(update={"rationale": "\n".join(steps)[:4000]}))
            why["reasoning kept"] += 1
        else:
            out.append(r)
            why["reasoning not grounded" if steps else "no reasoning for this family yet"] += 1
    return out, why


def clean_decisions(rows: list[L.DecisionSample]) -> tuple[list[L.DecisionSample], Counter]:
    """27.5: a sample whose features are not a pure state, or whose text holds personal data, is dropped; then duplicates
    and evaluation rows (``decontaminate``)."""
    from dclab_core import clean

    kept, why = [], Counter()
    for r in rows:
        if clean.impure(r.features):
            why["features not a pure state (a value)"] += 1
        elif clean.personal([r.state_text, r.subject_ref]):
            why["personal data"] += 1
        else:
            kept.append(r)
    out, more = decontaminate(kept, lambda r: L.fingerprint([r.decision_point, r.features, r.label]))
    return out, why + more


def clean_examples(rows: list[L.TrainingExample], seeds: set[int]) -> tuple[list[L.TrainingExample], Counter]:
    """27.5: personal data dropped; a self-play example of a table an evaluation reads kept out of train; exact and near
    duplicates (and train rows nearly equal to an evaluation row) dropped; each kept example scored."""
    from dclab_core import clean

    kept, why = [], Counter()
    for r in rows:
        if clean.personal([m.get("content") for m in r.messages] + [r.rationale]):
            why["personal data"] += 1
            continue
        group = r.provenance.group.split(":")
        if r.provenance.split in ("train", "dev") and group[0] == "selfplay" and len(group) == 3 and group[2].isdigit() and int(group[2]) in seeds:
            why[f"{r.provenance.split} example of a table an evaluation reads"] += 1  # dev is the training run's validation file
            continue
        kept.append(r.model_copy(update={"quality": clean.quality(r.family, r.messages, r.verification.passed, reasoned=bool(r.rationale))}))
    out, more = clean.dedupe(kept, lambda r: clean.near_key(r.messages), lambda r: r.provenance.split)
    return out, why + more


CLAIMS = {"decision_sample": (["How a model reads these decision points on the sources below, split by whole table or project."],
                               ["Nothing about tables of another kind than its sources.",
                                "That a model trained on it is ready for production: real tables and blind labels decide that."]),
          "training_example": (["How a model answers these task families on the sources below."],
                               ["That a model trained on it solves problems end to end: 27.6's benchmark measures that, once."]),
          "eval_record": (["Comparison and evaluation only: what another answerer (Claude) said on the same steps."],
                          ["Nothing may be trained on it (the owner's decision of 2026-10-16)."])}
BY_ORIGIN = {  # 27.5: what a set may claim follows the origins its rows have, never a fixed text
    "construction": "Labels planted by construction in invented tables: exact for the planted kinds, silent on others.",
    "public_catalog": "Labels from public datasets' own marks (OpenML's row identifier, default target and column types): real tables, "
                      "marks as their uploaders set them.",
    "verified_synthetic": "Invented problems solved by DCLab's engine and kept only when every planted trap was caught: verified, not real.",
    "research_record": "Answers from the R&D's own research records, each cited.",
    "human_exposed": "A person's corrections after seeing DCLab's answer (exposed, not blind).",
    "human_blind": "Labels from people who did not see any model's answer.",
    "measured_outcome": "Labels measured from what happened afterwards (an outcome, a holdout comparison).",
    "teacher_claude": "Claude's answers: evaluation only, never a training target.",
    "dev_record": "DCLab's build history, written in Claude Code sessions: evaluation only.",
}


def write_set(folder: Path, kind: str, name: str, rows: list[Any], inputs: list[dict[str, str]], dropped: Counter) -> dict[str, Any]:
    from dclab_core import clean

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
    claims, limits = CLAIMS[kind]
    origins = sorted({r.provenance.origin for r in rows})
    claims = [*claims, *(BY_ORIGIN[o] for o in origins if o in BY_ORIGIN)]
    if "public_catalog" not in origins and kind != "eval_record":
        limits = [*limits, "Nothing about real tables: none of its rows comes from one."]
    licences = Counter(str(r.provenance.licence or "not recorded") for r in rows)
    c = clean.card(name, kind, json.loads(manifest.model_dump_json()), dropped, licences, claims, limits)
    (folder / "card.json").write_text(json.dumps(c, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (folder / "CARD.md").write_text(clean.card_markdown(c), encoding="utf-8")
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
    examples += session_examples(others, run)  # 27.2: the sessions recorded under train consent, by who made their moves
    examples, reasoned = reason(examples)
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
    attached: Counter = Counter()
    for d in decisions:
        by_point[d.decision_point].append(d)
    from .answers import attach

    for point, rows in sorted(by_point.items()):
        kept, why = clean_decisions(rows)
        kept, answered = attach(kept)  # 26.9: every other answerer's captured answer beside its decision, never the label
        attached.update(answered)
        sources = {r.provenance.source for r in rows}
        sets.append(write_set(lake / "gold" / "decisions" / point / VERSION, "decision_sample", f"decisions/{point}", kept,
                              [i for i in run.inputs if point in i["path"] or "datasets" in i["path"] or "openml" in i["path"]],
                              _merged(run, sources, why)))
    by_family: dict[str, list[L.TrainingExample]] = defaultdict(list)
    for e in examples:
        by_family[e.family].append(e)
    seeds = eval_seeds()
    for family, rows in sorted(by_family.items()):
        kept, why = clean_examples(rows, seeds)
        sets.append(write_set(lake / "gold" / "solver" / family / VERSION, "training_example", f"solver/{family}", kept,
                              [i for i in run.inputs if "sft" in i["path"] or "selfplay" in i["path"]], _merged(run, {r.provenance.source for r in rows}, why)))
    for name, rows in sorted(teacher.items()):
        kept, why = decontaminate(rows, lambda r: L.fingerprint(r.messages))
        sets.append(write_set(lake / "gold" / "eval" / name / VERSION, "eval_record", f"eval/{name}", kept,
                              [i for i in run.inputs if "twin" in i["path"]], why))
    summary = {"name": "DCLab training lake", "version": VERSION, "builder_commit": commit(),
               "workspaces_read": len(homes) - len(run.skipped), "workspaces_skipped": [s["why"] for s in run.skipped], "inputs": len(run.inputs),
               "silver_only": {k: len(v) for k, v in sorted(others.items())},
               "reasoning": dict(reasoned),
               "other_answers": dict(attached),  # 26.9: captured answers attached to their decisions (lake/bronze/answers)  # 27.9: the examples that carry a verified trace (their `<think>`), and those without
               "dropped_by_source": {k: dict(v) for k, v in sorted(run.dropped.items()) if v},
               "sets": [{"name": s["name"], "written": s["written"], **({"problems": s["problems"]} if not s["written"] else {}),
                         **{k: s["manifest"][k] for k in ("rows", "by_split", "by_origin", "by_family", "groups", "dropped")}} for s in sets],
               "rule": "Counts only: the rows are in lake/ (not committed). Claude's answers and DCLab's build history are evaluation only "
                       "(the owner, 2026-10-16). Counts depend on the workspaces whose train consent is on."}
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
