"""D2 / P5 (2026-10-23): a set of the store as chat JSONL for tool-calling SFT, in the shape a model's own chat template reads.

    python -m dclab_rnd.store.export_chat --model-type m5_solver_llm --folder trajectories/reasoning --out DIR [--version v1]

One line per example: ``tools`` (the schemas the model was offered, as sent), ``messages`` with ``reasoning_content`` on every
assistant turn whose reasoning was recorded (never a visible, a rule's or a teacher's reasoning: the model is taught to think
only on real thinking), tool calls in the wire shape with their arguments as objects (what Qwen's template renders), and
``tool_call_id`` on tool turns. A trajectory becomes the same shape: the problem as the user's turn, each step an assistant
turn with its call and a tool turn with the whole result. ``train.jsonl`` and ``dev.jsonl`` as the set has them; a sealed
row is never exported. A set whose manifest's consent is not ``train`` (silver_frontier) is refused. The export's
MANIFEST.json carries the set's manifest, the commit and the counts. The training run itself is the owner's
(decisions/finetune/README.md); ``template_check`` checks the export against the model's template first.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dclab_core.store import ChatExample, SolverTrajectory

from .layout import Layout

SPLITS = ("train", "dev")  # a sealed row is never exported


class NotForTraining(ValueError):
    """The set's manifest does not allow training on it."""


def wire_call(call: dict[str, Any], n: int) -> dict[str, Any]:
    """A stored tool call ({"id", "name", "arguments": dict}, or the wire shape with a string) as the wire shape with its
    arguments as an object; arguments that are not valid JSON stay under ``_raw``, never silently dropped."""
    function = call.get("function") if isinstance(call.get("function"), dict) else None
    name = (function or call).get("name")
    arguments = (function or call).get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            arguments = {"_raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {} if arguments is None else {"_raw": arguments}
    return {"id": str(call.get("id") or f"call_{n}"), "type": "function", "function": {"name": str(name or ""), "arguments": arguments}}


def chat_line(ex: ChatExample, final_only: bool = False) -> dict[str, Any]:
    """``final_only`` (the final-turn sets): every assistant turn but the last carries ``weight: 0`` (context the model reads,
    never a target), the last ``weight: 1``; a trainer that reads per-turn weights then learns only the turn whose thinking
    was recorded with it."""
    messages: list[dict[str, Any]] = []
    last = max((i for i, m in enumerate(ex.messages) if m.role == "assistant"), default=-1)
    pending: list[str] = []  # the ids of the last assistant turn's calls, for tool turns recorded without their call id
    for n, m in enumerate(ex.messages):
        msg: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.role == "assistant":
            if m.reasoning and m.reasoning_source == "recorded":
                msg["reasoning_content"] = m.reasoning
            if m.tool_calls:
                msg["tool_calls"] = [wire_call(c, i) for i, c in enumerate(m.tool_calls) if isinstance(c, dict)]
                pending = [c["id"] for c in msg["tool_calls"]]
            if final_only:
                msg["weight"] = 1 if n == last else 0
        elif m.role == "tool":
            msg["tool_call_id"] = m.tool_call_id or (pending.pop(0) if pending else f"call_{n}")
        if m.name:
            msg["name"] = m.name
        messages.append(msg)
    return {"id": ex.id, "split": ex.split, "weight": ex.weight, "tools": list(ex.tools), "messages": messages}


def trajectory_line(t: SolverTrajectory) -> dict[str, Any]:
    problem = {"problem_kind": t.problem_kind, **{k: v for k, v in t.problem.items() if k in ("request", "outcome_named")}}
    messages: list[dict[str, Any]] = [{"role": "user", "content": json.dumps(problem, ensure_ascii=False, default=str)}]
    for s in t.steps:
        turn: dict[str, Any] = {"role": "assistant", "content": ""}
        if s.thought and s.thought_source == "recorded":
            turn["reasoning_content"] = s.thought
        if s.tool:
            call_id = f"call_{s.n}"
            turn["tool_calls"] = [{"id": call_id, "type": "function", "function": {"name": s.tool, "arguments": dict(s.arguments or {})}}]
            messages.append(turn)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(s.result, ensure_ascii=False, default=str)})
        else:
            turn["content"] = s.thought if s.thought and s.thought_source != "recorded" else ""
            messages.append(turn)
    return {"id": t.id, "split": t.split, "weight": t.weight, "tools": list(t.tools), "messages": messages}


def lines_of(rows: Iterable[dict[str, Any]], final_only: bool = False) -> Iterable[tuple[dict[str, Any] | None, str]]:
    """Each stored example as an export line, or None with why it is skipped (a preference pair is not an SFT line)."""
    for row in rows:
        shape = str(row.get("schema_version") or "").split(".")[-2] if row.get("schema_version") else ""
        if shape == "chat_example":
            yield chat_line(ChatExample.model_validate(row), final_only), ""
        elif shape == "solver_trajectory":
            yield trajectory_line(SolverTrajectory.model_validate(row)), ""
        else:
            yield None, f"not an SFT line: {shape or 'unknown shape'}"


def _commit() -> str | None:
    """This repository's commit, so an export says which code shaped it; None outside its checkout (a package installed inside
    another repository must not report that repository's commit)."""
    import subprocess

    root = Path(__file__).resolve().parents[2]
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=10, cwd=root)
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root:
            return None
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, cwd=root)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def export(layout: Layout, model_type: str, folder: str, version: str, out: Path) -> dict[str, Any]:
    base = layout.sets(model_type, *folder.split("/"), version)
    manifest_path = base / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no set at {base}: make capture builds it")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("consent") != "train":
        raise NotForTraining(f"{model_type}/{folder} {version} is not for training: its manifest's consent is {manifest.get('consent')!r}")
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    skipped: dict[str, int] = {}
    mismatch: dict[str, dict[str, int]] = {}  # the set's files against its manifest: said, never hidden
    calls_without_tools, longest = 0, 0
    for split in SPLITS:
        source = base / f"{split}.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()] if source.is_file() else []
        said = int((manifest.get("counts") or {}).get(split) or 0)
        if said != len(rows):
            mismatch[split] = {"manifest": said, "read": len(rows)}
        kept = []
        for row in rows:
            if row.get("split") not in SPLITS:  # a hand-edited or older set: the rule holds whatever the file says
                skipped["a row whose split is not train or dev"] = skipped.get("a row whose split is not train or dev", 0) + 1
                continue
            line, why = next(lines_of([row], folder.startswith("reasoning/final_turn")))
            if line is None:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            if not line["tools"] and any(m.get("tool_calls") for m in line["messages"]):
                calls_without_tools += 1  # the model would be taught to call tools it was never shown
            text = json.dumps(line, ensure_ascii=False, default=str)
            longest = max(longest, len(text))
            kept.append(text)
        (out / f"{split}.jsonl").write_text("".join(x + "\n" for x in kept), encoding="utf-8")
        counts[split] = len(kept)
    doc = {"set": manifest, "export": {"model_type": model_type, "folder": folder, "version": version, "source": str(base),
                                      "commit": _commit(), "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                      "counts": counts, "skipped": skipped, "mismatch": mismatch, "sealed": "never exported",
                                      "lines_with_calls_but_no_tools": calls_without_tools,  # before a run: a set that kept no schemas
                                      "longest_line_chars": longest,  # against a trainer's max_seq_length, which cuts silently
                                      "shape": "tools + messages; reasoning_content on assistant turns with recorded reasoning; arguments as objects"}}
    (out / "MANIFEST.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc["export"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--folder", required=True, help="the set's folder under the model type, e.g. trajectories/reasoning or chat/reasoning")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--store", type=Path, default=None, help="the store's root (default: training_records/store)")
    args = parser.parse_args(argv)
    layout = Layout(args.store) if args.store else Layout()
    try:
        report = export(layout, args.model_type, args.folder, args.version, args.out)
    except (NotForTraining, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
