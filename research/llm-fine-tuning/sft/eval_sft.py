#!/usr/bin/env python3
"""Deterministic evaluation of any model on the DCLab v3 validation set.

Most v3 tasks have a checkable core, so a model can be scored without an LLM judge:

* ``apply_selection_rule`` - the answer names the option the rule selects.
* ``leakage_judgment``     - the answer says exclude / do-not-auto-delete / proceed, matching the label.
* ``grounded_qa``          - the answer cites the correct record ID and no distractor ID.
* ``explain_experiment``   - the answer has all five sections (Evidence, Interpretation, Decision, Risks, Next test).
* ``critique_claim``       - the answer raises at least one gap and assigns a severity.
* other tasks              - format check only (non-empty answer).

Usage::

    # 1) Score predictions produced by any system (one JSON object per line: {"id": ..., "prediction": ...})
    python research/llm-fine-tuning/sft/eval_sft.py score --predictions preds.jsonl

    # 2) Sanity check: score the reference answers themselves (should be ~100%)
    python research/llm-fine-tuning/sft/eval_sft.py score --reference

    # 3) Generate with a local Hugging Face model (+ optional LoRA adapter), then score
    python research/llm-fine-tuning/sft/eval_sft.py generate --model Qwen/Qwen2.5-1.5B-Instruct [--adapter sft/runs/dclab-lora] --out preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]  # research/llm-fine-tuning/sft -> repo root
SFT_DIR = Path(__file__).resolve().parent
VAL = SFT_DIR / "out_v3" / "val.chat.jsonl"
SECTIONS = ("Evidence", "Interpretation", "Decision", "Risks", "Next test")


def load_val(path: Path = VAL) -> list[dict[str, Any]]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            row = json.loads(line)
            row["id"] = i
            rows.append(row)
    return rows


def _ids_in(text: str) -> set[str]:
    return set(re.findall(r"\[([A-Z]+-[A-Za-z0-9_\-]+)\]", text))


def score_one(row: dict[str, Any], prediction: str) -> tuple[bool, str]:
    meta = row["metadata"]
    task = meta["task"]
    reference = row["messages"][2]["content"]
    pred = prediction or ""
    low = pred.lower()
    if not pred.strip():
        return False, "empty"
    if task == "apply_selection_rule":
        chosen = re.search(r"\*\*([^*]+)\*\*", reference).group(1)
        others = re.findall(r"selects \*\*([^*]+)\*\*", pred)
        ok = (others and others[0].strip() == chosen) or (not others and re.search(rf"\b{re.escape(chosen)}\b", pred) is not None)
        return bool(ok), f"expected {chosen}"
    if task == "leakage_judgment":
        label = meta.get("label")
        if label == "exclude":
            return bool(re.search(r"\b(exclude|drop|remove|do not use|must not)\b", low)), "expected exclude"
        if label == "review":
            return bool(re.search(r"(not auto-?delete|do not (auto-?)?delete|review|confirm)", low)) and "exclude `" not in low, "expected review"
        return bool(re.search(r"(proceed|no column|none)", low)), "expected proceed"
    if task == "grounded_qa":
        target = meta.get("record_id")
        user_ids = _ids_in(row["messages"][1]["content"])
        cited = _ids_in(pred)
        ok = target in cited and not (cited & (user_ids - {target}))
        return ok, f"expected citation [{target}] only"
    if task == "explain_experiment":
        missing = [s for s in SECTIONS if s.lower() not in low]
        return not missing, ("missing " + ", ".join(missing)) if missing else "ok"
    if task == "critique_claim":
        return bool(re.search(r"\b(high|medium|low)\b", low)), "expected severity labels"
    return True, "format only"


def score(rows: list[dict[str, Any]], predictions: dict[int, str]) -> dict[str, Any]:
    by_task: dict[str, list[bool]] = defaultdict(list)
    failures = []
    for row in rows:
        ok, why = score_one(row, predictions.get(row["id"], ""))
        by_task[row["metadata"]["task"]].append(ok)
        if not ok:
            failures.append({"id": row["id"], "task": row["metadata"]["task"], "why": why})
    tasks = {t: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for t, v in sorted(by_task.items())}
    overall = sum(sum(v) for v in by_task.values()) / max(sum(len(v) for v in by_task.values()), 1)
    return {"overall": round(overall, 4), "by_task": tasks, "failures": failures[:25]}


def generate(rows: list[dict[str, Any]], model: str, adapter: str | None, max_new_tokens: int) -> dict[int, str]:  # pragma: no cover
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="auto")
    if adapter:
        from peft import PeftModel

        lm = PeftModel.from_pretrained(lm, adapter)
    out = {}
    for row in rows:
        prompt = tok.apply_chat_template(row["messages"][:2], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(lm.device)
        with torch.no_grad():
            gen = lm.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        out[row["id"]] = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--predictions", type=Path)
    s.add_argument("--reference", action="store_true", help="score the reference answers (sanity check)")
    s.add_argument("--val", type=Path, default=VAL)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--adapter")
    g.add_argument("--out", type=Path, default=Path("preds.jsonl"))
    g.add_argument("--max-new-tokens", type=int, default=700)
    g.add_argument("--val", type=Path, default=VAL)
    args = parser.parse_args(argv)
    rows = load_val(args.val)
    if args.command == "generate":
        preds = generate(rows, args.model, args.adapter, args.max_new_tokens)
        args.out.write_text("".join(json.dumps({"id": k, "prediction": v}) + "\n" for k, v in preds.items()))
    elif args.reference:
        preds = {row["id"]: row["messages"][2]["content"] for row in rows}
    elif args.predictions:
        preds = {}
        for line in args.predictions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                preds[int(item["id"])] = item.get("prediction", "")
    else:
        parser.error("score needs --predictions or --reference")
    print(json.dumps(score(rows, preds), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
