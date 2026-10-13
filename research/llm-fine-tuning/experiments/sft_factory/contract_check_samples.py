"""
contract_check_samples.py
=========================

SFT samples for the link between problem understanding and data understanding: before any analysis, check the
REAL table against the problem contract.

    problem contract + data profile ──► contract_check ──► checks, decision, contract changes, questions

One mismatch is PLANTED per sample (or none), so the right answer is known by construction:

    target_renamed     the contract's target column is not in the table (the data names it differently)
    task_wrong         the contract's inferred task does not fit the target's values
    labels_missing     some target values are blank, while the contract says labels are complete
    target_constant    the target never varies in the data
    time_missing       the contract's inferred time column is not in the table
    entity_missing     the column the person said rows repeat for is not in the table
    clean              nothing wrong

The checks are COMPUTED on the table (presence, distinct values, blanks, dtypes); the checker keeps a sample only if
exactly the planted check fails. Decisions follow fixed rules, written in the stage card:

* a stated fact the data contradicts, or a target the data cannot answer: ask_owner (never overwrite the person);
* an inferred fact the data contradicts: fix_contract (correct the inference, say why);
* nothing contradicted: proceed.

Usage
-----
    python research/llm-fine-tuning/experiments/sft_factory/contract_check_samples.py
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import problem_to_plan_samples as PP
from problem_to_plan_samples import (HERE, J, SCENARIOS, SPLIT_OF, SYSTEM_PROMPT, CoverageFact, TaskFact, TextFact,
                                     NamesFact, build_case, compact, numbers_ok, table_for)
from dclab_core.contracts import ProblemBrief

PLANTS = ["clean", "target_renamed", "task_wrong", "labels_missing", "target_constant", "time_missing", "entity_missing"]
CHECKS = ["target_present", "target_fits_task", "target_labelled", "target_varies", "time_column_present", "entities_present"]

CC_CARD = (
    "Contract check: before any analysis, check the real table against the problem contract. Checks, in order: "
    "target_present (the contract's target column is in the table); target_fits_task (binary: exactly 2 distinct "
    "values; multiclass: 3 to 100; regression: numeric with more than 20); target_labelled (no blank target values); "
    "target_varies (more than one distinct value); time_column_present (only if the contract names one); "
    "entities_present (only if the contract names them). Decide: ask_owner when the data contradicts a STATED fact or "
    "the target cannot be used; fix_contract when it contradicts only an INFERRED fact (correct it and say why); "
    "proceed when every check passes. Output JSON."
)
ACTIONS = ["proceed", "fix_contract", "ask_owner"]


def kind_of(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    sample = s.dropna().astype(str).head(200)
    if len(sample) and pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() > 0.9 \
            and sample.str.contains(r"\d{4}-\d{2}-\d{2}").mean() > 0.9:
        return "date"
    return "text"


def profile(df: pd.DataFrame, table: str) -> str:
    lines = [f"TABLE: {table} — {len(df):,} rows, {df.shape[1]} columns", "| column | kind | distinct | blank |", "|---|---|---|---|"]
    for c in df.columns:
        lines.append(f"| {c} | {kind_of(df[c])} | {df[c].nunique(dropna=True):,} | {df[c].isna().mean():.1%} |")
    return "\n".join(lines)


def run_checks(df: pd.DataFrame, brief: ProblemBrief) -> dict:
    """Each check: pass, fail or n/a, with the numbers it rests on (all visible in the profile)."""
    out = {}
    target = brief.target_column.value if brief.target_column.known else None
    task = brief.task.value if brief.task.known else None
    present = target in df.columns if target else False
    out["target_present"] = ("pass" if present else "fail") if target else "n/a"
    if present:
        y = df[target]
        distinct = y.nunique(dropna=True)
        fits = {"binary": distinct == 2, "multiclass": 3 <= distinct <= 100,
                "regression": pd.api.types.is_numeric_dtype(y) and distinct > 20}.get(task)
        out["target_fits_task"] = "n/a" if fits is None else ("pass" if fits else "fail")
        out["target_labelled"] = "pass" if y.isna().mean() == 0 else "fail"
        out["target_varies"] = "pass" if distinct > 1 else "fail"
    else:
        out.update({"target_fits_task": "n/a", "target_labelled": "n/a", "target_varies": "n/a"})
    tc = brief.time_column.value if brief.time_column.known else None
    out["time_column_present"] = ("pass" if tc in df.columns else "fail") if tc else "n/a"
    ents = brief.entities.value if brief.entities.known else None
    out["entities_present"] = ("pass" if all(e in df.columns for e in ents) else "fail") if ents else "n/a"
    return out


def observed_task(y: pd.Series) -> str:
    d = y.nunique(dropna=True)
    if d == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(y) and d > 20:
        return "regression"
    return "multiclass" if 3 <= d <= 100 else "unknown"


def plant(name, key, spec, df, brief, rng):
    """Apply one planted mismatch. Returns (df, brief) or None when it does not apply to this scenario."""
    df = df.copy()
    target = spec["target"]
    if name == "clean":
        return df, brief
    if name == "target_renamed":
        new = rng.choice([f"is_{target}", f"{target}_flag", f"{target}_label", f"label_{target}"])
        return df.rename(columns={target: new}), brief
    if name == "task_wrong":
        if not (brief.task.known and brief.task.status == "inferred"):
            return None
        wrong = {"binary": "regression", "multiclass": "binary", "regression": "multiclass"}[brief.task.value]
        b = brief.model_copy(update={"task": TaskFact(value=wrong, status="inferred",
                                                      reason="the goal was read as asking for " + {"binary": "a yes or no", "multiclass": "one of several classes", "regression": "a quantity"}[wrong])})
        return df, b
    if name == "labels_missing":
        m = rng.random(len(df)) < rng.uniform(0.1, 0.4)
        df.loc[m, target] = np.nan
        return df, brief
    if name == "target_constant":
        if spec["task"] == "regression":
            return None
        keep = df[target] == df[target].mode().iloc[0]
        return df[keep].reset_index(drop=True), brief
    if name == "time_missing":
        if not (brief.time_column.known and brief.time_column.status == "inferred"):
            return None
        return df.drop(columns=[brief.time_column.value]), brief
    if name == "entity_missing":
        if not (brief.entities.known and brief.entities.status == "stated"):
            return None
        return df.drop(columns=list(brief.entities.value)), brief
    raise ValueError(name)


# A target with one value also cannot fit a binary or multiclass task, so that plant fails two checks.
EXPECTED_FAIL = {"target_renamed": ["target_present"], "task_wrong": ["target_fits_task"], "labels_missing": ["target_labelled"],
                 "target_constant": ["target_fits_task", "target_varies"], "time_missing": ["time_column_present"],
                 "entity_missing": ["entities_present"]}


def answer(name, df, brief, checks):
    """The decision, contract changes, questions and next step, by the stage card's rules."""
    target = brief.target_column.value
    changes, questions, details = [], [], {}
    cols = list(map(str, df.columns))
    for c in CHECKS:
        details[c] = {"n/a": "not applicable: the contract does not name it", "pass": "matches the table"}.get(checks[c], "")
    if name == "clean":
        decision, nxt = "proceed", "The contract and the table agree: run the data analysis on the training partition."
    elif name == "target_renamed":
        close = difflib.get_close_matches(target, cols, n=1, cutoff=0.5)
        details["target_present"] = f"`{target}` is not a column of the table" + (f"; the closest name is `{close[0]}`" if close else "")
        questions.append({"field": "target_column", "text": f"The table has no `{target}` column." + (f" Is `{close[0]}` the target?" if close else " Which column holds the label?"), "impact": 1})
        decision, nxt = "ask_owner", "Wait for the owner's answer: no analysis can test columns against the outcome without the target."
    elif name == "task_wrong":
        y = df[target]
        obs = observed_task(y)
        details["target_fits_task"] = f"`{target}` has {y.nunique(dropna=True):,} distinct values, which fits {obs}, not {brief.task.value}"
        changes.append({"field": "task", "from": brief.task.value, "to": obs,
                        "reason": f"`{target}` has {y.nunique(dropna=True):,} distinct values; the earlier reading was an inference, so the data corrects it"})
        decision, nxt = "fix_contract", f"Record the task as {obs}, then run the data analysis."
    elif name == "labels_missing":
        blank = df[target].isna().mean()
        details["target_labelled"] = f"{blank:.1%} of `{target}` values are blank"
        changes.append({"field": "label_coverage", "from": brief.label_coverage.value, "to": "partial",
                        "reason": f"{blank:.1%} of rows have no label; unlabelled is not the same as negative"})
        questions.append({"field": "label_coverage", "text": "Are the rows without a label unknown, or do they mean something (for example 'not checked')?", "impact": 2})
        decision, nxt = "fix_contract", "Mark coverage partial, keep unlabelled rows out of training and evaluation, then run the data analysis."
    elif name == "target_constant":
        v = df[target].dropna().unique()
        details["target_varies"] = f"`{target}` has a single value ({v[0]}) in all {len(df):,} rows"
        details["target_fits_task"] = f"one distinct value cannot be a {brief.task.value} target"
        questions.append({"field": "target_column", "text": f"Every row has the same `{target}`. Is this an extract of one class only, or is another file needed?", "impact": 1})
        decision, nxt = "ask_owner", "Stop: a model cannot learn from a target that never varies. Ask for the full data."
    elif name == "time_missing":
        tc = brief.time_column.value
        details["time_column_present"] = f"`{tc}` is not a column of the table"
        changes.append({"field": "time_column", "from": tc, "to": None,
                        "reason": f"`{tc}` was inferred, and the table has no such column"})
        questions.append({"field": "time_column", "text": "Which column says when each row happened? Without one, a time-ordered split is not possible.", "impact": 2})
        decision, nxt = "fix_contract", "Mark the time column unknown, ask for it, and run the analysis without a time split until it is answered."
    elif name == "entity_missing":
        ents = brief.entities.value
        details["entities_present"] = f"{', '.join('`' + e + '`' for e in ents)} {'is' if len(ents) == 1 else 'are'} not in the table"
        questions.append({"field": "entities", "text": f"You said rows repeat for {', '.join(ents)}, but the table has no such column. Which column identifies them?", "impact": 2})
        decision, nxt = "ask_owner", "Ask before splitting: a split that ignores repeating entities lets the same entity appear in training and test."
    out = {"checks": [{"check": c, "status": checks[c], "detail": details[c]} for c in CHECKS],
           "decision": decision, "contract_changes": changes, "questions": questions, "next": nxt}
    return out


def build(keys, seeds, per_plant, seed0):
    samples, report = [], {"dropped": Counter(), "by_plant": Counter()}
    for key in keys:
        spec = SCENARIOS[key]
        rng = np.random.default_rng(int(hashlib.sha256(("cc" + key).encode()).hexdigest()[:8], 16) + seed0)
        for seed in (seeds if spec["source"] == "replica" else [1]):
            df0, table = table_for(key, spec, seed)
            reps = per_plant if spec["source"] == "replica" else per_plant * 2
            for _ in range(reps):
                description, brief, connected = build_case(key, spec, rng, list(df0.columns), True)
                for name in PLANTS:
                    got = plant(name, key, spec, df0, brief, rng)
                    if got is None:
                        continue
                    df, b = got
                    checks = run_checks(df, b)
                    fails = [c for c in CHECKS if checks[c] == "fail"]
                    if fails != (EXPECTED_FAIL[name] if name != "clean" else []):
                        report["dropped"][f"checks_disagree_{name}"] += 1
                        continue
                    ans = answer(name, df, b, checks)
                    user = (f"STAGE: contract_check\n{CC_CARD}\n\nPROBLEM CONTRACT:\n{J(compact(b))}\n\n"
                            f"DATA PROFILE:\n{profile(df, table)}\n\nALLOWED DECISIONS: {' | '.join(ACTIONS)}")
                    out = J(ans)
                    bad = numbers_ok(out, user)
                    if bad:
                        report["dropped"]["number_not_in_input"] += 1
                        report.setdefault("number_examples", []).append({"scenario": key, "plant": name, "numbers": bad[:3]})
                        continue
                    report["by_plant"][name] += 1
                    samples.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                                 {"role": "user", "content": user},
                                                 {"role": "assistant", "content": out}],
                                    "metadata": {"scenario": key, "dataset": key, "seed": seed, "stage": "contract_check",
                                                 "plant": name, "action": ans["decision"], "source": spec["source"]}})
        print(f"{key:20s} {sum(1 for s in samples if s['metadata']['scenario'] == key):4d} samples", flush=True)
    return samples, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--per-plant", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "out")
    args = ap.parse_args()
    samples, report = build(args.scenarios, args.seeds, args.per_plant, args.seed)
    seen, splits = set(), {"train": [], "val": [], "test": []}
    for s in samples:
        h = hashlib.sha256(J(s["messages"][1:]).encode()).hexdigest()
        if h in seen:
            report["dropped"]["duplicate"] += 1
            continue
        seen.add(h)
        splits[SPLIT_OF.get(s["metadata"]["scenario"], "train")].append(s)
    random.Random(args.seed).shuffle(splits["train"])
    out = args.out / "contract_check"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(J(r) + "\n" for r in rows)
    report = {"splits": {k: len(v) for k, v in splits.items()},
              "by_decision": dict(Counter(s["metadata"]["action"] for v in splits.values() for s in v)),
              "by_plant": dict(report["by_plant"]), "dropped": dict(report["dropped"]),
              "number_examples": report.get("number_examples", [])[:10]}
    (out / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
