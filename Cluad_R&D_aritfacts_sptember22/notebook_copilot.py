#!/usr/bin/env python3
"""
notebook_copilot.py
=====================
A working prototype of "an intelligent agent that writes a comment beside
every cell of code" — grounded in your evidence registry, not a generic
linter opinion. Every comment comes with a "Show proof" citation: the
DCLab rule it's based on, plus a real precedent from your 50-experiment
registry — the StackOverflow-style "someone else hit this, here's what
happened" trust mechanism, applied to ML methodology instead of syntax
errors.

WHAT IT DETECTS (v0 — a small, honest starting set, not an exhaustive linter)
------------------------------------------------------------------------------
1. Preprocessing fit BEFORE train_test_split — classic preprocessing leakage
   (DCLAB-R03). Detected structurally: a `.fit(` / `.fit_transform(` call on
   a transformer-like object appears in the code before `train_test_split(`
   is called.
2. A feature column whose name matches a known leakage-pattern vocabulary
   (built from real columns your registry already found leaky — `duration`,
   `cancellation_date`, `outcome_date`, etc. — plus generic suffixes like
   `_after`, `post_`, `final_`, `actual_`, `resolved`).
3. A single model fit with no comparison against any other candidate —
   flags the "no universal best model, always screen a few families"
   finding (DCLAB-R13/R14).
4. `train_test_split(` called with no `random_state=` — a reproducibility
   gap, not a correctness bug, but worth flagging with lower severity.

Each finding is matched to a real rule + a real precedent experiment via
`build_rag_index.py`'s retrieval index (same file, unmodified — this is
additive tooling on top of it, not a fork of it).

Usage
-----
    python notebook_copilot.py --notebook path/to/notebook.ipynb \
        --rules knowledge/model_building_rules.jsonl \
        --claims campaigns/model_building_50_v1/agent_memory.jsonl \
        --blocks knowledge/workflow_blocks.json

This is v0: a real, working, testable core — not the full IDE integration.
See the accompanying NOTEBOOK_COPILOT_SPEC.md for the inline-comment UX,
the "Show proof" panel design, and how this becomes a Jupyter/VS Code
extension or an MCP tool Claude Code/Cursor can call directly.
"""

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

from build_rag_index import RetrievalIndex, load_all

KNOWN_LEAKY_TOKENS = {
    "duration", "cancellation_date", "pagevalues", "final_customer_fare", "customers",
}
LEAKY_SUFFIXES = ("_after", "_post", "_final", "_actual", "_resolved", "_outcome")
LEAKY_PREFIXES = ("post_", "final_", "actual_", "resolved_", "outcome_")

TRANSFORMER_FIT_RE = re.compile(r"\b(\w+)\.fit(_transform)?\(")
SPLIT_RE = re.compile(r"\btrain_test_split\(")
MODEL_FIT_RE = re.compile(r"\b(\w+)\s*=\s*(RandomForestClassifier|LogisticRegression|XGBClassifier|"
                           r"LGBMClassifier|CatBoostClassifier|GradientBoostingClassifier|SVC)\(")
FEATURE_LIST_RE = re.compile(r"\[\s*((?:['\"]\w[\w]*['\"]\s*,?\s*)+)\]")
RANDOM_STATE_RE = re.compile(r"random_state\s*=")


@dataclass
class Finding:
    rule_category: str
    severity: str
    comment: str
    query_for_proof: str


def _extract_string_list(code: str) -> list[str]:
    names = []
    for m in FEATURE_LIST_RE.finditer(code):
        names += re.findall(r"['\"](\w[\w]*)['\"]", m.group(1))
    return names


def detect_fit_before_split(code: str):
    fit_match = TRANSFORMER_FIT_RE.search(code)
    split_match = SPLIT_RE.search(code)
    if fit_match and split_match and fit_match.start() < split_match.start():
        return Finding(
            rule_category="leakage",
            severity="high",
            comment=(
                f"`{fit_match.group(1)}.fit(...)` runs BEFORE `train_test_split(...)` in this cell. "
                "That means the transformer sees the test rows during fitting — a classic "
                "preprocessing-leakage pattern, not a data problem."
            ),
            query_for_proof="fit transformer before split preprocessing leakage",
        )
    return None


def detect_leaky_column_names(code: str):
    names = _extract_string_list(code)
    hits = [
        n for n in names
        if n.lower() in KNOWN_LEAKY_TOKENS
        or n.lower().endswith(LEAKY_SUFFIXES)
        or n.lower().startswith(LEAKY_PREFIXES)
    ]
    if hits:
        return Finding(
            rule_category="leakage",
            severity="high",
            comment=(
                f"Column(s) {hits} are in your feature list and match names that have been "
                "confirmed leaky in past experiments (only knowable after the outcome already "
                "happened). Worth an explicit availability check before training."
            ),
            query_for_proof=" ".join(hits) + " leakage safe unsafe",
        )
    return None


def detect_single_model_no_comparison(code: str, all_cells_code: str):
    models_in_cell = MODEL_FIT_RE.findall(code)
    if not models_in_cell:
        return None
    distinct_models_in_notebook = set(m[1] for m in MODEL_FIT_RE.findall(all_cells_code))
    if len(distinct_models_in_notebook) == 1:
        model_name = models_in_cell[0][1]
        return Finding(
            rule_category="model_selection",
            severity="medium",
            comment=(
                f"Only `{model_name}` is fit anywhere in this notebook, with no side-by-side "
                "comparison. Across a real 10-dataset registry, no single algorithm won every "
                "time — it's worth screening at least a linear baseline and one boosting model "
                "on identical folds before committing."
            ),
            query_for_proof="no universal best model algorithm screen comparison",
        )
    return None


def detect_missing_random_state(code: str):
    if SPLIT_RE.search(code) and not RANDOM_STATE_RE.search(code):
        return Finding(
            rule_category="evaluation",
            severity="low",
            comment=(
                "`train_test_split(...)` has no `random_state=`, so this split (and anything "
                "measured on it) won't be reproducible between runs."
            ),
            query_for_proof="reproducibility deterministic split",
        )
    return None


def review_cell(code: str, all_cells_code: str, index: RetrievalIndex) -> list[dict]:
    detectors = [
        detect_fit_before_split,
        detect_leaky_column_names,
        lambda c: detect_single_model_no_comparison(c, all_cells_code),
        detect_missing_random_state,
    ]
    findings = [f for f in (d(code) for d in detectors) if f]

    reviewed = []
    for f in findings:
        hits = index.search(f.query_for_proof, filters={"kind": "rule", "category": f.rule_category}, k=1)
        hits += index.search(f.query_for_proof, filters={"kind": "experiment_claim"}, k=1)
        proof = "\n\n---\n\n".join(r.display for r, _ in hits) or "(no matching precedent found — flagged on general methodology grounds only)"
        reviewed.append({"severity": f.severity, "comment": f.comment, "proof": proof})
    return reviewed


def load_notebook_cells(path: Path) -> list[str]:
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        return ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
    return [path.read_text(encoding="utf-8")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--rules", type=Path, default=Path("knowledge/model_building_rules.jsonl"))
    parser.add_argument("--claims", type=Path, default=Path("campaigns/model_building_50_v1/agent_memory.jsonl"))
    parser.add_argument("--blocks", type=Path, default=Path("knowledge/workflow_blocks.json"))
    args = parser.parse_args()

    records = load_all(args.rules, args.claims, args.blocks)
    index = RetrievalIndex(records)

    cells = load_notebook_cells(args.notebook)
    all_code = "\n".join(cells)

    for i, cell in enumerate(cells):
        try:
            ast.parse(cell)
        except SyntaxError:
            pass
        findings = review_cell(cell, all_code, index)
        print(f"\n{'='*70}\nCELL {i}\n{'='*70}")
        print(cell.strip())
        if not findings:
            print("\n  (no findings)")
            continue
        for f in findings:
            print(f"\n  \u26a0 [{f['severity'].upper()}] {f['comment']}")
            print(f"    Show proof:\n    " + f['proof'].replace("\n", "\n    "))


if __name__ == "__main__":
    main()
