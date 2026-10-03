"""Offline, read-only notebook advice with exact DCLab rule references.

The output contract is shared by the CLI and the VS Code notebook extension.
Notebook text is untrusted input; it is inspected, never executed or sent away.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "knowledge/model_building_rules.jsonl"
CLAIMS = ROOT / "campaigns/model_building_50_v1/agent_memory.jsonl"
MODEL_NAMES = ("RandomForestClassifier", "LogisticRegression", "XGBClassifier", "LGBMClassifier",
               "CatBoostClassifier", "GradientBoostingClassifier", "SVC", "HistGradientBoostingClassifier")
SUSPICIOUS_NAMES = {"duration", "cancellation_date", "final_customer_fare", "final_biker_fare",
                    "outcome_date", "pagevalues"}
SUSPICIOUS_PREFIXES = ("post_", "final_", "actual_", "outcome_")
SUSPICIOUS_SUFFIXES = ("_after", "_final", "_actual", "_outcome")


def fingerprint(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_text(value) -> str:
    return "".join(value) if isinstance(value, list) else str(value or "")


def calls(node: ast.AST) -> set[str]:
    names = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Call):
            name = item.func
            if isinstance(name, ast.Name):
                names.add(name.id)
            elif isinstance(name, ast.Attribute):
                names.add(name.attr)
    return names


def code_facts(code: str) -> dict:
    """Extract narrow syntax facts; do not infer runtime ordering from definitions."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"parseable": False, "operations": [], "unseeded_split": False,
                "feature_names": [], "df_feature_write": False, "model_types": []}
    wrappers = {}
    for item in tree.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found = calls(item)
            wrappers[item.name] = ({"fit"} if found & {"fit", "fit_transform", "fit_predict"} else set()) | (
                {"split"} if "train_test_split" in found else set())
    operations = []
    df_feature_write = False
    model_types = []
    feature_lists = {}
    uses_feature_list = set()
    for item in tree.body:
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            value = item.value
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            for target in targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "df":
                    df_feature_write = True
                if isinstance(target, ast.Name) and isinstance(value, (ast.List, ast.Tuple)):
                    strings = [v.value for v in value.elts if isinstance(v, ast.Constant) and isinstance(v.value, str)]
                    if len(strings) == len(value.elts):
                        feature_lists[target.id] = strings
                if isinstance(target, ast.Name) and target.id in {"X", "features", "feature_cols", "feature_columns"}:
                    for child in ast.walk(value):
                        if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name) and child.value.id == "df":
                            if isinstance(child.slice, ast.Name):
                                uses_feature_list.add(child.slice.id)
                            elif isinstance(child.slice, (ast.List, ast.Tuple)):
                                vals = [v.value for v in child.slice.elts if isinstance(v, ast.Constant) and isinstance(v.value, str)]
                                feature_lists["__direct_X__"] = vals
        found = calls(item) if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) else set()
        has_fit = bool(found & {"fit", "fit_transform", "fit_predict"}) or any("fit" in wrappers.get(n, set()) for n in found)
        has_split = "train_test_split" in found or any("split" in wrappers.get(n, set()) for n in found)
        if has_fit and has_split:
            operations.append("ambiguous_same_statement")
        elif has_fit:
            operations.append("fit")
        elif has_split:
            operations.append("split")
        for name in MODEL_NAMES:
            if name in found:
                model_types.append(name)
    # Direct calls within a statement such as split(scaler.fit_transform(X)) are
    # intentionally not ordered. They need manual review rather than a false claim.
    unseeded_split = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                         and node.func.id == "train_test_split"
                         and not any(k.arg == "random_state" for k in node.keywords)
                         for node in ast.walk(tree))
    chosen = feature_lists.get("__direct_X__", [])[:]
    for name in uses_feature_list:
        chosen.extend(feature_lists.get(name, []))
    # `features = [...]` is only treated as a model feature list when X uses it.
    return {"parseable": True, "operations": operations, "unseeded_split": unseeded_split,
            "feature_names": sorted(set(chosen)), "df_feature_write": df_feature_write,
            "model_types": sorted(set(model_types))}


def rule_proof(rule_ids: list[str], rules: dict, direct_claim: dict | None = None) -> dict:
    records = []
    for rule_id in rule_ids:
        rule = rules[rule_id]
        records.append({"type": "methodological_rule", "id": rule_id,
                        "statement": rule["statement"], "why": rule["why"],
                        "source_references": rule["evidence"], "confidence": rule["confidence"]})
    if direct_claim:
        claim = direct_claim["claim"]
        records.append({"type": "recorded_experiment_claim", "id": claim["claim_id"],
                        "statement": claim["statement"], "limitations": claim["limitations"],
                        "source_path": direct_claim["source_path"],
                        "dataset": direct_claim["dataset"]})
    return {"records": records, "exact_analogous_experiment": bool(direct_claim)}


def finding(kind, severity, confidence, title, explanation, next_step, rule_ids, rules, direct_claim=None):
    return {"kind": kind, "severity": severity, "confidence": confidence,
            "title": title, "explanation": explanation, "next_step": next_step,
            "proof": rule_proof(rule_ids, rules, direct_claim), "auto_fix_available": False}


def inferred_dataset(path: str) -> str | None:
    lower = path.lower()
    if "hyper" in lower: return "hyperack"
    if "telco_churn" in lower or "telco-customer-churn" in lower: return "telco_churn"
    if "bank_marketing" in lower: return "bank_marketing"
    return None


def review_document(document: dict, rules_path: Path = RULES, claims_path: Path = CLAIMS) -> dict:
    rules_raw = rules_path.read_bytes()
    rules = {r["rule_id"]: r for r in load_jsonl(rules_path)}
    claims = {c["claim"]["claim_id"]: c for c in load_jsonl(claims_path)}
    raw_cells = document.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) > 1000:
        raise ValueError("cells must be a list of at most 1000 items")
    notebook_path = str(document.get("notebook_path") or "")
    dataset = inferred_dataset(notebook_path)
    cells = []
    for index, raw in enumerate(raw_cells):
        if not isinstance(raw, dict):
            raise ValueError("Each cell must be an object")
        kind = raw.get("cell_type", raw.get("kind", ""))
        if kind not in ("code", "markdown"):
            continue
        code = source_text(raw.get("source"))
        if len(code) > 200000:
            raise ValueError("A cell exceeds the 200,000 character review limit")
        cells.append({"notebook_cell_index": index, "cell_type": kind,
                      "source_sha256": fingerprint(code.encode()), "code": code,
                      "facts": code_facts(code) if kind == "code" else None})
    types = {name for c in cells if c["facts"] for name in c["facts"]["model_types"]}
    output = []
    code_index = 0
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        facts = cell["facts"]
        findings = []
        ops = facts["operations"]
        if "fit" in ops and "split" in ops and ops.index("fit") < ops.index("split"):
            findings.append(finding("preprocessing_before_split", "high", "code_order_observed",
                "Preprocessing appears to fit before the split",
                "A fitted transform runs before the train/test split in this cell's top-level flow. Evaluation rows may influence the transform.",
                "Split first; fit transformations inside training folds or a Pipeline, then compare on unchanged folds.",
                ["DCLAB-R03", "DCLAB-R04"], rules))
        if "fit" in ops and facts["df_feature_write"] and "split" not in ops:
            later = cells[i + 1:]
            if any(c["facts"] and "split" in c["facts"]["operations"] and
                   ("preprocess_data(df)" in c["code"] or re.search(r"\bX\s*=\s*df\b", c["code"]))
                   for c in later):
                findings.append(finding("cross_cell_preprocessing_review", "high", "execution_order_unverified",
                    "Check a fitted feature created before a later split",
                    "This cell fits a transform and writes a dataframe feature; a later cell builds X from that dataframe and splits it. Notebook execution order and lineage need confirmation.",
                    "If this feature enters evaluation, move its fitted steps inside each training fold; rerun from a clean kernel and compare.",
                    ["DCLAB-R03", "DCLAB-R04"], rules))
        for name in facts["feature_names"]:
            lower = name.lower()
            if lower in SUSPICIOUS_NAMES or lower.startswith(SUSPICIOUS_PREFIXES) or lower.endswith(SUSPICIOUS_SUFFIXES):
                direct = claims.get("EXP-007-C2") if dataset == "bank_marketing" and lower == "duration" else None
                findings.append(finding("prediction_time_availability", "review", "needs_product_contract",
                    f"Verify when `{name}` becomes available",
                    f"`{name}` is selected as an input and its name suggests it may be recorded after the prediction moment. A name alone does not prove leakage.",
                    "Record the decision time, source timestamp and lineage. Exclude it only if unavailable at prediction time; then run a safe-feature ablation.",
                    ["DCLAB-R01", "DCLAB-R05"], rules, direct))
        if facts["unseeded_split"]:
            findings.append({"kind": "unseeded_split", "severity": "low", "confidence": "syntax_observed",
                "title": "Make this split repeatable", "explanation": "train_test_split has no explicit random_state in this call.",
                "next_step": "Set a seed and preserve the split identity before comparing experiments.",
                "proof": {"records": [], "exact_analogous_experiment": False,
                          "note": "No directly matching DCLab rule or experiment is cited for this syntax suggestion."},
                "auto_fix_available": False})
        if facts["model_types"] and len(types) == 1:
            findings.append(finding("single_model_comparison", "suggestion", "not_a_correctness_fault",
                "Consider a comparison baseline",
                f"This notebook instantiates only {next(iter(types))}. That may be intentional, but one model cannot establish which family works best here.",
                "On identical folds, compare a transparent baseline with one justified alternative before tuning.",
                ["DCLAB-R13"], rules))
        review_status = ("unsupported_syntax" if not facts["parseable"] else
                         "findings" if findings else
                         "partial_static_review" if "ambiguous_same_statement" in ops else
                         "no_rule_matched_not_certified")
        output.append({"notebook_cell_index": cell["notebook_cell_index"],
                       "code_cell_index": code_index, "source_sha256": cell["source_sha256"],
                       "findings": findings, "review_status": review_status})
        code_index += 1
    return {"schema_version": 1, "notebook_path": notebook_path, "dataset_inferred": dataset,
            "rules_sha256": fingerprint(rules_raw), "code_cells": output,
            "finding_count": sum(len(c["findings"]) for c in output),
            "limitations": ["Static review only; notebook execution order and runtime values are not observed.",
                            "No model-generated code or automatic notebook edits.",
                            "A DCLab rule is methodology, not proof that this notebook has a bug.",
                            "Only an exact matching experiment claim is labeled analogous; otherwise no direct precedent is asserted."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--notebook", type=Path)
    source.add_argument("--stdin", action="store_true")
    parser.add_argument("--output", type=Path, help="Immutable JSON report; omit to print JSON on stdout")
    args = parser.parse_args()
    if args.notebook:
        document = json.loads(args.notebook.read_text(encoding="utf-8"))
        document["notebook_path"] = str(args.notebook)
    else:
        document = json.load(sys.stdin)
    report = review_document(document)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
        print(json.dumps({"output": str(args.output), "code_cells": len(report["code_cells"]),
                          "finding_count": report["finding_count"]}))
    else:
        print(serialized)


if __name__ == "__main__":
    main()
