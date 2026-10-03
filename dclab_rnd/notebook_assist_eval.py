"""Evaluate the new notebook companion on frozen HyperAck/churn development cases."""
import argparse
import hashlib
import json
from pathlib import Path

from .notebook_assist import MANIFEST_CANDIDATES, ROOT, manifest_path, resolve_path, review_document


SIGNALS = {
    "preprocessing_leakage": {"preprocessing_before_split", "cross_cell_preprocessing_review"},
    "availability_warning": {"prediction_time_availability"},
    "missing_random_state": {"unseeded_split"},
    "single_model_suggestion": {"single_model_comparison"},
    "leakage": {"preprocessing_before_split", "cross_cell_preprocessing_review", "prediction_time_availability"},
}


def evaluate(manifest: dict):
    cached = {}
    rows = []
    for case in manifest["cases"]:
        path = resolve_path(case["notebook"])
        if path not in cached:
            document = json.loads(path.read_text(encoding="utf-8"))
            document["notebook_path"] = case["notebook"]
            cached[path] = review_document(document)
        cell = next((c for c in cached[path]["code_cells"] if c["code_cell_index"] == case["code_cell"]), None)
        if not cell or cell["source_sha256"] != case["cell_sha256"]:
            raise ValueError(f"Frozen notebook cell changed: {case['id']}")
        finding = next((f for f in cell["findings"] if f["kind"] in SIGNALS[case["signal"]]), None)
        observed = finding is not None
        expected_rule = case.get("expected_proof_rule")
        proof_ids = [r["id"] for r in finding["proof"]["records"]] if finding else []
        rows.append({"id": case["id"], "observed": observed, "expected": case["expected"],
                     "pass": observed == case["expected"],
                     "expected_proof_rule": expected_rule,
                     "proof_rule_pass": expected_rule in proof_ids if expected_rule else None,
                     "safe_wording": ("confirmed leaky" not in json.dumps(finding).lower())
                     if finding and case["signal"] == "availability_warning" else None,
                     "finding_kind": finding["kind"] if finding else None,
                     "cell_sha256": cell["source_sha256"]})
    return {"suite_id": "dclab_notebook_companion_development_v1",
            "scope": "The same researcher-authored development cases used to find prior prototype flaws; not blinded generalization",
            "cases_passed": sum(r["pass"] for r in rows), "cases_total": len(rows),
            "proof_rules_passed": sum(r["proof_rule_pass"] is True for r in rows),
            "proof_rules_scored": sum(r["proof_rule_pass"] is not None for r in rows),
            "availability_wording_safe": all(r["safe_wording"] is not False for r in rows),
            "overall_release_ready": False, "records": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=manifest_path() or MANIFEST_CANDIDATES[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path; reports are immutable snapshots")
    raw = args.manifest.read_bytes()
    report = evaluate(json.loads(raw))
    report["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: report[k] for k in ("cases_passed", "cases_total", "proof_rules_passed",
                                             "proof_rules_scored", "availability_wording_safe")}))


if __name__ == "__main__":
    main()
