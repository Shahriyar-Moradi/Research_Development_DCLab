"""Frozen cell-level evaluation of the existing notebook copilot; no notebook execution."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .agentic.catalog import ROOT


def signal_for(cell, all_code, signal, copilot):
    if signal == "preprocessing_leakage":
        finding = copilot.detect_fit_before_split(cell)
    elif signal == "availability_warning":
        finding = copilot.detect_leaky_column_names(cell)
    elif signal == "missing_random_state":
        finding = copilot.detect_missing_random_state(cell)
    elif signal == "single_model_suggestion":
        finding = copilot.detect_single_model_no_comparison(cell, all_code)
    elif signal == "leakage":
        finding = copilot.detect_fit_before_split(cell) or copilot.detect_leaky_column_names(cell)
    else:
        raise ValueError("Unknown signal")
    return finding


def evaluate(manifest, root=ROOT):
    # Import the user-provided prototype from its existing location, without edits.
    prototype = root / "Cluad_R&D_aritfacts_sptember22"
    sys.path.insert(0, str(prototype))
    import notebook_copilot as copilot
    from build_rag_index import RetrievalIndex, load_all

    index = RetrievalIndex(load_all(root / "knowledge/model_building_rules.jsonl",
                                    root / "campaigns/model_building_50_v1/agent_memory.jsonl",
                                    root / "knowledge/workflow_blocks.json"))

    notebooks = {}
    rows = []
    for case in manifest["cases"]:
        path = root / case["notebook"]
        if path not in notebooks:
            cells = copilot.load_notebook_cells(path)
            notebooks[path] = (cells, "\n".join(cells))
        cells, all_code = notebooks[path]
        cell = cells[case["code_cell"]]
        fingerprint = hashlib.sha256(cell.encode()).hexdigest()
        if fingerprint != case["cell_sha256"]:
            raise ValueError(f"Frozen cell changed: {case['id']}; relabel before scoring")
        finding = signal_for(cell, all_code, case["signal"], copilot)
        observed = finding is not None
        reviewed = copilot.review_cell(cell, all_code, index)
        matched = next((item for item in reviewed if finding and item["comment"] == finding.comment), None)
        proof = matched["proof"] if matched else None
        rule = case.get("expected_proof_rule")
        proof_rule_pass = (rule in proof) if rule and proof else (False if rule else None)
        safe_wording = ("confirmed leaky" not in finding.comment.lower()) if finding and case["signal"] == "availability_warning" else None
        rows.append({"id": case["id"], "notebook": case["notebook"], "code_cell": case["code_cell"],
                     "cell_sha256": fingerprint, "signal": case["signal"], "expected": case["expected"],
                     "observed": observed, "pass": observed == case["expected"], "why": case["why"],
                     "comment": finding.comment if finding else None,
                     "expected_proof_rule": rule, "proof_rule_pass": proof_rule_pass,
                     "proof_excerpt": proof[:400] if proof else None, "safe_wording": safe_wording,
                     "human_review": None})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "evaluation/notebook_cases_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path; evaluations are immutable snapshots")
    raw = args.manifest.read_bytes()
    rows = evaluate(json.loads(raw))
    result = {"suite_id": "dclab_hyperack_churn_notebook_pilot_v1",
              "manifest_sha256": hashlib.sha256(raw).hexdigest(),
              "created_at": datetime.now(timezone.utc).isoformat(),
              "passed": sum(r["pass"] for r in rows), "total": len(rows),
              "proof_rule_passed": sum(r["proof_rule_pass"] is True for r in rows),
              "proof_rule_scored": sum(r["proof_rule_pass"] is not None for r in rows),
              "availability_wording_safe": all(r["safe_wording"] is not False for r in rows),
              "records": rows,
              "overall_release_ready": False,
              "limitations": ["Development cases from two known notebooks; no blinded test set or user study.",
                              "Notebook cells are inspected but not executed; output correctness and runtime are not evaluated.",
                              "Cancellation-date leakage depends on an owner-approved prediction-time contract.",
                              "The prototype's wording says name-matched columns are confirmed leaky; that is stronger than this evidence supports."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"output": str(args.output), "passed": result["passed"], "total": len(rows),
                      "proof_rule_passed": result["proof_rule_passed"],
                      "proof_rule_scored": result["proof_rule_scored"],
                      "availability_wording_safe": result["availability_wording_safe"],
                      "failed_cases": [r["id"] for r in rows if not r["pass"]]}))


if __name__ == "__main__":
    main()
