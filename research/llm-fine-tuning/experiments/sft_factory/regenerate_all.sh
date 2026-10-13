#!/usr/bin/env bash
# Rebuild every stage of the SFT factory, then check and merge them (about 2 hours on 2 CPUs).
# Run from the repository root: bash research/llm-fine-tuning/experiments/sft_factory/regenerate_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY problem_to_plan_samples.py
$PY contract_check_samples.py
$PY leakage_audit_samples.py --variants 8
$PY end_to_end_samples.py --variants 3
$PY workflow_samples.py --variants 24   # all five stages, the balance stage included
$PY code_samples.py --variants 4 --review-variants 2
$PY check_and_merge.py
$PY make_preview.py
$PY build_training_mix.py   # training items for the Mac (out/mix, not committed)
$PY replay_tasks.py
echo "done"
