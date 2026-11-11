# DCLab GLiNER2.5-Decide / IBM Telco Churn pilot

## Aim
Train a task-conditioned text classifier on serialized historical customer features, compare its results with two tabular baselines on identical splits, and avoid claiming prospective 30-day risk from a cross-sectional sample.

Source data is already present in DCLab: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Get started (from DCLab repository root)

```bash
cd Research_Development_DCLab
# Unzip the starter files into research/churn-prediction/experiments/gliner_decide
python3.11 -m venv .venv-gliner
source .venv-gliner/bin/activate
python -m pip install --upgrade pip
pip install -r research/churn-prediction/experiments/gliner_decide/requirements.txt
python -c 'import torch; print("MPS available:", torch.backends.mps.is_available())'
```

### 1. Prepare

```bash
python research/churn-prediction/experiments/gliner_decide/prepare.py
```

Creates train.jsonl, dev.jsonl, test.jsonl and splits.csv. It never places `customerID` or the `Churn` column in model inputs. Identical serialized feature rows stay in the same split.

### 2. Baseline models

```bash
python research/churn-prediction/experiments/gliner_decide/baselines.py
```

Fits logistic regression and histogram gradient boosting using only train; evaluates only on dev. These are fresh same-split comparisons, not CHURN-014's repeated OOF CV.

### 3. Check training using 64 customers

```bash
python research/churn-prediction/experiments/gliner_decide/train_gliner.py --smoke --out research/churn-prediction/experiments/gliner_decide/runs/gliner_smoke
```

This smoke run uses 64 train examples and one epoch only to confirm the training stack and GPU backend function. Do NOT evaluate it as a research model.

### 4. Full fine-tuning

```bash
python research/churn-prediction/experiments/gliner_decide/train_gliner.py --epochs 3 --batch-size 2
```

This uses the official `ExtractorTrainer` and starts with modest batch size for Apple Silicon. GPU/MPS trainer compatibility is **not verified** in this kit; if MPS fails, use a Linux NVIDIA CUDA environment. Full encoder tuning may take significant memory and compute; consider documented LoRA support in a follow-up experiment.

### 5. Evaluate on development split

```bash
python research/churn-prediction/experiments/gliner_decide/evaluate.py
```

GLiNER label-based metrics and score-derived ranking metrics are compared with baselines. Its winning-label confidence is NOT a calibrated churn probability. Do not treat Brier/log loss from that confidence as deployment-ready calibration evidence.

### 6. Final one-time test (only after all settings are frozen)

```bash
python research/churn-prediction/experiments/gliner_decide/evaluate.py --split test --final
```

A `SEALED_TEST_OPENED.json` marker blocks accidentally running the final test again. Do not tune based on its results. A future experiment needs fresh, independent cohorts.

## Important scientific limitations

- IBM Telco is cross-sectional: the original `Churn` label indicates churn, but event times and the precise prediction moment are not verified. This is a retrospective classification exercise, not validated future 30-day churn prediction.
- Real deployment requires features demonstrably available at decision time, time-aware and entity-grouped validation, leakage review, probability calibration, company-level transfer evaluation, and consented data.
- Never include customer IDs or observed outcome labels in the text input.
- This is a separate R&D experiment and does not modify DCLab core or its sealed release benchmarks.
- The repo's CHURN-014 historical 0.850 ROC-AUC is *not comparable directly* to this different train/dev/test protocol.
- Model checkpoint is Apache 2.0 per the official Fastino model card. Verify all data licenses and model terms before distribution.

## References

- https://huggingface.co/fastino/GLiNER2.5-Decide
- https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/9-training.md
- https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv
