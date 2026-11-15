# DCLab GLiNER2.5-Decide Telecom Churn: Training-Readiness Audit

**Date:** 2026-11-09  
**Scope:** `DCLab_GLiNER25_Decide_Telco_Churn_Finetuning_Master_Plan.md`, `DCLab_GLiNER_Telco_Data_Engineering_and_Dataset_Creation_Plan.md`, supplied starter (`common.py`, `prepare.py`, `baselines.py`, `train_gliner.py`, `evaluate.py`), and DCLab source/experiment records at `08c7bac`.  
**Review method:** Source/code/document inspection and comparison with official Fastino GLiNER2.5 model/training docs. No real 340M training run was performed.  
**Overall status:** **NOT READY for a controlled, reportable full fine-tuning study**. A local experimental fine-tune might execute after dependencies are installed, but the present scripts cannot establish trustworthy performance or protected test custody.

## 1. Summary and decision

The fundamental approach is sound: map `Churn` (`Yes`/`No`) to single-label supervision, serialize all 19 permitted source features into deterministic text, fine-tune the 340M GLiNER2.5-Decide checkpoint, and compare with baselines. The official GLiNER2 format accepts `{"input":..., "output":{"classifications":[{"task":...,"labels":...,"true_label":[...] }]}}`; `ExtractorTrainer(...).train(train_data=..., eval_data=...)` is supported.

**What is not proven:** data rights for the specific original CSV and derived checkpoint; production-grade dataset factory; a successful GLiNER/MPS backward pass; correct checkpoint reload; calibrated churn scores; an independent external evaluation.

**Important scientific qualification:** the source IBM Telco rows have already been used across DCLab's 15-experiment churn CV campaign. Reserving a new holdout partition protects this fine-tuning run from future direct optimization, but **does not turn the source into an untouched external benchmark**. Label it *internal frozen test for this study*, not independent cross-company evidence.

## 2. Blocking gaps (P0)

| ID | Gap | Evidence | Precise change / pass condition |
|---|---|---|---|
| **TR-01** | Hardened source audit and data release **not implemented** | `common.py` uses `pd.to_numeric(..., errors="coerce")`; `prepare.py` lacks full categorical/rights/token-length audits | Implement DATA-01–DATA-08, fail on unreviewed invalids, enforce source SHA, use strict GLiNER validator and DCLab QC gate. No labeled test JSONL in trainer view. |
| **TR-02** | Labeled test examples and raw truth accessible to training process | Starter writes `processed/test.jsonl` with `true_label`; `splits.csv` includes `target`; repo source CSV also includes `Churn` | Data-preparation/evaluator process keeps original labeled CSV and test truth in a **separate protected directory/container/user**, inaccessible to trainer. Give trainer only permitted train/dev data and unlabeled test inputs only when scoring is authorized. Merely moving `test_labels.jsonl` is **not sufficient** while original CSV remains readable. |
| **TR-03** | GLiNER runtime training not verified | No 340M smoke-training or checkpoint reload evidence; starter sets `PYTORCH_ENABLE_MPS_FALLBACK=1` automatically | Pin model/package revisions; run import, CPU forward, MPS backward, 16–64-row train, save, reload and inference. Record actual device, peak memory, throughput; explicitly configure mixed precision. Do not hide unsupported MPS operators by automatic CPU fallback. |
| **TR-04** | No validation loop in current starter training | `train_gliner.py`: `ExtractorTrainer(...).train(train_data=str(src))`, no `eval_data` | Pass frozen `dev.jsonl`, set `eval_strategy`, `save_best`, `metric_for_best`, log validation loss and performance on dev. Save best and final separately and reload the actual selected checkpoint. |
| **TR-05** | Incorrect classification-confidence usage | `evaluate.py` uses winner confidence for `churn`, `1-confidence` for `not_churn` | Confirm exact scoring semantics in the **pinned** GLiNER version. Extract actual positive-class score if available; otherwise report label-only balanced accuracy/F1 and mark ROC-AUC/AP/Brier/log-loss as unavailable for GLiNER. Add invalid-output tests. |
| **TR-06** | Sealed evaluation can be bypassed | `evaluate.py` uses a writable `SEALED_TEST_OPENED.json` under user-selectable output folder | Independent evaluator, evaluator-only ground truth, approval with frozen model hashes, atomic registry lock keyed by `(source_sha, split_version, study_family)`, immutable final report. New output path must never reopen the same test as sealed. |
| **TR-07** | Training-use rights not yet affirmatively cleared | Plans flag source license review; model's Apache-2.0 license does not cover data rights | Record specific IBM dataset source/license, permitted model training and distribution, and internal vs public checkpoint release policy. If uncertain, restrict to local exploratory work pending review; block distribution/hosted uploads. |
| **TR-08** | Reproducibility and artifact identity incomplete | Current dependencies unpinned; mutable model `main`; training manifest only a few fields | Store resolved checkpoint commit SHA, package lock, Python/OS/MPS/CUDA versions, dataset hashes, serializer/split version, RNG seeds, full config, checkpoint hashes, output logs, allowed inputs and consent policy. |

## 3. Gaps that weaken scientific conclusions (P1)

| ID | Gap | Fix / acceptance condition |
|---|---|---|
| **SC-01** | **Missing zero-shot arm in runnable evaluator.** Plan describes G0 but `evaluate.py` only tests tuned GLiNER and two baselines. | Score the **untuned original GLiNER** on identical dev/test examples, frozen schema, and runtime version. |
| **SC-02** | Test evidence is not independent of older IBM Telco development campaigns. | Keep internal test designation; source and freeze a **different labeled churn dataset** or a later real-company cohort for external confirmation. No invented 30-day target. |
| **SC-03** | Dataset schema, task name and label consistency are not all enforced at the trainer boundary. | Assert `task=churn_status`, `labels=[not_churn,churn]`, exactly one valid `true_label`, train/inference schema parity, zero unapproved truncations, explicit `Churn` label exclusion. |
| **SC-04** | No explicit target-label/duplicate-identity audit or tokenization coverage in starter. | Validate duplicate `customerID` and identical allowed-feature fingerprints as connected groups, contradictory outcomes, category domains, `TotalCharges` blanks, every tokenizer length, and expected source digest. Freeze exact membership. |
| **SC-05** | Validation overfitting / model selection risk. | Limit and record experiments; predefine score/threshold-selection plan; where possible run multiple seeds. Never tune on locked-test feedback. |
| **SC-06** | Performance may be affected by label imbalance and numeric serialization. | Report prevalence; compare majority-class dummy, logistic, nonlinear tabular and G0/G1. Measure minority-class recall/AP and a preregistered numeric-format sensitivity study. Never rebalance dev/test. |
| **SC-07** | Latency, memory and model artifact size not benchmarked. | Log throughput, peak memory, CPU/GPU utilization, checkpoint size, inference latency and batch behavior. Measure cost per valid prediction against baselines. |
| **SC-08** | Training checkpoint is not a fully resumable state. | GLiNER docs say checkpoint loads weights but not optimizer/scheduler state. Treat a restart as a **new training segment**, log it and avoid claiming exact resume equivalence. |
| **SC-09** | Prediction semantics are retrospective only. | Report `Churn` on the IBM cross-sectional snapshot, not `churn within next 30 days`. Prospective deployment needs explicit prediction cutoffs, label maturity, out-of-time splits and downstream decisions/outcomes. |
| **SC-10** | Model robustness, fairness and distribution shift remain untested. | Add abstention/invalid response rate, score by contract/tenure strata with uncertainty, optional demographic-excluded ablation, serialization perturbations, missing-feature tests, and independent dataset test. Avoid overclaiming small subgroup results. |

## 4. Proposed precise training configuration (candidate, not executed)

First verify `inspect.signature(TrainingConfig)` in the installed pinned version; then implement a frozen configuration equivalent to:

```python
config = TrainingConfig(
    output_dir="runs/GLCH-001",
    num_epochs=3,
    batch_size=2,
    gradient_accumulation_steps=8,
    encoder_lr=1e-5,
    task_lr=5e-4,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_best=True,
    metric_for_best="eval_loss",
    greater_is_better=False,
    seed=42,
    fp16=False,  # initial MPS compatibility candidate; measure before optimizing
    bf16=False,
    report_to_wandb=False,
)
trainer = ExtractorTrainer(model, config)
trainer.train(train_data="train.jsonl", eval_data="dev.jsonl")
```

**These are sensible initial settings, not known-good MPS results or optimal hyperparameters.** Select the best candidate on dev *task metrics* (with valid score semantics) after checkpoint screening, not on the locked test. Optionally test LoRA as a separately versioned run; do not silently change training method.

## 5. Execution sequence and minimal release gates

1. **Data rights + source identity (TR-07, TR-01):** confirm dataset hash, usage permission, prediction contract, fixed allowed columns and validated label mappings.
2. **Data factory (TR-01, TR-02):** generate train/dev datasets and restricted locked-test artifacts; freeze split registry and quality report. Training runtime has no access to raw labeled CSV or test labels.
3. **Environment smoke (TR-03, TR-08):** pin model/dependencies; verify backward pass, save/reload and prediction on a small subset; no silent CPU fallback.
4. **Train and evaluate baselines (SC-01, SC-06):** dummy, logistic, boosting, zero-shot GLiNER on the frozen dev cohort.
5. **Full GLiNER experiment (TR-04):** train with dev eval, capture run, validate outputs; choose checkpoint and threshold on dev only.
6. **Scoring audit (TR-05, SC-05):** verify positive-class semantics or restrict to label metrics; produce paired dev comparison, error analysis and uncertainty.
7. **Freeze candidate and independent evaluator (TR-06):** close knobs; copy immutable model artifacts; run one final internal test with protected truth.
8. **External evaluation later (SC-02, SC-09):** choose a new legitimate dataset with the same target meaning or document how it differs; evaluate separately and do not call internal test an external success.

### Training may begin when
- **GO for smoke:** rights permit local use, source parsing and train JSONL validate, pinned GLiNER environment imports, and no private/locked test data are supplied.
- **GO for full controlled fit:** DATA gates G1–G8 pass, train/dev only are accessible, reproducibility configuration frozen, MPS/CUDA smoke passes, and dev metrics/logging are enabled.
- **GO for reportable final comparison:** reliable output parsing/metrics, all baseline arms, model freeze and independent locked-test custody are implemented and tested.
- **GO for generalizable/production claims:** independent dataset, time-valid churn contract, calibration where required, privacy/authorization, deployment monitoring and measured decision outcomes. Not achievable from this IBM snapshot alone.

## 6. Must-change files from the starter

| File | Required edits |
|---|---|
| `common.py` | strict canonical parsing, schema validation, no silent coercion, score/label metric separation, text contracts |
| `prepare.py` | connected-group splits, full data/QC manifests, train/dev JSONL, restricted source/test labels, opaque sample-ID joins, no standard labeled test JSONL |
| `train_gliner.py` | pinned checkpoint, `eval_data`, controlled eval/checkpoint/precision settings, smoke backward and reload, full run manifest; remove automatic fallback or log exactly what executes |
| `baselines.py` | read approved train/dev view, frozen split joins; include dummy and applicable baseline ablations |
| `evaluate.py` | separate dev and locked evaluator; zero-shot G0; no proxy churn probability; stable IDs; global one-use test gate; archive full outputs |
| `requirements.txt` | lock resolved tested versions; include hardware-compatibility notes; isolate env |

## 7. Source references

- GLiNER2.5-Decide model card and classification JSONL: https://huggingface.co/fastino/GLiNER2.5-Decide
- GLiNER2 training data schemas: https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/8-train_data.md
- GLiNER2 training, configuration, LoRA and checkpoint behavior: https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/9-training.md
- DCLab `research/churn-prediction/README.md` and `CHURN_BENCHMARK.md` at reviewed commit `08c7bac`

**Audit verdict:** The original research plan is substantially complete as a **design**. What remains is a short list of consequential **implementation and verification blockers**. Fix them before relying on the training results. A successful training command alone is not evidence of an effective or safely evaluated model.
