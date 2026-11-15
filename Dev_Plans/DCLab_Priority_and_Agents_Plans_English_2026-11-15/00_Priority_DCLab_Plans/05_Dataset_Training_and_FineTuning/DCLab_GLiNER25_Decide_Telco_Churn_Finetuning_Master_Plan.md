# DCLab | GLiNER2.5-Decide for Telecom Churn

**Complete experimental design, data engineering, fine-tuning, evaluation, testing retention, and research-to-product plan**

| Field | Value |
|---|---|
| Document type | Implementation-ready research plan (not an execution report) |
| Version | 1.0 |
| Prepared | November 9, 2026 |
| Repository | `Shahriyar-Moradi/Research_Development_DCLab` |
| Repo revision reviewed | `08c7bac622db6355a59e376eabc02f9489085d08` |
| Foundation checkpoint | `fastino/GLiNER2.5-Decide`, 340M parameters, English classification |
| First source | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (IBM Telco sample, 7,043 records) |
| Primary outcome | Correctly classify the original `Churn` field as `churn` or `not_churn` |
| Status | **Planned. Full real-data fine-tuning and this protocol's final test have not been run.** |
| Excluded from scope | DCLab Hierarchical Planning V2, live retention interventions, prospective 30-day churn claims, general foundation-model pretraining |

> **Read this first:** This is a *retrospective supervised classification* experiment on a single cross-sectional IBM sample. The source label is `Churn` (Yes/No). We do **not** have a verified prediction-time cutoff or month-by-month histories. Even if the model performs very well, the result is not evidence that it forecasts churn *30 days in the future* or that a retention action prevents churn. Those require separate, temporally valid datasets and studies.

---

## 0. Executive decision and success definition

### What we will build

1. Preserve and audit the original IBM Telco dataset.
2. Freeze **train, development/validation, and locked test** customer groups, and register a **separate external test collection** to be populated from another source later.
3. Convert each eligible customer into a deterministic, task-conditioned text input and a GLiNER single-label supervision record.
4. Fine-tune GLiNER2.5-Decide, beginning with a tiny integration smoke test and then a supervised experiment on the training partition.
5. Train classical tabular baselines on **exactly the same allowed input columns and customer partitions**.
6. Evaluate and select candidates on development data. If a separate calibration cohort is needed, carve it from training data *before* final training and never reuse it to update weights for that fitted model.
7. Freeze the winning model, source revision, serializer, thresholds, scoring rules, and analysis plan **before** opening the locked test.
8. Run one authorized final evaluation and archive per-customer predictions, labels, confidence semantics, metrics, environment, and hashes.
9. Decide objectively whether this representation merits further research. **A negative result is a valid success of the experiment.**

### Hypotheses (pre-register before training)

- **H1:** Fine-tuning GLiNER improves over its own zero-shot classification baseline on held-out customers.
- **H2:** Fine-tuned GLiNER approaches a strong conventional tabular model when given the same permitted features.
- **H3 (exploratory):** Changing the order or wording of serialized features exposes measurable robustness limitations, or useful invariance.
- **H4 (later):** Multi-task training on churn plus other *independently labeled* business datasets improves held-out-task performance. **Not tested by this first Telco run.**

### Minimum technical success

- Correctly formatted JSONL accepted by the installed GLiNER training data validator.
- Complete source, partition, serializer, library, checkpoint, and model-version provenance.
- Full smoke training completes and the saved checkpoint reloads for real inference.
- Training and development evaluation complete without data contamination or unauthorized exposure of the locked test.
- All baseline and GLiNER predictions are joined to the **same split IDs**, in the original row order.

### Scientific success and decision rule

Before opening the test, specify a *primary metric based on capabilities actually available from the checkpoint*:

- **If reliable positive-class scores are exposed:** Primary metric = ROC-AUC; secondary = average precision (AP), F1, precision, recall, Brier/log loss *only for calibrated or otherwise valid class probabilities*.
- **If the API only gives a winning class and winning-class confidence:** Primary metric = balanced accuracy or F1; treat winning-class confidence as an uncalibrated model score, **not** as `P(churn)` and do not claim comparable ROC-AUC/AP or calibration until class-score semantics are verified.
- Always include the conventional model on the same split. Do **not** compare a new locked-test AUC directly against CHURN-014's older, repeatedly reused out-of-fold development AUC.

**Suggested go/no-go rubric (must be frozen using dev data only):** retain GLiNER as a research candidate if it materially improves over its untuned counterpart, has no data/policy violations, and is either competitive with the baseline within a predeclared, uncertainty-aware tolerance or shows a measured transfer/operational advantage that justifies its greater complexity. Otherwise keep the tabular baseline and archive the GLiNER result as a negative experiment. Do not select the tolerance after seeing test outcomes.

---

## 1. Repository facts and evidence, not assumptions

| Verified in repository | Location | Implication |
|---|---|---|
| Full IBM-style Telco CSV present | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | No new dataset download required for local pilot |
| Retrospective churn benchmark and baseline records | `research/churn-prediction/experiments/churn_exp/` | Reuse definitions and review benchmark failure modes |
| Best recorded *development CV* candidate, logistic with charge features, AUC ~0.8499; AP ~0.6708 | `.../results/CHURN-014/result.json` | Strong historical reference only, **not** apples-to-apples evidence for this new split |
| Excluded `customerID` from modeling | CHURN-014 recorded metadata | Preserve this policy |
| Existing GLiNER inference wrapper | `dclab_rnd/learning/gliner_run.py` | Useful to inspect output structure and library version |
| Decision-model ledger, training sets and quality controls | `dclab_rnd/ledger/`, `dclab_rnd/store/sets/decisions.py` | Integrate through existing provenance and consent contracts later |
| Existing notebook kernel is not a container sandbox | `dclab_rnd/studio/kernels.py` | Fine-tuning here is a separate, supervised local experiment, not an autonomous notebook execution lane |
| Evaluation/scoring machinery exists | `dclab_rnd/agent_eval/`, `dclab_rnd/bench/` | Future integration should reuse gates instead of creating parallel release authority |

The repository's original Telco research states that the prediction timestamp and outcome window cannot be independently established from this public file. This plan honors that limitation.

### Dataset schema

The original file has **21 columns**: 19 permitted candidate predictors, one identifier, one target. Permitted candidate predictors are:

- **Numeric:** `tenure`, `MonthlyCharges`, `TotalCharges`.
- **Categorical/binary:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.
- **Always exclude from model-visible input:** `customerID`, `Churn`, row index, split assignment, any derived label, and outcome-dependent metadata.

**No columns** named `support_tickets`, `payment_failures`, `last_login` or similar are in this dataset. Do not synthesize them and misrepresent them as historical customer facts.

### Known dataset limitations

- A single snapshot rather than verified per-customer longitudinal records.
- IBM describes the sample as a **fictional telco**; it is useful for research, not proof of live customer behavior.
- Correlations involving tenure and charges may reflect the dataset's snapshot construction. Audit prediction-time availability before applying learned patterns to production.
- Customer demographics include potentially sensitive or proxy attributes; document inclusion, compare a demographic-excluded sensitivity variant, and perform subgroup checks where sample size allows. Do not claim fairness because a field was dropped.
- Confirm the applicable **dataset-specific** usage terms before using its rows to train or distribute a commercial model. Apache licensing of the model weights or IBM example code alone does not automatically establish data rights for every use.

---

## 2. Dataset custody: preserve original, testing data, and every derivative

**This is mandatory.** Separate data *capture and retention* from *data allowed to influence learning*.

### Four collections

| Collection | Source | Permitted use | Restrictions |
|---|---|---|---|
| `TRAIN` | Frozen IBM train partition | Weight updates, fitting transformations, training augmentations | No test rows, no dev labels for weights |
| `DEV` | Frozen IBM development partition | Select hyperparameters, checkpoints, thresholds and analyses | No gradient updates on dev; repeat permitted but logged |
| `LOCKED_TEST` | Frozen IBM test partition | **One final read** after run configuration is frozen | No model selection, prompt tuning, calibration, few-shot examples or future training |
| `EXTERNAL_TEST` | **Different dataset/source, not available in this pilot** | Later out-of-domain validation | Source provenance, legitimate label mapping and leakage review required; must remain out of training |

A fifth, optional collection, `CALIB`, may be carved **only from TRAIN before fitting**. It is used to fit a probability calibrator or operating threshold and is distinct from weight-training rows. If the dataset is too small, report uncalibrated scores rather than create a misleading calibration claim.

### Recommended physical layout

```text
research/churn-prediction/experiments/gliner_decide/
├── README.md
├── PLAN.md                              # this document
├── configs/
│   ├── experiment_v1.yaml               # frozen plan, hyperparameter search space, metrics
│   └── labels_v1.json                   # task/label mapping
├── scripts/
│   ├── audit_source.py
│   ├── make_splits.py
│   ├── serialize.py
│   ├── validate_jsonl.py
│   ├── train_baselines.py
│   ├── smoke_gliner.py
│   ├── train_gliner.py
│   ├── evaluate_dev.py
│   ├── freeze_candidate.py
│   ├── evaluate_locked.py
│   └── export_results.py
├── tests/
│   ├── test_source_schema.py
│   ├── test_split_disjointness.py
│   ├── test_no_target_leakage.py
│   ├── test_serializer_determinism.py
│   ├── test_jsonl_contract.py
│   ├── test_split_order.py
│   ├── test_sealed_access.py
│   └── test_metric_semantics.py
├── artifacts/                         # immutable/reproducible local data; git-ignored
│   └── data_v1/
│       ├── source_manifest.json
│       ├── source_checks.json
│       ├── splits_manifest.json
│       ├── splits.csv                  # hashed ID + source index + assigned split; restricted
│       ├── train.jsonl                 # GLiNER examples with correct labels
│       ├── dev.jsonl                   # development examples with correct labels
│       ├── test_inputs.jsonl           # sealed input ONLY, no labels
│       ├── test_labels.jsonl           # restricted evaluator-only truth
│       ├── external_test/              # EMPTY until a second licensed dataset is accepted
│       ├── files.sha256
│       └── AUDIT.json
├── runs/                               # immutable run IDs, not overwritten
│   └── <run_id>/
│       ├── config.json
│       ├── environment.txt
│       ├── logs/
│       ├── checkpoints/
│       ├── dev_predictions.jsonl
│       └── dev_metrics.json
└── evaluations/
    └── <final_eval_id>/
        ├── approval.json
        ├── frozen_candidate.json
        ├── test_predictions.jsonl
        ├── test_metrics.json
        ├── comparison.json
        └── report.md
```

**Storage controls:** put immutable dataset snapshots in a protected, versioned local store or artifact backend. Store only a manifest and safe hashes in Git. Keep customer-level rows and labels out of Git; for any real customer dataset, use encryption, access controls, consent, retention and deletion rules. A local `.gitignore` alone is not access control.

### Keys, references and joins

Each customer needs a stable internal **evaluation ID**, generated from the dataset version and an opaque row identity (for the IBM sample, a salted hash of source row ID; do not put raw `customerID` into the input). Persist it in the restricted split index, not in model input text. The same ID links **feature text, original target, model score, prediction, errors, and audit findings**. The evaluator checks a one-to-one ID join and rejects missing, duplicated or reordered IDs.

### Test permanence and retirement

- Store the locked test permanently with a dataset version and content hash.
- A sealed test is **not a reusable tuning set**. Its final aggregate result may be reported, but its individual labels and mistakes must not feed later tuning of the same benchmark family.
- If an existing locked test is retired and reused for training, first create a new independent, genuinely unseen sealed test. Record the retirement and invalidate comparisons that would otherwise appear independent.
- Preserve the original sealed files and audit entries even after retirement according to applicable retention rules; *retire* means change in eligibility, not secretly erase scientific history.
- For the first run, the simple starter's `SEALED_TEST_OPENED.json` marker is a convenience, **not sufficient enforcement**: implement an atomic one-use approval state and evaluator-only label access. A fresh output directory must not reopen the same sealed set under another name.

---

## 3. Data preparation and frozen split protocol

### Step 3.1. Audit the source (stop on failures)

1. Read source CSV *without modifying it*; calculate file SHA-256, byte length and row count.
2. Verify 21 expected columns, exact target classes `Yes` and `No`, and expected source profile.
3. Verify `customerID` non-nullness and inspect duplicate IDs (do not silently select one outcome).
4. Convert `TotalCharges` to numeric; the original dataset has blank values associated with zero tenure. Set these to zero **only if that rule is validated**; otherwise leave missing and impute within training folds.
5. Inspect rates of empty fields, invalid categories, contradictory duplicate feature rows, numeric ranges and label prevalence.
6. Compare the resulting file hash with the recorded DCLab CHURN-014 source hash (`88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`). If different, stop and investigate version mismatch; do not assert it is the same benchmark.
7. Emit `source_manifest.json` and `source_checks.json`. Never delete the raw file.

### Step 3.2. Fix the prediction contract

```yaml
prediction_contract: telco_churn_status_v1
unit: one customer in the IBM Telco sample
target_source_column: Churn
positive_class: Yes
negative_class: No
output_labels: [not_churn, churn]
task_name: churn_status
training_task_kind: single_label_classification
prediction_horizon: unknown_from_this_file
forbidden_input_fields:
  - customerID
  - Churn
  - source_index
  - split
  - label
```

No phrasing in text or inference prompt should assert `within 30 days` or `next month` unless supported by an independently verified time contract. The IBM sample's label indicating departure in a prior period is **not the same** as a future 30-day forecasting label.

### Step 3.3. Freeze the splits **before** training

**Compatible with the provided starter v1:**

```python
from sklearn.model_selection import StratifiedGroupKFold

# groups = SHA-256 of exact, deterministic, allowed-feature serialization.
# The target and customer identifier are NOT inside this key.
outer = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)
remaining_indices, test_indices = next(
    outer.split(indices, y, groups=groups)
)

inner = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=43
)
train_local, dev_local = next(
    inner.split(
        remaining_indices,
        y[remaining_indices],
        groups=groups[remaining_indices],
    )
)
train_indices = remaining_indices[train_local]
dev_indices = remaining_indices[dev_local]
```

The intended rough ratio is **64% train, 16% dev, 20% locked test**. `StratifiedGroupKFold` is approximate and does **not** guarantee exact proportions. Save the actual counts and class distributions. If the split deviates materially from the intended balance, revise the protocol using a **new split version before training**.

**Assertions (all must pass):**

- Exactly one split assignment per original row; union equals source rows and all splits are nonempty.
- No `customerID` straddles partitions; no identical allowed-input *feature fingerprint* straddles partitions.
- Both classes are present in train/dev/test; class fractions and sample counts are documented.
- Features are generated only from original allowed fields, using frozen deterministic code.
- `train.jsonl` and `dev.jsonl` contain their labels. Locked test labels exist **only in `test_labels.jsonl`, accessible to the independent evaluator**, not as a training-style test JSONL in the default training tree.
- Source, serializer, partition assignments and every output file receive SHA-256 fingerprints.

**No true temporal split is possible from this source.** A future real-company dataset should instead use known as-of timestamps, future outcome windows, and customer/group-safe temporal evaluation.

### Step 3.4. Avoid information leaks in transformations

- No target encoding, per-class feature statistics, or data-derived text embellishments computed across the whole dataset.
- Any learned vocabulary, imputation, normalization, frequency encoding or token selection must fit on `TRAIN` only.
- The deterministic hand-written serializer can use fixed constant field names, units and numeric display formats, because these do not learn from labels.
- If computing bins or percentiles for textual features, fit bin edges on training rows and freeze them for dev/test.
- Preserve missingness explicitly. Never stringify `NaN` as an unvalidated numeric value.
- Keep class weighting and oversampling within train only, if introduced as separately preregistered variants.

---

## 4. Serialization: structured telecom features → model-readable text

### Frozen serializer contract: `telecom_text_v1`

A single field-order template, identical across training and inference:

```text
Telecom customer subscription profile:
gender: Female
SeniorCitizen: no
Partner: No
Dependents: No
tenure: 2
PhoneService: Yes
MultipleLines: No
InternetService: DSL
OnlineSecurity: Yes
OnlineBackup: Yes
DeviceProtection: No
TechSupport: No
StreamingTV: No
StreamingMovies: No
Contract: Month-to-month
PaperlessBilling: Yes
PaymentMethod: Mailed check
MonthlyCharges: 53.85
TotalCharges: 108.15
```

The values are illustrative allowed-field values. Real training examples must come **directly** from the corresponding original record. For tests, use a fixture with a known label and source row identity; never mix attributes from different customers.

**Rules:**

| Rule | Requirement |
|---|---|
| Order | Always the same 19-field order |
| Field names | Stable original identifiers or documented natural-language aliases, not mixed at random during the first run |
| Numeric values | `tenure` integer months; charges finite numbers with two decimal places; missing explicitly `unknown` |
| Categorical values | Preserve category distinctions such as `No internet service` vs `No` |
| Binary flags | Translate consistently, e.g. `SeniorCitizen: yes` / `no` |
| No target leakage | No `Churn`, `customerID`, future label, index, split name, or post-outcome explanation in input |
| Length | Measure actual tokenizer lengths for every row; fail if any record would be truncated unless a reviewed, versioned policy is in place |
| Unicode | UTF-8; normalize deterministic whitespace and line endings |
| Versions | Embed serializer version/hash in manifests, run configs and model cards |

### Exact GLiNER supervised record shape

This is the **officially documented** single-label JSONL schema:

```json
{
  "input": "Telecom customer subscription profile:\ntenure: 2\nContract: Month-to-month\nInternetService: DSL\nMonthlyCharges: 53.85\nTotalCharges: 108.15",
  "output": {
    "classifications": [
      {
        "task": "churn_status",
        "labels": ["not_churn", "churn"],
        "true_label": ["churn"]
      }
    ]
  }
}
```

This example abbreviates the text for readability. **Actual file generation includes all 19 allowed predictors.** The label comes only from mapping `Churn == "Yes"` to `churn` and `Churn == "No"` to `not_churn`. `true_label` is **supervision for training, not a field supplied at inference**. `multi_label` must be absent or false for this mutually exclusive problem.

One full JSON object per line in `train.jsonl` and `dev.jsonl`. Do not put IDs or model metadata into the GLiNER input schema unless the installed validator explicitly supports them. Keep identifiers, rights and source references in **sidecar manifests**.

### Avoid these data errors

- `true_label` generated from the model's own predicted label, rather than the historical `Churn` column.
- A textual instruction such as `The customer is likely to churn` in the *input*.
- Synthetic support tickets or failure rates added to match a proposed example.
- Adding outcomes or labels as few-shot examples from the test cohort.
- Calculating global target-frequency words such as `this cohort has a 92% churn rate` into each input.
- Rewriting dev/test text into a different field order from training without explicitly declaring a robustness evaluation.

### Validate both library and policy contracts

```python
from gliner2.training.data import TrainingDataset

train = TrainingDataset.load(".../train.jsonl")
train.validate(strict=True, raise_on_error=True)

dev = TrainingDataset.load(".../dev.jsonl")
dev.validate(strict=True, raise_on_error=True)
```

Then run custom policy validators: schema/label set, row counts, full-ID joins, no prohibited keys or target leakage, immutable split fingerprints, tokenizer-length coverage and source-license eligibility. The library's syntax validator does **not** guarantee scientific validity.

---

## 5. Model, environment and hardware plan

### Model identity

- Source: <https://huggingface.co/fastino/GLiNER2.5-Decide>
- Model type: 340M-parameter **English** classification specialist built around DeBERTa-v3-large.
- Weight license on model card: **Apache-2.0**.
- Official loader: `AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")`.
- **Pin the exact Hugging Face model revision** actually downloaded. Do not rely only on a mutable `main` reference.
- **Pin `gliner2`, `torch`, `transformers`, `tokenizers`, `numpy`, `pandas`, `scikit-learn`, and `joblib` to tested versions** after compatibility smoke tests. Record `pip freeze`, hardware and driver/OS versions.

### Training API, verified against primary documentation

```python
from gliner2 import AutoExtractor
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")
config = TrainingConfig(
    output_dir=".../runs/<run_id>/checkpoints",
    num_epochs=3,
    batch_size=2,
)
trainer = ExtractorTrainer(model, config)
result = trainer.train(
    train_data=".../artifacts/data_v1/train.jsonl",
    eval_data=".../artifacts/data_v1/dev.jsonl",
)
```

The exact `TrainingConfig` options for LoRA, precision, evaluation and checkpointing should be checked with `inspect.signature(TrainingConfig)` in the installed version, as documented names have evolved. The code above is a **reference snippet**, not a claim it has run on your device.

### Apple Silicon versus CUDA

1. Create a **separate** virtual environment; do not change the main DCLab product environment.
2. Verify Python 3.11/3.12, package imports, `torch.backends.mps.is_available()`, model download and CPU forward pass.
3. Verify **MPS inference** independently from **MPS backward/training**. Some trainer operations, fp16/bf16 settings, optimizers or dataloaders may fail or fall back to CPU.
4. Run 16–64 labeled examples for one epoch; reload output and predict a real input. Only then schedule full fitting.
5. Start with batch size 1–2, inspect GPU/CPU memory and step time. Do not claim a fixed runtime or memory requirement until measured.
6. If full-parameter tuning becomes unstable, trial **LoRA** using the official trainer's current `use_lora`, `lora_r`, `lora_alpha`, `lora_target_modules` and save semantics, with a separate experiment ID. Never quietly switch training method halfway through a run.
7. If MPS compatibility is blocked, use a **Linux CUDA runner** with the same data and frozen config, subject to dataset usage rights; archive environment differences. Never upload private customer data to a cloud GPU without authorization.

**Precision rule:** Disable unsupported mixed precision on MPS, if necessary. Avoid interpreting `PYTORCH_ENABLE_MPS_FALLBACK=1` as evidence of GPU training: it can hide CPU fallbacks and slow a run substantially.

### Minimal setup from DCLab root

```bash
cd /path/to/Research_Development_DCLab
python3.11 -m venv .venv-gliner
source .venv-gliner/bin/activate
python -m pip install --upgrade pip
python -m pip install 'gliner2[train]' pandas numpy scikit-learn joblib
python -c 'import torch; print("MPS:", torch.backends.mps.is_available()); print("CUDA:", torch.cuda.is_available())'
python -c 'from gliner2.training.trainer import TrainingConfig; import inspect; print(inspect.signature(TrainingConfig))'
```

These prepare the environment; they do not start a long paid or GPU job.

---

## 6. Experimental arms and baseline comparison

The model study must compare configurations using the **same frozen allowed features**, equivalent preprocessing visibility, and exactly the same customer partition/labels.

| ID | Arm | Fit data | What it establishes |
|---|---|---|---|
| B0 | Prior-only dummy (`DummyClassifier`) | Train | Minimum reference and class prevalence |
| B1 | Logistic regression pipeline, C=0.2 initial | Train | Strong simple supervised baseline |
| B2 | Histogram boosting or LightGBM with permitted features | Train | Nonlinear supervised baseline |
| G0 | Untuned GLiNER2.5-Decide | None | Zero-shot model with frozen label schema |
| G1 | GLiNER2.5-Decide fine-tuned | Train | Primary research hypothesis |
| G2 (optional) | Same GLiNER using reviewed LoRA variant | Train | Training efficiency and comparable quality |
| G3 (later) | Alternative, preregistered serializer version | Train | Robustness to text rendering / numeric handling |

**Historical control:** CHURN-014 reported 2×3-fold development CV, AUC ~0.8499 and AP ~0.6708. It is not a same-partition test for this study. Report it separately as research context; fit B1 and B2 again on the new train split and evaluate them on the new dev/test splits.

### Baseline fitting requirements

- Scikit-learn `Pipeline` and `ColumnTransformer`.
- Numeric median imputation and scaling fit within training only.
- Categorical imputation and one-hot encoding fit within training only.
- All models see the same 19 allowed columns; track separately if charge-derived features are added to **all** arms, since GLiNER would need an equivalent text serialization of those derived features.
- Model selection and hyperparameters are tuned on the development set, not the locked test.
- Persist `joblib` artifacts, preprocessors, source hashes and per-customer outputs. Never load untrusted pickle/joblib files.

### Predeclared GLiNER tuning budget

- **Smoke run:** 16–64 training rows, 1 epoch; purely operational.
- **Pilot G1:** 1 seed, 3 epochs, batch size 2 (or the smallest stable size); use documented optimizer defaults unless explicitly frozen.
- **Optional confirmatory G1 variant:** one alternative learning-rate or LoRA setup, chosen *before* dev analysis or added later as an openly exploratory run. Log every attempt, not only the winner.
- Track step count, memory, duration, loss curves, output validity, and model revision.
- **Do not** perform dozens of unseen tuning attempts on this 7,043-row sample and then present the best dev number as independent generalization.

---

## 7. Inference outputs, confidence semantics and calibration

The vendor documents the public inference call:

```python
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained(".../best_or_final")
result = model.classify_text(
    "Telecom customer subscription profile:\ntenure: 2\n...",
    {"churn_status": ["not_churn", "churn"]},
    include_confidence=True,
)
# Typical structure per official docs:
# {"churn_status": {"label": "churn", "confidence": 0.82}}
```

**Critical:** `confidence` here is documented for the **selected class**. This is **not enough to assume** the number is a calibrated probability that the customer will churn. Some implementations may internally normalize class scores, but that behavior must be verified from the installed checkpoint and library source, including how the negative class is handled.

### Non-negotiable evaluation fix to the starter kit

The previous `evaluate.py` converts the selected-label confidence `c` into `c` for `churn` and `1-c` for `not_churn`. This is an **unverified approximation**, unsuitable as a research-grade churn probability. Before reporting AUC/AP/Brier/log loss, either:

1. Obtain **documented, validated scores for both classes** from the current GLiNER model, and prove positive-class-score ordering and normalization semantics; or
2. Clearly distinguish an **uncalibrated ranking proxy** from probability, report the primary label-based metrics instead, and mark probability metrics unavailable.

For a valid raw positive-class score, use dev data to measure ranking. If a distinct calibration cohort exists, fit Platt scaling or another simple calibrator there (checking monotonicity and data size). Preserve the calibrator as a separate versioned artifact. Never calibrate on the locked test.

Also test returned schema variations and missing/invalid output handling before benchmarking. A malformed answer is an evaluation failure, not a reason to skip a difficult customer record.

---

## 8. Metric definitions, confidence intervals and statistical tests

All metrics have fixed conventions. Positive class is `churn` = 1. Threshold is 0.5 **only if a meaningful calibrated probability exists**; otherwise choose a threshold on development data under a defined score contract, or use GLiNER's single-label argmax for class decisions.

| Metric | Why | Requirement |
|---|---|---|
| Accuracy | Overall correct labels | Report with class prevalence; not enough alone |
| Balanced accuracy | Equal weight to both classes | Strong default when only predicted labels are available |
| Precision and recall, positive class | False alarms vs missed churners | Show both and confusion matrix |
| F1 | Trade-off at chosen threshold | Fix decision rule on dev only |
| ROC-AUC | Ranking irrespective of threshold | Only if a justified continuous churn score exists |
| Average precision | Positive-class ranking under imbalance | Same score requirement as ROC-AUC |
| Brier / log loss | Probability quality | Only with a defensible positive-class probability; otherwise N/A |
| Calibration curve / ECE | Reliability of stated probabilities | Report bin counts and uncertainty; ECE is secondary |
| Top-k recall / precision | Operational retention capacity | Use a predeclared k or fraction, e.g. 10% of customers |
| Inference latency and memory | Deployability | Include hardware, batch size and tokenizer overhead |
| Subgroup metrics | Failure analysis and harmful imbalance | Report uncertainty and suppress tiny groups; not proof of fairness |

### Paired comparison and uncertainty

- Evaluate **the same customers** for every arm.
- Bootstrap customers or independent groups as appropriate (e.g., 2,000 bootstrap replicates with fixed seed) for 95% confidence intervals of each metric and pairwise differences. If groups represent correlated customers, resample groups rather than individual rows.
- Show the difference `GLiNER - baseline` with interval; avoid making significance claims from overlapping standalone CIs.
- Log all attempted hyperparameter variants and development comparisons.
- For final locked-test analysis, execute only the **predeclared** comparisons, plots and robustness checks. Any post-hoc inspection is labeled exploratory and cannot be used as an independent confirmation.
- Original IBM data lack true time cohorts. Do not invent temporal drift tests for this file; reserve them for the external/production dataset.

### Suggested result table template

| Arm | Split | n | AUC | AP | Balanced accuracy | F1 | Brier | Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B1 logistic | dev | measured | measured | measured | measured | measured | measured | measured |
| B2 boost | dev | measured | measured | measured | measured | measured | measured | measured |
| G0 untuned GLiNER | dev | measured | if valid | if valid | measured | measured | N/A until calibrated | measured |
| G1 tuned GLiNER | dev | measured | if valid | if valid | measured | measured | N/A until calibrated | measured |
| Frozen finalists | locked test | measured once | predeclared only | predeclared only | measured once | measured once | predeclared only | measured once |

---

## 9. Locked test protocol: preserve first, open exactly once

### Before opening: formal approval checklist

- [ ] Source, splits and serialization hashes are immutable; test labels have not been exposed to training jobs.
- [ ] Training and dev workflows complete and audited.
- [ ] Exact winning model checkpoint and source revision recorded.
- [ ] Baseline artifacts frozen; every arm's preprocessing is frozen.
- [ ] Model-output parsing and score semantics tested on dev.
- [ ] Metrics, class positive direction, thresholds, comparison plan and intervals frozen.
- [ ] No further feature selection, search prompt editing, learning-rate tuning or calibration is permitted after sealed scoring.
- [ ] Evaluator is the only process allowed to read test labels. It receives **unlabeled test inputs and frozen model files**.
- [ ] A test authorization record is signed or explicitly approved and a one-time evaluation ID is allocated.

### One-time evaluation transaction

1. Atomically acquire the final evaluation privilege for `(dataset_sha, split_version, experiment_family)`; a new output directory or run ID must **not** bypass it.
2. Persist the authorization and frozen model hashes **before** releasing test labels.
3. Evaluate frozen baselines and GLiNER on the **same unlabeled test inputs**; persist per-record predictions and stable evaluation IDs.
4. Load restricted test truth only inside the independent evaluator; join one-to-one by ID and score.
5. Generate fixed metrics, intervals and reports; write immutable result and audit manifest; mark evaluated regardless of whether the model succeeded or failed.
6. Do **not** use test mistakes or subgroup findings to tune another version of this family while calling the same data sealed.

The prior starter `evaluate.py --split test --final` is useful as a demonstration but its file marker alone is insufficient for a rigorous sealed evaluation. Upgrade it as above before a real promotion claim.

### What to retain from testing

- The **original dataset** and original label column, with provenance.
- Exact split mapping plus hashes, class distributions and source indices held in restricted storage.
- Test inputs without truth (`test_inputs.jsonl`).
- True labels in a separately protected file (`test_labels.jsonl`).
- Frozen base model and fine-tuned checkpoint identities.
- Predictions, raw returned classification responses, parsing/validity flags, scoring semantics, and preprocessing versions.
- Baseline scores, fixed comparison outputs, intervals, latency, hardware and cost.
- Approval event, timestamp, reviewer, reproducible evaluator build and one-use lock.

**Never overwrite the test set or discard failed test results.** The archive is part of DCLab's scientific record, not the training pool.

---

## 10. External test data: a truly independent generalization check

A new external test dataset must be sourced from a **different data-generation process**, not a random re-split of the same IBM Telco rows. It should be registered as a **new source** with its own license and schema manifest.

### External acceptance checks

1. Document owner/publisher, license, provenance, collection period, available features and target definition.
2. Determine whether churn labels mean past churn, forward churn, account closure or something else. Do not silently map incompatible target meanings.
3. Identify what features existed at the prediction cutoff; remove post-outcome fields.
4. Map external columns into a versioned adapter with missing-feature handling; reject unsupported schema transformations rather than fabricating values.
5. Check duplicate or shared customer identity against all model-development collections; avoid source contamination.
6. Freeze and hash the external collection before scoring. No gradient updates or prompt optimization on external labels.
7. Score once as external confirmation; document domain shift and limits of transfer.

**Initially external test is `pending`, not empty-but-successful.** A model passing IBM held-out customers does not thereby pass company/domain transfer.

### Later real-company dataset design

For actual prospective churn prediction, construct one record per customer at an explicit **as-of date** with features available no later than that date, then label future churn within a fixed horizon (e.g. `(as_of, as_of+30 days]`). Add a label-maturity/exclusion period and use customer-level temporal train/dev/test; resolve return-to-service and censoring. This is a **different study** from the IBM retrospective demonstration.

---

## 11. Data quality, provenance, licensing and training eligibility

DCLab's strategic training-data goal requires more than JSONL files. Every dataset release should ship a machine-readable **dataset card**, a **quality report**, and an explicit **training eligibility decision**.

### Required manifest fields

```json
{
  "schema_version": "dclab.training_dataset_manifest.v1",
  "dataset_id": "telco_gliner_churn_v1",
  "source": {
    "name": "IBM Telco Customer Churn sample",
    "path": "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "sha256": "COMPUTE_FROM_LOCAL_BYTES",
    "license_status": "REVIEW_REQUIRED",
    "source_type": "public_sample"
  },
  "target": {
    "column": "Churn",
    "map": {"Yes": "churn", "No": "not_churn"},
    "prediction_horizon": "not_verified"
  },
  "input_contract": {
    "serializer": "telecom_text_v1",
    "excluded": ["customerID", "Churn", "source_index", "split"],
    "text_hash": "COMPUTE"
  },
  "partitions": {
    "train": {"rows": "COMPUTE", "sha256": "COMPUTE"},
    "dev": {"rows": "COMPUTE", "sha256": "COMPUTE"},
    "locked_test": {"rows": "COMPUTE", "sha256": "COMPUTE", "access": "evaluator_only"},
    "external_test": {"status": "pending"}
  },
  "training_use": "conditional_on_dataset_terms_review",
  "dataset_version": 1
}
```

The placeholder values are intentional. Never invent hashes, split counts, or legal approvals.

### Label and example quality hierarchy

1. Original observed `Churn` labels for this sample, with provenance and limitations.
2. Future real-company outcomes with verified cutoff and follow-up, when consent and licensing permit.
3. Human-reviewed corrections supported by primary records.
4. Model-generated/synthetic labels as **silver or exploratory**; not substitutes for independent truth.

Never train on commercial third-party model output solely because the workspace owner opted in. Provider terms, dataset rights and appropriate consent are separate checks. Store permissions to use a record for **evaluation**, **training**, or **neither** distinctly.

### Quality tests

- Source hash and dataset schema reproducible.
- Valid classes, positive/negative mappings and plausible class distributions.
- No duplicate feature fingerprint crossing splits.
- Identical serialization across train and inference.
- Per-example schema and label validity.
- No unknown feature fabrication, label leakage, test few-shot injection or target-derived prompts.
- Complete provenance for each training label and every transformed row.
- Exact split membership retained and independently verifiable.
- Privacy policies satisfied and test ACLs enforced.

---

## 12. Execution runbook: do this in order

The downloadable starter from the earlier conversation is a **prototype**, not yet a fully compliant implementation of this plan. Use it to accelerate steps 1–5, then implement the hardening items before any sealed evaluation.

### Stage A: Setup and source validation

```bash
# All commands from the existing DCLab repository root.
cd /path/to/Research_Development_DCLab
ls data/WA_Fn-UseC_-Telco-Customer-Churn.csv
shasum -a 256 data/WA_Fn-UseC_-Telco-Customer-Churn.csv
python3.11 -m venv .venv-gliner
source .venv-gliner/bin/activate
python -m pip install 'gliner2[train]' pandas numpy scikit-learn joblib
```

**Stop** if the file is missing, its identity is unclear, the GLiNER installation fails, the source schema has unexpected columns, or training-use terms have not been reviewed where required.

### Stage B: Prepare and inspect JSONL

```bash
# Prototype location from the earlier supplied starter kit:
python research/churn-prediction/experiments/gliner_decide/prepare.py

head -n 1 research/churn-prediction/experiments/gliner_decide/processed/train.jsonl \
  | python -m json.tool

wc -l research/churn-prediction/experiments/gliner_decide/processed/*.jsonl
```

**Before accepting the dataset:** migrate the generated `test.jsonl` into protected `test_inputs.jsonl` and `test_labels.jsonl`, maintain the stable evaluation-ID sidecar and freeze all hashes. The prototype currently emits a labeled `test.jsonl` in the same directory; that is **not** the desired final custody design.

### Stage C: Validate format and scientific checks

```bash
python - <<'PY'
from gliner2.training.data import TrainingDataset
path = 'research/churn-prediction/experiments/gliner_decide/processed/train.jsonl'
dataset = TrainingDataset.load(path)
dataset.validate(strict=True, raise_on_error=True)
print('GLiNER training JSONL validated')
PY
```

Run custom tests for forbidden strings/keys, group disjointness, class balance, stable IDs and tokenizer length. A successful `TrainingDataset.validate()` **alone** is not enough.

### Stage D: New same-partition baselines

```bash
python research/churn-prediction/experiments/gliner_decide/baselines.py
```

Save dev scores plus pipelines. If dataset preparation changes, invalidate and refit all arms on the new frozen partition.

### Stage E: Model environment smoke

```bash
python research/churn-prediction/experiments/gliner_decide/train_gliner.py \
  --smoke \
  --out research/churn-prediction/experiments/gliner_decide/runs/gliner_smoke
```

Success requires a real forward/backward update, clean completion, **reloading** the saved checkpoint and a valid prediction. Don't assume a printed completion line proves the artifact can be served. Verify its exact `best`/`final` output path for the installed library version.

### Stage F: Controlled full run

```bash
python research/churn-prediction/experiments/gliner_decide/train_gliner.py \
  --epochs 3 --batch-size 2
```

Use the versioned run directory, training config, train/dev file hashes, model commit/revision and environment manifest. Check that evaluation data were supplied where best-checkpoint selection requires them; the prototype train script currently calls the trainer with **train only**, so add reviewed dev evaluation support.

### Stage G: Development comparison

```bash
python research/churn-prediction/experiments/gliner_decide/evaluate.py
```

**Before adopting this result as scientific evidence:** fix positive-class score extraction/semantics, stable ID alignment and metric gating described in §§7–8. The prototype's `1-winning_confidence` proxy must not be reported as calibrated churn probability.

### Stage H: Freeze and final test

Implement an evaluator-only entrypoint, a run approval and an atomic one-time dataset-family lock. Only after all checks, execute the final comparison and archive its result. The earlier command below is **not acceptable as the sole release gate** without these protections:

```bash
# Prototype only: NOT the hardened final evaluator
python research/churn-prediction/experiments/gliner_decide/evaluate.py \
  --split test --final
```

**No official locked-test evaluation should be attempted until the hardened evaluator passes its access-control and alignment tests.**

---

## 13. Critical fixes to the existing starter kit

These are concrete action items identified by inspecting the previously generated starter files. Do them before promoting a result:

| ID | File | Current issue | Mandatory change |
|---|---|---|---|
| FIX-01 | `prepare.py` | Writes labeled `test.jsonl` beside training files | Write unlabeled locked input and restricted label sidecar; use immutable split version |
| FIX-02 | `evaluate.py` | Treats `1 - confidence_of_winner` as positive probability | Validate score semantics and retrieve class scores, or use label-only metrics |
| FIX-03 | `evaluate.py` | Joins JSONL examples and tabular records mostly by order/length | Use stable, unique evaluation IDs and hash-verified joins |
| FIX-04 | `evaluate.py` | Final test protected by local output-file marker | Atomic one-time permission keyed by test dataset + experiment family; evaluator-only truth |
| FIX-05 | `train_gliner.py` | Trains with no explicit `eval_data` in trainer call | Add versioned dev evaluation and documented checkpoint selection, no test read |
| FIX-06 | `train_gliner.py` | Mutable base checkpoint reference, minimal run manifest | Pin exact model revision, environment, training data and config hashes |
| FIX-07 | `common.py` | Minimal metric guard and no independent score semantics test | Reject invalid probabilities, missing outputs and invalid scores; report N/A where needed |
| FIX-08 | all scripts | No explicit source dataset legal-status gate | Add provenance/usage-policy checks before export, training or external upload |
| FIX-09 | all scripts | First-run permissions are only local file layout | Restrict/test read capability, keep sealed labels out of training/search indexes |
| FIX-10 | experiments | Historical CHURN-014 uses different folds | Refit baseline on identical split, keep historical figure contextual |
| FIX-11 | evaluation | No true external-domain set | Add external dataset registry and pending status; no invented external score |
| FIX-12 | packaging | No full unit/integration and failure-injection suite | Add determinism, split, leak, validator, checkpoint, test-access and metric tests |

The starter was previously tested on a **small synthetic fixture for preparation/baselines only**. That does not establish that the 340M trainer or the full IBM dataset experiment has been executed successfully.

---

## 14. Test suite and failure injection

### Unit tests

1. `test_mapping`: target `Yes` → `churn`; `No` → `not_churn`; invalid values rejected.
2. `test_serializer`: same input row gives byte-identical text, independent of process/run.
3. `test_forbidden_features`: identifier, target, source index, split and label never included.
4. `test_jsonl`: classification schema, exact task and labels, one true label, no duplicate JSON keys.
5. `test_numeric`: missing and zero-tenure `TotalCharges`, numeric formatting, unknown values.
6. `test_group_split`: exact group disjointness and split completeness.
7. `test_split_freeze`: reruns with same source/serializer/seeds produce exactly the same hashes.
8. `test_token_budget`: no unintended truncation.
9. `test_baselines`: model preprocessing fitted on train only; feature names match.
10. `test_output_parse`: valid/malformed GLiNER outputs handled deterministically.
11. `test_probability`: confidence not mislabeled as churn probability; invalid scores rejected.
12. `test_eval_join`: shuffled order, missing predictions, duplicate IDs all cause hard failures.

### Access and recovery tests

1. Training process trying to open `test_labels.jsonl` is denied.
2. Final-test command without approval is denied and records no test metrics.
3. Re-running final test under a different output directory is denied for the same dataset/family.
4. Process crash during final scoring cannot permit an unrecorded second release of labels; recovery is audited.
5. Corrupt source hash or modified serializer invalidates the frozen model/evaluation pairing.
6. A failed GLiNER checkpoint reload is a failed run, not a test result.
7. A single malformed prediction fails or explicitly counts as an error according to the frozen scoring protocol.
8. No test labels become searchable in DCLab memory, vector stores or training collection pipelines.
9. Dataset consent/license failure blocks prohibited training exports.

### Scientific checks

- Compare dev metrics with and without potentially sensitive/proxy attributes.
- Test invariance under harmless formatting changes on **dev only**.
- Test stress on missing allowed features on **dev only**; record changes without calling it temporal shift.
- Audit cases where tabular models and GLiNER disagree (dev only before the test is opened).
- Record sample counts and confidence intervals in all subgroup comparisons.

---

## 15. Phased engineering tasks for Codex or a human engineer

Each phase is atomic. Do not proceed automatically to the next phase without its exit checks.

| Phase | Work packages | Deliverable | Exit gate |
|---|---|---|---|
| P0: Protocol | P0.1 benchmark definition; P0.2 source and license review; P0.3 dataset policy | `experiment_v1.yaml`, model card draft | Scope, metrics and rights agreed; no test touched |
| P1: Data Factory | P1.1 source audit; P1.2 split freeze; P1.3 serializer; P1.4 JSONL; P1.5 test custody | Versioned data artifacts + manifest | 100% schema validity, no split/label leak, stable hashes |
| P2: Baselines | P2.1 dummy; P2.2 logistic; P2.3 boosted model; P2.4 dev report | Same-partition baseline artifacts | Comparable inputs, proper train-only pipelines |
| P3: GLiNER | P3.1 dependency pinning; P3.2 CPU/MPS smoke; P3.3 training; P3.4 checkpoint reload | G0 and G1 versioned run folders | End-to-end train/infer works, reproducible metadata |
| P4: Scoring | P4.1 output semantics; P4.2 dev comparison; P4.3 calibration decision; P4.4 freeze winner | Pre-registered final candidate | No test leakage; fixed scoring rule |
| P5: Locked evaluation | P5.1 evaluator ACL; P5.2 atomic access; P5.3 one-time scoring; P5.4 immutable report | Complete final test evidence | One read, exact joins, saved metrics and audit |
| P6: Independent validation | P6.1 new dataset source; P6.2 label mapping; P6.3 new locked eval | External benchmark | Source rights and outcome definition compatible |
| P7: DCLab integration | P7.1 result/model card; P7.2 shadow serving; P7.3 governed decision API | Optional product candidate | Must beat agreed baseline and safety/latency criteria |

### Dependencies

```text
Source/license acceptance ──> Data audit ──> Frozen splits ──> JSONL validation
                                                │
                         ┌──────────────────────┴─────────────────────┐
                         ▼                                            ▼
                 Tabular baselines                            GLiNER smoke + fit
                         └──────────────────────┬─────────────────────┘
                                                ▼
                                      Development comparison
                                                ▼
                                      Candidate freeze
                                                ▼
                                Authorized one-time locked test
                                                ▼
                                   Archive and go/no-go decision
                                                ▼
                           External-domain validation (later)
```

**Parallelization:** baseline development and GLiNER fitting may run in parallel once the split is frozen. The final evaluator is independent and must not be imported into training processes.

### Initial Codex task contract

> Implement **P1 only** under `research/churn-prediction/experiments/gliner_decide`, reusing the previously supplied starter where safe. Do not train a model and do not evaluate the locked test. Freeze the source schema, source hash, deterministic split assignments, strict GLiNER JSONL export, `train.jsonl` and `dev.jsonl`, restricted `test_inputs.jsonl` and `test_labels.jsonl`, stable evaluation IDs, dataset manifest and SHA-256 file inventory. Reject target/id leakage, mixed groups, invalid labels and nonreproducible serialization. Write and run tests for all acceptance gates. Preserve the original CSV. Finish with a list of exact generated files, sample counts, hashes, passed tests and any blocker; do not continue to P2 without a separate instruction.

This is deliberately a bounded first task: it prevents a coding agent from touching the test or starting an expensive training run before the data contract is verified.

---

## 16. Model deployment, long-term data collection, and future foundation-model research

**Do not promote a checkpoint because it fits IBM Telco.** The following belong after independent confirmation:

1. Register GLiNER as a `decision_model` behind DCLab's existing model gateway and versioned model card; retain policy and human approval gates.
2. Add an optional shadow predictor returning `label`, model identity, confidence *as defined by the API*, and validation status. Until calibrated, do **not** expose a numerical churn-risk probability as if it were trustworthy.
3. Monitor latency, memory, error rate, score distributions and input-schema drift. Real outcome monitoring requires explicit identity-safe joins and delayed labels.
4. Collect eligible real-company supervision through DCLab's ledger and data factory, with verifiable time cutoffs and licensing/consent.
5. Expand to at least two additional independent task families (e.g. fraud and conversion), **each with real labels, its own held-out data and documented class meaning**.
6. Compare single-task GLiNER with multi-task training and conventional per-task tabular models. Evaluate entirely held-out domains to prove transfer rather than reusing training-style datasets.
7. Only if transfer and performance justify it, research a hybrid numeric/tabular encoder plus task-conditioned label representation. Dynamic-label classification alone does not imply general tabular in-context learning.

### Retain data for future training **without contaminating tests**

- Maintain an append-only, versioned **training candidate pool** built from consented, license-eligible examples.
- Keep **dev and sealed test IDs and content fingerprints** in an exclusion registry used by every dataset builder, embedding-retrieval system and fine-tuning job.
- Store model predictions and mistakes as research artifacts, with a separate training-eligibility flag. A failed prediction on a sealed row does **not** grant permission to train on its label.
- Maintain source group IDs (customer, company, dataset, time cohort) so future train/dev/test partitions do not mix correlated examples.
- Record model version, dataset revision and external evaluation for every future release.

---

## 17. Risk register and stop conditions

| Risk | Severity | Mitigation / stop condition |
|---|---|---|
| Label or ID leaks into text | Critical | Fail preprocessing and invalidate all derived artifacts |
| Training can read sealed labels | Critical | Stop work; redesign protected data store and reissue fresh test if exposed |
| Repeated test peeking | Critical | Seal per dataset/family, not per local output directory |
| Wrong positive-class confidence semantics | High | Only label-based metrics until valid class scores verified |
| Overfitting tiny, single-source sample | High | Small preregistered search, independent external validation |
| Outcome horizon misrepresented | High | Label as retrospective classification; no prospective claim |
| Missing source rights/consent | High | Use permitted local/evaluation context only until reviewed; do not distribute trained weights |
| Mac MPS training incompatibility | Medium | First smoke; documented precision/LoRA fallback; controlled CUDA alternative |
| Numeric representation harms generalization | Medium | Track baseline gap, test numerical and order robustness on dev |
| Customer subgroup disparities | Medium/High | Segment evaluation with uncertainty, review business and legal use restrictions |
| Artifact cannot be reloaded | High | Required end-to-end checkpoint reload smoke |
| Model beats baseline only on previously explored dev set | High | No product promotion without untouched test plus external confirmation |
| Data stored in Git or leaked externally | High | Git-ignored local artifacts, explicit access controls, no unauthorized uploads |

### Hard stopping conditions

Stop and investigate if any of the following occurs: schema mismatch, source hash mismatch, unknown label values, split overlap, prohibited training use, missing artifact provenance, unauthorized test read, silent tokenizer truncation, invalid model-output parsing, unrecoverable checkpoint state, or any attempt to relax DCLab scientific policy to improve the metric.

---

## 18. Deliverables and final Definition of Done

**The experimental program is fully complete only when all applicable items are satisfied:**

- [ ] Approved objective: `Churn` retrospective classification, not invented forward-horizon label.
- [ ] Data source rights reviewed and recorded; original source preserved.
- [ ] Source hash, 21-column schema, 7,043-row profile verified from local bytes.
- [ ] Stable train/dev/locked-test assignments and fingerprints persisted; external test tracked as pending.
- [ ] Sealed test labels separated and inaccessible to training code; one-use evaluator implemented and tested.
- [ ] Deterministic 19-feature serializer, no target/id in inputs, full tokenizer coverage.
- [ ] Official GLiNER classification JSONL schema validated and independently tested.
- [ ] Baselines trained and measured on the exact same partitions.
- [ ] Untuned GLiNER measured on dev; training smoke complete; full model fine-tuned.
- [ ] Checkpoint reloaded, model/library/OS and dataset versions pinned.
- [ ] Positive-score confidence semantics verified or unsupported metrics deliberately marked N/A.
- [ ] Development comparisons, uncertainty intervals, failures and resource consumption recorded.
- [ ] Exact candidate and scoring protocol frozen before the single locked test.
- [ ] Locked test evaluated once, independently, with audit, per-record scores and immutable final report.
- [ ] Honest result against conventional baselines, including negative or inconclusive outcomes.
- [ ] External-domain validation remains an explicit separate gate; product deployment not implied by IBM test.
- [ ] Training eligibility, provenance, consent/licensing and test-exclusion registry preserved for future model versions.

### Suggested experiment report headings

1. Objective and prediction-time limitations.
2. Data source and license, fingerprint and split table.
3. Serialization, token budget, missingness and leakage checks.
4. Training configuration, hardware, steps, runtime, and checkpoint references.
5. Baselines and dev results, all variants attempted.
6. Frozen final candidate and locked-test authorization.
7. Locked-test table with uncertainty and predeclared pairwise differences.
8. Calibration/score-semantics statement.
9. Subgroup, numerical robustness and qualitative error analysis (dev unless explicitly predeclared for final).
10. External validation status, limitations, decision and reproducibility commands.

---

## 19. Canonical primary sources and DCLab repository references

**Official GLiNER2.5 and GLiNER2 documentation (checked November 9, 2026)**

1. Fastino GLiNER2.5-Decide model card, official label-conditioned classification and JSONL fine-tuning format: <https://huggingface.co/fastino/GLiNER2.5-Decide>.
2. GLiNER2 GitHub repository, loaders, classification and licensing: <https://github.com/fastino-ai/GLiNER2>.
3. GLiNER2 training data formats, classification records and strict validation: <https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/8-train_data.md>.
4. GLiNER2 trainer configuration, evaluation data, checkpointing and LoRA: <https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/9-training.md>.
5. GLiNER2 classification output with confidence examples: <https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/1-classification.md>.
6. IBM Telco data description (fictional sample, departure in last month): <https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn>.
7. Public IBM churn code pattern and sample CSV: <https://github.com/IBM/telco-customer-churn-on-icp4d>.

**DCLab files used as repository evidence (commit `08c7bac...`)**

- `research/churn-prediction/README.md`
- `research/churn-prediction/experiments/churn_exp/CHURN_BENCHMARK.md`
- `research/churn-prediction/experiments/churn_exp/results/CHURN-014/recipe.json`
- `research/churn-prediction/experiments/churn_exp/results/CHURN-014/result.json`
- `dclab_rnd/learning/gliner_run.py`
- `dclab_rnd/studio/kernels.py`
- `dclab_rnd/ledger/`
- `dclab_rnd/store/sets/decisions.py`
- `dclab_rnd/agent_eval/`

**Starter-kit audit source:** Earlier conversation artifact `DCLab_GLiNER_Telco_Starter.zip`, containing `common.py`, `prepare.py`, `baselines.py`, `train_gliner.py` and `evaluate.py`. This audit covers the previously generated files; the hardened scripts in this plan are **specified work**, not yet implemented.

---

## Final recommendation

**Implement P1 (data factory and locked-test custody) before spending compute on the full model.** Next run B1/B2 and a GLiNER smoke test; then execute one controlled fine-tune, analyze development data, freeze the winning candidate, and open the locked test once. Store everything with hashes and rights, but never silently turn test labels into training examples.

The experiment answers a useful and falsifiable question: *Can a 340M label-conditioned text encoder learn telecom customer churn from serialized structured features, and is it competitive enough with simpler tabular models to warrant a broader decision-foundation-model program?* A rigorous answer matters more than a high score from a contaminated test.
