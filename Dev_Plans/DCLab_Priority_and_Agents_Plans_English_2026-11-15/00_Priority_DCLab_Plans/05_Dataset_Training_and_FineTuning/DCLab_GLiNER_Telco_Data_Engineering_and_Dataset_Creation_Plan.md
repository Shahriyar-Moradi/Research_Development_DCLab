# DCLab | Telecom-to-GLiNER Data Engineering and Fine-Tuning Dataset Factory

**Detailed implementation specification, v1.0 | 9 November 2026**  
**Status:** Design and implementation plan. Data-pipeline hardening described here has **not** been executed on the full IBM CSV in this session.  
**Parent plan:** `DCLab_GLiNER25_Decide_Telco_Churn_Finetuning_Master_Plan.md`  
**Repository:** `Shahriyar-Moradi/Research_Development_DCLab` at reviewed commit `08c7bac622db6355a59e376eabc02f9489085d08`  
**Model:** `fastino/GLiNER2.5-Decide` (340M parameters); pin its exact revision at execution time.

> **Purpose.** Produce *scientifically valid, reproducible, well-documented, versioned, legally eligible* fine-tuning data for supervised telecom churn classification; reserve independent test data permanently, and block training unless every data gate passes. This document specifies **how data are collected, checked, transformed, split, serialized, validated, stored, and consumed**. It is deliberately more precise than the original starter code.

## 0. The target result and the non-negotiable rules

### Final products from one dataset release

1. **Immutable source snapshot**, SHA-256 and source provenance; never overwrite source bytes.
2. **Audited canonical customer table**, keeping exact source links outside model inputs, with row-level transformation decisions.
3. **Frozen split registry:** train, development validation, protected locked test. An **external test registry** exists but is `pending` until a truly independent dataset is obtained. Optionally, a calibration subset carved out of train *before* training.
4. **Fine-tuning JSONL:** labeled `train.jsonl` and `dev.jsonl` in GLiNER's official classification format.
5. **Sealed test artifacts:** `test_inputs.jsonl` **without truth**, plus restricted `test_labels.jsonl` and a separate protected join index; no train process may access test truth.
6. **Reproducible baseline input:** the exact allowed-feature table and split IDs for Logistic Regression / LightGBM / other classical models, with each model fitting any learned preprocessing on train only.
7. **Dataset card, manifest, per-file hashes, quality report, code/config revision, license/consent decision**, plus an auditable list of exclusions/quarantines.
8. **Gates and tests** ensuring dataset quality, no leakage, no split crossover, and GLiNER parser compatibility.

### Rules that cannot be relaxed to get a higher score

- This IBM sample has a source column named `Churn` (Yes/No), **not** a verified *churn-within-next-30-days* label. Use task identifier **`churn_status`**, not `churn_prediction_30d`.
- The user-facing model text may contain **only fields available in the source and approved as features**. Never manufacture support tickets, payment failures, last login, later cancellations, or retention outcomes.
- `Churn` and `customerID` are never model-visible input fields. Nor are source index, split ID, target codes, evaluation score, or any post-outcome information.
- All fitted preprocessing, class weights, normalization and learned vocabularies use **TRAIN** only; dev and sealed test cannot influence their fitted values.
- Dev labels support model selection but **never parameter updates**. Locked and external test labels support **independent evaluation only**.
- Permanently archive test membership and hashes. If a test is ever retired and legitimately reused for training, it ceases to be a sealed test; create a new independent test **before** doing so and document the retirement.
- Model-weight license, data-source license and third-party output-training rights are different questions. Keep separate eligibility flags; fail closed on uncertain rights for distribution or external training.

## 1. Source contract and data audit (DATA-01)

### 1.1 Source of truth

- Path: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` in the DCLab repository.
- Expected shape per existing research: **7,043 rows, 21 source columns** = 19 allowed predictor candidates + `customerID` + `Churn`.
- Existing DCLab CHURN-014 records a source SHA-256:
  `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`.
- **Check this against the bytes present on the machine.** A mismatch is a blocker until provenance is reviewed, not permission to silently reset the expected hash.
- The original IBM sample is cross-sectional. There is no verified event-time cutoff or prospective prediction window.

### 1.2 Exact allowed predictor list (19)

| Feature | Raw category | Canonical display | Validation |
|---|---|---|---|
| `gender` | categorical | preserve original category | reject unexpected tokens until reviewed |
| `SeniorCitizen` | binary numeric category | `yes` for `1`, `no` for `0` | any other value quarantined |
| `Partner` | categorical | `Yes` / `No` | preserve domain |
| `Dependents` | categorical | `Yes` / `No` | preserve domain |
| `tenure` | integer numeric | months, integer string | nonnegative integral finite value |
| `PhoneService` | categorical | preserve category | preserve `No` |
| `MultipleLines` | categorical | preserve category | keep `No phone service` distinct from `No` |
| `InternetService` | categorical | preserve category | `DSL`, `Fiber optic`, `No` remain distinct |
| `OnlineSecurity` | categorical | preserve category | keep `No internet service` distinct from `No` |
| `OnlineBackup` | categorical | preserve category | same distinction |
| `DeviceProtection` | categorical | preserve category | same distinction |
| `TechSupport` | categorical | preserve category | same distinction |
| `StreamingTV` | categorical | preserve category | same distinction |
| `StreamingMovies` | categorical | preserve category | same distinction |
| `Contract` | categorical | preserve category | keep `Month-to-month`, `One year`, `Two year` distinct |
| `PaperlessBilling` | categorical | preserve category | preserve Yes/No |
| `PaymentMethod` | categorical | preserve category | preserve exact recorded method |
| `MonthlyCharges` | decimal numeric | fixed two decimals, original unit | finite; no silent coercion to 0 |
| `TotalCharges` | decimal numeric | fixed two decimals, original unit | special documented blank-zero-tenure rule below |

**Forbidden:** `customerID`, `Churn`, output label, row number, split, source fingerprint, performance score. The target appears only in the protected label record and training supervision output.

### 1.3 Source read and schema validation

1. Read CSV as source text first so blanks and irregular tokens are not silently converted. Keep raw source bytes unchanged and record CSV parsing options.
2. Assert exactly the required fields and stable, unique header names. Unexpected columns must trigger schema review, even if the model would ignore them.
3. Verify every row has an identity and label; measure missing/duplicate identities.
4. Check `Churn` is exactly `Yes` or `No` after narrowly defined whitespace normalization. Any other value is a hard error or an explicitly quarantined row; never guess a label.
5. Record observed categories, frequencies, invalid categories, numeric ranges, nulls and suspicious values. Review category changes before accepting a new source version.
6. Check duplicates and contradictory labels: identical permitted features may legitimately belong to different customers with different outcomes. **Do not silently deduplicate or flip labels.** Keep the records, flag contradictory same-feature groups, and ensure they do not cross partitions.
7. Review demographic features (`gender`, `SeniorCitizen`, etc.). For this controlled benchmark they may remain with documented approval; plan a **demographic-excluded** sensitivity variant later. It is not proof of fairness or safe production eligibility.
8. Record source dataset license/use permission separately from checkpoint license. Training or publishing a derived checkpoint needs the relevant permissions.

### 1.4 Numeric parsing policy

- `tenure`: parse as integer; reject nonintegral, nonfinite or negative values; do not truncate floats to integers.
- `MonthlyCharges` and `TotalCharges`: parse with an explicit decimal parser; reject strings that are neither a valid decimal nor an approved missing token. Do not use `errors="coerce"` without retaining error provenance.
- **Special known IBM case:** blank `TotalCharges` when `tenure == 0`: candidate rule is `0.00` after an audit confirms that every affected case matches the source's zero-tenure interpretation. Record counts and affected record IDs; if not confirmed, represent `missing` and compare via a separately versioned policy.
- Blank `TotalCharges` with positive tenure, invalid `MonthlyCharges`, or other unexpected numeric states must be reported and reviewed. No global median imputation before splitting.
- Canonical text formatting must be deterministic (`0`, `29.85`, `108.15`, `unknown`); never emit `nan`, `inf`, or Python float representations with machine-dependent notation.
- Do not infer a monetary currency that is not documented. Use the source feature name, not `$` or `AED` unless justified by the source.

### 1.5 Audit outputs

`SOURCE.json` (source identity and rights), `SOURCE_AUDIT.json` (row count, schema, missingness, label prevalence, duplicate analysis), `ROW_DECISIONS.jsonl` (per-row parse/quarantine/normalization provenance, **restricted**), and source SHA-256. Each can be recreated from the frozen source and code revision.

**Gate G1:** no unreviewed schema/label/parse anomalies; source hash and source-use policy are registered. **Do not produce training JSONL before G1.**

---

## 2. Prediction and label contract (DATA-02)

```yaml
contract_id: telco_churn_status_v1
unit_of_prediction: one customer snapshot
source_label: Churn
positive_source_label: 'Yes'
negative_source_label: 'No'
positive_model_label: churn
negative_model_label: not_churn
model_task: churn_status
multi_label: false
horizon: unknown_from_this_sample
allowed_source_features: 19_fixed_features_in_section_1
forbidden: [customerID, Churn, source_index, split, actual_outcome_after_snapshot]
```

- Map `Yes → churn` and `No → not_churn` in one pure, versioned function. Unknown label = **error**, not `not_churn`.
- Fix label order once: `["not_churn", "churn"]`. Preserve everywhere, including training JSONL, inference schema, decoder, confusion matrices, and model cards.
- The source label represents churn as defined by the IBM sample. Do not claim a model score is an actionable future retention-risk estimate.
- For the **future genuine 30-day task**, rebuild data from customer snapshots (`as_of`), label only churn events within `(as_of, as_of+30 days]`, ensure observation windows mature, handle right censoring and reactivation, then use temporal/customer-group isolation. That is a **new dataset and new contract**, not a rename of this label.

**Gate G2:** target mapping and interpretation are tested and signed off.

---

## 3. Customer identity, grouping, and split design (DATA-03)

### 3.1 Why splits must happen before preprocessing learned from data

We want an honest answer to “will the model work on *new customers*?” If identical feature vectors or the same customer are in both train and test, the model could memorize. All data-derived operations must be fitted after partitioning.

### 3.2 Stable records and group identity

- Preserve `source_index` and `customerID` only in a restricted **source-to-evaluation join table**, never in GLiNER text.
- Generate an opaque `evaluation_id` with a dataset-specific keyed hash/HMAC over canonical identity plus source version (for public samples a stable opaque key is sufficient; for real data use privacy-preserving HMAC).
- Generate `feature_fingerprint = SHA256(canonical allowed-feature tuple)` **excluding target, customerID, split and anything learned from labels**.
- Group all rows with identical feature fingerprints together. Also group duplicate `customerID` values together. If these grouping edges connect, compute their connected components to ensure neither constraint can be violated.
- Do not assume every duplicate feature vector is an invalid example. Keep contradictory labels in the **same** group; report counts to show irreducible ambiguity.

### 3.3 Freeze partitions once

Use deterministic **stratified, group-aware** splitting based on the original labels and pre-approved groups. For the initial pilot, the specification is a roughly **64% TRAIN, 16% DEV, 20% LOCKED_TEST** split using `StratifiedGroupKFold` in two stages (5 folds, seeds 42 and 43). This is approximate; stratification can be imperfect because entire groups must remain together.

Pseudo-implementation:

```python
outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
remaining_ix, test_ix = next(outer.split(indices, target, groups=group_ids))
inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
train_local, dev_local = next(inner.split(
    remaining_ix, target[remaining_ix], groups=group_ids[remaining_ix]
))
train_ix = remaining_ix[train_local]
dev_ix = remaining_ix[dev_local]
```

**Never rely on the same seed alone to preserve a split.** Serialize exact membership and version it. `split_v1` must be immutable. A change in source, grouping algorithm, seed, or target mapping creates `split_v2` and invalidates old apples-to-apples comparisons.

### 3.4 Split checks

- Every accepted source row belongs to exactly one partition; no unexpected row lost or duplicated.
- No duplicate customer ID, identical feature fingerprint, or connected identity group crosses a partition.
- TRAIN, DEV, TEST are nonempty and each contains both classes. Report counts, group counts, churn prevalence and ratio deviations from planned values.
- All model arms and text variants use **identical split membership**.
- Fit any numeric statistics, percentile bins or encoders on TRAIN only; test must not be opened for data exploration beyond strictly non-outcome metadata needed for custody and integrity.
- If dev/test quality reveals a group-imbalance problem **before any training or exposure**, create a reviewed new split version. After training starts, do not reassign rows just to improve a score.

### 3.5 Optional calibration cohort

If GLiNER eventually exposes a defensible raw score for the **positive class**, reserve an optional `CALIB` subset from TRAIN before fitting, with group safety. Fit any probability calibrator there and never use these rows for model weight updates in that fitted run. If calibration is not supportable, report **classification metrics only** and explicitly mark Brier/log loss unavailable for GLiNER.

**Gate G3:** split assignments, group disjointness, class balance, and partition hashes pass. No model training until frozen.

---

## 4. Canonical feature preprocessing and text serialization (DATA-04)

### 4.1 Policy: deterministic transforms first, learned transforms later

Use a single pure function `serialize_customer(canonical_row, serializer_version)` to build the text. Each invocation must produce **byte-identical UTF-8 output** from the same canonical features, regardless of Python process, machine or run order.

**First experiment uses raw numerical values with fixed formatting.** Do not introduce learned numeric bins, z-scores, quantiles, or synthetic features in version 1. Those are separate, controlled variants only after the baseline is complete.

### 4.2 Exact v1 text representation

Use one fixed field order and fixed names:

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

This is a **format example**, not a claim all those values belong to a particular real row. Every actual training text must be made from its own row only.

### 4.3 Serialization details

| Concern | Exact rule |
|---|---|
| Header | Fixed, neutral; no answer hint such as “high churn risk” |
| Feature order | The exact 19-field order in Section 1 |
| Whitespace | Normalize input categories using a documented narrow policy; output LF (`\n`), no trailing spaces |
| Numeric precision | `tenure` as integer; charges to 2 decimals, with explicit rounding policy |
| Missing | Use literal `unknown` only for an approved, audited missing field; never silently substitute a target-correlated value |
| Source categorical values | Preserve valid categories, including “No internet service” versus “No” |
| Binary mapping | `SeniorCitizen: yes/no` from 1/0; all other fields retain source conventions |
| Escaping | Escape/control newline, quote or control characters from any arbitrary text; no user content may inject a new feature line |
| Size | Enforce token budget on **text plus task schema/labels**, not just the text string |
| Pretraining text | No customer ID, churn label, label prevalence, index, split name or predicted score |
| Metadata | Stored in sidecar only; never appended to model-visible text |

### 4.4 Token budget

- Load the **actual pinned model tokenizer** and assemble an inference/training sample in the form consumed by GLiNER (text + classification schema).
- Compute token-length distribution (min, p50, p95, max) for every partition, and the number of rows exceeding the model/trainer limit.
- **Zero silent truncation.** If any row would truncate, first investigate why, then create a new `serializer_v2` (e.g. shortened field aliases) and **rebuild all partitions** from the same frozen IDs.
- Keep a **text SHA-256 per sample** and the serializer source/hash in every artifact manifest.

### 4.5 Leakage unit tests (structural, not naive substring checks)

- Build a deliberate row with fake `customerID` and a `Churn` value, serialize, and verify that neither field value can be introduced into text by the serializer.
- Verify that adding/changing the target or the raw customer identifier alone **cannot change the serialized text**.
- Verify changing an approved predictor does change its field's text.
- Verify a categorical value containing a newline or `Churn:` cannot inject extra lines or a plausible target field.
- Verify same canonical row gets the same fingerprint and text on repeated runs.
- Verify actual text includes exactly 19 approved features (by parsed field names), not merely “no forbidden word detected.”

**Gate G4:** deterministic serializer + policy leak checks + tokenizer coverage all pass.

---

## 5. Generate GLiNER training JSONL with trusted supervision (DATA-05)

### 5.1 Official schema

For this **single-label**, mutually exclusive classification problem, each training example is exactly:

```json
{
  "input": "Telecom customer subscription profile:\ntenure: 2\nContract: Month-to-month\nMonthlyCharges: 53.85\nTotalCharges: 108.15",
  "output": {
    "classifications": [
      {
        "task": "churn_status",
        "labels": ["not_churn", "churn"],
        "multi_label": false,
        "true_label": ["churn"]
      }
    ]
  }
}
```

The short text above is **illustrative**. Every generated example must contain **all 19 approved fields**.

- Positive original `Churn=Yes` → `true_label:["churn"]`.
- Negative original `Churn=No` → `true_label:["not_churn"]`.
- `task`, `labels`, `multi_label`, and any optional task prompt must match the documented inference-time schema. For v1, omit the optional `prompt` to minimize variation.
- JSONL is **one complete JSON object per physical line**, UTF-8, with escaped newlines in the JSON string and no trailing commas.
- No record ID, source path, split ID or other keys in `input` or `output`; put traceability in a protected **line-index-to-evaluation-ID** sidecar with file/content hashes.
- Preserve the exact original source label in restricted source labels, not in model-visible input.

The format is based on the official model card and GLiNER2 `tutorial/8-train_data.md` (primary sources listed in §14). The official guide accepts `true_label` as a list/string; one-element list is our fixed house convention.

### 5.2 Files produced

| File | Row structure | Allowed readers |
|---|---|---|
| `train.jsonl` | GLiNER `input` + correct `output` | Training pipeline |
| `dev.jsonl` | Same supervised format, *validation only* | Training evaluator / dev analysis, not weight updates |
| `dev_inputs.jsonl` | Input only + opaque evaluation ID sidecar | Dev inference pipeline |
| `test_inputs.jsonl` | Input only, no `output` and no label | Frozen candidate inference only |
| `test_labels.jsonl` | evaluation ID → true class; **no model input** | Independent final evaluator, restricted |
| `split_index.parquet` | restricted source/eval ID, split and fingerprint joins | Dataset-custody tooling |
| `external_test/` | `pending` manifest only | None until external data reviewed |

**Important:** The current older `prepare.py` writes a **labeled `test.jsonl` next to train/dev**. This is an identified **FIX**, not approved final behavior. Do not use it unmodified as a sealed-data custody system.

### 5.3 Training-validation examples vs inference inputs

The model sees `input` as customer context and `labels` as available answer options. During fine-tuning, `true_label` is the target **only in the supervised training/evaluation loader**. At inference it is **never provided**. Example:

```python
model.classify_text(
    customer_text,
    {"churn_status": ["not_churn", "churn"]},
    include_confidence=True,
)
```

The returned selected-label confidence is **not automatically a calibrated probability of churn**. Do not construct it with `1 - winning_confidence` unless the positive-score semantics are explicitly verified in the pinned model implementation; this issue belongs to the model-evaluation phase but must not be hidden in data preparation.

### 5.4 Library validator and independent policy validator

```python
from gliner2.training.data import TrainingDataset
train = TrainingDataset.load(".../train.jsonl")
train.validate(strict=True, raise_on_error=True)
dev = TrainingDataset.load(".../dev.jsonl")
dev.validate(strict=True, raise_on_error=True)
```

The GLiNER validator verifies schema correctness, **not** source truth, leakage, group splitting, rights or token-budget compliance. Run DCLab's additional policy tests independently. Record library version and validator result in `QUALITY.json`.

**Gate G5:** all JSONL rows valid, equal to the accepted split row counts, exact labels/task/options, stable ID join, and no training/test overlap.

---

## 6. Class distribution, weighting, augmentation and curriculum (DATA-06)

- **Report churn prevalence** in all partitions; do **not** rebalance the held-out evaluation sets. They must retain their original observed prevalence.
- Default experiment **no resampling and no augmentation**. Train on the naturally imbalanced customer distribution first.
- If the model underlearns the minority class, trial loss weighting or balanced sampling **only as a separate experiment**, fitted exclusively from TRAIN, with a documented seed and sampled-ID list; never duplicate dev/test rows.
- Avoid generating synthetic customer profiles with a guessed churn label. A realistic-looking synthetic example is not verified truth.
- Label-conditioned augmentation (e.g. reordering features or paraphrasing categorical names) is permitted **only** if the original factual values and labels are preserved, transformations are deterministic/versioned, and a distinct experiment proves it helps. Never create an augmented twin of a test row in train.
- Never use model-produced churn predictions as replacement labels for missing historical outcomes.
- Avoid spurious shortcut learning: compare the canonical serializer to one preregistered, independently evaluated **order-perturbation** or **numeric-format** variant only on DEV before final candidate freeze.
- Track demographics and rare categories carefully; small groups give noisy estimates. Fairness requires its own evaluation, not an accuracy-only acceptance gate.

**Gate G6:** any resampling or augmentation is declared, limited to TRAIN, and not applied to evaluation cohorts.

---

## 7. Dataset release, provenance, licenses and access control (DATA-07)

### 7.1 Folder layout (final target)

```text
research/churn-prediction/experiments/gliner_decide/
├── PLAN.md
├── DATA_FACTORY_PLAN.md                 # this document
├── config/
│   ├── source_contract_v1.yaml
│   ├── target_contract_v1.yaml
│   ├── features_v1.yaml
│   ├── serializer_v1.yaml
│   └── split_policy_v1.yaml
├── scripts/
│   ├── 01_audit_source.py
│   ├── 02_canonicalize.py
│   ├── 03_freeze_splits.py
│   ├── 04_serialize.py
│   ├── 05_build_jsonl.py
│   ├── 06_validate_release.py
│   └── 07_publish_dataset.py
├── tests/
│   ├── test_source.py
│   ├── test_normalization.py
│   ├── test_split_contract.py
│   ├── test_serialization.py
│   ├── test_gliner_jsonl.py
│   ├── test_sealed_isolation.py
│   └── test_reproduction.py
└── artifacts/                           # private/versioned/git-ignored data store
    └── telco_churn_gliner_v1/
        ├── source/SOURCE.json
        ├── audit/SOURCE_AUDIT.json
        ├── audit/ROW_DECISIONS.jsonl
        ├── partitions/SPLITS.json
        ├── partitions/split_index.parquet
        ├── samples/train.jsonl
        ├── samples/dev.jsonl
        ├── samples/dev_inputs.jsonl
        ├── samples/test_inputs.jsonl
        ├── protected/test_labels.jsonl
        ├── protected/evaluation_join_index.parquet
        ├── external_test/PENDING.json
        ├── DATASET_CARD.md
        ├── QUALITY.json
        ├── MANIFEST.json
        └── SHA256SUMS.txt
```

The **source CSV stays unchanged at its original path**, or is copied byte-identically into an immutable source store if policy requires a snapshot. Protected test labels should ideally be in a different ACL-restricted storage location/container rather than relying only on two folders under the same local user account. At minimum, training scripts must not receive that path or credentials and automated tests must prove they cannot access it.

### 7.2 Mandatory manifest

```json
{
  "manifest_schema": "dclab.gl_id_dataset.v1",
  "dataset_id": "telco_churn_gliner_v1",
  "created_at": "FILLED_AT_BUILD_TIME",
  "source": {
    "path": "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "sha256": "COMPUTE_FROM_BYTES",
    "license_id": "REVIEW_REQUIRED",
    "allowed_uses": ["eval_only_until_cleared"]
  },
  "target": {
    "source": "Churn",
    "model_task": "churn_status",
    "classes": ["not_churn", "churn"],
    "horizon": "not_verified"
  },
  "policy": {
    "allowed_feature_count": 19,
    "serializer": "telecom_text_v1",
    "split": "group_stratified_v1",
    "train_eligible": false,
    "locked_test_train_eligible": false
  },
  "partitions": {
    "train": {"count": "COMPUTE", "hash": "COMPUTE"},
    "dev": {"count": "COMPUTE", "hash": "COMPUTE"},
    "locked_test": {"count": "COMPUTE", "input_hash": "COMPUTE", "labels_hash": "COMPUTE", "access": "evaluator_only"},
    "external_test": {"status": "pending"}
  },
  "code_revision": "PIN_COMMIT_OR_CONTENT_HASH",
  "gliner2_version": "RECORD_AT_VALIDATION",
  "quality_status": "blocked_until_all_gates_pass"
}
```

The example fields are deliberately placeholders, not actual computed results. Record model rights separately (e.g., checkpoint Apache 2.0) and dataset rights separately; neither grants the other.

### 7.3 One authoritative source of truth

Reuse DCLab's existing `dclab_rnd/store/` manifest, provenance and contamination conventions where compatible. This experiment's files are immutable **artifacts**, not a new global training database. Avoid duplicating `dclab_rnd/ledger/`, `dclab_rnd/studio/trajectories.py` or the existing store catalogs.

### 7.4 Data-retention and test-exclusion registry

Maintain a central registry of `(source_hash, dataset_id, group_id, split, test_status, train_eligibility)`. Every later dataset builder, future fine-tune and retrieval-based augmentation must **reject sealed/external test data as training inputs**. Store hashes of sealed content, and carefully limit access to labels. When a sealed set is opened for final evaluation, keep the result immutable; do not use its exposed individual failures to adapt the same benchmark and still claim independence.

**Gate G7:** complete manifest, provenance and eligible-use decision; protected storage and cross-run test exclusion implemented.

---

## 8. Data QC report and acceptance test matrix (DATA-08)

### 8.1 Required QC report sections

1. **Identity:** SHA-256; source location, parser config, row/column counts.
2. **Legality and scope:** source/weight licenses, approved uses, privacy/PII policy.
3. **Schema:** 21 fields, exactly 19 candidate predictors, identifier and target.
4. **Missingness:** each feature's blank count and approved handling; zero-tenure TotalCharges case count.
5. **Validity:** categorical domain, numeric ranges and parse errors, quarantine/exclusion records.
6. **Labels:** Yes/No counts, churn prevalence, invalid/ambiguous labels.
7. **Duplicates:** exact source duplicates, same customer ID, same input features, conflicting label groups.
8. **Splits:** counts/groups/churn prevalence per partition, membership hashes, overlap=0.
9. **Serialization:** 19 features per input, numeric examples, label mapping, input text SHA-256, deterministic rerun test.
10. **Tokens:** min/median/p95/max full model-encoded lengths and truncation count (must be 0).
11. **Training format:** GLiNER strict validator, per-line parser, task/options/label consistency.
12. **Security:** sealed-label isolation, file access test, exclusion registry, no accidental Git tracking.
13. **Reproducibility:** code and dependency revision, same-run and fresh-run file hashes, artifact lineage.
14. **Final outcome:** `accepted` or `blocked` plus a reason list. No incomplete release marked accepted.

### 8.2 Test matrix

| Test ID | Check | Failure severity |
|---|---|---|
| T-01 | Raw source hash matches known/approved version | BLOCK |
| T-02 | 21 fields and 7,043 expected source rows, or an explicitly approved new version | BLOCK |
| T-03 | Exactly `Yes` / `No`, no lost/guessed labels | BLOCK |
| T-04 | `customerID` never in any model-visible sample | BLOCK |
| T-05 | `Churn` never in model-visible *feature text* | BLOCK |
| T-06 | All 19 fields accounted for; no imaginary features | BLOCK |
| T-07 | Numeric parser handles blanks/invalid tokens only by approved policy | BLOCK |
| T-08 | Identity/feature duplicate groups are partition-disjoint | BLOCK |
| T-09 | Every source row included once or has explicit quarantined reason | BLOCK |
| T-10 | Train/dev/test labels and class counts documented | BLOCK |
| T-11 | All files regenerate identically from same frozen source/config | BLOCK |
| T-12 | Strict GLiNER JSONL validation succeeds | BLOCK |
| T-13 | No silent truncation in real model tokenizer | BLOCK |
| T-14 | Trained preprocessing uses TRAIN only | BLOCK |
| T-15 | Training process cannot read test truth | BLOCK |
| T-16 | Train/development examples have no test IDs/content | BLOCK |
| T-17 | Labels stored only in supervision fields, never in inputs | BLOCK |
| T-18 | Legal training eligibility explicit, not inferred from open weights | BLOCK |
| T-19 | Each line joins one-to-one to a protected evaluation ID | BLOCK |
| T-20 | External test remains `pending` until approved actual data arrives | BLOCK if falsely marked ready |
| T-21 | Train-only resampling or augmentation with frozen seeds | BLOCK if enabled incorrectly |
| T-22 | Sample records undergo manual spot-check against original CSV | REVIEW |

### 8.3 Manual data inspection

Before training, manually inspect a reproducible sample of e.g. 40 TRAIN examples, oversampling cases with zero tenure, missing TotalCharges, minority churn labels, rare categories and duplicated feature fingerprints. For each compare:

- `input` text with permitted raw feature values;
- `true_label` with original `Churn`;
- source identity through the restricted index;
- no forbidden fields, normalization errors or fabricated facts;
- valid class schema and parseable JSON.

If any error is found, fix the converter, **rebuild the dataset with a new manifest version**, and rerun the entire quality suite. Do **not** silently patch a few JSONL lines.

**Gate G8:** automated tests pass and manual spot-check completed; release hash pinned.

---

## 9. Implementation stages, dependencies, and deliverables

| Stage | Work | Output | Definition of done |
|---|---|---|---|
| **D01** | Define source, feature, target and rights contracts | `config/*.yaml` | Reviewed and immutable v1 specs |
| **D02** | Audit source and canonicalize values | source/QC artifacts | Every row accounted for and parse policy audited |
| **D03** | Build identity groups, freeze splits | split index/manifest | Exactly one partition, zero group overlap |
| **D04** | Implement deterministic text serializer | `serialize.py`, tests | 19 source features, no leakage, stable bytes, no truncation |
| **D05** | Convert and validate GLiNER examples | train/dev JSONL, test input/label split | Official GLiNER validator + custom gates pass |
| **D06** | Implement sealed test custody and exclusion registry | protected labels, evaluator-only access | No training or preview path can read test truth |
| **D07** | Publish versioned dataset and QC report | dataset card, MANIFEST, SHA256SUMS | Rebuild from source yields same release hashes |
| **D08** | Baseline & GLiNER training handoff | training manifest & allowed paths | Trainer sees train/dev only; no test path or credential |

### Training handoff contract

The fine-tuning stage may start only when `QUALITY.json.status == "accepted"` and train-use eligibility is approved. It may read:

- the immutable `train.jsonl` and `dev.jsonl`;
- input contract, label schema, model checkpoint revision;
- train/dev sample hashes, examples/weights manifest and frozen training config.

It must **not** receive the original target-bearing CSV, `test_labels.jsonl`, test-join index, external-test truth or any dataset whose rights/status are unknown. For classical baselines, provide a separately controlled allowed-features view of the training/dev split, never unfiltered source.

The current official trainer example accepts `ExtractorTrainer(...).train(train_data=..., eval_data=...)` (verify against installed `gliner2` version). Use **DEV** for checkpoint selection, not the sealed test. The actual full-data training experiment has not yet been run.

---

## 10. CLI requirements for a Codex implementation

Commands below are a **proposed interface**, not commands that already exist:

```bash
python -m dclab_rnd.experiments.gl_telco audit \
  --source data/WA_Fn-UseC_-Telco-Customer-Churn.csv

python -m dclab_rnd.experiments.gl_telco prepare \
  --source data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --config research/churn-prediction/experiments/gliner_decide/config/source_contract_v1.yaml \
  --release telco_churn_gliner_v1

python -m dclab_rnd.experiments.gl_telco validate \
  --release telco_churn_gliner_v1

python -m dclab_rnd.experiments.gl_telco verify-rebuild \
  --release telco_churn_gliner_v1

python -m pytest research/churn-prediction/experiments/gliner_decide/tests -q
```

An implementation may instead use straightforward scripts inside `scripts/`; choose **one** entrypoint convention and document it. Do not create a parallel generalized training framework during the first experiment.

### Fail-fast behavior

On an invalid source hash, unexpected label, missing required feature, parse error, duplicate cross-split group, incomplete JSONL, mismatch in IDs, missing rights approval or inaccessible protected test store, the command must **exit nonzero before publishing a usable release**. Write to a temporary `.part` directory and promote to an immutable final directory only after all checks pass.

---

## 11. Future dataset expansion and multi-task decision model

This IBM dataset is a useful first benchmark, **not** sufficient proof of a general decision foundation model. When expanding to fraud, conversion, credit default or churn at another company:

1. Register a separate source and license/consent record; verify all new feature and label meanings.
2. Create a task-specific input contract. Different numeric features and units must not be invented or silently renamed.
3. Use group/time-aware splits appropriate to that task, ideally by company and prediction period for true transfer evaluation.
4. Maintain task-conditioned `labels` and `true_label` fields, preserving a source-specific target definition. A generic `"churn"` label is insufficient without a horizon/contract.
5. Train multi-task models on only permitted TRAIN groups across tasks; evaluate each task on its own frozen DEV and sealed sets.
6. Add **unseen-dataset / unseen-company** test tasks to measure transfer, not just familiar classification labels.
7. Record baseline tabular models and positive-class score/calibration semantics separately for every target.
8. Maintain a **global test-exclusion registry** so no future retrieval, instruction-tuning or distillation dataset contains sealed customers or labels.

Research outcome sought: does a 340M label-conditioned text encoder learn reusable decision representations **beyond the specific telecom dataset**? A single high churn accuracy does not establish that.

---

## 12. Review of the earlier starter kit: actionable fixes

The previously supplied `DCLab_GLiNER_Telco_Starter.zip` is a **working prototype concept**, but not a compliant data release. The following changes are mandatory:

| Starter file | Identified behavior | Required fix |
|---|---|---|
| `common.py` | `pd.to_numeric(errors="coerce")` could collapse unexpected numeric strings into missing | Explicit parse failures and per-row audit trail |
| `common.py` | Source labels and allowed features handled, but basic schema auditing | Validate actual source domains, anomalies, category differences, duplicate identity and token budget |
| `common.py` | `serialize_customer` produces readable text | Add serializer hash/version and malicious newline/control-character checks |
| `prepare.py` | Group by serialized text hash | Prefer canonical feature fingerprints independent of text wording; also group repeated customer IDs |
| `prepare.py` | Two-stage group-stratified split | Freeze exact membership/opaque IDs and comprehensive class/group audit |
| `prepare.py` | Writes `test.jsonl` including true labels beside training | Replace with unlabeled test inputs + restricted labels; no training-reader access |
| `prepare.py` | `splits.csv` stores source index + target | Restrict; use protected join map and opaque evaluation IDs |
| `prepare.py` | Manifest has no per-file checksums or training rights | Immutable dataset card, hashes, rights, source→sample lineage |
| `train_gliner.py` | Reads only `train.jsonl` | Add proper dev validation/checkpoint selection; cannot access locked-test storage |
| `evaluate.py` | Aligns predictions by order/length | Join by protected unique evaluation ID and enforce exact split/hash match |
| `evaluate.py` | Converts winner confidence into positive-class probability proxy | Separate scoring semantics; do not claim calibrated churn probability |
| `evaluate.py` | `SEALED_TEST_OPENED.json` beside output | Enforce a global dataset-family one-use sealed-test authorization, not a replaceable local marker |

Also note that the starter's earlier **synthetic 320-row check** did not verify full IBM-data conversion or a successful GLiNER training pass. Treat any successful full run as new evidence, not inherited success.

---

## 13. Required final report and release acceptance checklist

### Output artifacts to attach to the experiment record

- [ ] `SOURCE.json`, `SOURCE_AUDIT.json`, source hash and rights approval.
- [ ] `target_contract_v1.yaml`, `features_v1.yaml`, `serializer_v1.yaml`, `split_policy_v1.yaml`.
- [ ] Immutable group-safe split registry and exact row/group counts.
- [ ] `train.jsonl` and `dev.jsonl`, strict GLiNER-compatible and accepted.
- [ ] `test_inputs.jsonl` **without labels**.
- [ ] Restricted `test_labels.jsonl` and protected mapping; confirmed unreadable by training roles.
- [ ] External test registry present as **pending**.
- [ ] Exactly 19 permitted fields serialized, no invented columns or target leakage.
- [ ] Audited numeric missing-value policy and categorical-domain report.
- [ ] Token-length audit with zero unintended truncation.
- [ ] All data generation deterministic and reproducible with matching SHA-256 hashes.
- [ ] Training permissions explicitly verified, not inferred from model license.
- [ ] All data and sealed-test gates pass before any full fine-tune begins.
- [ ] Dataset card and versioned `QUALITY.json`, including honest limitations.
- [ ] Permanent test-exclusion registration to prevent contamination during future training.

### Definition of Done

A dataset release may be marked **READY_FOR_TRAINING** only when all eight gates (G1–G8) pass, the source and train-use rights are explicitly approved, train/dev JSONL validate against the actual pinned GLiNER library, and the training process cannot read sealed test labels. A missing or failed check leaves the release **BLOCKED**, even if the model could train on its files.

**The test set remains retained and protected for a single final evaluation; it is never silently folded back into training.**

---

## 14. Primary references

1. [Fastino GLiNER2.5-Decide model card](https://huggingface.co/fastino/GLiNER2.5-Decide): model identity, weights/license, classification output and exact JSONL training form.
2. [GLiNER2 training-data format guide](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/8-train_data.md): `input`/`output`, classification schema and validation rules.
3. [GLiNER2 training guide](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/9-training.md): `TrainingDataset.validate`, trainer arguments, evaluation split, LoRA, checkpointing.
4. [DCLab Telco research README](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac/research/churn-prediction/README.md): prediction limitation and baselines.
5. [DCLab churn benchmark](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac/research/churn-prediction/experiments/churn_exp/CHURN_BENCHMARK.md): cross-validation methodology and historical development results.
6. [DCLab CHURN-014 recipe](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/08c7bac/research/churn-prediction/experiments/churn_exp/results/CHURN-014/recipe.json): source hash, feature creation, leakage policy and fit protocol.

**End of data-engineering plan.**
