# Scientific ML Reliability, Experimentation & Memory Reuse Plan (SCI-01 … SCI-05, MEM-01 … MEM-05)

> [!NOTE]
> Approved by the owner (2026-11-10) and implemented in commit `69fcb92d`.

## 1. Goal Description

Establish scientific reliability across DCLab's machine-learning engine and autonomous experimentation, diagnose and repair demonstrated failures from the sealed `SYS-002` evaluation (segmentation, uplift, regression), and enable reliable transfer of verified experience across ML tasks without rediscovering solutions repeatedly.

A central principle of this plan: **we keep leakage data and models for contrastive comparison** (auditing, explanation, and training contrastive pairs) while ensuring the primary recommended pipeline is provably leakage-safe.

```mermaid
flowchart TD
  subgraph Scientific ML Reliability & Experimentation [SCI Tracks]
    SCI01["SCI-01: Diagnosis & Repair of Known Failures<br/>(Segmentation noise filtering, Uplift Qini bounds, Regression vs Baseline)"]
    SCI02["SCI-02: Scientific Correctness & Leakage Comparison<br/>(Splits, temporal integrity, calibration, + shadow leakage contrast model)"]
    SCI03["SCI-03: Autonomous Experimentation Operators<br/>(Guarded draft, debug, improve, evaluate from AIDE & R&D-Agent)"]
    SCI04["SCI-04: Capability Coverage Inventory<br/>(Machine-readable capability.json + explicit blockers for unsupported tasks)"]
    SCI05["SCI-05: Independent Evaluation on Real Unseen Datasets<br/>(UCI/Kaggle + customer datasets under blind review & reproducible code)"]
  end

  subgraph Knowledge, Memory & Skill Reuse [MEM Tracks]
    MEM01["MEM-01: Past Experiment & Outcome Retrieval<br/>(Structured search by dataset stats, constraints, outcomes)"]
    MEM02["MEM-02: Evidence-Backed Memory & Applicability Checks<br/>(Provenance, applicability guards against semantic mismatch)"]
    MEM03["MEM-03: Scalable Semantic Retrieval Evaluation<br/>(Benchmark pgvector vs local embedding cache on test questions)"]
    MEM04["MEM-04: Reusable, Versioned & Reviewed Agent Skills<br/>(Harness²-style verified skill library with immutable safeguards)"]
    MEM05["MEM-05: Independent Verification of Transfer<br/>(3-arm ablation: disabled vs brief vs retrieved skills)"]
  end

  SCI01 --> SCI02 --> SCI03 --> SCI04 --> SCI05
  SCI03 -. "Experiment logs" .-> MEM01
  MEM01 --> MEM02 --> MEM03 --> MEM04 --> MEM05
```

---

## 2. Design Principles & Safeguards

> [!IMPORTANT]
> **Leakage Data & Model Retention for Comparison.**
> Rather than discarding leakage features silently upon detection, the system constructs a **contrastive shadow comparison**:
> 1. An honest, leakage-safe model (recommended for deployment/export).
> 2. A benchmark shadow model fitted with the suspect/leakage features.
> 3. Contrastive metrics showing the delta between cross-validation and temporal/holdout evaluation (demonstrating why leakage is hazardous, providing training data for preference pairs and human explanations).

> [!CAUTION]
> **No Tuning on Sealed Sets (SYS-002).**
> `SYS-002_sealed.json` was run once and pinned. To fix the 3 failure modes identified in SYS-002 (segmentation, uplift, regression), independent development fixtures and replica seeds (`dev` split seeds) are used rather than repeatedly testing against the sealed seeds.

---

## 3. Implemented Components

### Component 1: Scientific Failure Repairs & Correctness (SCI-01, SCI-02)

#### 1.1 Segmentation Feature Selection (`dclab_core/segment.py`)
- **Root Cause:** For multi-level categorical noise columns, finite-sample random variation can exceed a low static threshold on 3,000 rows, preventing noise columns from being recognized as unused.
- **Repair:** Updated noise-filtering thresholds (`PLAUSIBLE = 0.03`, `UNUSED = 0.05`) and adjusted variance share evaluation so random noise columns are accurately identified and excluded.

#### 1.2 Uplift Model Evaluation Against Random Targeting (`dclab_core/uplift.py`)
- **Root Cause:** Unregularized T-Learner estimators overfit treatment arms on imbalanced responsiveness, degrading Qini confidence bounds against random targeting.
- **Repair:** Regularized T-Learner and X-Learner estimators (`max_leaf_nodes=6`, `min_samples_leaf=200`) and preserved Künzel et al. propensity weighting (`e * g0 + (1 - e) * g1`).

#### 1.3 Regression Performance Against Median Baseline (`dclab_rnd/kinds/regression.py`)
- **Root Cause:** On heavy-tailed and censored targets, standard squared-error estimators chase tail outliers and boundary floating-point comparisons fail strict thresholds.
- **Repair:** Added robust absolute-error estimators and stabilized boundary rounding comparisons against the `0.9 * median` baseline.

#### 1.4 Contrastive Leakage Benchmarking (`dclab_core/leakage_contrast.py`, `dclab_rnd/studio/engine.py`)
- Implemented `LeakageContrast` contract and integrated contrastive shadow model evaluation into `stage_leakage` and `stage_final`.
- Records clean vs. leakage cross-validation and holdout metrics, computing `cv_apparent_lift`, `holdout_delta`, and `generalization_collapse`.

---

### Component 2: Autonomous Experimentation Operators (SCI-03)

Implemented `dclab_rnd/studio/experiment_operators.py`:
- **`DraftOperator`**: Formulates initial candidate pipeline configurations under Part C rules.
- **`DebugOperator`**: Diagnoses convergence, regularization, or feature issues and applies family-appropriate corrective hyperparameters.
- **`ImproveOperator`**: Proposes bounded tuning variations based strictly on validation-fold diagnostics.
- **`EvaluateOperator` & `AutonomousExperimentLoop`**: Evaluates candidates on cross-validation folds without touching the holdout set, recording full attempt provenance and costs.

---

### Component 3: Capability Coverage & Independent Real Evaluation (SCI-04, SCI-05)

- **Capability Validation (`dclab_rnd/kinds/base.py`, `dclab_rnd/kinds/__init__.py`):** Implemented `UnsupportedTaskBlocker` and `validate_task_capability` against `dclab_rnd/kinds/capability.json`.
- **Real-World Evaluation (`tests/test_real_world_evaluation.py`):** Verified end-to-end execution, leakage auditing, and holdout reproducibility on real public datasets (Adult Census, Bank Marketing).

---

### Component 4: Knowledge, Memory & Verified Skill Reuse (MEM-01 to MEM-05)

- **MEM-01 (`dclab_rnd/memory/experiment_retrieval.py`):** Structured retrieval of past experiments by problem kind, dataset characteristics, constraints, and verified outcomes (both successes and failures).
- **MEM-02 (`dclab_rnd/lessons.py`):** Added `applicability_of` and `check_applicability` guards so remembered lessons are only applied when target problem constraints match.
- **MEM-03 (`tests/test_retrieval_semantic_benchmark.py`):** Benchmarked in-process embedding cache vs. simulated pgvector roundtrips across the evaluation suite.
- **MEM-04 (`.agents/skills/dclab-scientific-experiment/SKILL.md`):** Packaged reusable, reviewed agent skill enforcing Part C scientific safeguards.
- **MEM-05 (`dclab_rnd/agent_eval/memory_ablation.py`):** 3-arm ablation suite comparing memory-disabled, brief-memory, and retrieved-experience agents.
