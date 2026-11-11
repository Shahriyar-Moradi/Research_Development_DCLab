# Training Data Factory and Model Learning (DATA-01 … DATA-05)

> [!NOTE]
> Approved by the owner (2026-11-10). **Status: Active.**
> **Crucial directive:** Model training runs are **NOT to be run now**. Training execution is explicitly deferred until critical restriction data, provider licensing, provenance, and data-factory governance (D1–D4) are fully captured, audited, and verified.

---

## 1. Goal

Turn what DCLab already records into training datasets that are **eligible** (rights, restriction data, licensing, and consent), **truthfully labelled**, **diverse**, and **reproducible by manifest hash**.

Dataset size does not count as success. The priority is capturing and verifying critical restriction data and governance before any model training occurs.

| Exists | Where | Gap it leaves |
|---|---|---|
| Append-only store, records, example shapes (m1–m8), set manifests | `dclab_core/store.py`, `dclab_rnd/store/` | `Provenance` has `consent` only: no source class, licence, provider rights, critical restriction data, redaction version, or revocation |
| Lake (gold sets by origin, `TRAINABLE`, `eval_only`) | `dclab_core/lake.py`, `dclab_rnd/lake/` | **Conflicting rights rule**: lake keeps Claude answers eval-only; store `owner_decisions.json` allowed them for training |
| Trajectories, SFT from projects | `studio/trajectories.py`, `studio/sft.py`, `store/sets/llm.py` | Each shape is assembled separately; no unified record links request → context → action → tool → result → verification → outcome |
| `Quality` keep/silver/out, `LabelSource`, curators | `dclab_core/store.py`, `store/curators/` | No deterministic graders per target, no failure types, no label strength beyond gold/silver, no contrastive or preference roles |
| Coverage report (36.1) | `store/coverage.py` | Axes: agent × track only; counts raw records, not eligible graded examples; last measured 2026-10-28 (36 gaps) |
| Frozen exams, replay, versions, retire (36.4–36.7) | `store/exams.py`, `store/replay.py`, `store/versions.py`, `models/tuned.py` | Contracts exist, but training run is **on hold** pending restriction capture |
| LoRA script, MLX export | `research/llm-fine-tuning/experiments/sft/train_lora.py`, `lake/train.py` | Training runs deferred per owner directive |

---

## 2. Governance, Rights & Critical Restriction Data (DATA-02 / D1)

> [!IMPORTANT]
> **Commercial Provider-Output Restriction Policy:**
> Commercial model outputs (Claude answers, Claude thinking summaries, GPT) default to **restricted** for competitive model training. Customer consent alone does not override third-party model terms. Any training eligibility requires documented, scoped rights review. Synthetic and first-party verified data remain the primary eligible training sources.

Every source must carry:
1. **Source Class:** `synthetic`, `public`, `first_party`, `customer`, `human_labelled`, `commercial_model`, `open_weights_model`.
2. **Licensing & Terms:** Explicit licence identifier (SPDX id or proprietary), provider terms, and permitted use policy (`train`, `evaluate`, `restricted`, `excluded`).
3. **Critical Restriction Data:** Documented usage bounds, redistribution limits, competitive restriction flags, retention classes, and revocation handlers.
4. **Redaction & Privacy Version:** Verification that raw cell values, secrets, or unredacted personal text never enter trainable records.
5. **Sealed Exclusion:** Sealed evaluation examples and frozen exams are permanently prohibited from training sets.

---

## 3. Five Sub-Gaps & Implementation Packages

```mermaid
flowchart TD
  subgraph Data Factory Foundations (Active Focus)
    D1["Package D1: Source Rights & Restriction Inventory (DATA-02)<br/>(P0 - SourceClass, licence, permitted-use, restrictions)"]
    D2["Package D2: Unified Training-Grade Capture Contract (DATA-01)<br/>(P0 - AgentTrajectory v1, step links, outcomes, verification)"]
    D3["Package D3: Curation, Graders & Truthful Supervision (DATA-03)<br/>(P0 - Deterministic graders, failure types, label strength)"]
    D4["Package D4: Balanced Coverage & Leakage-Safe Splits (DATA-04)<br/>(P1 - 6 Home tracks × roles × problem kinds, no contamination)"]
  end
  
  subgraph Training Readiness (Deferred Execution)
    D5["Package D5: Replay & No-Forgetting Promotion Readiness (DATA-05)<br/>(P1 - Contracts, exam readiness; execution on hold)"]
  end

  D1 --> D2 --> D3 --> D4 -.-> D5
```

### Package D1: Source & Rights Inventory and Critical Restriction Capture (DATA-02, P0)
- Define `SourceRights` and `RightsReview` in `dclab_core/rights.py`.
- Capture critical restriction attributes: source class, licence, provider restrictions, customer consent scope, retention class, redaction version, and revocation status.
- Implement `dclab_rnd/datafactory/rights_inventory.py` to scan all store kinds, lake origins, and dataset manifests, producing `training_records/RIGHTS_INVENTORY.json`.
- Unify the rights decision between `dclab_core/lake.py` and `dclab_rnd/store/provider_outputs.py`.

### Package D2: Unified Training-Grade Data Capture Contract (DATA-01, P0)
- Create `dclab_core/trajectory.py` defining `AgentTrajectory` (v1) and `TrajectoryStep`.
- Link request, agent context, proposed action, tool call, observed result, verification, and task outcome.
- Explicitly prevent inventing hidden chain-of-thought or relabeling generated text as native reasoning.
- Build assembler `dclab_rnd/datafactory/assemble.py` and target projection logic (`project.py`) for routing, decisions, workflows, tool-calling, and solvers.

### Package D3: Dataset Cleaning, Quality Scoring & Truthful Supervision (DATA-03, P0)
- Create `dclab_core/curation.py` defining `Grade`, `LabelStrength` (`gold`, `verified`, `silver`, `weak`), and `FailureType`.
- Build deterministic graders in `dclab_rnd/datafactory/graders/`:
  - `tool_args.py`: Validate tool schema compliance.
  - `artifact.py`: Independent execution re-verification (successful tool call != successful task).
  - `decision.py`: Trap and ground-truth validation.
  - `routing.py`: Grounded in `track.outcome` user verdicts.
  - `safety.py`: Flag leakage and consent boundary violations as unsafe.
- Remove near-duplicates via fingerprinting.
- Add `make quality-report`.

### Package D4: Balanced Coverage Matrix & Contamination Prevention (DATA-04, P1)
- Extend `dclab_rnd/store/coverage.py` across 6 Home tracks, agent roles, ML problem kinds, languages (EN/FA), difficulty, and failure/recovery patterns.
- Implement strict family-grouped splits in `dclab_rnd/datafactory/splits.py` ensuring zero overlap across train/dev/sealed and frozen exams.
- Contamination assertion engine run automatically before any export.

### Package D5: Model Training & Continuous Improvement (DATA-05, P1) — DEFERRED
- Training execution is **on hold** per user instruction until restriction data is fully captured and audited.
- Architecture and contracts ready: replay sets, frozen exams, `TrainingRound` manifest, `no_forgetting_v1` gate, and instant rollback.

---

## 4. Acceptance Criteria

1. **Restriction & Rights Integrity:** 100% of exported training samples carry verified `SourceRights` with allowed licences and non-restricted status.
2. **Zero Contamination:** Automated gate proves zero leakage of sealed or evaluation items into training candidates.
3. **Truthful Supervision:** Every trainable trajectory has an assigned `Grade` with explicit failure typing and verified label strength.
4. **Reproducibility:** All datasets are reproducible from raw entries by manifest hash.
5. **No Premature Training:** Training runs remain blocked until critical restriction data capture is complete and signed off.
