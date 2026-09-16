"""Run HyperAck-style playbook for external dataset projects.

Usage:
  .venv/bin/python general_pipeline/playbook/run_playbook.py --dataset all
  .venv/bin/python general_pipeline/playbook/run_playbook.py --dataset bank_marketing
  .venv/bin/python general_pipeline/playbook/run_playbook.py --dataset adult --reports-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from general_pipeline.external_catalog import DATASET_CATALOG  # noqa: E402
from general_pipeline.external_project import scaffold_project  # noqa: E402
from general_pipeline.playbook.ladder import results_frame, run_project_ladder  # noqa: E402
from general_pipeline.playbook.notebooks import generate_all_notebooks  # noqa: E402
from general_pipeline.playbook.reports import write_detailed_reports  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="External dataset HyperAck playbook runner")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--reports-only", action="store_true")
    parser.add_argument("--notebooks-only", action="store_true")
    args = parser.parse_args()

    keys = [s.key for s in DATASET_CATALOG] if args.dataset == "all" else [d.strip() for d in args.dataset.split(",")]
    t0 = time.perf_counter()

    if args.notebooks_only:
        generate_all_notebooks(keys)
        return

    for key in keys:
        print("\n" + "=" * 72)
        print(f"PLAYBOOK: {key}")
        print("=" * 72)
        scaffold_project(key)
        generate_all_notebooks([key])
        if not args.reports_only:
            run_project_ladder(key)
        df = results_frame(key)
        write_detailed_reports(key, df)

        # Update project README pointer
        pdir = ROOT / "external_projects" / f"{key}_exp"
        readme = (pdir / "README.md").read_text() if (pdir / "README.md").exists() else ""
        extra = """

## HyperAck-style playbook artifacts

| Document | Content |
|---|---|
| `COMPLETE_MASTER_REPORT.md` | Executive summary + links |
| `FEATURE_ENGINEERING_REPORT.md` | FE stages & selection |
| `OPTIMIZATION_REPORT.md` | Tuning / ensembles ranking |
| `notebooks/` | Step-by-step Jupyter notebooks |
| `results/ladder/` | Exp 01–15 JSON (safe & unsafe) |

```bash
.venv/bin/python general_pipeline/playbook/run_playbook.py --dataset %s
```
""" % key
        if "COMPLETE_MASTER_REPORT.md" not in readme:
            (pdir / "README.md").write_text(readme.rstrip() + "\n" + extra)

    # Future-dataset pipeline doc at playbook root
    future = ROOT / "general_pipeline" / "playbook" / "FUTURE_DATASET_PIPELINE.md"
    future.write_text(
        """# Future Dataset Pipeline (HyperAck Playbook)

Reusable path to produce the same depth of analysis as HyperAck for any new tabular binary classification problem.

## Steps

1. **Register** the dataset in `general_pipeline/external_catalog.py`.
2. **Download / cache** via `download_external_datasets.py` (or drop parquet into `external_data/<key>/`).
3. **Define leakage policy** in `general_pipeline/playbook/policy.py`:
   - List post-outcome columns in `unsafe_only_features`.
   - Write a clear decision-time rationale.
4. **Run the playbook**:
   ```bash
   .venv/bin/python general_pipeline/playbook/run_playbook.py --dataset <key>
   ```
5. **Read** in `external_projects/<key>_exp/`:
   - `FEATURE_ENGINEERING_REPORT.md`
   - `OPTIMIZATION_REPORT.md`
   - `COMPLETE_MASTER_REPORT.md`
   - `notebooks/*.ipynb`
6. **Iterate** optimization: extend `ladder_specs()` or add Optuna trials following the tabular playbook rule.

## Ladder (mirrors HyperAck 01–15)

| ID | Focus |
|---|---|
| 01–02 | Baselines |
| 03–06 | FE families (logs, ratios, interactions, clusters/bins) |
| 07 | Mutual-information feature selection |
| 08–09 | RandomizedSearchCV (XGB, LGBM) |
| 10–11 | HistGB / ExtraTrees |
| 12 | Isotonic calibration |
| 13–14 | Stacking / soft-vote |
| 15 | Leakage honesty marker |

## Quadrants compared

- Safe baseline vs Safe optimized
- Unsafe baseline vs Unsafe optimized (when leakage exists)
- Safe vs Unsafe (leakage cost)
- FE stage ablation
- Optimization method ranking
"""
    )
    print(f"\nAll playbooks finished in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
