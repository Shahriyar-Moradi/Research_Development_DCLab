"""Generate HyperAck-style step notebooks for each external project."""

from __future__ import annotations

import nbformat as nbf
from pathlib import Path

from general_pipeline.external_catalog import DATASET_CATALOG
from general_pipeline.external_project import project_dir
from general_pipeline.playbook.policy import get_policy


def _md(text: str):
    return nbf.v4.new_markdown_cell(text)


def _code(text: str):
    return nbf.v4.new_code_cell(text)


# Robust when cwd is notebooks/, <key>_exp/, external_projects/, or repo root.
_FIND_ROOT = '''
from pathlib import Path
import sys

def _find_repo_root() -> Path:
    starts = [Path.cwd().resolve()]
    # Jupyter may set cwd to the notebook folder; also walk from this file if present
    try:
        starts.append(Path(__file__).resolve().parent)  # type: ignore[name-defined]
    except NameError:
        pass
    for start in starts:
        for p in [start, *start.parents]:
            if (p / "general_pipeline").is_dir() and (p / "hyperack_exp").is_dir():
                return p
    raise RuntimeError(
        "Could not find repo root containing general_pipeline/. "
        "Open the notebook from the R&D repo or set the kernel cwd to the repo root."
    )

ROOT = _find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "hyperack_exp") not in sys.path:
    sys.path.insert(0, str(ROOT / "hyperack_exp"))
print("REPO ROOT:", ROOT)
'''.strip()


def build_notebook(key: str) -> Path:
    spec = next(s for s in DATASET_CATALOG if s.key == key)
    policy = get_policy(key)
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(_md(f"""# {spec.name} — Full Experiment Notebook

HyperAck-style walkthrough for **{spec.name}**.

- Source: {spec.url}
- Leakage columns (unsafe-only): `{policy.unsafe_only_features}`
- Has leakage: **{policy.has_leakage}**
- Policy: {policy.rationale}

This notebook mirrors the HyperAck ladder: baseline → FE families → selection → tuning → calibration → ensembles → safe vs unsafe honesty check.
"""))

    cells.append(_md("## 0. Setup"))
    cells.append(_code(f"""
{_FIND_ROOT}

from general_pipeline.playbook.ladder import run_project_ladder, results_frame, load_raw_xy
from general_pipeline.playbook.features import build_feature_matrix
from general_pipeline.playbook.reports import write_detailed_reports
from general_pipeline.playbook.policy import get_policy

KEY = '{key}'
policy = get_policy(KEY)
print(policy)
"""))

    cells.append(_md("## 1. Load safe vs unsafe splits\nSafe drops post-outcome / contested columns when defined."))
    cells.append(_code("""
Xtr_s, ytr_s, Xte_s, yte_s, meta_s = load_raw_xy(KEY, 'safe')
Xtr_u, ytr_u, Xte_u, yte_u, meta_u = load_raw_xy(KEY, 'unsafe')
print('Safe features:', Xtr_s.shape, 'dropped:', meta_s.get('dropped_leakage'))
print('Unsafe features:', Xtr_u.shape, 'dropped:', meta_u.get('dropped_leakage'))
print('Pos rate:', meta_s.get('pos_rate', ytr_s.mean()))
"""))

    cells.append(_md("## 2. Feature engineering stages\nInspect how each FE stage expands the matrix (train-fit stateful steps)."))
    cells.append(_code("""
for stage in ['raw', 'logs', 'ratios', 'interactions', 'full_fe', 'selected']:
    Xtr, Xte, fe_meta = build_feature_matrix(Xtr_s, ytr_s, Xte_s, stage=stage)
    print(f'{stage:12s} -> train {Xtr.shape} | top MI: {fe_meta.get(\"top_mi\", [])[:5]}')
"""))

    cells.append(_md("## 3. Run full ladder (safe + unsafe)\n15 experiments × modes — same protocol as HyperAck 01–15."))
    cells.append(_code("""
# Uncomment to re-run (can take several minutes):
# df = run_project_ladder(KEY)
df = results_frame(KEY)
df.sort_values(['mode', 'exp_id']).head(20)
"""))

    cells.append(_md("## 4. Safe vs Unsafe leakage cost"))
    cells.append(_code("""
import pandas as pd
safe = df[df['mode']=='safe'].set_index('exp_name')['roc_auc']
unsafe = df[df['mode']=='unsafe'].set_index('exp_name')['roc_auc']
cmp = pd.DataFrame({'safe': safe, 'unsafe': unsafe}).dropna()
cmp['leakage_gap'] = cmp['unsafe'] - cmp['safe']
cmp.sort_values('leakage_gap', ascending=False)
"""))

    cells.append(_md("## 5. Which optimization method wins?"))
    cells.append(_code("""
opt = df[df['mode']=='safe'][['exp_name','optimization_method','roc_auc','f1','feature_count']].sort_values('roc_auc', ascending=False)
opt
"""))

    cells.append(_md("## 6. Regenerate detailed reports + figures"))
    cells.append(_code("""
write_detailed_reports(KEY, df)
print('Reports written to external_projects/%s_exp/' % KEY)
"""))

    cells.append(_md("""## 7. Next steps for a new dataset

1. Add entry to `general_pipeline/external_catalog.py` and download.
2. Define leakage in `general_pipeline/playbook/policy.py`.
3. Run `python general_pipeline/playbook/run_playbook.py --dataset <key>`.
4. Open the generated notebook and reports in `external_projects/<key>_exp/`.
"""))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    pdir = project_dir(key)
    ndir = pdir / "notebooks"
    ndir.mkdir(parents=True, exist_ok=True)
    path = ndir / f"01_{key}_full_ladder.ipynb"
    path.write_text(nbf.writes(nb))

    nb2 = nbf.v4.new_notebook()
    nb2["cells"] = [
        _md(f"# {spec.name} — Feature Engineering Deep Dive"),
        _code(f"""
{_FIND_ROOT}
from general_pipeline.playbook.ladder import load_raw_xy
from general_pipeline.playbook.features import build_feature_matrix
from shared.protocol import evaluate
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd

KEY = '{key}'
model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', LGBMClassifier(random_state=42, verbosity=-1)),
])
rows = []
for mode in ['safe', 'unsafe']:
    Xtr, ytr, Xte, yte, meta = load_raw_xy(KEY, mode)
    for stage in ['raw', 'logs', 'ratios', 'interactions', 'full_fe', 'selected']:
        A, B, fe = build_feature_matrix(Xtr, ytr, Xte, stage=stage)
        m = evaluate(model, A, ytr, B, yte)
        rows.append({{'mode': mode, 'stage': stage, 'feats': A.shape[1], 'roc_auc': m['roc_auc'], 'f1': m['f1']}})
pd.DataFrame(rows).sort_values(['mode', 'roc_auc'], ascending=[True, False])
"""),
    ]
    nb2["metadata"] = nb["metadata"]
    (ndir / f"02_{key}_feature_engineering.ipynb").write_text(nbf.writes(nb2))

    nb3 = nbf.v4.new_notebook()
    nb3["cells"] = [
        _md(f"# {spec.name} — Optimization Methods Deep Dive"),
        _code(f"""
{_FIND_ROOT}
from general_pipeline.playbook.ladder import results_frame
KEY = '{key}'
df = results_frame(KEY)
safe = df[df['mode']=='safe'].sort_values('roc_auc', ascending=False)
print(safe[['exp_name','optimization_method','roc_auc','f1','feature_count']].to_string(index=False))
print('\\nBest method:', safe.iloc[0]['optimization_method'], safe.iloc[0]['exp_name'])
"""),
    ]
    nb3["metadata"] = nb["metadata"]
    (ndir / f"03_{key}_optimization_methods.ipynb").write_text(nbf.writes(nb3))

    return path


def generate_all_notebooks(keys=None):
    keys = keys or [s.key for s in DATASET_CATALOG]
    paths = []
    for key in keys:
        project_dir(key).mkdir(parents=True, exist_ok=True)
        paths.append(build_notebook(key))
        print(f"[{key}] notebooks -> notebooks/")
    return paths
