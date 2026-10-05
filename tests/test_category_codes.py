"""DCLAB-R11: numbers that stand for categories never feed a log, ratio or product feature,
and the DCLab notebook one-hot encodes them inside the training folds instead of passing them as numbers.

The cached UCI tables store categories as factorized codes (bank_marketing: May=0, Jun=1, ...).
These tests pin the rule on every path that derives or encodes features: the playbook FeatureEngineer,
the expansion recipes the DCLab notebook runs, and the notebook it exports.
"""

import importlib.util
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.categoricals import HEURISTIC_RULE, category_code_columns, declared_categorical, looks_like_codes  # noqa: E402
from dclab_rnd.agentic.catalog import DATASETS  # noqa: E402
from dclab_rnd.expansion import datasets as ds  # noqa: E402
from dclab_rnd.expansion import runner  # noqa: E402

try:
    from dclab_rnd.studio import ProjectStore, data as sd, engine, export
    from dclab_rnd.studio.contract import Contract
except ImportError as error:  # pydantic is not part of requirements/ci.txt
    STUDIO_MISSING = str(error)
else:
    STUDIO_MISSING = ""


def _playbook_features():
    """Load features.py on its own: general_pipeline/__init__ imports the GBDT stack, the module needs only scikit-learn."""
    spec = importlib.util.spec_from_file_location("playbook_features", ROOT / "general_pipeline/playbook/features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


features = _playbook_features()


def bank_like(n: int = 900) -> pd.DataFrame:
    """Factorized month codes stored as floats (as in data/public) beside skewed quantities, plus a target."""
    rng = np.random.default_rng(0)
    weights = 0.6 ** np.arange(12)
    month = rng.choice(12, n, p=weights / weights.sum())  # skewed codes: most calls in the first months seen
    frame = pd.DataFrame({
        "month": month.astype(float),
        "duration": rng.lognormal(5.0, 1.0, n).round(),  # skewed, non-negative, hundreds of values
        "balance": (rng.lognormal(7.0, 1.2, n) - 600.0).round(),  # skewed, about a third negative
        "age": rng.integers(18, 90, n),
    })
    logit = -1.5 + 0.004 * (frame["duration"] - 150) + 1.2 * np.isin(month, [2, 5])
    frame["subscribed"] = np.where(rng.uniform(size=n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    return frame


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=["subscribed"]), (frame["subscribed"] == "yes").astype(int)


def _derived(frame: pd.DataFrame, X: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in X.columns]


class DetectionTests(unittest.TestCase):
    def test_heuristic_flags_codes_not_quantities(self):
        X, _ = _xy(bank_like())
        self.assertTrue(looks_like_codes(X["month"]))
        for column in ("duration", "balance", "age"):
            self.assertFalse(looks_like_codes(X[column]), column)
        self.assertTrue(looks_like_codes(pd.Series([0, 1, 1, 0])))
        self.assertFalse(looks_like_codes(pd.Series([3, 3, 3])))  # a constant is not a category
        self.assertFalse(looks_like_codes(pd.Series(["a", "b", "a"])))  # strings are categorical by dtype already
        self.assertEqual(category_code_columns(X), ["month"])

    def test_a_declaration_replaces_the_heuristic(self):
        X, _ = _xy(bank_like())
        self.assertEqual(category_code_columns(X, declared=[]), [])
        self.assertEqual(category_code_columns(X, declared=["age", "not_a_column"]), ["age"])

    def test_catalog_declares_the_cached_codes(self):
        columns = list(pd.read_parquet(ROOT / "data/public/bank_marketing/X.parquet").columns)
        declared = declared_categorical("bank_marketing", columns)
        self.assertTrue({"month", "poutcome", "job", "day_of_week"} <= set(declared))
        self.assertFalse({"balance", "duration", "age"} & set(declared))
        self.assertEqual(declared_categorical("mushroom", ["cap-shape", "odor"]), ["cap-shape", "odor"])
        self.assertIsNone(declared_categorical("not_a_dataset", columns))
        for key, policy in DATASETS.items():  # every declared name exists in its cached table
            path = ROOT / "data/public" / key / "X.parquet"
            if path.exists() and policy["categorical"] != "ALL":
                missing = set(policy["categorical"]) - set(pd.read_parquet(path).columns)
                self.assertFalse(missing, f"{key}: {sorted(missing)}")


class PairedDifferenceTests(unittest.TestCase):
    def test_corrected_interval_is_wider_than_the_naive_one(self):
        from dclab_rnd.code_encoding import paired

        diff = paired([0.80, 0.82, 0.79, 0.81], [0.78, 0.80, 0.80, 0.79], test_train_ratio=0.5)
        self.assertAlmostEqual(diff["mean"], 0.0125)
        self.assertEqual((diff["wins"], diff["losses"], diff["folds"]), (3, 1, 4))
        naive, corrected = diff["naive_ci95"], diff["corrected_ci95"]
        self.assertLess(corrected[0], naive[0])
        self.assertGreater(corrected[1], naive[1])


class FeatureEngineerTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _xy(bank_like())

    def test_month_codes_get_no_log_while_duration_does(self):
        before = features.FeatureEngineer(stage="logs", categorical=[]).fit(self.X, self.y).transform(self.X)
        self.assertIn("log1p_month", before.columns)  # what the ladder built before DCLAB-R11 was enforced
        for categorical in (["month"], None):  # declared, then found by the heuristic
            out = features.FeatureEngineer(stage="logs", categorical=categorical).fit(self.X, self.y).transform(self.X)
            self.assertNotIn("log1p_month", out.columns)
            self.assertIn("log1p_duration", out.columns)
            self.assertIn("month", out.columns)  # the code itself stays an input

    def test_no_ratio_product_cluster_or_bin_uses_a_code(self):
        engineer = features.FeatureEngineer(stage="full_fe", categorical=["month"]).fit(self.X, self.y)
        out = engineer.transform(self.X)
        derived = _derived(out, self.X)
        self.assertTrue(any(c.startswith("ratio_") for c in derived) and any(c.startswith("inter_") for c in derived))
        self.assertEqual([c for c in derived if "month" in c], [])
        self.assertFalse([c for c in engineer.kmeans_cols_ + engineer.bin_cols_ if "month" in c])
        self.assertEqual(engineer.meta()["categorical"], ["month"])

    def test_raw_stage_keeps_codes_as_numbers_by_default(self):
        out = features.FeatureEngineer(stage="raw", categorical=["month"]).fit(self.X, self.y).transform(self.X)
        self.assertEqual(list(out.columns), list(self.X.columns))  # the model_building_50_v1 matrix, unchanged

    def test_one_hot_codes_uses_fit_row_levels_only(self):
        fit = self.X[self.X["month"] != 3].reset_index(drop=True)  # level 3 never seen while fitting
        y_fit = self.y[self.X["month"] != 3].reset_index(drop=True)
        for stage in ("raw", "full_fe"):
            engineer = features.FeatureEngineer(stage=stage, categorical=["month"], one_hot_codes=True).fit(fit, y_fit)
            out = engineer.transform(self.X)
            self.assertNotIn("month", out.columns)  # never a number in the matrix
            indicators = [c for c in out.columns if c.startswith("month=")]
            self.assertEqual(sorted(indicators), sorted(f"month={float(v)}" for v in fit["month"].unique()))
            self.assertTrue(set(np.unique(out[indicators].to_numpy())) <= {0.0, 1.0})
            unseen = (self.X["month"] == 3).to_numpy()
            self.assertTrue(unseen.any() and (out.loc[unseen, indicators].sum(axis=1) == 0).all())
            self.assertFalse([c for c in out.columns if "month" in c and not c.startswith("month=")])
        self.assertTrue(engineer.meta()["one_hot_codes"])


class RecipeTests(unittest.TestCase):
    def test_log_numeric_and_poly2_skip_declared_codes(self):
        frame, y = _xy(bank_like())
        spec = replace(ds.SPECS["letter_recognition"], key="codes_test", categorical_columns=("month",))
        bundle = ds.prepare_bundle(spec, frame, y, cv_folds=2)
        logs = runner._derive(bundle, "log_numeric", frame)
        self.assertEqual(sorted(logs.columns), ["log_age", "log_balance", "log_duration"])
        self.assertTrue((logs["log_balance"] < 0).any())  # signed log keeps negative balances
        self.assertEqual([c for c in runner._derive(bundle, "poly2", frame).columns if "month" in c], [])
        bundle.spec = replace(spec, categorical_columns=())
        self.assertIn("log_month", runner._derive(bundle, "log_numeric", frame).columns)

    def test_encoder_one_hot_encodes_codes_with_fit_fold_levels(self):
        frame, _ = _xy(bank_like())
        fit, apply = frame.iloc[:600], frame.iloc[600:].copy()
        apply.loc[apply.index[0], "month"] = 99.0  # a level the fit fold never saw
        encoder = runner._Encoder().fit(fit, categorical=["month"])
        self.assertEqual(encoder.numeric, ["duration", "balance", "age"])
        out = encoder.transform(apply)
        self.assertNotIn("month", out.columns)
        indicators = [c for c in out.columns if c.startswith("month=")]
        self.assertEqual(sorted(indicators), sorted(f"month={v}" for v in fit["month"].astype(str).unique()))
        self.assertEqual(out.loc[apply.index[0], indicators].sum(), 0.0)
        self.assertTrue((out.loc[apply.index[1:], indicators].sum(axis=1) == 1).all())
        capped = runner._Encoder().fit(fit, max_categories=3, categorical=["month"])
        self.assertEqual(len(capped.categories["month"]), 3)  # most frequent fit-fold levels only
        self.assertIn("month", runner._Encoder().fit(fit).numeric)  # no declaration: a number, as before

    def test_recipes_encode_codes_inside_each_training_fold(self):
        frame, y = _xy(bank_like())
        spec = replace(ds.SPECS["letter_recognition"], key="codes_test", categorical_columns=("month",))
        bundle = ds.prepare_bundle(spec, frame, y, cv_folds=2)
        runner.RECIPES["codes_test"] = {"raw": {"description": "raw", "derive": ()}, "log_numeric": {"description": "logs", "derive": ("log_numeric",)}}
        try:
            for fit_idx, valid_idx in bundle.cv_splits:
                X_fit = bundle.X_train.iloc[fit_idx]
                for name in ("raw", "log_numeric"):
                    recipe = runner.make_recipe(bundle, name)
                    recipe.fit(X_fit, bundle.y_train.iloc[fit_idx])
                    matrix, names = recipe.transform(bundle.X_train.iloc[valid_idx])
                    self.assertNotIn("month", names)
                    self.assertFalse([n for n in names if "month" in n and not n.startswith("month=")])
                    levels = [n for n in names if n.startswith("month=")]
                    self.assertEqual(sorted(levels), sorted(f"month={v}" for v in X_fit["month"].astype(str).unique()))
                    self.assertTrue(set(np.unique(matrix[:, [names.index(n) for n in levels]])) <= {0.0, 1.0})
            numbers = replace(spec, settings={**spec.settings, "one_hot_codes": False})  # the paired "before" arm
            bundle.spec = numbers
            self.assertEqual(runner.encoded_codes(numbers), ())
            recipe = runner.make_recipe(bundle, "raw").fit(bundle.X_train, bundle.y_train)
            self.assertIn("month", recipe.transform(bundle.X_train)[1])
        finally:
            runner.RECIPES.pop("codes_test", None)

    def test_declared_quantities_keep_their_products(self):
        rng = np.random.default_rng(1)
        columns = ["x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]
        frame = pd.DataFrame(rng.integers(0, 16, size=(240, len(columns))), columns=columns)  # integers 0-15 that measure shape
        bundle = ds.prepare_bundle(ds.SPECS["letter_recognition"], frame, pd.Series(np.repeat([0, 1, 2], 80)), cv_folds=2)
        self.assertEqual(runner._derive(bundle, "poly2", frame).shape[1], 120)  # the spec declares no codes, so no guessing


@unittest.skipIf(STUDIO_MISSING, f"DCLab notebook dependencies not installed: {STUDIO_MISSING}")
class NotebookTests(unittest.TestCase):
    CONTRACT = {"target": "subscribed", "task": "binary", "positive_label": "yes",
                "prediction_moment": "Before the call, from the client's record and the month of contact."}

    def setUp(self):
        self.store = ProjectStore(Path(tempfile.mkdtemp()))

    def _project(self, frame: pd.DataFrame) -> dict:
        project = self.store.create("Codes", "fintech and banking", "Predict subscription")
        path = self.store.data_dir(project["id"]) / "data.csv"
        frame.to_csv(path, index=False)
        loaded = sd.load_table(path)
        project["data"] = {"filename": "data.csv", "rows": len(loaded), "columns": list(loaded.columns), "sha256": sd.sha256(path), "profile": sd.profile_table(loaded)}
        project["contract"] = Contract(**self.CONTRACT).model_dump()
        project["settings"]["quick"] = True
        return self.store.save(project)

    def test_uploaded_codes_are_found_and_skipped(self):
        project = self._project(bank_like())
        p = engine.prepare(self.store, project)
        self.assertEqual(p.bundle.spec.categorical_columns, ("month",))
        self.assertEqual(p.code_rule, HEURISTIC_RULE)
        logs = runner._derive(p.bundle, "log_numeric", p.bundle.X_train)
        self.assertEqual(sorted(logs.columns), ["log_age", "log_balance", "log_duration"])
        self.assertIn("1 category-code column left out", p.recipes["log_numeric"]["description"])

    def test_samples_use_the_catalog_declaration(self):
        project = self.store.create("Bank", "fintech and banking", "Predict subscription")
        project = sd.use_sample(self.store, project["id"], "bank_marketing")
        self.assertEqual(project["data"]["categorical"], declared_categorical("bank_marketing", project["data"]["columns"][:-1]))
        suggestion = project["suggestion"]
        project["contract"] = Contract(target=suggestion["target"], task="binary", positive_label="1", forbidden=suggestion["forbidden"],
                                       prediction_moment=suggestion["prediction_moment"]).model_dump()
        project["settings"]["quick"] = True
        p = engine.prepare(self.store, self.store.save(project))
        self.assertIn("day_of_week", p.bundle.spec.categorical_columns)  # 31 levels: only the declaration catches it
        self.assertIn("catalog", p.code_rule)
        logs = runner._derive(p.bundle, "log_numeric", p.bundle.X_train)
        self.assertIn("log_balance", logs.columns)
        self.assertFalse({"log_month", "log_poutcome", "log_job", "log_day_of_week", "log_duration"} & set(logs.columns))

    def test_feature_record_and_export_carry_the_codes(self):
        project = self._project(bank_like())
        with runner.fast_profile():
            for stage in ("data", "leakage", "features"):
                record = engine.execute(self.store, project["id"], stage)
        self.assertEqual(record["evidence"]["category_code_columns"], ["month"])
        self.assertTrue(record["evidence"]["category_code_encoding"].startswith("one-hot"))
        raw = next(r for r in record["evidence"]["stage_results"] if r["recipe"] == "raw")
        bundle = engine.prepare(self.store, self.store.get(project["id"])).bundle
        per_fold = [3 + bundle.X_train["month"].iloc[fit_idx].nunique() for fit_idx, _ in bundle.cv_splits]
        self.assertAlmostEqual(raw["feature_count_mean"], float(np.mean(per_fold)))  # 3 quantities + one indicator per month seen in the fit fold
        self.assertTrue(any(c["claim_id"].endswith("-C3") and "`month`" in c["statement"] for c in record["claims"]))
        records = self.store.records(project["id"])
        records["features"]["decision"]["chosen"] = "log_numeric"
        notebook = export.notebook(self.store.get(project["id"]), records)
        cell = next(c["source"] for c in notebook["cells"] if "def add_logs" in c["source"])
        self.assertIn("CATEGORY_CODES = ['month']", cell)
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        X, _ = _xy(bank_like())
        namespace = {"np": np, "pd": pd, "X_train": X, "ColumnTransformer": ColumnTransformer, "SimpleImputer": SimpleImputer,
                     "Pipeline": Pipeline, "OneHotEncoder": OneHotEncoder, "StandardScaler": StandardScaler}
        exec(cell, namespace)  # the exported notebook's own code
        self.assertEqual(_derived(namespace["add_logs"](X), X), ["log_duration", "log_balance", "log_age"])
        preprocess = namespace["features"].fit(X).named_steps["preprocess"]
        self.assertEqual(namespace["features"].transform(X).shape[1], 6 + 12)  # 3 quantities + 3 logs + 12 month indicators
        scaled, encoded = preprocess.transformers_[0], preprocess.transformers_[1]
        self.assertEqual(scaled[0], "numeric")
        self.assertNotIn("month", scaled[2])  # never scaled as a number
        self.assertEqual(encoded[0], "categorical")
        self.assertEqual(encoded[2], ["month"])
        self.assertEqual(len(encoded[1].named_steps["onehot"].categories_[0]), 12)

    def test_export_matches_records_made_before_codes_were_encoded(self):
        project = self._project(bank_like())
        with runner.fast_profile():
            for stage in ("data", "leakage", "features"):
                engine.execute(self.store, project["id"], stage)
        records = self.store.records(project["id"])
        del records["features"]["evidence"]["category_code_encoding"]  # as recorded by the engine before this change
        cell = next(c["source"] for c in export.notebook(self.store.get(project["id"]), records)["cells"] if "ColumnTransformer(" in c["source"])
        self.assertIn(".select_dtypes('number').columns.tolist()\ncategorical", cell)  # codes stay in the numeric branch
        self.assertIn("never logged or multiplied (DCLAB-R11)", cell)


if __name__ == "__main__":
    unittest.main()
