"""Package 17.6: review and improve (dclab_core.review, the engine's review rounds, the runner's changes, the export).

Weaknesses come from out-of-fold errors on training rows; each hypothesis cites one; a round runs one change on the
same folds and keeps it only by the paired rule; the holdout is used once, by the final stage.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pydantic  # noqa: F401
except ImportError as error:
    PYDANTIC_MISSING = str(error)
else:
    PYDANTIC_MISSING = ""


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class ReviewTest(unittest.TestCase):
    def test_a_lopsided_capped_target_gets_the_median_loss_first_with_its_reasons(self):
        from dclab_core.review import CATALOGUE, check, review

        rng = np.random.default_rng(0)
        base = rng.gamma(4, 4, 4000)
        y = np.minimum(base + np.where(rng.random(4000) < 0.25, rng.exponential(20, 4000), 0), 60)
        pred = base + 4  # a mean model: the tail pulls it up
        out = review("regression", y, pred, metric="mae", cap=60.0)
        ids = {w.id for w in out.weaknesses}
        self.assertTrue({"F-weak-residual_skew", "F-weak-cap"} <= ids)
        first = out.next_experiments[0]
        self.assertEqual(first.name, "loss.median")
        self.assertTrue(CATALOGUE[first.name].runnable)
        hypothesis = out.hypotheses[first.hypothesis]
        self.assertTrue(set(hypothesis.cites) <= ids)
        self.assertIsNone(check(hypothesis, out.weaknesses))
        invented = hypothesis.model_copy(update={"cites": ["F-weak-the-moon"]})
        self.assertIn("not measured", check(invented, out.weaknesses))
        self.assertNotIn("loss.median", [e.name for e in review("regression", y, pred, metric="rmse", cap=60.0).next_experiments])

    def test_a_good_model_on_clean_data_has_no_weakness(self):
        from dclab_core.review import review

        rng = np.random.default_rng(1)
        y = rng.normal(50, 10, 3000)
        frame = pd.DataFrame({"region": rng.choice(["n", "s", "e", "w"], 3000), "store": rng.integers(0, 200, 3000)})
        out = review("regression", y, y + rng.normal(0, 2, 3000), metric="mae", frame=frame, entities=["store"])
        self.assertEqual(out.weaknesses, [])  # no false alarm: nothing to propose
        self.assertEqual(out.next_experiments, [])

    def test_classes_segments_and_entities(self):
        from dclab_core.review import review, runnable

        rng = np.random.default_rng(2)
        n = 3000
        y = rng.choice(["a", "b", "c", "rare"], n, p=[0.45, 0.35, 0.18, 0.02])
        pred = np.where(y == "rare", "a", y)  # the rare class is never found
        pred = np.where((y == "c") & (rng.random(n) < 0.5), "b", pred)
        frame = pd.DataFrame({"channel": np.where(rng.random(n) < 0.15, "phone", "web"), "customer": rng.integers(0, 300, n)})
        out = review("multiclass", y, pred, frame=frame, entities=["customer"])
        names = [e.name for e in out.next_experiments]
        self.assertEqual([e.name for e in runnable(out.next_experiments)], ["weights.balanced"])
        self.assertIn("classes.merge_tiny", names)  # proposed for the person, never run unattended
        self.assertLess(names.index("weights.balanced"), names.index("classes.merge_tiny"))

        error = rng.random(n) < 0.05
        hot = frame["customer"].to_numpy() < 30
        error = error | (hot & (rng.random(n) < 0.7))
        y_bin = rng.integers(0, 2, n)
        p_bin = np.where(error, 1 - y_bin, y_bin).astype(float)
        frame["segment"] = np.where(hot, "x", rng.choice(["y", "z"], n))
        found = {w.id.split("-")[2] for w in review("binary", y_bin, p_bin, frame=frame, entities=["customer"]).weaknesses}
        self.assertTrue({"entity", "segment"} <= found)

    def test_noise_and_honest_differences_raise_no_alarm(self):
        from dclab_core.review import weaknesses

        for seed in range(5):
            rng = np.random.default_rng(seed)
            n = 3000
            who = pd.DataFrame({"customer": rng.integers(0, 300, n)})
            y = (rng.random(n) < 0.1).astype(int)
            self.assertEqual(weaknesses("binary", y, np.full(n, 0.1), frame=who, entities=["customer"]), [], seed)
            labels = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
            proba = np.tile([0.5, 0.3, 0.2], (n, 1))
            multi = weaknesses("multiclass", labels, proba, frame=who, entities=["customer"], classes=["a", "b", "c"])
            self.assertEqual([w.topic for w in multi if w.topic in ("entity", "segment")], [], seed)  # (it never finds b or c: that is real)
            plan = rng.choice(["free", "pro"], n, p=[0.7, 0.3])
            rate = np.where(plan == "pro", 0.4, 0.1)  # a higher base rate the model already predicts
            y2 = (rng.random(n) < rate).astype(int)
            self.assertEqual(weaknesses("binary", y2, rate, frame=pd.DataFrame({"plan": plan})), [], seed)
            noisy = np.where(plan == "pro", 15.0, 3.0)  # a noisier segment, no bias
            truth = rng.normal(50, 1, n)
            self.assertEqual(weaknesses("regression", truth + rng.normal(0, noisy), truth, metric="rmse",
                                        frame=pd.DataFrame({"plan": plan})), [], seed)
        biased = weaknesses("regression", truth + np.where(plan == "pro", 8.0, 0.0) + rng.normal(0, 3, n), truth, metric="rmse",
                            frame=pd.DataFrame({"plan": plan}))
        self.assertEqual([w.id for w in biased], ["F-weak-segment-plan-pro"])  # too low where plan is pro: a real weakness

    def test_adopt_needs_a_paired_gain_beyond_noise(self):
        from dclab_core.review import adopt

        self.assertTrue(adopt([9.4, 9.2, 9.4], [8.6, 8.5, 8.9], higher=False, margin=0.01)["adopted"])
        noisy = adopt([8.0, 9.0, 8.5], [7.5, 9.4, 8.3], higher=False, margin=0.01)
        within = adopt([8.4, 8.3, 8.5], [8.1, 8.35, 8.35], higher=False)
        self.assertGreater(within["mean_gain"], within["standard_error"])  # above one standard error ...
        self.assertFalse(within["adopted"])                                # ... but under 1.89 of them: not kept
        self.assertFalse(noisy["adopted"])
        self.assertIn("does not beat", noisy["why"])
        self.assertFalse(adopt([0.5], [0.9], higher=True)["adopted"])  # one fold is not a comparison
        self.assertFalse(adopt([0.5, 0.5], [0.501, 0.502], higher=True, margin=0.01)["adopted"])


class RunnerChangeTest(unittest.TestCase):
    def test_each_change_is_learned_on_the_fit_rows(self):
        from dclab_rnd.expansion import runner

        self.assertEqual(runner.change_params({"loss": "median"}, "lightgbm", "timeseries_regression"), {"model__objective": "l1"})
        self.assertIsNone(runner.change_params({"loss": "median"}, "ridge", "timeseries_regression"))
        self.assertIsNone(runner.change_params({"class_weight": "balanced"}, "ridge", "timeseries_regression"))
        self.assertEqual(runner.change_params({"target": "log1p"}, "ridge", "timeseries_regression"), {"review__target": "log1p"})

        class Bundle:
            task_type = "timeseries_regression"
            spec = type("Spec", (), {"settings": {}, "key": "t", "categorical_columns": ()})()

        rng = np.random.default_rng(0)
        X = pd.DataFrame({"x": rng.normal(size=300)})
        y = pd.Series(np.exp(1 + X["x"]) * 3)
        recipe = mock.Mock(sparse=False)
        recipe.transform = lambda frame: (frame.to_numpy(), ["x"])
        with mock.patch.object(runner, "make_recipe", return_value=recipe):
            plain = runner.fit_and_predict(Bundle, "raw", "ridge", None, X, y, X)["predictions"]
            logged = runner.fit_and_predict(Bundle, "raw", "ridge", {"review__target": "log1p"}, X, y, X)["predictions"]
            held = runner.fit_and_predict(Bundle, "raw", "ridge", {"review__clip": "fit_range"}, X, y, X * 3)["predictions"]
        self.assertLess(np.mean(np.abs(logged - y)), np.mean(np.abs(plain - y)))  # an exponential target fits on a log scale
        self.assertGreaterEqual(held.min(), y.min())
        self.assertLessEqual(held.max(), y.max())


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class EngineReviewTest(unittest.TestCase):
    def run_project(self, frame, solution):
        from dclab_rnd.studio import ProjectStore, data as sd, engine
        from dclab_rnd.studio.solution import Solution

        store = ProjectStore(Path(tempfile.mkdtemp()))
        project = store.create("Replica", "general", "Review")
        path = store.data_dir(project["id"]) / "data.csv"
        frame.to_csv(path, index=False)
        loaded = sd.load_table(path)
        project["data"] = {"filename": "data.csv", "rows": len(loaded), "columns": list(loaded.columns), "sha256": sd.sha256(path),
                           "profile": sd.profile_table(loaded)}
        project["solution"] = Solution(**solution).model_dump()
        project["settings"]["quick"] = True
        store.save(project)
        engine.run_all(store, project["id"])
        return store, project["id"]

    DELIVERY = {"target": "actual_minutes", "task": "regression", "identifiers": ["order_id"],
                "prediction_moment": "When the customer places the order.", "group_column": "restaurant_id"}

    def test_the_delivery_replica_finds_the_cap_and_a_round_tests_a_median_loss_by_the_rule(self):
        """On every seed the cap and the lopsided errors are found and a median loss is tried on the same folds; it is
        kept exactly when its paired gain clears the margin and the t bound (seed 1 sits on that bound, seed 0 under it)."""
        from dclab_core.replicas import delivery_time

        for seed in (0, 1):
            with self.subTest(seed=seed):
                frame = delivery_time.generate(seed=seed, n=8000).orders.drop(columns=["delivered_at", "courier_rating"])
                store, pid = self.run_project(frame, self.DELIVERY)
                models, final = store.read_stage(pid, "models")["evidence"], store.read_stage(pid, "final")["evidence"]
                review = models["review"]
                self.assertIn("F-weak-cap", {w["id"] for w in review["review"]["weaknesses"]})
                [round_] = review["rounds"]
                self.assertEqual(round_["experiment"], "loss.median")
                self.assertIn("F-weak-cap", round_["cites"])
                self.assertAlmostEqual(round_["margin"], 0.01 * round_["base_mean"])  # the studio's relative tuning margin
                self.assertGreater(round_["base_mean"], round_["variant_mean"])        # a gain on average ...
                clears = round_["mean_gain"] > round_["margin"] and round_["mean_gain"] > round_["bound"]
                self.assertEqual(round_["adopted"], clears and round_.get("own_family_gain", {"adopted": True})["adopted"])
                self.assertEqual(final["review_change"], review["adopted_params"])
                self.assertEqual(final["holdout_uses_in_this_project"], 1)  # the review never touched the holdout

    def test_a_family_switch_keeps_the_record_consistent(self):
        from dclab_core import review as core_review
        from dclab_core.replicas import delivery_time
        from dclab_rnd.expansion import runner

        kept = {"adopted": True, "why": "forced", "mean_gain": 1.0, "standard_error": 0.1, "bound": 0.2, "fold_gains": [1.0]}
        frame = delivery_time.generate(seed=0, n=1500).orders.drop(columns=["delivered_at", "courier_rating"])
        with runner.fast_profile(), mock.patch.object(core_review, "adopt", return_value=kept):
            store, pid = self.run_project(frame, self.DELIVERY)
        models, final = store.read_stage(pid, "models")["evidence"], store.read_stage(pid, "final")["evidence"]
        [round_] = [r for r in models["review"]["rounds"] if r["experiment"] == "loss.median"]
        if round_["family"] != round_["base_family"]:  # the leader could not make the change: another family did
            self.assertEqual(models["selected_model"], round_["family"])
            self.assertEqual(models["screen_leader"], round_["base_family"])
            self.assertEqual(models["selected_metric_mean"], models["review"]["selected_mean"])
        self.assertEqual(final["model"], models["selected_model"])
        self.assertEqual(final["review_change"], models["review"]["adopted_params"])

    def test_a_kept_change_reaches_the_final_stage_and_the_record(self):
        from dclab_core import review as core_review
        from dclab_core.contracts import Experiment, Hypothesis, Review
        from dclab_core.replicas import delivery_time
        from dclab_rnd.expansion import runner

        forced = Review(hypotheses=[Hypothesis(statement="Errors grow.", cites=["F-weak-errors_grow"], test="a fold test")],
                        next_experiments=[Experiment(name="target.log", hypothesis=0, expected_gain="smaller errors", cost="low")])
        frame = delivery_time.generate(seed=0, n=1200).orders.drop(columns=["delivered_at", "courier_rating"])
        with runner.fast_profile(), mock.patch.object(core_review, "review", return_value=forced), \
                mock.patch.object(core_review, "adopt", return_value={"adopted": True, "why": "forced"}):
            store, pid = self.run_project(frame, self.DELIVERY)
        models, final = store.read_stage(pid, "models")["evidence"], store.read_stage(pid, "final")["evidence"]
        self.assertEqual(models["review"]["adopted_params"], {"review__target": "log1p"})
        self.assertEqual(final["review_change"], {"review__target": "log1p"})
        self.assertEqual(final["selected_model_params"].get("review__target"), "log1p")
        self.assertEqual(runner._jsonable({"a": {"review__clip": "fit_range"}}), {"a": {"review__clip": "fit_range"}})

    def test_a_review_that_fails_leaves_the_models_stage_as_it_was(self):
        from dclab_core.replicas import delivery_time
        from dclab_rnd.expansion import runner

        frame = delivery_time.generate(seed=0, n=1200).orders.drop(columns=["delivered_at", "courier_rating"])
        with runner.fast_profile(), mock.patch("dclab_core.review.review", side_effect=ValueError("broken")):
            store, pid = self.run_project(frame, self.DELIVERY)
        models = store.read_stage(pid, "models")["evidence"]
        self.assertNotIn("review", models)
        self.assertEqual(store.read_stage(pid, "final")["evidence"]["review_change"], {})


class ExportChangeTest(unittest.TestCase):
    def test_the_notebook_replays_a_kept_change(self):
        from dclab_rnd.studio import export

        self.assertIn("TransformedTargetRegressor", export._change_code({"review__target": "log1p"}))
        self.assertEqual(export._change_code({}), "")
        self.assertIn("np.clip", export._cv_code("regression", "m", True, clip=True))
        self.assertIn("np.clip", export._score_code("regression", "m", clip=True))
        self.assertIn("model__sample_weight=balanced(y_train)", export._fit_code("binary", "X_train", "y_train", weighted=True))
        namespace = {"np": np, "labels": ["a", "b", "c"]}
        exec(export.MULTICLASS_HELPERS + export.WEIGHT_HELPER, namespace)  # noqa: S102 - the notebook's own helpers
        self.assertAlmostEqual(namespace["balanced"](np.array([0, 0, 0, 1])).tolist()[3], 2.0)
        for source in (export._cv_code("regression", "m", True, clip=True), export._score_code("multiclass", "m")):
            compile(source, "<cell>", "exec")


if __name__ == "__main__":
    unittest.main()
