"""Package 34.5, the model bake-off: on a tiny table, DCLab's engine and FLAML (tiny budgets) on the same split and the
same training folds, each reading the holdout once; the paired interval; the rule fixed before the run and refused when
changed; a stored result never replaced."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd

    from dclab_rnd import bakeoff
    from dclab_rnd.expansion import runner
except ImportError as error:  # pandas is not in every environment
    MISSING = str(error)
else:
    MISSING = "" if importlib.util.find_spec("flaml") else "flaml is not installed (pip install flaml)"

FIELDS = {"target": "bought", "task": "binary", "positive_label": "1", "group_column": "store",
          "prediction_moment": "when the visit starts, before anything is bought", "identifiers": ["visit_id"]}


def tiny_table(n=420, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"visit_id": np.arange(n), "store": rng.integers(0, 30, n).astype(str),
                          "pages": rng.poisson(4, n).astype(float), "minutes": rng.exponential(5, n),
                          "channel": rng.choice(["ads", "mail", "direct"], n)})
    logit = -1.0 + 0.4 * frame["pages"] - 0.1 * frame["minutes"] + (frame["channel"] == "mail") * 0.8
    frame["bought"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return frame


@unittest.skipIf(MISSING, MISSING)
class BakeoffRunTest(unittest.TestCase):
    """One run of the real orchestration on a tiny table: both engines in this process, FLAML with a 3 s budget."""

    @classmethod
    def setUpClass(cls):
        from flaml import AutoML

        cls.tmp = Path(tempfile.mkdtemp())
        cls.campaign, cls.work_root = cls.tmp / "evidence" / "bakeoff_test", cls.tmp / "work"
        cls.calls = {"fit": [], "predict_proba": [], "predict": []}
        real_fit, real_proba, real_predict = AutoML.fit, AutoML.predict_proba, AutoML.predict

        def fit(self, *args, **kwargs):
            cls.calls["fit"].append({"rows": len(kwargs["X_train"]), "split_type": kwargs.get("split_type"), "kwargs": kwargs})
            return real_fit(self, *args, **kwargs)

        def predict_proba(self, X, *args, **kwargs):
            cls.calls["predict_proba"].append(len(X))
            return real_proba(self, X, *args, **kwargs)

        def predict(self, X, *args, **kwargs):  # FLAML's own search scores its folds through its estimators, not this
            cls.calls["predict"].append(len(X))
            return real_predict(self, X, *args, **kwargs)

        def inline(work, engine_name, budget_s):
            return bakeoff.run_dclab(work) if engine_name == "dclab" else bakeoff.run_flaml(work, budget_s, n_jobs=1)

        frame = tiny_table()
        with mock.patch.dict(os.environ, {"DCLAB_AGENT_HOME": str(cls.tmp / "home"), "DCLAB_DECISION_SHADOW": "0"}), \
                runner.fast_profile(), mock.patch.object(AutoML, "fit", fit), mock.patch.object(AutoML, "predict_proba", predict_proba), \
                mock.patch.object(AutoML, "predict", predict):
            cls.summary = bakeoff.run(3, datasets=("tiny",), campaign=cls.campaign, work_root=cls.work_root, wait=False,
                                      prepare=lambda work, name: bakeoff.prepare_frame(work, name, frame, FIELDS), runner=inline)
        cls.work = cls.work_root / "tiny"
        cls.result = json.loads(bakeoff.result_path(cls.campaign, "tiny", ("tiny",)).read_text())

    def test_both_engines_score_the_same_holdout_rows_with_intervals(self):
        r = self.result
        self.assertTrue(r["same_split"])
        self.assertEqual(r["split"]["strategy"], "group")
        self.assertEqual(set(r["engines"]), {"dclab", "flaml"})
        for e in r["engines"].values():
            self.assertLessEqual(e["interval_95"]["low"], e["holdout_score"])
            self.assertGreaterEqual(e["interval_95"]["high"], e["holdout_score"])
            self.assertIn("cluster bootstrap", e["interval_95"]["method"])  # whole stores: the split is by store
            self.assertGreater(e["peak_memory_mb"], 0)
        self.assertLessEqual(r["paired"]["low"], r["paired"]["high"])
        self.assertIn(r["comparison"], ("flaml ahead", "flaml behind", "no clear difference"))
        self.assertNotIn("verdict", r)  # other readers take a results file's "verdict" as pass or fail
        d, f = (np.load(self.work / f"{e}.npz") for e in ("dclab", "flaml"))
        np.testing.assert_array_equal(d["holdout_rows"], f["holdout_rows"])
        np.testing.assert_array_equal(d["y_test"], f["y_test"])

    def test_flaml_searches_on_the_engine_folds_and_reads_the_holdout_once(self):
        from dclab_rnd.studio import engine

        store, pid = bakeoff._open(self.work)
        prepared = engine.prepare(store, store.get(pid))
        self.assertEqual(len(self.calls["fit"]), 1)
        call = self.calls["fit"][0]
        self.assertEqual(call["rows"], len(prepared.bundle.train_idx))  # FLAML is fitted on the training rows only
        self.assertIsInstance(call["split_type"], bakeoff.FixedFolds)
        self.assertEqual(len(call["split_type"].splits), len(prepared.bundle.cv_splits))
        for (a, b), (c, e) in zip(call["split_type"].splits, prepared.bundle.cv_splits):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, e)
        self.assertEqual(self.calls["predict_proba"], [len(prepared.bundle.test_idx)])  # one read of the holdout
        self.assertEqual(self.calls["predict"], [])
        given = call["kwargs"]["X_train"]
        for column in ("visit_id", "store", "bought"):  # an identifier, the group and the target never reach FLAML
            self.assertNotIn(column, given.columns)

    def test_the_engine_predictions_reproduce_its_own_record(self):
        info = json.loads((self.work / "dclab.json").read_text())
        store, pid = bakeoff._open(self.work)
        final = store.read_stage(pid, "final")
        self.assertAlmostEqual(info["holdout_score"], final["evidence"]["holdout_metrics"][info["metric"]], places=12)
        self.assertAlmostEqual(self.result["engines"]["dclab"]["holdout_score"], info["holdout_score"], places=12)

    def test_the_rule_is_fixed_before_the_run_and_a_change_is_refused(self):
        manifest = json.loads((self.campaign / "manifest.json").read_text())
        self.assertEqual(manifest["rule_sha256"], bakeoff.rule_sha256())
        self.assertLessEqual(manifest["frozen_at"], json.loads((self.work / "dclab.json").read_text())["finished_at"])  # frozen first
        changed = {**bakeoff.RULE, "min_datasets": 3}
        with mock.patch.object(bakeoff, "RULE", changed), self.assertRaises(SystemExit):
            bakeoff.freeze(self.campaign, 3, ("tiny",))
        with self.assertRaises(SystemExit):  # the protocol too: another budget is another campaign
            bakeoff.freeze(self.campaign, 30, ("tiny",))

    def test_a_stored_result_is_never_replaced_and_the_run_resumes(self):
        path = bakeoff.result_path(self.campaign, "tiny", ("tiny",))
        before = path.read_text()
        again = bakeoff.run(3, datasets=("tiny",), campaign=self.campaign, work_root=self.work_root, wait=False,
                            prepare=lambda *a: self.fail("nothing is prepared again"), runner=lambda *a: self.fail("nothing runs again"))
        self.assertEqual(again, {"tiny": "stored before"})
        self.assertEqual(path.read_text(), before)
        with self.assertRaises(FileExistsError):
            bakeoff.store_result(self.campaign, "tiny", ("tiny",), {})

    def test_compare_refuses_two_different_holdouts(self):
        import shutil

        other = self.tmp / "other"
        shutil.copytree(self.work, other)
        arrays = dict(np.load(other / "flaml.npz"))
        arrays["holdout_rows"] = arrays["holdout_rows"][::-1].copy()
        np.savez_compressed(other / "flaml.npz", **arrays)
        with self.assertRaises(ValueError):
            bakeoff.compare(other)

    def test_a_failing_dataset_is_recorded_and_the_campaign_goes_on(self):
        campaign, work_root = self.tmp / "evidence" / "failing", self.tmp / "work_failing"
        calls = []

        def failing(work, engine_name, budget_s):
            calls.append(engine_name)
            (work / f"{engine_name}.failed.json").write_text("{}")
            return {"status": "failed", "exit": 1}

        def broken(work, name):
            if name == "a":
                raise OSError("the download failed")
            return bakeoff.prepare_frame(work, name, tiny_table(), FIELDS)

        out = bakeoff.run(3, datasets=("a", "b"), campaign=campaign, work_root=work_root, wait=False, prepare=broken, runner=failing)
        self.assertEqual(out["a"], "failed: OSError")
        self.assertTrue((work_root / "a" / "dataset.failed.json").is_file())
        self.assertEqual(out["b"], "failed: dclab, flaml")
        self.assertEqual(calls, ["dclab", "dclab"])  # retried once; FLAML never runs without the engine's run
        self.assertEqual(bakeoff.failures(work_root, ["a", "b"]), {"a": ["dclab", "flaml"], "b": ["dclab"]})
        decision = bakeoff.report(campaign, final=True, work_root=work_root)
        self.assertEqual(decision["outcome"], "undecided")
        with self.assertRaises(SystemExit):  # a decision never comes while a dataset is neither compared nor failed
            (work_root / "b" / "dclab.failed.json").unlink()
            (campaign / "results" / "BKO-100_decision.json").unlink()
            bakeoff.report(campaign, final=True, work_root=work_root)

    def test_the_decision_waits_below_its_minimum_and_the_report_is_written(self):
        decision = json.loads((self.campaign / "results" / "BKO-100_decision.json").read_text())
        self.assertEqual(decision["outcome"], "undecided")  # one dataset, fewer than the rule's ten
        self.assertEqual(decision["rule_sha256"], bakeoff.rule_sha256())
        text = (self.campaign / "CAMPAIGN_REPORT.md").read_text()
        self.assertIn("| tiny |", text)
        self.assertTrue((self.work / "flaml_search.log").is_file())  # every FLAML trial kept beside the result


@unittest.skipIf(MISSING, MISSING)
class BakeoffRuleTest(unittest.TestCase):
    def rows(self, intervals):
        return [{"dataset": f"d{i}", "paired": {"low": low, "high": high}} for i, (low, high) in enumerate(intervals)]

    def test_go_needs_a_third_ahead_and_more_ahead_than_behind(self):
        ahead, behind, tie = (0.01, 0.05), (-0.05, -0.01), (-0.01, 0.01)
        self.assertEqual(bakeoff.decide(self.rows([ahead] * 4 + [tie] * 8))["outcome"], "go")  # 4 of 12 = a third
        self.assertEqual(bakeoff.decide(self.rows([ahead] * 3 + [tie] * 9))["outcome"], "no go")
        self.assertEqual(bakeoff.decide(self.rows([ahead] * 5 + [behind] * 5 + [tie] * 2))["outcome"], "no go")
        self.assertEqual(bakeoff.decide(self.rows([ahead] * 9))["outcome"], "undecided")  # fewer than ten datasets
        self.assertEqual(bakeoff.decide(self.rows([ahead] * 4 + [tie] * 8))["package_34_8"], "goes ahead")

    def test_a_flaml_failure_counts_behind_and_the_text_dataset_is_never_judged(self):
        ahead, tie = (0.01, 0.05), (-0.01, 0.01)
        rows = self.rows([ahead] * 4 + [tie] * 8)
        self.assertEqual(bakeoff.decide(rows, failures={"x": ["flaml"]})["outcome"], "no go")  # 4 of 13 is under a third
        out = bakeoff.decide(rows, failures={"x": ["dclab"]})
        self.assertEqual((out["outcome"], out["left_out_dclab_failed"]), ("go", ["x"]))
        text = rows + [{"dataset": "ecommerce_clothing_reviews", "paired": {"low": -0.2, "high": -0.1}}]
        out = bakeoff.decide(text)
        self.assertEqual((out["outcome"], out["not_judged"], out["datasets_judged"]), ("go", ["ecommerce_clothing_reviews"], 12))

    def test_the_solution_from_a_suggestion_names_the_positive_class(self):
        import pandas as pd

        telco = bakeoff.solution_from_suggestion({"target": "Churn", "prediction_moment": "at the end of the snapshot month",
                                                  "forbidden": [], "identifiers": ["customerID"]}, pd.DataFrame({"Churn": ["Yes", "No"]}))
        self.assertEqual(telco["positive_label"], "Yes")
        fraud = bakeoff.solution_from_suggestion({"target": "Class", "task": "binary", "prediction_moment": "at authorization time",
                                                  "time_column": "Time", "forbidden": [{"column": "Time", "reason": "a counter"}],
                                                  "identifiers": []}, pd.DataFrame({"Class": [0, 1]}))
        self.assertEqual((fraud["forbidden"], fraud["time_column"], fraud["positive_label"]), ([], "Time", "1"))

    def test_the_paired_difference_is_oriented_so_positive_means_flaml_better(self):
        from sklearn.metrics import mean_absolute_error, roc_auc_score

        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 300)
        good, noise = y + rng.normal(0, 0.3, 300), rng.normal(0, 1, 300)
        auc = lambda t, p: float(roc_auc_score(t, p))
        self.assertGreater(bakeoff.paired_bootstrap(y, good, noise, auc, True, draws=200)["low"], 0)
        target = rng.normal(10, 2, 300)
        mae = lambda t, p: float(mean_absolute_error(t, p))
        out = bakeoff.paired_bootstrap(target, target + rng.normal(0, 0.1, 300), target + rng.normal(0, 2, 300), mae, False, draws=200)
        self.assertGreater(out["low"], 0)  # lower error for the first: positive

    def test_busy_names_a_gate_a_store_run_or_a_loaded_model(self):
        self.assertIn("gate", bakeoff.busy("123 .venv/bin/python tests/run_parallel.py test_x.py"))
        self.assertIn("local model", bakeoff.busy("77 /Applications/Ollama.app/Contents/Resources/ollama runner --model x"))
        self.assertIn("store run", bakeoff.busy("88 python -m dclab_rnd.store run"))
        with mock.patch.object(bakeoff.shutil, "disk_usage", return_value=mock.Mock(free=10**12)), \
                mock.patch.object(bakeoff.subprocess, "run", return_value=mock.Mock(stdout="1")):
            self.assertIsNone(bakeoff.busy("99 /usr/bin/ollama serve"))  # the server alone holds no model

    def test_every_dataset_is_one_the_studio_offers_and_hyperack_is_left_out(self):
        from dclab_rnd.studio import data as sd

        offered = {s["key"] for s in sd.sample_catalog()}
        self.assertLessEqual(set(bakeoff.DATASETS), offered)
        self.assertNotIn("hyperack", bakeoff.DATASETS)
        self.assertEqual(len(set(bakeoff.DATASETS)), 14)


if __name__ == "__main__":
    unittest.main()
