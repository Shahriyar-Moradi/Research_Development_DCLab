"""Package 29.3: truth by measured outcome. A choice among measured options (a feature recipe, a model family) is labelled
by validation only where validation can tell the options apart: one option ahead of every other by more than their
combined fold spread, in the metric's own direction (an error is better when lower), never an option the rule may not
choose; inside the noise nothing is labelled. A planted bad choice (a family validation clearly beats) is labelled wrong
and read into its set; self-play's solves carry the labels, a half-written or failed solve gives nothing."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd  # noqa: F401
except ImportError as error:  # the agent environment has no pandas
    MISSING = str(error)
else:
    MISSING = ""


def opt(i, mean, std, **extra):
    return {"id": i, "mean": mean, "std": std, **extra}


class ClearBestTests(unittest.TestCase):
    def test_only_an_option_ahead_of_every_other_by_the_combined_spread_is_the_truth(self):
        from dclab_core.measured import KIND, clear_best

        best, evidence = clear_best([opt("raw", 0.70, 0.01), opt("poly2", 0.75, 0.01), opt("lags", None, None)], True)
        self.assertEqual((best, evidence["kind"], evidence["ref"]), ("poly2", KIND, "decision.options.poly2"))
        self.assertAlmostEqual(evidence["value"], 0.05)
        self.assertIsNone(clear_best([opt("raw", 0.70, 0.04), opt("poly2", 0.72, 0.04)], True))  # inside the noise: nothing said
        self.assertIsNone(clear_best([opt("a", 0.4535, 0.244), opt("b", 0.4922, 0.19)], True))  # the review's case: a sixth of a fold SD
        self.assertIsNone(clear_best([opt("a", 0.70, 0.0), opt("b", 0.70, 0.0)], True))  # no spread to measure the noise by
        self.assertIsNone(clear_best([opt("a", 0.70, None), opt("b", 0.90, 0.01)], True))  # a spread unknown
        self.assertIsNone(clear_best([opt("a", 0.70, 0.01)], True))  # one option: no choice to judge
        self.assertIsNone(clear_best([opt("a", 0.70, 0.01), opt("b", 0.80, 0.01)], None))  # the direction unknown

    def test_an_error_is_better_when_lower_and_an_ineligible_option_is_never_the_truth(self):
        from dclab_core.measured import clear_best

        errors = [opt("ridge", 8.58, 0.40), opt("hist_gradient_boosting", 8.94, 0.05), opt("extra_trees", 9.80, 0.10)]
        self.assertIsNone(clear_best(errors, False))  # ridge leads, but not by the combined spread over the boosting
        self.assertEqual(clear_best([opt("ridge", 7.9, 0.1), opt("hgb", 8.94, 0.05), opt("et", 9.8, 0.1)], False)[0], "ridge")
        forecasts = [opt("seasonal_naive", 0.40, 0.01, eligible=False), opt("global_gbm", 0.66, 0.02, eligible=True), opt("per_series", 0.84, 0.03, eligible=True)]
        self.assertEqual(clear_best(forecasts, False)[0], "global_gbm")  # the baseline measured lower but is never chosen


@unittest.skipIf(MISSING, MISSING)
class CaptureTests(unittest.TestCase):
    def setUp(self):
        from dclab_rnd import ledger

        self.book = ledger.FileLedger(Path(tempfile.mkdtemp()), "w")
        ledger.install(self.book)
        self.addCleanup(ledger.install, None)

    def labels(self, point):
        return [(l.value, l.source, l.evidence.kind if l.evidence else None) for l in self.book.labels(point=point)]

    def stage(self, stage, options, chosen, higher=True, project="p-planted"):
        from dclab_rnd.ledger import capture

        prepared = SimpleNamespace(bundle=SimpleNamespace(spec=SimpleNamespace(higher_is_better=higher)))
        record = {"decision": {"chosen": chosen, "selected": chosen, "options": options}, "evidence": {}, "primary_metric": "roc_auc", "task_type": "binary"}
        capture.stage_decisions(stage, record, prepared, {"id": project, "data": {"rows": 1000, "columns": ["a", "b"]}, "settings": {}}, project=project)

    def test_a_planted_family_validation_clearly_beats_is_labelled_wrong_and_reaches_its_set(self):
        from dclab_core import datasets

        self.stage("models", [opt("logistic_regression", 0.70, 0.01), opt("lightgbm", 0.80, 0.01)], "logistic_regression")
        self.assertEqual(self.labels("model.family"), [("lightgbm", "measured", "validation_clear_best")])
        rec = next(r for r in self.book.records(point="model.family"))
        self.assertEqual(rec.value_used, "logistic_regression")  # the rule chose it: the measured label says it was wrong
        doc = {**rec.model_dump(mode="json"), "labels": [l.model_dump(mode="json") for l in self.book.labels(point="model.family")]}
        rows, manifest = datasets.decisions([doc], "model.family")
        self.assertEqual((rows[0]["labels"].get("measured"), rows[0]["label"]), ("lightgbm", {"value": "lightgbm", "source": "measured"}))
        self.assertIn("measured", manifest.counts["labels_by_source"])

    def test_a_right_choice_is_labelled_right_and_noise_or_an_unknown_direction_says_nothing(self):
        self.stage("features", [opt("raw", 0.80, 0.01), opt("poly2", 0.70, 0.01)], "raw", project="p-right")
        self.assertEqual(self.labels("features.recipe"), [("raw", "measured", "validation_clear_best")])
        self.stage("features", [opt("raw", 0.80, 0.05), opt("poly2", 0.78, 0.05)], "raw", project="p-noise")
        self.stage("models", [opt("a", 0.70, 0.01), opt("b", 0.90, 0.01)], "a", higher=None, project="p-unknown")
        self.assertEqual(len(self.book.labels(point="features.recipe")), 1)
        self.assertEqual(self.book.labels(point="model.family"), [])

    def test_the_holdout_judges_no_tuning_and_a_reuse_gets_no_made_up_label(self):
        from dclab_rnd.ledger import capture

        record = {"decision": {}, "primary_metric": "roc_auc", "task_type": "binary",
                  "evidence": {"tuning_decision": {"accepted": True, "gain": 0.01}, "holdout_primary_metric_ci": {"low": 0.6, "high": 0.68},
                               "configuration_results": [{"config_id": "lightgbm_baseline", "metrics": {"roc_auc": {"mean": 0.7}}}]}}
        capture.stage_decisions("final", record, None, {"id": "p-t", "data": {}, "settings": {}}, project="p-t")
        self.assertEqual([r.point for r in self.book.records(point="tuning.accept")], ["tuning.accept"])  # recorded, the rule's answer on it
        self.assertEqual(self.book.labels(point="tuning.accept"), [])  # the default never read on that holdout: the owner's call (holdout once)
        verdict = SimpleNamespace(move="run_stage", args={"stage": "final", "reuse_reason": "a person's reason"}, actor="human", allowed=True,
                                  node_from="done", message="")
        capture.move(verdict, {"allowed": ["run_stage"]}, {"holdout_uses": 1, "stages": {}}, "p-reuse")
        self.assertEqual(self.book.labels(point="holdout.reuse"), [])

    def test_an_error_metric_labels_the_lowest_and_a_persons_clearly_worse_choice_is_labelled_wrong(self):
        from dclab_rnd.ledger import capture

        self.stage("models", [opt("ridge", 7.9, 0.1), opt("extra_trees", 9.8, 0.1)], "extra_trees", higher=False, project="p-mae")
        self.assertEqual(self.labels("model.family"), [("ridge", "measured", "validation_clear_best")])
        record = {"decision": {"chosen": "lightgbm", "selected": "lightgbm", "options": [opt("lightgbm", 0.80, 0.01), opt("logistic_regression", 0.70, 0.01)]},
                  "primary_metric": "roc_auc"}  # the person keeps logistic regression over the rule's clearly better pick
        capture.approval("models", record, "logistic_regression", "human", "p-person")
        rec = next(r for r in self.book.records(point="model.family") if r.project == "p-person")
        self.assertEqual(rec.value_used, "logistic_regression")
        self.assertEqual(sorted((l.value, l.source) for l in self.book.labels(record=rec.id)),
                         [("lightgbm", "measured"), ("logistic_regression", "user_decision_exposed")])  # the person's own choice is kept apart
        from dclab_core import ledger as contract

        self.assertEqual(contract.effective_label(rec, self.book.labels(record=rec.id)).value, "lightgbm")  # the measurement counts first

    def test_real_runs_label_what_validation_tells_apart_in_each_direction(self):
        from dclab_core.measured import clear_best
        from dclab_rnd import kinds, ledger
        from dclab_rnd.expansion import runner
        from dclab_rnd.kinds import acceptance as A

        labelled = 0
        for key, seed, higher in (("binary", 3, True), ("forecasting", 1, False)):  # roc_auc; MASE, lower is better
            book = ledger.FileLedger(Path(tempfile.mkdtemp()), "w")
            ledger.install(book)
            kind = kinds.get(key)
            with runner.fast_profile():
                run = A.engine_run(kind.case(seed), getattr(kind, "engine_key", key))
            for point, stage in (("features.recipe", "features"), ("model.family", "models")):
                want = clear_best(run.records[stage]["decision"]["options"], higher)
                got = [x.value for x in book.labels(point=point) if x.source == "measured"]
                self.assertEqual(got, [want[0]] if want else [], (key, point))
                labelled += len(got)
        self.assertGreaterEqual(labelled, 2)  # binary 3's family and forecasting's family: the test reads real labels


@unittest.skipIf(MISSING, MISSING)
class FreshDrawTests(unittest.TestCase):
    """A self-play tuning judged on new rows of its replica's process: a planted tuning that a fresh draw contradicts is
    labelled wrong (the rule kept it) and read into its set; the project's own holdout is never read for it."""

    def test_a_planted_tuning_a_fresh_draw_contradicts_is_labelled_wrong_and_reaches_its_set(self):
        from dclab_core import datasets
        from dclab_rnd import kinds, ledger
        from dclab_rnd.expansion import runner
        from dclab_rnd.kinds import acceptance as A
        from dclab_rnd.lake import selfplay
        from dclab_rnd.lake.replicate import KIND

        book = ledger.FileLedger(Path(tempfile.mkdtemp()), "selfplay")
        ledger.install(book)
        self.addCleanup(ledger.install, None)
        kind = kinds.get("regression")
        with runner.fast_profile():
            run = A.engine_run(kind.case(1), kind.key)
        final = run.records["final"]["evidence"]
        self.assertEqual(final["model"], "ridge")
        # the plant: a candidate that shrinks every coefficient to nothing, kept by the rule as if it had measured a gain
        final["configuration_results"].append({"config_id": "ridge_planted", "model_params": {"model__alpha": 1e9}})
        final["tuning_decision"] = {"accepted": True, "best_candidate": "ridge_planted", "gain": 0.5, "required_gain": 0.001}
        from types import SimpleNamespace

        from dclab_rnd.ledger import capture

        prepared = SimpleNamespace(bundle=SimpleNamespace(spec=SimpleNamespace(higher_is_better=False)))
        capture.stage_decisions("final", run.records["final"], prepared, {"id": "p-tune", "data": {}, "settings": {}}, project="p-tune")
        selfplay._label_tuning(book, "regression", 1, run)
        rec = next(r for r in book.records(point="tuning.accept") if r.project == "p-tune")
        self.assertIs(rec.value_used, True)  # the rule kept it
        got = [(l.value, l.source, l.evidence.kind) for l in book.labels(record=rec.id)]
        self.assertEqual(got, [(False, "measured", KIND)])  # a fresh draw says keeping it was wrong
        self.assertLess(book.labels(record=rec.id)[0].evidence.value, 0)
        doc = {**rec.model_dump(mode="json"), "labels": [l.model_dump(mode="json") for l in book.labels(record=rec.id)]}
        rows, _ = datasets.decisions([doc], "tuning.accept")
        self.assertEqual(rows[0]["label"], {"value": False, "source": "measured"})

    def test_no_candidate_or_another_process_gives_no_label(self):
        from dclab_rnd import kinds
        from dclab_rnd.expansion import runner
        from dclab_rnd.kinds import acceptance as A
        from dclab_rnd.lake.replicate import tuning_on_fresh_draw

        kind = kinds.get("forecasting")  # its own machine tunes nothing
        with runner.fast_profile():
            run = A.engine_run(kind.case(1), kind.engine_key if hasattr(kind, "engine_key") else kind.key)
        self.assertIsNone(tuning_on_fresh_draw(kind, 1, run))


@unittest.skipIf(MISSING, MISSING)
class SelfPlayTests(unittest.TestCase):
    """Self-play's ledgers give decision rows in their solve's group and split, with every label the ledger holds; a solve
    not verified, one recorded again that failed, and one stopped before its holdout give nothing."""

    @classmethod
    def setUpClass(cls):
        from unittest import mock

        from dclab_rnd import kinds
        from dclab_rnd.expansion import runner
        from dclab_rnd.kinds import acceptance as A
        from dclab_rnd.lake import selfplay

        cls.out = Path(tempfile.mkdtemp())
        kind = kinds.get("binary")
        with mock.patch.object(selfplay, "OUT", cls.out), runner.fast_profile():
            for seed in (3, 4, 6, 7):
                selfplay._run_recorded(A, kind.case(seed), kind.key, f"selfplay:binary:{seed}")
        half = cls.out / "ledgers" / "binary" / "7" / "outcomes"  # seed 7: a run stopped before its holdout
        for path in half.rglob("*.jsonl"):
            path.unlink()
        done = [{"group": "selfplay:binary:3", "verified": True}, {"group": "selfplay:binary:4", "verified": False},
                {"group": "selfplay:binary:6", "verified": True}, {"group": "selfplay:binary:7", "verified": True}]
        (cls.out / "done.jsonl").write_text("".join(json.dumps(d) + "\n" for d in done), encoding="utf-8")
        (cls.out / "backfill_done.jsonl").write_text(json.dumps({"group": "selfplay:binary:6", "verified": False}) + "\n", encoding="utf-8")

    def test_only_a_whole_verified_solve_gives_rows_in_its_group_and_split(self):
        from dclab_core import lake as L
        from dclab_rnd import ledger
        from dclab_rnd.lake import selfplay

        excluded = {}
        rows = selfplay.decision_rows(self.out, excluded)
        self.assertEqual(excluded, {"a solve not verified": 1, "a solve that failed its checks when recorded again": 1,
                                    "a ledger whose run stopped before its holdout": 1})  # each said, with why
        groups = {r["group"] for rs in rows.values() for r in rs}
        self.assertEqual(groups, {"selfplay:binary:3"})  # 4 not verified, 6 failed when recorded again, 7 stopped half way
        self.assertTrue(all(r["split"] == L.split_of("selfplay:binary:3") for rs in rows.values() for r in rs))
        book = ledger.FileLedger(self.out / "ledgers" / "binary" / "3", "selfplay")
        measured = {(x.point, json.dumps(x.value)) for x in book.labels() if x.source == "measured"}
        in_rows = {(p, json.dumps(r["labels"]["measured"])) for p, rs in rows.items() for r in rs if r["labels"].get("measured") is not None}
        self.assertEqual(in_rows, measured)  # every measured label the solve wrote, and nothing else

    def test_a_time_asked_about_is_read_as_a_time_never_as_text(self):
        from dclab_rnd.lake.selfplay import _when

        self.assertEqual(_when("2026-10-18 17:30"), _when("2026-10-18T17:30:00+00:00"))  # a space, no offset: UTC
        self.assertEqual(_when("2026-10-18T17:30:00Z"), _when("2026-10-18T17:30:00+00:00"))
        self.assertLess(_when("2026-10-18T09:00:00+00:00"), _when("2026-10-18 17:30"))  # as text the 'T' would sort it after
        with self.assertRaises(ValueError):
            _when("yesterday")

    def test_the_lake_reads_them_and_keeps_an_evaluated_table_out(self):
        from unittest import mock

        from dclab_rnd import lake
        from dclab_rnd.lake import selfplay

        with mock.patch.object(selfplay, "OUT", self.out):
            run = lake.Run()
            kept = lake.selfplay_decisions(run, seeds=set())
            gone = lake.selfplay_decisions(lake.Run(), seeds={3})
        self.assertTrue(kept and all(s.provenance.source == "selfplay/ledgers" and s.provenance.group == "selfplay:binary:3" for s in kept))
        self.assertTrue(any(i["path"].endswith("done.jsonl") for i in run.inputs))
        self.assertTrue(all(s.provenance.split == "sealed" for s in gone))  # seed 3 read by an evaluation: only a sealed row could stay


if __name__ == "__main__":
    unittest.main()
