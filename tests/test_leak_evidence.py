"""2026-11-19: a column suspected of leakage by its name alone stayed in an unattended build (the held-out comparison
E2E-004: rules alone claimed macro F1 0.94 on tickets and got 0.28). Each suspect is now weighed on the rows a build
reads before its holdout: its timing, whether it is too good alone, and the score with and without it; the person is told."""
import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import leak_evidence as L  # noqa: E402


def table(n=800, seed=0):
    """Rows in time; ``minutes`` from distance and noise; honest and late columns beside it."""
    rng = np.random.default_rng(seed)
    t = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 300, n)), unit="D")
    distance = rng.uniform(0, 10, n)
    load = rng.uniform(0, 1, n)
    minutes = 10 + 3 * distance + 5 * load + rng.normal(0, 4, n)
    return pd.DataFrame({
        "at": t, "distance": distance, "load": load,
        "delivered_at": t + pd.to_timedelta(minutes, unit="m"),  # after the row's own time: late
        "signup_at": t - pd.to_timedelta(rng.uniform(1, 500, n), unit="D"),  # before it: existed in time
        "noise_rating": rng.normal(0, 1, n),  # a scary name, no signal, no time: nothing proves it either way
        "minutes": minutes,
    })


def by_column(evidence):
    return {c.column: c for c in evidence.columns}


class LeakEvidenceTests(unittest.TestCase):
    def test_a_time_after_the_rows_own_time_is_late_and_kept_out(self):
        c = by_column(L.weigh(table(), "minutes", "regression", ["delivered_at"], ["distance", "load"], "at"))["delivered_at"]
        self.assertEqual((c.decision, c.basis), ("keep_out", "late"))
        self.assertEqual(c.late_share, 1.0)
        self.assertIn("does not exist yet when you predict", c.why)

    def test_a_time_before_the_rows_own_time_stays_in(self):
        c = by_column(L.weigh(table(), "minutes", "regression", ["signup_at"], ["distance", "load"], "at"))["signup_at"]
        self.assertEqual((c.decision, c.basis), ("keep_in", "early"))

    def test_a_column_that_alone_beats_every_other_column_is_too_good(self):
        t = table()
        t["final_minutes"] = t["minutes"].round()  # the outcome itself, written down after the event
        c = by_column(L.weigh(t, "minutes", "regression", ["final_minutes"], ["distance", "load"], "at"))["final_minutes"]
        self.assertEqual((c.decision, c.basis), ("keep_out", "too_good"))
        self.assertLess(c.alone, c.rest)  # MAE: lower is better
        self.assertGreater(c.gain, 0)

    def test_too_good_on_classes_too(self):
        t = table()
        t["team"] = np.where(t["distance"] > 5, "far", "near")
        t["resolved_by"] = t["team"].map({"far": "ann", "near": "bob"})
        c = by_column(L.weigh(t.drop(columns="minutes"), "team", "binary", ["resolved_by"], ["load"], "at"))["resolved_by"]
        self.assertEqual(c.basis, "too_good")
        self.assertEqual(c.alone, 1.0)

    def test_no_evidence_either_way_follows_unsure_and_says_so(self):
        e = L.weigh(table(), "minutes", "regression", ["noise_rating"], ["distance", "load"], "at")
        c = by_column(e)["noise_rating"]
        self.assertEqual((c.decision, c.basis, e.unsure), ("keep_out", "unsure", "keep_out"))
        self.assertIn("put it back", c.why)
        self.assertIn("Keeping it out changes the score", c.why)  # the cost, so the person can judge
        kept = by_column(L.weigh(table(), "minutes", "regression", ["noise_rating"], ["distance", "load"], "at", unsure="keep_in"))["noise_rating"]
        self.assertEqual(kept.decision, "keep_in")

    def test_blank_on_the_most_recent_rows_is_filled_in_later(self):
        t = table(n=1000)
        t["courier_rating"] = np.random.default_rng(1).normal(4, 0.5, len(t))
        from dclab_core.modifiers import provisional_reading

        rows, _ = provisional_reading(t.index, t["at"])
        recent = rows[int(len(rows) * 0.8):]
        t.loc[recent, "courier_rating"] = np.nan  # not rated yet on the latest rows a build reads
        c = by_column(L.weigh(t, "minutes", "regression", ["courier_rating"], ["distance", "load"], "at"))["courier_rating"]
        self.assertEqual((c.decision, c.basis), ("keep_out", "late"))
        self.assertGreaterEqual(c.blank_recent, L.FILLED_LATER)

    def test_only_the_rows_before_the_holdout_are_read(self):
        """A column perfect only on the latest quarter (the rows a later holdout takes) is never judged on them."""
        t = table(n=1000)
        t["leaky_later"] = np.random.default_rng(2).normal(0, 1, len(t))
        latest = t.index[t["at"] >= t["at"].quantile(0.8)]
        t.loc[latest, "leaky_later"] = t.loc[latest, "minutes"]
        e = L.weigh(t, "minutes", "regression", ["leaky_later"], ["distance", "load"], "at")
        self.assertTrue(e.by_time)
        self.assertLessEqual(e.rows_read, 750)
        self.assertNotEqual(by_column(e)["leaky_later"].basis, "too_good")

    def test_too_few_rows_fits_no_model(self):
        c = by_column(L.weigh(table(n=40), "minutes", "regression", ["noise_rating"], ["distance"], "at"))["noise_rating"]
        self.assertEqual((c.alone, c.rest, c.with_it, c.gain), (None, None, None, None))
        self.assertEqual(c.basis, "unsure")

    def test_the_contract_reads_back_and_holds_no_table_value(self):
        e = L.weigh(table(), "minutes", "regression", ["delivered_at", "noise_rating"], ["distance", "load"], "at")
        again = L.LeakEvidence.read(e.model_dump_json())
        self.assertEqual(again, e)
        self.assertEqual(again.schema_version, "dclab_core.leak_evidence.v1")


    def test_dates_read_from_a_csv_are_still_times(self):
        """A CSV keeps dates as text: the timing check reads them (the review: both came out unsure)."""
        import tempfile

        path = Path(tempfile.mkdtemp()) / "t.csv"
        table().to_csv(path, index=False)
        t = pd.read_csv(path)
        self.assertFalse(pd.api.types.is_datetime64_any_dtype(t["delivered_at"]))
        e = by_column(L.weigh(t, "minutes", "regression", ["delivered_at", "signup_at"], ["distance", "load"], "at"))
        self.assertEqual((e["delivered_at"].basis, e["signup_at"].basis), ("late", "early"))

    def test_a_number_as_the_time_column_never_makes_a_date_late(self):
        t = table().assign(elapsed=np.arange(800) * 60.0)
        c = by_column(L.weigh(t, "minutes", "regression", ["signup_at"], ["distance", "load"], "elapsed"))["signup_at"]
        self.assertIsNone(c.late_share)
        self.assertNotEqual(c.basis, "late")

    def test_a_time_with_a_zone_beside_one_without_is_compared(self):
        t = table()
        t["delivered_at"] = t["delivered_at"].dt.tz_localize("UTC")  # a zone beside a naive time used to raise TypeError
        c = by_column(L.weigh(t, "minutes", "regression", ["delivered_at"], ["distance", "load"], "at"))["delivered_at"]
        self.assertEqual(c.basis, "late")

    def test_noise_is_never_called_too_good(self):
        """The review: with no margin, a noise column tied a weak rest by chance on 7 of 20 seeds (1 of 60 with a 0.05 or 0.10 margin)."""
        for seed in range(60):
            t = table(seed=seed)
            t["outcome"] = np.where(np.random.default_rng(seed + 100).random(len(t)) < 0.3, "yes", "no")
            c = by_column(L.weigh(t, "outcome", "binary", ["noise_rating"], ["load"], "at"))["noise_rating"]
            self.assertNotEqual(c.basis, "too_good", seed)

    def test_a_long_column_name_is_kept(self):
        t = table().rename(columns={"noise_rating": "x" * 300 + "_rating"})
        e = L.weigh(t, "minutes", "regression", ["x" * 300 + "_rating"], ["distance", "load"], "at")
        self.assertEqual(len(e.columns), 1)


def frame_proposal(t):
    return {"task": "regression", "forbidden": [
        {"column": "delivered_at", "reason": "name suggests a post-outcome value"},
        {"column": "noise_rating", "reason": "name suggests a post-outcome value"},
        {"column": "distance", "reason": "single column tracks the target (|Spearman| 0.950)"},  # a signal, not a name
        {"column": "proven", "reason": "name suggests a post-outcome value"},
        {"column": "order_id", "reason": "name suggests a post-outcome value; identifier-like"},
        {"column": "review_text", "reason": "name suggests a post-outcome value"}]}


class StandardPlanTests(unittest.TestCase):
    """The intern's unattended plan: a column suspected by its name is weighed, kept out on evidence, and the report says why."""

    def weigh(self, t, suggestion=None, **kw):
        from dclab_rnd.intern.loop import Intern

        return Intern._weigh_suspects(t, "minutes", kw.get("proposal") or frame_proposal(t), suggestion or {"text_columns": ["review_text"]},
                                      {"proven": {}}, {"at", "order_id"}, "at")

    def test_only_columns_suspected_by_their_name_are_weighed(self):
        """Not a proven leak, a role, a text column, or a column flagged by a signal of its values (often an honest strong feature)."""
        e = self.weigh(table().assign(review_text="ok"))
        self.assertEqual([c["column"] for c in e["columns"]], ["delivered_at", "noise_rating"])
        self.assertIsNone(self.weigh(table(), proposal={"task": "regression", "forbidden": []}))

    def test_a_studied_samples_columns_are_left_as_the_rnd_decided(self):
        self.assertIsNone(self.weigh(table(), {"sample": "bike_sharing_daily"}))

    def test_a_check_that_cannot_run_keeps_the_suspects_out_and_says_why(self):
        from dclab_rnd.intern.loop import Intern

        with unittest.mock.patch.object(L, "weigh", side_effect=TypeError("boom")):
            e = self.weigh(table())
        self.assertEqual(e["error"], "TypeError")
        self.assertEqual({c["decision"] for c in e["columns"]}, {"keep_out"})
        self.assertIn("could not run (TypeError)", Intern._report(None, "p1", {}, e))

    def test_the_report_tells_the_person_each_decision_and_which_rows(self):
        from dclab_rnd.intern.loop import Intern

        e = self.weigh(table())
        text = Intern._report(None, "p1", {}, e)
        self.assertIn("checked on the earliest training rows", text)
        self.assertIn("`delivered_at`", text)
        self.assertIn("Kept out of the model", text)
        no_time = L.weigh(table(), "minutes", "regression", ["noise_rating"], ["distance"], None).model_dump(mode="json")
        self.assertIn("a random holdout may overlap", Intern._report(None, "p1", {}, no_time))


if __name__ == "__main__":
    unittest.main()
