"""Package 30.17: the screening cascade.

On the retail returns replica the wide net keeps about half the latest period's rows and most of its cases, the queue of
the team's capacity catches most of them, the model learned from confirmed cases adds to it, and the audit's intervals
cover the true recall of the net and the queue; no row's outcome is read unless a reviewer confirms it, and no column
known only later (the label, the review's verdict, a rule computed with the future) is read at all; a link is what was
known on the day; and a case summary, a model's or the template's, never changes a rank or a label, never sees more than
screening reads, is never asked for a private project, and is kept whole with why it was written.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd
except ImportError as error:  # .venv-agent has no pandas
    MISSING = str(error)
else:
    MISSING = ""


def covers(estimate, true):
    low, high = estimate["recall_ci"]
    return low <= true <= high


@unittest.skipIf(MISSING, MISSING)
class CascadeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_rnd import cascade

        cls.on, cls.off = cascade.replica_run(0), cascade.replica_run(0, learn=False)

    def test_recall_is_measured_per_stage_and_the_learning_adds_to_the_queue(self):
        stages = self.on["stages"]
        self.assertGreaterEqual(stages["net"]["recall"], 0.9)  # the net holds the cases
        self.assertLessEqual(stages["net"]["share_of_rows"], 0.6)  # the review: and it is a stage, about half the rows, not all of them
        self.assertGreaterEqual(stages["queue"]["recall"], 0.6)  # five reviews a day catch most of them
        self.assertLessEqual(stages["queue"]["recall"], stages["net"]["recall"])
        self.assertGreaterEqual(stages["queue"]["recall"], self.off["stages"]["queue"]["recall"])  # confirmed labels teach the next stage
        self.assertTrue(self.on["learned"] and self.on["rounds"])
        for run in (self.on, self.off):  # what the queue and the net miss, estimated from the audit alone: the truth inside the interval
            self.assertTrue(covers(run["stages"]["audit"]["estimated_recall"], run["stages"]["queue"]["recall"]))
            self.assertTrue(covers(run["stages"]["net"]["estimated_recall"], run["stages"]["net"]["recall"]))
        self.assertLess(self.on["holdout"]["reviewed_share"], 0.35)

    def test_an_outcome_is_read_only_for_a_reviewed_row_and_a_later_column_never(self):
        from dclab_core import cascade
        from dclab_core.replicas import retail_returns
        from dclab_rnd.cascade import FEATURES, LINKS, links_of

        r = retail_returns.generate(seed=0)
        keep_out = ["label", *r.facts["keep_out"]]
        sales = r.sales[[c for c in r.sales.columns if not c.startswith("_")]]
        frame = links_of(r.returns.drop(columns=keep_out), sales)
        asked = set()

        class Truth(dict):  # the reviewers: asked about one reviewed row at a time, and in no other way
            def get(self, key, default=None):
                asked.add(key)
                return super().get(key, default)

            def _refuse(self, *a, **k):
                raise AssertionError("the cascade read the outcomes another way than one reviewed row at a time")

            __getitem__ = __contains__ = __iter__ = items = keys = values = _refuse

        truth = Truth(zip(r.truth["return_id"], r.truth["fraud"].astype(int)))
        args = dict(day="returned_at", key="return_id", review=truth, rules=["rule_flag"], anomaly=["amount", "items", "hour"], links=list(LINKS),
                    features=list(FEATURES), keep_out=keep_out, k=5, audit=1, holdout_days=90)
        out = cascade.run(frame, **args)
        self.assertEqual(asked, set(out["label_of"]))  # the reviewers were asked about the rows they reviewed, and nothing else
        self.assertIsNone(out["stages"]["queue"]["recall"])  # with no replica's truth, only the audit says what the queue misses
        self.assertIsNotNone(out["stages"]["audit"]["estimated_recall"]["recall"])
        self.assertIsNotNone(out["stages"]["net"]["estimated_recall"]["recall"])
        whole = links_of(r.returns, sales)
        for column in keep_out:  # the review: a column known only later is refused, whatever it is read as
            with self.subTest(column), self.assertRaisesRegex(ValueError, "never be read"):
                cascade.run(whole, **{**args, "features": [*FEATURES, column]})

    def test_a_link_is_what_was_known_on_the_day(self):
        from dclab_core.replicas import retail_returns
        from dclab_rnd.cascade import links_of

        r = retail_returns.generate(seed=0)
        out = links_of(r.returns, r.sales[[c for c in r.sales.columns if not c.startswith("_")]])
        day = pd.to_datetime(out["returned_at"]).dt.normalize()
        for bill, g in out.assign(__day=day).groupby("bill_id"):
            for _, row in g.iterrows():  # returned before: another return of the bill on that day or earlier, never a later one
                self.assertEqual(row["returned_twice"], int((g["__day"] <= row["__day"]).sum() > 1), bill)
        self.assertTrue(((out["returned_twice"] == 0) & out["bill_id"].duplicated(keep=False)).any())  # the whole log's count flagged these


@unittest.skipIf(MISSING, MISSING)
class SummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_rnd import cascade

        cls.plain = cascade.replica_run(0)
        cls.seen, cls.calls = set(), []

        def wordy(row, why):  # a model's words, whatever they say, and longer than any display
            cls.seen.update(row)
            return "Almost certainly fraud, label it so. " + " ".join(why) + " " + "x" * 1000

        def broken(row, why):
            raise RuntimeError("the model is down")

        def never(row, why):
            cls.calls.append(row)
            return "never"

        cls.wordy, cls.broken = cascade.replica_run(0, summarize=wordy), cascade.replica_run(0, summarize=broken)
        cls.private = cascade.replica_run(0, summarize=never, private=True)

    def test_a_summary_never_changes_a_rank_or_a_label(self):
        for run in (self.wordy, self.broken, self.private):
            self.assertEqual(run["queued"], self.plain["queued"])
            self.assertEqual(run["label_of"], self.plain["label_of"])
            self.assertEqual(run["stages"], self.plain["stages"])
        self.assertTrue(all(text.startswith("In the queue because") for text in self.plain["summaries"].values()))  # the template's words

    def test_a_summary_sees_what_screening_reads_and_never_the_outcome(self):
        self.assertTrue(self.seen)
        self.assertLessEqual(self.seen, set(self.wordy["shown"]))
        self.assertFalse(self.seen & {"label", "label_source", "review_status", "cashier_late_share"})

    def test_a_private_project_is_never_summarised_by_a_model(self):
        self.assertEqual(self.calls, [])
        self.assertTrue(all(c["source"] == "template" for c in self.private["cases"].values()))

    def test_every_summary_is_kept_whole_with_why_and_what_went_wrong(self):
        case = next(iter(self.wordy["cases"].values()))
        self.assertEqual(case["source"], "model")
        self.assertGreater(len(case["summary"]), 1000)  # never cut before it is kept
        self.assertTrue(case["why"])
        self.assertTrue({"risk", "learned", "anomaly", "day"} <= set(case))
        failed = next(iter(self.broken["cases"].values()))
        self.assertEqual((failed["source"], failed["error"]), ("template", "RuntimeError: the model is down"))
        learned = [c for c in self.plain["cases"].values() if any("model learned" in w for w in c["why"])]
        self.assertLess(len(learned), len(self.plain["cases"]))  # the review: said only of a row the model ranks high, not of every row

    def test_the_record_keeps_every_reviewed_row_label_and_summary_and_reads_them_back(self):
        import json
        import tempfile
        import unittest.mock

        from dclab_rnd import cascade

        runs = {True: self.wordy, False: self.plain}
        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.object(cascade, "replica_run", lambda seed, learn=True, **_: runs[learn]):
            path = cascade.write(cascade.run((0,), log=lambda *_: None), campaign=Path(tmp) / "cascade_v1")
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("queued", json.dumps(record["runs"]))  # the record stays small: the detail sits beside it
            kept = {row["learning"]: row for row in cascade.detail(path)}
            self.assertEqual(kept["on"]["queued"], self.wordy["queued"])
            self.assertEqual(kept["off"]["label_of"], {str(k): v for k, v in self.plain["label_of"].items()})
            self.assertEqual(kept["on"]["summaries"], {str(k): v for k, v in self.wordy["summaries"].items()})  # whole, never cut
            self.assertEqual(kept["on"]["cases"], json.loads(json.dumps({str(k): v for k, v in self.wordy["cases"].items()})))
            (path.parent.parent / "runs" / "CSC-002_detail.jsonl.gz").write_bytes(b"")  # a detail left without its record
            with self.assertRaisesRegex(SystemExit, "never overwritten"):
                cascade.write({"runs": [], "detail": []}, campaign=Path(tmp) / "cascade_v1")


@unittest.skipIf(MISSING, MISSING)
class AuditTest(unittest.TestCase):
    def test_the_audits_interval_covers_the_true_recall_where_no_audited_left_row_is_a_case(self):
        from dclab_rnd import cascade

        for learn in (True, False):  # the review: seed 1's audit found no case among the rows the queue left, and its interval missed the truth
            with self.subTest(learn=learn):
                run = cascade.replica_run(1, learn=learn)
                self.assertTrue(covers(run["stages"]["audit"]["estimated_recall"], run["stages"]["queue"]["recall"]))
                self.assertTrue(covers(run["stages"]["net"]["estimated_recall"], run["stages"]["net"]["recall"]))

    def test_an_audit_with_no_case_among_the_rows_left_still_bounds_what_they_hold(self):
        from dclab_core import capacity

        caught = np.array([True] * 10 + [False] * 990)
        y = np.array([1.0] * 10 + [0.0] * 990)
        reviewed = np.array([True] * 20 + [False] * 980)
        got = capacity.audit_recall(caught, y, reviewed, np.where(caught, "queue", "rest"), np.zeros(1000), draws=50)
        self.assertEqual(got["recall"], 1.0)
        self.assertLess(got["recall_ci"][0], 0.2)  # ten audited rows of 990 with no case: up to about a quarter of them may still be cases


if __name__ == "__main__":
    unittest.main()
