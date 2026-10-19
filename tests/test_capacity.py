"""Package 30.12 (its statistics): a queue of K a period, its threshold, its precision on confirmed labels, its recall
estimated only from the rows reviewed at random within their stratum and period, and the rule that built a noisy
source's labels. On the retail returns replica the recall's interval covers the true recall on every seed."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401

    HAVE_ML = True
except ImportError:  # the agent environment
    HAVE_ML = False


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class QueueTests(unittest.TestCase):
    def test_a_queue_takes_k_a_period_and_its_threshold_is_the_kth_score_of_an_average_period(self):
        import numpy as np

        from dclab_core import capacity as C

        scores = [0.9, 0.8, 0.1, 0.7, 0.6, 0.5, 0.4]
        periods = [1, 1, 1, 2, 2, 2, 2]
        self.assertEqual(C.queue(scores, periods, 2).tolist(), [True, True, False, True, True, False, False])
        self.assertEqual(C.kth_threshold(scores, periods, 2), round(float(np.median([0.8, 0.6])), 6))
        y = [1, np.nan, 0, 0, 1, 1, np.nan]  # a queued case without a confirmed label is not counted, and said
        p = C.precision_at_k(scores, periods, 2, y)
        self.assertEqual((p["queued"], p["queued_confirmed"], p["hits"], p["precision"], p["unconfirmed_share"]), (4, 3, 2, 0.6667, 0.25))

    def test_the_queues_precision_weighs_each_reviewed_row_by_its_cell(self):
        import numpy as np

        from dclab_core import capacity as C

        # one period: the old rule's stratum (2 rows, both reviewed) and the rest (8 rows, 1 reviewed at random)
        scores = [0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        y = [1, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        reviewed = [True, True, True] + [False] * 7
        stratum = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        got = C.precision_at_k(scores, [1] * 10, 3, y, reviewed, stratum)
        self.assertEqual(got["raw_precision"], 0.3333)  # 1 of 3, as counted
        self.assertEqual(got["precision"], 0.1)  # the audited row stands for 8: 1 of (1 + 1 + 8)

    def test_a_stratum_never_reviewed_is_said_not_guessed(self):
        import numpy as np

        from dclab_core import capacity as C

        caught = [True, False, True, False]
        y = [1, 0, np.nan, np.nan]
        got = C.audit_recall(caught, y, [True, True, False, False], ["a", "a", "b", "b"], [1, 1, 1, 1], draws=10)
        self.assertEqual((got["recall"], got["unsampled_share"]), (1.0, 0.5))

    def test_a_noisy_sources_rule_is_found_and_its_agreement_measured(self):
        import numpy as np
        import pandas as pd

        from dclab_core import capacity as C

        frame = pd.DataFrame({"late": [1] * 10 + [0] * 30, "amount": range(40)})
        source = ["rule"] * 10 + ["expert"] * 10 + [""] * 20
        self.assertEqual([r["column"] for r in C.rule_columns(frame, source, ["rule"])], ["late"])
        got = C.agreement([1, 1, 1, 0], [1, 0, np.nan, 1])
        self.assertEqual((got["checked"], got["right"], got["agreement"]), (2, 1, 0.5))


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class ReplicaTests(unittest.TestCase):
    """The replica's team checks the rule's hits and a random part of the rest each day: the estimate from those rows
    alone must cover what the queue truly catches, on every seed."""

    def test_the_audit_estimate_covers_the_true_recall_and_the_rules_are_found(self):
        import numpy as np
        import pandas as pd

        from dclab_core import capacity as C
        from dclab_core.replicas import retail_returns as rr

        for seed in (0, 1, 2):
            rep = rr.generate(seed=seed)
            r = rep.returns.copy()
            r["returned_at"] = pd.to_datetime(r["returned_at"])
            fraud = r["return_id"].map(rep.truth.set_index("return_id")["fraud"]).to_numpy()
            day = (r["returned_at"] - r["returned_at"].min().normalize()).dt.days.to_numpy()
            reviewed = r["label_source"].isin(["expert", "expert_audit"]).to_numpy()
            y = np.where(reviewed, r["label"], np.nan)
            hold = day >= day.max() - 89
            hour = (r["returned_at"].dt.hour + r["returned_at"].dt.minute / 60).to_numpy()
            q = C.queue(hour[hold], day[hold], rr.CAPACITY)
            true = float((q & (fraud[hold] == 1)).sum() / (fraud[hold] == 1).sum())
            got = C.audit_recall(q, y[hold], reviewed[hold], r["rule_flag"].to_numpy()[hold], day[hold], draws=500, seed=seed)
            self.assertTrue(got["recall_ci"][0] <= true <= got["recall_ci"][1], (seed, true, got))
            found = {(x["source"], x["column"]) for x in C.rule_columns(r.drop(columns=["label"]), r["label_source"], ["rule", "cluster"],
                                                                           exclude=["label_source", "review_status"])}
            self.assertIn(("rule", "rule_flag"), found)  # the late-evening rule that labelled the first months
            self.assertIn(("cluster", "refund_method"), found)  # "similar to a confirmed case": cash refunds


if __name__ == "__main__":
    unittest.main()
