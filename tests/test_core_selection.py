"""34.6, honest selection under many attempts: the best of many candidates scored on the same folds is flattered by
its luck, more so the more attempts; a selection fold set aside before the search, read once to choose among the top
K, keeps the chosen one's score honest whatever the number of attempts.

Simulated candidates with a known true score each: their search score is the truth plus fold noise, the selection fold
and the holdout each give another noisy reading. Plain cross-validation's winner is flattered more as attempts grow;
the protocol's selection score stays near the holdout's at every count.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

try:
    from dclab_core import selection  # noqa: F401

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)


def trial(attempts, seed, top_k=1):
    """One search: (plain winner's search score - its holdout), (protocol's selection score - its holdout)."""
    from dclab_core.selection import choose

    rng = np.random.default_rng(seed)
    truth = rng.normal(0.70, 0.01, attempts)  # candidates of about equal true quality: the differences are mostly luck
    search = truth + rng.normal(0, 0.02, attempts)
    on_selection = truth + rng.normal(0, 0.03, attempts)
    holdout = truth + rng.normal(0, 0.03, attempts)
    names = [f"c{i}" for i in range(attempts)]
    record = choose(dict(zip(names, search)), lambda k, rows: on_selection[names.index(k)] if rows is None else None,
                    metric="roc_auc", top_k=top_k, bootstrap=0)
    plain = int(np.argmax(search))
    chosen = names.index(record.chosen)
    return search[plain] - holdout[plain], record.selection_score - holdout[chosen]


@unittest.skipIf(MISSING, MISSING)
class SelectionTests(unittest.TestCase):
    def test_plain_cross_validation_is_flattered_more_as_attempts_grow_and_the_protocol_is_not(self):
        """Measured 2026-10-25 (400 searches each): plain 0.019, 0.039, 0.051 at 5, 50 and 200 attempts; confirming the
        winner on the selection fold (K=1) 0.002, -0.003, 0.002; choosing among three there 0.024, 0.023, 0.027."""
        seeds = range(400)
        plain, honest, among3 = {}, {}, {}
        for attempts in (5, 50, 200):
            gaps = np.array([trial(attempts, s) for s in seeds])
            plain[attempts], honest[attempts] = gaps[:, 0].mean(), gaps[:, 1].mean()
            among3[attempts] = np.array([trial(attempts, s, top_k=3) for s in seeds])[:, 1].mean()
        self.assertLess(plain[5], plain[50])
        self.assertLess(plain[50], plain[200])
        self.assertGreater(plain[200], 0.04)  # the winner of 200 lucky draws looks four points better than it is
        for attempts in (5, 50, 200):
            self.assertLess(abs(honest[attempts]), 0.006, (attempts, honest[attempts]))  # the default: honest at every count
        self.assertLess(abs(among3[200] - among3[5]), 0.01)  # choosing among three: its own optimism, not growing with attempts
        self.assertGreater(among3[5], honest[5] + 0.01)  # and said in the record

    def test_the_record_says_what_happened(self):
        from dclab_core.selection import SelectionRecord, choose

        scores = {"logistic": 0.80, "forest": 0.83, "boosting": 0.82, "knn": 0.70}
        held = {"logistic": 0.79, "forest": 0.76, "boosting": 0.81, "knn": 0.69}
        rng_rows = 50
        record = choose(scores, lambda k, rows: held[k] if rows is None else held[k] + 0.01 * (rows.mean() / rng_rows - 0.5),
                        metric="roc_auc", split="group", search_rows=400, selection_rows=rng_rows, bootstrap=100, top_k=3)
        self.assertEqual(record.plain_choice, "forest")
        self.assertEqual(record.shortlist, ["forest", "boosting", "logistic"])
        self.assertEqual(record.chosen, "boosting")
        self.assertAlmostEqual(record.optimism, 0.01, places=6)
        self.assertIn("plain cross-validation would have kept forest", record.note)
        self.assertIn("upper estimate", record.note)  # chosen among three on the selection fold
        self.assertEqual(record.schema_version, "dclab_core.selection_record.v1")
        lo, hi = record.selection_interval
        self.assertLessEqual(lo, record.selection_score + 0.01)
        self.assertIsInstance(SelectionRecord.model_validate(record.model_dump()), SelectionRecord)
        lower = choose({"a": 0.5, "b": 0.4}, lambda k, rows: {"a": 0.55, "b": 0.45}[k], metric="mae", higher_is_better=False, bootstrap=0, top_k=2)
        self.assertEqual(lower.chosen, "b")  # a lower error is better
        self.assertAlmostEqual(lower.optimism, 0.05, places=6)  # 0.40 on the search rows, 0.45 on the selection fold
        with self.assertRaises(ValueError):
            choose({"a": float("nan")}, lambda k, rows: 0.5, metric="roc_auc")

    def test_the_selection_fold_is_cut_the_way_the_project_splits(self):
        from dclab_core.selection import selection_split

        n = 1000
        order = np.random.default_rng(0).permutation(n)
        search, held, kind = selection_split(n, order=order, share=0.2)
        self.assertEqual(kind, "time")
        self.assertGreater(order[held].min(), order[search].max())  # the latest rows, never mixed in time
        groups = np.repeat(np.arange(100), 10)
        search, held, kind = selection_split(n, groups=groups, share=0.2)
        self.assertEqual(kind, "group")
        self.assertFalse(set(groups[search]) & set(groups[held]))  # whole entities on one side
        y = (np.arange(n) % 10 == 0).astype(int)
        search, held, kind = selection_split(n, y=y, share=0.2)
        self.assertEqual(kind, "stratified")
        self.assertEqual(int(y[held].sum()), 20)  # each class keeps its share
        self.assertEqual(len(search) + len(held), n)
        with self.assertRaises(ValueError):
            selection_split(n, share=0.6)
        with self.assertRaises(ValueError):
            selection_split(n, groups=np.zeros(n), share=0.2)  # one entity cannot be set aside whole


if __name__ == "__main__":
    unittest.main()
