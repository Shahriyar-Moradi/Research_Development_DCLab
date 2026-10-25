"""dclab-core 0.110.1: a running number in both files is never a label-join key, whatever its name; a real key whose
table ids run 1 to n stays one (the planted trap of 2026-10-25: the first rows marked as the cases)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core.analyse import _running, label_join  # noqa: E402


def files(seed: int = 0, n: int = 600, share: float = 0.1, name: str = "line"):
    rng = np.random.default_rng(seed)
    orders = rng.permutation(np.arange(100000, 100000 + n))
    table = pd.DataFrame({"order_id": orders, "amount": rng.gamma(2, 40, n).round(2), name: np.arange(1, n + 1)})
    k = 60
    labels = pd.DataFrame({"order_id": orders[rng.choice(n, k, replace=False)], "reason": "chargeback", name: np.arange(1, k + 1)})
    missing = labels.sample(frac=share, random_state=seed).index  # labelled cases this export does not hold
    labels.loc[missing, "order_id"] = 999999 - np.arange(len(missing))
    return table, labels


class RunningNumberTest(unittest.TestCase):
    def test_a_running_number_in_both_files_is_never_a_key(self):
        for name in ("line", "S.No", "ردیف", "n"):
            table, labels = files(name=name)
            keys = [j["key"] for j in label_join(table, labels)]
            self.assertNotIn(name, keys, name)
            self.assertIn("order_id", keys, name)

    def test_a_real_key_whose_table_ids_run_one_to_n_stays_a_key(self):
        n = 400
        table = pd.DataFrame({"order_id": np.arange(1, n + 1), "amount": np.arange(n) * 1.5})
        picked = np.sort(np.random.default_rng(1).choice(n, 40, replace=False)) + 1
        labels = pd.DataFrame({"order_id": picked, "reason": "chargeback"})  # a subset of the table's ids, not 1 to k again
        self.assertTrue(_running(table["order_id"]) and not _running(labels["order_id"]))
        self.assertEqual([j["key"] for j in label_join(table, labels)][0], "order_id")

    def test_what_counts_as_a_running_number(self):
        self.assertTrue(_running(pd.Series([0, 1, 2, 3])) and _running(pd.Series(["1", "2", "3"])))
        self.assertFalse(_running(pd.Series([3, 1, 2])))  # not in the file's order
        self.assertFalse(_running(pd.Series([2, 3, 4])))  # starts elsewhere
        self.assertFalse(_running(pd.Series([1.0, 2.5, 3.0])) or _running(pd.Series([1, None, 3])) or _running(pd.Series([1, 2])))


if __name__ == "__main__":
    unittest.main()
