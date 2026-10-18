"""Package 30.9: the retail returns replica (the owner's interview patterns 1 to 5): every planted trap is real on the
invented tables, on several seeds (it bites a careless solution and no single column gives the fraud away), the truth is
never written, and the files say they are synthetic."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core.replicas import retail_returns as rr  # noqa: E402

SEEDS = (0, 1, 2)


def _auc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    ranks = pd.Series(score).rank().to_numpy()
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def _view(seed):
    r = rr.generate(seed)
    m = r.returns.merge(r.truth[["return_id", "fraud", "how", "copy_of"]], on="return_id")
    m["ts"] = pd.to_datetime(m["returned_at"])
    cut = m["ts"].max().normalize() - pd.Timedelta(days=rr.DRIFT_DAYS - 1)
    return r, m, cut


def _near_bills(sales):
    """Bills with a near-duplicate before them within two hours (the same items, the amount within 5%): blocking on the items,
    then every pair inside the window, as a matcher would."""
    found, pairs = set(), 0
    for _, g in sales.sort_values("sold").groupby("items"):
        t = g["sold"].to_numpy()
        a = g["amount"].to_numpy()
        ids = g["bill_id"].to_numpy()
        lo = np.searchsorted(t, t - np.timedelta64(2, "h"))
        for i in range(len(g)):
            j = np.arange(lo[i], i)
            hit = j[np.abs(a[j] / a[i] - 1) <= 0.05] if len(j) else j
            if len(hit):
                found.add(ids[i])
                pairs += len(hit)
    return found, pairs


class RetailReturnsReplicaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.views = {s: _view(s) for s in SEEDS}

    def test_deterministic_synthetic_the_truth_never_written_and_every_return_follows_its_sale(self):
        r, m, _ = self.views[0]
        pd.testing.assert_frame_equal(r.returns, rr.generate(0).returns)
        paths = rr.save(r, Path(tempfile.mkdtemp()))
        self.assertTrue(json.loads(paths["info"].read_text())["synthetic"])
        written = pd.read_csv(paths["returns"])
        self.assertFalse({"fraud", "how", "copy_of"} & set(written.columns))
        sales = pd.read_csv(paths["sales"]).set_index("bill_id")
        self.assertEqual(set(written["bill_id"]) - set(sales.index), set())  # every return points at a bill in the sales table
        self.assertTrue((pd.to_datetime(written["returned_at"]) > pd.to_datetime(written["bill_id"].map(sales["sold_at"]))).all())
        rr.generate(0, n_dishonest=1)  # a single dishonest cashier is a valid replica

    def test_rare_positives_and_a_review_capacity_that_binds(self):
        for seed, (r, m, _) in self.views.items():
            self.assertTrue(0.01 <= m["fraud"].mean() <= 0.06, seed)
            per_day = m.groupby(m["ts"].dt.date).size()
            self.assertLess((per_day <= rr.CAPACITY).mean(), 0.1, seed)  # the team cannot review every return of a day
            reviewed = m[m["label_source"].isin(["expert", "expert_audit"])]
            self.assertLessEqual(reviewed.groupby(reviewed["ts"].dt.date).size().max(), rr.CAPACITY, seed)

    def test_rule_labels_reward_the_rule_and_the_holdout_has_reviewed_labels(self):
        for seed, (r, m, cut) in self.views.items():
            ruled = m[m["label_source"] == "rule"]
            self.assertTrue((ruled["label"] == 1).all() and ruled["fraud"].mean() < 0.5, seed)  # most late returns are honest
            labelled = m[m["label_source"] != ""]
            self.assertGreater(_auc(labelled["rule_flag"], labelled["label"]) - _auc(m["rule_flag"], m["fraud"]), 0.1, seed)
            reviewed = m[m["label_source"].isin(["expert", "expert_audit"])]
            self.assertTrue((reviewed["label"] == reviewed["fraud"]).all(), seed)
            evaluated = (m["ts"] >= cut) & m["label_source"].isin(["expert", "expert_audit"]) & (m["label"] == 1)
            self.assertGreaterEqual(int(evaluated.sum()), 12, seed)  # the holdout can be judged on reviewed labels

    def test_the_cashier_statistic_reads_the_future_and_review_status_is_the_label(self):
        for seed, (r, m, cut) in self.views.items():
            ms = m.sort_values("ts")
            earlier = ms.groupby("cashier_id")["rule_flag"].transform(lambda s: s.shift().expanding().mean()).fillna(0)
            train = ms["ts"] < cut  # the leak shows where a model would learn it: the training period
            gain = _auc(ms.loc[train, "cashier_late_share"], ms.loc[train, "fraud"]) - _auc(earlier[train], ms.loc[train, "fraud"])
            self.assertGreater(gain, 0.03, seed)
            reviewed = m[m["review_status"] != ""]
            self.assertTrue(((reviewed["review_status"] == "confirmed") == (reviewed["fraud"] == 1)).all(), seed)
        self.assertEqual(set(self.views[0][0].facts["keep_out"]), {"review_status", "cashier_late_share", "label_source"})

    def test_copies_are_near_duplicates_found_only_across_the_tables(self):
        for seed, (r, m, cut) in self.views.items():
            sales = r.sales.assign(sold=pd.to_datetime(r.sales["sold_at"]))
            by_bill = sales.set_index("bill_id")
            copies = m[m["copy_of"] != ""]
            self.assertGreater(len(copies), 50, seed)
            copy, orig = by_bill.loc[copies["bill_id"]], by_bill.loc[copies["copy_of"]]
            self.assertTrue((copy["items"].to_numpy() == orig["items"].to_numpy()).all(), seed)
            self.assertTrue((np.abs(copy["amount"].to_numpy() / orig["amount"].to_numpy() - 1) <= 0.051).all(), seed)
            gap = (copy["sold"].to_numpy() - orig["sold"].to_numpy()) / np.timedelta64(1, "m")
            self.assertTrue(((gap > 0) & (gap <= 90)).all(), seed)
            self.assertGreaterEqual(int(((copies["ts"].to_numpy() >= cut) & (orig["sold"].to_numpy() < cut)).sum()), 1, seed)  # across the cut
            # no column of the returns table alone gives the fraud away ...
            singles = {c: _auc(m[c].astype("category").cat.codes if m[c].dtype == object else m[c], m["fraud"])
                       for c in ("refund_method", "reason_code", "amount", "items", "rule_flag")}
            best = max(max(v, 1 - v) for v in singles.values())
            self.assertLess(best, 0.85, (seed, singles))
            # ... while the cross-table link does: the bill has a near-duplicate minutes before it, issued by the cashier who takes it back
            near_bills, pairs = _near_bills(sales)
            link = (m["bill_id"].isin(near_bills) & (m["bill_id"].map(by_bill["cashier_id"]) == m["cashier_id"])).astype(int)
            self.assertGreater(_auc(link, m["fraud"]), best, seed)
            self.assertGreater(pairs - len(copies), 5 * len(copies), seed)  # honest repeat purchases outnumber copies: a match alone is not fraud

    def test_a_few_cashiers_and_new_ones_in_the_last_months(self):
        for seed, (r, m, cut) in self.views.items():
            fraud = m["fraud"] == 1
            self.assertLessEqual(m.loc[fraud, "cashier_id"].nunique(), r.facts["dishonest_cashiers"], seed)
            new = set(m.loc[fraud & (m["ts"] >= cut), "cashier_id"]) - set(m.loc[fraud & (m["ts"] < cut), "cashier_id"])
            self.assertGreaterEqual(len(new), 1, seed)  # a model that learnt who they are misses the next ones

    def test_the_holdout_is_the_drift_and_the_rule_loses_its_recall(self):
        for seed, (r, m, cut) in self.views.items():
            fraud = m["fraud"] == 1
            before, after = m.loc[fraud & (m["ts"] < cut), "rule_flag"].mean(), m.loc[fraud & (m["ts"] >= cut), "rule_flag"].mean()
            self.assertGreater(before - after, 0.5, (seed, before, after))
            self.assertGreater(int(m.loc[m["ts"] >= m["ts"].max() - pd.Timedelta(days=7), "fraud"].sum()), 0, seed)  # fraud to the log's last week
        f = self.views[0][0].facts
        self.assertEqual((f["time_column"], f["holdout_days"], f["capacity_per_day"]), ("returned_at", rr.DRIFT_DAYS, rr.CAPACITY))
        self.assertIn("never accuracy", f["metric"])
        self.assertEqual(f["evaluate_on"], ["expert", "expert_audit"])


if __name__ == "__main__":
    unittest.main()
