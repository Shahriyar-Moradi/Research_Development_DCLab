"""Package 34.13 in the library: would more rows help? A learning curve on the training folds only, its last step's gain
with an interval, a plain answer, and the same per segment where errors concentrate.

The replicas have a known curve: ``rising`` (many small weights: the score still climbs) must say more rows likely
help; ``flat`` (one strong column, noisy labels: learnt early) must say they are unlikely to help."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import mean_absolute_error, roc_auc_score
    from sklearn.model_selection import KFold, StratifiedKFold

    from dclab_core import data_value as dv
    from dclab_core.replicas import learning_curves
except ImportError as error:  # pydantic is not part of requirements/ci.txt
    MISSING = str(error)
else:
    MISSING = ""


def _setup(shape, seed=0):
    """Training rows (80 %) and a holdout (20 %) the curve must never see, with three stratified folds on training."""
    frame = learning_curves.generate(shape, seed).rows
    y_all = (frame["outcome"] == "yes").astype(int).to_numpy()
    X_all = frame.drop(columns=["row_id", "region", "outcome"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    cut = int(0.8 * len(frame))
    train, holdout = np.sort(order[:cut]), np.sort(order[cut:])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(train, y_all[train]))
    return frame, X_all, y_all, train, holdout, folds


def _logistic(X, y, seen=None):
    def fit_predict(fit_rows, valid_rows):
        if seen is not None:
            seen.append((np.asarray(fit_rows), np.asarray(valid_rows)))
        model = LogisticRegression(C=1.0, max_iter=500).fit(X[fit_rows], y[fit_rows])
        return model.predict_proba(X[valid_rows])[:, 1]
    return fit_predict


def _curve(shape, seed=0, **kw):
    frame, X_all, y_all, train, holdout, folds = _setup(shape, seed)
    X, y = X_all[train], y_all[train]
    segments = {"region": frame["region"].to_numpy()[train]}
    return dv.learning_curve(y, folds, _logistic(X, y), roc_auc_score, metric="roc_auc", strata=y, segments=segments,
                             bootstrap=kw.pop("bootstrap", 100), seed=seed, **kw)


@unittest.skipIf(MISSING, MISSING)
class AnswerOnKnownCurves(unittest.TestCase):
    def test_a_rising_curve_says_more_rows_likely_help(self):
        result = _curve("rising")
        self.assertEqual(result.answer, dv.LIKELY, result.plain)
        self.assertGreater(result.slope.low, 0)
        scores = [p.score for p in result.points]
        self.assertLess(scores[0], scores[-1])  # 25 % of the rows score below all of them
        for p in result.points:
            self.assertLessEqual(p.low, p.score + 1e-12)
            self.assertLessEqual(p.score, p.high + 1e-12)
        self.assertIn("likely help", result.plain)
        self.assertIn("same process", result.assumption)
        self.assertFalse(result.holdout_read)

    def test_a_flat_curve_says_more_rows_are_unlikely_to_help(self):
        result = _curve("flat")
        self.assertEqual(result.answer, dv.UNLIKELY, result.plain)
        self.assertLess(result.slope.high, result.practical)
        self.assertIn("unlikely", result.plain)

    def test_the_answers_hold_on_other_seeds(self):
        for seed in (1, 2):
            self.assertEqual(_curve("rising", seed).answer, dv.LIKELY, seed)
            self.assertEqual(_curve("flat", seed).answer, dv.UNLIKELY, seed)

    def test_the_segment_where_errors_concentrate_gets_its_own_curve(self):
        result = _curve("rising")
        found = [(s.column, s.level) for s in result.segments]
        self.assertIn(("region", "east"), found)
        east = next(s for s in result.segments if s.level == "east")
        self.assertGreater(east.error_share, east.row_share)
        self.assertIn(east.answer, (dv.LIKELY, dv.UNLIKELY, dv.CANNOT))
        self.assertTrue(east.reason)


@unittest.skipIf(MISSING, MISSING)
class NeverTheHoldout(unittest.TestCase):
    def test_every_fit_and_score_reads_training_rows_only(self):
        frame, X_all, y_all, train, holdout, folds = _setup("rising")
        seen = []

        def fit_predict(fit_rows, valid_rows):  # positions are of the training rows: map them back to the table
            seen.append((train[fit_rows], train[valid_rows]))
            return _logistic(X_all[train], y_all[train])(fit_rows, valid_rows)

        dv.learning_curve(y_all[train], folds, fit_predict, roc_auc_score, metric="roc_auc", strata=y_all[train], bootstrap=20)
        self.assertEqual(len(seen), 3 * (3 * 2 + 1))  # 3 folds x (3 smaller sizes x 2 repeats + every row once)
        held = set(holdout.tolist())
        for fit, valid in seen:
            self.assertFalse(held & set(fit.tolist()))
            self.assertFalse(held & set(valid.tolist()))
            self.assertFalse(set(fit.tolist()) & set(valid.tolist()))

    def test_rows_outside_the_training_folds_are_refused(self):
        y = np.array([0, 1] * 20)
        folds = [(np.arange(0, 30), np.arange(30, 45))]  # 45 is past the 40 training rows given
        with self.assertRaises(ValueError):
            dv.learning_curve(y, folds, lambda a, b: np.zeros(len(b)), roc_auc_score, metric="roc_auc")


@unittest.skipIf(MISSING, MISSING)
class Subsampling(unittest.TestCase):
    def test_a_grouped_split_keeps_whole_groups(self):
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(50), 4)
        rows = np.arange(200)
        picked = dv.subsample(rows, 0.5, how="group", rng=rng, groups=groups)
        kept = set(groups[picked].tolist())
        self.assertEqual(sorted(picked.tolist()), sorted(np.flatnonzero(np.isin(groups, list(kept))).tolist()))
        self.assertAlmostEqual(len(picked) / 200, 0.5, delta=0.03)

    def test_a_timed_split_keeps_the_latest_rows(self):
        order = np.arange(100)[::-1].astype(float)  # row 0 is the latest
        picked = dv.subsample(np.arange(100), 0.25, how="time", rng=np.random.default_rng(0), order=order)
        self.assertEqual(sorted(picked.tolist()), list(range(25)))

    def test_a_stratified_sample_keeps_every_class(self):
        strata = np.array([1] * 6 + [0] * 194)
        picked = dv.subsample(np.arange(200), 0.25, how="stratified", rng=np.random.default_rng(0), strata=strata)
        self.assertGreaterEqual(int(strata[picked].sum()), 1)
        self.assertEqual(len(dv.subsample(np.arange(200), 1.0, how="random", rng=np.random.default_rng(0))), 200)


@unittest.skipIf(MISSING, MISSING)
class RuleAndContract(unittest.TestCase):
    def test_the_rule(self):
        s = lambda g, lo, hi: dv.Slope(gain=g, low=lo, high=hi, per_doubling=g * 2.4)
        self.assertEqual(dv.answer_of(s(0.02, 0.01, 0.03), 0.005)[0], dv.LIKELY)
        self.assertEqual(dv.answer_of(s(0.001, -0.002, 0.004), 0.005)[0], dv.UNLIKELY)
        self.assertEqual(dv.answer_of(s(0.004, 0.001, 0.0045), 0.005)[0], dv.UNLIKELY)  # surely positive, surely small
        self.assertEqual(dv.answer_of(s(0.01, -0.01, 0.03), 0.005)[0], dv.CANNOT)
        self.assertEqual(dv.answer_of(None, 0.005)[0], dv.CANNOT)

    def test_an_error_metric_counts_a_fall_as_a_gain(self):
        rng = np.random.default_rng(0)
        n, k = 900, 30
        X = rng.normal(size=(n, k))
        y = X @ rng.uniform(0.3, 0.6, k) + rng.normal(scale=1.0, size=n)
        folds = list(KFold(3, shuffle=True, random_state=0).split(X))

        def fit_predict(fit_rows, valid_rows):
            from sklearn.linear_model import LinearRegression

            return LinearRegression().fit(X[fit_rows], y[fit_rows]).predict(X[valid_rows])

        result = dv.learning_curve(y, folds, fit_predict, mean_absolute_error, metric="mae", higher_is_better=False, bootstrap=100)
        self.assertGreater(result.points[0].score, result.points[-1].score)  # the error falls with rows
        self.assertGreater(result.slope.gain, 0)
        self.assertAlmostEqual(result.practical, 0.01 * result.points[-1].score, places=9)

    def test_the_record_is_a_versioned_contract(self):
        from dclab_core.contracts import CONTRACTS

        self.assertIs(CONTRACTS["data_value"], dv.DataValue)
        result = _curve("flat", bootstrap=30)
        self.assertEqual(result.schema_version, "dclab_core.data_value.v1")
        again = dv.DataValue.model_validate_json(result.model_dump_json())
        self.assertEqual(again, result)
        self.assertEqual(result.rule, dv.RULE)


if __name__ == "__main__":
    unittest.main()
