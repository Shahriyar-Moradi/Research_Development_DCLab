"""34.11, retrain and champion/challenger: a challenger fitted on the newer labelled rows replaces the champion only when
it is better beyond its interval on the latest labelled window neither trained on.

On a replica whose outcome's relation changes, the challenger is promoted after the change; on one that does not change,
it is refused (a fit on fewer rows of the same world is no better). A retrain starts from the live report's alerts or a
schedule, and waits for enough new labels.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

try:
    from sklearn.linear_model import LogisticRegression

    from dclab_core import retrain  # noqa: F401

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)


def rows(n, rng, changed=False):
    x = rng.normal(0, 1, (n, 3))
    logit = 1.2 * x[:, 0] + (1.5 * x[:, 2] if changed else -1.0 * x[:, 1]) - 0.3
    return x, (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)


def contest(seed, changed):
    from dclab_core.retrain import compare

    rng = np.random.default_rng(seed)
    X_old, y_old = rows(3000, rng)
    X_new, y_new = rows(3000, rng, changed=changed)
    cut = int(0.8 * len(y_new))  # the latest fifth of the new labelled rows: neither model trains on it
    champion = LogisticRegression().fit(X_old, y_old)
    challenger = LogisticRegression().fit(X_new[:cut], y_new[:cut])
    window_X, window_y = X_new[cut:], y_new[cut:]
    return compare("binary", window_y, champion.predict_proba(window_X)[:, 1], challenger.predict_proba(window_X)[:, 1],
                   window="the latest fifth of the new labelled rows", bootstrap=300, seed=seed)


@unittest.skipIf(MISSING, MISSING)
class ChampionChallengerTests(unittest.TestCase):
    def test_promoted_after_the_change_and_refused_without_it(self):
        """Measured 2026-10-25 over 20 seeds each: promoted after the change in 20 of 20 (better by about 0.19 ROC AUC);
        without it, 1 of 20 promoted, as a one-sided 95% bound allows: an equally good challenger slips through about once
        in twenty (a needless redeploy, never a worse model; a margin above zero makes it rarer)."""
        after = [contest(seed, True) for seed in range(20)]
        without = [contest(seed, False) for seed in range(20)]
        self.assertGreaterEqual(sum(r.promoted for r in after), 19)
        self.assertLessEqual(sum(r.promoted for r in without), 1)
        record = after[0]
        self.assertEqual((record.metric, record.window_rows, record.schema_version), ("roc_auc", 600, "dclab_core.challenger_record.v1"))
        self.assertIn("promoted", record.reason)
        self.assertGreater(record.interval[0], 0)
        kept = without[0]
        self.assertFalse(kept.promoted)
        self.assertIn("the champion stays", kept.reason)

    def test_a_lower_error_is_better_and_bad_inputs_are_refused(self):
        from dclab_core.retrain import compare

        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 500)
        record = compare("regression", y, y + rng.normal(0, 1.0, 500), y + rng.normal(0, 0.3, 500), bootstrap=300)
        self.assertEqual(record.metric, "mae")
        self.assertTrue(record.promoted)  # the challenger's error is lower
        self.assertGreater(record.difference, 0)  # positive is better, whichever way the measure points
        with self.assertRaises(ValueError):
            compare("binary", [1, 0, 1], [0.2, 0.1], [0.3, 0.2, 0.9])
        with self.assertRaises(ValueError):
            compare("binary", [1, 1, 1], [0.2, 0.1, 0.4], [0.3, 0.2, 0.9])  # one class only: no ROC AUC
        with self.assertRaises(ValueError):
            compare("binary", [1, 0], [float("nan"), 0.1], [0.3, 0.2])

    def test_tiny_windows_never_promote_and_the_hosts_measure_and_groups_are_read(self):
        """The review of 34.11: 1 to 4 rows promoted on a zero-width interval; the comparison used another measure than the
        project's; the interval drew rows, not whole entities."""
        from dclab_core.retrain import MIN_WINDOW, compare

        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, 10)
        tiny = compare("regression", y, y + 1.0, y + 0.1, bootstrap=100)
        self.assertFalse(tiny.promoted)
        self.assertIn("too few rows", tiny.reason)
        yb = np.array([1, 0, 1, 0] * 20, dtype=float)
        champion, challenger = rng.random(80), yb * 0.8 + rng.random(80) * 0.2
        own = compare("binary", yb, champion, challenger, metric="precision_at_half", higher_is_better=True,
                      metric_fn=lambda t, p: float(t[p >= 0.5].mean()) if (p >= 0.5).any() else 0.0, bootstrap=200)
        self.assertEqual(own.metric, "precision_at_half")
        self.assertTrue(own.promoted)
        with self.assertRaises(ValueError):
            compare("binary", yb, champion, challenger, metric_fn=lambda t, p: 0.5)  # a host's measure needs its name and direction
        groups = np.repeat(np.arange(8), 10)  # ten rows of each of eight entities: the interval over entities is wider
        by_rows = compare("binary", yb, champion, challenger, bootstrap=300)
        by_groups = compare("binary", yb, champion, challenger, groups=groups, bootstrap=300)
        self.assertGreaterEqual(by_groups.interval[1] - by_groups.interval[0], by_rows.interval[1] - by_rows.interval[0])
        self.assertIn("over whole groups", by_groups.reason)
        self.assertEqual(MIN_WINDOW, 30)

    def test_a_retrain_starts_from_an_alert_or_the_schedule_and_waits_for_labels(self):
        from dclab_core.retrain import RetrainPolicy, should_retrain

        report = {"windows": [{"window": "2026-10-07", "alerts": ["tenure shifted (PSI 0.400, the alert line 0.25)"]}]}
        policy = RetrainPolicy(min_new_labelled=500)
        self.assertFalse(should_retrain(policy, live_report={"windows": []}, new_labelled=5000)["retrain"])
        waiting = should_retrain(policy, live_report=report, new_labelled=120)
        self.assertFalse(waiting["retrain"])
        self.assertIn("waits for 500", waiting["wait"])
        go = should_retrain(policy, live_report=report, new_labelled=800)
        self.assertTrue(go["retrain"])
        self.assertIn("shifted", go["because"][0])
        scheduled = should_retrain(RetrainPolicy(every_days=30, on_drift=False), last_trained="2026-09-01T00:00:00+00:00",
                                   now="2026-10-25T00:00:00+00:00", new_labelled=900)
        self.assertTrue(scheduled["retrain"])
        self.assertIn("the schedule", scheduled["because"][0])
        self.assertFalse(should_retrain(RetrainPolicy(on_drift=False), live_report=report, new_labelled=900)["retrain"])


if __name__ == "__main__":
    unittest.main()
