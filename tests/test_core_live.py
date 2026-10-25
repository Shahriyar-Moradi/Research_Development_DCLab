"""34.10, monitoring that runs: a model in use for ten days, its log read window by window. The inputs and the outcome's
relation change on day 6; outcomes become known two days after scoring.

The first alert is day 6's and no earlier day raises one; on the undrifted days the live AUC agrees with the holdout's
within the intervals; the drifted input is named from the batch profiles; the last two days are unread for performance
(their outcomes are not known yet), not called good; no value of a row is in any record.
"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

try:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from dclab_core import live  # noqa: F401

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)

DRIFT_DAY = 6


def rows(n, rng, drifted=False):
    x = rng.normal(0, 1, (n, 3))
    if drifted:
        x[:, 0] += 1.2  # the inputs move, and the outcome no longer follows the second column
    logit = 1.5 * x[:, 0] - (0.0 if drifted else 1.0) * x[:, 1] - 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(x, columns=["tenure", "calls", "fee"]), y


def world(seed=0, days=10, per_day=600, delay=None):
    """``delay(score) -> days``: when each outcome becomes known (two days for every row by default)."""
    from dclab_core import monitoring, serving

    rng = np.random.default_rng(seed)
    X, y = rows(4000, rng)
    Xh, yh = rows(1500, rng)
    model = LogisticRegression().fit(X, y)
    train_scores = model.predict_proba(X)[:, 1]
    plan = monitoring.plan(X, scores=train_scores, threshold=0.5, rule="score at least 0.5")
    hs = model.predict_proba(Xh)[:, 1]
    auc = live._auc(yh, hs)
    draws = [live._auc(yh[r], hs[r]) for r in (rng.integers(0, len(yh), len(yh)) for _ in range(200))]
    holdout = {"metric": "roc_auc", "value": round(auc, 4), "interval": [round(float(np.quantile(draws, 0.05)), 4), round(float(np.quantile(draws, 0.95)), 4)]}
    start = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
    predictions, outcomes, profiles = [], [], []
    for day in range(days):
        Xd, yd = rows(per_day, rng, drifted=day >= DRIFT_DAY)
        scores = model.predict_proba(Xd)[:, 1]
        batch = serving.new_key()
        at = (start + timedelta(days=day)).isoformat()
        profiles.append(live.profile(plan, Xd, model_id="p1", model_version="v1", at=at))
        for i in range(per_day):
            key = serving.new_key()
            predictions.append(serving.PredictionRecord(model_id="p1", model_version="v1", batch=batch, position=i, key=key, task="binary",
                                                        score=float(scores[i]), decision="alert" if scores[i] >= 0.5 else "no_alert",
                                                        threshold=0.5, at=at))
            known = (start + timedelta(days=day + (2 if delay is None else delay(scores[i])))).isoformat()  # two days later by default
            outcomes.append(serving.OutcomeRecord(key=key, outcome=float(yd[i]), known_at=known, recorded_at=known))
    as_of = (start + timedelta(days=days - 1, hours=12)).isoformat()
    return plan, predictions, outcomes, profiles, holdout, as_of


@unittest.skipIf(MISSING, MISSING)
class LiveMonitorTests(unittest.TestCase):
    def test_the_planted_drift_is_the_first_alert_and_undrifted_days_agree_with_the_holdout(self):
        plan, predictions, outcomes, profiles, holdout, as_of = world()
        report = live.monitor_log(plan, predictions, outcomes, profiles=profiles, holdout=holdout, as_of=as_of)
        days = [w.window for w in report.windows]
        self.assertEqual(len(days), 10)
        self.assertEqual(report.first_alert, days[DRIFT_DAY])  # not before
        self.assertTrue(all(w.status != "alert" for w in report.windows[:DRIFT_DAY]), [w.alerts for w in report.windows[:DRIFT_DAY]])
        self.assertIn("tenure", report.windows[DRIFT_DAY].columns_shifted)  # named from the batch profiles
        read = [w for w in report.windows[:DRIFT_DAY] if w.vs_holdout != "unread"]
        self.assertEqual(len(read), DRIFT_DAY)
        self.assertTrue(all(w.vs_holdout == "within" for w in read), [(w.value, w.interval) for w in read])
        self.assertEqual([w.vs_holdout for w in report.windows[-2:]], ["unread", "unread"])  # outcomes not known yet: unread
        self.assertEqual([w.labelled for w in report.windows[-2:]], [0, 0])
        self.assertIn("worse", [w.vs_holdout for w in report.windows[DRIFT_DAY:DRIFT_DAY + 2]])  # once its outcomes are known, the drift costs
        said = " ".join(t for w in report.windows[DRIFT_DAY:DRIFT_DAY + 2] for t in w.alerts + w.warnings)
        self.assertIn("worse than the holdout", said)
        self.assertEqual(report.schema_version, "dclab_core.live_report.v1")

    def test_one_worse_window_warns_and_two_running_alert(self):
        """Measured 2026-10-25 on twenty runs: one undrifted window alone was worse than the holdout by chance in 2 of 120;
        alerting on one window gave 2 of 20 runs a false alert before the drift, two running gave none."""
        plan, predictions, outcomes, profiles, holdout, as_of = world()
        strict = dict(holdout, interval=[holdout["value"] + 0.2, holdout["value"] + 0.3])  # a holdout no window can match
        report = live.monitor_log(plan, predictions[:1200], outcomes, holdout=strict, as_of=as_of)  # two undrifted days
        first, second = report.windows
        self.assertEqual((first.vs_holdout, first.status), ("worse", "warn"))
        self.assertEqual((second.vs_holdout, second.status), ("worse", "alert"))
        self.assertIn("the second window running", second.alerts[0])

    def test_no_value_of_a_row_is_in_a_profile_or_the_report(self):
        plan, predictions, outcomes, profiles, holdout, as_of = world(seed=1, days=3, per_day=300)
        report = live.monitor_log(plan, predictions, outcomes, profiles=profiles, holdout=holdout, as_of=as_of)
        text = report.model_dump_json() + "".join(p.model_dump_json() for p in profiles)
        for value in pd.DataFrame(rows(5, np.random.default_rng(1))[0]).to_numpy().ravel():
            self.assertNotIn(f"{value:.6f}", text)
        with self.assertRaises(ValueError):  # shares that are not shares are refused
            live.BatchProfile(model_id="p1", model_version="v1", at=profiles[0].at, rows=30, shares={"fee": [0.5, 0.9]})
        with self.assertRaises(ValueError):  # a profile of a few rows would show their own bins (the review of 34.10)
            live.BatchProfile(model_id="p1", model_version="v1", at=profiles[0].at, rows=1, shares=profiles[0].shares)
        with self.assertRaises(ValueError):
            live.profile(plan, pd.DataFrame(rows(3, np.random.default_rng(2))[0]), model_id="p1", model_version="v1", at=profiles[0].at)
        gather = live.Profiler(plan, "p1", "v1")  # one row at a time: nothing written until twenty are in
        made = [gather.add(pd.DataFrame(rows(1, np.random.default_rng(i))[0]), profiles[0].at) for i in range(25)]
        self.assertEqual([i for i, m in enumerate(made) if m is not None], [19])
        self.assertEqual(made[19].rows, 20)

    def test_a_quiet_log_says_so_and_bad_windows_are_refused(self):
        plan, predictions, _, _, _, _ = world(seed=2, days=2, per_day=250)
        report = live.monitor_log(plan, predictions, window="week")
        self.assertEqual(report.status, "stable")
        self.assertEqual(len(report.windows), 1)
        self.assertEqual(live.monitor_log(plan, []).status, "no_rows")
        with self.assertRaises(ValueError):
            live.monitor_log(plan, predictions, window="hour")

    def test_outcomes_known_first_for_the_flagged_rows_raise_no_false_alert(self):
        """The review of 34.10: flagged rows labelled after a day, the rest after thirty; read at once, an undrifted day
        looked worse than the holdout (about 0.69 against 0.82); a window is now read only once mature."""
        plan, predictions, outcomes, profiles, holdout, as_of = world(seed=3, days=6, delay=lambda s: 1 if s >= 0.5 else 30)
        report = live.monitor_log(plan, predictions, outcomes, profiles=profiles, holdout=holdout, as_of=as_of)
        self.assertIsNone(report.first_alert, report.alerts)
        self.assertTrue(all(w.vs_holdout == "unread" and not w.mature for w in report.windows))
        later = live.monitor_log(plan, predictions, outcomes, profiles=profiles, holdout=holdout, as_of="2026-11-15T00:00:00+00:00")
        self.assertTrue(all(w.mature and w.vs_holdout == "within" for w in later.windows), [(w.value, w.interval) for w in later.windows])
        horizon = live.monitor_log(plan, predictions, outcomes, holdout=holdout, as_of=as_of, label_horizon_days=40)
        self.assertTrue(all(not w.mature for w in horizon.windows))

    def test_a_column_no_longer_sent_is_seen_and_a_reading_as_of_a_moment_sees_nothing_later(self):
        plan, predictions, outcomes, profiles, holdout, as_of = world(seed=4, days=3, per_day=400)
        gone = [live.profile(plan, pd.DataFrame({"calls": [0.0] * 400, "fee": [0.0] * 400}), model_id="p1", model_version="v1", at=p.at)
                for p in profiles]  # requests that stopped sending tenure
        report = live.monitor_log(plan, predictions, outcomes, profiles=gone, holdout=holdout, as_of=as_of)
        self.assertTrue(all("tenure" in w.columns_shifted for w in report.windows))
        early = live.monitor_log(plan, predictions, outcomes, profiles=profiles, holdout=holdout, as_of="2026-10-01T10:00:00+00:00")
        self.assertEqual([w.window for w in early.windows], ["2026-10-01"])  # nothing scored after the moment
        self.assertEqual(early.windows[0].labelled, 0)  # nor an outcome recorded after it

    def test_a_worse_window_of_few_rows_is_not_hidden_and_a_large_log_reads_fast(self):
        import time

        plan, predictions, outcomes, profiles, holdout, as_of = world(seed=5, days=2, per_day=150)
        strict = dict(holdout, interval=[holdout["value"] + 0.2, holdout["value"] + 0.3])
        report = live.monitor_log(plan, predictions, outcomes, holdout=strict, as_of="2026-10-10T00:00:00+00:00")
        self.assertEqual([w.status for w in report.windows], ["warn", "alert"])  # 150 rows: below min_rows, still said
        plan, predictions, outcomes, profiles, holdout, as_of = world(seed=6, days=20, per_day=2500)
        started = time.perf_counter()
        live.monitor_log(plan, predictions, outcomes, profiles=profiles, holdout=holdout, as_of=as_of)
        self.assertLess(time.perf_counter() - started, 20)  # 50,000 rows in twenty windows: well inside a request's timeout

    def test_every_import_order_works_in_a_fresh_interpreter(self):
        """The review of 34.10: contracts imports live at its end; importing monitoring first met a half-made module."""
        import subprocess

        for module in ("dclab_core.monitoring", "dclab_core.serving", "dclab_core.live", "dclab_core.contracts", "dclab_core.retrain"):
            done = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r}); import {module}"],
                                  capture_output=True, text=True, timeout=120)
            self.assertEqual(done.returncode, 0, (module, done.stderr[-500:]))


if __name__ == "__main__":
    unittest.main()
