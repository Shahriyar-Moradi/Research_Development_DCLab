"""34.11 and 34.12 in the studio: newer labelled rows kept beside a project, a challenger fitted the champion's way with
them, compared on the latest of those rows neither trained on; promoted only when better beyond its interval; every
comparison kept, its event and its ledger point recorded, and a lesson proposed for a reviewer.

The churn replica: newer rows from the same world keep the champion; newer rows whose outcome follows another rule promote
the challenger. The review of 34.11: a re-export of the project's own rows is dropped and never compared on; a bad time
stamp, a code read as 1.0, a tiny window and an invalid live report are each handled; the measure is the project's; a later
comparison is against the promoted fit; the promotion is in the ledger. The project's holdout is never read.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd

    from dclab_rnd import kinds
    from dclab_rnd.kinds import acceptance
except ImportError as error:  # pragma: no cover
    MISSING = str(error)
else:
    MISSING = ""


def built():
    from dclab_rnd.expansion import runner
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import engine

    store, pid = acceptance.build(kinds.get("binary").case(0), Path(tempfile.mkdtemp()))
    with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
        engine.run_all(store, pid)
    return store, pid


def newer(changed: bool, seed: int = 1) -> pd.DataFrame:
    """Newer rows of the same table (a later signup time); with ``changed`` their outcome follows another rule."""
    table = kinds.get("binary").case(seed).table.copy()
    table["signup_time"] = (pd.to_datetime(table["signup_time"]) + pd.DateOffset(years=2)).dt.strftime("%Y-%m-%d %H:%M:%S")
    if changed:
        rng = np.random.default_rng(seed)
        risk = (table["plan"] == "plus").astype(float) * 1.6 + 0.9 * (table["tenure_months"] > 24) - 1.2
        table["churned"] = np.where(rng.random(len(table)) < 1 / (1 + np.exp(-2 * risk)), "yes", "no")
    return table


def attach(store, pid, frame, name):
    from dclab_rnd.studio import retrain

    retrain.add_newer_rows(store, pid, frame.to_csv(index=False).encode(), f"{name}.csv", name)


def report(pid):
    from dclab_core.live import LiveReport, WindowReading

    return LiveReport(model_id=pid, window="day", as_of="2027-01-10T00:00:00+00:00", status="alert", first_alert="2027-01-04",
                      note="read from the service's logs",
                      windows=[WindowReading(window="2027-01-04", start="2027-01-04T00:00:00+00:00", end="2027-01-04T23:00:00+00:00", rows=600,
                                             status="alert", alerts=["plan shifted (PSI 0.410, the alert line 0.25)"])]).model_dump(mode="json")


@unittest.skipIf(MISSING, MISSING)
class RetrainStudioTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd import experience, ledger

        home = Path(tempfile.mkdtemp())
        self.log, self.book = experience.open_log(home, "wtest"), ledger.FileLedger(home / "ledger", "wtest")
        experience.install(self.log)
        ledger.install(self.book)
        self.addCleanup(experience.install, None)
        self.addCleanup(ledger.install, None)

    def test_kept_without_a_change_promoted_after_one_and_every_comparison_recorded(self):
        from dclab_rnd import lessons
        from dclab_rnd.studio import retrain

        store, pid = built()
        holdout_uses = store.get(pid).get("holdout_uses")
        lesson_store = lessons.open_lessons(Path(tempfile.mkdtemp()))
        attach(store, pid, newer(False), "same_world")
        attach(store, pid, newer(True), "new_rule")
        self.assertEqual(store.get(pid)["stages"]["final"]["status"], "completed")  # newer rows clear no stage
        kept = retrain.run(store, pid, "same_world", lesson_store=lesson_store)
        promoted = retrain.run(store, pid, "new_rule", lesson_store=lesson_store, live_report=report(pid))
        self.assertFalse(kept["promoted"], kept["reason"])
        self.assertTrue(promoted["promoted"], promoted["reason"])
        self.assertEqual(promoted["metric"], store.read_stage(pid, "final")["primary_metric"])  # the project's own measure
        self.assertIn("the inputs or the score shifted", " ".join(promoted["because"]))
        self.assertIn("after the project's latest time", promoted["window"])
        self.assertEqual(promoted["against"], "the approved model")
        view = retrain.view(store, pid)
        self.assertEqual([r["promoted"] for r in view["retrains"]], [False, True])
        self.assertFalse(view["preferred_fit"]["production_approved"])
        self.assertEqual([i["name"] for i in view["preferred_fit"]["includes"]], ["new_rule"])
        self.assertEqual(store.get(pid).get("holdout_uses"), holdout_uses)  # a comparison never reads the holdout
        decided = [r for r in self.book.records() if r.point == "model.promote"]
        self.assertEqual([r.value_used for r in decided], [False, True])  # the review of 34.11: refused by the ledger before
        proposed = lesson_store.list(project_id=pid, status="proposed")
        self.assertEqual(len(proposed), 3)  # the two comparisons and the live alert
        self.assertTrue(any("was better than the approved model" in l["claim"] for l in proposed))
        self.assertTrue(any("first alert in the window 2027-01-04" in l["claim"] for l in proposed))
        self.assertTrue(all(l["written_by"] == "code" and l["scope"]["from"] == "use" for l in proposed))
        lessons.review(lesson_store, proposed[0]["id"], "accept", "reviewer", reason="checked against the record")
        self.assertEqual(lesson_store.get(proposed[0]["id"])["status"], "accepted")  # reviewed as any lesson
        attach(store, pid, newer(True, seed=2), "new_rule_later")
        later = retrain.run(store, pid, "new_rule_later", lesson_store=lesson_store, live_report=report(pid))
        self.assertEqual(later["against"], "the last promoted fit")  # the champion is now the promoted fit (the review of 34.11)
        self.assertIn("the newer rows of new_rule", later["trained_on"]["champion"])
        self.assertEqual(len(lesson_store.list(project_id=pid)), 4)  # one more comparison; the same live alert never twice

    def test_a_re_export_of_the_projects_own_rows_is_dropped_and_never_compared_on(self):
        from dclab_rnd.studio import retrain

        store, pid = built()
        own = kinds.get("binary").case(0).table  # the project's own table again: every row repeats one it holds
        attach(store, pid, own, "re_export")
        with self.assertRaisesRegex(retrain.RetrainRefused, "repeat rows the project holds"):
            retrain.run(store, pid, "re_export")
        mixed = pd.concat([own, newer(False)], ignore_index=True)
        attach(store, pid, mixed, "mixed")
        done = retrain.run(store, pid, "mixed")
        self.assertEqual(done["repeats_dropped"], len(own))

    def test_bad_times_codes_and_tiny_windows_are_handled(self):
        from dclab_core.retrain import RetrainPolicy
        from dclab_rnd.studio import retrain

        store, pid = built()
        rows = newer(False)
        rows.loc[3, "signup_time"] = "not a date"  # one bad stamp: that row out, never every time made missing
        attach(store, pid, rows, "one_bad_stamp")
        done = retrain.run(store, pid, "one_bad_stamp")
        self.assertIn("after the project's latest time", done["window"])
        self.assertEqual(done["window_rows"], int(round(0.2 * (len(rows) - 1))))
        tiny = retrain.run(store, pid, "one_bad_stamp", policy=RetrainPolicy(window_share=0.002, min_new_labelled=1))
        self.assertFalse(tiny["promoted"])
        self.assertIn("too few rows to compare", tiny["reason"])
        p_rows = newer(True)
        p_rows["support_calls"] = p_rows["support_calls"].astype(float)  # a CSV that wrote the counts as 1.0
        attach(store, pid, p_rows, "floats")
        self.assertTrue(retrain.run(store, pid, "floats")["promoted"])
        with self.assertRaisesRegex(retrain.RetrainRefused, "not a monitor's answer"):
            retrain.run(store, pid, "floats", live_report={"first_alert": "x", "windows": []})

    def test_a_table_without_the_columns_too_few_rows_or_a_changed_file_is_refused(self):
        from dclab_core.retrain import RetrainPolicy
        from dclab_rnd.studio import retrain

        store, pid = built()
        attach(store, pid, newer(False).drop(columns=["plan"]), "no_plan")
        with self.assertRaisesRegex(retrain.RetrainRefused, "lack the project's columns"):
            retrain.run(store, pid, "no_plan")
        attach(store, pid, newer(False).head(50), "few")
        with self.assertRaisesRegex(retrain.RetrainRefused, "waits for"):
            retrain.run(store, pid, "few", policy=RetrainPolicy(min_new_labelled=200))
        with self.assertRaisesRegex(retrain.RetrainRefused, "no newer rows named"):
            retrain.run(store, pid, "never_added")
        with self.assertRaisesRegex(retrain.RetrainRefused, "do not read as a table"):
            retrain.add_newer_rows(store, pid, b"\x00\x01 not a table", "bad.parquet", "bad")
        attach(store, pid, newer(False), "january")
        attach(store, pid, newer(True), "february")  # two names, two files, whatever the uploads were called
        self.assertNotEqual(store.get(pid)["newer_rows"]["january"]["filename"], store.get(pid)["newer_rows"]["february"]["filename"])
        (Path(store.data_dir(pid)) / store.get(pid)["newer_rows"]["few"]["filename"]).write_text("changed")
        with self.assertRaisesRegex(retrain.RetrainRefused, "changed or are gone"):
            retrain.run(store, pid, "few")


if __name__ == "__main__":
    unittest.main()
