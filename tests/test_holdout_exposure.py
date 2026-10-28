"""H8's gap (rule 2), closed 2026-10-28: a holdout the chat's answers already looked at is said beside its number.

``dclab_core.exposure.holdout_exposure`` reads the exposure an analysis carried into the project against the split that ran;
the final stage keeps the verdict in its evidence and, unless the holdout starts after every date the answers covered,
adds it to the holdout claim's limits.
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from dclab_core.exposure import holdout_exposure  # noqa: E402


def carried(date=None, column="signup_date", related=("plan",)):
    out = {"answers": 2, "target": "churned", "related_to_target": list(related)}
    if date:
        out["holdout_after"] = {"column": column, "date": date}
    else:
        out["holdout_seen"] = True
    return out


class RuleTests(unittest.TestCase):
    def test_after_seen_and_may_hold(self):
        self.assertIsNone(holdout_exposure(None, "signup_date", ["2022-07-01", "2022-08-14"]))
        self.assertIsNone(holdout_exposure({"answers": 0}, "signup_date", ["2022-07-01", "2022-08-14"]))
        first = holdout_exposure(carried("2022-03-31"), "signup_date", ["2022-07-01 06:00:00", "2022-08-14"])
        self.assertEqual((first["status"], first["explored_until"], first["holdout_from"]), ("after", "2022-03-31", "2022-07-01"))
        seen = holdout_exposure(carried("2022-12-31"), "signup_date", ["2022-07-01", "2022-08-14"])
        self.assertEqual(seen["status"], "seen")
        self.assertIn("not a first look", seen["note"])
        self.assertIn("plan", seen["note"])  # what the chat compared with the outcome
        same_day = holdout_exposure(carried("2022-07-01"), "signup_date", ["2022-07-01 18:00:00", "2022-08-14"])
        self.assertEqual(same_day["status"], "seen")  # a day the answers covered is a day they saw
        # the review: the answers read times in UTC, the split in the table's clock; 02:00 at UTC+4 is the day before in UTC,
        # so a holdout starting the day after the answers' last date is never called a first look
        next_day = holdout_exposure(carried("2022-06-30"), "signup_date", ["2022-07-01 02:00:00", "2022-08-14"])
        self.assertEqual(next_day["status"], "may_hold")
        self.assertIn("time zones", next_day["note"])
        two_days = holdout_exposure(carried("2022-06-30"), "signup_date", ["2022-07-02 00:00:00", "2022-08-14"])
        self.assertEqual(two_days["status"], "after")
        for exp, t, rng in ((carried(), "signup_date", ["2022-07-01", "2022-08-14"]), (carried("2022-03-31"), None, None),
                            (carried("2022-03-31", column="renewal_date"), "signup_date", ["2022-07-01", "2022-08-14"])):
            self.assertEqual(holdout_exposure(exp, t, rng)["status"], "may_hold")


class EngineTests(unittest.TestCase):
    """A project built from an analysis: the final stage's evidence and its holdout claim say what the chat saw."""

    @classmethod
    def setUpClass(cls):
        from dclab_rnd.expansion.runner import fast_profile
        from dclab_rnd.studio import ProjectStore, engine
        from test_studio import CHURN_CONTRACT, churn_frame, new_project

        cls.finals = {}
        store = ProjectStore(Path(tempfile.mkdtemp()))
        with fast_profile():
            for name, exposure in (("before", carried("2022-03-31")), ("over", carried("2022-12-31")), ("unbounded", carried()),
                                   ("none", None)):
                project = new_project(store, churn_frame(), CHURN_CONTRACT)
                if exposure:
                    project["exposure"] = exposure
                    store.save(project)
                for stage in ("data", "leakage", "features", "models", "final"):
                    engine.execute(store, project["id"], stage)
                cls.finals[name] = store.read_stage(project["id"], "final")
                cls.stores = getattr(cls, "stores", {}) | {name: (store, project["id"])}

    def holdout_claim(self, record):
        return next(c for c in record["claims"] if c["claim_id"].endswith("-C1"))

    def test_a_holdout_after_the_explored_dates_is_a_first_look(self):
        ev = self.finals["before"]["evidence"]
        self.assertEqual(ev["holdout_exposure"]["status"], "after")
        self.assertNotIn("first look", " ".join(ev.get("limits") or []))
        self.assertFalse(any("first look" in x for x in self.holdout_claim(self.finals["before"])["limitations"]))

    def test_a_holdout_the_chat_saw_is_said_beside_its_number_once(self):
        for name, status in (("over", "seen"), ("unbounded", "may_hold")):
            ev = self.finals[name]["evidence"]
            note = ev["holdout_exposure"]["note"]
            self.assertEqual(ev["holdout_exposure"]["status"], status)
            self.assertIn(note, self.holdout_claim(self.finals[name])["limitations"])
            self.assertNotIn(note, ev.get("limits") or [])  # on the claim only: a report lists both (the review: said twice)
        from dclab_core.report import to_markdown
        from dclab_rnd.studio.solution_report import build
        store, pid = self.stores["over"]
        self.assertEqual(to_markdown(build(store, pid, analyse=False)).count("not a first look"), 1)

    def test_no_analysis_no_mark(self):
        self.assertNotIn("holdout_exposure", self.finals["none"]["evidence"])


class MachineTests(unittest.TestCase):
    """A kind with its own machine (forecasting) reads the holdout in its own final stage: marked there too."""

    def test_a_forecast_after_an_analysis_is_marked(self):
        from unittest import mock

        from dclab_rnd import kinds
        from dclab_rnd.kinds import acceptance
        from dclab_rnd.models.gateway import private_scope
        from dclab_rnd.studio import engine

        store, pid = acceptance.build(kinds.get("forecasting").case(0), Path(tempfile.mkdtemp()))
        project = store.get(pid)
        project["exposure"] = carried("2099-12-31", column=project["solution"].get("time_column") or "date")
        store.save(project)
        with private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.run_all(store, pid)
        final = store.read_stage(pid, "final")
        seen = final["evidence"]["holdout_exposure"]
        self.assertIn(seen["status"], ("seen", "may_hold"))
        self.assertTrue(any(seen["note"] in (c.get("limitations") or []) for c in final["claims"]))


if __name__ == "__main__":
    unittest.main()
