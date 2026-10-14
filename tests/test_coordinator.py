"""Package 21.2: the coordinator. One lead agent walks a draft through the problem card, its data, the links between
files, the meet check, the analysis, the workflow and the build, each through a deterministic gate; it never builds while
a blocker is open and says which one; a role's hand-off that names a column the tables do not hold is refused. The
owner's morning, replayed (two files in two drafts, "I want build anomaly detection"), ends in the fraud project:
labels joined on the order, partial, the bikers as groups."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core import coordinator as machine
    from dclab_core.agents import ScriptedAdvisor
    from dclab_core.replicas import delivery_fraud as dfr
    from dclab_rnd import experience
    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.draft.coordinator import Coordinator
    from dclab_rnd.draft.store import DraftStore
    from dclab_rnd.studio import ProjectStore, team
    from tests.test_morning_guard import MOMENT, PROBLEM, upload
except ImportError as error:  # pydantic and pandas are not in every environment
    MISSING = str(error)
else:
    MISSING = ""

ANSWERS = {"prediction_moment": MOMENT, "entity": "The biker", "labels": "A sample of the cases we found",
           "budget": "About 80 a day", "target": "The confirmed frauds", "family": "Find the fraudulent orders"}


def person(answers=ANSWERS):
    return team.Person(answer=lambda field, question: answers.get(field), approve=lambda gate, summary: True, who="scripted person")


@unittest.skipIf(MISSING, MISSING)
class MachineTest(unittest.TestCase):
    def test_the_states_come_in_order_and_a_question_the_data_can_settle_waits_for_it(self):
        labels_q = {"field": "labels", "text": "Is there a file of known cases?"}
        s = machine.situation(machine.Facts(blocking_questions=[labels_q], files_ready=0))
        self.assertEqual((s.state, s.next), ("data", "stop"))  # not asked first: a file may settle it
        s = machine.situation(machine.Facts(blocking_questions=[labels_q, {"field": "prediction_moment", "text": "When?"}],
                                            pending_question="prediction_moment", files_ready=1))
        self.assertEqual((s.state, s.next, s.blockers[0].field), ("problem", "answer", "prediction_moment"))
        s = machine.situation(machine.Facts(files_ready=1, cases_only=True, offers=[{"draft_id": "d2", "key": "order_id", "coverage": 0.97}]))
        self.assertEqual((s.state, s.next, s.blockers[0].evidence["draft_id"]), ("links", "bring_table", "d2"))
        s = machine.situation(machine.Facts(files_ready=1, labels_joined=True, meet=[{"code": "target_absent", "question": "Which column?"}]))
        self.assertEqual((s.state, s.passed), ("meet", ["problem", "data", "links"]))
        s = machine.situation(machine.Facts(files_ready=1, analysed=True, workflow_ready=True,
                                            build_refusals=[{"rule": "family not built yet", "message": "Anomaly detection is not built yet"}]))
        self.assertEqual((s.state, s.next), ("build", "stop"))
        self.assertIn("not built yet", s.says)  # never built while a blocker is open, and it says which one
        s = machine.situation(machine.Facts(files_ready=1, analysed=True, workflow_ready=True, pending_question="entity"))
        self.assertEqual((s.state, s.next, s.blockers[0].code), ("build", "answer", "question_open"))  # never with a question open
        s = machine.situation(machine.Facts(files_ready=1, analysed=True, workflow_ready=True))
        self.assertEqual((s.state, s.next), ("build", "build"))

    def test_a_handoff_naming_a_column_the_tables_do_not_hold_is_refused_whole(self):
        card = {"tables": [{"columns": [{"name": "biker_id"}, {"name": "fare"}]}]}
        good = machine.investigator_handoff({"entities": [{"column": "biker_id", "why": "repeats"}], "after": []})
        bad = machine.investigator_handoff({"entities": [{"column": "biker_id", "why": "repeats"}],
                                            "after": [{"column": "driver_rating", "why": "rated after the ride"}]})
        self.assertTrue(machine.gate(good, card).allowed)
        verdict = machine.gate(bad, card)
        self.assertEqual((verdict.allowed, verdict.refused_columns), (False, ["driver_rating"]))
        self.assertFalse(machine.gate(machine.Handoff(role="critic", state="nowhere"), card).allowed)
        self.assertFalse(machine.gate(good, {}).allowed)  # nothing to check against: never taken on trust
        empty = machine.Handoff(role="data_investigator", state="analysis")
        self.assertFalse(machine.gate(empty, {}, extra_columns=("is_case",)).allowed)  # the host's own column is no data card
        odd = machine.investigator_handoff({"after": {"driver_rating": "after"}, "entities": []})  # not a list of {column, why}
        verdict = machine.gate(odd, card)
        self.assertFalse(verdict.allowed)
        self.assertIn("after is not a list", verdict.reasons[0])

    def test_a_file_that_could_not_be_read_stops_the_walk(self):
        s = machine.situation(machine.Facts(files_ready=1, files_failed=1, analysed=True, workflow_ready=True))
        self.assertEqual((s.state, s.next, s.blockers[0].code), ("data", "stop", "files_failed"))

    def test_from_the_tables_draft_the_walk_points_to_the_draft_with_its_cases(self):
        s = machine.situation(machine.Facts(files_ready=1, blocking_questions=[{"field": "labels", "text": "Known cases?"}],
                                            cases_elsewhere=[{"draft_id": "d9", "key": "order_id", "coverage": 0.95}]))
        self.assertEqual((s.state, s.next, s.blockers[0].evidence["draft_id"]), ("links", "stop", "d9"))
        s = machine.situation(machine.Facts(files_ready=1, analysed=True, workflow_ready=True, workflow_problems=["no steps"]))
        self.assertEqual((s.state, s.next), ("workflow", "stop"))


@unittest.skipIf(MISSING, MISSING)
class MorningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        replica = dfr.generate(seed=0, n_orders=6000, n_bikers=240)
        cls.orders, cls.cases = replica.rides, replica.sample.assign(final_status="DELIVERED")

    def setUp(self):
        home = Path(tempfile.mkdtemp())
        self.store, self.projects = DraftStore(home / "drafts"), ProjectStore(home / "projects")
        self.agent = HomeAgent(self.store, None)
        self.orders_draft = self.store.create(PROBLEM)["id"]
        self.agent.start(self.orders_draft)
        upload(self.store, self.agent, self.orders_draft, "part2-dataset.csv", self.orders)
        self.cases_draft = self.store.create(PROBLEM)["id"]
        self.agent.start(self.cases_draft)
        upload(self.store, self.agent, self.cases_draft, "part2-fraud-sample.csv", self.cases)
        self.log = experience.FileLog(home / "experience", "wtest")
        experience.install(self.log)
        self.addCleanup(experience.install, None)

    def walk(self, advisor=None, answers=ANSWERS):
        return Coordinator(self.store, self.projects, self.agent, advisor).run(self.cases_draft, person(answers))

    def test_the_mornings_walk_is_a_run_of_the_coordinator(self):
        bad = ScriptedAdvisor({"data_investigator": {"after": [{"column": "driver_rating", "why": "rated after the ride"}],
                                                     "entities": [{"column": "biker_id", "why": "the bikers repeat"}]}})
        out = self.walk(bad)
        self.assertEqual((out["state"], out["next"]), ("built", "done"), out["says"])
        actions = [s["action"] for s in out["steps"]]
        self.assertIn("bring_table", actions)  # the orders came from the other draft
        self.assertLess(actions.index("bring_table"), actions.index("build"))
        project = self.projects.get(out["project_id"])
        self.assertEqual(project["labels"]["key"], ["order_id"])  # labels joined on the order
        self.assertEqual((project["suggestion"]["labels"], project["suggestion"]["group_column"]), ("partial", "biker_id"))
        self.assertEqual(project["draft"]["card"]["family"]["value"], "rare_event")
        handoff = self.store.get(self.cases_draft)["coordinator"]["handoffs"][0]
        self.assertEqual((handoff["allowed"], handoff["refused_columns"], handoff["source"]), (False, ["driver_rating"], "rules"))
        self.assertNotIn("why", json.dumps(handoff["output"]))  # columns only, never the model's words
        events = self.log.read()
        refused = [e for e in events if e.kind == "decision.refused" and e.payload.get("point") == "handoff"]
        self.assertTrue(refused and "driver_rating" in refused[0].payload["reason"])
        steps = [e for e in events if e.kind == "agent.step" and e.payload["agent"] == "coordinator"]
        self.assertEqual([e.payload.get("tool") for e in steps], actions)
        ended = [e for e in events if e.kind == "run.recorded" and e.payload["name"] == "coordinator"]
        self.assertEqual(ended[-1].payload["status"], "completed")

    def test_a_table_that_could_not_be_read_is_never_built_around(self):
        from unittest import mock

        from dclab_rnd.draft import analyze

        real, calls = analyze.analyze, {"n": 0}

        def breaks(frame, **kw):  # the brought table's analysis fails, after the cases moved aside for it
            calls["n"] += 1
            if len(frame) > 1000:
                raise RuntimeError("analysis broke")
            return real(frame, **kw)

        with mock.patch.object(analyze, "analyze", side_effect=breaks):
            out = self.walk()
        self.assertEqual(out["next"], "stop", out)
        self.assertIsNone(out["project_id"])
        self.assertIn("could not be read", out["says"])
        from dclab_rnd.draft.api import BuildRefused, build_project

        with self.assertRaises(BuildRefused):  # the build route refuses it too: the cases are not the table
            build_project(self.store, self.projects, self.cases_draft)

    def test_from_the_orders_draft_it_points_to_the_cases(self):
        out = Coordinator(self.store, self.projects, self.agent).run(self.orders_draft, person())
        self.assertEqual((out["next"], out["project_id"]), ("stop", None))
        self.assertIn(self.cases_draft, out["says"])

    def test_it_never_builds_while_a_blocker_is_open_and_says_which(self):
        asked = person({})
        out = Coordinator(self.store, self.projects, self.agent).run(self.cases_draft, asked)  # the person answers nothing
        self.assertEqual(out["next"], "stop")
        self.assertEqual(len(asked.asked), 1)  # the question that stopped it is in the person's record
        self.assertIsNone(out["project_id"])
        self.assertTrue(out["says"] and out["blockers"] or "Waiting for the person" in out["says"], out)
        self.assertNotIn("build", [s["action"] for s in out["steps"]])
        ended = [e for e in self.log.read() if e.kind == "run.recorded" and e.payload["name"] == "coordinator"]
        self.assertEqual(ended[-1].payload["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
