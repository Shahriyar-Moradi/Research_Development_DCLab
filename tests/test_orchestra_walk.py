"""Package 28.3: the coordinator's state machine as the orchestration layer's graph: its nodes call the coordinator's own
actions, the person is a pause, a build is approved first, every node advances the board, and the dev journeys reach
the same result through the graph as through the loop."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import langgraph  # noqa: F401

    from dclab_rnd import experience
    from dclab_rnd.agent_eval import journeys
    from tests.test_coordinator import ANSWERS, MISSING as COORD_MISSING, PROBLEM, dfr, upload
except ImportError as error:  # pandas or langgraph are not in every environment
    MISSING = str(error)
else:
    MISSING = COORD_MISSING


@unittest.skipIf(MISSING, MISSING)
class WalkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        replica = dfr.generate(seed=0, n_orders=6000, n_bikers=240)
        cls.orders, cls.cases = replica.rides, replica.sample.assign(final_status="DELIVERED")

    def setUp(self):
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore
        from dclab_rnd.studio import ProjectStore

        self.home = Path(tempfile.mkdtemp())
        self.store, self.projects = DraftStore(self.home / "drafts"), ProjectStore(self.home / "projects")
        self.agent = HomeAgent(self.store, None)
        self.orders_draft = self.store.create(PROBLEM)["id"]
        self.agent.start(self.orders_draft)
        upload(self.store, self.agent, self.orders_draft, "part2-dataset.csv", self.orders)
        self.cases_draft = self.store.create(PROBLEM)["id"]
        self.agent.start(self.cases_draft)
        upload(self.store, self.agent, self.cases_draft, "part2-fraud-sample.csv", self.cases)
        self.log = experience.FileLog(self.home / "experience", "wtest")
        experience.install(self.log)
        self.addCleanup(experience.install, None)

    def walk(self, home):
        from dclab_rnd.draft.coordinator import Coordinator
        from dclab_rnd.orchestra import checkpoints
        from dclab_rnd.orchestra.walk import Walk

        saver_cm = checkpoints.open_checkpointer(home, database_url="")
        saver = saver_cm.__enter__()
        self.addCleanup(saver_cm.__exit__, None, None, None)
        return Walk(Coordinator(self.store, self.projects, self.agent), saver, home=home)

    def test_the_walk_is_a_session_of_the_orchestra_walk_agent(self):  # the harness's probe; the morning, through the graph
        from dclab_rnd.orchestra.walk import drive
        from dclab_rnd.studio import team

        asked, gates = [], []
        person = team.Person(answer=lambda field, question: asked.append(field) or ANSWERS.get(field),
                             approve=lambda gate, summary: gates.append(gate) or True, who="scripted person")
        out = drive(self.walk(self.home), self.cases_draft, person)
        self.assertEqual(out["next"], "done", out["says"])
        self.assertTrue(out["project_id"])
        self.assertIn("build", gates)  # a build is approved first
        self.assertIn("build", [s["action"] for s in out["steps"]])
        events = self.log.read()
        steps = [e for e in events if e.kind == "agent.step" and e.payload["agent"] == "orchestra.walk"]
        self.assertIn("build", [e.payload.get("tool") for e in steps])
        handoffs = [e.payload for e in events if e.kind == "team.handoff"]
        self.assertEqual(len(handoffs), len(out["steps"]))  # every action node advanced the board
        board = json.loads((self.home / "orchestra" / "boards.jsonl").read_text().splitlines()[-1])["board"]
        self.assertEqual(board["state"], "built")  # the board after a node is the draft after it (the review of 28.3)
        self.assertEqual(board["problem"]["outcome"] is None or isinstance(board["problem"]["outcome"], str), True)
        teams = [json.loads(x) for x in (self.home / "orchestra" / "teams.jsonl").read_text().splitlines()]
        self.assertEqual(teams[-1]["status"], "completed")
        self.assertEqual(teams[-1]["interrupts"], len(asked) + len(gates))  # every pause closed on the board, answered
        self.assertLess(teams[-1]["started"], teams[-1]["finished"])  # the walk's start, not its end
        self.assertEqual([e.payload["interrupts"] for e in events if e.kind == "team.recorded"], [teams[-1]["interrupts"]])
        self.assertEqual([e.payload["status"] for e in steps if e.payload.get("verdict") == "error"], [])  # a pause is no failed step
        closed = [e.payload for e in events if e.kind == "team.interrupt" and e.payload["status"] != "open"]
        self.assertEqual({p["interrupt"] for p in closed}, {"question", "approval", *({"confirmation"} if "map" in gates else set())})
        self.assertEqual({p["gate"] for p in closed if p["interrupt"] == "confirmation"}, {"map"} & set(gates))  # the map is confirmed

    def test_a_declined_build_stops_and_says_why(self):
        from dclab_rnd.orchestra.walk import drive
        from dclab_rnd.studio import team

        person = team.Person(answer=lambda field, question: ANSWERS.get(field), approve=lambda gate, summary: gate != "build", who="p")
        out = drive(self.walk(self.home), self.cases_draft, person)
        self.assertEqual(out["next"], "stop")
        self.assertIn("approval to build", out["says"])
        self.assertFalse(self.store.get(self.cases_draft).get("project_id"))  # never built without the person's approval
        events = self.log.read()
        declined = [e.payload for e in events if e.kind == "team.interrupt" and e.payload["status"] == "declined"]
        self.assertEqual([(p["interrupt"], p["gate"]) for p in declined], [("approval", "build")])  # the decline is recorded
        approve = [e.payload for e in events if e.kind == "agent.step" and e.payload.get("tool") == "approve"]
        self.assertTrue(approve)

    def _to_the_build_approval(self, walk):
        out = walk.run(self.cases_draft)
        while out["status"] == "paused" and out["asks"]["kind"] != "approval":
            if out["asks"]["kind"] == "question":
                out = walk.run(self.cases_draft, resume=ANSWERS.get(out["asks"]["field"]) or "I don't know")
            else:
                out = walk.run(self.cases_draft, resume=True)
        self.assertEqual((out["status"], out["asks"].get("gate")), ("paused", "build"))
        return out

    def test_a_build_made_while_its_approval_waited_is_not_made_twice(self):  # the review of 28.3, finding 1
        from dclab_rnd.draft.api import build_project

        walk = self.walk(self.home)
        self._to_the_build_approval(walk)
        build_project(self.store, self.projects, self.cases_draft)  # the page's Build button, meanwhile
        out = walk.run(self.cases_draft, resume=True)
        self.assertEqual(out["next"], "done")
        self.assertEqual(len(self.projects.list()), 1)  # one draft, one project

    def test_the_walk_resumes_from_a_new_walk_on_the_same_checkpoints(self):  # another process: the checkpoint holds the walk
        first = self.walk(self.home)
        self._to_the_build_approval(first)
        out = self.walk(self.home).run(self.cases_draft, resume=True)
        self.assertEqual(out["next"], "done", out["says"])
        self.assertTrue(self.store.get(self.cases_draft).get("project_id"))

    def test_an_answer_is_taken_only_for_the_question_it_was_given_to(self):  # the review of 28.3, finding 2
        walk = self.walk(self.home)
        out = walk.run(self.cases_draft)
        self.assertEqual((out["status"], out["asks"]["kind"]), ("paused", "question"), out)
        first = out["asks"]["field"]
        self.agent.reply(self.cases_draft, ANSWERS.get(first) or "I don't know")  # answered in the chat while the walk waited
        now = self.agent.pending(self.store.get(self.cases_draft))
        out = walk.run(self.cases_draft, resume="An answer to the first question")
        u = self.store.get(self.cases_draft).get("understanding") or {}
        self.assertNotIn("An answer to the first question", json.dumps(u))  # never recorded against another question
        self.assertIsNotNone(now)  # the scenario: another question is open when the walk resumes
        self.assertNotEqual(now["field"], first)
        self.assertEqual((out["status"], out["asks"].get("field")), ("paused", now["field"]))  # asked what is open now

    def test_deleting_the_draft_deletes_its_walk_checkpoints(self):  # the review of 28.3, finding 8: they hold the answers
        from dclab_rnd.experience import retention
        from dclab_rnd.orchestra import checkpoints

        self._to_the_build_approval(self.walk(self.home))
        self.log.home = self.home
        retention.forget(draft=self.cases_draft, log=self.log)
        with checkpoints.open_checkpointer(self.home, database_url="") as saver:
            self.assertIsNone(saver.get_tuple({"configurable": {"thread_id": f"draft:{self.cases_draft}"}}))


@unittest.skipIf(MISSING, MISSING)
class JourneyParityTests(unittest.TestCase):
    def test_dev_journeys_reach_the_same_result_through_the_graph(self):
        people = [j for j in journeys.load("dev") if j["id"] in ("JD-01", "JD-03", "JD-04")]
        for person in people:
            loop = journeys.play(person)
            graph = journeys.play(person, walk="graph")
            self.assertEqual((graph["right"], graph["parts"]), (loop["right"], loop["parts"]), person["id"])


if __name__ == "__main__":
    unittest.main()
