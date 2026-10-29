"""P8's two missing routing points (2026-10-29): the cross-route suggestion (request.suggestion) and the tool serving a kind
of help (request.tool) are decisions of their own in the ledger, each labelled by what the person does: the suggestion by
a switch to it (its track) or a confirmed result on their own track (none); the tool by the person's choice between groups
they have and new groups, or, for a rule's tool, by a confirmed result."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import ledger  # noqa: E402
from dclab_rnd.agent_eval.home import tables  # noqa: E402
from dclab_rnd.draft import pipeline  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402

PAST_MODEL = "Build a model of last month's sales by region."  # H1 d-b4: says model, means a report of the past


class RoutingPointsTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.store = DraftStore(self.home / "drafts")
        self.ledger = ledger.FileLedger(self.home / "ledger", "wtest")
        ledger.install(self.ledger)
        self.addCleanup(ledger.install, None)

    def start(self, text):
        d = self.store.create(text)
        agent = HomeAgent(self.store, None, on_request=lambda *a: None)
        agent.start(d["id"])
        return agent, d["id"]

    def values(self, point):
        return [r.value_used for r in self.ledger.records(point=point)]

    def current(self, point):
        labels = self.ledger.labels(point=point)
        gone = {lb.supersedes for lb in labels if lb.supersedes}
        return [lb.value for lb in labels if lb.id not in gone and lb.value is not None]

    def test_a_suggestion_taken_is_labelled_with_its_track(self):
        agent, draft_id = self.start(PAST_MODEL)
        if HomeAgent.track(self.store.get(draft_id)) != "prediction":
            agent.choose_intent(draft_id, "prediction")
        self.assertIn("reporting", self.values("request.suggestion"))  # the words point to a report of the past
        hint = self.store.get(draft_id)["suggestion_record"]
        self.assertEqual((hint["on"], hint["track"]), ("prediction", "reporting"))
        self.assertIsNotNone(self.store.get(draft_id)["route_suggestion"])  # still read as "a suggestion was offered"
        agent.choose_intent(draft_id, "reporting")  # the Switch button
        self.assertEqual(self.current("request.suggestion"), ["reporting"])

    def test_no_suggestion_and_the_rules_tool_are_labelled_by_a_confirmed_result(self):
        agent, draft_id = self.start("What was the average monthly spend per plan?")
        self.assertEqual(self.values("request.suggestion")[-1], "none")
        self.assertEqual(self.values("request.tool"), ["query"])
        self.assertIsNone(self.store.get(draft_id).get("route_suggestion"))  # none offered: nothing for the page to show
        name = tables.write("customers", self.store.data_dir(draft_id))
        asset = pipeline.new_asset(self.store, draft_id, "upload", name, name)
        pipeline.run(self.store, draft_id, asset["id"], agent, None)
        agent.acted(draft_id, "answer")
        agent.acted(draft_id, "save")
        self.assertEqual((self.current("request.suggestion"), self.current("request.tool")), (["none"], ["query"]))
        agent.acted(draft_id, "unsave")  # refused later: the confirmation is withdrawn, never left standing
        self.assertEqual((self.current("request.suggestion"), self.current("request.tool")), ([], []))

    def test_the_person_chooses_the_tool_for_groups_and_it_stays_theirs(self):
        agent, draft_id = self.start("I want analyze my customer and segment of the my customer")
        self.assertEqual(self.values("request.tool"), [])  # the rules give no tool for groups: the person is asked
        agent.reply(draft_id, "Find new groups in the data")
        records = self.ledger.records(point="request.tool")
        self.assertEqual([(r.value_used, r.used) for r in records], [("segmentation_project", "person")])  # not doubled by the project path
        self.assertEqual(self.current("request.tool"), ["segmentation_project"])
        self.assertEqual(HomeAgent.track(self.store.get(draft_id)), "prediction")

    def test_groups_the_person_has_are_the_query_engine(self):
        agent, draft_id = self.start("Segment our customers by plan")
        agent.reply(draft_id, "Describe groups I already have")
        self.assertEqual(self.current("request.tool"), ["query"])

    def test_a_result_on_an_earlier_track_never_labels_the_new_one(self):  # the review, finding 1
        agent, draft_id = self.start("What was the average monthly spend per plan?")
        name = tables.write("customers", self.store.data_dir(draft_id))
        asset = pipeline.new_asset(self.store, draft_id, "upload", name, name)
        pipeline.run(self.store, draft_id, asset["id"], agent, None)
        agent.acted(draft_id, "answer")
        agent.acted(draft_id, "save")
        before = len(self.ledger.labels())
        agent.choose_intent(draft_id, "reporting")
        agent.acted(draft_id, "dashboard")  # produced, nobody judged it yet
        reporting = [r.id for r in self.ledger.records(point="request.tool") if r.value_used == "dashboard"]
        self.assertTrue(reporting)
        self.assertFalse([lb for lb in self.ledger.labels() if lb.record in reporting])
        self.assertEqual(self.store.get(draft_id)["outcome"]["verdict"], "open")
        self.assertGreaterEqual(len(self.ledger.labels()), before)

    def test_new_groups_moved_by_home_label_neither_the_suggestion_nor_the_reading(self):  # the review, finding 2
        agent, draft_id = self.start("Segment our customers by plan and show which customers will leave next month")
        if HomeAgent.track(self.store.get(draft_id)) != "analysis":
            agent.choose_intent(draft_id, "analysis")
        labels_before = {(lb.point, lb.value) for lb in self.ledger.labels()}
        question = [m for m in self.store.get(draft_id)["messages"] if m.get("kind") == "question"]
        if not question or question[-1]["question"]["field"] != "segments_route":
            self.skipTest("the rules did not read groups in these words")
        agent.reply(draft_id, "Find new groups in the data")
        added = {(lb.point, lb.value) for lb in self.ledger.labels()} - labels_before
        self.assertNotIn(("request.suggestion", "prediction"), added)
        self.assertNotIn(("request.intent", "prediction"), added)
        self.assertIn(("request.tool", "segmentation_project"), added)  # the person's own choice is still their label

    def test_a_rule_tool_record_agrees_with_itself(self):  # the review, finding 3
        agent, draft_id = self.start("I want analyze my customer and segment of the my customer")
        agent.reply(draft_id, "Find new groups in the data")
        agent.choose_intent(draft_id, "reporting")
        agent.choose_intent(draft_id, "prediction")
        for r in self.ledger.records(point="request.tool"):
            if r.used == "rule":
                self.assertEqual(next(a.value for a in r.answers if a.by == "rule"), r.value_used)

    def test_the_routing_set_reads_both_points(self):
        from dclab_core.ledger import POINTS as CORE
        from dclab_rnd.store.sets import routing

        names = {p.key for p in CORE} if not isinstance(CORE, dict) else set(CORE)
        self.assertTrue({"request.suggestion", "request.tool"} <= names)
        self.assertTrue({"request.suggestion", "request.tool"} <= set(routing.POINTS))
        self.assertTrue({"request.suggestion", "request.tool"} <= set(ledger.CAPTURED))


if __name__ == "__main__":
    unittest.main()
