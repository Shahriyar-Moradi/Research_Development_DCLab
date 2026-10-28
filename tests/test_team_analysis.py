"""Package 28.4 (the rest): the understanding team's analysis as two nodes in parallel. Once a table is there the walk sends
the problem analyst (the first sentence and every answer read into the brief, with the table's columns) and the data
investigator (the coordinator's own analysis) in the same step, joins them on the board as one hand-off, and asks nothing
twice. The critic's questions are marked in the conversation events, so its questions and the person's answers are
conversation examples of their own."""

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
    from tests.test_coordinator import ANSWERS, MISSING as COORD_MISSING, PROBLEM, dfr, upload
except ImportError as error:  # pandas or langgraph are not in every environment
    MISSING = str(error)
else:
    MISSING = COORD_MISSING

ACTION = "to send each one to the review team"


@unittest.skipIf(MISSING, MISSING)
class ParallelAnalysisTests(unittest.TestCase):
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
        orders = self.store.create(PROBLEM)["id"]  # the table the cases belong to, brought by the walk
        self.agent.start(orders)
        upload(self.store, self.agent, orders, "part2-dataset.csv", self.orders)
        self.draft = self.store.create(PROBLEM)["id"]
        self.agent.start(self.draft)
        upload(self.store, self.agent, self.draft, "part2-fraud-sample.csv", self.cases)
        self.log = experience.FileLog(self.home / "experience", "atest")
        experience.install(self.log)
        self.addCleanup(experience.install, None)

    def walk(self, **kw):
        from dclab_rnd.draft.coordinator import Coordinator
        from dclab_rnd.orchestra import checkpoints
        from dclab_rnd.orchestra.walk import Walk

        saver_cm = checkpoints.open_checkpointer(self.home, database_url="")
        saver = saver_cm.__enter__()
        self.addCleanup(saver_cm.__exit__, None, None, None)
        return Walk(Coordinator(self.store, self.projects, self.agent), saver, home=self.home, **kw)

    def drive(self, walk, answers=ANSWERS):
        from dclab_rnd.orchestra.walk import drive
        from dclab_rnd.studio import team

        asked = []
        person = team.Person(answer=lambda field, question: asked.append(field) or answers.get(field), approve=lambda gate, summary: True, who="p")
        return drive(walk, self.draft, person), asked

    def test_the_analyst_and_the_investigator_run_in_the_same_step_and_join_as_one_hand_off(self):
        from dclab_rnd.orchestra import thread_id

        walk = self.walk()
        out, asked = self.drive(walk)
        self.assertEqual(out["next"], "done", out["says"])
        config = {"configurable": {"thread_id": thread_id("draft", self.draft)}}
        scheduled = [tuple(sorted(s.next)) for s in walk.graph.get_state_history(config)]
        self.assertIn(("analyse_data", "analyse_problem"), scheduled)  # both in one superstep: the two readers run in parallel
        analysed = [s for s in out["steps"] if s["action"] == "analyse"]
        self.assertEqual(len(analysed), 1)
        self.assertEqual(set(analysed[0]["result"]), {"problem", "data"})
        self.assertEqual(analysed[0]["result"]["data"]["role"], "data_investigator")  # the coordinator's own analysis, unchanged
        self.assertEqual(analysed[0]["result"]["problem"]["role"], "problem_analyst")
        events = self.log.read()
        tools = [e.payload.get("tool") for e in events if e.kind == "agent.step" and e.payload["agent"] == "orchestra.walk"]
        self.assertIn("analyse_problem", tools)
        self.assertIn("analyse_data", tools)
        handoffs = [e.payload for e in events if e.kind == "team.handoff"]
        self.assertEqual(len(handoffs), len(out["steps"]))  # the join advanced the board once for the pair
        self.assertEqual(len(asked), len(set(asked)))  # nothing asked twice

    def test_the_analyst_reads_every_answer_into_the_brief_and_never_overwrites_a_field(self):
        answers = {**ANSWERS, "budget": f"About 80 a day, {ACTION}"}
        out, _ = self.drive(self.walk(), answers)
        self.assertEqual(out["next"], "done", out["says"])
        draft = self.store.get(self.draft)
        self.assertEqual(draft["understanding"].get("action"), ACTION)  # the person's words, said in an answer to another question
        self.assertEqual(draft["plan"]["action"]["source"], "analyst")
        reread = [s for s in out["steps"] if s["action"] == "analyse_problem"]  # the answers after the analysis, read before the critic
        self.assertEqual([s["result"]["last"] for s in reread], [["action"]])
        actions = [s["action"] for s in out["steps"]]
        self.assertLess(actions.index("analyse_problem"), actions.index("critic"))  # the critic reads the card with every answer in it
        own = draft["coordinator"]["analyst"]
        self.assertEqual(own["filled"], ["action"])
        self.assertEqual(own["source"], "rules")  # no model serves the analyst until its gate passed
        self.assertNotIn(ACTION, json.dumps(own))  # the draft keeps the words in the understanding, the record names the fields

        before = dict(self.store.get(self.draft)["understanding"])
        result = self.agent.read_answers(self.draft)  # again: every field it could read is set, nothing is overwritten
        self.assertEqual((result["last"], result["filled"]), ([], ["action"]))  # nothing new this time; what it ever filled is kept (finding 4)
        self.assertEqual(self.store.get(self.draft)["understanding"], before)

    def test_the_analyst_never_takes_the_outcome_from_an_answer_the_data_settles_it(self):  # the review of 28.4, finding 1
        self.draft = self.store.create(PROBLEM)["id"]  # the owner's morning: the orders and the known cases in one draft, the outcome settled by the cases
        self.agent.start(self.draft)
        upload(self.store, self.agent, self.draft, "part2-dataset.csv", self.orders)
        upload(self.store, self.agent, self.draft, "part2-fraud-sample.csv", self.cases)
        answers = {**ANSWERS, "prediction_moment": "After delivery, in the daily audit; the courier app records the fare in biker_fare once the order is delivered",
                   "labels": "A sample of the cases we found; status is the outcome the app writes after delivery"}
        out, asked = self.drive(self.walk(), answers)
        self.assertEqual(out["next"], "done", out["says"])
        draft = self.store.get(self.draft)
        self.assertNotEqual((draft["plan"].get("target") or {}).get("source"), "analyst")
        self.assertNotIn("target", draft["coordinator"]["analyst"]["filled"])
        self.assertNotIn(str(draft["understanding"].get("target") or ""), ("status", "records the fare in biker_fare"))
        self.assertNotIn("target", asked)  # the outcome is never asked: the cases settled it, and the critic never sends it back

    def test_a_failing_analyst_is_recorded_and_the_walk_goes_on_to_the_critic(self):  # the review of 28.4, finding 2
        from unittest import mock

        with mock.patch.object(self.agent, "read_answers", side_effect=RuntimeError("the reader broke")):
            out, _ = self.drive(self.walk())
        self.assertEqual(out["next"], "done", out["says"])
        actions = [s["action"] for s in out["steps"]]
        self.assertIn("critic", actions)
        self.assertEqual(self.store.get(self.draft)["coordinator"]["analyst"]["error"], "RuntimeError")
        analysed = next(s for s in out["steps"] if s["action"] == "analyse")
        self.assertEqual(analysed["result"]["problem"]["error"], "RuntimeError")  # recorded, never the walk's failure

    def test_the_analysts_readings_are_understanding_examples_with_the_brief_before_and_after(self):  # finding 3
        from dclab_core import datasets as builders
        from dclab_rnd import consent, datasets as studio_sets
        from dclab_rnd.lake import Run, session_examples

        consent.set_workspace_consent(self.home, "train", True)  # before the walk: its boards are written at that level
        answers = {**ANSWERS, "budget": f"About 80 a day, {ACTION}"}
        out, _ = self.drive(self.walk(), answers)
        project = self.projects.get(out["project_id"])
        project["settings"]["consent"] = {"train": True}
        self.projects.save(project)
        items = studio_sets._team_items(self.home, {"train": True}, {project["id"]: project}, {self.draft: self.store.get(self.draft)})
        rows, manifest = builders.team_walks(items)
        mine = [r for r in rows if r["node"] in ("analyse", "analyse_problem")]
        self.assertEqual({r["family"] for r in mine}, {"understand_problem"})
        reread = next(r for r in mine if r["node"] == "analyse_problem")
        self.assertIsNone(reread["before"]["problem"].get("action"))  # the brief before the analyst read the answers
        self.assertTrue(reread["after"]["problem"].get("action"))  # and after
        examples = session_examples({"team_walks/boards": [{"row": r, "source": "w", "from_sealed": False} for r in rows]}, Run())
        self.assertIn("understand_problem", {e.family for e in examples})

    def test_without_the_parallel_analysis_the_walk_reaches_the_same_project(self):
        out, _ = self.drive(self.walk(parallel=False))
        self.assertEqual(out["next"], "done", out["says"])
        analysed = [s for s in out["steps"] if s["action"] == "analyse"]
        self.assertEqual(len(analysed), 1)
        self.assertNotIn("problem", analysed[0]["result"])  # the coordinator's analysis alone, as the loop does it

    def test_a_critics_question_is_marked_in_the_conversation_events(self):
        q = self.agent.ask(self.draft, "entity", "What is one row: an order or a biker?", source="critic")
        self.assertIsNotNone(q)
        self.agent.reply(self.draft, "One order")
        self.agent.say(self.draft, "Thanks: one row is one order.", kind="text")  # an example ends on the agent's turn
        events = [e for e in self.log.read() if e.kind == "chat.agent" and e.payload.get("question_field") == "entity"]
        self.assertEqual(events[-1].payload.get("question_source"), "critic")
        from dclab_core import datasets

        ids = {"conversation": self.draft}
        logged = [e.model_dump(mode="json") for e in self.log.read()]
        rows, _ = datasets.conversations([{"kind": "draft.created", "ids": ids, "payload": {"problem": PROBLEM}, "at": "0000"}, *logged,
                                          {"kind": "draft.built", "ids": ids, "payload": {}, "at": "9999"}])  # a conversation that ended in a build
        critic = rows[0]["critic"]
        self.assertEqual([c["field"] for c in critic], ["entity"])
        turn = rows[0]["messages"][critic[0]["turn"]]
        self.assertEqual(turn["role"], "assistant")
        self.assertIn("What is one row", turn["content"])
        self.assertIn("One order", rows[0]["messages"][critic[0]["turn"] + 1]["content"])  # the person's answer is the next turn
        self.agent.ask(self.draft, "action", "What happens with each flagged order?", source="critic")  # asked last, never answered
        logged = [e.model_dump(mode="json") for e in self.log.read()]
        rows, _ = datasets.conversations([{"kind": "draft.created", "ids": ids, "payload": {"problem": PROBLEM}, "at": "0000"}, *logged,
                                          {"kind": "draft.built", "ids": ids, "payload": {}, "at": "9999"}])
        self.assertEqual([c["field"] for c in rows[0]["critic"]], ["entity"])  # a question whose answer the example lacks is not listed (finding 6)



@unittest.skipIf(MISSING, MISSING)
class PostgresParallelTests(unittest.TestCase):
    """The two readers' writes to one draft on PostgreSQL, from two threads while the walk holds the draft's turn."""

    def test_the_parallel_analysis_on_postgresql_reaches_the_build_with_both_readers_writes(self):
        sys.path.insert(0, str(ROOT / "tests"))
        import pgtest

        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.coordinator import Coordinator
        from dclab_rnd.orchestra import checkpoints, thread_id
        from dclab_rnd.orchestra.walk import Walk, drive
        from dclab_rnd.storage import postgres
        from dclab_rnd.studio import team

        url = pgtest.require()
        pgtest.empty(url)
        home = Path(tempfile.mkdtemp())
        projects, drafts, _ = postgres.open_stores(home, url)
        experience.install(experience.FileLog(home / "experience", "pg"))
        self.addCleanup(experience.install, None)
        agent = HomeAgent(drafts, None)
        replica = dfr.generate(seed=0, n_orders=6000, n_bikers=240)
        draft = drafts.create(PROBLEM)["id"]
        agent.start(draft)
        upload(drafts, agent, draft, "part2-dataset.csv", replica.rides)
        upload(drafts, agent, draft, "part2-fraud-sample.csv", replica.sample.assign(final_status="DELIVERED"))
        answers = {**ANSWERS, "budget": f"About 80 a day, {ACTION}"}
        person = team.Person(answer=lambda field, question: answers.get(field), approve=lambda gate, summary: True, who="p")
        with checkpoints.open_checkpointer(home, database_url="") as saver:
            walk = Walk(Coordinator(drafts, projects, agent), saver, home=home)
            out = drive(walk, draft, person)
            scheduled = [tuple(sorted(s.next)) for s in walk.graph.get_state_history({"configurable": {"thread_id": thread_id("draft", draft)}})]
        self.assertEqual(out["next"], "done", out["says"])
        self.assertIn(("analyse_data", "analyse_problem"), scheduled)
        own = drafts.get(draft)["coordinator"]
        self.assertTrue(own["handoffs"] and own["analyst"]["read"])  # the investigator's and the analyst's writes both kept
        self.assertEqual(drafts.get(draft)["understanding"]["action"], ACTION)


class ContractTests(unittest.TestCase):
    def test_the_critic_is_named_as_meta_on_a_v11_event_and_a_v10_event_reads_as_v11(self):
        from dclab_core import experience as x

        ev = x.make("chat.agent", "agent", {"workspace": "w1", "conversation": "d1"},
                    {"text": "What is one row: an order or a biker?", "kind": "question", "question_field": "entity", "question_source": "critic"},
                    id="ev0000101", at="2026-10-18T10:00:00+00:00", seq=1)
        self.assertEqual((ev.schema_version, ev.classes["question_source"]), ("dclab_core.experience_event.v16", "meta"))
        old = {"schema_version": "dclab_core.experience_event.v10", "id": "ev0000102", "at": "2026-10-18T10:00:00+00:00", "seq": 0, "ids": {"workspace": "w1"},
               "actor": "agent", "kind": "chat.agent", "payload": {"kind": "question", "question_field": "entity"}, "classes": {"kind": "meta", "question_field": "meta"}}
        self.assertEqual(x.ExperienceEvent.read(old).schema_version, "dclab_core.experience_event.v16")


if __name__ == "__main__":
    unittest.main()
