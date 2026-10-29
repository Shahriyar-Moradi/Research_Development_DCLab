"""Home's understanding conversation (the owner's Telco test, 2026-10-29: "talk naturally for about three turns before
offering options, and make the options fit the context"). The model leads; code checks every sentence, card field and
option, and decides when Home has understood enough to offer them."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import ledger  # noqa: E402
from dclab_rnd.draft import pipeline  # noqa: E402
from dclab_rnd.draft import understand as U  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402

FIRST = "how can I analyze my customer"


class Script:
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def complete(self, messages, tools=None, *args, **kwargs):
        self.seen.append(messages)
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        return {"content": reply if isinstance(reply, str) else json.dumps(reply)}


def turn(say, enough=False, options=None, **card):
    return {"say": say, "card": card, "enough": enough, "options": options or []}


SEGMENTS = {"title": "Find groups of customers who behave alike", "what": "Groups found in your data, each described.",
            "track": "prediction", "tool": "segmentation_project", "columns": []}
CHURN = {"title": "Predict who will leave next month", "what": "A score for each customer.", "track": "prediction", "tool": "ml_project",
         "columns": []}
COMPARE = {"title": "Compare churn by contract", "what": "Churn per contract type.", "track": "analysis", "tool": "query", "columns": []}


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.store = DraftStore(self.home / "drafts")
        self.book = ledger.FileLedger(self.home / "ledger", "wtest")
        ledger.install(self.book)
        self.addCleanup(ledger.install, None)

    def agent(self, understand):
        # the model's first reading of the owner's sentence was unclear (their session): the understanding conversation begins
        unclear = Script(json.dumps({"tracks": [], "best_question": "What would you like to find out about your customers?"}))
        return HomeAgent(self.store, None, on_request=lambda *a: None, intent_client=unclear, understand_client=understand)

    def questions(self, draft_id):
        return [m for m in self.store.get(draft_id)["messages"] if m.get("kind") == "question"]

    def test_three_natural_turns_then_options_from_the_context_and_the_pick_routes(self):
        model = Script(
            turn("You want to understand your customers better. What are you hoping to change or decide?", tracks=["analysis", "prediction"]),
            turn("Keeping more customers from leaving is a clear goal. What data do you have about them?",
                 goal={"value": "keep more customers from leaving", "quote": "keep more customers from leaving"}, tracks=["prediction"]),
            turn("Thanks, a customer table with who left is a good start.", enough=True, options=[SEGMENTS, CHURN, COMPARE],
                 goal={"value": "keep more customers from leaving", "quote": "keep more customers from leaving"},
                 data_said={"value": "a customer table with who left", "quote": "a customer table with who left"}, tracks=["prediction", "analysis"]))
        d = self.store.create(FIRST)
        agent = self.agent(model)
        agent.start(d["id"])
        first = self.questions(d["id"])[-1]
        self.assertEqual(first["question"]["options"], [U.SHOW_OPTIONS])  # a conversation, never six buttons on the first turn
        self.assertTrue(first["question"]["understand"])
        self.assertIn("What are you hoping to change", first["text"])
        agent.reply(d["id"], "I want to keep more customers from leaving")
        self.assertIn("What data do you have", self.questions(d["id"])[-1]["text"])
        agent.reply(d["id"], "I have a customer table with who left and their contract")
        offer = self.questions(d["id"])[-1]
        self.assertEqual(offer["question"]["options"], [SEGMENTS["title"], CHURN["title"], COMPARE["title"], U.SOMETHING_ELSE])
        sent = model.seen[-1]
        self.assertEqual([m["role"] for m in sent[2:]], ["user", "assistant", "user", "assistant", "user"])  # it sees its own questions
        agent.reply(d["id"], SEGMENTS["title"])
        draft = self.store.get(d["id"])
        self.assertEqual((HomeAgent.track(draft), draft["understanding"].get("family")), ("prediction", "segmentation"))
        self.assertEqual([r.value_used for r in self.book.records(point="request.understand_turn")], ["ask", "ask", "offer"])
        self.assertEqual([lb.value for lb in self.book.labels(point="request.options")], ["prediction:segmentation_project"])
        self.assertEqual([lb.value for lb in self.book.labels(point="request.tool")], ["segmentation_project"])

    def test_the_turn_limit_offers_the_rules_options_when_the_model_never_settles(self):
        model = Script(*[turn("Tell me more about what you need?", tracks=["analysis"]) for _ in range(5)])
        d = self.store.create(FIRST)
        agent = self.agent(model)
        agent.start(d["id"])
        for _ in range(U.TURNS):
            agent.reply(d["id"], "I'm not sure")
        offer = self.questions(d["id"])[-1]["question"]
        self.assertTrue(offer.get("offer"))
        self.assertEqual(offer["options"][0], "Numbers from my data")  # the likeliest kind of help first, then the rest
        self.assertEqual(len([q for q in self.store.get(d["id"])["questions"] if q.get("understand")]), U.TURNS)

    def test_show_me_options_skips_ahead_and_labels_the_turn(self):
        model = Script(turn("What would you like to decide?", tracks=["analysis"]),
                       turn("Here is what I can do.", options=[COMPARE, CHURN], tracks=["analysis"]))
        d = self.store.create(FIRST)
        agent = self.agent(model)
        agent.start(d["id"])
        agent.reply(d["id"], U.SHOW_OPTIONS)
        self.assertEqual(self.questions(d["id"])[-1]["question"]["options"][:2], [COMPARE["title"], CHURN["title"]])
        self.assertEqual([lb.value for lb in self.book.labels(point="request.understand_turn")], ["offer"])

    def test_a_failed_call_falls_back_to_the_open_question_and_its_buttons(self):
        d = self.store.create(FIRST)
        self.agent(Script(RuntimeError("down"))).start(d["id"])
        q = self.questions(d["id"])[-1]["question"]
        self.assertEqual(len(q["options"]), 6)
        self.assertEqual(self.store.get(d["id"])["understand"]["failed"], "down")

    def test_a_sentence_with_a_number_nobody_gave_is_dropped_and_a_card_needs_a_quote(self):
        model = Script(turn("Most firms see 30% churn. What do you want to find out about them?",
                            goal={"value": "cut churn by 30%", "quote": "cut churn by 30%"}, tracks=["analysis"]))
        d = self.store.create(FIRST)
        self.agent(model).start(d["id"])
        self.assertEqual(self.questions(d["id"])[-1]["text"], "What do you want to find out about them?")
        self.assertNotIn("goal", self.store.get(d["id"])["understand"]["card"])  # not the person's words


class ReviewTests(ConversationTests):  # the review of C2-C4 (2026-10-29)
    def test_a_file_attached_with_the_first_words_still_gets_a_question_first(self):
        model = Script(turn("Sure.", enough=True, options=[COMPARE, CHURN], goal={"value": "analyze my customer", "quote": "analyze my customer"},
                            tracks=["analysis"]))
        d = self.store.create(FIRST)
        self.store.update(d["id"], lambda x: x["assets"].append({"id": "a1", "status": "queued", "name": "telco.csv"}))
        self.agent(model).start(d["id"])
        q = self.questions(d["id"])[-1]
        self.assertTrue(q["question"].get("understand"))  # never options on the first turn
        self.assertNotEqual(q["text"], "Sure.")  # a lead-in to options is never asked as a question
        self.assertTrue(q["text"].endswith("?"))

    def test_the_tool_decides_the_route_of_an_option(self):
        kept, _ = U.validate([{"title": "Find groups of customers who behave alike", "track": "analysis", "tool": "segmentation_project"}], {})
        self.assertEqual((kept[0]["track"], kept[0]["tool"]), ("prediction", "segmentation_project"))

    def test_free_words_after_an_offer_reach_the_model_with_what_it_offered(self):
        model = Script(turn("What do you want to decide?", tracks=["analysis"]), turn("What data do you have?", tracks=["analysis"]),
                       turn("Here is how I can help.", enough=True, options=[COMPARE, CHURN], tracks=["analysis"],
                            goal={"value": "why customers leave us", "quote": "why customers leave us"},
                            data_said={"value": "a table of our customers", "quote": "a table of our customers"}),
                       turn("Comparing churn by contract, then.", tracks=["analysis"]))
        d = self.store.create(FIRST)
        agent = self.agent(model)
        agent.start(d["id"])
        agent.reply(d["id"], "I want to know why customers leave us")
        agent.reply(d["id"], "I have a table of our customers")
        agent.reply(d["id"], "the first one please")
        last = model.seen[-1]
        self.assertIn(COMPARE["title"], " ".join(m["content"] for m in last if m["role"] == "assistant"))
        offers = [r for r in self.book.records(point="request.options")]
        self.assertIn("analysis:query", offers[0].options)  # a pick is one of the options, as its label is

    def test_a_repeated_question_is_never_asked_twice(self):
        model = Script(*[turn("Most firms see 30% churn.", tracks=["analysis"]) for _ in range(3)])
        d = self.store.create(FIRST)
        agent = self.agent(model)
        agent.start(d["id"])
        agent.reply(d["id"], "hmm")
        agent.reply(d["id"], "not sure")
        texts = [q["text"] for q in self.store.get(d["id"])["questions"] if q.get("understand")]
        self.assertEqual(len(texts), len(set(texts)))

    def test_the_model_is_sent_no_cell_value(self):
        frame = pd.DataFrame({"customerID": [f"QX{i}ZZ" for i in range(40)], "Contract": ["Monthy-Q7K", "Yearly-Q7K"] * 20,
                              "Churn": ["Yes", "No"] * 20})
        model = Script(turn("What would you like to find out?", tracks=["analysis"]))
        d = self.store.create(FIRST)
        agent = self.agent(model)
        path = self.store.data_dir(d["id"]) / "t.csv"
        frame.to_csv(path, index=False)
        asset = pipeline.new_asset(self.store, d["id"], "upload", path.name, path.name)
        pipeline.run(self.store, d["id"], asset["id"], HomeAgent(self.store, None, on_request=lambda *a: None), None)
        agent.start(d["id"])
        sent = json.dumps(model.seen)
        self.assertIn("Contract", sent)
        self.assertNotIn("Q7K", sent)
        self.assertNotIn("QX1ZZ", sent)

    def test_after_two_offers_home_never_starts_with_a_route_turned_down(self):
        offer = turn("Here is how I can help.", enough=True, options=[COMPARE, CHURN], tracks=["analysis", "prediction", "reporting"],
                     goal={"value": "know more about our customers", "quote": "know more about our customers"},
                     data_said={"value": "a table of our customers", "quote": "a table of our customers"})
        model = Script(turn("What do you want?", tracks=["analysis"]), turn("What data?", tracks=["analysis"]), offer, offer, offer)
        d = self.store.create(FIRST)
        agent = self.agent(model)
        agent.start(d["id"])
        agent.reply(d["id"], "I want to know more about our customers")
        agent.reply(d["id"], "I have a table of our customers")
        agent.reply(d["id"], U.SOMETHING_ELSE)
        agent.reply(d["id"], U.SOMETHING_ELSE)
        self.assertEqual(HomeAgent.track(self.store.get(d["id"])), "reporting")  # the likeliest one not passed over

    def test_a_reply_that_is_not_json_falls_back_to_the_buttons(self):
        d = self.store.create(FIRST)
        self.agent(Script("I think you want analysis!")).start(d["id"])
        self.assertEqual(len(self.questions(d["id"])[-1]["question"]["options"]), 6)


class OptionTests(unittest.TestCase):
    def draft_with_table(self, frame):
        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, True)
        store = DraftStore(home / "drafts")
        d = store.create(FIRST)
        agent = HomeAgent(store, None, on_request=lambda *a: None)
        path = store.data_dir(d["id"]) / "customers.csv"
        frame.to_csv(path, index=False)
        asset = pipeline.new_asset(store, d["id"], "upload", path.name, path.name)
        pipeline.run(store, d["id"], asset["id"], agent, None)
        return store.get(d["id"])

    def test_options_name_only_real_columns_and_never_what_the_table_cannot_support(self):
        frame = pd.DataFrame({"customerID": [f"C{i}" for i in range(60)], "tenure": list(range(60)),
                              "Contract": ["a", "b", "c"] * 20, "Churn": ["Yes", "No"] * 30})
        draft = self.draft_with_table(frame)
        options = [{"title": "Compare churn by contract", "track": "analysis", "tool": "query", "columns": ["Churn", "Contract"]},
                   {"title": "Average discount per customer", "track": "analysis", "tool": "query", "columns": ["discount"]},
                   {"title": "Why did churn change this month", "track": "business", "tool": "business_brief", "columns": []},
                   {"title": "Predict churn for the top 500", "track": "prediction", "tool": "ml_project", "columns": ["Churn"]},
                   {"title": "Predict who will leave", "track": "prediction", "tool": "ml_project", "columns": ["Churn"]}]
        kept, refused = U.validate(options, draft)
        self.assertEqual([o["title"] for o in kept], ["Compare churn by contract", "Predict who will leave"])
        self.assertEqual(len(refused), 3)  # a missing column, periods with no dates, a number nobody gave
        no_outcome = self.draft_with_table(frame.drop(columns=["Churn"]))
        kept, _ = U.validate([{"title": "Predict who will leave", "track": "prediction", "tool": "ml_project"},
                              {"title": "Find groups of customers", "track": "prediction", "tool": "segmentation_project"}], no_outcome)
        self.assertEqual([o["title"] for o in kept], ["Find groups of customers"])  # groups need no outcome; a prediction does

    def test_the_model_sees_columns_and_what_was_noticed_never_a_value(self):
        frame = pd.DataFrame({"customerID": [f"C{i}" for i in range(60)], "SeniorCitizen": [0, 1] * 30, "Churn": ["Yes", "No"] * 30})
        draft = self.draft_with_table(frame)
        view = {c["name"]: c for c in U.columns_view(draft)}
        self.assertIn("identifier", view["customerID"]["noticed"])
        self.assertIn("small code or flag", view["SeniorCitizen"]["noticed"])
        self.assertIn("outcome-like name", view["Churn"]["noticed"])
        self.assertNotIn("C1", json.dumps(view))


class RunnerTests(unittest.TestCase):
    """The conversation's measurement plays a written person from their hidden facts (agent_eval/home/conversation.py)."""

    def test_the_person_answers_from_their_facts_and_picks_their_route(self):
        from dclab_rnd.agent_eval.home import conversation as C

        people = [{"id": "t-1", "opening": "can you help with our customers?", "upload": None, "route": "prediction", "tool": "segmentation_project",
                   "facts": {"situation": "We run a gym chain.", "goal": "I want to find types of members to target offers",
                             "data": "a members table with visits", "decision": "send each group its own offer"}},
                  {"id": "t-2", "opening": "my data is a mess", "upload": None, "route": "preparation", "tool": "recipe",
                   "facts": {"goal": "I want one clean table", "data": "two exports from our shop"}}]
        replies = {"t-1": [turn("What are you hoping to achieve with your customers?", tracks=["analysis"]),
                           turn("Great. What data do you have?", tracks=["prediction"]),
                           turn("Here is how I can help.", enough=True, options=[SEGMENTS, CHURN], tracks=["prediction"],
                                goal={"value": "find types of members", "quote": "find types of members"},
                                data_said={"value": "a members table with visits", "quote": "a members table with visits"})],
                   "t-2": [turn("What would you like to end up with?", tracks=["preparation"]),
                           turn("One clean table makes sense. What data do you have?", tracks=["preparation"]),
                           turn("Here is how I can help.", enough=True, tracks=["preparation"],
                                goal={"value": "one clean table", "quote": "one clean table"},
                                data_said={"value": "two exports from our shop", "quote": "two exports from our shop"},
                                options=[{"title": "Clean and combine your exports", "track": "preparation", "tool": "recipe"},
                                         {"title": "Build a sales dashboard", "track": "reporting", "tool": "dashboard"}])]}
        unclear = lambda pid: Script(json.dumps({"tracks": []}))  # noqa: E731
        result = C.measure(people, lambda pid: Script(*replies[pid]), unclear)
        rows = {r["id"]: r for r in result["rows"]}
        self.assertTrue(rows["t-1"]["right"] and rows["t-1"]["family"] == "segmentation")
        self.assertEqual(rows["t-1"]["turns_before_options"], 2)
        self.assertTrue(rows["t-2"]["right"])
        self.assertEqual(result["summary"]["silent_turns"], 0)
        self.assertEqual(result["summary"]["route_right"], 2)
        self.assertEqual(C.answer_for("What data do you have about them?", {"data": "a table", "goal": "x"}, set()), "a table")


if __name__ == "__main__":
    unittest.main()
