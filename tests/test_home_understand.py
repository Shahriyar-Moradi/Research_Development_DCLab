"""2026-10-29 (the owner: "understand before routing; AI first, rules as the fallback"): Home's model reads the request
first, the rules check it, an unclear request is asked with an open question, a failed reading is said, and a prediction
is never assumed. The two sentences the owner tested are here; the others are written for these tests.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import ledger  # noqa: E402
from dclab_rnd.draft import intent  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402

VAGUE = "I have data analysis problems"  # the owner's first test: shown as a prediction, asked what the model should predict
SEGMENTS = "I want analyze my customer and segment of the my customer"  # the owner's second


class Scripted:
    def __init__(self, *replies):
        self.replies, self.seen, self.outputs = list(replies), [], []

    def complete(self, messages, tools=None):
        self.seen.append(messages)
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        return {"content": reply}

    def output(self, ok, note=""):
        self.outputs.append(ok)


def reading(**fields):
    return json.dumps(fields)


class ReadTests(unittest.TestCase):
    def test_nothing_read_is_asked_never_a_prediction(self):
        r = intent.read(VAGUE)
        self.assertEqual((r["status"], r["ask"], r["track"]), ("asked", "open", None))
        self.assertEqual(r["question"], intent.OPEN_QUESTION)

    def test_a_failed_model_call_is_said_and_never_becomes_a_prediction(self):
        r = intent.read(VAGUE, Scripted(RuntimeError("refused: the workspace has used its monthly budget")))
        self.assertIn("monthly budget", r["model_failed"])
        self.assertEqual((r["status"], r["ask"]), ("asked", "open"))
        r = intent.read("Predict which subscribers will cancel next quarter", Scripted(RuntimeError("down")))
        self.assertEqual((r["track"], r["source"], r["status"]), ("prediction", "rules", "stated"))  # the rules read alone
        self.assertEqual(r["model_failed"], "down")

    def test_the_model_reads_first_even_when_a_rule_matches(self):
        client = Scripted(reading(tracks=["analysis"], quote="segment of the my customer", confidence=0.9,
                                  need="find groups among their customers"))
        r = intent.read(SEGMENTS, client)
        self.assertEqual(len(client.seen), 1)  # asked although "analyze my" matches a rule
        self.assertEqual((r["track"], r["source"], r["status"], r["need"]), ("analysis", "model", "inferred", "find groups among their customers"))
        self.assertEqual(r["rule_track"], "analysis")  # the rules' own answer is kept beside the model's

    def test_a_report_that_names_nothing_it_shows_stays_open(self):
        """HOMECONV-003 p-03 and the sealed s-04 (2026-10-29): "I need some kind of weekly report on our subscribers" was read
        as a clear report and routed at once; the person wanted a Monday list of who will cancel (a prediction)."""
        vague = "I need some kind of weekly report on our subscribers"
        r = intent.read(vague)
        self.assertEqual((r["status"], r["ask"]), ("asked", "open"))
        self.assertIn("report_content_unsaid", r["signals"])
        self.assertEqual(r["tracks"][0], "reporting")  # still the first guess among the options
        client = Scripted(reading(tracks=["reporting"], quote="weekly report on our subscribers", confidence=0.9, need="a weekly report"))
        self.assertEqual(intent.read(vague, client)["status"], "asked")  # the model's confident reading is checked too
        for clear in ("a weekly report of churn by region", "monthly dashboard of sales and refunds per store",
                      "a weekly report with the number of new subscribers and the cancellation rate",
                      # the review of 2026-10-30: content in other words, a column name, a report that exists, Persian
                      "Want a dashboard refreshing daily with delivery times.", "Dashboard tracking enrollment and attendance numbers",
                      "Monthly report of active users per country", "A monthly report of complaints by branch",
                      "Weekly report: customers with more than 5 support_tickets_90",
                      "Same Excel report every week, rebuilding it by hand is killing me. Can it just update itself?",
                      "یک داشبورد روزانه از فروش هر کانال برای مدیر بازاریابی می‌خواهم", "گزارش هفتگی فروش"):
            self.assertIn(intent.read(clear)["status"], ("stated", "inferred"), clear)
        self.assertEqual(intent.read("Give me a report of which customers will leave next month")["track"], "prediction")

    def test_the_report_check_opens_no_clear_report_in_the_written_sets(self):
        """The review of 2026-10-30: the first version asked 30 of the 77 reporting requests the rules read right."""
        home = ROOT / "dclab_rnd" / "agent_eval" / "home"
        opened = []
        for name in ("intent_requests_v1.json", "requests_dev.json"):
            for case in json.loads((home / name).read_text(encoding="utf-8")):
                text = case.get("text")
                gold = case.get("track") or ((case.get("gold") or {}).get("tracks") or [None])[0]
                before = intent._read(text) if text else {}
                if gold == "reporting" and before.get("track") == "reporting" and before.get("status") in ("stated", "inferred") \
                        and intent.read(text)["status"] == "asked":
                    opened.append(text[:80])
        self.assertEqual(opened, [])

    def test_an_unclear_request_gets_the_models_own_question(self):
        q = "What would you like to find out from your data, and what decision would it help?"
        r = intent.read(VAGUE, Scripted(reading(tracks=[], best_question=q, unknowns=["the goal"])))
        self.assertEqual((r["status"], r["ask"], r["question"], r["source"]), ("asked", "open", q, "model"))

    def test_a_low_confidence_reading_is_asked_with_its_tracks_first(self):
        r = intent.read(VAGUE, Scripted(reading(tracks=["analysis", "business"], quote="data analysis problems", confidence=0.4)))
        self.assertEqual((r["status"], r["ask"], r["tracks"][:2]), ("asked", "open", ["analysis", "business"]))

    def test_the_rules_check_the_model_and_a_clash_is_asked(self):
        client = Scripted(reading(tracks=["advice"], quote="which subscribers will cancel", confidence=0.9))
        r = intent.read("Predict which subscribers will cancel next quarter", client)
        self.assertEqual((r["status"], r["ask"], r["tracks"]), ("asked", "either", ["advice", "prediction"]))

    def test_a_question_with_a_number_nobody_gave_is_not_used(self):
        r = intent.read(VAGUE, Scripted(reading(tracks=[], best_question="Do you have about 1000 rows?", need="analyse 30 tables")))
        self.assertEqual(r["question"], intent.OPEN_QUESTION)  # the model's question named a number the person never wrote
        self.assertNotIn("need", r)

    def test_which_alone_is_not_a_prediction_with_no_model(self):  # live 2026-10-29: "which products sell best" is the past
        self.assertNotEqual(intent.read("I want to see which products and regions sell best each month")["track"], "prediction")
        self.assertEqual(intent.read("Find the customers who will cancel")["track"], "prediction")

    def test_a_quote_the_person_did_not_write_is_a_failed_reading(self):
        r = intent.read(VAGUE, Scripted(reading(tracks=["prediction"], quote="predict churn", confidence=0.9)))
        self.assertEqual(r["model_failed"], "the reading's quote is not the person's words")
        self.assertEqual(r["status"], "asked")


class HomeTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.store = DraftStore(self.home / "drafts")
        ledger.install(ledger.FileLedger(self.home / "ledger", "wtest"))
        self.addCleanup(ledger.install, None)

    def agent(self, intent_client=None):
        return HomeAgent(self.store, None, intent_client=intent_client, on_request=lambda *a: None)

    def said(self, draft_id):
        return [m for m in self.store.get(draft_id)["messages"] if m.get("role") != "user"]

    def test_a_vague_request_gets_an_open_question_not_the_ml_path(self):
        d = self.store.create(VAGUE)
        self.agent().start(d["id"])
        draft = self.store.get(d["id"])
        text = " ".join(m.get("text") or "" for m in self.said(d["id"]))
        self.assertIn("Tell me a bit more", text)
        self.assertNotIn("What exactly should the model predict", text)
        self.assertIsNone(draft.get("pack"))
        self.assertFalse(draft.get("workflow"))
        question = next(m for m in self.said(d["id"]) if m.get("kind") == "question")
        self.assertEqual(len(question["question"].get("options") or []), len(intent.TRACKS))  # every kind of help as a button
        self.assertEqual(HomeAgent.track(draft), "undecided")

    def test_an_answer_in_words_is_read_again_and_routes(self):
        d = self.store.create(VAGUE)
        agent = self.agent()
        agent.start(d["id"])
        agent.reply(d["id"], "I need a weekly dashboard of sales per region for my manager")
        draft = self.store.get(d["id"])
        self.assertEqual(HomeAgent.track(draft), "reporting")

    def test_with_a_model_the_answer_is_read_by_the_model_with_everything_said(self):
        client = Scripted(reading(tracks=[], best_question="What would you like to end up with?"),
                          reading(tracks=["analysis"], quote="groups of customers", confidence=0.9, need="They want to see which groups of customers they have"))
        d = self.store.create(VAGUE)
        agent = self.agent(client)
        agent.start(d["id"])
        self.assertIn("What would you like to end up with?", " ".join(m.get("text") or "" for m in self.said(d["id"])))
        agent.reply(d["id"], "I want to see groups of customers")
        self.assertIn("I want to see groups of customers", client.seen[1][1]["content"])  # the second reading sees the answer
        draft = self.store.get(d["id"])
        self.assertEqual(HomeAgent.track(draft), "analysis")
        self.assertIn("I understood: you want to see which groups of customers they have", " ".join(m.get("text") or "" for m in self.said(d["id"])))

    def test_after_two_unclear_answers_the_buttons_decide(self):
        d = self.store.create(VAGUE)
        agent = self.agent()
        agent.start(d["id"])
        agent.reply(d["id"], "hmm not sure")
        agent.reply(d["id"], "still thinking")
        last = [m for m in self.said(d["id"]) if m.get("kind") == "question"][-1]
        self.assertEqual(last["text"].split("\n")[0], "Which of these is closest to what you need?")
        agent.reply(d["id"], intent.LABELS["preparation"])
        self.assertEqual(HomeAgent.track(self.store.get(d["id"])), "preparation")

    def test_segments_get_their_two_ways_and_new_groups_need_no_outcome(self):  # the owner's second test
        d = self.store.create(SEGMENTS)
        agent = self.agent()
        agent.start(d["id"])
        question = [m for m in self.said(d["id"]) if m.get("kind") == "question"][-1]
        self.assertEqual(question["question"]["field"], "segments_route")
        self.assertNotIn("What should I measure", " ".join(m.get("text") or "" for m in self.said(d["id"])))
        agent.reply(d["id"], "Find new groups in the data")
        draft = self.store.get(d["id"])
        self.assertEqual((HomeAgent.track(draft), draft["understanding"]["family"]), ("prediction", "segmentation"))
        text = " ".join(m.get("text") or "" for m in self.said(d["id"]))
        self.assertIn("no outcome to predict", text)
        self.assertNotIn("What exactly should the model predict", text)

    def test_groups_the_person_has_stay_an_analysis(self):
        d = self.store.create("Segment our customers by plan")
        agent = self.agent()
        agent.start(d["id"])
        agent.reply(d["id"], "Describe groups I already have")
        self.assertEqual(HomeAgent.track(self.store.get(d["id"])), "analysis")

    def test_persian_segments_are_read(self):
        r = intent.read("مشتری‌هایم را خوشه‌بندی کن")
        self.assertEqual((r["track"], r["status"]), ("analysis", "stated"))
        self.assertIn("fa_segments", r["signals"])


class ReviewTests(unittest.TestCase):
    """The review of 2026-10-29: each finding's scenario, held."""

    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.store = DraftStore(self.home / "drafts")
        self.book = ledger.FileLedger(self.home / "ledger", "wtest")
        ledger.install(self.book)
        self.addCleanup(ledger.install, None)

    def agent(self, intent_client=None, client=None):
        return HomeAgent(self.store, client, intent_client=intent_client, on_request=lambda *a: None)

    def text(self, draft_id):
        return " ".join(m.get("text") or "" for m in self.store.get(draft_id)["messages"] if m.get("role") != "user")

    def test_a_bare_future_is_not_a_prediction(self):  # finding 2
        for sentence in ("I'm going to upload my customer table", "What will help me with my data?"):
            self.assertEqual(intent.read(sentence)["status"], "asked", sentence)
        self.assertEqual(intent.read("I want to know if each delivery will be late")["track"], "prediction")

    def test_an_edit_never_lets_the_fallback_replace_a_models_reading(self):  # finding 3
        client = Scripted(reading(tracks=["analysis"], quote="matters most", confidence=0.9))
        d = self.store.create("Tell me what matters most in my sales data")
        agent = self.agent(client)
        agent.start(d["id"])
        self.store.update(d["id"], lambda x: x.update(problem="Tell me what matters most in my fraud data"))  # only the kind-word fallback reads it
        agent.replan(d["id"])
        self.assertEqual(self.store.get(d["id"])["intent"]["track"], "analysis")

    def test_words_after_an_either_question_are_read_by_the_model(self):  # finding 4
        client = Scripted(reading(tracks=["advice"], quote="subscribers will cancel", confidence=0.9),
                          reading(tracks=["reporting"], quote="weekly dashboard", confidence=0.9))
        d = self.store.create("Predict which subscribers will cancel next quarter")
        agent = self.agent(client)
        agent.start(d["id"])
        self.assertEqual(self.store.get(d["id"])["intent"]["ask"], "either")
        agent.reply(d["id"], "Neither, I want a weekly dashboard of cancellations")
        self.assertEqual(HomeAgent.track(self.store.get(d["id"])), "reporting")

    def test_weak_words_are_never_the_persons_choice(self):  # finding 5
        d = self.store.create(VAGUE)
        agent = self.agent()
        agent.start(d["id"])
        agent.reply(d["id"], "not sure, maybe compare a few things")
        draft = self.store.get(d["id"])
        self.assertEqual(HomeAgent.track(draft), "undecided")
        self.assertEqual(self.book.labels(point="request.intent"), [])

    def test_the_rules_have_a_new_version(self):  # finding 6
        self.assertEqual(intent.RULES_VERSION, "intent_rules_v6")  # v6 (2026-11-12): hypothetical "would" alone does not state a prediction task

    def test_an_open_reading_is_recorded_and_the_choice_labels_it(self):  # finding 7
        d = self.store.create(VAGUE)
        agent = self.agent()
        agent.start(d["id"])
        self.assertEqual(len(self.book.records(point="request.intent")), 1)
        agent.reply(d["id"], intent.LABELS["reporting"])
        self.assertEqual([lab.value for lab in self.book.labels(point="request.intent")], ["reporting"])

    def test_segments_from_the_example_chip_the_model_and_plain_words(self):  # finding 8
        self.assertIn("segments", intent.read("I want to find groups of customers who behave alike, to target offers.")["signals"])
        client = Scripted(reading(tracks=["analysis"], quote="our buyers", confidence=0.9, need="find groups among your buyers"))
        d = self.store.create("Help me understand our buyers")
        agent = self.agent(client)
        agent.start(d["id"])
        self.assertIn("two ways to look at groups", self.text(d["id"]))
        agent.reply(d["id"], "I don't need new groups, just describe the plans")
        draft = self.store.get(d["id"])
        self.assertEqual(HomeAgent.track(draft), "analysis")
        self.assertIn(intent.NEXT["analysis"], self.text(d["id"]))  # nothing to ask before the table: what happens next is said

    def test_the_memory_says_undecided_while_home_asks(self):  # finding 9
        from dclab_rnd.draft import memory
        d = self.store.create(VAGUE)
        self.agent().start(d["id"])
        self.assertEqual(memory.summary(self.store.get(d["id"]))["track"], "undecided")

    def test_the_second_open_question_is_not_the_first_again(self):
        d = self.store.create(VAGUE)
        agent = self.agent()
        agent.start(d["id"])
        agent.reply(d["id"], "hmm")
        questions = [m["text"] for m in self.store.get(d["id"])["messages"] if m.get("kind") == "question"]
        self.assertEqual(len(set(questions)), len(questions))

    def test_not_sure_forever_is_not_asked_forever(self):  # HOME-015: seven questions and no route
        d = self.store.create(VAGUE)
        agent = self.agent()
        agent.start(d["id"])
        for _ in range(6):
            if agent.pending(self.store.get(d["id"])) is None:
                break
            agent.reply(d["id"], "I'm not sure")
        asked = [q for q in self.store.get(d["id"])["questions"] if q.get("field") == "intent"]
        self.assertLessEqual(len(asked), HomeAgent.UNDERSTAND_TURNS + 1)
        self.assertIsNone(agent.pending(self.store.get(d["id"])))
        client = Scripted(reading(tracks=["analysis", "reporting"], quote="data analysis problems", confidence=0.4),
                          *[reading(tracks=["analysis"], quote="data analysis problems", confidence=0.4)] * 4)
        d = self.store.create(VAGUE)
        agent = self.agent(client)
        agent.start(d["id"])
        for _ in range(6):
            if agent.pending(self.store.get(d["id"])) is None:
                break
            agent.reply(d["id"], "I'm not sure")
        self.assertEqual(HomeAgent.track(self.store.get(d["id"])), "analysis")  # its best reading, said and changeable

    def test_the_eval_runs_home_with_its_intent_reader(self):  # HOME-015: the model arm had no home_intent client
        import inspect
        from dclab_rnd.agent_eval import home as H
        self.assertIn("intent_for", inspect.signature(H.observe).parameters)
        self.assertIn("intent_for", inspect.signature(H.measure).parameters)

    def test_a_pack_set_on_a_prediction_is_cleared_when_the_track_changes(self):  # P4
        d = self.store.create("Predict which customers will cancel next month")
        agent = self.agent()
        agent.start(d["id"])
        self.assertTrue(self.store.get(d["id"]).get("pack"))
        agent.choose_intent(d["id"], "analysis")
        self.assertIsNone(self.store.get(d["id"]).get("pack"))


class RoutingRecordTests(unittest.TestCase):
    """P8: each intent reading is one routing record with every answerer, the model's joined to its exchange (and so to
    its thinking summary), and the person's choice labels it."""

    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.store = DraftStore(self.home / "drafts")
        self.book = ledger.FileLedger(self.home / "ledger", "wtest")
        ledger.install(self.book)
        self.addCleanup(ledger.install, None)

    def test_the_models_reading_carries_its_exchange_and_the_rules_answer_beside_it(self):
        client = Scripted(reading(tracks=["analysis"], quote="segment of the my customer", confidence=0.9))
        client.last_request = "mr0123456789abcdef"
        d = self.store.create(SEGMENTS)
        HomeAgent(self.store, None, intent_client=client, on_request=lambda *a: None).start(d["id"])
        rec = self.book.records(point="request.intent")[0]
        answers = {a.by: a for a in rec.answers}
        self.assertEqual((answers["model_role"].value, answers["model_role"].source), ("analysis", "mr0123456789abcdef"))
        self.assertEqual(answers["rule"].value, "analysis")
        self.assertEqual(self.store.get(d["id"])["intent"]["model_request"], "mr0123456789abcdef")

    def test_a_failed_model_call_is_in_the_record_with_its_reason(self):
        d = self.store.create(VAGUE)
        HomeAgent(self.store, None, intent_client=Scripted(RuntimeError("refused: the workspace has used its monthly budget")),
                  on_request=lambda *a: None).start(d["id"])
        rec = self.book.records(point="request.intent")[0]
        model = next(a for a in rec.answers if a.by == "model_role")
        self.assertIsNone(model.value)
        self.assertIn("monthly budget", model.reason)


class NumbersTests(unittest.TestCase):
    """P5: every track's model reply keeps only numbers someone gave (HOME-013's three uncited numbers, now spent dev
    material: two prediction requests and one advice question read as business)."""

    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.store = DraftStore(self.home / "drafts")
        ledger.install(ledger.FileLedger(self.home / "ledger", "wtest"))
        self.addCleanup(ledger.install, None)

    def checked(self, problem, reply, track):
        d = self.store.create(problem)
        self.store.update(d["id"], lambda x: x.update(intent=intent.chosen(track)))
        return HomeAgent(self.store, None).checked_reply(self.store.get(d["id"]), reply)

    def test_a_predictions_invented_example_is_dropped_and_the_rest_kept(self):
        problem = ("need to forcast how much each customer will spend next month, based on whatever we know at month end. "
                   "finance wants it for budgeting. we have like 2 yrs of history across customers and sales")
        out = self.checked(problem, "Got it, a monthly spend forecast per customer. Say you have 1000 customers. When is it made?", "prediction")
        self.assertNotIn("1000", out)
        self.assertIn("monthly spend forecast per customer", out)
        self.assertIn("2 yrs", self.checked(problem, "With 2 yrs of history we can use last year's month as a baseline.", "prediction"))

    def test_homes_own_words_hold_no_number(self):  # HOME-013 s2-v4: "two periods" in the business next step read as 2
        from dclab_rnd import cited
        for track, text in intent.NEXT.items():
            self.assertEqual(cited.numbers(text), set(), track)
        self.assertEqual(cited.numbers(intent.OPEN_QUESTION) | cited.numbers(intent.OPEN_AGAIN), set())

    def test_a_number_from_nowhere_on_another_track_is_replaced(self):
        out = self.checked("how do i set up a time based train/test split for a churn model so i dont leak future info into training",
                           "Hold out the last 30 days.", "advice")
        self.assertNotIn("30", out)


class FieldsTests(unittest.TestCase):
    """HOME-017 (sealed v4): Home read the fields of each kind 17 right against Claude alone's 64, because only the rules read
    them. The model reads them in the same call as the kind of help; the rules fill only what it left."""

    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.store = DraftStore(self.home / "drafts")
        ledger.install(ledger.FileLedger(self.home / "ledger", "wtest"))
        self.addCleanup(ledger.install, None)

    def test_only_quoted_fields_of_the_kind_are_kept(self):
        text = "Can you show monthly revenue for each sales region over the last quarter?"
        r = intent.read(text, Scripted(reading(tracks=["analysis"], quote="monthly revenue", confidence=0.9, fields={"analysis": {
            "measure": {"value": "monthly revenue", "quote": "monthly revenue"},
            "breakdown": {"value": "sales region", "quote": "each sales region"},
            "period": {"value": "last 90 days", "quote": "the last 90 days"},          # not the person's words
            "audience": {"value": "the board", "quote": "monthly revenue"},            # not a field of an analysis
            "filters": {"value": "top 10 regions", "quote": "each sales region"}}})))  # a number nobody wrote
        self.assertEqual(set(r["fields"]["analysis"]), {"measure", "breakdown"})
        self.assertEqual(r["fields"]["analysis"]["measure"]["source"], "model")

    def test_a_reworded_value_is_kept_in_the_persons_words(self):  # HOME-018: "هر هفته" came back as "every week"
        text = "I want a KPI dashboard of bookings that is refreshed every Monday morning for the ops team"
        r = intent.read(text, Scripted(reading(tracks=["reporting"], quote="KPI dashboard", confidence=0.9, fields={"reporting": {
            "refresh": {"value": "weekly", "quote": "refreshed every Monday morning"},
            "audience": {"value": "ops team", "quote": "for the ops team"}}})))
        self.assertEqual(r["fields"]["reporting"]["refresh"]["value"], "refreshed every Monday morning")
        self.assertEqual(r["fields"]["reporting"]["audience"]["value"], "ops team")  # already the person's words: kept

    def test_the_models_fields_come_first_and_the_rules_fill_the_rest(self):
        text = "Can you show monthly revenue for each sales region over the last quarter?"
        client = Scripted(reading(tracks=["analysis"], quote="monthly revenue", confidence=0.9,
                                  fields={"analysis": {"measure": {"value": "monthly revenue", "quote": "monthly revenue"}}}))
        d = self.store.create(text)
        HomeAgent(self.store, None, intent_client=client, on_request=lambda *a: None).start(d["id"])
        brief = self.store.get(d["id"])["track_brief"]["analysis"]
        self.assertEqual((brief["measure"]["value"], brief["measure"]["source"]), ("monthly revenue", "model"))
        self.assertIn("rule_value", brief["measure"])  # the rules' own reading kept beside it, for the routing data
        self.assertNotEqual(brief["period"].get("source"), "model")  # what the model left, the rules or the default fill

    def test_a_field_naming_a_column_the_table_lacks_is_refused(self):
        text = "Show the average discount_pct per channel this year"
        client = Scripted(reading(tracks=["analysis"], quote="average discount_pct", confidence=0.9,
                                  fields={"analysis": {"measure": {"value": "average discount_pct", "quote": "average discount_pct"}}}))
        d = self.store.create(text)
        self.store.update(d["id"], lambda x: x.update(analysis={"columns": [{"name": "amount", "kind": "numeric", "unique": 90},
                                                                           {"name": "channel", "kind": "categorical", "unique": 3}]}))
        HomeAgent(self.store, None, intent_client=client, on_request=lambda *a: None).start(d["id"])
        draft = self.store.get(d["id"])
        self.assertNotEqual((draft["track_brief"]["analysis"].get("measure") or {}).get("source"), "model")
        self.assertIn("no column named `discount_pct`", " ".join(m.get("text") or "" for m in draft["messages"]))

    def agent_with(self, text, fields):
        client = Scripted(reading(tracks=["analysis"], quote=text.split()[1] + " " + text.split()[2], confidence=0.9, fields={"analysis": fields}))
        d = self.store.create(text)
        return HomeAgent(self.store, None, intent_client=client, on_request=lambda *a: None), d

    def table(self, draft_id, *columns):
        self.store.update(draft_id, lambda x: x.update(analysis={"columns": [{"name": c, "kind": "numeric", "unique": 50} for c in columns]}))

    def test_a_model_field_on_a_missing_column_is_withdrawn_after_the_intent_changed(self):  # review finding 1
        text = "Show the average discount_pct per channel this year"
        agent, d = self.agent_with(text, {"measure": {"value": "average discount_pct", "quote": "average discount_pct"}})
        agent.start(d["id"])
        agent.choose_intent(d["id"], "reporting")
        agent.choose_intent(d["id"], "analysis")
        self.table(d["id"], "amount", "channel")
        agent.open_brief(d["id"])
        self.assertNotEqual((self.store.get(d["id"])["track_brief"]["analysis"].get("measure") or {}).get("value"), "average discount_pct")

    def test_not_sure_is_never_overwritten_by_the_model(self):  # review finding 2
        text = "Show the average discount_pct per channel this year"
        agent, d = self.agent_with(text, {"measure": {"value": "average discount_pct", "quote": "average discount_pct"}})
        self.table(d["id"], "amount", "channel")
        agent.start(d["id"])
        q = agent.pending(self.store.get(d["id"]))
        if q is not None and q.get("field") == "measure":
            agent.reply(d["id"], "Not sure")
        self.store.update(d["id"], lambda x: x["track_brief"]["analysis"].update(measure={"value": None, "source": None, "asked": True}))
        self.table(d["id"], "amount", "channel", "discount_pct")
        agent.open_brief(d["id"])
        self.assertNotEqual(self.store.get(d["id"])["track_brief"]["analysis"]["measure"].get("source"), "model")  # never the model over an answer

    def test_the_models_fields_reach_the_project_memory(self):  # review finding 3
        from dclab_rnd.draft import memory
        text = "Show the total revenue per region for last month"
        agent, d = self.agent_with(text, {"measure": {"value": "total revenue", "quote": "total revenue"},
                                          "breakdown": {"value": "region", "quote": "per region"}})
        agent.start(d["id"])
        facts = memory.facts(self.store.get(d["id"]))
        self.assertTrue(any("total revenue" in f for f in facts), facts)

    def test_a_predictions_reworded_value_says_the_persons_words(self):  # HOME-021: the plan model's own sentences
        text = "I want to predict how much each customer will spend, when customer put it in cart, so warehouse can prepare stock."

        class Planner:
            def complete(self, messages, tools=None, max_tokens=1800):
                return {"content": json.dumps({"prediction_moment": {"status": "stated", "value": "At the moment the customer puts the order in the cart.",
                                                                     "quote": "when customer put it in cart"}})}
        d = self.store.create(text)
        self.store.update(d["id"], lambda x: x.update(intent=intent.chosen("prediction")))
        HomeAgent(self.store, Planner(), on_request=lambda *a: None).start(d["id"])
        self.assertEqual(self.store.get(d["id"])["understanding"]["prediction_moment"], "when customer put it in cart")

    def test_a_predictions_value_with_an_invented_number_says_the_persons_words(self):  # HOME-017's "10%"
        text = "Score everyone at month start; the top decile goes to the save-desk, and we predict who will churn."

        class Planner:
            def complete(self, messages, tools=None, max_tokens=1800):
                return {"content": json.dumps({"action": {"status": "stated", "value": "top 10% go to the save-desk",
                                                          "quote": "the top decile goes to the save-desk"}})}
        d = self.store.create(text)
        self.store.update(d["id"], lambda x: x.update(intent=intent.chosen("prediction")))
        HomeAgent(self.store, Planner(), on_request=lambda *a: None).start(d["id"])
        u = self.store.get(d["id"])["understanding"]
        self.assertEqual(u["action"], "the top decile goes to the save-desk")
        self.assertNotIn("10%", " ".join(m.get("text") or "" for m in self.store.get(d["id"])["messages"]))


if __name__ == "__main__":
    unittest.main()
