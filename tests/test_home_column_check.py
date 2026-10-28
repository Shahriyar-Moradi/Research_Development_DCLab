"""H3.1: a column kept out on statistics alone is asked about before a build (live, HyperAck, 2026-10-28).

The owner's HyperAck table went through DCLab Code's chat with a person who answered "Use your recommendation" to every
question: `deliverey_category_id` was kept out only because it alone separated the classes (AUC 0.816), DCLab's leakage
reviewer read it as chosen when the order is placed, Home never asked, and the holdout was 0.664 against the research's
honest 0.945. The owner's rule: Home asks, with the evidence; "use your recommendation" keeps the column only when the
reviewer says "available" and names the moment.
"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_rnd.draft import column_check as checks  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent, counted  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402

REVIEWER = ("The delivery category is chosen when the order is placed, so it is known 'At order creation, before dispatch'; "
            "a high single-column AUC alone does not prove leakage and calls for semantic review only.")


def hyperack_proposal(reviewer_says=REVIEWER, verdict="available"):
    """The proposal of the live run, as the build read it (draft d033dcbcab6a)."""
    return {
        "prediction_moment": "At order creation, before dispatch.",
        "forbidden": [
            {"column": "final_biker_fare", "reason": "name suggests a post-outcome value; known DCLab leakage precedent LEAK-hyperack"},
            {"column": "final_customer_fare", "reason": "name suggests a post-outcome value; known DCLab leakage precedent LEAK-hyperack"},
            {"column": "deliverey_category_id", "reason": "single column separates the classes (AUC 0.816)"},
        ],
        "review": {"verdicts": [{"column": "final_customer_fare", "verdict": "after the moment"},
                                {"column": "final_biker_fare", "verdict": "after the moment"},
                                {"column": "deliverey_category_id", "verdict": verdict}],
                   "disagreements": [{"column": "deliverey_category_id", "reason": reviewer_says}] if verdict == "available" else []},
    }


class ColumnCheckRules(unittest.TestCase):
    def test_only_the_statistics_only_column_is_asked_never_a_named_leak(self):
        items = checks.disputed(hyperack_proposal(), time_column="created_date")
        self.assertEqual([i["column"] for i in items], ["deliverey_category_id"])
        item = items[0]
        self.assertTrue(item["statistics_only"] and item["names_moment"])
        self.assertEqual(item["reviewer"], "available")

    def test_the_question_shows_the_evidence_and_the_reviewer_and_the_recommendation(self):
        item = checks.disputed(hyperack_proposal())[0]
        q = checks.question(item, "At order creation, before dispatch.")
        self.assertIn("AUC 0.816", q)
        self.assertIn("chosen when the order is placed", q)
        self.assertIn("At order creation, before dispatch", q)
        self.assertIn("I recommend to keep it", q)

    def test_use_your_recommendation_follows_the_reviewer_only_when_it_names_the_moment(self):
        item = checks.disputed(hyperack_proposal())[0]
        self.assertEqual(checks.read_answer("Use your recommendation."), "recommendation")
        self.assertEqual(checks.decide(item, "recommendation")[0], True)
        # a reason that quotes the moment (as every reviewer reason must) but hedges, or says it changes later, names no moment
        for hedged in ("Probably set at order creation, before dispatch, though it could be updated after delivery.",
                       "It may be known 'At order creation, before dispatch' depending on the region.",
                       "Set at order creation, before dispatch, and changed after the courier answers."):
            vague = checks.disputed(hyperack_proposal(reviewer_says=hedged))[0]
            self.assertFalse(vague["names_moment"], hedged)
            self.assertEqual(checks.decide(vague, "recommendation")[0], False, hedged)
        silent = checks.disputed(hyperack_proposal(verdict="after the moment"))[0]
        self.assertEqual(checks.decide(silent, "recommendation")[0], False, "the reviewer agrees it is late: it stays out")

    def test_a_late_or_unsure_answer_is_never_read_as_known(self):  # the review of H3.1
        for said in ("yes it is written later", "Yes, but only after dispatch", "known later", "Known after the delivery",
                     "correct, it is filled after", "available after the order is accepted", "right after the trip", "no"):
            self.assertEqual(checks.read_answer(said), "later", said)
        for said in ("I don't know if it is known", "not sure it is known", "maybe, I think so", "Use your recommendation."):
            self.assertEqual(checks.read_answer(said), "recommendation", said)
        self.assertEqual(checks.read_answer("yes, it is set when the order is created"), "known")
        self.assertEqual(checks.read_answer("yes, I guess"), "recommendation")
        self.assertFalse(checks.names_moment("It is known only when the delivery is finished; set at completion."))
        self.assertTrue(checks.names_moment("The delivery category is chosen when the order is placed, so it is known 'At order creation, "
                                            "before dispatch'; a high single-column AUC alone does not prove leakage."))

    def test_a_named_leak_or_a_precedent_is_never_asked_even_if_the_reviewer_says_available(self):  # the review of H3.1
        p = hyperack_proposal()
        p["review"]["verdicts"][0]["verdict"] = "available"  # the reviewer model wrong about final_customer_fare
        p["review"]["disagreements"].append({"column": "final_customer_fare", "reason": "Known at order creation, before dispatch."})
        p["forbidden"].append({"column": "is_delivered", "reason": "single column separates the classes (AUC 0.990)", "proof": ["DCLAB-R01"]})
        self.assertEqual([i["column"] for i in checks.disputed(p)], ["deliverey_category_id"])

    def test_an_answer_counts_only_for_its_outcome_and_moment(self):  # the review of H3.1: a new version keeps the understanding
        given = {"answer": "known", "target": "hyper_ack", "moment": "At order creation, before dispatch."}
        self.assertTrue(checks.current(given, "hyper_ack", "At order creation, before dispatch."))
        self.assertFalse(checks.current(given, "late_delivery", "At order creation, before dispatch."))
        self.assertFalse(checks.current(given, "hyper_ack", "After dispatch."))
        self.assertFalse(checks.current({"answer": "known"}, "hyper_ack", "At order creation, before dispatch."), "an old answer without them")

    def test_the_persons_own_answer_wins_either_way(self):
        item = checks.disputed(hyperack_proposal())[0]
        self.assertEqual(checks.read_answer(checks.OPTIONS[0]), "known")
        self.assertEqual(checks.read_answer(checks.OPTIONS[1]), "later")
        self.assertEqual(checks.read_answer(checks.OPTIONS[2]), "recommendation")
        self.assertEqual(checks.read_answer("No, it is filled in later by the dispatcher"), "later")
        self.assertEqual(checks.decide(item, "later")[0], False)
        vague = checks.disputed(hyperack_proposal(reviewer_says="Probably known then, it depends."))[0]
        self.assertEqual(checks.decide(vague, "known")[0], True)
        self.assertIn("person said", checks.decide(vague, "known")[1])


class ColumnCheckInHome(unittest.TestCase):
    def setUp(self):
        self.store = DraftStore(Path(tempfile.mkdtemp()))
        self.builds = []

        def on_request(draft_id, what, args):
            self.builds.append(what)
            return {"project_id": "p1", "started": True}
        self.agent = HomeAgent(self.store, None, on_request=on_request)
        self.draft = self.store.create("When a customer creates an order, predict whether a courier will accept it")
        item = checks.disputed(hyperack_proposal())[0]
        self.store.update(self.draft["id"], lambda d: d.update(column_check_pending={"item": item, "moment": "At order creation, before dispatch.", "target": "hyper_ack"}))

    def test_a_refused_build_asks_the_column_question_outside_the_question_cap(self):
        self.agent.built(self.draft["id"], {"refused": "column", "rule": "column check"})
        d = self.store.get(self.draft["id"])
        q = [x for x in d["questions"] if x["field"] == checks.FIELD]
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0]["column"], "deliverey_category_id")
        self.assertEqual(q[0]["options"], list(checks.OPTIONS))
        self.assertNotIn(q[0], counted(d), "a column check never takes a place among the prediction's questions")

    def test_the_answer_is_kept_then_the_build_goes_on(self):
        self.agent.built(self.draft["id"], {"refused": "column", "rule": "column check"})
        self.agent.reply(self.draft["id"], "Use your recommendation.")
        d = self.store.get(self.draft["id"])
        kept = d["understanding"]["column_checks"]["deliverey_category_id"]
        self.assertEqual((kept["answer"], kept["target"], kept["moment"]), ("recommendation", "hyper_ack", "At order creation, before dispatch."))
        self.assertNotIn("column_check_pending", d)
        self.assertEqual(self.builds, ["build"], "the build is asked again after the answer")

    def test_the_rules_recommendation_is_a_decision_the_answer_labels(self):  # S4: column.statistics_only
        from dclab_rnd import ledger
        led = ledger.FileLedger(Path(tempfile.mkdtemp()), "wtest")
        ledger.install(led)
        self.addCleanup(ledger.install, None)
        for said, label in (("Use your recommendation.", None), (checks.OPTIONS[0], "feature"), (checks.OPTIONS[1], "keep_out")):
            item = checks.disputed(hyperack_proposal())[0]
            self.store.update(self.draft["id"], lambda d: d.update(column_check_pending={"item": item, "moment": "At order creation, before dispatch.",
                                                                                         "target": "hyper_ack"}))
            self.agent.built(self.draft["id"], {"refused": "column", "rule": "column check"})
            self.agent.reply(self.draft["id"], said)
            record = led.records(point="column.statistics_only")[-1]
            self.assertEqual(record.answers[0].value, "feature" if checks.recommended(item) else "keep_out")  # the rule's recommendation
            self.assertEqual([x.value for x in led.labels(record=record.id)], [label] if label else [])  # "not sure" is no label
            self.assertEqual((record.subject.kind, record.subject.key, record.draft), ("column", item["column"], self.draft["id"]))
            self.assertEqual(record.state["column"], item["column"])  # joined with the column's other records by name
        from dclab_rnd.ledger import capture

        late = {**item, "reviewer": "after the moment"}  # the review of S4: the usual keep_out case, refused by the state's codes before
        rid = capture.column_checked(self.draft["id"], late, "later", {"columns": [{"name": item["column"], "kind": "numeric"}]})
        record = next(r for r in led.records(point="column.statistics_only") if r.id == rid)
        self.assertEqual((record.state["reviewer"], record.answers[0].value), ("after_moment", "keep_out"))
        self.assertEqual(record.state["dtype"], "float")  # with the column profiled, its bands are in the state


def wait(fn, timeout=60.0, every=0.1):
    end = time.time() + timeout
    while time.time() < end:
        value = fn()
        if value:
            return value
        time.sleep(every)
    raise AssertionError("timed out waiting")


class ColumnCheckThroughTheRoutes(unittest.TestCase):
    """Problem, upload, "build it": a column that alone separates the outcome is asked about; the answer decides."""

    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def draft(self, did):
        return self.client.get(f"/api/drafts/{did}").json()

    def ready_draft(self):
        """A draft whose outcome and moment are answered, on a table where `segment` alone separates the outcome."""
        c = self.client
        d = c.post("/api/drafts", json={"problem": "Predict which customers cancel next month"}, headers=self.h).json()
        # `segment` alone separates the outcome (its value is the outcome on 6 rows of 7, in both classes): strong, not late by its name
        rows = "tenure,plan,charges,segment,left\n" + "\n".join(
            f"{i % 60},{'ab'[i % 2]},{20 + i % 70},{(int(i % 5 == 0) if i % 7 else 1 - int(i % 5 == 0))},{int(i % 5 == 0)}" for i in range(400))
        self.assertEqual(c.put(f"/api/drafts/{d['id']}/data?filename=c.csv", content=rows.encode(), headers=self.h).status_code, 200)
        wait(lambda: next((a for a in self.draft(d["id"])["assets"] if a["status"] == "ready"), None))
        answers = {"target": "left", "prediction_moment": "At the monthly snapshot"}
        for _ in range(2):
            q = wait(lambda: next((q for q in self.draft(d["id"])["questions"] if not q.get("answered") and q["field"] in answers), None))
            text = answers.pop(q["field"])
            wait(lambda: c.post(f"/api/drafts/{d['id']}/messages", json={"text": text}, headers=self.h).status_code == 202)
            wait(lambda: any(x["id"] == q["id"] and x.get("answered") for x in self.draft(d["id"])["questions"]))
        return d

    def test_a_strong_column_is_asked_before_the_build_and_kept_when_known(self):
        c = self.client
        d = self.ready_draft()
        wait(lambda: c.post(f"/api/drafts/{d['id']}/messages", json={"text": "build it"}, headers=self.h).status_code == 202)
        q = wait(lambda: next((q for q in self.draft(d["id"])["questions"] if q["field"] == checks.FIELD and not q.get("answered")), None))
        self.assertEqual((q["column"], q["options"]), ("segment", list(checks.OPTIONS)))
        self.assertIn("I recommend to keep it out", q["text"], "no reviewer named the moment: the recommendation keeps it out")
        wait(lambda: c.post(f"/api/drafts/{d['id']}/messages", json={"text": checks.OPTIONS[0]}, headers=self.h).status_code == 202)
        built = wait(lambda: self.draft(d["id"]).get("project_id"))
        project = c.get(f"/api/projects/{built}").json()
        self.assertNotIn("segment", [f["column"] for f in project["solution"]["forbidden"]], "the person said it is known: a feature")
        self.assertIn("the person said it is known", project["solution"]["notes"])

    def test_a_button_build_asks_the_column_question_in_the_chat(self):  # the review of 2026-10-28: the Team panel's Build
        c = self.client
        d = self.ready_draft()
        for body in ({}, {"team": True}):
            refused = c.post(f"/api/drafts/{d['id']}/build/run", json=body, headers=self.h)
            self.assertEqual(refused.status_code, 409, refused.text)
            self.assertIn("segment", refused.json()["detail"])
            self.assertEqual(refused.headers.get("X-DCLab-Rule"), "column check")
        open_ = [q for q in self.draft(d["id"])["questions"] if q["field"] == checks.FIELD and not q.get("answered")]
        self.assertEqual([q["column"] for q in open_], ["segment"])  # asked once, in the chat, however many times refused
        wait(lambda: c.post(f"/api/drafts/{d['id']}/messages", json={"text": checks.OPTIONS[0]}, headers=self.h).status_code == 202)
        self.assertTrue(wait(lambda: self.draft(d["id"]).get("project_id")))  # the reply in the chat is the answer, and the build goes on


if __name__ == "__main__":
    unittest.main()
