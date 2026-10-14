"""Package 19.9: signals of quality. Each failure detector has a session it must flag and one it must not; the owner's
morning (anomaly detection built as a model of customer ids, the labels question unanswered) is flagged by more than
two; feedback joins the session record and "wrong" on a yes/no decision is a person's correction; the Sessions page
shows the flags."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from dclab_core import experience as x, signals
    from dclab_rnd import experience, ledger
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""


def session(*steps, visit="v1", start=1):
    return [x.make(kind, actor, {"workspace": "w1", "visit": visit, "conversation": "d1"}, payload, id=f"ev-{visit}-{i:04d}",
                   at=f"2026-10-13T09:{i:02d}:00+00:00") for i, (kind, actor, payload) in enumerate(steps, start=start)]


MORNING = session(("draft.created", "person", {"problem": "I want build anomaly detection"}),
                  ("data.uploaded", "person", {"name": "orders.csv"}),
                  ("data.analysed", "system", {"rows": 9000, "columns": 12, "roles": {"customer_id": "identifier", "order_request_id": "identifier"}}),
                  ("chat.agent", "agent", {"text": "Are some cases already confirmed?", "question_field": "labels"}),
                  ("solution.saved", "person", {"target": "customer_id", "task": "multiclass"}),
                  ("draft.built", "person", {"project": "p1"}))
GOOD = session(("draft.created", "person", {"problem": "Predict which orders arrive late, at the moment the order is placed"}),
               ("chat.agent", "agent", {"text": "When is the prediction made?", "question_field": "prediction_moment"}),
               ("chat.person", "person", {"text": "When the order is placed"}),
               ("data.analysed", "system", {"rows": 400, "columns": 6, "roles": {"order_id": "identifier", "late": "outcome-like"}}),
               ("solution.saved", "person", {"target": "late", "task": "binary"}),
               ("draft.built", "person", {"project": "p2"}),
               ("model.request", "model", {"purpose": "draft_chat", "cost": 0.001, "currency": "EUR"}),
               ("stage.finished", "system", {"stage": "data"}))

def other(*steps, visit="v1", conversation="d2", start=40):
    return [x.make(kind, actor, {"workspace": "w1", "visit": visit, "conversation": conversation}, payload, id=f"ev-{conversation}-{i:04d}",
                   at=f"2026-10-13T09:{i:02d}:00+00:00") for i, (kind, actor, payload) in enumerate(steps, start=start)]


ASK_TARGET = ("chat.agent", "agent", {"text": "What is the outcome?", "question_field": "target"})
CASES = {  # detector: (a session it must flag, the sessions it must not: the near misses the product makes every day)
    "identifier_target": (MORNING, [GOOD,
                                    session(("data.analysed", "system", {"roles": {"country_code": "categorical"}}),
                                            ("solution.saved", "person", {"target": "country_code", "task": "multiclass"})),
                                    session(("solution.saved", "person", {"target": "diagnosis_code", "task": "multiclass"})),
                                    session(("solution.saved", "person", {"target": "id_verified", "task": "binary"}))]),
    "built_with_question_open": (MORNING, [GOOD,
                                           session(("chat.agent", "agent", {"text": "What happens with each prediction?", "question_field": "action"}),
                                                   ("draft.built", "person", {"project": "p1"})),
                                           session(ASK_TARGET, ("data.uploaded", "person", {"name": "labels.csv"}),  # withdrawn by a label file
                                                   ("data.analysed", "system", {"rows": 10, "columns": 3}), ("draft.built", "person", {"project": "p1"}))]),
    "task_contradicts_words": (MORNING, [session(("draft.created", "person", {"problem": "Find the rare fraud cases among the orders"}),
                                                 ("solution.saved", "person", {"target": "is_fraud", "task": "binary"})),
                                         session(("draft.created", "person", {"problem": "I want build anomaly detection"})) +  # two drafts
                                         other(("draft.created", "person", {"problem": "Predict delivery time in minutes"}),
                                               ("solution.saved", "person", {"target": "minutes", "task": "regression"})),
                                         session(("draft.created", "person", {"problem": "I want build anomaly detection"}),
                                                 ("draft.edited", "person", {"field": "problem", "before": "I want build anomaly detection",
                                                                             "after": "Predict delivery time in minutes"}),
                                                 ("solution.saved", "person", {"target": "minutes", "task": "regression"}))]),
    "rephrased": (session(("draft.edited", "person", {"field": "problem", "before": "a", "after": "b"}),
                          ("draft.edited", "person", {"field": "problem", "before": "b", "after": "c"})),
                  [session(("draft.edited", "person", {"field": "problem", "before": "a", "after": "b"}), ("chat.person", "person", {"text": "yes"})),
                   session(("chat.agent", "agent", {"text": "?", "question_field": "labels"}), ("chat.person", "person", {"text": "Yes"}),
                           ("chat.agent", "agent", {"text": "?", "question_field": "action"}), ("chat.person", "person", {"text": "yes"}))]),
    "error": (session(("error.raised", "system", {"where": "/api/drafts/{draft_id}/build", "code": "E_PARSE"})), [GOOD]),
    "spend_without_result": (session(("draft.created", "person", {"problem": "x y z w"}), ("model.request", "model", {"cost": 0.002, "currency": "EUR"})),
                             [GOOD,
                              session(("chat.person", "person", {"text": "Why this metric?", "surface": "project"}),
                                      ("model.request", "model", {"cost": 0.002, "currency": "EUR"}),
                                      ("chat.agent", "agent", {"text": "Because...", "kind": "answer", "surface": "project"})),
                              session(("stage.started", "system", {"stage": "features"}), ("model.request", "model", {"cost": 0.002, "currency": "EUR"}))]),
    "build_refused_again": (session(("draft.refused", "system", {"rule": "problem card not ready"}), ("draft.refused", "system", {"rule": "problem card not ready"})),
                            [session(("draft.refused", "system", {"rule": "problem card not ready"}), ("draft.built", "person", {"project": "p3"}))]),
    "question_asked_again": (session(("chat.agent", "agent", {"text": "When?", "question_field": "prediction_moment"}), ("chat.person", "person", {"text": "x"}),
                                     ("chat.agent", "agent", {"text": "When, again?", "question_field": "prediction_moment"})),
                             [GOOD,
                              session(ASK_TARGET, ("chat.person", "person", {"text": "whether a customer cancels"}),  # the follow-up names the columns
                                      ("chat.agent", "agent", {"text": "Which column records that outcome?", "question_field": "target",
                                                               "options": ["churned", "Something else"]}))]),
    "left_unbuilt": (session(("draft.created", "person", {"problem": "Find late orders please"}), ("chat.agent", "agent", {"text": "hello"})), [GOOD]),
}
MORE_BAD = {  # what the near misses must not hide
    "task_contradicts_words": [session(("draft.created", "person", {"problem": "Detect anomalies in sensor readings, nothing is labelled"}),
                                       ("solution.saved", "person", {"target": "fault", "task": "binary"})),
                               session(("draft.created", "person", {"problem": "I want build anomaly detection"}), ("draft.built", "person", {"project": "p9"}))
                               + [x.make("solution.saved", "person", {"workspace": "w1", "visit": "v1", "project": "p9"},
                                         {"target": "segment", "task": "multiclass"}, id="ev-p9-save", at="2026-10-13T09:30:00+00:00")]],
    "rephrased": [session(("chat.person", "person", {"text": "Predict churn"}), ("chat.agent", "agent", {"text": "Sorry?"}),
                          ("chat.person", "person", {"text": "predict  churn"}))],
    "built_with_question_open": [session(("chat.agent", "agent", {"text": "?", "question_field": "labels"}),
                                         ("chat.person", "person", {"text": "an answer in the intern"}), ("draft.built", "person", {"project": "p1"}))[:1]
                                 + other(("chat.person", "person", {"text": "an answer in another conversation"}), start=2)
                                 + session(("draft.built", "person", {"project": "p1"}), start=3)],
}


@unittest.skipIf(MISSING, MISSING)
class DetectorTest(unittest.TestCase):
    def test_every_detector_has_a_session_it_flags_and_one_it_does_not(self):
        self.assertEqual(set(CASES), {fn.__name__ for fn in signals.DETECTORS})
        for name, (bad, goods) in CASES.items():
            for i, one in enumerate([bad, *MORE_BAD.get(name, [])]):
                with self.subTest(detector=name, flags=i):
                    self.assertTrue(signals.detect(one, only=[name]), "missed")
            for i, good in enumerate(goods):
                with self.subTest(detector=name, must_not=i):
                    self.assertEqual(signals.detect(good, only=[name]), [], "false alarm")

    def test_the_owners_morning_is_flagged(self):
        found = {f.detector for f in signals.detect(MORNING)}
        self.assertTrue({"identifier_target", "built_with_question_open", "task_contradicts_words"} <= found, found)
        self.assertEqual(signals.detect(GOOD), [])
        for flag in signals.detect(MORNING):
            self.assertTrue(flag.events and all(e in {ev.id for ev in MORNING} for e in flag.events))

    def test_an_answered_question_and_a_key_named_outcome(self):
        answered = session(("chat.agent", "agent", {"text": "?", "question_field": "labels"}), ("chat.person", "person", {"text": "Nothing is labelled"}),
                           ("draft.built", "person", {"project": "p1"}))
        self.assertEqual(signals.detect(answered, only=["built_with_question_open"]), [])
        keyed = session(("solution.saved", "person", {"target": "order_request_id", "task": "binary"}))
        self.assertEqual([f.severity for f in signals.detect(keyed, only=["identifier_target"])], ["medium"])  # by name only: medium
        self.assertEqual([f.severity for f in signals.detect(MORNING, only=["identifier_target"])], ["high"])  # the analysis said so
        self.assertEqual(signals.detect(session(("solution.saved", "person", {"target": "is_paid", "task": "binary"})), only=["identifier_target"]), [])


@unittest.skipIf(MISSING, MISSING)
class ProductTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_rnd.agentic.server import create_app

        cls.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        cls.services = cls.client.app.state.services
        cls.h = {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}
        cls.services.experience.append([e.model_copy(update={"ids": e.ids.model_copy(update={"workspace": cls.services.experience.workspace})})
                                        for e in MORNING])

    def test_the_sessions_page_shows_the_flags(self):
        card = next(c for c in self.client.get("/api/experience/sessions").json()["items"] if c["visit"] == "v1")
        self.assertIn("identifier_target", {f["detector"] for f in card["flags"]})
        self.assertGreaterEqual(self.client.get("/api/experience/sessions").json()["totals"]["flagged"], 1)
        one = self.client.get("/api/experience/sessions/v1").json()
        self.assertTrue(one["session"]["flags"])

    def test_feedback_joins_the_record_and_wrong_corrects_a_decision(self):
        r = self.client.post("/api/feedback", json={"on": "answer", "ref": "m1", "rating": "unhelpful", "reason": "it asked twice", "draft": "d1"},
                             headers=self.h)
        self.assertEqual(r.status_code, 200, r.text)
        given = [e for e in self.services.experience.read(kinds=["feedback.given"])]
        self.assertIn(("unhelpful", "it asked twice", "d1"), [(e.payload["rating"], e.payload.get("reason"), e.ids.conversation) for e in given])
        decided = ledger.decide("column.is_identifier", ("column", "customer_id"), answers=[ledger.answer("rule", "rule:id_like", False)],
                                value_used=False, ledger=self.services.ledger, project="p1")
        out = self.client.post("/api/feedback", json={"on": "decision", "ref": decided.id, "rating": "wrong"}, headers=self.h).json()
        self.assertIsNotNone(out["label"])
        corrected = self.services.ledger.labels(record=decided.id)
        self.assertEqual([(x.source, x.value) for x in corrected], [("user_decision_exposed", True)])
        self.assertEqual(self.client.post("/api/feedback", json={"on": "answer", "rating": "great"}, headers=self.h).status_code, 422)
        again = self.client.post("/api/feedback", json={"on": "decision", "ref": decided.id, "rating": "wrong"}, headers=self.h).json()
        self.assertIsNone(again["label"])  # one correction per person, however often it is sent
        self.assertEqual(len(self.services.ledger.labels(record=decided.id)), 1)

    def test_feedback_failure_paths(self):
        from dclab_rnd.accounts.principal import Principal, acting
        from dclab_rnd.agentic.pages.ledger import feedback

        planted = "jane.doe" + "@" + "example.com"
        self.client.post("/api/feedback", json={"on": "answer", "ref": planted, "rating": "helpful"}, headers=self.h)
        given = [e for e in self.services.experience.read(kinds=["feedback.given"]) if e.payload["rating"] == "helpful"]
        self.assertEqual([e.payload["on"] for e in given], ["answer"])  # "on" is meta and leaves with an export: a free-text ref is dropped
        self.assertEqual(self.client.post("/api/feedback", json={"on": "decision", "ref": "dl_no_such_one", "rating": "wrong"},
                                          headers=self.h).status_code, 404)
        book = self.services.ledger
        yes_no = ledger.decide("column.is_identifier", ("column", "order_id"), answers=[ledger.answer("rule", "rule:id_like", True)],
                               value_used=True, ledger=book, project="p1")
        with acting(Principal(role="viewer", user_id="u_viewer")):  # a viewer's "wrong" is feedback, not a label
            self.assertIsNone(feedback({"on": "decision", "ref": yes_no.id, "rating": "wrong"}, book)["label"])
        self.assertEqual(book.labels(record=yes_no.id), [])
        with acting(Principal(role="reviewer", user_id="u_rev")):
            self.assertIsNotNone(feedback({"on": "decision", "ref": yes_no.id, "rating": "wrong"}, book)["label"])
        self.assertEqual([(x.by, x.value) for x in book.labels(record=yes_no.id)], [("person:u_rev", False)])
        family = ledger.decide("model.family", ("project", "p1"), answers=[ledger.answer("rule", "r", "ridge")], value_used="ridge",
                               ledger=book, project="p1")
        self.assertIsNone(feedback({"on": "decision", "ref": family.id, "rating": "wrong"}, book)["label"])  # not a yes or no: no opposite


if __name__ == "__main__":
    unittest.main()
