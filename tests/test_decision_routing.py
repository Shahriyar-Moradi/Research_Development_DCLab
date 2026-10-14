"""Package 22.6, choosing a model per decision: switching a point's answerer is one audited change (a reviewer or the
owner, a reason, an answerer that passed its sealed gate above shadow, and to act one measured at least as right as the
rule in shadow), the rule answers when a routed model abstains, fails or disagrees at "act with a check", and a model
whose weighted accuracy on new labels falls below its bar is demoted by DCLab and falls back to the rule."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import ledger as contract  # noqa: E402

try:
    import pandas  # noqa: F401

    from dclab_rnd import audit
    from dclab_rnd.ledger import routing
    from dclab_rnd.models.routing import FileRouting
except ImportError as error:
    MISSING = str(error)
else:
    MISSING = ""

POINT = "feature.leakage_suspect"


class Fake:
    """An answerer that says what it is told, or fails."""

    by = "classifier"

    def __init__(self, name, version, says):
        self.name, self.version, self.points, self.says = name, version, frozenset({POINT}), says

    def answer(self, point, state, options):
        if isinstance(self.says, Exception):
            raise self.says
        return None if self.says is None else contract.Answer(by="classifier", answerer=f"{self.name}@{self.version}", value=self.says)


def rule(value):
    return contract.Answer(by="rule", answerer="rule:column_audit", value=value)


@unittest.skipIf(MISSING, MISSING)
class DecisionRoutingTest(unittest.TestCase):
    def setUp(self):
        tmp = Path(tempfile.mkdtemp())
        self.store, self.audit = FileRouting(tmp / "model_routing.json"), audit.FileAudit(tmp / "audit.jsonl")
        self.passed = next(n for n, i in routing.passed().items() if i["point"] == POINT)  # the leakage classifier that passed 22.3
        self.failed = "column.time.boosting@1186fbf4bce4"  # judged and failed: shadow only
        self.good = {self.passed: {"labelled_disagreements": 25, "weighted_right": 20.0, "weighted_rule_right": 5.0}}  # measured in shadow

    def act(self, level=3):
        return routing.switch(self.store, self.audit, POINT, self.passed, level, "reviewer", "passed its gate, right more often in shadow",
                              measured=self.good)

    def test_one_audited_change_by_a_reviewer_with_a_reason(self):
        with self.assertRaises(PermissionError):
            routing.switch(self.store, self.audit, POINT, self.passed, 1, "data_scientist", "it passed")
        with self.assertRaises(routing.DecisionRoutingError):
            routing.switch(self.store, self.audit, POINT, self.passed, 1, "reviewer", "")
        with self.assertRaises(routing.DecisionRoutingError):  # did not pass its gate: shadow only
            routing.switch(self.store, self.audit, "column.time", self.failed, 1, "reviewer", "try it")
        with self.assertRaises(routing.DecisionRoutingError):  # judged at another point
            routing.switch(self.store, self.audit, "column.time", self.passed, 1, "reviewer", "try it")
        for bad in (0, 1.5, "high"):
            with self.assertRaises(routing.DecisionRoutingError):
                routing.switch(self.store, self.audit, POINT, self.passed, 1, "reviewer", "a bar", bar=bad)
        doc = self.act()
        self.assertEqual((routing.entry(doc, POINT)["answerer"], routing.entry(doc, POINT)["level"]), (self.passed, 3))
        self.act()  # the same again changes nothing
        total, rows = self.audit.query(10, 0, kind="routing")
        self.assertEqual(total, 1)
        self.assertEqual((rows[0]["move"], rows[0]["detail"]["args"]["to"]), ("set_decision_routing", {"answerer": self.passed, "level": 3, "bar": 0.9}))
        routing.switch(self.store, self.audit, "column.time", self.failed, 0, "owner", "measure it in shadow")  # shadow is allowed

    def test_to_act_a_model_must_have_been_measured_right_in_shadow(self):
        with self.assertRaises(routing.DecisionRoutingError):  # passed its sealed gate, never measured on the studio's decisions
            routing.switch(self.store, self.audit, POINT, self.passed, 2, "reviewer", "it passed")
        worse = {self.passed: {"labelled_disagreements": 25, "weighted_right": 4.0, "weighted_rule_right": 21.0}}
        with self.assertRaises(routing.DecisionRoutingError):  # measured, and wrong where it differs from the rule
            routing.switch(self.store, self.audit, POINT, self.passed, 2, "reviewer", "it passed", measured=worse)
        routing.switch(self.store, self.audit, POINT, self.passed, 1, "reviewer", "flag only")  # flagging needs only the gate
        self.assertEqual(routing.entry(self.act(2), POINT)["level"], 2)

    def test_the_rule_answers_when_the_model_abstains_fails_disagrees_at_a_check_or_only_flags(self):
        name, version = self.passed.split("@")
        says_yes = Fake(name, version, True)
        doc = self.act(3)
        self.assertEqual(routing.answer(doc, POINT, {}, ["yes", "no"], rule(False), [says_yes])[:3], (True, "classifier", 3))
        for silent in (Fake(name, version, None), Fake(name, version, RuntimeError("broken")), Fake(name, version, "maybe")):
            self.assertEqual(routing.answer(doc, POINT, {}, ["yes", "no"], rule(False), [silent])[:2], (False, "rule"))
        check = self.act(2)
        value, used, level, answers, review = routing.answer(check, POINT, {}, ["yes", "no"], rule(False), [says_yes])
        self.assertEqual((value, used, level, review), (False, "rule", 2, True))  # they differ: the rule stands, a person decides
        self.assertEqual(routing.answer(check, POINT, {}, ["yes", "no"], rule(True), [says_yes])[:2], (True, "classifier"))  # they agree
        flag = routing.switch(self.store, self.audit, POINT, self.passed, 1, "reviewer", "flag only")
        value, used, level, answers, review = routing.answer(flag, POINT, {}, ["yes", "no"], rule(False), [says_yes])
        self.assertEqual((value, used, level, review), (False, "rule", 1, False))
        self.assertEqual([a.value for a in answers], [False, True])  # the model's flag is beside the rule's answer
        self.assertEqual(routing.answer({}, POINT, {}, ["yes", "no"], None, [says_yes]), (None, "none", 0, [], False))  # no rule, not routed

    def test_a_model_below_its_bar_on_new_labels_is_demoted_by_dclab_and_the_rule_answers(self):
        since = routing.entry(self.act(3), POINT)["since"]
        records, labels = [], []
        for i in range(30):
            r = contract.DecisionRecord(id=f"dr{i:012d}", point=POINT, subject=contract.Subject(kind="column", key=f"c{i}"), state={},
                                        options=["yes", "no"], answers=[rule(False), contract.Answer(by="classifier", answerer=self.passed, value=True)],
                                        value_used=True, used="classifier", level=3, at=since)
            records.append(r)
            labels.append(contract.DecisionLabel(id=f"dl{i:012d}", record=r.id, point=POINT, value=i < 10, source="benchmark", by="t", at=since))
        self.assertEqual(routing.demote(self.store, self.audit, records[:15], labels), [])  # too few labels yet
        out = routing.demote(self.store, self.audit, records, labels)
        self.assertEqual((out[0]["accuracy"], out[0]["labelled"]), (0.3333, 30))
        doc = self.store.get()
        self.assertEqual(routing.entry(doc, POINT)["level"], 0)
        name, version = self.passed.split("@")
        self.assertEqual(routing.answer(doc, POINT, {}, ["yes", "no"], rule(False), [Fake(name, version, True)])[:2], (False, "rule"))
        _, rows = self.audit.query(10, 0, kind="routing")
        self.assertIn("demoted to shadow", rows[0]["detail"].get("message", rows[0].get("message", "")))
        self.assertEqual((rows[0]["actor"], rows[0]["who"]), ("agent", "DCLab (automatic demotion)"))  # DCLab demoted it, not a person
        self.assertEqual(doc["history"][-1]["by"], routing.SYSTEM)

    def test_blind_labels_weigh_as_they_were_drawn(self):
        since = routing.entry(self.act(3), POINT)["since"]
        records, labels = [], []
        for i in range(150):  # 90 agreeing decisions drawn at 0.1 (weight 10), 60 differing ones at 0.6 (weight 1.667)
            agree = i < 90
            r = contract.DecisionRecord(id=f"dr{i:012d}", point=POINT, subject=contract.Subject(kind="column", key=f"c{i}"), state={},
                                        options=["yes", "no"], answers=[rule(True if agree else False), contract.Answer(by="classifier", answerer=self.passed, value=True)],
                                        value_used=True, used="classifier", level=3, at=since)
            records.append(r)
            right = agree or i < 126  # right on every agreement and on 36 of the 60 differences
            labels.append(contract.DecisionLabel(id=f"dl{i:012d}", record=r.id, point=POINT, value=right, source="blind", by="reviewer", at=since,
                                                 weight=10.0 if agree else round(1 / 0.6, 6)))
        self.assertEqual(routing.demote(self.store, self.audit, records, labels), [])  # weighted 0.96: above its bar; unweighted 0.84 would demote it
        self.assertEqual(routing.entry(self.store.get(), POINT)["level"], 3)

    def test_the_candidates_are_those_that_passed_best_first(self):
        got = routing.candidates(POINT)
        self.assertEqual([c["answerer"] for c in got], [self.passed])
        self.assertGreater(got[0]["sealed_accuracy"], 0.9)
        self.assertEqual(routing.candidates("column.time"), [])

    def test_the_audit_backfill_reads_a_decision_change_as_one(self):
        from dclab_rnd.studio import ProjectStore

        routing.switch(self.store, self.audit, POINT, self.passed, 1, "reviewer", "flag only")
        rows = audit.history(ProjectStore(Path(tempfile.mkdtemp())), self.store)
        mine = [r for r in rows if r["kind"] == "routing"]
        self.assertEqual(mine[0]["move"], "set_decision_routing")
        self.assertEqual(mine[0]["detail"]["args"]["decision_point"], POINT)

    def test_the_routes(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        body = {"point": POINT, "answerer": self.passed, "level": 1, "reason": "passed its sealed gate"}
        self.assertEqual(client.post("/api/ledger/routing", json=body, headers=h).status_code, 403)  # no reviewer named
        r = client.post("/api/ledger/routing", json=body, headers={**h, "X-DCLab-Role": "reviewer"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["routed"]["level"], 1)
        r = client.post("/api/ledger/routing", json={**body, "level": 2}, headers={**h, "X-DCLab-Role": "reviewer"})
        self.assertEqual(r.status_code, 422, r.text)  # no shadow measurement in this workspace: it may not act
        got = client.get("/api/ledger/routing", headers=h).json()
        self.assertEqual(got["points"][POINT]["routed"]["answerer"], self.passed)
        self.assertEqual(client.post("/api/ledger/routing", json={**body, "answerer": self.failed, "point": "column.time"},
                                     headers={**h, "X-DCLab-Role": "reviewer"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
