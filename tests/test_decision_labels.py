"""Package 19.8: labels from every source. Every decision on a replica about a planted trap gets its benchmark label; a
person's override becomes an exposed label that supersedes the rule's answer only by demoting it; a measured number
(a leak's lift, a split's optimism) is a measured label; a blind label carries its sampling weight; the precedence
(blind, benchmark, measured, exposed) is one function. The routes are the owner's and reviewers'."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from dclab_core import ledger as L
    from dclab_rnd import ledger
    from dclab_rnd.ledger import capture, labels
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""

T0, T1, T2 = "2026-10-14T08:00:00+00:00", "2026-10-14T09:00:00+00:00", "2026-10-14T10:00:00+00:00"


def record(**over):
    base = dict(id="dr-0000001", point="column.is_identifier", subject=L.Subject(kind="column", key="order_id"), at=T0,
                answers=[L.Answer(by="rule", answerer="rule:id_like", value=True)], value_used=True, used="rule")
    return L.DecisionRecord(**{**base, **over})


def lab(i, source, value, at=T1, **over):
    return L.DecisionLabel(**{"id": f"dl-{i:06d}", "record": "dr-0000001", "point": "column.is_identifier", "value": value, "source": source,
                              "by": "x", "at": at, **({"weight": 2.0} if source == "blind" else {}), **over})


@unittest.skipIf(MISSING, MISSING)
class PrecedenceTest(unittest.TestCase):
    def test_the_strongest_source_wins_and_a_later_label_supersedes(self):
        r = record()
        self.assertIsNone(L.effective_label(r, []))
        exposed, measured, bench, blind = lab(1, "user_decision_exposed", False), lab(2, "measured", True), lab(3, "benchmark", False), lab(4, "blind", True)
        self.assertEqual(L.effective_label(r, [exposed]).id, exposed.id)
        self.assertEqual(L.effective_label(r, [exposed, measured]).id, measured.id)
        self.assertEqual(L.effective_label(r, [exposed, measured, bench]).id, bench.id)
        self.assertEqual(L.effective_label(r, [exposed, measured, bench, blind]).id, blind.id)
        newer = lab(5, "benchmark", True, at=T2, supersedes=bench.id)
        self.assertEqual(L.effective_label(r, [bench, newer]).id, newer.id)
        self.assertEqual([x.id for x in L.current_labels([bench, newer])], [newer.id])

    def test_an_exposed_label_may_demote_never_promote(self):
        r = record()  # the rule said yes and yes was used
        agreeing = lab(1, "user_decision_exposed", True)  # a person who saw "yes" and said yes: no evidence
        self.assertIsNone(L.effective_label(r, [agreeing]))
        demoting = lab(2, "user_decision_exposed", False, at=T2)
        self.assertEqual(L.effective_label(r, [agreeing, demoting]).id, demoting.id)

    def test_the_blind_draw_is_known_and_over_samples_disagreements(self):
        agree = [record(id=f"dr-agree-{i:04d}", answers=[L.Answer(by="rule", answerer="r", value=True), L.Answer(by="classifier", answerer="c", value=True)])
                 for i in range(2000)]
        disagree = [record(id=f"dr-dis-{i:04d}", answers=[L.Answer(by="rule", answerer="r", value=True), L.Answer(by="classifier", answerer="c", value=False)])
                    for i in range(2000)]
        share_agree = sum(L.blind_draw(r) for r in agree) / len(agree)
        share_disagree = sum(L.blind_draw(r) for r in disagree) / len(disagree)
        self.assertAlmostEqual(share_agree, L.BLIND_RATES["agree"], delta=0.03)
        self.assertAlmostEqual(share_disagree, L.BLIND_RATES["disagree"], delta=0.04)
        self.assertEqual([L.blind_draw(r) for r in agree[:50]], [L.blind_draw(r) for r in agree[:50]])  # reproducible
        view = L.blind_view(disagree[0])
        self.assertNotIn("answers", view)
        self.assertNotIn("value_used", view)
        self.assertNotIn("used", view)


@unittest.skipIf(MISSING, MISSING)
class StudioLabelsTest(unittest.TestCase):
    def setUp(self):
        self.store = ledger.FileLedger(Path(tempfile.mkdtemp()) / "ledger", "wtest")
        ledger.install(self.store)

    def tearDown(self):
        ledger.install(None)

    def test_a_replica_labels_the_decisions_about_its_planted_traps(self):
        from dclab_rnd import kinds
        from dclab_rnd.kinds import acceptance

        case = kinds.get("regression").case(0)  # delivery time: delivered_at and courier_rating are written after the moment
        acceptance.engine_run(case)
        leaks = set(case.facts["leaky_columns"])
        benchmark = [x for x in self.store.labels() if x.source == "benchmark"]
        labelled = {next(r for r in self.store.records() if r.id == x.record).subject.key for x in benchmark if x.point == "feature.leakage_suspect"}
        self.assertTrue(leaks & labelled, f"no planted leak labelled: {labelled}")
        self.assertTrue(all(x.value is True and x.evidence.kind == "planted_truth" for x in benchmark))
        entity = case.facts["entity"]
        about_entity = [r for r in self.store.records(point="column.group") if r.subject.key == entity]
        self.assertEqual(len([x for x in benchmark if x.record in {r.id for r in about_entity}]), len(about_entity))  # each one made, labelled
        before = len(self.store.labels())
        labels.from_replica(next(iter({r.project for r in self.store.records() if r.project})), case.facts, "delivery_time")
        self.assertEqual(len(self.store.labels()), before)  # labelled once, never again

    def test_a_person_who_chose_otherwise_writes_an_exposed_label(self):
        profile = {"rows": 300, "columns": [{"name": "fare", "kind": "numeric", "unique_ratio": 0.7, "missing_rate": 0.0, "unique": 210}]}
        capture.solution_saved({"target": "late", "rule": {"forbidden": [], "identifiers": []}}, profile,
                               {"target": "late", "task": "binary", "forbidden": [{"column": "fare"}], "identifiers": []}, "human", project="p1")
        fare = next(r for r in self.store.records(point="feature.leakage_suspect") if r.subject.key == "fare")
        exposed = [x for x in self.store.labels(record=fare.id) if x.source == "user_decision_exposed"]
        self.assertEqual([(x.value, x.by) for x in exposed], [(True, "owner")])
        same = next(r for r in self.store.records(point="column.is_identifier") if r.subject.key == "fare")
        self.assertEqual(self.store.labels(record=same.id), [])  # agreeing with the rule labels nothing
        self.assertEqual(L.effective_label(fare, self.store.labels()).id, exposed[0].id)  # it demotes the rule's "no": it counts

    def test_heuristics_nobody_saw_write_no_exposed_label(self):
        capture.stage_decisions("leakage", {"task_type": "binary", "evidence": {"declared_leakage_features": ["fare"],
                                                                               "heuristic_review_candidates": [{"feature": "zone"}]}},
                                None, {"id": "p1", "data": {}}, project="p1")
        self.assertEqual([x for x in self.store.labels() if x.source == "user_decision_exposed"], [])

    def test_replicas_name_their_traps_their_own_way(self):
        self.assertEqual(labels.truth_of({"identifier": "subscriber_id", "leak": "final_refund_amount", "time_column": "signup_time"}),
                         {"identifiers": ["subscriber_id"], "leaky_columns": ["final_refund_amount"], "time": ["signup_time"]})
        self.assertEqual(labels.truth_of({"leaks": ["a", "b"], "group": "customer_id"}), {"leaky_columns": ["a", "b"], "entity": ["customer_id"]})

    def test_a_measured_optimism_labels_the_split(self):
        split = ledger.decide("split.strategy", ("project", "p1"), answers=[ledger.answer("rule", "rule:kind_split", "time")], value_used="time",
                              project="p1")
        older = ledger.decide("split.strategy", ("project", "p1"), answers=[ledger.answer("rule", "rule:kind_split", "group")], value_used="group",
                              project="p1")  # a later split that is not a time split: the gap says nothing about it
        capture.stage_decisions("leakage", {"task_type": "binary", "evidence": {"random_vs_time_cv": {"optimism_gap": 0.06}}}, None,
                                {"id": "p1", "data": {}}, project="p1")
        measured = self.store.labels(record=split.id)
        self.assertEqual([(x.source, x.value, x.evidence.kind, x.evidence.value) for x in measured], [("measured", "time", "optimism_gap", 0.06)])
        self.assertEqual(self.store.labels(record=older.id), [])

    def test_a_blind_reviewer_answers_only_what_the_state_can_answer(self):
        hidden = ledger.decide("split.strategy", ("project", "p1"), answers=[ledger.answer("rule", "r", "time")], value_used="time", project="p1")
        self.assertFalse(L.blind_eligible(hidden))  # a value with no options: nothing to choose among, never drawn (never a stall)
        move = ledger.decide("workflow.move", ("workflow", "run_stage:final"), answers=[ledger.answer("rule", "r", True)], value_used=True,
                             options=["run_stage", "approve_stage"], project="p1")
        self.assertEqual(L.blind_options(move), ["yes", "no"])  # a yes/no point is answered yes or no, whatever moves were allowed
        revisit = ledger.decide("workflow.revisit", ("workflow", "run_stage:models"), answers=[ledger.answer("person", "human", "models")],
                                value_used="models", used="person", state={"stage": "models"}, project="p1")
        self.assertFalse(L.blind_eligible(revisit))  # its state names the answer
        view = L.blind_view(ledger.decide("metric.primary", ("project", "p1"), answers=[ledger.answer("rule", "r", "roc_auc")], value_used="roc_auc",
                                          options=["roc_auc", "average_precision"], state={"metric": "roc_auc", "rows": "small"}, project="p1"))
        self.assertNotIn("metric", view["state"])  # a state field equal to the answer is not shown

    def test_a_disagreement_is_drawn_more_often(self):
        def rec(i, person):
            return L.DecisionRecord(id=f"dr-strat-{i:05d}", point="column.is_identifier", subject=L.Subject(kind="column", key="c"), at=T0,
                                    answers=[L.Answer(by="rule", answerer="r", value=True), L.Answer(by="person", answerer="p", value=person)],
                                    value_used=person, used="person")
        self.assertEqual((L.stratum(rec(0, False)), L.stratum(rec(1, True))), ("disagree", "agree"))
        drawn = sum(L.blind_draw(rec(i, False)) for i in range(1000)) / 1000
        self.assertGreater(drawn, 0.5)

    def test_a_blind_label_carries_its_weight_and_only_a_drawn_decision_takes_one(self):
        records = [ledger.decide("column.is_identifier", ("column", f"c{i}"), answers=[ledger.answer("rule", "r", True)], value_used=True, project="p1")
                   for i in range(80)]
        item = labels.next_blind()
        self.assertIsNotNone(item)
        self.assertNotIn("value_used", item)
        out = labels.blind_label(item["record"], False, "reviewer")
        self.assertEqual((out.source, out.weight), ("blind", item["weight"]))
        self.assertNotEqual(labels.next_blind()["record"], item["record"])
        undrawn = next(r for r in records if not L.blind_draw(r))
        with self.assertRaises(ValueError):
            labels.blind_label(undrawn.id, True, "reviewer")
        with self.assertRaises(ValueError):  # one blind label per reviewer and decision: a second would count its weight twice
            labels.blind_label(item["record"], True, "reviewer")
        with self.assertRaises(ValueError):  # a label says what was right
            labels.blind_label(labels.next_blind()["record"], None, "reviewer")
        with self.assertRaises(ValueError):  # not an answer at this point
            labels.blind_label(labels.next_blind()["record"], "Jane Roe", "reviewer")


@unittest.skipIf(MISSING, MISSING)
class RoutesTest(unittest.TestCase):
    def test_the_blind_labelling_routes(self):
        from dclab_rnd.accounts import guard
        from dclab_rnd.accounts.principal import Principal
        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        book = client.app.state.services.ledger
        for i in range(60):
            ledger.decide("column.is_identifier", ("column", f"c{i}"), answers=[ledger.answer("rule", "r", True)], value_used=True, ledger=book, project="p1")
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        summary = client.get("/api/ledger/summary").json()
        self.assertEqual((summary["decisions"], summary["by_point"]["column.is_identifier"]), (60, 60))
        item = client.get("/api/ledger/blind/next").json()["item"]
        done = client.post(f"/api/ledger/blind/{item['record']}", json={"value": True}, headers=h)
        self.assertEqual(done.status_code, 200, done.text)
        self.assertEqual(done.json()["label"]["source"], "blind")
        self.assertEqual(client.post("/api/ledger/blind/dr-missing-01", json={"value": True}, headers=h).status_code, 404)
        self.assertEqual(client.post(f"/api/ledger/blind/{item['record']}", json={}, headers=h).status_code, 422)
        self.assertEqual(client.post(f"/api/ledger/blind/{item['record']}", json={"value": True}, headers=h).status_code, 422)  # twice
        for i in range(60, 300):  # enough drawn decisions that one is left after a skip (the ids, so the draw, are random)
            ledger.decide("column.is_identifier", ("column", f"c{i}"), answers=[ledger.answer("rule", "r", True)], value_used=True, ledger=book, project="p1")
        following = client.get("/api/ledger/blind/next").json()["item"]
        after = client.get("/api/ledger/blind/next", params={"skip": following["record"]}).json()["item"]
        self.assertNotEqual(after["record"], following["record"])  # a decision the reviewer cannot tell is passed, not guessed
        self.assertEqual(client.get("/api/ledger/blind/next").json()["item"]["record"], following["record"])  # and stays unlabelled
        self.assertEqual(client.get("/api/ledger/blind/next", params={"skip": "a b"}).status_code, 422)
        with mock.patch.object(guard, "current", return_value=Principal(role="viewer", user_id="u1", via="session")):
            self.assertEqual(client.get("/api/ledger/summary").status_code, 403)
        with mock.patch.object(guard, "current", return_value=Principal(role="reviewer", user_id="u1", via="session")):
            self.assertEqual(client.get("/api/ledger/blind/next").status_code, 200)


if __name__ == "__main__":
    unittest.main()
