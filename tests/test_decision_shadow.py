"""Package 22.4, the decision classifiers in shadow: a classifier answers beside the rule at its decision point, at
level 0; its answer is recorded in the decision record, never used; agreement and disagreement are counted, with its
accuracy on the labelled disagreements; disagreements reach the blind labelling page first. The fitted models stay on
the owner's machine, so these tests train small classifiers of their own."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import ledger as contract  # noqa: E402
from dclab_core.answerers import SHADOW, ClassifierAnswerer  # noqa: E402

try:
    import pandas  # noqa: F401

    from dclab_rnd import ledger
    from dclab_rnd.ledger import labels as ledger_labels
    from dclab_rnd.ledger import shadow
except ImportError as error:  # the agent environment has no pandas
    MISSING = str(error)
else:
    MISSING = ""


def leak_classifier():
    """Says a column named like a refund is a leak: trained on a handful of states."""
    rows = [({"column": f"{w}_{i}", "name_tokens": [w], "dtype": "float"}, w == "refund") for w in ("refund", "fee", "age") for i in range(6)]
    model = ClassifierAnswerer("feature.leakage_suspect", name="feature.leakage_suspect.test", version="t1").fit(rows)
    return model


def record(name, rule_says, rid=None):
    return contract.DecisionRecord(id=rid or f"dr{abs(hash(name)) % 10**12:012d}", point="feature.leakage_suspect",
                                   subject=contract.Subject(kind="column", key=name), state={"column": name, "name_tokens": [name.split("_")[0]],
                                                                                              "dtype": "float"},
                                   options=["yes", "no"], answers=[contract.Answer(by="rule", answerer="rule:column_audit", value=rule_says)],
                                   value_used=rule_says, used="rule", level=0, at="2026-10-15T00:00:00+00:00")


@unittest.skipIf(MISSING, MISSING)
class ShadowTest(unittest.TestCase):
    def setUp(self):
        self.model = leak_classifier()

    def test_a_shadow_is_recorded_beside_the_rule_and_never_used(self):
        r = shadow.add(record("refund_1", False), [self.model])
        mine = [a for a in r.answers if a.by == "classifier"]
        self.assertEqual(len(mine), 1)
        self.assertEqual((mine[0].source, mine[0].value), (SHADOW, True))
        self.assertEqual((r.used, r.value_used, r.level), ("rule", False, 0))  # level 0: the rule's answer stands
        self.assertEqual(contract.stratum(r), "disagree")  # the shadow differs: the blind draw's most drawn stratum

    def test_every_recorded_decision_carries_the_shadow_and_a_broken_one_stops_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ledger.FileLedger(Path(tmp), "w-test")
            on = mock.patch.dict(os.environ, {"DCLAB_DECISION_SHADOW": "1"})
            on.start()
            self.addCleanup(on.stop)
            with mock.patch.object(shadow, "answerers", return_value=(self.model,)):
                ledger.decide("feature.leakage_suspect", ("column", "refund_2"), answers=[ledger.answer("rule", "rule:column_audit", False)],
                              value_used=False, state={"column": "refund_2", "name_tokens": ["refund"], "dtype": "float"},
                              options=["yes", "no"], ledger=store)
            broken = mock.Mock(points=frozenset({"feature.leakage_suspect"}), by="classifier")
            broken.answer.side_effect = RuntimeError("model file damaged")
            with mock.patch.object(shadow, "answerers", return_value=(broken,)):
                ledger.decide("feature.leakage_suspect", ("column", "fee_2"), answers=[ledger.answer("rule", "rule:column_audit", False)],
                              value_used=False, state={"column": "fee_2"}, options=["yes", "no"], ledger=store)
            with mock.patch.dict(os.environ, {"DCLAB_DECISION_SHADOW": "0"}), mock.patch.object(shadow, "answerers", return_value=(self.model,)):
                ledger.decide("feature.leakage_suspect", ("column", "refund_3"), answers=[ledger.answer("rule", "rule:column_audit", False)],
                              value_used=False, state={"column": "refund_3", "name_tokens": ["refund"]}, options=["yes", "no"], ledger=store)
            got = {r.subject.key: r for r in store.records()}
        self.assertEqual(set(got), {"refund_2", "fee_2", "refund_3"})  # the broken shadow's decision is still recorded
        self.assertEqual([a.source for a in got["refund_2"].answers if a.by == "classifier"], [SHADOW])
        self.assertFalse([a for a in got["fee_2"].answers if a.by == "classifier"])
        self.assertFalse([a for a in got["refund_3"].answers if a.by == "classifier"])  # shadows turned off

    def test_agreement_disagreement_and_accuracy_on_the_labelled_disagreements(self):
        records = [record("refund_a", False, "dr000000000001"), record("refund_b", False, "dr000000000002"),
                   record("fee_a", False, "dr000000000003"), record("age_a", True, "dr000000000004")]
        labels = [contract.DecisionLabel(id=f"dl00000000000{n}", record=f"dr00000000000{n}", point="feature.leakage_suspect", value=True,
                                         source="benchmark", by="t", at="2026-10-15T00:00:00+00:00") for n in (1, 4)]
        got = shadow.measure(records, labels, [self.model], answer=True)["feature.leakage_suspect.test@t1"]
        self.assertEqual(shadow.measure(records, labels, [self.model])["feature.leakage_suspect.test@t1"]["answered"], 0)  # a page reads what is recorded
        # refund_a, refund_b: the shadow says yes, the rule no; fee_a: both no; age_a: the shadow no, the rule yes
        self.assertEqual((got["agree"], got["disagree"], got["labelled_disagreements"]), (1, 3, 2))
        self.assertEqual((got["right_on_disagreements"], got["rule_right_on_disagreements"]), (1, 1))
        self.assertEqual((got["agreement"], got["accuracy_on_disagreements"]), (0.25, 0.5))
        shadowed = [shadow.add(r, [self.model]) for r in records]  # shadows already recorded measure alike, without answering again
        self.assertEqual(shadow.measure(shadowed, labels, [self.model])["feature.leakage_suspect.test@t1"], got)
        silent = record("refund_c", None, "dr000000000005")  # the rule gave no answer: nothing to agree or differ with
        more = shadow.measure(records + [silent], labels, [self.model], answer=True)["feature.leakage_suspect.test@t1"]
        self.assertEqual((more["decisions"], more["with_rule"], more["disagree"]), (5, 4, 3))

    def test_the_blind_page_shows_disagreements_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ledger.FileLedger(Path(tmp), "w-test")
            calm = [record(f"fee_{i}", False, f"dr1000000000{i:02d}") for i in range(40)]
            loud = [shadow.add(record(f"refund_{i}", False, f"dr2000000000{i:02d}"), [self.model]) for i in range(40)]
            for r in calm + loud:
                store.add(r)
            first = ledger_labels.next_blind(store)
            drawn_disagreements = [r for r in loud if contract.blind_draw(r, "blind-v1")]
        self.assertTrue(drawn_disagreements)
        self.assertIn(first["record"], {r.id for r in drawn_disagreements})
        self.assertEqual(first["weight"], round(1 / contract.BLIND_RATES["disagree"], 6))  # the weight still says how it was drawn
        self.assertFalse({"answers", "value_used", "used"} & set(first))  # a reviewer never sees what the rule or the shadow said

    def test_a_batch_is_answered_at_once_and_a_name_alone_gets_no_shadow(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ledger.FileLedger(Path(tmp), "w-test")
            on = mock.patch.dict(os.environ, {"DCLAB_DECISION_SHADOW": "1"})
            on.start()
            self.addCleanup(on.stop)
            with mock.patch.object(shadow, "answerers", return_value=(self.model,)), mock.patch.object(shadow, "_answers", wraps=shadow._answers) as spy:
                with ledger.batch():
                    for i in range(20):
                        ledger.decide("feature.leakage_suspect", ("column", f"refund_{i}"), answers=[ledger.answer("rule", "rule:column_audit", False)],
                                      value_used=False, state={"column": f"refund_{i}", "name_tokens": ["refund"], "dtype": "float"},
                                      options=["yes", "no"], ledger=store)
                    ledger.decide("feature.leakage_suspect", ("column", "refund_x"), answers=[ledger.answer("rule", "rule:column_audit", False)],
                                  value_used=False, state={"column": "refund_x", "task": "binary_classification"}, options=["yes", "no"], ledger=store)
                self.assertEqual(spy.call_count, 1)  # one prediction for the whole batch
            got = {r.subject.key: r for r in store.records()}
        self.assertEqual(sum(1 for r in got.values() if any(a.source == SHADOW for a in r.answers)), 20)
        self.assertFalse([a for a in got["refund_x"].answers if a.source == SHADOW])  # no band: the classifier would answer from nothing

    def test_a_shadow_is_no_answer_of_the_record_for_labels_agreement_or_export(self):
        from dclab_rnd import consent

        r = shadow.add(record("refund_9", False, "dr000000000009"), [self.model])
        person = r.model_copy(update={"answers": [*r.answers, contract.Answer(by="person", answerer="solution", value=True)],
                                      "used": "person", "value_used": True})
        self.assertIsNone(person.answer("classifier"))
        self.assertEqual(person.agreement(), "unavailable")  # a shadow is not the model's answer
        # the person marks "no" right after all: an exposed label agreeing with every answer shown (the rule's no) says
        # nothing; a shadow's yes, never shown, must not make it count
        exposed = contract.DecisionLabel(id="dl000000000009", record=person.id, point=person.point, value=False, source="user_decision_exposed",
                                         by="p", at="2026-10-15T00:00:00+00:00")
        self.assertIsNone(contract.effective_label(person, [exposed]))
        with mock.patch.object(consent, "_allowed_for", return_value=True), mock.patch.object(consent, "_project_of", return_value=None):
            docs = consent.export_decisions([person], [], workspace={"train": True}, projects={})
        sources = [a.get("source") for a in docs[0]["answers"]]
        self.assertIn("shadow", sources)  # the shadow stays marked when it leaves the workspace
        self.assertEqual(sources.count(None), len(sources) - 1)  # every other source stays home

    def test_a_reviewer_sees_every_band(self):
        r = contract.DecisionRecord(id="dr000000000010", point="column.semantic_role", subject=contract.Subject(kind="column", key="signup"),
                                    state={"column": "signup", "dtype": "datetime", "value_pattern": "date_like"},
                                    options=["numeric", "categorical_code", "identifier", "datetime", "free_text", "other"],
                                    answers=[contract.Answer(by="rule", answerer="rule:prepare_roles", value="datetime")], value_used="datetime",
                                    used="rule", level=0, at="2026-10-15T00:00:00+00:00")
        self.assertEqual(contract.blind_view(r)["state"]["dtype"], "datetime")  # left out, its absence would give the answer away

    def test_a_model_file_that_is_not_the_one_built_is_never_loaded(self):
        import joblib

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            joblib.dump(self.model, folder / "feature.leakage_suspect.joblib")
            (folder / "feature.leakage_suspect.json").write_text('{"point": "feature.leakage_suspect", "version": "t1", "name": "x"}')
            from dclab_rnd.decision_models_build import record_fitted

            record_fitted(folder)
            self.assertEqual(len(shadow.answerers(str(folder))), 1)
            joblib.dump(leak_classifier(), folder / "feature.leakage_suspect.joblib")  # swapped after it was built
            (folder / "feature.leakage_suspect.joblib").write_bytes((folder / "feature.leakage_suspect.joblib").read_bytes() + b"\0")
            shadow.answerers.cache_clear()
            with mock.patch("joblib.load", wraps=joblib.load) as load:
                self.assertEqual(shadow.answerers(str(folder)), ())
            load.assert_not_called()  # refused before it was read: unpickling runs code
            shadow.answerers.cache_clear()

    def test_the_stages_decisions_carry_the_bands_a_classifier_reads(self):
        from dclab_rnd import kinds
        from dclab_rnd.expansion import runner
        from dclab_rnd.kinds import acceptance
        from dclab_rnd.models.gateway import private_scope
        from dclab_rnd.studio import engine

        with tempfile.TemporaryDirectory() as tmp:
            store = ledger.FileLedger(Path(tmp), "w-test")
            previous = ledger._INSTALLED.get("ledger")
            ledger.install(store)
            try:
                project_store, pid = acceptance.build(kinds.get("binary").case(0), Path(tmp) / "projects")
                with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
                    engine.run_all(project_store, pid)
            finally:
                ledger.install(previous)
            roles = [r for r in store.records() if r.point == "column.semantic_role"]
            leaks = [r for r in store.records() if r.point == "feature.leakage_suspect" and "apparent_lift" in r.state]
        self.assertTrue(roles and all(shadow.banded(r.state) for r in roles))
        self.assertTrue(leaks and all(shadow.banded(r.state) and r.state.get("target") for r in leaks))

    def test_a_columns_state_carries_its_value_pattern(self):
        import pandas as pd

        from dclab_rnd.studio.data import profile_table

        frame = pd.DataFrame({"order_id": [f"A{i:05d}" for i in range(50)], "zip": [f"{i:05d}" for i in range(50)], "fee": range(50)})
        states = {c["name"]: ledger.column_state(c, 50) for c in profile_table(frame)["columns"]}
        self.assertEqual((states["order_id"]["value_pattern"], states["zip"]["value_pattern"], states["fee"]["value_pattern"]),
                         ("mixed", "digits", "not_text"))
        self.assertEqual((states["fee"]["dtype"], states["zip"]["dtype"]), ("integer", "string"))  # as the classifiers were trained to read it
        contract.check_state(states["zip"])  # a band the ledger accepts


if __name__ == "__main__":
    unittest.main()
