"""Package 19.6: the decision ledger's contract. A column's identifier decision, a split decision and a person's
override round-trip; a state field carrying a table's value is refused; the points name the main product's governed
points and Jev's purposes; a main-product ``semantic_decision_answers`` row becomes a record (and its label) and goes
back unchanged; ``project_decision_records`` rows become records at their points."""

import json
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from pydantic import ValidationError

    from dclab_core import ledger as L
    from dclab_core.adapters import main_product_ledger as mpl
    from dclab_core.adapters.main_product import recorded
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""

T0 = "2026-10-14T08:00:00+00:00"


def identifier_record(**over):
    base = dict(id="dr-ident-1", point="column.is_identifier", subject=L.Subject(kind="column", key="order_request_id"),
                state={"column": "order_request_id", "name_tokens": ["order", "request", "id"], "dtype": "string", "uniqueness": "all",
                       "nulls": "none", "value_pattern": "mixed"},
                options=["yes", "no"], answers=[L.Answer(by="rule", answerer="rule:key_named", value=True, reason="named like a key, all unique"),
                                                L.Answer(by="classifier", answerer="identifier-gbm@1", value=True, raw=0.97, confidence=0.97)],
                value_used=True, used="rule", project="p1", at=T0)
    return L.DecisionRecord(**{**base, **over})


@unittest.skipIf(MISSING, MISSING)
class ContractTest(unittest.TestCase):
    def test_three_decisions_round_trip(self):
        ident = identifier_record()
        split = L.DecisionRecord(id="dr-split-1", point="split.strategy", subject=L.Subject(kind="project", key="p1"),
                                 state={"rows": "medium", "has_time": True, "has_groups": True, "task": "binary_classification",
                                        "optimism_gap": 0.041},
                                 options=["random", "group", "time_group"], answers=[L.Answer(by="rule", answerer="rule:split", value="time_group")],
                                 value_used="time_group", used="rule", at=T0)
        override = L.DecisionRecord(id="dr-target-1", point="target.column", subject=L.Subject(kind="draft", key="d1"),
                                    state={"labels": "file", "family": "rare_event", "columns": 14},
                                    answers=[L.Answer(by="model_role", answerer="home_agent@gpt", value="customer_id", confidence=0.6),
                                             L.Answer(by="person", answerer="owner", value="is_fraud", reason="the label file says which orders were fraud")],
                                    value_used="is_fraud", used="person", draft="d1", at=T0)
        for record in (ident, split, override):
            with self.subTest(point=record.point):
                again = L.DecisionRecord.read(record.model_dump_json())
                self.assertEqual(again, record)
        self.assertEqual((ident.kind, split.kind, override.kind), ("technical", "technical", "technical"))
        self.assertEqual(ident.agreement(), "agree")
        self.assertEqual(override.agreement(), "unavailable")  # a person overrode a model: no rule to compare

    def test_a_v1_record_reads_as_v2_at_operate_and_the_stamp_is_kept(self):
        import json

        record = identifier_record(consent="train")
        self.assertEqual(record.consent, "train")
        doc = json.loads(record.model_dump_json())
        old = {k: v for k, v in doc.items() if k != "consent"} | {"schema_version": "dclab_core.decision_record.v1"}
        self.assertEqual((L.DecisionRecord.read(old).schema_version, L.DecisionRecord.read(old).consent), ("dclab_core.decision_record.v2", "operate"))
        self.assertEqual(L.DecisionRecord.read(doc).consent, "train")

    def test_a_table_value_never_enters_a_state(self):
        refused = [{"sample_value": "Jane Roe"},              # not a state field
                   {"dtype": "Jane Roe"},                     # not one of its bands
                   {"column": "x" * 300},                     # too long to be a name
                   {"column": "first line\nJane Roe"},        # not a name
                   {"name_tokens": ["jane roe"]},             # a token is one lower-case word
                   {"uniqueness": 0.53},                      # a band, never the number
                   {"columns": -1}, {"share": 1.5}, {"has_time": "yes"}, {"move": "Run Stage!"}]
        for state in refused:
            with self.subTest(state=state), self.assertRaises(ValidationError):
                identifier_record(state=state)

    def test_points_subjects_and_answers_are_checked(self):
        with self.assertRaises(ValidationError):
            identifier_record(point="column.colour")
        with self.assertRaises(ValidationError):  # an identifier decision is about a column
            identifier_record(subject=L.Subject(kind="project", key="p1"))
        with self.assertRaises(ValidationError):  # the value came from someone who gave no answer
            identifier_record(used="person")
        self.assertTrue({p.kind for p in L.POINTS.values()} == {"technical", "workflow"})
        workflow = [k for k, p in L.POINTS.items() if p.kind == "workflow"]
        for key in ("problem.family", "workflow.next_question", "workflow.move", "agent.tool_choice", "experiment.next", "workflow.stop"):
            self.assertIn(key, workflow)

    def test_labels_supersede_and_a_blind_one_carries_its_weight(self):
        planted = L.DecisionLabel(id="dl-0001", record="dr-ident-1", point="column.is_identifier", value=True, source="benchmark",
                                  by="replica:delivery_fraud", at=T0, evidence=L.Evidence(kind="planted_truth", ref="delivery_fraud:order_request_id"))
        blind = L.DecisionLabel(id="dl-0002", record="dr-ident-1", point="column.is_identifier", value=True, source="blind", by="reviewer",
                                at=T0, supersedes="dl-0001", weight=4.0)
        self.assertEqual(L.DecisionLabel.read(blind.model_dump_json()), blind)
        with self.assertRaises(ValidationError):  # a blind label without its sampling weight
            L.DecisionLabel(**{**blind.model_dump(), "weight": None})
        with self.assertRaises(ValidationError):  # only a blind label is weighted
            L.DecisionLabel(**{**planted.model_dump(), "weight": 2.0})
        with self.assertRaises(ValidationError):
            L.DecisionLabel(**{**blind.model_dump(), "supersedes": "dl-0002"})

    def test_the_points_name_the_main_products_and_jevs(self):
        record = recorded()
        governed = {p for entry in record["decision_points"]["points"] for p in ([entry] if isinstance(entry, str) else [entry.get("key")])}
        names = {p.main_product for p in L.POINTS.values() if p.main_product}
        self.assertEqual(governed - names, set(), "every governed point of the main product has a ledger point")
        for key, point in L.POINTS.items():
            if point.main_product:
                with self.subTest(point=key):
                    self.assertTrue(point.main_product in governed or point.main_product in record["experience_vocabularies"]["jev.purposes"],
                                    point.main_product)
        self.assertEqual({p.jev for p in L.POINTS.values() if p.jev}, set(record["experience_vocabularies"]["jev.purposes"]))
        bands = record["experience_vocabularies"]["jev.bands"]
        self.assertEqual((list(L.DTYPE_KINDS), list(L.RATIO_BANDS), list(L.CARDINALITY_BANDS), list(L.VALUE_PATTERNS), list(L.TASKS),
                          list(L.ROLES), list(L.AVAILABILITY), list(L.COUNT_BANDS)),
                         (bands["dtype"], bands["ratio"], bands["cardinality"], bands["value_pattern"], bands["task"], bands["role"],
                          bands["availability"], bands["count"]))
        self.assertEqual(mpl.JEV_BANDS, record["experience_vocabularies"]["jev.acting_bands"])


def uid():
    return str(uuid.uuid4())


@unittest.skipIf(MISSING, MISSING)
class MainProductTest(unittest.TestCase):
    def setUp(self):
        self.columns = recorded()["experience_tables"]

    def answer_row(self, **values):
        return {c: None for c in self.columns["semantic_decision_answers"]} | {
            "id": uid(), "workspace_id": uid(), "project_id": uid(), "llm_invocation_id": uid(), "purpose": "column.is_identifier",
            "decision_point_key": "column.is_identifier", "release_version": "1", "model_id": "jev-1.13.0", "question_key": "customer_id",
            "question_digest": "a" * 64, "data_class": "metadata", "evidence_partition": "metadata", "primitive": "noul",
            "answer": {"value": 0.96}, "confidence": 0.96, "in_acting_band": True, "rule_answer": {"value": False}, "agreement": "disagree",
            "level": 1, "policy_outcome": "rule", "value_used": {"value": False}, "created_at": T0, "labels_version": 0} | values

    def test_a_jev_answer_becomes_a_record_and_goes_back(self):
        row = self.answer_row()
        record, label = mpl.from_semantic_answer(row)
        self.assertIsNone(label)
        self.assertEqual((record.point, record.subject.kind, record.subject.key, record.used, record.level), ("column.is_identifier", "column", "customer_id", "rule", 1))
        row["workspace_id"] = record.workspace
        self.assertEqual(record.answer("model_role").value, True)        # 0.96 is past the purpose's yes band (0.90)
        self.assertEqual(record.agreement(), row["agreement"])
        back = mpl.to_semantic_answer(record, row["model_id"], row["release_version"], row["question_digest"])
        for column in ("purpose", "decision_point_key", "question_key", "answer", "confidence", "rule_answer", "agreement", "level",
                       "policy_outcome", "value_used", "in_acting_band", "primitive"):
            with self.subTest(column=column):
                self.assertEqual(back[column], row[column])
        self.assertTrue(set(back) <= set(self.columns["semantic_decision_answers"]))

    def test_an_answer_outside_its_band_abstains(self):
        record, _ = mpl.from_semantic_answer(self.answer_row(answer={"value": 0.5}, confidence=0.5, agreement="abstain", in_acting_band=False))
        self.assertIsNone(record.answer("model_role").value)
        self.assertEqual(record.answer("model_role").raw, 0.5)
        self.assertEqual(record.agreement(), "abstain")
        role, _ = mpl.from_semantic_answer(self.answer_row(purpose="column.semantic_role", decision_point_key="column.semantic_role", primitive="choice",
                                                           answer={"value": "identifier"}, confidence=0.7, rule_answer={"value": "numeric"},
                                                           value_used={"value": "numeric"}))
        self.assertIsNone(role.answer("model_role").value)  # 0.7 is below the choice's 0.80

    def test_every_row_shape_goes_back_unchanged(self):
        shapes = {
            "leakage, rule exclude, agree": dict(purpose="feature.leakage_suspect", decision_point_key="feature.leakage_suspect", evidence_partition="train",
                                                 rule_answer={"value": "exclude"}, value_used={"value": "exclude"}, agreement="agree"),
            "unavailable": dict(answer={}, confidence=None, in_acting_band=False, agreement="unavailable", value_used={"value": False}),
            "score": dict(purpose="proposal.completeness", decision_point_key="proposal.completeness", primitive="score", answer={"value": 3},
                          confidence=None, in_acting_band=False, rule_answer={"value": 2}, agreement="abstain", value_used={"value": 2},
                          question_key="proposal"),
            "review": dict(policy_outcome="review", agreement="disagree"),
            "ai at level 2": dict(policy_outcome="ai", level=2, value_used={"value": True}, agreement="disagree"),
        }
        for name, values in shapes.items():
            with self.subTest(shape=name):
                row = self.answer_row(**values)
                record, _ = mpl.from_semantic_answer(row)
                self.assertEqual(record.agreement(), row["agreement"])
                back = mpl.to_semantic_answer(record, row["model_id"], row["release_version"], row["question_digest"])
                for column in ("answer", "rule_answer", "agreement", "policy_outcome", "value_used", "in_acting_band", "evidence_partition", "level"):
                    self.assertEqual(back[column], row[column], column)

    def test_a_decision_model_trained_here_answers_as_ai(self):
        record = identifier_record(answers=[L.Answer(by="rule", answerer="rule:key_named", value=False),
                                            L.Answer(by="classifier", answerer="identifier-gbm@1", value=True, raw=0.97, confidence=0.97)],
                                   value_used=True, used="classifier", level=2, workspace=uid())
        back = mpl.to_semantic_answer(record, "dclab-identifier-gbm", "1", "b" * 64)
        self.assertEqual((back["policy_outcome"], back["agreement"], back["answer"], back["value_used"]), ("ai", "disagree", {"value": 0.97}, {"value": True}))
        with self.assertRaises(ValueError):  # a row needs its workspace
            mpl.to_semantic_answer(identifier_record(), "m", "1", "b" * 64)
        with self.assertRaises(ValidationError):  # a model's answer is used only from level 2
            identifier_record(used="classifier", level=0)

    def test_a_label_row_labels_its_decision_and_supersedes_only_labels(self):
        answer = self.answer_row()
        first = self.answer_row(label_source="benchmark", ground_truth={"value": True}, labeled_at=T0, labels_version=1, supersedes_label_id=answer["id"])
        record, label = mpl.from_semantic_answer(first)
        self.assertIsNone(record)  # a label row is not a second decision
        self.assertEqual((label.source, label.value, label.record, label.supersedes), ("benchmark", True, f"mp-sda-{answer['id']}", None))
        second = self.answer_row(label_source="blind", ground_truth={"value": False}, labeled_at=T0, labels_version=2, supersedes_label_id=first["id"],
                                 labeled_by_user_id=uid())
        with self.assertRaises(ValueError):  # without its chain the decision it labels is unknown
            mpl.from_semantic_answer(second)
        _, blind = mpl.from_semantic_answer(second, chain={first["id"]: first, answer["id"]: answer})
        self.assertEqual((blind.record, blind.supersedes, blind.weight), (f"mp-sda-{answer['id']}", f"mp-sdl-{first['id']}", None))
        self.assertIn("sampling probability", blind.weight_note)  # never an invented weight of 1

    def test_answers_and_options_hold_no_row_text(self):
        role = L.DecisionRecord(id="dr-role-01", point="column.semantic_role", subject=L.Subject(kind="column", key="note"), at=T0,
                                answers=[L.Answer(by="rule", answerer="rule:kind", value="free_text")], value_used="free_text", used="rule")
        for change in ({"options": ["Jane Roe"]}, {"value_used": "Jane Roe"}, {"options": ["x" * 300]},
                       {"answers": [L.Answer(by="rule", answerer="rule:kind", value="Jane Roe lives at 12 Main St")], "value_used": None}):
            with self.subTest(change=list(change)), self.assertRaises(ValidationError):
                L.DecisionRecord(**{**role.model_dump(), **change})
        with self.assertRaises(ValidationError):
            identifier_record(value_used="yes please")  # a yes/no point answers a flag
        with self.assertRaises(ValidationError):
            L.DecisionLabel(id="dl-0009", record="dr-role-01", point="target.column", value="line one\nline two", source="benchmark", by="x", at=T0)

    def test_a_renamed_column_fails_loudly(self):
        with self.assertRaises(ValueError):
            mpl.from_semantic_answer(self.answer_row(answer_json={"value": 1}))

    def test_project_decisions_take_their_value_from_their_facts(self):
        base = {c: None for c in self.columns["project_decision_records"]} | {"id": uid(), "workspace_id": uid(), "project_id": uid(),
                                                                               "event_at": T0, "state": "accepted"}
        split = mpl.from_decision_record(base | {"decision_type": "split_plan_created", "actor_kind": "rule", "actor_rule": "holdout_planner.v3",
                                                 "facts": {"holdout_strategy": "group_disjoint", "validation_strategy": "group_kfold"},
                                                 "details": {"plan_digest": "d"}})
        self.assertEqual((split.point, split.value_used, split.used, split.answers[0].answerer), ("split.strategy", "group_disjoint", "rule", "holdout_planner.v3"))
        point = mpl.from_decision_record(base | {"decision_type": "operating_point_chosen", "actor_kind": "human", "actor_user_id": uid(),
                                                 "facts": {"threshold": 0.42, "precision": 0.8}})
        self.assertEqual((point.value_used, point.used), (0.42, "person"))
        winner = mpl.from_decision_record(base | {"decision_type": "winner_locked", "actor_kind": "rule", "candidate_id": "c-7",
                                                  "facts": {"selection_metric": "roc_auc"}})
        self.assertEqual(winner.value_used, "candidate:c-7")
        unknown = mpl.from_decision_record(base | {"decision_type": "split_plan_created", "actor_kind": "rule", "facts": {}})
        self.assertIsNone(unknown.value_used)  # never the row's state
        rejected = mpl.from_decision_record(base | {"decision_type": "champion_promoted", "actor_kind": "human", "state": "rejected"})
        self.assertEqual((rejected.point, rejected.value_used, rejected.used), ("model.promote", False, "person"))
        proposed = mpl.from_decision_record(base | {"decision_type": "champion_promoted", "actor_kind": "agent", "state": "proposed",
                                                    "rationale": "trust me", "rationale_untrusted": True})
        self.assertEqual((proposed.used, proposed.value_used, proposed.answers[0].value, proposed.answers[0].reason), ("none", None, True, None))
        vocab = set(recorded()["experience_vocabularies"]["project_decision_records.decision_type"])
        self.assertEqual(set(mpl.DECISION_TYPE_POINTS) | set(mpl.DECISION_TYPES_NOT_LEDGERED), vocab)
        self.assertIsNone(mpl.from_decision_record(base | {"decision_type": "ref_moved", "actor_kind": "rule"}))


if __name__ == "__main__":
    unittest.main()
