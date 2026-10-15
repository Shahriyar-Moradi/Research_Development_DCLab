"""Package 22.9, decision models inside the workflow: where the studio decides, it asks the answerer routed for the point
and acts on its answer only after the point's validator accepts it; nothing routed, the rules decide and no model is
loaded. A model may forbid a column, never free one the audit flagged nor forbid the time that orders the split; may mark
an identifier, nothing else; may choose the next experiment among the runnable ones only; checks an agent's tool and never
replaces it. The measurement compares each routed model with the rules on the journeys, each on its own."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd

    from dclab_core import ledger as contract
    from dclab_rnd import kinds
    from dclab_rnd.ledger import shadow, use
except ImportError as error:  # pandas is not in every environment
    MISSING = str(error)
else:
    MISSING = ""


class Says:
    """A decision model that answers what it is told, per column (or one value for all)."""

    by = "classifier"

    def __init__(self, point, says, version="v1"):
        self.name, self.version, self.points, self.says, self.asked = f"fake.{point}", version, frozenset({point}), says, []

    def answer(self, point, state, options):
        self.asked.append(state.get("column"))
        value = self.says(state) if callable(self.says) else self.says
        return None if value is None else contract.Answer(by="classifier", answerer=f"{self.name}@{self.version}", value=value)


def routed(*answerers, level=3):
    return {"decisions": {next(iter(a.points)): {"answerer": f"{a.name}@{a.version}", "level": level, "bar": 0.9, "since": "2026-10-15"}
                          for a in answerers}}


@unittest.skipIf(MISSING, MISSING)
class ChooseTest(unittest.TestCase):
    def test_nothing_routed_the_rule_decides_and_no_model_is_loaded(self):
        with mock.patch.object(shadow, "answerers", side_effect=AssertionError("a model was loaded")), use.scope({"decisions": {}}):
            choice = use.choose("feature.leakage_suspect", {"column": "x"}, ["yes", "no"], False, "rule:column_audit")
        self.assertEqual((choice.value, choice.used, choice.level), (False, "rule", 0))

    def test_a_routed_answer_acts_after_its_validator_and_a_refused_one_leaves_the_rule(self):
        model = Says("feature.leakage_suspect", True)
        with mock.patch.object(shadow, "answerers", return_value=[model]), use.scope(routed(model)), use.recording() as seen:
            acted = use.choose("feature.leakage_suspect", {"column": "a"}, ["yes", "no"], False, "rule:column_audit", subject="a")
            refused = use.choose("feature.leakage_suspect", {"column": "b"}, ["yes", "no"], False, "rule:column_audit", subject="b",
                                 check=lambda v: ["never here"])
        self.assertEqual((acted.value, acted.used, acted.level), (True, "classifier", 3))
        self.assertEqual((refused.value, refused.used, refused.refused, refused.model_value), (False, "rule", ["never here"], True))
        self.assertEqual([c.subject for c in seen], ["a", "b"])

    def test_with_a_check_a_difference_waits_for_a_person(self):
        model = Says("feature.leakage_suspect", True)
        with mock.patch.object(shadow, "answerers", return_value=[model]), use.scope(routed(model, level=2)):
            choice = use.choose("feature.leakage_suspect", {"column": "a"}, ["yes", "no"], False, "rule:column_audit")
        self.assertEqual((choice.value, choice.used, choice.review), (False, "rule", True))

    def test_a_state_the_ledger_refuses_is_never_shown_to_a_model(self):
        model = Says("feature.leakage_suspect", True)
        with mock.patch.object(shadow, "answerers", return_value=[model]), use.scope(routed(model)):
            choice = use.choose("feature.leakage_suspect", {"column": "a", "rows": 12345}, ["yes", "no"], False, "rule:column_audit")
        self.assertEqual((choice.value, model.asked), (False, []))


@unittest.skipIf(MISSING, MISSING)
class WiredTest(unittest.TestCase):
    def proposal(self, model):
        from dclab_rnd.studio.data import profile_table
        from dclab_rnd.studio.solution import propose

        case = kinds.get("regression").case(0)
        with mock.patch.object(shadow, "answerers", return_value=[model]), use.scope(routed(model)):
            return propose(case.table, profile_table(case.table), case.target, "regression")

    def test_the_proposal_adds_a_models_caution_never_frees_the_audits_nor_forbids_the_time(self):
        plain = self.proposal(Says("other.point", None))
        flagged = {f["column"] for f in plain["forbidden"]}
        self.assertTrue(flagged)
        model = Says("feature.leakage_suspect", lambda state: state["column"] == "distance_km" or (False if state["column"] in flagged else
                                                                                                state["column"] == plain["time_candidates"][0]))
        got = self.proposal(model)
        forbidden = {f["column"]: f for f in got["forbidden"]}
        self.assertEqual(forbidden["distance_km"]["proof"], [f"decision-model:{model.name}@v1"])  # added, said as the model's
        self.assertTrue(flagged <= set(forbidden))  # the audit's stay
        self.assertNotIn(plain["time_candidates"][0], forbidden)  # the time that orders the split
        refused = {d["column"] for d in got["decided_by_models"] if d["refused"]}
        self.assertTrue(flagged <= refused and plain["time_candidates"][0] in refused, got["decided_by_models"])

    def test_the_engine_lets_a_model_mark_an_identifier_and_nothing_else_and_records_it(self):
        from dclab_rnd.expansion import runner
        from dclab_rnd.kinds import acceptance
        from dclab_rnd.ledger import FileLedger, install
        from dclab_rnd.studio import engine

        model = Says("column.semantic_role", lambda s: "identifier" if s["column"] == "basket_items" else "free_text")
        book = FileLedger(Path(tempfile.mkdtemp()), "wtest")
        install(book)
        self.addCleanup(install, None)
        store, pid = acceptance.build(kinds.get("regression").case(0), Path(tempfile.mkdtemp()))
        with mock.patch.object(shadow, "answerers", return_value=[model]), use.scope(routed(model)), runner.fast_profile():
            engine.execute(store, pid, "data")
        evidence = store.read_stage(pid, "data")["evidence"]
        self.assertEqual(evidence["column_roles"]["basket_items"], "identifier")
        self.assertTrue(all(v != "free_text" for v in evidence["column_roles"].values()))  # another role changes nothing
        decided = {d["column"]: d for d in evidence["decided_by_models"]}
        self.assertEqual(decided["basket_items"]["used"], "classifier")
        self.assertTrue(any(d["refused"] for d in decided.values()))
        record = next(r for r in book.records(point="column.semantic_role") if r.subject.key == "basket_items")
        self.assertEqual((record.used, record.value_used), ("classifier", "identifier"))
        self.assertIn(f"{model.name}@v1", [a.answerer for a in record.answers])

    def test_the_next_experiment_is_one_of_the_runnable_ones(self):
        from dclab_rnd.studio import engine

        p = SimpleNamespace(bundle=SimpleNamespace(X_train=pd.DataFrame({"a": range(50)})), task="binary")
        for says, expected in (("more_trees", "more_trees"), ("delete_the_holdout", "shallower")):
            model = Says("experiment.next", says)
            with mock.patch.object(shadow, "answerers", return_value=[model]), use.scope(routed(model)):
                choice = engine._model_experiment(p, ["shallower", "more_trees"], 1)
            self.assertEqual(choice.value, expected)
        with use.scope({"decisions": {}}):
            self.assertIsNone(engine._model_experiment(p, ["shallower"], 1))

    def test_an_agents_tool_is_checked_never_replaced(self):
        from dclab_rnd.ledger import FileLedger, capture, install

        book = FileLedger(Path(tempfile.mkdtemp()), "wtest")
        install(book)
        self.addCleanup(install, None)
        model = Says("agent.tool_choice", "search_evidence")
        policy, step = SimpleNamespace(name="intern", max_steps=8), SimpleNamespace(n=1, tool="profile_data")
        with mock.patch.object(shadow, "answerers", return_value=[model]), use.scope(routed(model)), use.recording() as seen:
            capture.tool_choice(policy, step, ["profile_data", "search_evidence"], None)
        record = book.records(point="agent.tool_choice")[0]
        self.assertEqual(record.value_used, "profile_data")  # the agent's
        self.assertIn("search_evidence", [a.value for a in record.answers])  # the model's, beside it
        self.assertTrue(seen[0].refused)


@unittest.skipIf(MISSING, MISSING)
class ReviewFixesTest(unittest.TestCase):
    """The review's findings, each pinned."""

    def test_a_refusal_is_in_the_session_record(self):
        from dclab_rnd import experience

        log = experience.FileLog(Path(tempfile.mkdtemp()), "wtest")
        experience.install(log)
        self.addCleanup(experience.install, None)
        model = Says("feature.leakage_suspect", True)
        with mock.patch.object(shadow, "answerers", return_value=[model]), use.scope(routed(model)):
            use.choose("feature.leakage_suspect", {"column": "a"}, ["yes", "no"], False, "rule:column_audit", check=lambda v: ["no"])
        refused = [e for e in log.read() if e.kind == "decision.refused"]
        self.assertTrue(refused and model.name in refused[0].payload["reason"], [e.kind for e in log.read()])

    def test_the_rule_beside_the_next_experiments_model_is_the_rules_own_pick_not_a_critics(self):
        from dclab_rnd.studio import engine

        p = SimpleNamespace(bundle=SimpleNamespace(X_train=pd.DataFrame({"a": range(50)})), task="binary")
        model = Says("experiment.next", "more_trees")
        with mock.patch.object(shadow, "answerers", return_value=[model]), use.scope(routed(model, level=2)):
            choice = engine._model_experiment(p, ["more_trees", "shallower"], 1, rule_first="shallower")  # a critic put more_trees first
        self.assertEqual((choice.value, choice.used, choice.review), ("shallower", "rule", True))
        self.assertEqual(choice.state["step"], 1)

    def test_a_routed_model_is_an_input_of_a_run_and_an_export_refuses_a_changed_routing(self):
        from dclab_rnd.expansion import runner
        from dclab_rnd.kinds import acceptance
        from dclab_rnd.studio import code, engine, lineage

        store, pid = acceptance.build(kinds.get("binary").case(0), Path(tempfile.mkdtemp()))
        model = Says("column.semantic_role", "numeric")
        with use.scope({"decisions": {}}):
            self.assertNotIn("decision_routing", lineage.inputs_of(store.get(pid), {}).refs)
        with mock.patch.object(shadow, "answerers", return_value=[model]), use.scope(routed(model)):
            self.assertIn("decision_routing", lineage.inputs_of(store.get(pid), {}).refs)
        with use.scope({"decisions": {}}), runner.fast_profile():
            engine.run_all(store, pid)
        with mock.patch.object(shadow, "answerers", return_value=[model]), use.scope(routed(model)), self.assertRaises(code.CodeRefused) as caught:
            code.export(store, pid, Path(tempfile.mkdtemp()))
        self.assertIn("changed since the stages ran", str(caught.exception))

    def test_a_held_difference_is_kept_for_a_person_in_the_ledger(self):
        from dclab_rnd.expansion import runner
        from dclab_rnd.kinds import acceptance
        from dclab_rnd.ledger import FileLedger, install
        from dclab_rnd.studio import engine

        book = FileLedger(Path(tempfile.mkdtemp()), "wtest")
        install(book)
        self.addCleanup(install, None)
        model = Says("column.semantic_role", lambda s: "identifier" if s["column"] == "basket_items" else None)
        store, pid = acceptance.build(kinds.get("regression").case(0), Path(tempfile.mkdtemp()))
        with mock.patch.object(shadow, "answerers", return_value=[model]), use.scope(routed(model, level=2)), runner.fast_profile():
            engine.execute(store, pid, "data")
        evidence = store.read_stage(pid, "data")["evidence"]
        self.assertNotEqual(evidence["column_roles"]["basket_items"], "identifier")  # with a check, the rule stands
        held = {d["column"]: d for d in evidence["decided_by_models"]}["basket_items"]
        self.assertEqual((held["review"], held["changed"]), (True, False))
        record = next(r for r in book.records(point="column.semantic_role") if r.subject.key == "basket_items")
        self.assertEqual((record.review, record.used), (True, "rule"))

    def test_a_model_never_forbids_a_column_the_split_may_be_built_on(self):
        from dclab_rnd.studio.data import profile_table
        from dclab_rnd.studio.solution import propose

        case = kinds.get("places").case(0)
        model = Says("feature.leakage_suspect", lambda s: s["column"] in ("lat", "lon", "rooms"))
        with mock.patch.object(shadow, "answerers", return_value=[model]), use.scope(routed(model)):
            got = propose(case.table, profile_table(case.table), case.target, "regression")
        forbidden = {f["column"] for f in got["forbidden"]}
        self.assertIn("rooms", forbidden)
        self.assertFalse({"lat", "lon"} & forbidden)

    def test_the_acceptance_runs_on_the_rules_whatever_the_workspace_routes(self):
        from dclab_rnd.kinds import acceptance
        from dclab_rnd.studio import engine

        seen = {}

        def stop(store, pid, *a, **k):
            seen["doc"] = use.routing_doc()
            raise RuntimeError("stop")

        model = Says("column.semantic_role", "identifier")
        with use.scope(routed(model)), mock.patch.object(engine, "run_all", side_effect=stop), self.assertRaises(RuntimeError):
            acceptance.engine_run(kinds.get("binary").case(0))
        self.assertEqual(seen["doc"], {"decisions": {}})

    def test_a_routing_change_is_never_served_from_the_prepare_cache(self):
        from dclab_rnd.kinds import acceptance
        from dclab_rnd.studio import engine

        store, pid = acceptance.build(kinds.get("regression").case(0), Path(tempfile.mkdtemp()))
        with use.scope({"decisions": {}}):
            self.assertEqual(engine.prepare(store, store.get(pid)).roles["basket_items"], "feature")
        model = Says("column.semantic_role", lambda s: "identifier" if s["column"] == "basket_items" else None)
        with mock.patch.object(shadow, "answerers", return_value=[model]), use.scope(routed(model)):
            self.assertEqual(engine.prepare(store, store.get(pid)).roles["basket_items"], "identifier")

    def test_an_agreement_is_no_change(self):
        agree = use.Choice("p", "c", True, "classifier", 3, rule_value=True)
        differ = use.Choice("p", "c", True, "classifier", 3, rule_value=False)
        self.assertEqual((agree.changed, differ.changed, use.Choice("p", "c", True, rule_value=False).changed), (False, True, False))


@unittest.skipIf(MISSING, MISSING)
class MeasureTest(unittest.TestCase):
    def test_each_gated_model_is_an_arm_alone_acting_and_with_a_check(self):
        from dclab_rnd.agent_eval import journeys
        from dclab_rnd.ledger import routing

        arms = journeys.arms()
        self.assertEqual(arms["rules"], {"decisions": {}})
        for name, info in routing.passed().items():
            self.assertEqual(arms[info["point"]]["decisions"][info["point"]]["level"], 3)
            self.assertEqual(arms[info["point"] + "@check"]["decisions"][info["point"]]["level"], 2)

    def test_an_arm_worse_than_the_rules_on_a_part_fails_alone(self):
        from dclab_rnd.agent_eval import journeys

        def row(i, holdout=True):
            return {"id": f"J{i}", "right": holdout, "parts": {"card": True, "workflow": True, "pipeline": True, "holdout": holdout},
                    "refused_moves": 0, "decisions": [{"review": True}], "by_model": 1}

        out = journeys.compare({"rules": [row(1), row(2)], "a": [row(1), row(2)], "b": [row(1), row(2, holdout=False)]})
        self.assertEqual((out["arms"]["a"]["ok"], out["arms"]["b"]["ok"]), (True, False))
        self.assertEqual(out["arms"]["b"]["changed"], [{"id": "J2", "rules": True, "arm": False}])
        self.assertEqual(out["verdict"], "fail: b on holdout")
        self.assertEqual(out["arms"]["a"]["held_for_review"], 2)


if __name__ == "__main__":
    unittest.main()
