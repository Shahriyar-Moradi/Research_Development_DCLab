"""Package 17.4: the solution plan (dclab_core.plan): decisions with reasons and alternatives, a validator, overrides.

On each replica the plan must reach the decisions a careful data scientist reaches (the acceptance decisions), every
decision must cite findings that exist, the validator must refuse a plan that ignores a critical finding, and a person's
override must be kept with the old choice as an alternative.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import pydantic  # noqa: F401
except ImportError as error:
    PYDANTIC_MISSING = str(error)
else:
    PYDANTIC_MISSING = ""


def _cases():
    import test_core_understand as briefs

    from dclab_core.analyse import analyse
    from dclab_core.replicas import delivery_fraud, delivery_time, ticket_routing
    from dclab_core.understand import understand

    out = {}
    t = ticket_routing.generate(seed=0).tickets
    out["tickets"] = (understand(briefs.TICKETS, briefs.TICKETS_FILES),
                      analyse({"tickets.csv": t}, target="assigned_team", train_index=t.index[: int(len(t) * 0.8)]))
    d = delivery_time.generate(seed=0).orders
    out["delivery"] = (understand(briefs.DELIVERY, briefs.DELIVERY_FILES),
                       analyse({"orders.csv": d}, target="actual_minutes", train_index=d.index[: int(len(d) * 0.8)]))
    r = delivery_fraud.generate(seed=0)
    rides = r.rides[(r.rides["status"] == "DELIVERED") & r.rides["biker_id"].notna()].copy()
    rides["is_fraud"] = rides["order_id"].isin(set(r.sample["order_id"])).astype(int)
    brief = understand(briefs.FRAUD, {"rides.csv": list(r.rides.columns), "known_frauds.csv": list(r.sample.columns)})
    out["fraud"] = (brief, analyse({"rides.csv": rides, "known_frauds.csv": r.sample}, target="is_fraud", brief=brief,
                                   labels="known_frauds.csv", train_index=rides.index[: int(len(rides) * 0.8)]))
    return out


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class PlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_core.plan import plan

        cls.cases = _cases()
        cls.plans = {name: plan(brief, findings) for name, (brief, findings) in cls.cases.items()}

    def choice(self, case, key):
        decision = self.plans[case].decision(key)
        self.assertIsNotNone(decision, f"{case} has no {key}")
        return decision.choice

    def test_every_plan_is_valid_and_every_decision_cites_real_findings(self):
        from dclab_core.plan import validate

        for name, plan_ in self.plans.items():
            with self.subTest(case=name):
                self.assertEqual(validate(plan_, self.cases[name][1]), [])
                for decision in plan_.decisions:
                    self.assertGreaterEqual(len(decision.reason), 10)

    def test_the_fraud_decisions(self):
        self.assertEqual(self.choice("fraud", "labels.coverage"), "partial")
        self.assertEqual(self.choice("fraud", "labels.join"), {"key": "order_id"})
        refused = [a.choice for a in self.plans["fraud"].decision("labels.join").alternatives]
        self.assertIn("account_id+biker_id", refused)
        self.assertEqual(self.choice("fraud", "split.strategy")["group_column"], "biker_id")
        self.assertEqual(self.choice("fraud", "metric.primary"), "average_precision")
        self.assertIn("recall_at_budget", self.choice("fraud", "metric.secondary"))
        self.assertTrue({"pu_bagging", "isolation_forest"} <= set(self.choice("fraud", "model.families")))
        self.assertEqual(self.choice("fraud", "features.forbidden"), [])  # nothing written after the moment was planted

    def test_the_ticket_decisions(self):
        split = self.choice("tickets", "split.strategy")
        self.assertEqual((split["kind"], split["time_column"]), ("time", "created_at"))  # a class appears late; customers
        # repeat but do not tell the team out of fold, so grouping by them is not what the evidence asks for
        self.assertEqual(self.choice("tickets", "metric.primary"), "macro_f1")
        forbidden = {x["column"] for x in self.choice("tickets", "features.forbidden")}
        self.assertEqual(forbidden, {"resolved_by"})
        confirm = {x["column"] for x in self.choice("tickets", "features.to_confirm")}
        self.assertEqual(confirm, {"first_response_minutes"})  # sounds post-outcome, predicts little: ask, and keep out until then
        self.assertIn("tiny_classes", self.choice("tickets", "task.framing"))
        self.assertTrue(any("copies" in step for step in self.choice("tickets", "data.cleaning")))

    def test_the_delivery_decisions(self):
        self.assertEqual(self.choice("delivery", "split.strategy")["group_column"], "restaurant_id")
        self.assertEqual(self.choice("delivery", "metric.primary"), "mae")
        self.assertIn("share_within_tolerance", self.choice("delivery", "metric.secondary"))  # "within 5 minutes for 80%"
        self.assertIn("cap", self.choice("delivery", "task.framing"))
        forbidden = {x["column"] for x in self.choice("delivery", "features.forbidden")}
        self.assertEqual(forbidden, {"delivered_at", "courier_rating"})
        self.assertIn("quantile_gradient_boosting", self.choice("delivery", "model.families"))

    def test_a_metric_the_brief_names_leads(self):
        from dclab_core.plan import plan
        from dclab_core.understand import understand

        brief, findings = self.cases["tickets"]
        named = understand("Route each ticket to one of 12 teams at the moment it is created; it will be judged on accuracy.")
        primary = plan(named, findings).decision("metric.primary")
        self.assertEqual(primary.choice, "accuracy")
        self.assertIn("macro_f1", [a.choice for a in primary.alternatives])

    def test_the_validator_refuses_a_plan_that_ignores_the_findings(self):
        from dclab_core.plan import override, validate

        findings = self.cases["tickets"][1]
        bad = override(self.plans["tickets"], "features.forbidden", [], "keep every column in the model", by="advisor")
        bad = bad.model_validate({**bad.model_dump(), "decisions": [
            {**d.model_dump(), "status": "proposed"} if d.key == "features.forbidden" else d.model_dump() for d in bad.decisions]})
        self.assertTrue(any("F-leak-resolved_by" in p for p in validate(bad, findings)))
        random = override(self.plans["fraud"], "split.strategy", {"kind": "stratified"}, "a random split is faster", by="advisor")
        random = random.model_validate({**random.model_dump(), "decisions": [
            {**d.model_dump(), "status": "proposed"} if d.key == "split.strategy" else d.model_dump() for d in random.decisions]})
        self.assertTrue(any("entity" in p for p in validate(random, self.cases["fraud"][1])))
        wrong_metric = override(self.plans["delivery"], "metric.primary", "macro_f1", "a mistake in the metric")
        self.assertTrue(any("not a regression metric" in p for p in validate(wrong_metric, self.cases["delivery"][1])))

    def test_a_person_overrides_with_a_reason_and_the_old_choice_stays(self):
        from dclab_core.plan import override, validate

        new = override(self.plans["fraud"], "split.strategy", {"kind": "time", "time_column": "created_date"},
                       "The fraud team retrains weekly; we want a forward check first.")
        decision = new.decision("split.strategy")
        self.assertEqual((decision.status, decision.source), ("overridden", "person"))
        self.assertEqual(decision.alternatives[0].choice["group_column"], "biker_id")
        self.assertIn("overridden by person", decision.alternatives[0].why_not)
        self.assertEqual(validate(new, self.cases["fraud"][1]), [])  # an override with a reason is the person's call

    def test_the_plan_is_a_valid_contract(self):
        from dclab_core.contracts import SolutionPlan

        for plan_ in self.plans.values():
            self.assertEqual(SolutionPlan.read(plan_.model_dump_json()), plan_)


    def test_the_studio_reads_a_plan_as_a_valid_solution(self):
        from dclab_rnd.core_adapter import solution_fields_from_plan
        from dclab_rnd.studio.solution import Solution

        fields = solution_fields_from_plan(self.plans["delivery"], "actual_minutes", "When the customer places the order.")
        solution = Solution(**fields)
        self.assertEqual(solution.group_column, "restaurant_id")
        self.assertEqual({f.column for f in solution.forbidden}, {"delivered_at", "courier_rating"})
        self.assertEqual(solution.metric, "mae")
        tickets = Solution(**solution_fields_from_plan(self.plans["tickets"], "assigned_team", "When the ticket is created."))
        self.assertEqual(tickets.time_column, "created_at")
        self.assertEqual({f.column for f in tickets.forbidden}, {"resolved_by", "first_response_minutes"})  # confirm-first stays out


    def test_every_decision_cites_a_finding_except_an_empty_list_of_columns_to_keep_out(self):
        for name, plan_ in self.plans.items():
            for decision in plan_.decisions:
                if decision.key == "features.forbidden" and decision.choice == []:
                    continue
                with self.subTest(case=name, key=decision.key):
                    self.assertTrue(decision.cites)

    def test_without_a_brief_the_fraud_plan_groups_by_the_entity_that_tells_the_target(self):
        from dclab_core.plan import plan

        split = plan(None, self.cases["fraud"][1]).decision("split.strategy")
        self.assertEqual(split.choice["group_column"], "biker_id")
        self.assertTrue(any("account_id" in str(a.choice) and "too few" in a.why_not for a in split.alternatives))

    def test_a_named_metric_alias_is_understood_and_an_unknown_one_is_recorded(self):
        from dclab_core.plan import plan
        from dclab_core.understand import understand

        findings = self.cases["fraud"][1]
        auc = plan(understand("Detect fraudulent rides; the work is judged on AUC."), findings).decision("metric.primary")
        self.assertEqual(auc.choice, "roc_auc")
        odd = plan(understand("Detect fraudulent rides; it is measured by mape."), findings).decision("metric.primary")
        self.assertTrue(any(a.choice == "mape" for a in odd.alternatives))


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class ValidatorHolesTest(unittest.TestCase):
    """What the review found: a model must not be able to talk its way past the validator, and odd shapes are problems."""

    @classmethod
    def setUpClass(cls):
        from dclab_core.plan import plan

        cls.cases = _cases()
        cls.plans = {name: plan(brief, findings) for name, (brief, findings) in cls.cases.items()}

    def test_a_models_override_is_still_checked_and_a_persons_stands(self):
        from dclab_core.plan import override, validate

        fraud = self.cases["fraud"][1]
        by_model = override(self.plans["fraud"], "split.strategy", {"kind": "stratified"}, "a random split is quicker", by="advisor")
        self.assertTrue(any("entity" in p for p in validate(by_model, fraud)))
        by_alice = override(self.plans["fraud"], "split.strategy", {"kind": "stratified"}, "we accept the risk, for a quick check",
                            by="alice@example.com")
        self.assertEqual(by_alice.decision("split.strategy").source, "person")
        self.assertEqual(validate(by_alice, fraud), [])
        leaky = override(self.plans["tickets"], "features.forbidden", [], "keep every column in the model", by="advisor")
        self.assertTrue(any("F-leak-resolved_by" in p for p in validate(leaky, self.cases["tickets"][1])))

    def test_odd_shapes_are_problems_never_crashes(self):
        from dclab_core.plan import override, validate

        fraud = self.cases["fraud"][1]
        as_text = override(self.plans["fraud"], "split.strategy", "stratified", "sent as a plain string", by="advisor")
        self.assertTrue(any("entity" in p for p in validate(as_text, fraud)))
        names = override(self.plans["tickets"], "features.forbidden", ["resolved_by"], "names only, no reasons", by="advisor")
        self.assertEqual(validate(names, self.cases["tickets"][1]), [])
        nonsense = override(self.plans["fraud"], "split.strategy", 42, "a number for a split", by="advisor")
        self.assertTrue(any("must name a kind" in p for p in validate(nonsense, fraud)))

    def test_an_inflating_label_key_is_refused(self):
        from dclab_core.plan import override, validate

        bad = override(self.plans["fraud"], "labels.join", {"key": "biker_id"}, "the biker is the case", by="advisor")
        self.assertTrue(any("inflates" in p for p in validate(bad, self.cases["fraud"][1])))

    def test_many_overrides_and_many_findings_never_overflow_a_decision(self):
        from dclab_core.plan import override

        plan_ = self.plans["fraud"]
        for i in range(12):
            plan_ = override(plan_, "metric.primary", "roc_auc" if i % 2 else "average_precision", f"round {i} of changes")
        self.assertLessEqual(len(plan_.decision("metric.primary").alternatives), 8)

    def test_wide_tables_with_many_findings_still_plan(self):
        import numpy as np
        import pandas as pd

        from dclab_core.analyse import analyse
        from dclab_core.plan import plan

        rng = np.random.default_rng(0)
        y = rng.choice([0, 1], 3000, p=[0.9, 0.1])
        df = pd.DataFrame({f"m{i}": np.where((y == 1) & (rng.random(3000) < 0.7), np.nan, rng.normal(size=3000)) for i in range(25)})
        df["y"] = y
        result = plan(None, analyse({"t": df}, target="y", train_index=df.index[:2400]))
        self.assertLessEqual(len(result.decision("features.direction").cites), 20)

    def test_the_split_never_orders_by_a_column_it_keeps_out(self):
        import numpy as np
        import pandas as pd

        from dclab_core.analyse import analyse
        from dclab_core.plan import plan
        from dclab_rnd.core_adapter import solution_fields_from_plan
        from dclab_rnd.studio.solution import Solution

        rng = np.random.default_rng(0)
        n = 3000
        df = pd.DataFrame({"completed_at": pd.date_range("2025-01-01", periods=n, freq="h"), "x": rng.normal(size=n),
                           "y": np.r_[np.zeros(2000), (rng.random(1000) < 0.2).astype(int)]})
        p = plan(None, analyse({"t": df}, target="y", train_index=df.index[:2400]))
        self.assertNotEqual((p.decision("split.strategy").choice or {}).get("time_column"), "completed_at")
        Solution(**solution_fields_from_plan(p, "y", "When the order is placed, before it is completed."))

    def test_a_plan_without_a_training_split_says_so(self):
        from dclab_core.analyse import analyse
        from dclab_core.plan import plan
        from dclab_core.replicas import delivery_time

        frame = delivery_time.generate(seed=0, n=1500).orders
        p = plan(None, analyse({"o": frame}, target="actual_minutes"))
        self.assertIn("F-no-split", p.decision("split.strategy").cites)
        self.assertTrue(any("F-no-split" in r.cites for r in p.risks))


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class AdoptPlanTest(unittest.TestCase):
    """The plan reaches the engine through the governed move, and a model's change that ignores a finding is refused."""

    def setUp(self):
        import tempfile

        from dclab_core.replicas import ticket_routing
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.studio import ProjectStore, data as sd

        self.store = ProjectStore(Path(tempfile.mkdtemp()))
        project = self.store.create("Tickets", "support", "Route each ticket to one of 12 teams when it is created.")
        frame = ticket_routing.generate(seed=0, n=1500).tickets
        path = self.store.data_dir(project["id"]) / "tickets.csv"
        frame.to_csv(path, index=False)
        sd.attach_data(self.store, project["id"], "tickets.csv")
        self.pid = project["id"]
        self.box = Toolbox(self.store)

    def test_adopting_the_plan_saves_the_solution_and_logs_the_move(self):
        out = self.box.adopt_plan(self.pid, "assigned_team", "When the ticket is created, from its subject, body and plan.")
        self.assertTrue(out.get("saved"), out)
        project = self.store.get(self.pid)
        self.assertIn("resolved_by", {f["column"] for f in project["solution"]["forbidden"]})
        self.assertEqual(project["plan"]["schema_version"], "dclab_core.solution_plan.v1")

    def test_a_models_change_that_ignores_a_critical_finding_is_refused(self):
        out = self.box.adopt_plan(self.pid, "assigned_team", "When the ticket is created, from its subject, body and plan.",
                                  overrides=[{"key": "features.forbidden", "choice": [], "reason": "keep every column in"}])
        self.assertIn("not adopted", out["error"])
        self.assertIsNone(self.store.get(self.pid).get("solution"))


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class StudioProposalTest(unittest.TestCase):
    def test_the_proposal_carries_the_plan_and_keeps_its_own_fields(self):
        from dclab_core.replicas import delivery_time
        from dclab_rnd.studio import data as sd
        from dclab_rnd.studio.solution import propose

        frame = delivery_time.generate(seed=0, n=2000).orders
        frame["order_time"] = frame["order_time"].astype(str)
        frame["delivered_at"] = frame["delivered_at"].astype(str)
        proposal = propose(frame, sd.profile_table(frame), "actual_minutes", "regression")
        decisions = {d["decision"]: d for d in proposal["plan"]}
        self.assertEqual(decisions["metric.primary"]["choice"], "mae")
        self.assertIn("courier_rating", {f["column"] for f in decisions["features.forbidden"]["choice"]})
        self.assertIn("forbidden", proposal)  # the studio's own audit stays as it was
        self.assertLess(len(str(proposal["plan"])), 8000)  # compact: the intern reads it on every proposal


if __name__ == "__main__":
    unittest.main()
