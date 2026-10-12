"""Package 18.3: the data investigator (dclab_core.analyse's role) and its measurement (dclab_rnd/agent_eval/investigator).

The investigator reads column names and kinds, never values; each proposal is a test analyse runs, so it can add a
suspicion but never clear one. No model is called here.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pydantic  # noqa: F401
except ImportError as error:
    PYDANTIC_MISSING = str(error)
else:
    PYDANTIC_MISSING = ""


def demand(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    price = rng.normal(10, 2, n)
    y = np.round(50 - 2 * price + rng.normal(0, 5, n), 1)
    return pd.DataFrame({"price": price, "store_ref": rng.integers(0, 40, n), "promo": rng.integers(0, 2, n),
                         "units_returned_next_week": np.round(y * rng.uniform(0.05, 0.15, n), 2), "target": y})


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class InvestigatorTest(unittest.TestCase):
    def leaks(self, findings):
        return {f.evidence.get("column"): f for f in findings.findings if f.topic == "leakage" and f.severity != "info"}

    def test_a_proposal_becomes_a_test_and_the_test_decides(self):
        from dclab_core.agents import ScriptedAdvisor
        from dclab_core.analyse import analyse

        frame = demand()
        self.assertNotIn("units_returned_next_week", self.leaks(analyse({"t": frame}, target="target", task="regression")))
        advisor = ScriptedAdvisor({"data_investigator": {"after": [
            {"column": "units_returned_next_week", "why": "returns are counted the week after the sale"},
            {"column": "promo", "why": "a guess"},                      # tested, predicts little: an info note only
            {"column": "target", "why": "never the target"}, {"column": "ghost", "why": "not a column"}]}})
        found = analyse({"t": frame}, target="target", task="regression", advisor=advisor, moment="Each evening, for the next day.")
        leak = self.leaks(found)["units_returned_next_week"]
        self.assertEqual(leak.severity, "warning")
        self.assertEqual(leak.evidence["suspected_by"], "the data investigator")
        self.assertNotIn("promo", self.leaks(found))
        [(purpose, evidence)] = advisor.requests
        self.assertEqual(purpose, "data_investigator")
        text = json.dumps(evidence)
        self.assertIn("units_returned_next_week | number", text)
        for value in frame["price"].head(20):  # nothing but names, kinds and shares is sent
            self.assertNotIn(f"{value:.6f}"[:8], text)

    def test_it_can_add_suspicion_never_clear_it(self):
        from dclab_core.agents import ScriptedAdvisor
        from dclab_core.analyse import analyse

        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 800)
        frame = pd.DataFrame({"x": rng.normal(size=800), "refund_amount_final": y * 10 + rng.normal(0, 0.1, 800), "target": y})
        quiet = ScriptedAdvisor({"data_investigator": {"after": [], "entities": []}})
        found = analyse({"t": frame}, target="target", task="binary", advisor=quiet, moment="At checkout.")
        self.assertEqual(self.leaks(found)["refund_amount_final"].severity, "critical")

    def test_an_entity_it_names_is_checked_like_an_id(self):
        from dclab_core.agents import ScriptedAdvisor
        from dclab_core.analyse import analyse

        rng = np.random.default_rng(2)
        who = rng.integers(0, 200, 1000)
        label = rng.random(200) < 0.3
        frame = pd.DataFrame({"policy_holder": [f"E{w:05d}" for w in who], "x": rng.normal(size=1000), "target": label[who].astype(int)})
        advisor = ScriptedAdvisor({"data_investigator": {"entities": [{"column": "policy_holder", "why": "the person holding the policy"}]}})
        plain = {f.evidence.get("column") for f in analyse({"t": frame}, target="target", task="binary").findings if f.topic == "entities"}
        self.assertNotIn("policy_holder", plain)
        entities = {f.evidence.get("column"): f for f in analyse({"t": frame}, target="target", task="binary", advisor=advisor).findings
                    if f.topic == "entities"}
        self.assertEqual(entities["policy_holder"].severity, "warning")

    def test_the_rules_now_test_missingness_and_sums(self):
        from dclab_core.analyse import analyse

        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 1000)
        frame = pd.DataFrame({"survey": np.where(y == 1, np.nan, rng.normal(60, 12, 1000)), "x": rng.normal(size=1000), "target": y})
        leak = self.leaks(analyse({"t": frame}, target="target", task="binary"))["survey"]
        self.assertEqual(leak.severity, "critical")
        self.assertEqual(leak.evidence["missing_indicator_power"], 1.0)
        amount = np.round(rng.gamma(3, 10, 1000), 1)
        part = np.round(amount * rng.uniform(0.3, 0.7, 1000), 2)
        parts = pd.DataFrame({"morning": part, "afternoon": np.round(amount - part, 2), "x": rng.normal(size=1000), "target": amount})
        ids = {f.id for f in analyse({"t": parts}, target="target", task="regression").findings}
        self.assertIn("F-leak-sum-morning-afternoon", ids)
        honest = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(size=1000), "target": rng.normal(size=1000)})
        self.assertFalse(any(f.id.startswith("F-leak-sum") for f in analyse({"t": honest}, target="target", task="regression").findings))

    def test_the_plan_keeps_both_parts_of_a_sum_out_and_a_mostly_zero_column_is_no_part(self):
        from dclab_core.analyse import analyse
        from dclab_core.plan import plan, validate

        rng = np.random.default_rng(4)
        amount = np.round(rng.gamma(3, 10, 1000), 1)
        part = np.round(amount * rng.uniform(0.3, 0.7, 1000), 2)
        parts = pd.DataFrame({"morning": part, "afternoon": np.round(amount - part, 2), "x": rng.normal(size=1000), "target": amount})
        findings = analyse({"t": parts}, target="target", task="regression")
        solution = plan(None, findings)
        forbidden = {d["column"] for d in solution.decision("features.forbidden").choice}
        self.assertTrue({"morning", "afternoon"} <= forbidden)
        self.assertEqual(validate(solution, findings), [])
        tip = pd.DataFrame({"final_amount": amount, "discount": np.where(rng.random(1000) < 0.7, 0.0, rng.gamma(2, 3, 1000)),
                            "x": rng.normal(size=1000), "target": amount})
        ids = {f.id for f in analyse({"t": tip}, target="target", task="regression").findings}
        self.assertFalse(any(i.startswith("F-leak-sum") for i in ids))  # the copy is caught on its own, the discount is clean
        self.assertIn("F-leak-final_amount", ids)

    def test_proposed_entities_never_push_out_a_rule(self):
        from dclab_core.agents import ScriptedAdvisor
        from dclab_core.analyse import analyse

        rng = np.random.default_rng(5)
        who = rng.integers(0, 150, 1200)
        label = rng.random(150) < 0.3
        frame = pd.DataFrame({**{f"c{i}": rng.integers(0, 5, 1200) for i in range(6)}, "customer_id": who, "target": label[who].astype(int)})
        crowd = ScriptedAdvisor({"data_investigator": {"entities": [{"column": f"c{i}", "why": "maybe"} for i in range(6)]}})
        entities = {f.evidence.get("column"): f.severity for f in analyse({"t": frame}, target="target", task="binary", advisor=crowd).findings
                    if f.topic == "entities"}
        self.assertEqual(entities["customer_id"], "warning")

    def test_an_answer_that_is_not_lists_falls_back(self):
        from dclab_core.agents import ScriptedAdvisor, run_role
        from dclab_core.analyse import DATA_INVESTIGATOR

        inputs = {"features": demand().drop(columns="target"), "target": "target", "task": "regression", "moment": "now"}
        result = run_role(DATA_INVESTIGATOR, ScriptedAdvisor({"data_investigator": {"after": "units_returned_next_week"}}), inputs)
        self.assertEqual((result.source, result.reason), ("fallback", "invalid"))


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class InvestigatorMeasureTest(unittest.TestCase):
    def test_scoring_and_the_gate(self):
        from dclab_rnd.agent_eval.investigator import Item, score, verdict

        item = Item("x", lambda: None, "target", "binary", "now", ("late",), "who", ("honest",))
        self.assertEqual(score(item, {"late", "honest"}, {"who"}), {"traps": 2, "found": 2, "missed": [], "false_alarms": ["honest"]})
        result = {"failures": [], "summary": {"rules": {"found": 5, "false_alarms": 0}, "investigator": {"found": 6, "false_alarms": 1}}}
        self.assertEqual(verdict(result, "sealed")["checks"], {"more_found": True, "no_more_false_alarms": False})
        self.assertEqual(verdict({**result, "failures": ["x"]}, "sealed")["status"], "not scored")

    def test_the_dev_split_and_the_sealed_check(self):
        from dclab_rnd.agent_eval import benchmark, investigator

        items = investigator.load("dev")
        self.assertEqual(len(items), len(benchmark.split("dev")) + 2)
        self.assertTrue(all(i.clean for i in items))
        with mock.patch.object(benchmark, "sealed_fingerprints", return_value={}), self.assertRaises(ValueError):
            investigator.load("sealed")

    def test_a_scripted_investigator_on_dev_finds_the_unlisted_name(self):
        from dclab_core.agents import ScriptedAdvisor
        from dclab_rnd.agent_eval.investigator import load, measure

        items = [i for i in load("dev") if i.id == "BV2-DEMA-post_outcome_name"]
        advisor = ScriptedAdvisor({"data_investigator": {"after": [{"column": "units_returned_next_week", "why": "counted after the sale"}]}})
        result = measure(items, advisor)
        self.assertEqual((result["summary"]["rules"]["found"], result["summary"]["investigator"]["found"]), (0, 1))
        self.assertEqual(result["summary"]["investigator"]["false_alarms"], 0)

    def test_the_sealed_split_runs_once(self):
        from dclab_rnd.agent_eval.investigator import __main__ as cli

        with mock.patch("dotenv.load_dotenv"), mock.patch.object(cli, "sealed_runs", return_value=[Path("INV-001_sealed.json")]), \
                mock.patch("dclab_rnd.models.gateway.for_workspace") as gateway:
            self.assertEqual(cli.main(["--split", "sealed", "--yes", "--cap-eur", "1"]), 2)
        gateway.assert_not_called()
        tmp = Path(tempfile.mkdtemp())
        with mock.patch("dotenv.load_dotenv"), mock.patch("dclab_rnd.models.gateway.for_workspace"), \
                mock.patch.object(cli, "plan", return_value={"available": True, "local": False, "max_eur": 0.5}), \
                mock.patch.object(cli, "RESULTS", tmp):
            self.assertEqual(cli.main(["--split", "dev", "--yes", "--cap-eur", "0.1"]), 2)  # above the cap: refused
        self.assertEqual(list(tmp.iterdir()), [])

    def test_a_run_cut_short_is_stored_unscored(self):
        from dclab_rnd.agent_eval import investigator
        from dclab_rnd.agent_eval.investigator import __main__ as cli

        tmp = Path(tempfile.mkdtemp())
        small = [i for i in investigator.load("dev") if i.id.startswith("BV2-DEMA")][:2]
        with mock.patch("dotenv.load_dotenv"), mock.patch("dclab_rnd.models.gateway.for_workspace"), \
                mock.patch.object(cli, "load", return_value=small), \
                mock.patch.object(cli, "plan", return_value={"available": True, "local": False, "max_eur": 0.01}), \
                mock.patch.object(cli, "RESULTS", tmp), mock.patch.object(investigator, "RESULTS", tmp), \
                mock.patch("dclab_rnd.core_advisor.GatewayAdvisor.ask", side_effect=RuntimeError("the reply was cut at the token limit")), \
                mock.patch("dclab_rnd.core_advisor.GatewayAdvisor.spent_eur", return_value=0.001):
            self.assertEqual(cli.main(["--split", "sealed", "--yes", "--cap-eur", "1"]), 0)
        [stored] = list(tmp.iterdir())
        self.assertTrue(stored.name.endswith("_sealed_unscored.json"))
        record = json.loads(stored.read_text())
        self.assertNotIn("rows", record)
        self.assertEqual((record["verdict"]["status"], record["status"]), ("not scored", "failed"))
        with mock.patch.object(investigator, "RESULTS", tmp):
            self.assertEqual(investigator.sealed_runs(), [])


if __name__ == "__main__":
    unittest.main()
