"""Package 18.9: the agentic layer judged. The judged table is generated from the recorded results; a role is on by
default only where its gate passed; the owner can override; a role gets an advisor only when it is on, a model serves
it and the data is not private; the product's flows ask the roles that are on; the whole layer is compared on the
sealed benchmark with a gate fixed before its runs."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core.agents import ScriptedAdvisor
    from dclab_rnd import roles
    from dclab_rnd.agent_eval import judged, layer
except ImportError as error:  # pydantic is not part of requirements/ci.txt
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class JudgedTest(unittest.TestCase):
    def test_the_table_is_generated_and_a_role_is_on_only_where_its_gate_passed(self):
        doc = judged.judge()
        self.assertEqual(json.loads(judged.OUT.read_text(encoding="utf-8")), json.loads(json.dumps(doc)))  # make roles-judged
        rows = {r["role"]: r for r in doc["roles"]}
        layer_passed = rows["layer"]["verdict"] == "pass"
        for role, row in rows.items():
            expected = row["verdict"] == "pass" and role not in ("team", "layer") and (role not in judged.PIPELINE or layer_passed)
            self.assertEqual(row["on"], expected, role)
        if not layer_passed:  # the analyst, investigator and planner wait for the whole layer's comparison
            self.assertFalse(rows["problem_analyst"]["on"])
            self.assertIn("layer", rows["problem_analyst"]["waits_for"])
        self.assertFalse(rows["critic"]["on"])  # failed its gate (18.6)
        self.assertFalse(rows["brief_checker"]["on"])  # failed its gate (18.2's follow-up)
        self.assertTrue(rows["reporter"]["on"])
        self.assertEqual(rows["data_investigator"]["comparison"]["investigator"]["found"]["hits"], 35)
        self.assertIsNotNone(rows["data_investigator"]["comparison"]["investigator"]["found"]["ci95"])
        self.assertGreater(rows["solution_planner"]["eur_per_solved"], 0)

    def test_the_owner_can_switch_roles_on_or_off(self):
        with mock.patch.dict(os.environ, {"DCLAB_ROLES": "none"}):
            self.assertFalse(roles.on("problem_analyst"))
        with mock.patch.dict(os.environ, {"DCLAB_ROLES": "critic"}):
            self.assertTrue(roles.on("critic"))
            self.assertFalse(roles.on("reporter"))
        with mock.patch.dict(os.environ, {"DCLAB_ROLES": "judged"}):
            self.assertTrue(roles.on("reporter"))
            self.assertFalse(roles.on("critic"))

    def test_an_advisor_only_when_on_served_and_not_private(self):
        class Gateway:
            def __init__(self, available):
                self._available = available

            def available(self, purpose):
                return self._available

        with mock.patch.dict(os.environ, {"DCLAB_ROLES": "judged"}), mock.patch("dclab_rnd.models.installed", return_value=Gateway(True)):
            self.assertIsNotNone(roles.advisor("reporter", project={"id": "p", "settings": {}}))
            self.assertIsNone(roles.advisor("critic", project={"id": "p", "settings": {}}))  # off: its gate failed
            self.assertIsNone(roles.advisor("reporter", project={"id": "p", "settings": {"private": True}}))
        with mock.patch.dict(os.environ, {"DCLAB_ROLES": "judged"}), mock.patch("dclab_rnd.models.installed", return_value=Gateway(False)):
            self.assertIsNone(roles.advisor("reporter"))  # no model serves it


@unittest.skipIf(MISSING, MISSING)
class FlowsTest(unittest.TestCase):
    def test_adopt_plan_asks_the_investigator_the_planner_and_the_engineer_that_are_on(self):
        from dclab_rnd.agent_eval import team as harness
        from dclab_rnd.intern.tools import Toolbox

        store, pid, case = harness._project("binary")
        adv = ScriptedAdvisor({})
        with mock.patch("dclab_rnd.roles.advisor", lambda role, **kw: adv if role in ("data_investigator", "solution_planner", "feature_engineer") else None):
            saved = Toolbox(store).adopt_plan(pid, case.target, case.moment)
        self.assertTrue(saved.get("saved"), saved)
        asked = {p for p, _ in adv.requests}
        self.assertEqual(asked, {"data_investigator", "solution_planner", "feature_engineer"})

    def test_the_solution_report_route_uses_the_reporter_only_after_the_final_stage(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app
        from dclab_rnd.expansion import runner
        from dclab_rnd.kinds import acceptance
        from dclab_rnd import kinds
        from dclab_rnd.studio import ProjectStore, engine

        home = Path(tempfile.mkdtemp())
        client = TestClient(create_app(home))
        token = client.get("/api/config").json()["csrf"]
        store = ProjectStore(home / "projects")
        pid, case = None, kinds.get("binary").case(0)
        store_, pid = acceptance.build(case, home / "projects")
        self.assertEqual(client.get(f"/api/projects/{pid}/export/solution-report").status_code, 409)
        with runner.fast_profile():
            engine.run_all(store_, pid)
        echo = ScriptedAdvisor({"reporter": lambda ev: {"summaries": [{"title": s["title"], "text": list(s["lines"].values())[0] + " [L1]"}
                                                                      for s in ev["sections"]]}})
        with mock.patch("dclab_rnd.roles.advisor", lambda role, **kw: echo if role == "reporter" else None):
            response = client.get(f"/api/projects/{pid}/export/solution-report", headers={"X-DCLab-Token": token})
        self.assertEqual(response.status_code, 200)
        self.assertIn("written by a model", response.text)
        roles_page = client.get("/api/roles").json()
        self.assertTrue(any(r["role"] == "reporter" for r in roles_page["roles"]))


@unittest.skipIf(MISSING, MISSING)
class LayerTest(unittest.TestCase):
    def test_the_gate_compares_the_three_arms(self):
        def arm(caught, alarms, errors=0, eur=0.0):
            return {"spent_eur": eur, "summary": {k: {"leaks_caught": [caught, 55], "false_alarms": [alarms, 100], "errors": errors}
                                                  for k in ("rules", "plain", "roles")}}
        self.assertEqual(layer.judge(arm(36, 15), arm(30, 10, eur=0.05), arm(38, 14, eur=0.2))["status"], "pass")
        self.assertEqual(layer.judge(arm(36, 15), arm(30, 10), arm(38, 20))["status"], "fail")  # more false alarms
        self.assertEqual(layer.judge(arm(36, 15), arm(40, 10), arm(38, 14))["status"], "fail")  # below the plain model
        verdict = layer.judge(arm(36, 15), arm(30, 10, eur=0.05), arm(38, 14, eur=0.19))
        self.assertEqual(verdict["eur_per_caught_trap"]["roles"], 0.005)
        self.assertEqual(layer.judge(arm(36, 15), arm(30, 10, errors=4), arm(38, 14))["status"], "not scored")  # errors anywhere

    def run_two(self, arm, advisor=None):
        cases = layer.v3.split("dev")
        two = (next(c for c in cases if c.trap == "post_outcome_name"), next(c for c in cases if c.trap == "partial_labels"))
        with mock.patch.object(layer.v3, "split", lambda name: two):
            return layer.run_arm(arm, "dev", advisor=advisor, log=lambda *a: None)

    def test_roles_that_fail_or_never_answer_are_not_scored_never_passed(self):
        class Failing:
            def ask(self, purpose, evidence):
                raise RuntimeError("BudgetExceeded")

            def spent_eur(self):
                return 0.0
        rules = self.run_two("rules")
        failing = self.run_two("roles", Failing())
        self.assertEqual(failing["summary"]["roles"]["leaks_caught"], rules["summary"]["rules"]["leaks_caught"])  # silently the rules
        plain = {"spent_eur": 0.0, "summary": {"plain": {"leaks_caught": [0, 2], "false_alarms": [0, 10], "errors": 0}}}
        verdict = layer.judge(rules, plain, failing)
        self.assertEqual(verdict["status"], "not scored")
        self.assertIn("failed", verdict["why"])
        abstaining = self.run_two("roles", ScriptedAdvisor({}))
        self.assertEqual(layer.judge(rules, plain, abstaining)["status"], "not scored")  # never answered

    def test_partial_labels_reach_the_solution_in_both_pipeline_arms(self):
        rules = self.run_two("rules")
        row = next(r for r in rules["results"] if r["case"].endswith("partial_labels"))
        self.assertTrue(row["leak_caught"])  # the plan's coverage decision now becomes labels="partial"

    def test_a_sealed_arm_runs_once_and_only_from_committed_code(self):
        with mock.patch.object(layer, "frozen_problem", return_value=None), \
                mock.patch.object(layer, "_runs", return_value=[Path("BEN3-099_test_layer_rules.json")]):
            with self.assertRaises(SystemExit):
                layer.main(["rules", "--split", "test"])
        with mock.patch.object(layer, "frozen_problem", return_value="the gate has uncommitted changes"):
            with self.assertRaises(SystemExit):
                layer.main(["rules", "--split", "test"])


@unittest.skipIf(MISSING, MISSING)
class HomeTest(unittest.TestCase):
    def test_home_without_a_model_asks_no_role(self):
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore

        store = DraftStore(Path(tempfile.mkdtemp()))
        draft = store.create("Predict which customers churn next month")
        called = []
        with mock.patch("dclab_rnd.roles.advisor", lambda role, **kw: called.append(role)):
            HomeAgent(store, None).read_brief(draft["id"])
        self.assertEqual(called, [])  # replan and the pipeline run Home with no model: they stay free


if __name__ == "__main__":
    unittest.main()
