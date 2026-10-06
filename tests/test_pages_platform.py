"""Integrations and Admin routes: every number comes from the code that applies it, and nothing secret leaves the server."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PlatformApiTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"})
        env.start()
        self.addCleanup(env.stop)
        self.app = create_app(Path(tempfile.mkdtemp()))
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}
        self.projects = self.app.state.projects

    # ------------------------------------------------------------------ integrations
    def test_integrations_read_the_real_tools_routes_and_makefile(self):
        from dclab_rnd import mcp_server
        d = self.client.get("/api/platform/integrations").json()
        intern = self.client.get("/api/intern").json()
        m = d["mcp"]
        self.assertEqual([t["name"] for t in m["tools"]], intern["tools"])
        self.assertEqual(m["count"], len(intern["tools"]))
        self.assertEqual(m["mounted"], mcp_server.available())
        if m["mounted"]:
            self.assertEqual(m["url"], intern["mcp_url"])
        run_stage = next(t for t in m["tools"] if t["name"] == "run_stage")
        self.assertEqual((run_stage["required"], run_stage["group"]), (["project_id", "stage"], "project"))
        self.assertEqual({t["group"] for t in m["tools"] if t["name"] in ("get_graph", "check_move")}, {"graph"})
        # ML Intern's count is the README's recorded verification, carried forward honestly
        mi = m["ml_intern_tools"]
        if mi["verified_total"] is not None:
            self.assertEqual(mi["own"], mi["verified_total"] - mi["verified_dclab"])
            self.assertEqual(mi["expected_now"], m["count"] + mi["own"])
            self.assertEqual(mi["current"], mi["verified_dclab"] == m["count"])
        self.assertEqual(d["chat_ui"], intern["chat_ui"])
        rest = {(r["method"], r["path"]) for r in d["rest"]}
        self.assertIn(("GET", "/api/platform/integrations"), rest)
        self.assertIn(("PATCH", "/api/projects/{project_id}"), rest)
        self.assertNotIn(("PUT", "/api/projects/{project_id}/contract"), rest)  # a hidden alias
        self.assertTrue(all(p.startswith("/api/") for _, p in rest))
        self.assertEqual(sorted(d["rest_groups"]), sorted({r["group"] for r in d["rest"]}))
        cli = {t["target"]: t for t in d["cli"]}
        self.assertIn("rd-check", cli)
        self.assertIn("MCP", cli["mcp-serve"]["help"])
        self.assertEqual(cli["knowledge"]["help"], "rebuild the evidence index and the SFT v3 corpus after new results")
        self.assertEqual(d["vscode"]["status"], "not packaged yet")

    def test_connector_status_names_and_booleans_never_secrets(self):
        with mock.patch.dict(os.environ, {"KAGGLE_USERNAME": "someone", "KAGGLE_KEY": "kaggle-secret-123",
                                          "DCLAB_DB_WAREHOUSE": "postgresql://reader:pw-secret-456@db.internal/sales"}):
            r = self.client.get("/api/platform/integrations")
        c = r.json()["connectors"]
        self.assertTrue(c["kaggle"]["configured"])
        self.assertEqual(c["database"]["connections"], ["warehouse"])
        self.assertTrue(c["ready"]["database"])
        for secret in ("kaggle-secret-123", "pw-secret-456", "db.internal"):
            self.assertNotIn(secret, r.text)

    # ------------------------------------------------------------------ policies
    def test_policies_are_probed_against_the_validator(self):
        p = self.projects.create("Churn", "general", "who leaves")
        p["policy"] = {"require_holdout_approval": True}
        self.projects.save(p)
        self.projects.create("Fraud", "general", "")
        d = self.client.get("/api/platform/policies").json()
        ids = {i["id"] for i in d["invariants"]}
        self.assertTrue({"holdout-once", "holdout-reason", "person-approves-gates", "stages-in-order", "signed-solution"} <= ids)
        for inv in d["invariants"]:
            self.assertTrue(inv["locked"])
            if inv["checked"]:
                self.assertTrue(inv["holds"], inv["id"])
        sw = {s["key"]: s for s in d["switches"]}
        self.assertEqual(set(sw), {"require_solution_signoff", "require_holdout_approval"})
        self.assertEqual((sw["require_holdout_approval"]["on"], sw["require_holdout_approval"]["total"]), (1, 2))
        self.assertEqual(sw["require_solution_signoff"]["on"], 0)
        self.assertEqual({r["name"]: r["on"] for r in sw["require_holdout_approval"]["projects"]}, {"Churn": True, "Fraud": False})
        actors = {a["action"]: a for a in d["actors"]}
        gate = actors["Sign the solution (gate)"]
        self.assertEqual((gate["person"]["status"], gate["intern"]["status"]), ("allowed", "blocked"))
        rerun = actors["Rerun the final stage after the holdout is used"]
        self.assertEqual((rerun["person"]["status"], rerun["intern"]["status"]), ("needs_approval", "blocked"))
        self.assertIn("Holdout reuse confirmed", rerun["person"]["failed"])
        self.assertFalse(d["accounts"]["users"])
        self.assertIn("presentation only", d["accounts"]["view_as"])

    def test_a_rule_the_validator_stops_enforcing_shows_as_not_holding(self):
        from dclab_rnd.studio import graph
        real = graph.check

        def lax(project, move, actor="human", **args):
            v = real(project, move, "human", **args)  # every actor treated as a person: the intern could approve gates
            v.actor = actor
            return v

        with mock.patch.object(graph, "check", lax):
            d = self.client.get("/api/platform/policies").json()
        holds = {i["id"]: i["holds"] for i in d["invariants"]}
        self.assertFalse(holds["person-approves-gates"])
        self.assertFalse(holds["holdout-once"])
        self.assertTrue(holds["stages-in-order"])

    # ------------------------------------------------------------------ limits
    def test_limits_match_the_clamps_the_routes_apply(self):
        from dclab_rnd.agentic.pages import platform
        from dclab_rnd.intern.sessions import DEFAULT_BUDGET, SessionStore
        d = self.client.get("/api/platform/limits").json()
        self.assertEqual(d["intern_session"]["defaults"], DEFAULT_BUDGET)
        sessions = SessionStore(Path(tempfile.mkdtemp()))
        low = sessions.create("x" * 10, mode="standard", model=None, budget={"max_steps": 0, "max_minutes": 0})["budget"]
        high = sessions.create("x" * 10, mode="standard", model=None, budget={"max_steps": 10**6, "max_minutes": 10**6})["budget"]
        caps = d["intern_session"]["caps"]
        self.assertEqual((low["max_steps"], high["max_steps"]), tuple(caps["max_steps"]))
        self.assertEqual((low["max_minutes"], high["max_minutes"]), tuple(caps["max_minutes"]))
        draft = self.client.post("/api/drafts", json={"problem": "Predict which customers leave next month"}, headers=self.h).json()
        put = lambda body: self.client.put(f"/api/drafts/{draft['id']}/settings", json=body, headers=self.h).json()["settings"]
        default = put({})
        self.assertEqual({"max_rows": default["max_rows"], "folds": default["folds"], "calls": default["budget"]["max_steps"],
                          "minutes": default["budget"]["max_minutes"], "eur": default["budget"]["eur"]}, d["draft_settings"]["defaults"])
        lo = put({"max_rows": 1, "folds": 1, "budget": {"calls": -5, "minutes": -5, "eur": -5}})
        hi = put({"max_rows": 10**9, "folds": 99, "budget": {"calls": 10**6, "minutes": 10**6, "eur": 10**9}})
        got = {"max_rows": [lo["max_rows"], hi["max_rows"]], "folds": [lo["folds"], hi["folds"]],
               "calls": [lo["budget"]["max_steps"], hi["budget"]["max_steps"]], "minutes": [lo["budget"]["max_minutes"], hi["budget"]["max_minutes"]],
               "eur": [lo["budget"]["eur"], hi["budget"]["eur"]]}
        self.assertEqual(got, d["draft_settings"]["caps"])
        self.assertIn("200 * 1024 * 1024", (ROOT / "dclab_rnd/agentic/server.py").read_text(encoding="utf-8"))
        self.assertEqual(d["upload_max_bytes"], platform.UPLOAD_MAX_BYTES)
        self.assertTrue(d["spend"]["tracked"])  # the gateway counts every model request (package A1.2)
        self.assertEqual((d["spend"]["eur_this_month"], d["spend"]["priced_models"]), (0.0, []))
        self.assertIn("No model has a configured price yet", d["spend"]["note"])  # nothing is guessed
        sees = next(p for p in d["privacy"] if p["title"] == "What a model sees")
        self.assertFalse(sees["on"])
        self.assertIn("nothing is sent", sees["text"])

    def test_the_model_key_never_reaches_the_page(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-secret-789"}):
            limits = self.client.get("/api/platform/limits")
            intern = self.client.get("/api/intern")
        self.assertTrue(limits.json()["model"]["key_configured"])
        self.assertNotIn("sk-test-secret-789", limits.text + intern.text)

    # ------------------------------------------------------------------ audit
    def test_audit_merges_every_project_newest_first_with_paging(self):
        from dclab_rnd.studio import graph
        a = self.projects.create("Churn", "general", "")
        b = self.projects.create("Fraud", "general", "")
        move = lambda at, actor, mv, status, **args: {"at": at, "actor": actor, "move": mv, "args": args, "status": status, "message": "m",
                                                      "from": None, "to": None, "state": "", "failed_checks": [], "rules": [], "evidence": [], "side_effects": []}
        self.projects.transition(a["id"], move("2026-10-01T10:00:00+00:00", "human", "set_solution", "allowed", target="Churn"))
        self.projects.transition(a["id"], move("2026-10-01T10:00:00+00:00", "human", "run_stage", "allowed", stage="data"))
        self.projects.transition(b["id"], move("2026-10-02T09:00:00+00:00", "agent", "run_stage", "blocked", stage="final"))
        self.projects.transition(b["id"], move("2026-10-03T09:00:00+00:00", "agent", "approve_gate", "blocked", gate="holdout"))
        p = self.projects.get(b["id"])
        p["solution"] = {"target": "y"}
        self.projects.save(p)
        graph.approve_gate(self.projects, b["id"], "holdout", "owner", "checked the split")  # logs an allowed approve_gate too
        d = self.client.get("/api/platform/audit").json()
        self.assertEqual(d["projects"], 2)
        self.assertEqual(d["total"], 5)  # the allowed approve_gate transition is folded into its approval
        first = d["items"][0]
        self.assertEqual((first["kind"], first["who"], first["message"], first["project"]["name"]), ("approval", "Owner", "checked the split", "Fraud"))
        moves = [(e["move"], e["status"], e["who"]) for e in d["items"][1:]]
        self.assertEqual(moves, [("approve_gate", "blocked", "Intern"), ("run_stage", "blocked", "Intern"),
                                 ("run_stage", "allowed", "Person"), ("set_solution", "allowed", "Person")])  # same second: later line first
        self.assertEqual(d["counts"], {"moves": 4, "approvals": 1, "blocked": 2, "waiting": 0, "policy_changes": 0, "routing_changes": 0})
        page = self.client.get("/api/platform/audit?limit=2&offset=3").json()
        self.assertEqual([e["move"] for e in page["items"]], ["run_stage", "set_solution"])
        self.assertEqual((page["total"], page["offset"], page["limit"]), (5, 3, 2))
        self.assertEqual(self.client.get("/api/platform/audit?limit=0").status_code, 422)
        self.assertEqual(self.client.get("/api/platform/audit?offset=-1").status_code, 422)

    def test_gate_switch_changes_are_logged_and_audited(self):
        p = self.projects.create("Churn", "general", "")
        url = f"/api/projects/{p['id']}"
        self.assertEqual(self.client.patch(url, json={"policy": {"require_holdout_approval": True}}, headers=self.h).status_code, 200)
        self.client.patch(url, json={"policy": {"require_holdout_approval": True}}, headers=self.h)  # no change, nothing logged
        self.client.patch(url, json={"policy": {"require_contract_signoff": True}}, headers=self.h)  # the old name still works
        logged = [a for a in self.projects.activity(p["id"]) if a["kind"] == "policy_changed"]
        self.assertEqual([a["payload"]["changes"] for a in logged],
                         [{"require_holdout_approval": {"from": False, "to": True}}, {"require_solution_signoff": {"from": False, "to": True}}])
        d = self.client.get("/api/platform/audit").json()
        self.assertEqual([(e["kind"], e["args"]["policy"], e["args"]["on"]) for e in d["items"]],
                         [("policy", "require_solution_signoff", True), ("policy", "require_holdout_approval", True)])
        self.assertEqual(d["counts"]["policy_changes"], 2)


if __name__ == "__main__":
    unittest.main()
