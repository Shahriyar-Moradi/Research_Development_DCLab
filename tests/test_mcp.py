"""DCLab as an MCP server: the tool list is the intern toolbox, calls run the real tools, no CSRF token needed."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def rpc(method, params=None, id=1):
    return {"jsonrpc": "2.0", "id": id, "method": method, **({"params": params} if params is not None else {})}


class McpOverHttpTests(unittest.TestCase):
    def setUp(self):
        try:
            import mcp  # noqa: F401
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"dependencies not installed: {error}")
        self.home = Path(tempfile.mkdtemp())
        self.client = TestClient(create_app(self.home))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def post(self, body):
        response = self.client.post("/mcp", json=body, headers=HEADERS)
        self.assertEqual(response.status_code, 200, response.text[:300])
        return response.json()

    def test_initialize_list_and_call(self):
        init = self.post(rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}))
        self.assertEqual(init["result"]["serverInfo"]["name"], "dclab")
        self.assertIn("holdout", init["result"]["instructions"])
        tools = self.post(rpc("tools/list", {}, 2))["result"]["tools"]
        names = {t["name"] for t in tools}
        self.assertTrue({"search_evidence", "list_samples", "create_project", "set_solution", "run_all", "export_notebook"} <= names)
        self.assertEqual(next(t for t in tools if t["name"] == "set_solution")["inputSchema"]["required"], ["project_id", "target", "task", "prediction_moment"])
        hit = self.post(rpc("tools/call", {"name": "search_evidence", "arguments": {"query": "duration leakage bank marketing", "k": 2}}, 3))
        payload = json.loads(hit["result"]["content"][0]["text"])
        full = (ROOT / "evidence" / "knowledge" / "rag" / "records.jsonl").exists()  # the open edition packages the rules and pitfalls only
        prefixes = ("LEAK-", "EXP-") if full else ("DCLAB-", "PIT-")
        self.assertTrue(any(r["record_id"].startswith(prefixes) for r in payload["results"]), payload["results"])
        created = self.post(rpc("tools/call", {"name": "create_project", "arguments": {"name": "via mcp", "goal": "smoke"}}, 4))
        project_id = json.loads(created["result"]["content"][0]["text"])["project_id"]
        self.assertEqual(self.client.get(f"/api/projects/{project_id}").status_code, 200)
        bad = self.post(rpc("tools/call", {"name": "create_project", "arguments": {"name": "x"}}, 5))
        self.assertTrue(bad["result"].get("isError") or "error" in json.loads(bad["result"]["content"][0]["text"]))
        calls = [json.loads(line) for line in (self.home / "mcp" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]  # 32.7: every call kept
        self.assertEqual([c["tool"] for c in calls][:2], ["search_evidence", "create_project"])  # a call the protocol refused never reached a tool
        self.assertTrue(all(c["run"] for c in calls))

    def test_status_advertises_the_endpoint(self):
        info = self.client.get("/api/intern").json()
        self.assertTrue(info["mcp_url"].endswith("/mcp"))


@unittest.skipUnless((ROOT / "scripts" / "chat_ui.py").exists(), "scripts/chat_ui.py is not part of this edition")
class ChatUiConfigTests(unittest.TestCase):
    def test_env_local_points_chat_ui_at_dclab(self):
        import argparse
        import importlib.util
        import os
        from unittest import mock

        spec = importlib.util.spec_from_file_location("chat_ui", ROOT / "scripts" / "chat_ui.py")
        chat_ui = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chat_ui)
        args = argparse.Namespace(ml_intern=True, dclab_url="http://127.0.0.1:8765/")
        clean = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "HF_TOKEN", "OPENID_CLIENT_ID", "OPENID_CLIENT_SECRET", "OPENAI_BASE_URL", "DCLAB_CHAT_UI_MODELS", "DCLAB_CHAT_UI_EXTRA_ENV")}
        with mock.patch.dict(os.environ, clean, clear=True):
            text, warnings = chat_ui.env_local(args)
        self.assertIn('MCP_SERVERS=[{"name": "DCLab notebook", "url": "http://127.0.0.1:8765/mcp"}]', text)
        self.assertIn("OPENAI_BASE_URL=https://router.huggingface.co/v1", text)
        self.assertIn("ML_ASSISTANT_MODE=true", text)
        self.assertEqual(len(warnings), 2)  # no key, no OAuth app
        with mock.patch.dict(os.environ, {**clean, "OPENAI_API_KEY": "hf_x", "OPENID_CLIENT_ID": "id", "OPENID_CLIENT_SECRET": "s"}, clear=True):
            text, warnings = chat_ui.env_local(argparse.Namespace(ml_intern=False, dclab_url="http://127.0.0.1:8765"))
        self.assertEqual(warnings, [])
        self.assertIn("MCP_FORWARD_HF_USER_TOKEN=true", text)
        self.assertNotIn("ML_ASSISTANT_MODE", text)


class StandaloneMcpTests(unittest.TestCase):
    """A7.1: the standalone /mcp (what DCLab Code starts) runs in the workspace's whole services, as the product server's."""

    def setUp(self):
        try:
            import mcp  # noqa: F401
            from starlette.testclient import TestClient
            from dclab_rnd import mcp_server
        except ImportError as error:
            self.skipTest(f"dependencies not installed: {error}")
        self.workspace = Path(tempfile.mkdtemp())
        self.client = TestClient(mcp_server.standalone_app(self.workspace))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.services = self.client.app.state.services

    def call(self, name, arguments, id=1):
        response = self.client.post("/mcp", json=rpc("tools/call", {"name": name, "arguments": arguments}, id), headers=HEADERS)
        self.assertEqual(response.status_code, 200, response.text[:300])
        return json.loads(response.json()["result"]["content"][0]["text"])

    def test_a_call_records_its_events_decisions_and_itself_in_the_workspace(self):
        project = self.call("create_project", {"name": "standalone", "goal": "smoke"})["project_id"]
        self.call("check_move", {"project_id": project, "move": "run_stage", "stage": "data"}, 2)
        calls = [json.loads(x) for x in (self.workspace / "mcp" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([c["tool"] for c in calls], ["create_project", "check_move"])
        events = self.services.experience.read(project=project)  # files or PostgreSQL alike
        self.assertTrue(events, "the MCP agent's run, recorded in the workspace's experience log")
        self.assertTrue(all(e.ids.request and e.ids.trace for e in events), "each event carries its request and trace, as through the product server")
        self.assertTrue(any(str(e.ids.run or "").startswith("mcp-") for e in events), "inside the MCP agent's run")
        self.assertTrue(self.services.ledger.records(project=project), "the project's decisions in the workspace's ledger")

    def test_the_workspace_is_the_one_the_settings_name_and_consent_is_read_fresh(self):
        import os
        from unittest import mock

        from starlette.testclient import TestClient
        from dclab_rnd import consent, mcp_server

        other = Path(tempfile.mkdtemp())
        with mock.patch.dict(os.environ, {"DCLAB_AGENT_HOME": str(other)}):
            app = mcp_server.standalone_app()  # no --home: DCLAB_AGENT_HOME, as the product server and the worker choose
        with TestClient(app):
            self.assertEqual(Path(app.state.services.store.home).resolve(), other.resolve())
        self.assertEqual(consent.PROJECT_TTL, 0.0)

    def test_a_model_request_is_in_the_workspace_usage_log_and_refused_over_its_cap(self):
        import os
        from unittest import mock

        from dclab_rnd import models

        prices = self.workspace / "prices.json"
        prices.write_text(json.dumps({"priced-model": {"input": 2.0, "output": 8.0, "as_of": "2026-10-01", "source": "test"}}))

        class Scripted:
            def __init__(self, tier, purpose):
                pass

            def complete(self, messages, tools=None, max_tokens=1800, **options):
                return {"content": "ok", "usage": {"input_tokens": 500_000, "output_tokens": 100_000}}

        gateway = models.installed()
        self.assertIs(gateway, self.services.gateway, "the process's gateway is the workspace's")
        from dclab_rnd.models import settings as model_settings

        clean = {k: "" for k in ("OPENAI_BASE_URL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
                                 "DCLAB_BLOCK_PROVIDERS", "DCLAB_INTERN_MODEL", "DCLAB_WORKSPACE_MONTHLY_EUR")}
        clean |= {f"DCLAB_TIER_{t.upper()}_{n}": "" for t in model_settings.TIERS for n in ("BASE_URL", "MODEL", "API_KEY")}
        env = {**clean, "OPENAI_API_KEY": "k", "OPENAI_MODEL": "priced-model", "DCLAB_PRICES_FILE": str(prices),
               "DCLAB_NO_LIVE_MODELS": ""}  # the scripted transport never leaves the machine
        with mock.patch.dict(os.environ, env), mock.patch.object(gateway, "transport", lambda tier, purpose: Scripted(tier, purpose)):
            gateway.client("project_answer", project_id="p1").complete([{"role": "user", "content": "x"}], max_tokens=1000)
            self.assertTrue(gateway.usage.recent(), "the request is in the workspace's usage log")
            with mock.patch.dict(os.environ, {"DCLAB_WORKSPACE_MONTHLY_EUR": "0.5"}), self.assertRaises(RuntimeError) as over:
                gateway.client("project_answer", project_id="p2").complete([{"role": "user", "content": "x"}], max_tokens=1000)
            self.assertIn("the workspace has used its monthly budget", str(over.exception))


if __name__ == "__main__":
    unittest.main()
