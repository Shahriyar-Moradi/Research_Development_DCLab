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
        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))
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
        self.assertTrue(any(r["record_id"].startswith("LEAK-") or r["record_id"].startswith("EXP-") for r in payload["results"]))
        created = self.post(rpc("tools/call", {"name": "create_project", "arguments": {"name": "via mcp", "goal": "smoke"}}, 4))
        project_id = json.loads(created["result"]["content"][0]["text"])["project_id"]
        self.assertEqual(self.client.get(f"/api/projects/{project_id}").status_code, 200)
        bad = self.post(rpc("tools/call", {"name": "create_project", "arguments": {"name": "x"}}, 5))
        self.assertTrue(bad["result"].get("isError") or "error" in json.loads(bad["result"]["content"][0]["text"]))

    def test_status_advertises_the_endpoint(self):
        info = self.client.get("/api/intern").json()
        self.assertTrue(info["mcp_url"].endswith("/mcp"))


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


if __name__ == "__main__":
    unittest.main()
