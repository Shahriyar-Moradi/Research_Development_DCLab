"""DCLab Watch B4 (2026-11-03): a decision model's card and a bench run become MLflow runs through its REST interface; off by
default; MLflow away is counted, never raised."""

import http.server
import json
import os
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import watch_mlflow as wm  # noqa: E402


class FakeMLflow(http.server.BaseHTTPRequestHandler):
    calls: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        FakeMLflow.calls.append((self.path, body))
        answer = {"experiments/create": {"experiment_id": "7"}, "runs/create": {"run": {"info": {"run_id": "r1"}}}}
        key = self.path.split("/mlflow/", 1)[1]
        out = json.dumps(answer.get(key, {})).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


class WatchMlflowTests(unittest.TestCase):
    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {"DCLAB_TRACE_EXPORT": ""}):
            self.assertFalse(wm.log_run("x", "y", {}, {}))

    def test_a_card_becomes_a_run_with_names_and_numbers_only(self):
        from dclab_core.decision_models import ModelCard, Score

        card = ModelCard(point="column.semantic_role", name="column.semantic_role.boosting", version="abc", family="boosting", strength=0.1,
                         features=140, fields=["cardinality", "dtype", "name_tokens"], trained_on=4320, trained_set="f" * 64, groups=144,
                         chosen_by="out-of-fold accuracy", oof_accuracy={"boosting:0.1": 0.93}, dev=Score(n=10, accuracy=0.936),
                         rule_dev=Score(n=10, accuracy=0.835))
        FakeMLflow.calls = []
        server = http.server.HTTPServer(("127.0.0.1", 0), FakeMLflow)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            env = {"DCLAB_TRACE_EXPORT": "mlflow", "DCLAB_MLFLOW_URL": f"http://127.0.0.1:{server.server_port}"}
            with mock.patch.dict(os.environ, env):
                self.assertTrue(wm.log_card(card))
        finally:
            server.shutdown()
            server.server_close()
        paths = [p.split("/mlflow/", 1)[1] for p, _ in FakeMLflow.calls]
        self.assertEqual(paths, ["experiments/create", "runs/create", "runs/log-batch", "runs/update"])
        batch = FakeMLflow.calls[2][1]
        metrics = {m["key"]: m["value"] for m in batch["metrics"]}
        self.assertEqual((metrics["dev_accuracy"], metrics["rule_dev_accuracy"], metrics["trained_on"]), (0.936, 0.835, 4320.0))
        self.assertEqual({p["key"]: p["value"] for p in batch["params"]}["point"], "column.semantic_role")

    def test_mlflow_away_is_counted_never_raised(self):
        before = wm.stats()["failed"]
        with mock.patch.dict(os.environ, {"DCLAB_TRACE_EXPORT": "mlflow", "DCLAB_MLFLOW_URL": "http://127.0.0.1:9"}), \
                mock.patch.object(wm, "TIMEOUT", 0.5):
            self.assertFalse(wm.log_bench({"commit": "abc", "states": {"agents": "red"}, "commands": []}))
        self.assertEqual(wm.stats()["failed"], before + 1)


if __name__ == "__main__":
    unittest.main()
