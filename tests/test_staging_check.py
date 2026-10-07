"""Package 14.7: the staging check's pieces that can be tested without a cloud: the token header and the make target."""
import importlib
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class StagingCheckTests(unittest.TestCase):
    def test_the_flows_send_a_bearer_token_only_when_one_is_given(self):
        product = importlib.import_module("product_e2e")
        seen = []

        class Reply:
            status = 200

            def read(self):
                return b'{"csrf": "c"}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake(request, timeout=0):
            seen.append(dict(request.header_items()))
            return Reply()

        with mock.patch.object(product.urllib.request, "urlopen", fake), mock.patch.dict(os.environ, {"DCLAB_E2E_TOKEN": "dclab_test_token"}):
            product.Api("http://staging.example").call("POST", "/x", {})
        self.assertEqual(seen[-1]["Authorization"], "Bearer dclab_test_token")
        env = {k: v for k, v in os.environ.items() if k != "DCLAB_E2E_TOKEN"}
        seen.clear()
        with mock.patch.object(product.urllib.request, "urlopen", fake), mock.patch.dict(os.environ, env, clear=True):
            product.Api("http://staging.example").call("GET", "/x")
        self.assertNotIn("Authorization", seen[-1])

    def test_the_make_target_asks_for_an_address_and_the_runbook_states_what_it_cannot_do(self):
        done = subprocess.run(["make", "staging-check"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)
        self.assertIn("give the address", done.stdout + done.stderr)
        text = (ROOT / "deploy" / "STAGING.md").read_text()
        for needle in ("DCLAB_E2E_TOKEN", "model requests", "never run against a signed-in"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
