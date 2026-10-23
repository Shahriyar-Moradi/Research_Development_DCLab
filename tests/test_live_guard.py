"""D3 (2026-10-23): a test never reaches a live model unless it asks for one, whichever entry point started it."""

import os
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class LiveGuardTest(unittest.TestCase):
    def setUp(self):
        env = {k: v for k, v in os.environ.items() if k != "DCLAB_ALLOW_LIVE_MODELS_IN_TESTS"}
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_real_client_is_refused_in_a_test(self):
        from dclab_rnd.models.anthropic_client import AnthropicClient
        from dclab_rnd.models.client import ChatClient
        from dclab_rnd.models.live_guard import LiveModelInTest

        with self.assertRaises(LiveModelInTest):
            ChatClient(model="m", base_url="https://api.openai.com/v1", api_key="placeholder").client
        with self.assertRaises(LiveModelInTest):
            AnthropicClient(model="claude-sonnet-5-5", api_key="placeholder").client

    def test_an_injected_fake_still_works(self):
        from dclab_rnd.models.client import ChatClient

        fake = object()
        c = ChatClient(model="m", base_url="http://127.0.0.1:11434/v1", api_key="ollama")
        c._client = fake
        self.assertIs(c.client, fake)

    def test_a_test_may_ask_for_a_live_model(self):
        from dclab_rnd.models.client import ChatClient

        with mock.patch.dict(os.environ, {"DCLAB_ALLOW_LIVE_MODELS_IN_TESTS": "1"}):
            self.assertIsNotNone(ChatClient(model="m", base_url="http://127.0.0.1:9/v1", api_key="k").client)  # built, never called

    def test_the_mark_holds_in_a_background_thread(self):
        from dclab_rnd.models import live_guard

        seen = {}

        def turn():  # an API worker's turn: no test frame on its stack
            seen["marked"] = live_guard.in_test()

        with mock.patch.object(sys, "argv", ["server"]), mock.patch.dict(os.environ, {"DCLAB_TEST_RUN": "1"}), \
                mock.patch.dict(sys.modules, {"_pytest": None}):
            sys.modules.pop("_pytest", None)
            t = threading.Thread(target=turn)
            t.start()
            t.join()
        self.assertTrue(seen["marked"])

    def test_outside_a_test_nothing_is_refused(self):
        from dclab_rnd.models import live_guard

        seen = {}

        def turn():
            seen["marked"] = live_guard.in_test()

        with mock.patch.object(sys, "argv", ["dclab"]), mock.patch.dict(os.environ, {"DCLAB_TEST_RUN": ""}), \
                mock.patch.dict(sys.modules, {"_pytest": None}):
            sys.modules.pop("_pytest", None)
            t = threading.Thread(target=turn)
            t.start()
            t.join()
        self.assertFalse(seen["marked"])


if __name__ == "__main__":
    unittest.main()
