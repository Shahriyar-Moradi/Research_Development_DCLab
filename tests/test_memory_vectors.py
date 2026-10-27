"""35.5: memory searched by meaning. The local embedding model 35.0 chose (embeddinggemma) serves the gateway's embedding
purpose only, from DCLAB_EMBEDDING_BASE_URL and DCLAB_EMBEDDING_MODEL, on this machine only; each item's vector is kept in
the workspace, named by the model, and recall ranks by meaning when it is there, by words when it is not."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import memory as store_mod  # noqa: E402
from dclab_rnd.memory import reading as r  # noqa: E402
from dclab_rnd.memory import vectors as v  # noqa: E402
from dclab_rnd.memory import writing as w  # noqa: E402
from dclab_rnd.models import gateway as gw  # noqa: E402
from dclab_rnd.models import settings  # noqa: E402

AT = "2026-10-28T09:00:00+00:00"
CHAIN = [("global", "dclab"), ("workspace", "local"), ("person", "owner"), ("project", "p1")]
LOCAL = {"DCLAB_MEMORY_EMBEDDING_BASE_URL": "http://127.0.0.1:11434/v1", "DCLAB_MEMORY_EMBEDDING_MODEL": "embeddinggemma:300m-qat-q8_0", "DCLAB_MEMORY": ""}

TOPICS = {"money": ("cost", "price", "euro", "budget", "pay"), "time": ("week", "month", "day", "holdout", "later")}


class Words:
    """A scripted embedding transport: a vector of topic counts, so 'price' is near 'cost' though they share no word."""

    thinking = None

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            low = t.lower()
            out.append([float(sum(w in low for w in words)) + 0.01 for words in TOPICS.values()])
        return {"vectors": out, "usage": {"input_tokens": 0, "output_tokens": 0}}


class EndpointTest(unittest.TestCase):
    def test_the_local_endpoint_serves_the_embedding_purpose_only(self):
        with mock.patch.dict(os.environ, LOCAL):
            t = settings.embedding_tier()
            self.assertEqual((t["base_url"], t["model"]), (LOCAL["DCLAB_MEMORY_EMBEDDING_BASE_URL"], LOCAL["DCLAB_MEMORY_EMBEDDING_MODEL"]))
            g = gw.Gateway()
            g.transport = lambda tier, p: Words()
            self.assertEqual(g.client("memory_embedding").model, LOCAL["DCLAB_MEMORY_EMBEDDING_MODEL"])
            for other in ("embedding", "parse_pattern"):  # the evidence search's embedding and every other purpose stay where they were
                bound = g.client(other)
                self.assertTrue(bound is None or bound.model != LOCAL["DCLAB_MEMORY_EMBEDDING_MODEL"], other)
            self.assertEqual(g.client("memory_embedding", model="bge-m3:567m").model, "bge-m3:567m")  # a chosen model is the one asked

    def test_a_request_goes_to_the_local_model_not_the_routed_tier(self):
        """The live check (2026-10-28): before each request the routing moved the purpose back to the cheap tier (Claude)."""
        class Elsewhere:
            thinking = None

            def embed(self, texts):
                raise AssertionError("the request left the local model")

        with mock.patch.dict(os.environ, LOCAL):
            g = gw.Gateway()
            local = Words()
            g.transport = lambda tier, p: local if tier.get("name") == settings.EMBEDDING_LOCAL else Elsewhere()
            out = g.client("memory_embedding").embed(["a price"])
        self.assertEqual((len(out["vectors"]), local.calls), (1, 1))

    def test_an_endpoint_elsewhere_is_refused(self):
        with mock.patch.dict(os.environ, {**LOCAL, "DCLAB_MEMORY_EMBEDDING_BASE_URL": "https://api.example.com/v1"}):
            self.assertIsNone(settings.embedding_tier())

    def test_no_live_models_in_a_test_process(self):
        with mock.patch.dict(os.environ, {**LOCAL, "DCLAB_NO_LIVE_MODELS": "1"}):
            self.assertIsNone(gw.Gateway().client("memory_embedding"))  # the real transport never reaches Ollama in a test


class RecallByMeaningTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.store = store_mod.open_memory(self.home)
        self.env = mock.patch.dict(os.environ, LOCAL)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.words = Words()
        g = gw.Gateway()
        g.transport = lambda tier, p: self.words
        self.embedder = v.Embedder(self.home, g.client("memory_embedding"), "sha256:0123456789abcdef")

    def fact(self, text):
        return w.record_fact(self.store, scope="project", scope_id="p1", text=text, source="test:1", who="person:owner", at=AT)

    def test_recall_finds_by_meaning_where_words_find_nothing(self):
        money = self.fact("A retention call costs five euros.")
        self.fact("The holdout is the last four weeks.")
        self.assertEqual(r.recall(self.store, "what is the price", CHAIN), [])  # no shared word
        self.assertEqual([i.id for i in r.recall(self.store, "what is the price", CHAIN, embedder=self.embedder)][:1], [money.id])

    def test_vectors_are_kept_named_by_the_model_and_reused(self):
        self.fact("A retention call costs five euros.")
        r.recall(self.store, "price", CHAIN, embedder=self.embedder)
        first = self.words.calls
        r.recall(self.store, "budget", CHAIN, embedder=self.embedder)
        self.assertEqual(self.words.calls, first + 1)  # only the question is embedded again
        files = list((self.home / "memory").glob("vectors-*.json"))
        self.assertEqual(len(files), 1)
        self.assertIn("embeddinggemma", files[0].name)
        self.assertIn("0123456789ab", files[0].name)  # the weights' digest: weights pulled again under the tag start a new file

    def test_the_product_embedder_is_none_without_a_local_model(self):
        with mock.patch.dict(os.environ, {"DCLAB_MEMORY_EMBEDDING_BASE_URL": "", "DCLAB_MEMORY_EMBEDDING_MODEL": ""}):
            self.assertIsNone(v.embedder_for(self.home))


    def test_a_local_model_that_is_down_falls_back_to_words(self):
        class Down:
            model = "embeddinggemma:300m-qat-q8_0"

            def query(self, q):
                raise RuntimeError("connection refused")

            def docs(self, texts):
                raise RuntimeError("connection refused")

        cost = self.fact("A retention call costs five euros.")
        self.assertEqual([i.id for i in r.recall(self.store, "what does a call cost", CHAIN, embedder=Down())], [cost.id])

    def test_without_a_local_model_memory_gets_no_model_at_all(self):
        with mock.patch.dict(os.environ, {"DCLAB_MEMORY_EMBEDDING_BASE_URL": "", "DCLAB_MEMORY_EMBEDDING_MODEL": ""}):
            self.assertIsNone(gw.Gateway().client("memory_embedding"))


if __name__ == "__main__":
    unittest.main()
