"""Package A5.1 (embeddings): the gateway's embedding call, the neural retrievers, the cached record vectors, and the safe fallback."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dclab_rnd import retrieval, tools  # noqa: E402
from dclab_rnd.models import prices  # noqa: E402
from dclab_rnd.models.client import ChatClient  # noqa: E402
from dclab_rnd.models.gateway import Gateway  # noqa: E402
from dclab_rnd.models.usage import FileUsage  # noqa: E402
from test_models_gateway import CLEAN  # noqa: E402

TEXT_FIELDS = {"embedding": "ok"}


def fake_vector(text: str, dims: int = 64) -> list[float]:
    """A deterministic stand-in for a sentence embedding: hashed words, so texts that share words are close."""
    v = np.zeros(dims)
    for word in text.lower().split():
        v[hash(word) % dims] += 1.0
    return v.tolist()


class FakeEmbeddings:
    """What a gateway client offers the retrieval: embed(texts); counts requests."""
    spent_eur = 0.0

    def __init__(self):
        self.requests, self.texts = 0, []

    def embed(self, texts):
        self.requests += 1
        self.texts.extend(texts)
        return {"vectors": [fake_vector(t) for t in texts], "usage": {"input_tokens": 10, "output_tokens": 0}}


class ClientTests(unittest.TestCase):
    def test_the_sdk_result_is_returned_in_order_with_its_tokens(self):
        class Row:
            def __init__(self, index, vec):
                self.index, self.embedding = index, vec

        class Embeddings:
            def create(self, model, input):
                self.seen = (model, list(input))
                usage = type("U", (), {"prompt_tokens": 7})()
                return type("R", (), {"data": [Row(1, [0.0, 1.0]), Row(0, [1.0, 0.0])], "usage": usage})()

        sdk = type("S", (), {"embeddings": Embeddings()})()
        c = ChatClient(model="text-embedding-3-small", base_url="https://api.openai.com/v1", api_key="sk-test-not-real")
        c._client = sdk
        out = c.embed(["a", "b"])
        self.assertEqual(out["vectors"], [[1.0, 0.0], [0.0, 1.0]])  # the provider's index order, not its arrival order
        self.assertEqual(out["usage"], {"input_tokens": 7, "output_tokens": 0})
        self.assertEqual(sdk.embeddings.seen, ("text-embedding-3-small", ["a", "b"]))

    def test_a_provider_error_is_a_runtime_error_with_no_secret(self):
        class Embeddings:
            def create(self, **kw):
                raise ValueError("Bearer sk-secret failed")

        c = ChatClient(model="m", base_url="https://api.openai.com/v1", api_key="sk-test-not-real")
        c._client = type("S", (), {"embeddings": Embeddings()})()
        with self.assertRaises(RuntimeError) as caught:
            c.embed(["a"])
        self.assertNotIn("sk-secret", str(caught.exception))


class GatewayTests(unittest.TestCase):
    def test_an_embedding_request_is_counted_priced_and_capped_like_any_other(self):
        class Transport:
            def embed(self, texts):
                return {"vectors": [[1.0]] * len(texts), "usage": {"input_tokens": 1000, "output_tokens": 0}}

        with mock.patch.dict(os.environ, {**CLEAN, "OPENAI_API_KEY": "sk-never-logged", "OPENAI_MODEL": "std-model"}):
            usage = FileUsage(Path(tempfile.mkdtemp()) / "usage.jsonl")
            gw = Gateway(usage, transport=lambda tier, purpose: Transport(), sleep=lambda s: None)
            client = gw.client("embedding", model="text-embedding-3-small")
            self.assertEqual(len(client.embed(["a", "b"])["vectors"]), 2)
            row = usage.recent()[0]
            self.assertEqual((row["purpose"], row["model"], row["input_tokens"]), ("embedding", "text-embedding-3-small", 1000))
            self.assertEqual(row["cost_eur"], round(1000 * prices.load()["text-embedding-3-small"].input / 1_000_000, 6))

    def test_the_embedding_purpose_never_sees_workspace_data(self):
        from dclab_rnd.models import settings

        self.assertEqual(settings.PURPOSES["embedding"].cell_values, 0)
        self.assertIn("no rows", settings.PURPOSES["embedding"].may_see)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.index = tools._index()

    def test_the_record_vectors_are_built_once_and_then_read_from_the_file(self):
        client = FakeEmbeddings()
        first = retrieval.Embedder(client, self.folder, model="fake-model")
        self.assertIsNone(first.records(self.index))  # no build without asking: a search never spends silently
        self.assertEqual(client.requests, 0)
        ids, vectors = first.records(self.index, build=True)
        self.assertEqual((len(ids), vectors.shape[0]), (len(self.index.records), len(self.index.records)))
        built = client.requests
        self.assertGreater(built, 0)
        again = retrieval.Embedder(client, self.folder, model="fake-model")  # a new process: the file is read, nothing is sent
        self.assertEqual(list(again.records(self.index)[0]), ids)
        self.assertEqual(client.requests, built)
        other = retrieval.Embedder(client, self.folder, model="another-model")  # a different model has its own vectors
        self.assertIsNone(other.records(self.index))

    def test_a_question_finds_the_record_that_shares_its_words_by_embedding_and_by_hybrid(self):
        embedder = retrieval.Embedder(FakeEmbeddings(), self.folder, model="fake-model")
        embedder.records(self.index, build=True)
        record = self.index.records[5]
        question = " ".join(retrieval._text(record).split()[:30])
        self.assertEqual(retrieval.ranked(question, "embed", 5, self.index, embedder)[0], record["record_id"])
        self.assertIn(record["record_id"], retrieval.ranked(question, "hybrid_embed", 5, self.index, embedder))
        with self.assertRaises(ValueError):
            retrieval.ranked(question, "embed", 5, self.index)  # no embedder

    def test_questions_are_embedded_in_one_batch_when_measured(self):
        client = FakeEmbeddings()
        embedder = retrieval.Embedder(client, self.folder, model="fake-model")
        m = retrieval.measure(retrieval.questions()[:6], embedder=embedder)
        self.assertTrue({"embed", "hybrid_embed", "bm25"} <= set(m["recall"]))
        self.assertEqual(client.texts.count(retrieval.questions()[0]["question"]), 1)

    def test_a_neural_method_is_chosen_only_when_it_is_worse_on_no_group(self):
        groups = ["rule", "paraphrase"]
        bm = {"rule": 0.8, "paraphrase": 0.0, "overall": 0.5}
        self.assertEqual(retrieval.choose({"bm25": bm, "hybrid": {"rule": 0.7, "paraphrase": 0.4, "overall": 0.6},
                                           "hybrid_embed": {"rule": 0.8, "paraphrase": 0.6, "overall": 0.7}}, groups), "hybrid_embed")  # the best that qualifies
        self.assertEqual(retrieval.choose({"bm25": bm, "embed": {"rule": 0.7, "paraphrase": 0.9, "overall": 0.9}}, groups), "bm25")  # worse on the rules
        self.assertEqual(retrieval.choose({"bm25": bm, "hybrid_embed": {"rule": 0.8, "paraphrase": 0.0, "overall": 0.5}}, groups), "bm25")  # no better: stay

    def test_search_falls_back_to_keyword_search_without_vectors_or_a_model(self):
        question = "Should I oversample before splitting?"
        keyword = [h["record_id"] for h in self.index.search(question, k=5)]
        with mock.patch.object(retrieval, "chosen", return_value="hybrid_embed"), mock.patch.dict(os.environ, {"DCLAB_EMBEDDINGS_DIR": str(self.folder)}):
            self.assertEqual([h["record_id"] for h in retrieval.search(question, index=self.index)], keyword)  # no vectors built
            hits = retrieval.search(question, index=self.index)
            self.assertTrue(all(h.get("method", "bm25") == "bm25" or "method" not in h for h in hits))

    def test_a_damaged_vectors_file_is_no_file_and_a_build_replaces_it_whole(self):
        client = FakeEmbeddings()
        embedder = retrieval.Embedder(client, self.folder, model="fake-model")
        path = embedder._path(self.index)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PK\x03\x04 not really a zip")
        self.assertIsNone(embedder.records(self.index))  # a search treats it as not built
        with mock.patch.object(retrieval, "chosen", return_value="embed"), mock.patch.dict(os.environ, {"DCLAB_EMBEDDINGS_DIR": str(self.folder)}):
            hits = retrieval.search("Should I oversample before splitting?", index=self.index)  # never an error on the page
        self.assertEqual([h["record_id"] for h in hits], [h["record_id"] for h in self.index.search("Should I oversample before splitting?", k=5)])
        ids, _ = embedder.records(self.index, build=True)  # a build writes over it, atomically
        self.assertEqual(len(ids), len(self.index.records))
        self.assertFalse(list(self.folder.glob("*.part")))
        self.assertEqual(len(retrieval.Embedder(FakeEmbeddings(), self.folder, model="fake-model").records(self.index)[0]), len(ids))

    def test_a_plain_run_does_not_undo_what_the_neural_run_showed(self):
        rec = {"bm25": {"overall": 0.7}, "hybrid": {"overall": 0.69}}
        folder = Path(tempfile.mkdtemp())
        import json as _json
        (folder / "RET-001_x.json").write_text(_json.dumps({"measurement": {"recall": {**rec, "embed": {"overall": 0.9}}, "chosen": "embed"}}))
        (folder / "RET-002_x.json").write_text(_json.dumps({"measurement": {"recall": rec, "chosen": "bm25"}}))  # a later run without --embeddings
        with mock.patch.object(retrieval, "RESULTS", folder):
            self.assertEqual(retrieval.chosen(), "embed")
        (folder / "RET-003_x.json").write_text(_json.dumps({"measurement": {"recall": {**rec, "embed": {"overall": 0.5}}, "chosen": "bm25"}}))  # a new neural run decides again
        with mock.patch.object(retrieval, "RESULTS", folder):
            self.assertEqual(retrieval.chosen(), "bm25")

    def test_search_uses_the_embeddings_when_built_and_survives_a_failed_request(self):
        embedder = retrieval.Embedder(FakeEmbeddings(), self.folder, model="fake-model")
        embedder.records(self.index, build=True)
        record = self.index.records[7]
        question = " ".join(retrieval._text(record).split()[:30])
        with mock.patch.object(retrieval, "chosen", return_value="embed"), mock.patch.object(retrieval, "default_embedder", return_value=embedder):
            hits = retrieval.search(question, index=self.index)
            self.assertEqual((hits[0]["record_id"], hits[0]["method"]), (record["record_id"], "embed"))

            class Failing(FakeEmbeddings):
                def embed(self, texts):
                    raise RuntimeError("BudgetExceeded: the monthly cap is reached")

            broken = retrieval.Embedder(Failing(), self.folder, model="fake-model")  # the vectors are on disk; the question cannot be embedded
            with mock.patch.object(retrieval, "default_embedder", return_value=broken):
                fallback = retrieval.search(question, index=self.index)
        self.assertEqual([h["record_id"] for h in fallback], [h["record_id"] for h in self.index.search(question, k=5)])


if __name__ == "__main__":
    unittest.main()
