"""35.0: memory search measured before any vector store: the corpus, the questions, the gate fixed before the run,
and the dev and sealed steps on a scripted embedding client (never a live model)."""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy  # noqa: F401
    import sklearn  # noqa: F401

    from dclab_rnd.evidence_index import tokenize
    from dclab_rnd.retrieval import Embedder
    from dclab_rnd.retrieval import memory as mr

    MISSING = ""
except ImportError as error:  # pragma: no cover
    MISSING = str(error)


class Words:
    """A scripted embedding client: a bag of the question's and the item's words, hashed into 64 slots. It records every
    text it was sent, so a test can read the prefixes."""

    def __init__(self):
        self.sent: list[str] = []
        self.spent_eur = 0.0

    def embed(self, texts):
        import numpy as np

        self.sent.extend(texts)
        out = []
        for text in texts:
            v = np.zeros(64)
            for token in tokenize(text):
                v[sum(map(ord, token)) % 64] += 1.0
            v[63] += 0.01  # never a zero vector
            out.append(v.tolist())
        return {"vectors": out, "usage": {"input_tokens": 0, "output_tokens": 0}}


@unittest.skipIf(MISSING, MISSING)
class CorpusAndQuestionsTest(unittest.TestCase):
    def test_every_item_is_one_short_line_with_a_source(self):
        items = mr.items()
        self.assertGreaterEqual(len(items), 100)
        self.assertEqual(len({i["id"] for i in items}), len(items))
        for item in items:
            self.assertLessEqual(len(item["text"]), 300, item["id"])
            self.assertNotIn("\n", item["text"])
            self.assertTrue(item["source"])
            self.assertIn(item["kind"], ("fact", "know_how", "episode"))
            self.assertIn(item["scope"], ("global", "workspace", "person", "project"))

    def _check(self, qs):
        by_id = {i["id"]: i for i in mr.items()}
        for q in qs:
            self.assertIn(q["group"], mr.GROUPS, q["id"])
            self.assertTrue(q["answers"], q["id"])
            for a in q["answers"]:
                self.assertIn(a, by_id, q["id"])
            if q["group"] in mr.UNSEEN:
                answer_words = {t for a in q["answers"] for t in tokenize(by_id[a]["text"])}
                self.assertFalse(set(tokenize(q["question"])) & answer_words, q["id"])
            if q["group"] == "persian":
                self.assertIsNone(re.search(r"[A-Za-z0-9]", q["question"]), q["id"])
                self.assertRegex(q["question"], r"[؀-ۿ]")

    def test_dev_questions_keep_their_rules(self):
        qs = mr.questions("dev")
        self._check(qs)
        self.assertEqual({g: sum(q["group"] == g for q in qs) for g in mr.GROUPS}, {"direct": 12, "reworded": 12, "persian": 12})

    def test_the_sealed_set_is_pinned_and_refused_without_its_flag(self):
        with self.assertRaises(PermissionError):
            mr.questions("sealed")
        self.assertEqual(mr.SEALED_HASH.read_text().split()[0], mr.sha256_of(mr.SEALED))
        qs = mr.questions("sealed", sealed_ok=True)
        self._check(qs)
        self.assertEqual(len(qs), 30)

    def test_a_changed_sealed_file_is_refused(self):
        scratch = Path(tempfile.mkdtemp())
        changed = scratch / "questions_sealed.json"
        changed.write_text(mr.SEALED.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with mock.patch.object(mr, "SEALED", changed), self.assertRaises(SystemExit):
            mr.questions("sealed", sealed_ok=True)


@unittest.skipIf(MISSING, MISSING)
class GateTest(unittest.TestCase):
    def _rows(self, reworded=0.5, persian=0.5, direct=0.0, ms=50.0, n=10):
        gains = {"direct": direct, "reworded": reworded, "persian": persian}
        rows = []
        for g in mr.GROUPS:
            for i in range(n):
                base = 0.5 if (i % 2 and g == "direct") else 0.0
                gain = gains[g] if isinstance(gains[g], float) else gains[g](i)
                rows.append({"id": f"{g}{i}", "group": g, "hybrid": base, "embed": min(1.0, max(0.0, base + gain)), "ms": ms})
        return rows

    def test_embeddings_enter_only_when_the_reworded_gain_is_clear(self):
        self.assertTrue(mr.verdict(self._rows(), "embed", "m")["verdict"].startswith("embeddings enter"))
        self.assertTrue(mr.verdict(self._rows(reworded=0.0), "embed", "m")["verdict"].startswith("keywords stay"))

    def test_persian_hits_never_hide_a_reworded_failure(self):
        """The review: pooled with Persian, 6 of 10 Persian found and no reworded gain passed."""
        out = mr.verdict(self._rows(reworded=0.0, persian=lambda i: 1.0 if i < 6 else 0.0), "embed", "m")
        self.assertFalse(out["checks"]["reworded_gain_clear"])
        self.assertTrue(out["verdict"].startswith("keywords stay"))

    def test_a_few_lucky_hits_are_not_a_gain(self):
        """The review: a few hits against a baseline at 0 give an interval above 0; the minimum gain refuses them."""
        out = mr.verdict(self._rows(reworded=lambda i: 0.5 if i < 5 else 0.0), "embed", "m")  # half of one answer on 5 of 10
        self.assertGreater(out["reworded_interval_90"][0], 0)
        self.assertFalse(out["checks"]["reworded_gain_clear"])

    def test_a_loss_on_direct_or_persian_questions_keeps_keywords(self):
        out = mr.verdict(self._rows(direct=-0.5), "embed", "m")
        self.assertFalse(out["checks"]["direct_not_worse"])
        rows = self._rows()
        for r in rows:
            if r["group"] == "persian":
                r["hybrid"], r["embed"] = 1.0, 0.0
        self.assertFalse(mr.verdict(rows, "embed", "m")["checks"]["persian_not_worse"])

    def test_a_slow_model_keeps_keywords(self):
        out = mr.verdict(self._rows(ms=mr.LATENCY_MS + 1), "embed", "m")
        self.assertFalse(out["checks"]["fast_enough"])
        self.assertTrue(out["verdict"].startswith("keywords stay"))

    def _dev(self, model, reworded, direct=0.5, persian=0.5, base_persian=0.0, ms=50.0):
        recall = {"hybrid": {"direct": 0.5, "reworded": 0.1, "persian": base_persian},
                  "embed": {"direct": direct, "reworded": reworded, "persian": persian},
                  "hybrid_embed": {"direct": direct, "reworded": reworded - 0.05, "persian": persian}}
        return {"model": model, "measurement": {"recall": recall, "median_query_ms": ms}}

    def test_the_choice_is_made_on_dev_with_every_candidate_measured(self):
        names = list(mr.CANDIDATES)
        self.assertIsNone(mr.choose([self._dev(names[0], 0.6)])["model"])  # the others not measured yet
        results = [self._dev(n, 0.4) for n in names]
        results[1] = self._dev(names[1], 0.7)
        self.assertEqual((mr.choose(results)["model"], mr.choose(results)["method"]), (names[1], "embed"))
        results[1] = self._dev(names[1], 0.9, direct=0.3)  # better on reworded, worse on direct: excluded
        self.assertNotEqual(mr.choose(results)["model"], names[1])
        results[1] = self._dev(names[1], 0.9, persian=0.0, base_persian=0.2)  # worse than the baseline on Persian: excluded
        self.assertNotEqual(mr.choose(results)["model"], names[1])
        tied = [self._dev(n, 0.4) for n in names]  # a tie goes to the smallest download
        self.assertEqual(mr.choose(tied)["model"], min(names, key=lambda n: mr.CANDIDATES[n]["mb"]))
        slow = [self._dev(n, 0.4, ms=mr.LATENCY_MS + 1) for n in names]
        self.assertIsNone(mr.choose(slow)["model"])

    def test_a_changed_gate_is_refused(self):
        scratch = Path(tempfile.mkdtemp())
        (scratch / "c").mkdir()
        (scratch / "c" / "manifest.json").write_text(json.dumps({**mr.manifest(), "sha256": "0" * 64}))
        with mock.patch.object(mr, "CAMPAIGN", scratch / "c"), self.assertRaises(SystemExit):
            mr.freeze()

    def test_the_gate_cannot_be_frozen_again_after_a_result(self):
        """The review: delete the manifest after seeing dev numbers, change the gate, freeze again."""
        scratch = Path(tempfile.mkdtemp())
        (scratch / "c" / "results").mkdir(parents=True)
        (scratch / "c" / "results" / "MRT-001_dev_x.json").write_text("{}")
        with mock.patch.object(mr, "CAMPAIGN", scratch / "c"), self.assertRaises(SystemExit):
            mr.freeze()

    def test_only_the_first_dev_result_under_the_manifest_counts(self):
        scratch = Path(tempfile.mkdtemp())
        folder = scratch / "c" / "results"
        folder.mkdir(parents=True)
        for i, (sha, ms) in enumerate([("old", 10.0), ("held", 900.0), ("held", 20.0)]):
            (folder / f"MRT-{i + 1:03d}_dev_m.json").write_text(json.dumps({"model": "m", "manifest_sha256": sha, "measurement": {"median_query_ms": ms}}))
        with mock.patch.object(mr, "CAMPAIGN", scratch / "c"):
            got = mr.dev_results("held")
        self.assertEqual([d["measurement"]["median_query_ms"] for d in got], [900.0])  # a rerun never re-rolls it

    def test_the_gate_is_frozen_in_the_repository(self):
        held = json.loads((mr.CAMPAIGN / "manifest.json").read_text())
        self.assertEqual(held["sha256"], mr.manifest()["sha256"])
        self.assertEqual(held["code_sha256"], mr.code_sha256())
        self.assertEqual(held["files"]["questions_sealed.json"], mr.sha256_of(mr.SEALED))


@unittest.skipIf(MISSING, MISSING)
class LocalOnlyTest(unittest.TestCase):
    def test_the_embedder_goes_to_this_machine_whatever_the_settings_said(self):
        paid = {"DCLAB_LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "x", "DCLAB_TIER_CHEAP_BASE_URL": "https://api.openai.com/v1",
                "DCLAB_TIER_CHEAP_API_KEY": "x", "DCLAB_TIER_CHEAP_MODEL": "text-embedding-3-small", "DCLAB_NO_LIVE_MODELS": "1"}
        with mock.patch.dict("os.environ", paid), mock.patch.object(mr, "model_digest", return_value="sha256:abcdef0123456789"):
            embedder = mr.local_embedder("bge-m3:567m", folder=Path(tempfile.mkdtemp()))
            from dclab_rnd.models import settings

            served = embedder.client.gateway.route("embedding")[0]
            self.assertTrue(settings.is_local(settings.tier(served)["base_url"]))
            self.assertEqual(settings.tier(served)["base_url"], mr.OLLAMA)
            self.assertEqual(embedder.model, "bge-m3:567m@sha256:abcde")  # named by the weights
        self.assertEqual(__import__("os").environ.get("DCLAB_NO_LIVE_MODELS"), "1" if "DCLAB_NO_LIVE_MODELS" in paid else None)

    def test_a_tier_elsewhere_is_refused(self):
        from dclab_rnd.models import settings

        with mock.patch.dict("os.environ", {}), mock.patch.object(settings, "is_local", return_value=False), self.assertRaises(SystemExit):
            mr.local_embedder("bge-m3:567m")


@unittest.skipIf(MISSING, MISSING)
class RunTest(unittest.TestCase):
    def test_prefixes_reach_the_model_and_name_the_vector_file(self):
        scratch = Path(tempfile.mkdtemp())
        client = Words()
        plain = Embedder(client, folder=scratch, model="m")
        prefixed = Embedder(client, folder=scratch, model="m", query_prefix="q: ", doc_prefix="d: ")
        index = mr.index()
        self.assertNotEqual(plain._path(index), prefixed._path(index))
        prefixed.records(index, build=True)
        prefixed.query("hello")
        self.assertTrue(all(t.startswith("d: ") for t in client.sent[:-1]))
        self.assertEqual(client.sent[-1], "q: hello")

    def _small_sealed(self, scratch):
        """A stand-in for the sealed set (the review: the suite never measures the real one)."""
        qs = [q for q in mr.questions("dev") if q["id"] in ("D01", "D02", "D13", "D14", "D25", "D26")]
        path = scratch / "sealed.json"
        path.write_text(json.dumps({"questions": qs}, ensure_ascii=False), encoding="utf-8")
        pin = scratch / "sealed.sha256"
        pin.write_text(mr.sha256_of(path) + "  sealed.json\n")
        return path, pin

    def test_a_dev_run_and_the_sealed_run_on_a_scripted_client(self):
        scratch = Path(tempfile.mkdtemp())
        campaign = scratch / "c"
        sealed, pin = self._small_sealed(scratch)
        names = list(mr.CANDIDATES)

        def made(model, folder=None):
            e = Embedder(Words(), folder=scratch, model=model, query_prefix=mr.CANDIDATES[model]["query_prefix"], doc_prefix=mr.CANDIDATES[model]["doc_prefix"])
            e.digest = "sha256:test"
            return e

        with mock.patch.object(mr, "CAMPAIGN", campaign), mock.patch.object(mr, "local_embedder", made), \
                mock.patch.object(mr, "SEALED", sealed), mock.patch.object(mr, "SEALED_HASH", pin), mock.patch.object(mr, "loaded_mb", return_value=None):
            mr.freeze()
            with self.assertRaises(SystemExit):
                mr.run_sealed(yes=True)  # nothing measured on dev yet: refused before the sealed file is opened
            for name in names:
                doc = mr.run_dev(name)
                m = doc["measurement"]
                self.assertEqual(set(m["recall"]), {*mr.KEYWORD, *mr.NEURAL})
                self.assertEqual(m["recall"]["bm25"]["persian"], 0.0)  # keywords cannot read Persian, and a listing is not a hit
                row = m["rows"][0]  # captured whole: each method's first items, the answers, the time
                self.assertEqual(set(row["top"]), {*mr.KEYWORD, *mr.NEURAL})
                self.assertLessEqual(len(row["top"]["embed"]), mr.K)
                self.assertIn("ms", row)
                self.assertIn("answers", row)
                self.assertTrue(doc["training_use"].startswith("allowed"))
                self.assertEqual(doc["digest"], "sha256:test")
                self.assertEqual(set(m["intervals_90"]["embed"]), set(mr.GROUPS))
            with self.assertRaises(SystemExit):
                mr.run_sealed(yes=False)
            out = mr.run_sealed(yes=True)
            self.assertIn(out["verdict"]["verdict"].split(":")[0], ("embeddings enter", "keywords stay"))
            self.assertTrue(out["training_use"].startswith("never"))
            self.assertTrue((campaign / "results" / "MRT-100_sealed.json").is_file())
            with self.assertRaises(SystemExit):
                mr.run_sealed(yes=True)  # read once

    def test_a_crash_during_the_sealed_run_is_not_a_second_chance(self):
        scratch = Path(tempfile.mkdtemp())
        sealed, pin = self._small_sealed(scratch)
        dev = [{"model": n, "manifest_sha256": None, "measurement": {"median_query_ms": 5.0, "recall": {
            "hybrid": {"direct": 0.0, "reworded": 0.0, "persian": 0.0}, "embed": {"direct": 1.0, "reworded": 1.0, "persian": 1.0}}}} for n in mr.CANDIDATES]
        made = lambda model, folder=None: Embedder(Words(), folder=scratch, model=model)  # noqa: E731
        with mock.patch.object(mr, "CAMPAIGN", scratch / "c"), mock.patch.object(mr, "local_embedder", made), \
                mock.patch.object(mr, "SEALED", sealed), mock.patch.object(mr, "SEALED_HASH", pin), \
                mock.patch.object(mr, "dev_results", return_value=dev), mock.patch.object(mr, "measure", side_effect=RuntimeError("out of memory")):
            mr.freeze()
            with self.assertRaises(RuntimeError):
                mr.run_sealed(yes=True)
            with self.assertRaises(SystemExit):
                mr.run_sealed(yes=True)

    def test_a_model_outside_the_candidates_is_refused(self):
        with self.assertRaises(SystemExit):
            mr.run_dev("some-other-model")


if __name__ == "__main__":
    unittest.main()
