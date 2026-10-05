"""Evidence library routes: the live index in the page's shape, and answers that keep only checked sentences."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agentic.pages import evidence  # noqa: E402


class Scripted:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def complete(self, messages, max_tokens=0):
        self.calls.append(messages)
        if self.error:
            raise RuntimeError(self.error)
        return {"content": self.reply}


class EvidenceFunctionTests(unittest.TestCase):
    def test_library_is_the_index_in_the_page_shape(self):
        lines = [json.loads(l) for l in evidence.INDEX.read_text(encoding="utf-8").splitlines() if l.strip()]
        lib = evidence.library()
        self.assertEqual(lib["total"], len(lines))
        self.assertEqual([r["id"] for r in lib["records"]], [r["record_id"] for r in lines])
        first = lib["records"][0]
        self.assertEqual(set(first), {"id", "type", "title", "text", "citations", "meta"})
        self.assertLessEqual(max(len(r["citations"]) for r in lib["records"]), 6)
        self.assertEqual(sum(lib["counts"].values()), lib["total"])
        self.assertEqual(lib["source"], "evidence/knowledge/rag/records.jsonl")

    def test_library_follows_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            row = {"record_id": "PIT-900", "type": "pitfall", "title": "T", "text": "x", "metadata": {"dataset": "d", "nested": {"a": 1}}}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            lib = evidence.library(path)
            self.assertEqual((lib["total"], lib["records"][0]["meta"]), (1, {"dataset": "d"}))
            os.utime(path, (1, 1))
            path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "record_id": "PIT-901"}) + "\n", encoding="utf-8")
            self.assertEqual(evidence.library(path)["total"], 2)
            self.assertEqual(evidence.library(Path(tmp) / "missing.jsonl")["total"], 0)

    def test_without_a_model_the_answer_is_the_closest_records(self):
        a = evidence.ask("Should I oversample before splitting?")
        self.assertEqual((a["mode"], a["sentences"]), ("records", []))
        self.assertEqual(a["records"][0]["id"], "PIT-003")
        weak = evidence.ask("Do graph neural networks help for road speed?")
        self.assertEqual((weak["covered"], weak.get("weak")), (False, True))
        self.assertTrue(weak["note"].startswith(evidence.NOT_COVERED))
        self.assertEqual(evidence.ask("zzqx")["mode"], "none")

    def test_check_keeps_only_cited_sentences_with_known_numbers(self):
        hits = [{"record_id": "PIT-003", "title": "Oversampling", "text": "Holdout ROC-AUC rose by 0.3003 on 11 datasets."}]
        kept, dropped = evidence.check("No. It rose by 0.3003 [PIT-003]. It rose by 0.5 [PIT-003]. See [DCLAB-R99]. "
                                       "It held on 11 datasets [PIT-003].", hits)
        self.assertEqual([k["text"] for k in kept], ["It rose by 0.3003.", "It held on 11 datasets."])
        self.assertEqual(dropped, 3)

    def test_model_answers_are_checked_and_errors_fall_back(self):
        good = Scripted("Resample only inside the training folds [PIT-003]. It is 99.9% safe [PIT-003].")
        a = evidence.ask("Should I oversample before splitting?", good)
        self.assertEqual((a["mode"], a["covered"], a["dropped"]), ("model", True, 1))
        self.assertEqual(a["sentences"][0]["cites"], ["PIT-003"])
        self.assertIn("PIT-003", good.calls[0][1]["content"])
        self.assertEqual(evidence.ask("Should I oversample before splitting?", Scripted(evidence.NOT_COVERED))["covered"], False)
        failed = evidence.ask("Should I oversample before splitting?", Scripted(error="APIError: the model request failed"))
        self.assertEqual((failed["mode"], failed["records"][0]["id"]), ("records", "PIT-003"))
        unchecked = evidence.ask("Should I oversample before splitting?", Scripted("Trust me, it is fine."))
        self.assertEqual((unchecked["mode"], unchecked["sentences"]), ("records", []))


class EvidenceApiTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            self.skipTest(f"Studio dependencies not installed: {error}")
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}

    def test_routes(self):
        lib = self.client.get("/api/evidence").json()
        self.assertEqual(lib["total"], len(lib["records"]))
        self.assertEqual(self.client.get("/api/evidence/PIT-003").json()["record_id"], "PIT-003")  # the single-record route still works
        self.assertEqual(self.client.post("/api/evidence/ask", json={"question": "oversample?"}).status_code, 403)
        self.assertEqual(self.client.post("/api/evidence/ask", json={"question": " "}, headers=self.h).status_code, 422)
        self.assertEqual(self.client.post("/api/evidence/ask", json={"question": "x" * 501}, headers=self.h).status_code, 422)
        a = self.client.post("/api/evidence/ask", json={"question": "Should I oversample before splitting?"}, headers=self.h).json()
        self.assertEqual((a["mode"], a["records"][0]["id"]), ("records", "PIT-003"))


if __name__ == "__main__":
    unittest.main()
