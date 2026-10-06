"""Evidence search, measured (package A5.1): the question set, recall at 5 per method and group, and the choice rule."""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import retrieval, tools  # noqa: E402


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = retrieval.measure()

    def test_sixty_questions_whose_answers_exist(self):
        qs = retrieval.questions()
        known = {r["record_id"] for r in tools._index().records}
        self.assertEqual(len(qs), 60)
        self.assertEqual(len({q["id"] for q in qs}), 60)
        self.assertTrue(all(q["answers"] and set(q["answers"]) <= known for q in qs))
        self.assertEqual(set(self.m["groups"]), {"rule", "paraphrase", "pitfall", "precedent", "experiment", "workflow"})

    def test_recall_is_what_the_stored_measurement_says(self):
        stored = json.loads(sorted(retrieval.RESULTS.glob("RET-*.json"))[-1].read_text())
        self.assertEqual(stored["index_signature"], retrieval._signature(), "the index changed: run make retrieval-eval")
        self.assertEqual(self.m["recall"]["bm25"], stored["measurement"]["recall"]["bm25"])  # keyword search: exactly the same
        for method in ("lsa", "hybrid"):  # the SVD may differ in the last digits between scikit-learn versions
            for group, value in stored["measurement"]["recall"][method].items():
                self.assertAlmostEqual(self.m["recall"][method][group], value, delta=0.02, msg=f"{method} {group}")

    def test_the_hybrid_is_used_only_when_it_is_not_worse_on_any_group(self):
        groups = ["rule", "paraphrase"]
        self.assertEqual(retrieval.choose({"bm25": {"rule": 0.8, "paraphrase": 0.2, "overall": 0.5}, "hybrid": {"rule": 0.73, "paraphrase": 0.5, "overall": 0.6}}, groups), "bm25")
        self.assertEqual(retrieval.choose({"bm25": {"rule": 0.8, "paraphrase": 0.2, "overall": 0.5}, "hybrid": {"rule": 0.8, "paraphrase": 0.3, "overall": 0.55}}, groups), "hybrid")
        self.assertEqual(self.m["chosen"], "bm25")  # today the hybrid is worse on the rules group

    def test_the_product_searches_by_the_chosen_method(self):
        question = "Should I oversample before splitting?"
        self.assertEqual([h["record_id"] for h in retrieval.search(question)], [h["record_id"] for h in tools._index().search(question, k=5)])
        with mock.patch.object(retrieval, "chosen", return_value="hybrid"):
            hits = retrieval.search(question)
        self.assertEqual(len(hits), 5)
        self.assertTrue(all({"record_id", "title", "text", "score", "method"} <= set(h) for h in hits))
        self.assertIn("PIT-003", [h["record_id"] for h in hits])

    def test_a_paraphrase_shares_no_word_with_the_records_that_answer_it(self):
        from dclab_rnd.evidence_index import tokenize
        by = {r["record_id"]: r for r in tools._index().records}
        for q in retrieval.questions():
            if q["group"] == "paraphrase":
                for answer in q["answers"]:
                    shared = set(tokenize(q["question"])) & set(tokenize(by[answer]["title"] + " " + by[answer]["text"]))
                    self.assertEqual(shared, set(), (q["id"], answer))

    def test_the_evidence_page_searches_its_own_index_by_the_measured_method(self):
        from dclab_rnd.agentic.pages import evidence
        seen = []
        real = retrieval.search
        with mock.patch.object(retrieval, "search", side_effect=lambda q, k=5, index=None: seen.append(index) or real(q, k, index=index)):
            evidence.ask("Should I oversample before splitting?")
        self.assertTrue(seen and seen[0] is not None)  # the page's index (reloaded when the file changes), not a process-wide copy
        with mock.patch.object(retrieval, "chosen", return_value="hybrid"):
            a = evidence.ask("How often can I peek at the sealed exam?")
            strong = evidence.ask("Should I oversample before splitting?")
        self.assertEqual(strong.get("weak"), None)  # a strong keyword match anywhere in the hits is not a weak answer
        self.assertIn(a["mode"], ("records", "none"))


if __name__ == "__main__":
    unittest.main()
