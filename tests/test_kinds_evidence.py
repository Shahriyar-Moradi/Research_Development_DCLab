"""Package 16.14: agents and evidence per kind. Every kind names a research track that exists; its first campaign (the
kind's acceptance on three seeds of its replica) is stored with intervals and indexed as evidence of type ``kind``; a
method question about a kind is answered from that kind's own record; the intern can read a kind's checks through the
registry; trajectories carry the project's kind."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import kinds, retrieval
    from dclab_rnd.evidence_index import EvidenceIndex
except ImportError as error:  # the agent environment has no pandas
    MISSING = str(error)
else:
    MISSING = ""

RESULTS = ROOT / "evidence/campaigns/kinds_v1/results"
# a method question per kind in a person's own words, not the records' phrases (frozen here, written before the records
# were measured against them); its answer must cite the kind's own record
QUESTIONS = {
    "forecasting": "How do I evaluate a weekly sales forecast for many stores?",
    "time_to_event": "How should subscriptions that are still active be handled when predicting how long customers stay?",
    "ranking": "How do I test a recommender system offline?",
    "segmentation": "How do I choose the number of clusters for customer clustering?",
    "uplift": "How do I target a marketing campaign at customers who change their behaviour because of it?",
    "entity_tables": "How do I predict churn from separate customers, orders and tickets tables?",
    "places": "Should I use spatial cross validation for geographic data?",
    "streams": "How should I evaluate predictive maintenance alerts?",
    "embedded": "How should I split an image classification dataset?",
    "binary": "How should I validate a yes or no classifier on a customer table?",
    "multiclass": "How do I route support tickets to the right team with a classifier?",
    "regression": "How do I predict a delivery time in minutes?",
    "rare_event": "How do I detect fraud when only 1% of transactions are fraud?",
    "text_tabular": "How do I use a free-text column together with numeric columns?",
}

# a second set, written before the aliases were measured against it (16.14): reported, and required to reach most kinds
HELD_OUT = {
    "forecasting": "What baseline should a demand forecast beat?",
    "time_to_event": "How do I model churn timing when many customers have not churned yet?",
    "ranking": "Which metrics should I use for a product recommendation list?",
    "segmentation": "How do I know my customer segments are stable?",
    "uplift": "How can I measure the incremental effect of a promotion per customer?",
    "entity_tables": "How should I aggregate transactions per customer up to a prediction date?",
    "places": "Do nearby houses make random cross validation optimistic?",
    "streams": "How many alerts per day should an anomaly detector raise for machines?",
    "embedded": "Can duplicate photos inflate the accuracy of an image model?",
    "binary": "Should I use a time split for a churn model?",
    "multiclass": "Which metric should I use for a ticket category classifier with uneven classes?",
    "regression": "How do I evaluate a model predicting delivery minutes?",
    "rare_event": "How should I evaluate a classifier when positives are very rare?",
    "text_tabular": "How do I combine review text with numeric features?",
}

def latest_ids() -> dict:
    index = EvidenceIndex.load()
    out: dict = {}
    for r in sorted((r for r in index.records if r["type"] == "kind"), key=lambda r: r["record_id"]):
        out[r["metadata"]["problem_kind"]] = r["record_id"]  # the newest result of each kind
    return out


@unittest.skipIf(MISSING, MISSING)
class KindsEvidenceTest(unittest.TestCase):
    def test_every_kind_names_a_track_that_exists_and_has_its_campaign(self):
        for kind in kinds.all_kinds():
            self.assertTrue(kind.evidence_track, kind.key)
            self.assertTrue((ROOT / "research" / kind.evidence_track / "README.md").exists(), kind.key)
            found = sorted(RESULTS.glob(f"KND-*_{kind.key}.json"))
            self.assertTrue(found, kind.key)
            doc = json.loads(found[-1].read_text(encoding="utf-8"))
            rate = doc["evidence"]["rate"]
            self.assertEqual(rate["seeds"], 3)
            low, high = rate["ci95"]  # the interval over seeds, not over correlated checks
            self.assertTrue(low <= rate["seeds_all_held"] / rate["seeds"] <= high, kind.key)
            self.assertEqual(doc["track"], kind.evidence_track)
            self.assertTrue(doc["provenance"].get("code_committed"), kind.key)  # run from committed code
            for d in doc["evidence"]["decisions"]:
                if d["open"]:  # a decision waiting on the owner is never counted as failed
                    self.assertNotIn(d["decision"], " ".join(c["statement"] for c in doc["claims"] if c["kind"] == "risk"))

    def test_each_kinds_record_is_in_the_evidence_index(self):
        index = EvidenceIndex.load()
        by_kind = {r["metadata"]["problem_kind"]: r for r in index.records if r["type"] == "kind"}
        self.assertEqual(set(by_kind), {k.key for k in kinds.all_kinds()})

    def test_a_method_question_about_a_kind_cites_that_kinds_evidence(self):
        from dclab_rnd.agents import default_registry

        ids = latest_ids()
        registry = default_registry()
        for name, questions in (("worded", QUESTIONS), ("held out", HELD_OUT)):
            missed = {k: [h["record_id"] for h in retrieval.search(q, k=5)] for k, q in questions.items()}
            missed = {k: cited for k, cited in missed.items() if ids[k] not in cited}
            # measured at 13 of 14 for each set (16.14); keyword search, so a near miss is possible: at least 12
            self.assertLessEqual(len(missed), 2, (name, missed))
        # the same question through the tool the intern and /mcp call
        got = registry.call("search_evidence", {"query": QUESTIONS["places"], "type": "kind"}, scope="project")
        self.assertIn(ids["places"], [r["record_id"] for r in got["results"]])

    def test_the_intern_reads_a_kinds_checks_through_the_registry(self):
        from dclab_rnd.agents import default_registry
        from dclab_rnd.tools import kind_checks

        registry = default_registry()
        self.assertIn("kind_checks", registry.names("project"))
        self.assertEqual(registry.get("kind_checks").effect, "read")
        out = kind_checks("forecasting")
        self.assertEqual(out["track"], "time-series-forecasting")
        self.assertTrue(out["evidence"] and out["evidence"][0]["record_id"].startswith("KND-"))
        self.assertTrue(out["capability"]["decisions"])
        self.assertIn("PIT-005", [c["record_id"] for c in out["checks"]])
        self.assertIn("error", kind_checks("no_such_kind"))

    def test_trajectories_carry_the_projects_kind(self):
        from dclab_rnd.studio.trajectories import problem_kind

        self.assertEqual(problem_kind({"solution": {"target": "y", "task": "regression", "prediction_moment": "When the flat is listed",
                                                    "spatial": {"lat_column": "lat", "lon_column": "lon", "block_km": 5}}, "pack": "maps"}), "places")
        self.assertIsNone(problem_kind({}))


if __name__ == "__main__":
    unittest.main()
