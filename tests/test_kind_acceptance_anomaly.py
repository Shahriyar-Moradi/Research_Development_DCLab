"""Package 20.7, in the engine: anomaly detection without labels as its own kind. The owner's morning on the replica
(deliveries, no label) ends with a ranked review list, its evaluation on planted anomalies and its limits; the rows to
review reach the final stage only; confirmed alerts become labels and the problem a rare-event one."""

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd

    from dclab_core import anomaly
    from dclab_rnd import kinds
    from dclab_rnd.kinds import acceptance
    from dclab_rnd.studio.solution import Solution
    from tests.test_engine_machines import before_final, run_case
except ImportError as error:  # pydantic and pandas are not in every environment
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class AnomalyKindTest(unittest.TestCase):
    def test_the_mornings_replica_ends_with_a_ranked_review_list_its_evaluation_and_its_limits(self):
        result = acceptance.run(kinds.get("anomaly"))
        self.assertEqual(result["status"], "supported", result)
        self.assertTrue(all(c["ok"] for c in result["checks"]), [c for c in result["checks"] if not c["ok"]])

    def test_the_rows_to_review_reach_the_final_stage_only(self):
        case = kinds.get("anomaly").case(0)
        f = case.solution["kind_fields"]
        frm = anomaly.split(case.table, f["time_column"])["split"]["holdout_from"]
        table = case.table.copy()
        later = pd.to_datetime(table[f["time_column"]]) >= pd.Timestamp(frm)
        table.loc[later, "biker_fare"] = table.loc[later, "biker_fare"] * 7 + 3
        self.assertTrue(later.any())
        _, _, a = run_case("anomaly", case)
        _, _, b = run_case("anomaly", dataclasses.replace(case, table=table))
        first, second = before_final(a), before_final(b)
        for stage in first:
            self.assertEqual(first[stage], second[stage], f"the {stage} stage read the rows to review")
        self.assertNotEqual(a["final"]["evidence"]["review_list"], b["final"]["evidence"]["review_list"])

    def test_a_solution_reads_no_outcome_and_other_kinds_still_need_one(self):
        base = {"task": "binary", "prediction_moment": "After delivery, in the daily audit of the day's rides."}
        Solution(**base, kind="anomaly", labels="none", kind_fields={"entity_column": "biker_id"})
        with self.assertRaises(ValueError):
            Solution(**base, target="is_fraud", kind="anomaly")  # an outcome column is another kind
        with self.assertRaises(ValueError):
            Solution(**base)  # a binary problem learns from its outcome
        with self.assertRaises(ValueError):
            Solution(**base, kind="anomaly", labels="partial")  # with labels it is a rare-event problem
        with self.assertRaises(ValueError):
            Solution(**base, target="is_fraud", labels="none")  # an outcome with no labels is anomaly detection

    def test_confirmed_alerts_become_labels_and_the_problem_a_rare_event_one(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        home = Path(tempfile.mkdtemp())
        client = TestClient(create_app(home))
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        case = kinds.get("anomaly").case(0)
        store, pid = acceptance.build(case, home / "projects")  # where the server keeps the workspace's projects
        keys = list(case.table["order_id"].astype(str).head(3))
        r = client.post(f"/api/projects/{pid}/review", json={"confirmed": keys[:2]}, headers=h)
        self.assertEqual(r.status_code, 200, r.text)
        r = client.post(f"/api/projects/{pid}/review", json={"confirmed": keys[2:]}, headers=h)  # a second round adds to the first
        self.assertEqual(r.status_code, 200, r.text)
        project = store.get(pid)
        self.assertEqual(project["labels"]["key"], ["order_id"])
        self.assertEqual((project["suggestion"]["target"], project["suggestion"]["labels"]), ("is_case", "partial"))
        rows = pd.read_csv(store.data_dir(pid) / "reviewed_cases.csv", dtype=str)
        self.assertEqual(sorted(rows["order_id"]), sorted(keys))
        self.assertIsNone(project.get("pack"))  # no pack decides it: the suggestion names the kind
        moved = Solution(**{k: v for k, v in project["suggestion"].items() if k in Solution.model_fields},
                         prediction_moment=case.moment).problem_kind(project.get("pack"))
        self.assertEqual(moved.key, "rare_event")
        self.assertIn("picked by the detector", project["suggestion"]["label_provenance"])  # the labels' selection is said
        self.assertEqual(len(project["review"]["rounds"]), 2)
        self.assertEqual(client.post(f"/api/projects/{pid}/review", json={"confirmed": []}, headers=h).status_code, 422)  # nothing named
        before = (store.data_dir(pid) / "reviewed_cases.csv").read_text()
        r = client.post(f"/api/projects/{pid}/review", json={"confirmed": ["nope-1", "nope-2"]}, headers=h)
        self.assertEqual(r.status_code, 409, r.text)  # keys the table does not hold: refused, and nothing written
        self.assertEqual((store.data_dir(pid) / "reviewed_cases.csv").read_text(), before)

    def test_the_review_list_survives_the_first_confirmation(self):
        from dclab_rnd.kinds.anomaly import confirm

        case = kinds.get("anomaly").case(0)
        store, pid, records = run_case("anomaly", case)
        listed = [x["key"] for x in records["final"]["evidence"]["review_list"]]
        project = confirm(store, pid, listed[:2], "order_id", rejected=listed[2:4])
        self.assertEqual([x["key"] for x in project["review"]["review_list"]], listed)  # the other alerts are still to review
        self.assertEqual(project["review"]["rounds"][0]["rejected"], listed[2:4])


if __name__ == "__main__":
    unittest.main()
