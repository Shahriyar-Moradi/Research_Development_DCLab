"""Package 16.15 C, several tables in the engine: a table of customers with its orders and tickets attached as related
tables runs the entity machine. Every training row and target is built from the events before the holdout's cutoff;
the holdout is built and scored once, in the final stage; the model is chosen on the latest training cutoff; a join that
repeats rows is refused; a related table is checked against its fingerprint and never written over another file."""

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd

    from dclab_core import entities
    from dclab_rnd import kinds
    from dclab_rnd.kinds import acceptance
    from dclab_rnd.studio import ProjectStore, data as sd
    from dclab_rnd.studio.solution import Solution
    from unittest import mock

    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import engine
    from tests.test_engine_machines import before_final, built, run_case
except ImportError as error:  # pydantic and pandas are not in every environment
    MISSING = str(error)
else:
    MISSING = ""

FIELDS = {"key_column": "customer_id", "target_table": "orders", "horizon_days": 30, "cutoffs": ["2025-01-01", "2025-02-01"],
          "event_tables": [{"name": "orders", "time_column": "ordered_at"}]}
MOMENT = "On each cutoff date, from what the customer did before it."


class SimpleVerdict:
    allowed, message = False, "Blocked: a stage is running."


def engine_machine(store, pid):
    """The entity machine the engine would run on the project now."""
    from dclab_rnd.studio.machine import prepare

    project = store.get(pid)
    solution = Solution(**project["solution"])
    return prepare(store, project, solution, kinds.get("entity_tables")).machine


@unittest.skipIf(MISSING, MISSING)
class EntityTablesInTheEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kind = kinds.get("entity_tables")
        cls.case = cls.kind.case(0)
        cls.store, cls.pid, cls.records = run_case("entity_tables", cls.case)

    def test_the_replica_ends_with_the_holdout_scored_once_against_recency(self):
        result = acceptance.run(self.kind)
        self.assertEqual(result["status"], "supported", result)
        self.assertTrue(all(c["ok"] for c in result["checks"]), [c for c in result["checks"] if not c["ok"]])

    def test_the_holdout_reaches_the_final_stage_only(self):
        f = self.case.solution["kind_fields"]
        parts = self.records["data"]["evidence"]["split"]
        start, end = pd.Timestamp(parts["holdout"]), pd.Timestamp(parts["holdout"]) + pd.Timedelta(days=f["horizon_days"])
        orders = self.case.tables["orders"].copy()
        inside = (pd.to_datetime(orders["ordered_at"]) >= start) & (pd.to_datetime(orders["ordered_at"]) < end)
        self.assertGreater(int(inside.sum()), 50)
        rng = np.random.default_rng(1)  # the holdout's orders go to customers drawn at random: its targets change, nothing before it does
        orders.loc[inside, "customer_id"] = rng.choice(self.case.table["customer_id"].to_numpy(), int(inside.sum()))
        _, _, other = run_case("entity_tables", dataclasses.replace(self.case, tables={**self.case.tables, "orders": orders}))
        first, second = before_final(self.records), before_final(other)
        for stage in first:
            self.assertEqual(first[stage], second[stage], f"the {stage} stage read the holdout's horizon")
        self.assertNotEqual(self.records["final"]["evidence"]["holdout_metrics"], other["final"]["evidence"]["holdout_metrics"])
        self.assertEqual(self.records["final"]["evidence"]["holdout_uses_in_this_project"], 1)

    def test_the_model_is_chosen_on_validation_before_the_holdout_is_read(self):
        models, final = self.records["models"]["evidence"], self.records["final"]["evidence"]
        self.assertEqual(final["model"], models["selected_model"])
        self.assertEqual({r["model"] for r in models["ranking"]}, set(entities.CANDIDATES))
        self.assertTrue(final["beats_recency"])
        self.assertEqual(self.records["features"]["evidence"]["rows_that_read_after_the_holdout"], 0)
        self.assertEqual(self.records["features"]["evidence"]["events_after_cutoff_read"], 0)

    def test_a_table_changed_on_disk_is_refused_and_attached_again_clears_the_stages(self):
        store, pid = built("entity_tables", self.case)
        with private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.run_all(store, pid)
        orders = self.case.tables["orders"]
        later = orders.assign(ordered_at=(pd.to_datetime(orders["ordered_at"]) + pd.Timedelta(days=40)).astype(str))
        later.to_csv(store.data_dir(pid) / "orders.csv", index=False)  # the log now runs 40 days longer
        with self.assertRaises(sd.DataError):  # changed on disk without attaching it again: refused by its fingerprint
            sd.load_tables(store, store.get(pid), ["orders"])
        sd.attach_table(store, pid, "orders.csv", "orders")
        self.assertEqual({s.get("status") for s in store.get(pid)["stages"].values()}, {"pending"})  # a machine may read it: rerun
        self.assertIsNone(store.read_stage(pid, "final"))
        with private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.execute(store, pid, "data")  # the final stage would now ask a person: the holdout was read once already
        self.assertNotEqual(store.read_stage(pid, "data")["evidence"]["split"]["holdout"], self.records["data"]["evidence"]["split"]["holdout"])

    def test_the_final_stage_refuses_a_holdout_the_data_stage_did_not_lock(self):
        machine = engine_machine(self.store, self.pid)
        moved = {"data": {"evidence": {"split": {**self.records["data"]["evidence"]["split"], "holdout": "2025-10-28"}}}}
        with self.assertRaises(ValueError):
            machine.final("E-1", {**self.records, **moved})

    def test_a_related_table_is_never_written_over_another_file(self):
        project = self.store.get(self.pid)
        self.assertIn(project["data"]["filename"], sd.files_in_use(project))
        self.assertNotIn(project["tables"]["orders"]["filename"], sd.files_in_use(project, but="orders"))
        with self.assertRaises(sd.DataError):
            sd.attach_table(self.store, self.pid, project["data"]["filename"], "customers_again")
        with self.assertRaises(sd.DataError):
            sd.load_tables(self.store, project, ["payments"])

    def test_the_route_attaches_a_table_and_never_writes_over_the_main_one(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        pid = client.post("/api/projects", json={"name": "Shop", "goal": "g"}, headers=h).json()["id"]
        main = self.case.table.to_csv(index=False).encode()
        self.assertEqual(client.put(f"/api/projects/{pid}/data?filename=customers.csv", content=main, headers=h).status_code, 200)
        orders = self.case.tables["orders"].head(500).to_csv(index=False).encode()
        r = client.put(f"/api/projects/{pid}/tables/orders", content=orders, headers=h)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["tables"]["orders"]["rows"], 500)
        r = client.put(f"/api/projects/{pid}/tables/orders2?filename=customers.csv", content=orders, headers=h)
        self.assertEqual(r.status_code, 409, r.text)
        self.assertEqual(client.get(f"/api/projects/{pid}").json()["data"]["rows"], len(self.case.table))  # the main table untouched

    def test_a_broken_reupload_leaves_the_attached_table_as_it_was(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        root = Path(tempfile.mkdtemp())
        client = TestClient(create_app(root))
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        pid = client.post("/api/projects", json={"name": "Shop", "goal": "g"}, headers=h).json()["id"]
        client.put(f"/api/projects/{pid}/data?filename=customers.csv", content=self.case.table.to_csv(index=False).encode(), headers=h)
        orders = self.case.tables["orders"].head(300).to_csv(index=False).encode()
        first = client.put(f"/api/projects/{pid}/tables/orders", content=orders, headers=h).json()["tables"]["orders"]
        r = client.put(f"/api/projects/{pid}/tables/orders", content=b"only_one_column\n1\n2\n", headers=h)
        self.assertEqual(r.status_code, 400, r.text)
        project = client.get(f"/api/projects/{pid}").json()
        self.assertEqual(project["tables"]["orders"], first)
        import hashlib

        kept = [p for p in root.rglob("orders.csv") if pid in str(p)]
        self.assertEqual(len(kept), 1)  # still there, its fingerprint intact
        self.assertEqual(hashlib.sha256(kept[0].read_bytes()).hexdigest(), first["sha256"])
        self.assertEqual(list(root.rglob("*.part")), [])

    def test_no_model_reads_a_column_the_audit_kept_out(self):
        case = dataclasses.replace(self.case, table=self.case.table.assign(refund_at_export=np.arange(len(self.case.table)) % 7),
                                   solution={**self.case.solution, "forbidden": [{"column": "refund_at_export", "reason": "written at export"}]})
        _, _, records = run_case("entity_tables", case)
        self.assertIn("refund_at_export", records["leakage"]["evidence"]["kept_out"])
        self.assertNotIn("refund_at_export", records["features"]["evidence"]["features"])
        machine = engine_machine(*built("entity_tables", case))  # the library itself: an excluded column is never fitted
        picked = entities.choose(machine.view(), machine.parts, key="customer_id", exclude=["refund_at_export"])
        self.assertNotIn("refund_at_export", picked["features"])
        self.assertIn("refund_at_export", entities.choose(machine.view(), machine.parts, key="customer_id")["features"])

    def test_a_kept_column_that_follows_the_log_after_the_holdout_is_said_and_scored_as_kept(self):
        parts = self.records["data"]["evidence"]["split"]
        orders = self.case.tables["orders"]
        ids = self.case.table["customer_id"]
        after = ids.map(orders[pd.to_datetime(orders["ordered_at"]) >= pd.Timestamp(parts["holdout"])].groupby("customer_id").size()).fillna(0)
        total = ids.map(orders.groupby("customer_id").size()).fillna(0)  # at the export: the whole count, the holdout's months weighed six times
        table = self.case.table.assign(export_score=(total + 6 * after).to_numpy(float))
        _, _, records = run_case("entity_tables", dataclasses.replace(self.case, table=table))
        self.assertNotIn("export_score", records["leakage"]["evidence"]["kept_out"])  # invisible from the training period
        self.assertIn("export_score", records["features"]["evidence"]["features"])
        self.assertEqual(records["final"]["evidence"]["kept_but_written_later_at_the_holdout"], ["export_score"])
        self.assertTrue(any("export_score" in n["title"] for n in records["final"]["evidence"]["notes_for_person"]))

    def test_the_acceptance_fails_on_a_look_ahead_inside_the_horizon(self):
        import copy
        from types import SimpleNamespace

        records = copy.deepcopy(self.records)
        records["features"]["evidence"]["events_after_cutoff_read"] = 12  # the windows read past the cutoff, not past the holdout

        def run(r):
            return SimpleNamespace(records=r, case=self.case, evidence=lambda stage: r[stage]["evidence"])

        name = "in the engine no training feature reads an event after its cutoff, nor anything after the holdout's"
        verdict = lambda r: next(c["ok"] for c in self.kind.accept_engine(run(r)) if c["decision"] == name)  # noqa: E731
        self.assertTrue(verdict(self.records))
        self.assertFalse(verdict(records))

    def test_a_refused_attach_never_deletes_an_attached_table(self):
        store, pid = built("entity_tables", self.case)
        before = store.get(pid)["tables"]["orders"]
        refused = SimpleVerdict()
        with mock.patch("dclab_rnd.studio.graph.check", return_value=refused), mock.patch("dclab_rnd.studio.graph.log"):
            with self.assertRaises(sd.DataError):
                sd.attach_table(store, pid, "orders.csv", "orders")
        self.assertTrue((store.data_dir(pid) / "orders.csv").exists())
        self.assertEqual(store.get(pid)["tables"]["orders"], before)
        self.assertEqual(len(sd.load_tables(store, store.get(pid), ["orders"])["orders"]), len(self.case.tables["orders"]))

    def test_a_related_table_needs_the_main_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProjectStore(Path(tmp))
            project = store.create("Empty", "general", MOMENT)
            pd.DataFrame({"customer_id": [1], "ordered_at": ["2025-01-01"]}).to_csv(store.data_dir(project["id"]) / "orders.csv", index=False)
            with self.assertRaises(sd.DataError):
                sd.attach_table(store, project["id"], "orders.csv", "orders")


@unittest.skipIf(MISSING, MISSING)
class EntitySolutionTest(unittest.TestCase):
    def test_a_solution_names_its_event_tables_and_reads_no_outcome_column(self):
        Solution(task="binary", prediction_moment=MOMENT, kind="entity_tables", kind_fields=FIELDS)
        for fields, why in ((dict(FIELDS, event_tables=[]), "no event table"),
                            (dict(FIELDS, target_table="tickets"), "the target is not an event table"),
                            (dict(FIELDS, joined_tables=["orders"]), "a table both events and columns")):
            with self.assertRaises(ValueError, msg=why):
                Solution(task="binary", prediction_moment=MOMENT, kind="entity_tables", kind_fields=fields)
        with self.assertRaises(ValueError):  # the target is built from the events
            Solution(target="churned", task="binary", prediction_moment=MOMENT, kind="entity_tables", kind_fields=FIELDS)
        with self.assertRaises(ValueError):  # the machine chooses and reports by ROC-AUC
            Solution(task="binary", prediction_moment=MOMENT, kind="entity_tables", kind_fields=FIELDS, metric="f1")

    def test_the_kind_lists_the_tables_its_machine_reads(self):
        kind = kinds.get("entity_tables")
        solution = Solution(task="binary", prediction_moment=MOMENT, kind="entity_tables",
                            kind_fields=dict(FIELDS, event_tables=[*FIELDS["event_tables"], {"name": "tickets", "time_column": "opened_at"}],
                                             joined_tables=["addresses"]))
        self.assertEqual(kind.tables_of(solution), ["orders", "tickets", "addresses"])
        self.assertEqual(kind.table_columns_of(solution)["tickets"], ["customer_id", "opened_at"])
        self.assertEqual(kind.table_columns_of(solution)["addresses"], ["customer_id"])


if __name__ == "__main__":
    unittest.main()
