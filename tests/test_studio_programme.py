"""Package 24.5, the studio side: a project's programme. Built from the draft's confirmed plan, run as a job behind the graph's move, the holdout read once, a second read a
person's written decision, the pipeline and the code export covering it, and the exported code reproducing the number."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import pandas as pd
    from fastapi.testclient import TestClient

    from dclab_core import datacard, problem_map as pm, programme as core, solution_map as sm_mod, solutions
    from dclab_core.replicas import fraud_programme as fp
    from dclab_rnd.studio import data as sd, graph, pipeline as studio_pipeline, programme as sp
    from dclab_rnd.studio.store import ProjectStore
except ImportError as error:  # the agent environment may lack pandas
    MISSING = str(error)
else:
    MISSING = ""

WORDS = ("Find the fake rides and fake orders our bikers and customers make. We have a few confirmed cases, nothing else is labelled, "
         "and the team can review about 2% of the rides each day.")
SMALL = dict(n_orders=14_000, n_accounts=1_000, n_bikers=90, n_rings=6, n_markers=10)


def plan_for(rides, accounts=None):
    """What the draft keeps at the build: the confirmed map, its routes and the solution map (drawn from the tables the draft had: the rides, and the accounts when given)."""
    m = pm.decompose(WORDS)
    tables = {"rides": rides, **({"accounts": accounts} if accounts is not None else {})}
    card = datacard.card_of(tables, roles={"rides": "main", "accounts": "related"})
    routes = solutions.routes(m, card)
    sm = sm_mod.solution_map(m, routes, card)
    return {"problem_map": json.loads(m.model_dump_json()), "routes": {k: json.loads(v.model_dump_json()) for k, v in routes.items()},
            "solution_map": json.loads(sm.model_dump_json()), "confirmed": True}


def make_project(store, seed: int = 0, tables: bool = True, labels: bool = True, plan_with_accounts: bool = True, rides_extra: dict | None = None):
    r = fp.generate(seed=seed, **SMALL)
    for name, values in (rides_extra or {}).items():
        r.rides[name] = values(r.rides) if callable(values) else values
    project = store.create("Fraud programme", "general", WORDS)
    pid = project["id"]
    d = store.data_dir(pid)
    r.rides.to_csv(d / "rides.csv", index=False)
    sd.attach_data(store, pid, "rides.csv")
    if labels:
        r.cases.to_csv(d / "cases.csv", index=False)
        sd.attach_labels(store, pid, "cases.csv", target="is_case", key=["order_id"])
    if tables:
        r.accounts.to_csv(d / "accounts.csv", index=False)
        sd.attach_table(store, pid, "accounts.csv", "accounts")
    project = store.get(pid)
    project["programme_plan"] = plan_for(r.rides, r.accounts if plan_with_accounts else None)
    project["suggestion"] = {**(project.get("suggestion") or {}), "alert_budget": 0.02}
    store.save(project)
    return pid, r


@unittest.skipIf(MISSING, MISSING)
class RunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = ProjectStore(Path(tempfile.mkdtemp()))
        cls.pid, cls.replica = make_project(cls.store)
        cls.view = sp.run(cls.store, cls.pid)

    def test_the_programme_runs_and_the_holdout_is_read_once(self):
        project = self.store.get(self.pid)
        self.assertEqual(self.view["state"], "composed")
        self.assertEqual(self.view["holdout_reads"], 1)
        self.assertEqual(project["holdout_uses"], 1)
        p = sp.stored(project)
        self.assertEqual(p.holdout.reads, 1)
        self.assertEqual(core.validate(p), [])
        self.assertEqual({l.block: l.status for l in p.leaves}["blk3"], "built")

    def test_the_view_has_a_section_per_leaf_and_nothing_is_stale(self):
        titles = [s["title"] for s in self.view["sections"]]
        self.assertTrue(any(t.startswith("blk1:") for t in titles))
        self.assertIn("The holdout, read once", titles)
        self.assertEqual(self.view["stale"], [])

    def test_the_run_is_in_the_activity_the_audit_trail_and_the_graph(self):
        project = self.store.get(self.pid)
        kinds = [e["kind"] for e in self.store.activity(self.pid, limit=50)]
        self.assertIn("programme_run", kinds)
        moves = [t["move"] for t in self.store.transitions(self.pid)]
        self.assertIn("run_programme", moves)
        states = graph.node_states(project)
        self.assertEqual((states["WF-07"], states["WF-09"]), ("done", "done"))
        self.assertIn("run_programme", [m["move"] for m in graph.describe(project)["moves"]])

    def test_the_record_keeps_hashes_of_what_it_read_and_no_row(self):
        project = self.store.get(self.pid)
        inputs = project["programme"]["inputs"]
        self.assertEqual(inputs["data_sha256"], project["data"]["sha256"])
        self.assertEqual(set(inputs["tables"]), {"accounts"})
        text = json.dumps(project["programme"])
        for value in list(self.replica.rides["order_id"].iloc[:30].astype(str)) + list(self.replica.accounts["phone"].iloc[:30]):
            self.assertNotIn(value, text)

    def test_a_second_read_of_the_holdout_is_refused_for_an_agent_and_needs_a_reason_from_a_person(self):
        with self.assertRaises(sp.ProgrammeError) as cm:
            sp.run(self.store, self.pid, actor="agent")
        self.assertIn("may not run the programme again", str(cm.exception))
        with self.assertRaises(sp.ProgrammeError) as cm:
            sp.run(self.store, self.pid, actor="human")
        self.assertIn("written reason", str(cm.exception))
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], 1, "a refused run reads nothing")

    def test_the_pipeline_covers_the_programme_and_replays_to_the_same_document(self):
        pipe = studio_pipeline.of(self.store, self.pid)
        ids = [s.id for s in pipe.steps]
        self.assertTrue({"programme:blk1", "programme:blk2", "programme:blk3", "programme:composition", "programme:holdout"} <= set(ids))
        self.assertNotIn("programme:blk4", ids, "the budget step is not a model")
        self.assertNotIn("stage:final", ids)
        holdout = pipe.step("programme:holdout")
        self.assertEqual(holdout.status, "done")
        self.assertTrue(any(n.startswith("programme.document.holdout") for n in holdout.numbers))
        before = self.store.get(self.pid)["holdout_uses"]
        out = studio_pipeline.rerun(self.store, self.pid, "programme:holdout")
        self.assertTrue(out["replayed"])
        self.assertTrue(out["same"], out)
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], before, "a replay in a scratch copy consumes nothing")

    def test_a_changed_table_makes_the_programme_stale(self):
        store = ProjectStore(Path(tempfile.mkdtemp()))
        pid, replica = make_project(store, seed=1)
        sp.run(store, pid)
        project = store.get(pid)
        project["tables"]["accounts"]["sha256"] = "0" * 64  # as if another file were attached under the same name
        store.save(project)
        self.assertEqual(sp.stale_reasons(store.get(pid)), ["tables changed since the programme ran"])
        pipe = studio_pipeline.of(store, pid)
        self.assertEqual(pipe.step("programme:holdout").status, "stale")

    def test_the_exported_code_reproduces_the_holdout_number(self):
        from dclab_rnd.studio import code

        out = Path(tempfile.mkdtemp()) / "code"
        made = code.export(self.store, self.pid, out)
        self.assertEqual(made["kind"], "programme")
        for name in ("programme.py", "problem_map.json", "solution_map.json", "programme.json", "expected.json", "requirements.txt", "README.md"):
            self.assertTrue((out / name).exists(), name)
        self.assertTrue((out / "data" / "rides.csv").exists())
        self.assertFalse((out / "vendor" / "dclab_rnd").exists(), "the programme needs nothing of the studio")
        done = subprocess.run([sys.executable, "-I", str(out / "programme.py")], cwd=out, capture_output=True, text=True, timeout=600)
        self.assertEqual(done.returncode, 0, done.stdout[-1500:] + done.stderr[-1500:])
        last = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertTrue(last["same"])
        self.assertEqual(last["chosen"], sp.stored(self.store.get(self.pid)).holdout.chosen)


@unittest.skipIf(MISSING, MISSING)
class BlockedTest(unittest.TestCase):
    def setUp(self):
        self.store = ProjectStore(Path(tempfile.mkdtemp()))
        self.pid, self.replica = make_project(self.store, tables=False)

    def test_a_leaf_that_cannot_run_keeps_the_holdout_closed_and_unused(self):
        view = sp.run(self.store, self.pid)
        project = self.store.get(self.pid)
        self.assertEqual(view["state"], "planned")
        self.assertEqual(view["holdout_reads"], 0)
        self.assertEqual(int(project.get("holdout_uses", 0) or 0), 0, "nothing read the holdout, so nothing was consumed")
        self.assertTrue(any(b.startswith("blk3") for b in view["blockers"]), view["blockers"])
        self.assertTrue(any("blocked" in i for i in view["issues"]))

    def test_only_a_person_drops_a_blocked_leaf_with_a_reason_and_the_rest_is_composed(self):
        sp.run(self.store, self.pid)
        with self.assertRaises(sp.ProgrammeError):
            sp.drop(self.store, self.pid, "blk3", "not needed", actor="agent")
        with self.assertRaises(sp.ProgrammeError):
            sp.drop(self.store, self.pid, "blk3", " ", actor="human")
        dropped = sp.drop(self.store, self.pid, "blk3", "this company has no table of accounts")
        self.assertTrue(any("dropped by person" in n for s in dropped["sections"] for n in s["lines"]) or "blk3" in json.dumps(dropped["document"]))
        view = sp.run(self.store, self.pid)
        self.assertEqual(view["state"], "composed")
        self.assertEqual(self.store.get(self.pid)["holdout_uses"], 1)
        p = sp.stored(self.store.get(self.pid))
        self.assertEqual(next(l for l in p.leaves if l.block == "blk3").status, "dropped")
        self.assertNotIn("blk3", p.composition.candidates)
        pipe = studio_pipeline.of(self.store, self.pid)
        self.assertEqual(pipe.step("programme:blk3").status, "not_recorded", "the dropped leaf is listed, with the reason")
        self.assertNotIn("programme:blk3", pipe.step("programme:composition").depends_on, "and it is no input of the composition")
        self.assertEqual((pipe.step("programme:composition").status, pipe.step("programme:holdout").status), ("done", "done"), "so it makes nothing stale")

    def test_a_dropped_leaf_cannot_be_dropped_after_the_holdout_was_read(self):
        sp.run(self.store, self.pid)
        sp.drop(self.store, self.pid, "blk3", "no accounts table")
        sp.run(self.store, self.pid)
        with self.assertRaises(sp.ProgrammeError) as cm:
            sp.drop(self.store, self.pid, "blk1", "changed my mind")
        self.assertIn("already read", str(cm.exception))

    def test_a_second_read_by_a_person_with_a_reason_keeps_the_first_in_the_record(self):
        sp.run(self.store, self.pid)
        sp.drop(self.store, self.pid, "blk3", "no accounts table")
        sp.run(self.store, self.pid)
        first = sp.stored(self.store.get(self.pid)).holdout.composed.value
        view = sp.run(self.store, self.pid, actor="human", reuse_reason="the labels were corrected")
        project = self.store.get(self.pid)
        self.assertEqual(project["holdout_uses"], 2)
        self.assertEqual(view["holdout_reads"], 2)
        self.assertEqual(project["programme"]["earlier"][0]["composed"], first)
        self.assertEqual(project["programme"]["earlier"][0]["reason"], "the labels were corrected")


@unittest.skipIf(MISSING, MISSING)
class GraphTest(unittest.TestCase):
    def test_a_project_without_a_plan_has_no_programme_to_run(self):
        store = ProjectStore(Path(tempfile.mkdtemp()))
        project = store.create("Plain", "general", "")
        verdict = graph.check(store.get(project["id"]), "run_programme", "human")
        self.assertEqual(verdict.status, "blocked")
        self.assertIn("several parts", verdict.message)
        self.assertNotIn("run_programme", [m["move"] for m in graph.describe(store.get(project["id"]))["moves"]])

    def test_the_owners_approval_is_needed_for_an_agent_when_the_policy_asks(self):
        store = ProjectStore(Path(tempfile.mkdtemp()))
        pid, _ = make_project(store)
        project = store.get(pid)
        project["policy"] = {"require_holdout_approval": True}
        store.save(project)
        v = graph.check(store.get(pid), "run_programme", "agent")
        self.assertEqual(v.status, "needs_approval")
        self.assertIn("Owner approval", [c["name"] for c in v.checks if not c["ok"]])


@unittest.skipIf(MISSING, MISSING)
class ReviewTest(unittest.TestCase):
    """What the independent review of 24.5 found, each as the sequence that showed it."""

    def fresh(self, **kwargs):
        store = ProjectStore(Path(tempfile.mkdtemp()))
        pid, replica = make_project(store, **kwargs)
        return store, pid, replica

    def break_accounts(self, store, pid, replica):
        """Replace the table of accounts by one with no contact column, as a person correcting a file might."""
        d = store.data_dir(pid)
        replica.accounts[["account_id", "city"]].to_csv(d / "accounts.csv", index=False)
        sd.attach_table(store, pid, "accounts.csv", "accounts")

    def test_a_second_read_cannot_be_reached_by_dropping_a_leaf_after_seeing_the_first(self):
        store, pid, replica = self.fresh()
        sp.run(store, pid)
        self.assertEqual(sp.reads_of(store.get(pid)), 1)
        self.break_accounts(store, pid, replica)
        view = sp.run(store, pid, actor="human", reuse_reason="the accounts file was corrected")
        self.assertEqual(view["state"], "planned", "the rings cannot run on the new file")
        self.assertEqual(store.get(pid)["holdout_uses"], 1, "a run that reads nothing consumes nothing")
        with self.assertRaises(sp.ProgrammeError) as cm:
            sp.drop(store, pid, "blk3", "no accounts contacts any more")
        self.assertIn("already read", str(cm.exception))
        with self.assertRaises(sp.ProgrammeError):
            sp.run(store, pid, actor="agent")
        with self.assertRaises(sp.ProgrammeError) as cm:
            sp.run(store, pid, actor="human")
        self.assertIn("written reason", str(cm.exception))
        self.assertEqual(sp.stored(store.get(pid)).reads_before, 1)

    def test_a_replay_is_refused_when_what_the_run_read_has_changed(self):
        store, pid, replica = self.fresh()
        sp.run(store, pid)
        project = store.get(pid)
        project["suggestion"]["alert_budget"] = 0.05                                  # another budget is another question put to the same holdout
        store.save(project)
        self.assertEqual(sp.stale_reasons(store.get(pid)), ["budget changed since the programme ran"])
        with self.assertRaises(sp.ProgrammeError) as cm:
            sp.replay(store, pid)
        self.assertIn("same inputs", str(cm.exception))
        out = studio_pipeline.rerun(store, pid, "programme:holdout")
        self.assertFalse(out["replayed"])
        self.assertEqual(store.get(pid)["holdout_uses"], 1)

    def test_a_project_solved_by_the_stages_and_never_run_as_a_programme_keeps_its_stage_pipeline(self):
        store, pid, replica = self.fresh(tables=False)
        project = store.get(pid)
        project["stages"] = {"data": {"status": "completed"}}
        store.save(project)
        ids = [s.id for s in studio_pipeline.of(store, pid).steps]
        self.assertNotIn("programme:holdout", ids, "the programme never ran: its steps are not missing steps of the stages' pipeline")
        self.assertIn("stage:data", ids)

    def test_a_replay_of_a_programme_that_never_read_the_holdout_is_refused(self):
        store, pid, replica = self.fresh(tables=False, plan_with_accounts=False)
        sp.run(store, pid)
        self.assertIsNone(sp.stored(store.get(pid)).holdout)
        with self.assertRaises(sp.ProgrammeError) as cm:
            sp.replay(store, pid)
        self.assertIn("never read the holdout", str(cm.exception))

    def test_every_input_of_the_run_makes_it_stale_when_it_changes(self):
        store, pid, replica = self.fresh()
        sp.run(store, pid)
        self.assertEqual(sp.stale_reasons(store.get(pid)), [])
        for edit, expect in [(lambda p: p["suggestion"].update(identifiers=[*p["suggestion"].get("identifiers", []), "fare"]), "excluded"),
                             (lambda p: p["labels"].update(sha256="0" * 64), "labels_sha256"),
                             (lambda p: p["data"].update(sha256="0" * 64), "data_sha256")]:
            project = store.get(pid)
            edit(project)
            self.assertIn(f"{expect} changed since the programme ran", sp.stale_reasons(project), expect)

    def test_attaching_the_labels_again_does_not_forget_the_review_budget(self):
        store, pid, replica = self.fresh(labels=False)
        project = store.get(pid)
        project["suggestion"] = {**(project.get("suggestion") or {}), "alert_budget": 0.02}
        store.save(project)
        replica.cases.to_csv(store.data_dir(pid) / "cases.csv", index=False)
        sd.attach_labels(store, pid, "cases.csv", target="is_case", key=["order_id"])
        self.assertEqual(sp.budget_of(store.get(pid)), 0.02, "the budget a person said is not the label file's to forget")

    def test_no_table_can_be_attached_while_the_programme_runs_and_the_flag_is_released_after_a_failure(self):
        store, pid, replica = self.fresh()
        seen = {}
        real = sp.execute

        def during(store_, project, **kwargs):
            seen["running"] = store_.get(pid).get("running")
            verdict = graph.check(store_.get(pid), "attach_table", "human", filename="x.csv", name="x")
            seen["verdict"] = (verdict.status, verdict.message)
            raise RuntimeError("the run failed")

        with mock.patch.object(sp, "execute", during):
            with self.assertRaises(RuntimeError):
                sp.run(store, pid)
        self.assertEqual(seen["running"], "programme")
        self.assertEqual(seen["verdict"][0], "blocked")
        self.assertIn("still running", seen["verdict"][1])
        self.assertIsNone(store.get(pid).get("running"), "a failed run frees the project")
        sp.execute = real

    def test_a_column_the_project_forbids_never_reaches_a_leaf(self):
        extra = {"basket_total": lambda rides: rides["fare"] * 1.1 + rides["distance_m"] / 1000}
        store, pid, replica = self.fresh(rides_extra=extra)
        project = store.get(pid)
        project["suggestion"]["forbidden"] = ["basket_total"]
        store.save(project)
        sp.run(store, pid)
        for leaf in sp.stored(store.get(pid)).leaves:
            self.assertNotIn("basket_total", leaf.columns)
        self.assertIn("basket_total", sp.excluded(store.get(pid)))
        store2, pid2, _ = self.fresh(rides_extra=extra)
        sp.run(store2, pid2)
        self.assertIn("basket_total", next(l for l in sp.stored(store2.get(pid2)).leaves if l.block == "blk1").columns, "the column is a measure unless the project forbids it")

    def test_a_column_the_data_card_marks_as_a_leak_suspect_is_kept_out_without_anyone_saying_so(self):
        extra = {"final_fare": lambda rides: rides["fare"] * 1.1 + rides["distance_m"] / 1000}
        store, pid, replica = self.fresh(rides_extra=extra)
        project = store.get(pid)
        self.assertNotIn("final_fare", (project["suggestion"] or {}).get("forbidden", []))
        sp.run(store, pid)
        self.assertNotIn("final_fare", next(l for l in sp.stored(store.get(pid)).leaves if l.block == "blk1").columns)
        self.assertIn("final_fare", sp.excluded(store.get(pid)))

    def test_the_owners_signature_on_the_plan_is_needed_when_the_policy_asks(self):
        store, pid, replica = self.fresh()
        project = store.get(pid)
        project["policy"] = {"require_solution_signoff": True}
        store.save(project)
        with self.assertRaises(sp.ProgrammeError) as cm:
            sp.run(store, pid)
        self.assertIn("sign the plan", str(cm.exception))
        graph.approve_gate(store, pid, "solution", by="owner", reason="the plan is what we asked for")
        self.assertTrue(graph.plan_signed(store.get(pid)))
        self.assertEqual(sp.run(store, pid)["state"], "composed")

    def test_an_agents_approval_is_spent_by_a_read_and_not_by_a_run_that_reads_nothing(self):
        store, pid, replica = self.fresh(tables=False, plan_with_accounts=False)
        project = store.get(pid)
        project["policy"] = {"require_holdout_approval": True}
        project["approvals"] = [{"gate": "holdout", "by": "owner", "at": "2026-10-16T00:00:00Z"}]
        store.save(project)
        sp.run(store, pid, actor="agent")                                              # a leaf is blocked: the holdout is not read
        self.assertFalse(store.get(pid)["approvals"][0].get("used"), "the approval waits for the read")
        store.save({**store.get(pid), "approvals": [{"gate": "holdout", "by": "owner", "at": "2026-10-16T00:00:00Z"}]})
        store2, pid2, _ = self.fresh()
        p2 = store2.get(pid2)
        p2["policy"] = {"require_holdout_approval": True}
        p2["approvals"] = [{"gate": "holdout", "by": "owner", "at": "2026-10-16T00:00:00Z"}]
        store2.save(p2)
        sp.run(store2, pid2, actor="agent")
        self.assertTrue(store2.get(pid2)["approvals"][0].get("used"), "the read spent it")

    def test_a_stale_programme_is_not_exported_as_code(self):
        from dclab_rnd.studio import code

        store, pid, replica = self.fresh()
        sp.run(store, pid)
        project = store.get(pid)
        project["suggestion"]["alert_budget"] = 0.05
        store.save(project)
        with self.assertRaises(code.CodeRefused) as cm:
            code.export(store, pid, Path(tempfile.mkdtemp()) / "code")
        self.assertIn("stale", str(cm.exception))

    def test_a_parquet_table_asks_for_the_reader_that_opens_it_in_the_exported_requirements(self):
        from dclab_rnd.studio import code

        store = ProjectStore(Path(tempfile.mkdtemp()))
        r = fp.generate(seed=0, **SMALL)
        project = store.create("Fraud programme", "general", WORDS)
        pid = project["id"]
        d = store.data_dir(pid)
        r.rides.to_parquet(d / "rides.parquet", index=False)
        sd.attach_data(store, pid, "rides.parquet")
        r.cases.to_parquet(d / "cases.parquet", index=False)
        sd.attach_labels(store, pid, "cases.parquet", target="is_case", key=["order_id"])
        r.accounts.to_parquet(d / "accounts.parquet", index=False)
        sd.attach_table(store, pid, "accounts.parquet", "accounts")
        p = store.get(pid)
        p["programme_plan"] = plan_for(r.rides, r.accounts)
        p["suggestion"] = {**(p.get("suggestion") or {}), "alert_budget": 0.02}
        store.save(p)
        sp.run(store, pid)
        out = Path(tempfile.mkdtemp()) / "code"
        code.export(store, pid, out)
        self.assertIn("pyarrow", (out / "requirements.txt").read_text())

    def test_attaching_the_table_of_accounts_lets_the_rings_be_routed_and_makes_an_earlier_programme_stale(self):
        store, pid, replica = self.fresh(tables=False, plan_with_accounts=False)
        before = sp.plan_of(store.get(pid))["solution_map"]
        self.assertTrue(before["blockers"], "the router had no table for the rings when the draft was confirmed")
        sp.run(store, pid)
        replica.accounts.to_csv(store.data_dir(pid) / "accounts.csv", index=False)
        sd.attach_table(store, pid, "accounts.csv", "accounts")
        self.assertIn("tables changed since the programme ran", sp.stale_reasons(store.get(pid)))
        self.assertTrue(sp.refresh_plan(store, pid))
        after = sp.plan_of(store.get(pid))["solution_map"]
        self.assertEqual(after["blockers"], [])
        self.assertNotEqual(after["fingerprint"], before["fingerprint"])
        self.assertIn("programme_plan_refreshed", [e["kind"] for e in store.activity(pid, limit=50)])
        self.assertFalse(sp.refresh_plan(store, pid), "nothing changed the second time")
        self.assertEqual(sp.run(store, pid)["state"], "composed")


@unittest.skipIf(MISSING, MISSING)
class RouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_rnd.agentic.server import create_app

        cls.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        cls.client.__enter__()
        cls.h = {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}
        cls.s = cls.client.app.state.services

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_the_programme_is_shown_run_and_a_blocked_leaf_dropped_through_the_routes(self):
        pid, _ = make_project(self.s.projects, tables=False)
        shown = self.client.get(f"/api/projects/{pid}/programme").json()
        self.assertEqual((shown["available"], shown["state"]), (True, "not_run"))
        project = self.client.get(f"/api/projects/{pid}").json()
        self.assertEqual(set(project["programme_plan"]), {"map", "solution_map"}, "the page gets the plan's fingerprints, not the whole plan")
        ran = self.client.post(f"/api/projects/{pid}/programme/run", json={}, headers=self.h).json()
        self.assertEqual((ran["state"], ran["holdout_reads"]), ("planned", 0))
        self.assertEqual(ran["job"]["status"], "done")
        refused = self.client.post(f"/api/projects/{pid}/programme/drop", json={"block": "blk3", "reason": ""}, headers=self.h)
        self.assertEqual(refused.status_code, 409)
        dropped = self.client.post(f"/api/projects/{pid}/programme/drop", json={"block": "blk3", "reason": "no accounts table here"}, headers=self.h)
        self.assertEqual(dropped.status_code, 200)
        done = self.client.post(f"/api/projects/{pid}/programme/run", json={}, headers=self.h).json()
        self.assertEqual((done["state"], done["holdout_reads"]), ("composed", 1))
        again = self.client.post(f"/api/projects/{pid}/programme/run", json={}, headers=self.h)
        self.assertEqual(again.status_code, 409)
        self.assertIn("written reason", again.json()["detail"])

    def test_every_payload_that_carries_a_project_carries_the_plans_fingerprints_not_the_plan(self):
        pid, replica = make_project(self.s.projects, tables=False)
        body = replica.accounts.to_csv(index=False).encode()
        put = self.client.put(f"/api/projects/{pid}/tables/accounts?filename=accounts.csv", content=body, headers=self.h)
        self.assertEqual(put.status_code, 200, put.text)
        self.assertEqual(set(put.json()["programme_plan"]), {"map", "solution_map"})
        listed = next(p for p in self.client.get("/api/projects").json() if p["id"] == pid)
        self.assertEqual(set(listed["programme_plan"]), {"map", "solution_map"})
        shown = self.client.get(f"/api/projects/{pid}/programme").json()
        self.assertEqual(shown["solution_map"], put.json()["programme_plan"]["solution_map"])
        self.assertTrue(shown["blocks"] and shown["split"]["strategy"] == "time_group", "the page gets the plan's blocks and its split from /programme")

    def test_the_programme_has_a_report_with_a_section_per_leaf_and_every_number_traced(self):
        pid, _ = make_project(self.s.projects)
        refused = self.client.get(f"/api/projects/{pid}/export/solution-report")
        self.assertEqual(refused.status_code, 409, "nothing to report before the programme ran")
        self.client.post(f"/api/projects/{pid}/programme/run", json={}, headers=self.h)
        got = self.client.get(f"/api/projects/{pid}/export/solution-report")
        self.assertEqual(got.status_code, 200, got.text[:300])
        text = got.text
        for heading in ("## blk1:", "## blk2:", "## blk3:", "## The composition, chosen on validation", "## The holdout, read once"):
            self.assertIn(heading, text)
        self.assertIn("Composed minus best leaf", text)
        project = self.s.projects.get(pid)
        project["suggestion"]["alert_budget"] = 0.05
        self.s.projects.save(project)
        stale = self.client.get(f"/api/projects/{pid}/export/solution-report")
        self.assertEqual(stale.status_code, 409)
        self.assertIn("stale", stale.json()["detail"])

    def test_a_single_problem_has_no_programme(self):
        pid = self.s.projects.create("Plain", "general", "")["id"]
        self.assertEqual(self.client.get(f"/api/projects/{pid}/programme").json()["available"], False)
        self.assertEqual(self.client.post(f"/api/projects/{pid}/programme/run", json={}, headers=self.h).status_code, 409)


if __name__ == "__main__":
    unittest.main()
