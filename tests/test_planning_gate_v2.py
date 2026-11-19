"""hp_gate_v2 (2026-11-19): the planning exam's six checks have judging code, pinned before the run; the held-out tables
are built on the product's own unattended path. These tests use invented rows and an invented case, never a held-out one."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval.planning_gate_v2 import gate  # noqa: E402


def row(case, arm, read=2, enforced=2, planted=3, planted_enforced=2, before=("a",), after=("a",), seconds=1.0, eur=0.0, **events):
    return {"case": case, "arm": arm, "obeyed_read": read, "obeyed_done": enforced, "stopped_with_question": False, "routed_to": arm,
            "plan_status": {"status": "planned"}, "rules_catches": ["a"], "plan_catches": ["a"],
            "planted": planted, "planted_enforced": planted_enforced, "forbidden_before": list(before), "forbidden_after": list(after),
            "plan_seconds": seconds, "plan_eur": eur, **{name: events.get(name, 0) for name in gate.ZERO}}


def projects(model_planted=2, rules_planted=2, seconds=100.0, eur=0.01, **kw):
    out = []
    for i in range(8):
        out.append(row(f"c{i}", "rules", planted_enforced=rules_planted, **kw))
        out.append(row(f"c{i}", "model", planted_enforced=model_planted, seconds=seconds, eur=eur, **kw))
    return out


PEOPLE_OK = {"met": True}


class GateTests(unittest.TestCase):
    def test_the_terms_are_the_owners_choices(self):
        self.assertEqual((gate.MAX_SECONDS, gate.MAX_EUR, gate.TERMS["people"], gate.MIN_PROJECTS), (300.0, 0.03, "plan_held", 8))

    def test_a_level_model_gives_way_to_the_rules_and_the_gate_reads_the_rules_cost(self):
        v = gate.judge(projects(seconds=999.0), PEOPLE_OK)
        self.assertEqual(v["checks"]["5_model_step"]["serves"], "rules")
        self.assertTrue(v["checks"]["6_cost"]["met"])  # the rules' plan costs nothing: the slow model is off
        self.assertEqual(v["not_judged"], [])

    def test_a_model_that_catches_more_serves_and_must_fit_the_time(self):
        better = gate.judge(projects(model_planted=3, rules_planted=1, seconds=280.0), PEOPLE_OK)
        self.assertEqual(better["checks"]["5_model_step"]["serves"], "model")
        self.assertTrue(better["checks"]["6_cost"]["met"])
        slow = gate.judge(projects(model_planted=3, rules_planted=1, seconds=320.0), PEOPLE_OK)
        self.assertFalse(slow["checks"]["6_cost"]["met"])
        self.assertFalse(slow["passed"])

    def test_a_catch_not_enforced_fails_obeyed(self):
        rows = projects()
        rows[3]["obeyed_done"] = 1
        self.assertFalse(gate.judge(rows, PEOPLE_OK)["checks"]["1_obeyed"]["met"])

    def test_obeyed_is_not_read_when_no_catch_was_read_and_a_stop_excuses_only_itself(self):
        rows = projects(read=0, enforced=0)
        rows[0]["stopped_with_question"] = True
        self.assertIsNone(gate.judge(rows, PEOPLE_OK)["checks"]["1_obeyed"]["met"])  # nothing tested is never a pass

    def test_a_column_put_back_or_lost_by_the_model_fails_nothing_lost(self):
        rows = projects()
        rows[0]["forbidden_after"] = []  # the plan put back a column the solution had kept out
        self.assertFalse(gate.judge(rows, PEOPLE_OK)["checks"]["2_nothing_lost"]["met"])
        rows = projects()
        rows[1]["rules_catches"] = ["a", "b"]  # the rules' plan caught b on this project; the model's plan, served, did not
        check = gate.judge(rows, PEOPLE_OK)["checks"]["2_nothing_lost"]
        self.assertEqual((check["model_plans_dropping_a_rules_catch"], check["met"]), (1, False))

    def test_any_event_fails_and_a_missing_people_run_is_not_judged(self):
        self.assertFalse(gate.judge(projects(holdout_read_twice=1), PEOPLE_OK)["passed"])
        v = gate.judge(projects(), None)
        self.assertIn("3_people", v["not_judged"])
        self.assertFalse(v["passed"])

    def test_the_cost_is_read_only_where_a_plan_was_made(self):
        rows = projects(model_planted=3, rules_planted=1, seconds=280.0)
        for r in rows[1::2][:4]:
            r["plan_seconds"] = None  # stopped: no plan made, never a zero that lowers the mean
        self.assertEqual(gate.judge(rows, PEOPLE_OK)["checks"]["6_cost"]["projects"], 4)

    def test_the_committed_result_holds_no_case_or_column(self):
        rows = projects()
        rows[0]["forbidden_after"] = []
        text = json.dumps(gate.public(gate.judge(rows, PEOPLE_OK)))
        self.assertNotIn("c0", text)
        self.assertNotIn('"a"', text)

    def test_the_terms_pin_the_held_out_material(self):
        self.assertTrue(gate.TERMS["held_tables_pin"] and gate.TERMS["held_people_pin"])

    def test_a_pair_is_one_case_on_one_seed(self):
        """The second review: rows were keyed by case alone, so the second seed overwrote the first."""
        rows = []
        for case in ("c0", "c1", "c2", "c3"):
            for seed in (1, 2):
                rows += [{**row(case, "rules", planted_enforced=1), "seed": seed}, {**row(case, "model", planted_enforced=3), "seed": seed}]
        self.assertEqual(gate.model_step(rows)["pairs"], 8)

    def test_too_few_projects_decide_nothing(self):
        v = gate.judge(projects()[:6], PEOPLE_OK)
        self.assertIn("5_model_step", v["not_judged"])
        self.assertFalse(v["passed"])

    def test_people_are_read_as_hab_gate_v2_reads_them(self):
        def rows(planning, understanding):
            return [{"case": f"p{i}", "judged": True, "planning": planning, "understanding": understanding,
                     **{z: 0 for z in ("numbers_nobody_gave", "built_without_yes", "blocked_runs", "fallback_lines", "empty_answers",
                                       "cut_replies", "forbidden_done")}} for i in range(8)]

        judge = "nvidia/nemotron-3.5-super-vl:free"
        self.assertTrue(gate.people("qwen/qwen3.8-flash", judge, rows(3, 3), rows(3, 3))["met"])
        self.assertFalse(gate.people("qwen/qwen3.8-flash", judge, rows(2, 3), rows(3, 3))["met"])  # a whole point behind on the plan

    def test_the_gate_is_pinned_and_its_terms_unchanged(self):
        self.assertIsNone(gate.pinned(), "hp_gate_v2's terms changed after the pin: a changed gate is a new gate")


def invented_case():
    """One small invented case, never a held-out one: a late column, an entity, a time."""
    rng = np.random.default_rng(7)
    n = 400
    at = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 200, n)), unit="D")
    shop = rng.integers(0, 40, n)
    spend = rng.uniform(5, 100, n)
    churned = (rng.random(n) < 1 / (1 + np.exp(-(spend - 50) / 15))).astype(int)
    frame = pd.DataFrame({"order_id": np.arange(n), "created_at": at, "shop_id": shop, "spend": spend,
                          "closed_at": at + pd.to_timedelta(rng.uniform(1, 30, n), unit="D"), "churned": churned})
    case = {"id": "inv1", "brief": "Which orders lead to churn?", "known_facts": {"target": "churned"},
            "enforceable_traps": [{"kind": "late_column", "column": "closed_at"}, {"kind": "time_split", "column": "created_at"}]}
    return case, {"orders.csv": frame}


class RunTests(unittest.TestCase):
    def test_a_case_is_built_on_the_unattended_path_with_the_plan_before_the_stages(self):
        from dclab_rnd.agent_eval.planning_gate_v2 import run
        from dclab_rnd.expansion.runner import fast_profile

        case, frames = invented_case()
        with fast_profile():
            out = run.build(case, frames, "rules", Path(tempfile.mkdtemp()))
        self.assertEqual(out["plan_status"]["status"], "planned")  # the plan was made before run_all
        self.assertEqual(out["routed_to"], "rules")
        self.assertEqual(out["planted"], 2)
        self.assertIn("closed_at", out["forbidden_after"])
        self.assertEqual(out["holdout_read_twice"], 0)
        self.assertEqual(out["numbers_nobody_gave"], 0)
        self.assertEqual(out["plan_eur"], 0.0)

    def test_the_project_id_in_the_report_is_no_number_anyone_gave(self):
        """The second review: a hex id starting with digits was counted as a made-up number on about one project in five."""
        from types import SimpleNamespace

        from dclab_rnd.agent_eval.planning_gate_v2 import run
        from dclab_rnd.expansion.runner import fast_profile

        case, frames = invented_case()
        with fast_profile(), mock.patch("dclab_rnd.studio.store.uuid.uuid4", return_value=SimpleNamespace(hex="7070912960b0" + "0" * 20)):
            out = run.build(case, frames, "rules", Path(tempfile.mkdtemp()))
        self.assertEqual((out["project_id"], out["numbers_nobody_gave"]), ("7070912960b0", 0))

    def test_a_case_with_no_outcome_stops_with_a_question(self):
        from dclab_rnd.agent_eval.planning_gate_v2 import run

        case, frames = invented_case()
        case["known_facts"] = {"target": None}
        case["brief"] = "Help us with our orders."
        out = run.build(case, frames, "rules", Path(tempfile.mkdtemp()))
        self.assertTrue(out["stopped_with_question"])
        self.assertIsNone(out["plan_status"])

    def test_the_held_cases_are_refused_when_they_changed_after_their_pin(self):
        from dclab_rnd.agent_eval.planning_gate_v2 import run

        folder = Path(tempfile.mkdtemp())
        (folder / "cases.json").write_text(json.dumps({"cases": [], "seeds": [1]}), encoding="utf-8")
        (folder / "tables.py").write_text("", encoding="utf-8")
        (folder / "cases.sha256").write_text("0" * 64 + "  cases.json\n" + "0" * 64 + "  tables.py\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            run.load_held(folder)


class WatchTests(unittest.TestCase):
    def test_a_purpose_no_model_serves_is_written_down(self):
        from dclab_rnd.agent_eval.planning_gate_v2.run import Watched

        class View:
            def client(self, *a, **k):
                return None

        watched = Watched(View())
        self.assertIsNone(watched.client("task_decomposer"))  # the planner sees no model, as it would without the wrapper
        self.assertEqual(len(watched.errors), 1)

    def test_a_failed_model_request_is_written_down_and_still_raised(self):
        from dclab_rnd.agent_eval.planning_gate_v2.run import Watched

        class BudgetExceeded(RuntimeError):  # the cap's error is matched by its name (conversation_exam)
            pass

        class Client:
            def complete(self, *a, **k):
                raise BudgetExceeded("cap")

        class View:
            def client(self, *a, **k):
                return Client()

        watched = Watched(View())
        with self.assertRaises(BudgetExceeded):
            watched.client("task_decomposer").complete([])
        self.assertEqual([type(e).__name__ for e in watched.errors], ["BudgetExceeded"])


class CommandTests(unittest.TestCase):
    def test_preflight_names_what_would_waste_the_run_by_counts(self):
        from types import SimpleNamespace

        from dclab_rnd.agent_eval import planning_gate_v2 as cli

        cases = [{"id": f"k{i}", "enforceable_traps": [{"kind": "late_column", "column": "x"}]} for i in range(4)]
        tables = SimpleNamespace(**{f"make_k{i}": (lambda seed: {}) for i in range(4)})
        self.assertIsNone(cli.preflight(cases, [1, 2], tables))
        self.assertIn("1 cases without", cli.preflight(cases + [{"id": "k9", "enforceable_traps": []}], [1, 2], tables))
        cases[0]["enforceable_traps"] = [{"kind": "mystery", "column": "x"}]
        self.assertIn("1 unknown trap kinds", cli.preflight(cases, [1, 2], tables))

    def test_judge_is_refused_a_second_time_without_people_on_another_model_or_on_a_cut_run(self):
        from dclab_rnd.agent_eval import planning_gate_v2 as cli

        root = Path(tempfile.mkdtemp())
        run = root / "2026-11-19-planning-gate-v2-tables"
        run.mkdir()
        (run / "rows.json").write_text(json.dumps({"rows": projects(), "model": "qwen/qwen3.8-flash", "stopped": None, "of": 16}), encoding="utf-8")
        results = root / "results"
        people = ("qwen/qwen3.8-flash", "nvidia/nemotron-3.5-super-vl:free", [], [])
        with mock.patch.object(cli, "RUN_ROOT", root), mock.patch.object(cli, "RESULTS", results):
            self.assertEqual(cli.main(["judge"]), 2)  # no people run named: a reading without check 3 would lock the gate
            with mock.patch.object(cli, "_people_rows", return_value=("other/model", None, [], [])):
                self.assertEqual(cli.main(["judge", "--people", "HAB-999"]), 2)  # the people run used another model
            with mock.patch.object(cli, "_people_rows", return_value=people):
                self.assertEqual(cli.main(["judge", "--people", "HAB-999"]), 0)
                self.assertEqual(cli.main(["judge", "--people", "HAB-999"]), 2)  # judged once
        self.assertTrue((run / "verdict.json").exists())
        root2 = Path(tempfile.mkdtemp())
        run2 = root2 / "2026-11-19-planning-gate-v2-tables"
        run2.mkdir()
        (run2 / "rows.json").write_text(json.dumps({"rows": projects()[:10], "model": "qwen/qwen3.8-flash", "stopped": None, "of": 16}), encoding="utf-8")
        with mock.patch.object(cli, "RUN_ROOT", root2), mock.patch.object(cli, "RESULTS", root2 / "results"), \
                mock.patch.object(cli, "_people_rows", return_value=people):
            self.assertEqual(cli.main(["judge", "--people", "HAB-999"]), 0)
        self.assertFalse(json.loads((root2 / "results" / "HPG-001_gate.json").read_text())["passed"])  # a cut run never passes

    def test_an_incomplete_people_run_is_refused(self):
        from dclab_rnd.agent_eval import planning_gate_v2 as cli

        root = Path(tempfile.mkdtemp())
        folder = root / "evidence" / "campaigns" / "harness_ablation_v1" / "results"
        folder.mkdir(parents=True)
        (folder / "HAB-998_plan_held.json").write_text(json.dumps({"set": "plan_held", "incomplete": True, "models": {}, "scores": {}}), encoding="utf-8")
        with mock.patch.object(cli, "ROOT", root):
            with self.assertRaises(ValueError):
                cli._people_rows("HAB-998")

    def test_the_tables_run_is_refused_twice_unpinned_with_claude_or_without_yes(self):
        from dclab_rnd.agent_eval import planning_gate_v2 as cli
        from dclab_rnd.agent_eval.planning_gate_v2 import run

        empty = Path(tempfile.mkdtemp())
        self.assertEqual(cli.main(["tables", "--model", "anthropic/claude-x", "--yes"]), 2)
        with mock.patch.object(cli.gate, "pinned", return_value="the gate is not pinned"):
            self.assertEqual(cli.main(["tables", "--yes"]), 2)
        with mock.patch.object(cli, "_runs", return_value=[Path("2026-11-19-planning-gate-v2-tables")]):
            self.assertEqual(cli.main(["tables", "--yes"]), 2)
        # never the real held-out folder here: the loader and the preflight are stand-ins
        stand_ins = (mock.patch.object(cli, "_runs", return_value=[]), mock.patch.object(cli.gate, "pinned", return_value=None),
                     mock.patch.object(cli, "RESULTS", empty), mock.patch.object(run, "load_held", return_value=([], [1], None)))
        for patch in stand_ins:
            patch.start()
        try:
            with mock.patch.object(cli, "preflight", return_value=None):
                self.assertEqual(cli.main(["tables"]), 0)  # without --yes it only says what it would do
            with mock.patch.object(cli, "preflight", return_value="0 pairs"):
                self.assertEqual(cli.main(["tables", "--yes"]), 2)  # a preflight problem never starts the run
        finally:
            for patch in stand_ins:
                patch.stop()


if __name__ == "__main__":
    unittest.main()
