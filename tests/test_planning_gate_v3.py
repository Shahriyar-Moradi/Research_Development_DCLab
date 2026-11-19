"""hp_gate_v3 (2026-11-19): v2's six checks by v2's own code, plus how many planted traps the serving plan enforces and
no needless entity split. Nothing here reads the held-out material."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def row(case, arm, planted=4, enforced=2, kinds=None, needless=0, **kw):
    base = {"case": case, "seed": 1, "arm": arm, "obeyed_read": 1, "obeyed_done": 1, "asked_not_applied": 0, "stopped_with_question": False,
            "forbidden_before": [], "forbidden_after": [], "planted": planted, "planted_enforced": enforced,
            "planted_by_kind": kinds or {"late_column": [2, 1], "time_split": [2, 1]}, "needless_group": needless,
            "plan_seconds": 0.02, "plan_eur": 0.0, "holdout_read_twice": 0, "stages_past_a_needed_yes": 0,
            "failed_with_nothing_built": 0, "numbers_nobody_gave": 0, "routed_to": arm, "plan_status": {"status": "planned"}}
    return {**base, **kw}


class GateTests(unittest.TestCase):
    def test_check_7_reads_the_serving_arms_share_of_planted_traps(self):
        from dclab_rnd.agent_eval.planning_gate_v3 import gate

        rows = [row(f"c{i}", a, enforced=2 if a == "rules" else 3) for i in range(8) for a in ("rules", "model")]
        c = gate.caught(rows, "rules")
        self.assertEqual((c["rules"], c["model"], c["met"]), (0.5, 0.75, True))
        self.assertEqual(c["by_kind"]["late_column"]["rules"], 0.5)
        self.assertFalse(gate.caught([row(f"c{i}", "rules", enforced=1) for i in range(8)], "rules")["met"])
        self.assertIsNone(gate.caught(rows, None)["met"])

    def test_check_8_fails_on_one_needless_entity_split_and_is_not_read_with_nothing_to_test(self):
        from dclab_rnd.agent_eval.planning_gate_v3 import gate

        self.assertTrue(gate.needless_group([row("a", "rules", no_group_case=True)])["met"])
        self.assertFalse(gate.needless_group([row("a", "rules", no_group_case=True), row("b", "model", needless=1, no_group_case=True)])["met"])
        self.assertIsNone(gate.needless_group([row("a", "rules")])["met"])  # no case names a column: nothing was tested

    def test_check_7_leaves_out_projects_that_stopped_for_their_outcome(self):
        from dclab_rnd.agent_eval.planning_gate_v3 import gate

        rows = [row("a", "rules", planted=2, enforced=2), row("b", "rules", planted=2, enforced=0, stopped_with_question=True)]
        self.assertEqual(gate.caught(rows, "rules")["rules"], 1.0)

    def test_a_stale_v2_pin_problem_never_survives_a_pinned_v3(self):
        from dclab_rnd.agent_eval.planning_gate_v3 import gate

        rows = [row(f"c{i}", a, no_group_case=True) for i in range(8) for a in ("rules", "model")]
        with mock.patch.object(gate.v2, "pinned", return_value="v2 changed"), mock.patch.object(gate, "pinned", return_value=None):
            v = gate.judge(rows, {"met": True})
        self.assertNotIn("pin_problem", v)
        self.assertTrue(v["pinned"])

    def test_the_verdict_holds_v2s_six_checks_and_the_two_new_ones_and_needs_the_pin(self):
        from dclab_rnd.agent_eval.planning_gate_v3 import gate

        rows = [row(f"c{i}", a) for i in range(8) for a in ("rules", "model")]
        v = gate.judge(rows, {"met": True})
        self.assertEqual(sorted(v["checks"]), ["1_obeyed", "2_nothing_lost", "3_people", "4_zero_events", "5_model_step", "6_cost",
                                               "7_caught", "8_no_needless_group"])
        self.assertEqual(v["gate"], "hp_gate_v3")
        with mock.patch.object(gate, "PIN", Path(tempfile.mkdtemp()) / "none.sha256"):
            self.assertFalse(gate.judge(rows, {"met": True})["passed"])  # unpinned: never passed

    def test_the_committed_result_holds_no_case_or_column(self):
        from dclab_rnd.agent_eval.planning_gate_v3 import gate

        rows = [row(f"case{i}", a, forbidden_before=["secret_col"], forbidden_after=[]) for i in range(8) for a in ("rules", "model")]
        text = str(gate.public(gate.judge(rows, {"met": True})))
        self.assertNotIn("case", text.replace("_case", ""))
        self.assertNotIn("secret_col", text)

    def test_the_terms_pin_the_checks_the_builder_the_reader_and_the_held_material(self):
        from dclab_rnd.agent_eval.planning_gate_v3 import gate

        for key in ("checks_sha256", "builder_sha256", "reader_sha256"):
            self.assertTrue(gate.TERMS[key], key)
        self.assertIn("held_tables_pin", gate.TERMS)
        self.assertIn("held_people_pin", gate.TERMS)

    def test_pin_is_refused_before_the_held_material_exists(self):
        from dclab_rnd.agent_eval import planning_gate_v3 as cli

        with mock.patch.dict(cli.gate.TERMS, {"held_tables_pin": None}):
            self.assertEqual(cli._cmd_pin(), 2)

    def test_the_gate_is_pinned_and_its_terms_unchanged(self):
        from dclab_rnd.agent_eval.planning_gate_v3 import gate

        if not gate.PIN.exists():
            self.skipTest("hp_gate_v3 is not pinned yet")
        self.assertIsNone(gate.pinned(), "hp_gate_v3's terms changed after the pin: a changed gate is a new gate")


class CommandTests(unittest.TestCase):
    def people(self, folder, run="HAB-100", set_name="plan_held_v3", at="2099-01-01T00:00:00+00:00", sha=None, **more):
        from dclab_rnd.agent_eval import planning_gate_v3 as cli

        pinned = (cli.ROOT / "dclab_rnd/agent_eval/harness_ablation/cases_plan_held_v3.sha256").read_text().split()[0]
        doc = {"set": set_name, "at": at, "manifest": {"cases_sha256": sha or pinned}, "models": {"arms": "m", "judge": "j"},
               "scores": {"case": [], "model_only": []}, **more}
        (folder / f"{run}_dev.json").write_text(__import__("json").dumps(doc))

    def test_the_people_run_is_refused_unless_it_is_the_only_one_on_the_pinned_people_after_the_pin(self):
        from dclab_rnd.agent_eval import planning_gate_v3 as cli

        folder = Path(tempfile.mkdtemp())
        pin = Path(tempfile.mkdtemp()) / "pin"
        pin.write_text("x")
        with mock.patch.object(cli, "PEOPLE", folder), mock.patch.object(cli.gate, "PIN", pin):
            self.people(folder)
            self.assertEqual(cli._people_rows("HAB-100")[0], "m")
            self.people(folder, run="HAB-101")  # a second run on the same people: one could be picked
            with self.assertRaises(ValueError):
                cli._people_rows("HAB-100")
            (folder / "HAB-101_dev.json").unlink()
            self.people(folder, sha="0" * 64)
            with self.assertRaises(ValueError):  # not the pinned people
                cli._people_rows("HAB-100")
            self.people(folder, at="2000-01-01T00:00:00+00:00")
            with self.assertRaises(ValueError):  # before the pin
                cli._people_rows("HAB-100")
            self.people(folder, set_name="plan_held")
            with self.assertRaises(ValueError):  # v2's people
                cli._people_rows("HAB-100")

    def test_the_tables_are_refused_after_a_start_marker_and_not_by_v2s_run(self):
        from dclab_rnd.agent_eval import planning_gate_v3 as cli

        marker = Path(tempfile.mkdtemp()) / "tables_started.json"
        with mock.patch.object(cli.gate, "pinned", return_value=None), mock.patch.object(cli, "MARKER", marker), \
                mock.patch.object(cli, "_runs", return_value=[]), mock.patch.object(cli, "RESULTS", Path(tempfile.mkdtemp())):
            marker.write_text("{}")
            self.assertEqual(cli._cmd_tables("qwen/qwen3.8-flash", 0.5, False), 2)
        self.assertNotIn("v2", str(cli._runs.__code__.co_consts))

    def test_preflight_needs_a_real_column_for_no_group_needed(self):
        from types import SimpleNamespace

        import pandas as pd

        from dclab_rnd.agent_eval import planning_gate_v3 as cli

        tables = SimpleNamespace(**{f"make_k{i}": (lambda seed: {"t.csv": pd.DataFrame({"x": [1], "shop": [1]})}) for i in range(8)})
        cases = [{"id": f"k{i}", "enforceable_traps": [{"kind": "late_column", "column": "x"}]} for i in range(8)]
        self.assertIn("at least one", cli.preflight(cases, [1], tables))  # no case names one: check 8 would test nothing
        cases[0]["no_group_needed"] = "shop"
        self.assertIsNone(cli.preflight(cases, [1], tables))
        cases[0]["no_group_needed"] = ["shop"]
        self.assertIn("1 cases whose no_group_needed is not a column", cli.preflight(cases, [1], tables))


class ReaderTests(unittest.TestCase):
    def test_a_needless_entity_split_is_read_and_a_reader_error_never_loses_the_row(self):
        from types import SimpleNamespace

        from dclab_rnd.agent_eval.planning_gate_v3 import run

        built = {"project_id": "p1", "planted_traps": [{"kind": "late_column", "enforced": True}]}
        store = SimpleNamespace(get=lambda pid: {"solution": {"group_column": "shop", "time_column": "t"}})
        with mock.patch.object(run, "build_v2", return_value=dict(built)), mock.patch("dclab_rnd.storage.open_stores", return_value=(store,)):
            out = run.build({"no_group_needed": "shop"}, {}, "rules", Path(tempfile.mkdtemp()))
        self.assertEqual((out["needless_group"], out["no_group_case"], out["planted_by_kind"]), (1, True, {"late_column": [1, 1]}))
        with mock.patch.object(run, "build_v2", return_value=dict(built)), mock.patch("dclab_rnd.storage.open_stores", side_effect=OSError):
            out = run.build({"no_group_needed": "shop"}, {}, "rules", Path(tempfile.mkdtemp()))
        self.assertEqual((out["reader_error"], out["needless_group"], out["project_id"]), ("OSError", None, "p1"))

    def test_a_built_case_says_its_split_columns_and_a_needless_group(self):
        from dclab_rnd.agent_eval.planning_gate_v3 import run
        from dclab_rnd.expansion.runner import fast_profile
        from tests.test_planning_gate_v2 import invented_case

        case, frames = invented_case()
        case["no_group_needed"] = "shop_id"
        with fast_profile():
            out = run.build(case, frames, "rules", Path(tempfile.mkdtemp()))
        self.assertEqual(out["needless_group"], 0)  # the brief asks for no new shops: no entity split
        self.assertEqual(out["time_column"], "created_at")
        self.assertEqual(sorted(out["planted_by_kind"]), ["late_column", "time_split"])


if __name__ == "__main__":
    unittest.main()
