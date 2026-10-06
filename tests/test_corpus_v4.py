"""The corpus v4 recipe (package A6.2): nothing until the threshold, whole projects held out, v3's gates."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.studio import corpus_v4 as v4  # noqa: E402


def move(pid, n, state, mv, verdict, stage=None):
    return {"kind": "move", "project_id": pid, "n": n, "actor": "agent", "state": state, "node": "WF-04", "allowed": ["run_stage:data", "capture"],
            "move": mv, "arguments": {"stage": stage} if stage else {}, "verdict": verdict, "failed_checks": [] if verdict == "allowed" else ["solution_signed"],
            "cited": ["DCLAB-R01"], "next": {"outcome": "done" if verdict == "allowed" else None, "to": "WF-05", "state": None}}


def step(pid, run, n, state, tool, verdict="ok", **args):
    return {"kind": "step", "project_id": pid, "run": run, "agent": "intern", "n": n, "chosen_by": "plan", "state": state,
            "allowed": ["describe_data", "run_stage", "set_solution"], "tool": tool, "arguments": {"project_id": pid, **args}, "verdict": verdict,
            "cited": [], "next": {"state": None}}


def varied(projects=12, runs=4):
    """Projects that decide differently: their states differ, so every trajectory is distinct."""
    records = []
    for i in range(projects):
        pid = f"p{i:02d}"
        s = f"{'d' * (i % 5)}{'-' * (10 - i % 5)}"
        records += [move(pid, 1, s + f"/{i}", "set_solution", "allowed"), move(pid, 2, s + f"/{i}", "run_stage", "blocked", "final"),
                    move(pid, 3, s + f"/{i}", "run_stage", "allowed", "data"), move(pid, 4, s + f"/{i}", "capture", "allowed")]
        for r in range(runs):
            records += [step(pid, f"s{i}-{r}", n, f"{s}/{i}/{r}", tool) for n, tool in enumerate(("describe_data", "set_solution", "run_stage"), 1)]
    return records


def standard(projects=12):
    """What the standard plan really leaves: every project decides the same, only its ids differ."""
    records = []
    for i in range(projects):
        pid = f"std{i:02d}"
        records += [move(pid, 1, "----------", "set_solution", "allowed"), move(pid, 2, "dd--------", "run_stage", "allowed", "data"),
                    move(pid, 3, "dddd------", "capture", "allowed")]
        records += [step(pid, f"run{i}", n, st, tool, key="telco_churn") for n, (st, tool) in
                    enumerate((("----------", "describe_data"), ("----------", "set_solution"), ("dd--------", "run_stage")), 1)]
    return records


class CountTests(unittest.TestCase):
    def test_a_trajectory_needs_three_decisions_and_the_threshold_needs_both_counts(self):
        records = varied(projects=2, runs=1) + [step("p00", "short", 1, "x", "describe_data"), step("p00", "short", 2, "x", "run_stage")]
        self.assertEqual(len(v4.trajectories_of(records)), 4)  # two move logs and two runs; the two-step run does not count
        self.assertFalse(v4.status(v4.trajectories_of(varied(projects=9, runs=6)))["ready"])  # 63 trajectories, 9 projects
        self.assertFalse(v4.status(v4.trajectories_of(varied(projects=12, runs=2)))["ready"])  # 36 trajectories, 12 projects
        ready = v4.status(v4.trajectories_of(varied(projects=12, runs=4)))
        self.assertEqual((ready["trajectories"], ready["projects"], ready["ready"]), (60, 12, True))
        edge = v4.status(v4.trajectories_of(varied(projects=10, runs=4)))
        self.assertEqual((edge["trajectories"], edge["projects"], edge["ready"]), (50, 10, True))  # exactly at the threshold

    def test_identical_runs_count_once_and_ids_are_masked(self):
        trajs = v4.trajectories_of(standard(projects=30))
        state = v4.status(trajs)
        self.assertEqual((state["recorded"], state["trajectories"]), (60, 2))  # the same move log and the same run, thirty times
        self.assertFalse(state["ready"])
        text = json.dumps([(ex["user"], ex["assistant"]) for t in trajs for ex in v4.examples_of(t)])
        self.assertNotIn("std0", text)  # no project id in a prompt or a target (the metadata keeps it, for the split)
        self.assertIn("<project>", text)

    def test_below_the_threshold_nothing_is_written(self):
        home, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()) / "out_v4"
        with mock.patch.object(v4, "workspace_records", return_value=(varied(projects=3), {})):
            result = v4.build(home, out)
        self.assertEqual((result["written"], result["trajectories"]), (False, 15))
        self.assertFalse(out.exists())
        with self.assertRaises(SystemExit):
            v4.build(home, v4.V3 / "v4")  # never into the v3 corpus


class ProjectTests(unittest.TestCase):
    def test_a_project_on_a_held_out_dataset_is_recognised_whatever_built_it(self):
        self.assertEqual(v4.project_meta({"suggestion": {"sample": "spambase"}})["held_out_dataset"], "spambase")  # the intern's sample
        self.assertEqual(v4.project_meta({"data": {"filename": "heart_disease.parquet", "columns": ["a"]}})["held_out_dataset"], "heart_disease")
        columns = next(iter(v4._held_out_columns()), None)
        if columns:  # Home cleans and renames the file; the columns stay
            self.assertTrue(v4.project_meta({"data": {"filename": "upload.csv", "columns": sorted(columns)}})["held_out_dataset"])

    def test_a_project_on_a_judgment_suite_table_is_left_out(self):
        from dclab_rnd.agent_eval.cases import CASES
        meta = {"bench": v4.project_meta({"data": {"filename": "t.csv", "columns": list(CASES[0].table().columns)}}), "p00": {}}
        self.assertEqual(meta["bench"]["benchmark_case"], CASES[0].id)
        records = varied(projects=1) + [{**r, "project_id": "bench"} for r in varied(projects=1)]
        self.assertEqual({t["project_id"] for t in v4.trajectories_of(records, meta)}, {"p00"})

    def test_a_project_keeps_its_split_as_the_workspace_grows(self):
        small = v4.held_out_projects([f"p{i:02d}" for i in range(12)], {})
        large = v4.held_out_projects([f"p{i:02d}" for i in range(40)], {})
        self.assertEqual(small, {p for p in large if p in {f"p{i:02d}" for i in range(12)}})
        self.assertTrue(v4.held_out_projects(["only", "two"], {}))  # never zero with two projects or more
        ids = [f"q{i}" for i in range(40)]
        none_hash = [p for p in ids if v4._bucket(p) % v4.VAL_EVERY][:3]  # three projects none of which hashes to validation
        self.assertEqual(v4.held_out_projects(none_hash, {}), {v4.fallback_held_out(none_hash)})  # the fallback, and it is named
        self.assertIsNone(v4.fallback_held_out([f"p{i:02d}" for i in range(12)]))


class RecipeTests(unittest.TestCase):
    @classmethod
    def build(cls, records, meta):
        out = Path(tempfile.mkdtemp()) / "out_v4"
        with mock.patch.object(v4, "workspace_records", return_value=(records, meta)):
            result = v4.build(Path(tempfile.mkdtemp()), out)
        split = {s: [json.loads(l) for l in (out / f"{s}.chat.jsonl").read_text().splitlines()] for s in ("train", "val")}
        return result, split, json.loads((out / "MANIFEST.json").read_text())

    @classmethod
    def setUpClass(cls):
        cls.records = varied() + standard(projects=4)
        cls.meta = {"p03": {"held_out_dataset": "heart_disease"}}
        cls.result, cls.split, cls.manifest = cls.build(cls.records, cls.meta)

    def mine(self, split):
        return [e for e in self.split[split] if e["metadata"].get("corpus") == "v4_trajectory"]

    def test_whole_projects_are_held_out_and_v3_keeps_its_split(self):
        self.assertTrue(self.result["written"])
        val_p = {e["metadata"]["project_id"] for e in self.mine("val")}
        train_p = {e["metadata"]["project_id"] for e in self.mine("train")}
        self.assertEqual(val_p & train_p, set())
        self.assertEqual(self.manifest["projects_in_both_splits"], [])
        self.assertIn("p03", val_p)  # on a dataset v3 holds out: always validation
        v3 = v4.v3_examples()
        self.assertEqual(self.manifest["from_v3"], {"train": len(v3["train"]), "val": len(v3["val"])})
        self.assertEqual(self.split["val"][:len(v3["val"])], v3["val"])

    def test_the_gates_fire_on_what_the_standard_plan_leaves(self):
        gates = self.manifest["quality_gates"]
        self.assertGreater(gates["duplicates_removed"], 0)  # the standard runs repeat each other once ids are masked
        val_keys = {v4._key(e["messages"][1]["content"]) for e in self.split["val"]}
        val_sh = [v4._shingles(e["messages"][1]["content"]) for e in self.split["val"]]
        for e in self.mine("train"):
            self.assertNotIn(v4._key(e["messages"][1]["content"]), val_keys)
            self.assertFalse(any(v4.near_duplicate(v4._shingles(e["messages"][1]["content"]), s) for s in val_sh))
        self.assertNotIn("std0", json.dumps([e["messages"] for s in self.split.values() for e in s]))

    def test_the_split_does_not_depend_on_the_order_of_the_records(self):
        _, again, manifest = self.build(list(reversed(self.records)), self.meta)
        mine = lambda s: sorted(json.dumps(e, sort_keys=True) for e in s if e["metadata"].get("corpus") == "v4_trajectory")
        self.assertEqual((mine(again["train"]), mine(again["val"])), (mine(self.split["train"]), mine(self.split["val"])))
        self.assertEqual(manifest["quality_gates"], self.manifest["quality_gates"])

    def test_a_refused_move_is_judged_but_never_a_target(self):
        for e in self.mine("train") + self.mine("val"):
            answer = json.loads(e["messages"][-1]["content"])
            if e["metadata"]["task"] == "next_move":
                self.assertFalse(answer.get("move") == "run_stage" and answer.get("arguments", {}).get("stage") == "final", e)
        judged = [json.loads(e["messages"][-1]["content"]) for e in self.mine("train") + self.mine("val") if e["metadata"]["task"] == "judge_move"]
        self.assertIn("blocked", {j["verdict"] for j in judged})

    def test_an_example_with_a_path_or_a_hash_is_rejected(self):
        self.assertFalse(v4._clean({"user": "see /tmp/x", "assistant": "{}"}))
        self.assertFalse(v4._clean({"user": "x", "assistant": "a" * 40}))  # a 40-character hex string
        self.assertTrue(v4._clean({"user": "State: dd--------", "assistant": '{"tool": "run_stage"}'}))


class PolicyPageTests(unittest.TestCase):
    def test_stage_two_is_ready_only_at_both_thresholds(self):
        from dclab_rnd.agentic.pages import learn
        base = {"trajectories": 0, "moves": 0, "by_status": {}, "sharing": [], "opted_in": 0}
        for trajs, ready in ((varied(projects=9, runs=6), False), (varied(projects=12, runs=4), True)):
            use = v4.status(v4.trajectories_of(trajs))
            step2 = learn.curriculum({"available": True, "total": 1}, {"runs": []}, {**base, "usable": use})[1]
            self.assertEqual(step2["ready"], ready)
            self.assertIn(f"{use['trajectories']} of 50 trajectories", step2["detail"])

    def test_the_page_reads_the_real_count_and_says_when_it_cannot(self):
        from fastapi.testclient import TestClient
        from dclab_rnd.agentic.server import create_app

        with TestClient(create_app(Path(tempfile.mkdtemp()))) as c:
            with mock.patch.object(v4, "workspace_records", return_value=(varied(projects=12, runs=4), {})):
                d = c.get("/api/learn/policy").json()
            self.assertEqual((d["trajectories"]["usable"]["trajectories"], d["trajectories"]["usable"]["projects"]), (60, 12))
            self.assertTrue(d["curriculum"][1]["ready"])
            with mock.patch.object(v4, "workspace_records", side_effect=OSError("disk")):
                d = c.get("/api/learn/policy").json()
            self.assertEqual(d["trajectories"]["usable"]["error"], "OSError")
            self.assertIn("could not be read", d["curriculum"][1]["detail"])
            self.assertFalse(d["curriculum"][1]["ready"])


if __name__ == "__main__":
    unittest.main()
