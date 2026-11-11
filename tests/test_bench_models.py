"""Step 6 of the 2026-11-11 plan: Claude's runs beside Qwen's across results, paired on the same people when they can be,
with each side's judge named, a rerun of a seen set kept apart from a first run, and a report never written over."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.bench import models  # noqa: E402

CLAUDE = {"arms": "claude-sonnet-5-5 (the product's)", "person": "claude-haiku-5-5", "judge": "claude-opus-5-5"}
QWEN = {"arms": "qwen/qwen3.8-flash", "person": "qwen/qwen3.8-flash", "judge": "qwen/qwen3.8-flash"}
PEOPLE = ["s3-01", "s3-02", "s3-03", "s3-04", "s3-05", "s3-06"]


def person(pid, arm, understanding, route_right, **extra):
    return {"id": pid, "arm": arm, "judged": True, "failed": False, "unjudged": False, "understanding": understanding,
            "route_right": route_right, "safety_failures": 0, **extra}


def write(campaigns: Path, suite: str, ref: str, doc: dict) -> None:
    (campaigns / suite / "results").mkdir(parents=True, exist_ok=True)
    (campaigns / suite / "results" / f"{ref}.json").write_text(json.dumps(doc))


class ModelComparisonTests(unittest.TestCase):
    def setUp(self):
        self.campaigns = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.campaigns, True)
        # Claude's first run on sealed_v3 and Qwen's rerun of the same people (unseen: false)
        write(self.campaigns, "conversation_exam_v1", "CEX-032_sealed_v3", {
            "id": "CEX-032", "split": "sealed_v3", "at": "2026-11-01T20:00:00+00:00", "arms": ["plain", "case"], "models": CLAUDE,
            "scores": {"case": [person(p, "case", 1.0, True) for p in PEOPLE],
                       "plain": [person(p, "plain", 0.5, i % 2 == 0) for i, p in enumerate(PEOPLE)]}})
        write(self.campaigns, "conversation_exam_v1", "CEX-050_sealed_v3_rerun_qwen", {
            "id": "CEX-050", "split": "sealed_v3", "at": "2026-11-12T10:00:00+00:00", "arms": ["plain", "case"], "models": QWEN,
            "unseen": False, "unseen_why": "OWNER-2026-11-11: one rerun per set per model",
            "scores": {"case": [person(p, "case", 0.75, i < 3) for i, p in enumerate(PEOPLE)]
                       + [{"id": "s3-07", "failed": True, "understanding": 0.0}],  # a failed row is left out
                       "plain": [person(p, "plain", 0.5, True) for p in PEOPLE]}})
        # a Qwen first run on dev with other people than Claude's dev run: unpaired
        write(self.campaigns, "conversation_exam_v1", "CEX-010_dev", {
            "id": "CEX-010", "split": "dev", "at": "2026-10-20T10:00:00+00:00", "arms": ["case"], "models": CLAUDE,
            "scores": {"case": [person("d-01", "case", 1.0, True), person("d-02", "case", 0.5, True)]}})
        write(self.campaigns, "conversation_exam_v1", "CEX-051_dev", {
            "id": "CEX-051", "split": "dev", "at": "2026-11-12T11:00:00+00:00", "arms": ["case"], "models": QWEN,
            "scores": {"case": [person("d-03", "case", 0.5, False), person("d-04", "case", 0.5, True)]}})

    def doc(self, **kw):
        return models.build(self.campaigns, **kw)

    def rows(self, doc=None, **where):
        doc = doc or self.doc()
        return [r for r in doc["rows"] if all(r.get(k) == v for k, v in where.items())]

    def test_the_same_people_are_paired_with_mcnemar_and_the_judges_are_named(self):
        r = self.rows(split="sealed_v3", arm="case", metric="route_right")[0]
        self.assertEqual(r["mode"], "paired")
        p = r["paired"]
        self.assertEqual((p["n"], p["mean_a"], p["mean_b"], p["losses_b"], p["wins_b"]), (6, 1.0, 0.5, 3, 0))
        self.assertEqual(p["mcnemar_p"], 0.25)  # three discordant pairs, all one way
        self.assertEqual((r["baseline"]["model"], r["baseline"]["judge"], r["baseline"]["person"]),
                         ("claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"))
        self.assertEqual((r["target"]["model"], r["target"]["judge"]), ("qwen/qwen3.8-flash", "qwen/qwen3.8-flash"))
        self.assertTrue(r["judges_differ"])
        self.assertIn("judged by claude-opus-5-5 against qwen/qwen3.8-flash", r["note"])
        u = self.rows(split="sealed_v3", arm="case", metric="understanding")[0]
        self.assertEqual(u["paired"]["n"], 6)  # the failed row s3-07 is not a case
        self.assertEqual(u["paired"]["difference"], -0.25)

    def test_a_rerun_is_marked_and_kept_apart_from_a_first_run(self):
        write(self.campaigns, "conversation_exam_v1", "CEX-052_sealed_v3", {  # a first run of the target on the same split
            "id": "CEX-052", "split": "sealed_v3", "at": "2026-11-13T10:00:00+00:00", "arms": ["case"], "models": QWEN,
            "scores": {"case": [person(p, "case", 1.0, True) for p in PEOPLE]}})
        rows = self.rows(split="sealed_v3", arm="case", metric="understanding")
        by_run = {r["target"]["run"]: r for r in rows}
        self.assertEqual(set(by_run), {"rerun (not unseen)", "first run"})
        self.assertEqual(by_run["rerun (not unseen)"]["target"]["runs"], ["CEX-050_sealed_v3_rerun_qwen"])
        self.assertEqual(by_run["first run"]["target"]["runs"], ["CEX-052_sealed_v3"])
        self.assertIn("not unseen", by_run["rerun (not unseen)"]["note"])
        self.assertNotIn("not unseen", by_run["first run"]["note"])
        headers = {h["ref"]: h for h in self.doc()["runs"]}
        self.assertEqual(headers["CEX-050_sealed_v3_rerun_qwen"]["run"], "rerun (not unseen)")
        self.assertEqual(headers["CEX-050_sealed_v3_rerun_qwen"]["split"], "sealed_v3")

    def test_other_people_are_reported_unpaired_with_each_sides_interval(self):
        r = self.rows(split="dev", arm="case", metric="route_right")[0]
        self.assertEqual(r["mode"], "unpaired")
        self.assertNotIn("paired", r)
        self.assertIn("0 shared cases", r["note"])
        self.assertEqual((r["baseline"]["n"], r["baseline"]["mean"], r["target"]["mean"]), (2, 1.0, 0.5))
        self.assertIsNotNone(r["target"]["interval"][0])  # Wilson's for a yes/no metric

    def test_arms_inside_one_model_are_compared_across_its_runs(self):
        pairs = [r for r in self.doc()["arm_pairs"] if r["answered_by"] == "qwen/qwen3.8-flash" and r["metric"] == "route_right"]
        self.assertEqual([(r["arm"], r["arms"]) for r in pairs], [("case vs plain", ["plain", "case"])])
        self.assertEqual(pairs[0]["paired"]["difference"], -0.5)  # Qwen's lead agent right on 3, Qwen alone on 6

    def test_homes_arm_called_model_keeps_its_side_and_the_report_is_written_whole(self):
        """The review of 2026-11-11: Home's arm "model" overwrote the row's model, and render crashed after the JSON was written."""
        for ref, model in (("HOME-024_dev", "claude-sonnet-5-5"), ("HOME-031_dev", "qwen/qwen3.8-flash")):
            write(self.campaigns, "home_v1", ref, {"split": "dev", "arms": ["rules", "model", "plain"], "plan": {"model": model},
                                                    "rows": [{"case": c, "scores": {"rules": {"track": "right"}, "model": {"track": "right"},
                                                                                     "plain": {"track": "wrong" if c == "c1" else "right"}}}
                                                             for c in ("c1", "c2", "c3")]})
        doc = self.doc()
        home = [r for r in doc["arm_pairs"] if r["suite"] == "home_v1" and r["metric"] == "track_right"]
        self.assertEqual({r["answered_by"] for r in home}, {"claude-sonnet-5-5", "qwen/qwen3.8-flash"})
        for r in home:
            self.assertEqual((r["a"]["mean"], r["b"]["mean"], r["arms"]), (0.6667, 1.0, ["plain", "model"]))
        self.assertTrue(any(r["suite"] == "home_v1" and r["arm"] == "model" for r in doc["rows"]))
        js, md = models.write(doc, self.campaigns / "out")
        self.assertIn("home_v1", md.read_text())
        self.assertEqual(json.loads(js.read_text())["schema"], "dclab_rnd.bench_models.v1")

    def test_a_render_failure_leaves_no_file(self):
        out = self.campaigns / "out"
        with self.assertRaises(KeyError):
            models.write({"at": "2026-11-11T00:00:00+00:00", "target": "q", "baseline": "c"}, out)
        self.assertEqual(list(out.glob("*")), [])

    def test_a_run_answered_by_two_models_is_on_neither_side(self):
        mixed = {"live": True, "models": {"home_agent": "qwen/qwen3.8-flash", "data_investigator": "claude-sonnet-5-5"},
                 "rows": [{"id": i, "passed": True} for i in ("RG-01", "RG-02")]}
        write(self.campaigns, "harness_v1", "HRN-300", mixed)
        write(self.campaigns, "harness_v1", "HRN-301", {**mixed, "models": {"home_agent": "qwen/qwen3.8-flash"}})
        doc = self.doc()
        self.assertEqual({h["model"] for h in doc["runs"] if h["suite"] == "harness_v1"}, {"claude-sonnet-5-5 + qwen/qwen3.8-flash", "qwen/qwen3.8-flash"})
        self.assertEqual(self.rows(doc, suite="harness_v1"), [])  # Qwen is never compared with a half-Qwen baseline

    def test_an_edited_journey_set_is_another_split(self):
        for ref, model, digest in (("JRN-030_dev", "claude-sonnet-5-5", "a" * 64), ("JRN-031_dev", "qwen/qwen3.8-flash", "b" * 64)):
            write(self.campaigns, "journeys_v1", ref, {"split": "dev", "arm": "single_live", "journeys": digest,
                                                        "rows": [{"id": j, "right": True, "models": {"conversation_reader": model}} for j in ("JD-01", "JD-02")]})
        self.assertEqual(self.rows(suite="journeys_v1"), [])
        self.assertEqual({h["split"] for h in self.doc()["runs"] if h["suite"] == "journeys_v1"}, {"dev@" + "a" * 12, "dev@" + "b" * 12})

    def test_arms_without_a_model_and_scripted_runs_are_left_out_and_unknown_is_never_guessed(self):
        write(self.campaigns, "home_v1", "HOME-030_dev", {"split": "dev", "arms": ["rules", "model"], "plan": {"model": "qwen/qwen3.8-flash"},
                                                          "rows": [{"case": "c1", "scores": {"rules": {"track": "right"}, "model": {"track": "wrong"}}}]})
        write(self.campaigns, "harness_v1", "HRN-200", {"live": False, "models": {}, "rows": [{"id": "RG-01", "passed": True}]})
        write(self.campaigns, "journeys_v1", "JRN-040_dev", {"split": "dev", "arm": "single_live", "rows": [{"id": "JD-01", "right": True}]})
        runs = {(h["ref"], h["arm"]): h for h in self.doc()["runs"]}
        self.assertNotIn(("HOME-030_dev", "rules"), runs)
        self.assertEqual(runs[("HOME-030_dev", "model")]["model"], "qwen/qwen3.8-flash")
        self.assertEqual(runs[("HOME-030_dev", "model")]["model_source"], "plan.model")
        self.assertEqual(runs[("HOME-030_dev", "model")]["judge"], models.CODE_JUDGE)
        self.assertNotIn(("HRN-200", "harness"), runs)
        self.assertEqual(runs[("JRN-040_dev", "single_live")]["model"], models.UNKNOWN)

    def test_model_of_reads_every_recorded_place(self):
        self.assertEqual(models.model_of({"models": {"arms": {"case": "qwen/a", "plain": "claude-b"}}}, "plain"), ("claude-b", "models.arms.plain"))
        self.assertEqual(models.model_of({"models": {"arms": {"case": "qwen/a", "plain": "claude-b"}}})[0], "claude-b + qwen/a")
        self.assertEqual(models.model_of({"models": {"data_investigator": "claude-x", "home_agent": "claude-x"}}), ("claude-x", "models"))
        self.assertEqual(models.model_of({"rows": [{"models": {"conversation_reader": "claude-y"}}]}), ("claude-y", "rows[].models"))
        self.assertEqual(models.model_of({}), ("unknown", ""))

    def test_a_suite_without_case_rows_is_unpaired_on_its_totals(self):
        for ref, model, acc in (("SEX-005", "claude-sonnet-5-5", 0.84), ("SEX-008", "qwen/qwen3.8-flash", 0.80)):
            write(self.campaigns, "students_exam_v1", ref, {"model": model, "per_point": 300, "seed": 0, "scores": {
                "column.time": {"on_common": {"claude": {"n": 244, "accuracy": acc, "accuracy_ci": [acc - 0.05, acc + 0.04]}}}}})
        r = self.rows(suite="students_exam_v1")[0]
        self.assertEqual((r["mode"], r["metric"], r["split"]), ("unpaired", "column.time.accuracy", "per_point_300_seed_0"))
        self.assertIn("no per-case rows", r["note"])
        self.assertEqual((r["baseline"]["mean"], r["target"]["mean"], r["target"]["n"]), (0.84, 0.80, 244))
        write(self.campaigns, "students_exam_v1", "SEX-004", {"model": "claude-sonnet-5-5", "per_point": 300, "seed": 0, "scores": {
            "column.time": {"on_common": {"claude": {"n": 244, "accuracy": 0.5}}}}})
        r = self.rows(suite="students_exam_v1")[0]
        self.assertEqual((r["baseline"]["runs"], r["baseline"]["mean"]), (["SEX-005"], 0.84))  # the latest total, and only it listed

    def test_a_report_is_written_once_as_new_dated_files_never_over_one(self):
        out = self.campaigns / "out"
        doc = self.doc()
        js, md = models.write(doc, out)
        self.assertTrue(js.name.startswith("QWEN_VS_CLAUDE_") and js.suffix == ".json" and md.suffix == ".md")
        self.assertEqual(json.loads(js.read_text())["schema"], "dclab_rnd.bench_models.v1")
        text = md.read_text()
        self.assertIn("claude-opus-5-5", text)
        self.assertIn("rerun (not unseen)", text)
        self.assertIn("Limits", text)
        before = js.read_text()
        with self.assertRaises(FileExistsError):
            models.write(doc, out)  # the same time and process: refused, the first file kept
        self.assertEqual(js.read_text(), before)
        later = models.write({**doc, "at": "2099-01-01T00:00:00+00:00"}, out)
        self.assertNotEqual(later[0], js)
        self.assertEqual(len(list(out.glob("*.json"))), 2)


class ColumnTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)

    def test_the_exam_benchmark_has_a_model_and_a_judge_column(self):
        from dclab_rnd.agent_eval.conversation_exam import benchmark

        results = self.home / "results"
        results.mkdir()
        arm = {"people": 2, "understanding": 0.5}
        (results / "CEX-001_dev.json").write_text(json.dumps({"id": "CEX-001", "split": "dev", "models": CLAUDE, "summary": {"plain": arm}}))
        (results / "CEX-002_dev.json").write_text(json.dumps({"id": "CEX-002", "split": "dev", "models": QWEN, "summary": {"plain": arm}}))
        rows = {r["run"]: r for r in benchmark.runs(results)}
        self.assertEqual((rows["CEX-001"]["model"], rows["CEX-001"]["judge"]), ("claude-sonnet-5-5", "claude-opus-5-5"))
        self.assertEqual((rows["CEX-002"]["model"], rows["CEX-002"]["judge"]), ("qwen/qwen3.8-flash", "qwen/qwen3.8-flash"))
        text = benchmark.render({"runs": list(rows.values()), "labels": [], "capture": {"rule": "r", "workspace": {}, "vaults": {}}})
        self.assertIn("| student | model | judge |", text)
        self.assertIn("| CEX-001 | dev | Claude alone | claude-sonnet-5-5 | claude-opus-5-5 |", text)
        self.assertIn("| CEX-002 | dev | the model alone | qwen/qwen3.8-flash | qwen/qwen3.8-flash |", text)

    def test_bench_versions_names_the_model_of_each_live_result(self):
        from dclab_rnd.bench import versions

        write(self.home, "conversation_exam_v1", "CEX-001_dev", {"at": "2026-11-12T10:00:00+00:00", "arms": ["case"], "models": QWEN,
                                                                "scores": {"case": []}})
        live = versions.live_results(self.home)
        self.assertEqual({(r["model"], r["judge"]) for r in live["case_agent"]}, {("qwen/qwen3.8-flash", "qwen/qwen3.8-flash")})


if __name__ == "__main__":
    unittest.main()
