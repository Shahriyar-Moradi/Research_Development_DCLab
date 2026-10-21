"""The M5 solver trajectories, scored and checked (the owner, 2026-10-21: "the best agent for honest ML on business table data").

An intern session saved as ``intern/<id>.json`` (a twin lane, a live test) is an intern session; a Claude Code session keeps its
id through redaction; a trajectory carries its holdout score, the table it ran on and its planted truth; a run made by the rules
or a local model that passed its truth goes to ``trajectories/studio``, one made by Claude, GPT or any hosted model to
``trajectories/silver_frontier`` at half weight, its manifest not cleared for training; a run that never ran its final stage, one
that failed its planted truth, and a twin lane whose verdict is missing are placed nowhere; one table is never in two splits, and a
sealed session seals its table; a table's cells never reach a placed row (an old session's summary quotes them, so it is dropped).
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)

AT = "2026-10-21T10:00:00+00:00"
SESSION_UUID = "3f2b8a64-0c1d-4e5f-9a7b-1c2d3e4f5a6b"
DESCRIBE = {"filename": "orders.csv", "rows": 2000, "columns": [{"name": "late", "kind": "numeric"}]}


def final(project: str, sha: str, value: float, completed: str = AT) -> dict:
    return {"stage": "final", "status": "completed", "task_type": "binary", "primary_metric": "roc_auc", "completed_at": completed,
            "evidence": {"holdout_consumed": True, "holdout_rows": 400},
            "run": {"schema_version": "dclab_core.run_record.v2", "project": project, "kind": "stage", "name": "final",
                    "inputs": {"data_sha256": sha}, "outputs": {"metric": "roc_auc", "value": value}}}


def session(sid: str, project: str, mode: str = "standard", model=None, whole: bool = True, runs: bool = True, helper=None) -> dict:
    quoted = json.dumps({**DESCRIBE, "columns": [{"name": "late", "kind": "numeric", "examples": ["58", "44", "33"]}]})[:60] + "…"
    steps = [{"n": 1, "tool": "describe_data", "arguments": {"project_id": project}, "summary": quoted, "ok": True, "at": AT}]
    if runs:
        steps.append({"n": 2, "tool": "run_all", "arguments": {"project_id": project}, "summary": "{\"stages\": 5}", "ok": True, "at": AT,
                      "elapsed_seconds": 10})
    if whole:
        steps[0]["result"] = DESCRIBE
        if runs:
            steps[1]["result"] = {"stages": 5}
        if helper:  # a tool inside the run answered by a model
            steps[0]["result"] = {**DESCRIBE, "mode": "model", "model": helper, "brief": {"note": "write to maria.lopez@acme.com"}}
    return {"id": sid, "task": "Predict which orders arrive late.", "status": "completed", "mode": mode, "model": model,
            "project_id": project, "steps": steps, "created": AT, "updated": AT}


def verdict(run: str, system: bool, ai_first: bool) -> dict:
    return {"schema_version": "dclab_core.twin_step.v1", "id": f"{run}:solution.columns", "session": run, "step": "solution.columns",
            "truth": {"value": {"target": "late", "leakage": ["delivered_at"]}, "source": "planted"},
            "verdict": {"fields": ["target", "leakage"], "system": {"target": system, "leakage": system},
                        "ai_first": {"target": ai_first, "leakage": ai_first}}}


def project_in(split: str, prefix: str) -> str:
    from dclab_core.datasets import group_of, split_of

    return next(p for p in (f"{prefix}{i}" for i in range(500)) if split_of(group_of(project=p)) == split)


@unittest.skipIf(MISSING, MISSING)
class SolverTrajectoriesTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout, Store, curators, shapers

        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)  # the disk is small: a test leaves nothing behind
        self.layout = Layout(Path(folder) / "store")
        store = Store(self.layout)

        def keep(kind, workspace, source, doc):
            store.keep(kind, workspace, source, json.dumps(doc).encode(), consent="train")

        # run r1: the rules' lane passed its planted truth, Claude's lane too; both ran on table sha-one
        keep("conversations", "twin__runs__r1__workflow", "intern/aaaaaaaaaaaa.json", session("aaaaaaaaaaaa", "p1"))
        keep("workflows", "twin__runs__r1__workflow", "projects/p1/stages/final.json", final("p1", "sha-one", 0.77))
        keep("conversations", "twin__runs__r1__ai_first", "intern/bbbbbbbbbbbb.json", session("bbbbbbbbbbbb", "p2", "llm", "claude-sonnet-5-5"))
        keep("workflows", "twin__runs__r1__ai_first", "projects/p2/stages/final.json", final("p2", "sha-one", 0.79))
        # run r2: the rules' lane picked the wrong target; run r3: its verdict was never kept
        keep("conversations", "twin__runs__r2__workflow", "intern/cccccccccccc.json", session("cccccccccccc", "p3"))
        keep("workflows", "twin__runs__r2__workflow", "projects/p3/stages/final.json", final("p3", "sha-two", 0.70))
        keep("conversations", "twin__runs__r3__workflow", "intern/c3c3c3c3c3c3.json", session("c3c3c3c3c3c3", "p3b"))
        keep("workflows", "twin__runs__r3__workflow", "projects/p3b/stages/final.json", final("p3b", "sha-two-b", 0.71))
        store.keep_log("conversations", "dev", "twin/runs/r1/steps.jsonl", (json.dumps(verdict("r1", True, True)) + "\n").encode(), consent="train")
        store.keep_log("conversations", "dev", "twin/runs/r2/steps.jsonl", (json.dumps(verdict("r2", False, True)) + "\n").encode(), consent="train")
        # never reached a holdout; read a scored project without running its final; from before whole results were kept
        keep("conversations", "dev", "intern/dddddddddddd.json", session("dddddddddddd", "p4"))
        keep("conversations", "dev", "intern/d2d2d2d2d2d2.json", session("d2d2d2d2d2d2", "p5", runs=False))
        keep("conversations", "dev", "intern/eeeeeeeeeeee.json", session("eeeeeeeeeeee", "p5", whole=False))
        keep("workflows", "dev", "projects/p5/stages/final.json", final("p5", "sha-three", 0.66))
        # a hosted model that is not Claude or GPT, on PostgreSQL (its stage as a database row)
        keep("conversations", "pg", "intern/ffffffffffff.json", session("ffffffffffff", "p6", "llm", "mistral-large-latest"))
        store.keep_log("workflows", "pg", "db/stage_record.jsonl",
                       (json.dumps({"project_id": "p6", "stage": "final", "record": final("p6", "sha-four", 0.81)}) + "\n").encode(), consent="train")
        # a sealed project and a train project on one table: both sealed
        self.sealed_p, self.train_p = project_in("sealed", "ps"), project_in("train", "pt")
        keep("conversations", "dev", "intern/111111111111.json", session("111111111111", self.sealed_p))
        keep("workflows", "dev", f"projects/{self.sealed_p}/stages/final.json", final(self.sealed_p, "sha-shared", 0.60))
        keep("conversations", "dev", "intern/222222222222.json", session("222222222222", self.train_p))
        keep("workflows", "dev", f"projects/{self.train_p}/stages/final.json", final(self.train_p, "sha-shared", 0.61))
        # a final scored before the session (by a person, by Claude): never the session's
        keep("conversations", "dev", "intern/a1a1a1a1a1a1.json", session("a1a1a1a1a1a1", "p7"))
        keep("workflows", "dev", "projects/p7/stages/final.json", final("p7", "sha-seven", 0.9, "2026-10-20T10:00:00+00:00"))
        # a sealed project's final run from the screen (no session), and a train project's session on the same table
        self.sealed_q, self.train_q = project_in("sealed", "qs"), project_in("train", "qt")
        keep("workflows", "dev", f"projects/{self.sealed_q}/stages/final.json", final(self.sealed_q, "sha-screen", 0.62))
        keep("conversations", "dev", "intern/333333333333.json", session("333333333333", self.train_q))
        keep("workflows", "dev", f"projects/{self.train_q}/stages/final.json", final(self.train_q, "sha-screen", 0.63))
        # a rules session whose leakage review a hosted model answered (its text sits in the run), an email in its result
        keep("conversations", "dev", "intern/444444444444.json", session("444444444444", "p8", helper="claude-sonnet-5-5"))
        keep("workflows", "dev", "projects/p8/stages/final.json", final("p8", "sha-eight", 0.7))
        # two self-play solves: seed 0 verified, seed 1 not
        for seed in (0, 1):
            ws = f"selfplay__ledgers__binary__{seed}"
            dec = [{"schema_version": "dclab_core.decision_record.v2", "id": f"dr{seed}a", "point": "target.column", "state": {"rows": "medium"},
                    "options": ["churned", "plan"], "answers": [{"by": "person", "value": "churned"}], "value_used": "churned", "used": "person",
                    "project": f"sp{seed}", "workspace": "selfplay", "at": AT},
                   {"schema_version": "dclab_core.decision_record.v2", "id": f"dr{seed}b", "point": "feature.leakage_suspect",
                    "subject": {"kind": "column", "key": "final_refund_amount"}, "state": {"lift": "high"}, "options": [True, False],
                    "answers": [{"by": "rule", "value": True, "reason": "its lift alone is too high to be known before the outcome"}],
                    "value_used": True, "used": "rule", "project": f"sp{seed}", "workspace": "selfplay", "at": "2026-10-21T10:00:01+00:00"}]
            store.keep_log("decisions", ws, "ledger/decisions/2026/10/21.jsonl", "".join(json.dumps(x) + "\n" for x in dec).encode(), consent="train")
            store.keep_log("decisions", ws, "ledger/labels/2026/10/21.jsonl", (json.dumps(
                {"schema_version": "dclab_core.decision_label.v4", "id": f"dl{seed}", "record": f"dr{seed}b", "point": "feature.leakage_suspect",
                 "value": seed == 0, "source": "benchmark", "by": "planted", "at": AT}) + "\n").encode(), consent="train")  # seed 1: the engine was wrong
            store.keep_log("decisions", ws, "ledger/outcomes/2026/10/21.jsonl", (json.dumps(
                {"schema_version": "dclab_core.outcome.v1", "id": f"do{seed}", "record": f"dr{seed}a", "point": "target.column", "metric": "roc_auc",
                 "value": 0.74, "observed": "holdout", "project": f"sp{seed}", "at": AT}) + "\n").encode(), consent="train")
        store.keep_log("other", "selfplay", "done.jsonl", (json.dumps({"group": "selfplay:binary:0", "verified": True}) + "\n"
                                                         + json.dumps({"group": "selfplay:binary:1", "verified": False}) + "\n").encode(), consent="train")
        # a Claude Code session whose id is a UUID
        lines = [{"type": "user", "sessionId": SESSION_UUID, "timestamp": AT, "uuid": "u1", "message": {"role": "user", "content": "Fix the test"}},
                 {"type": "assistant", "sessionId": SESSION_UUID, "timestamp": AT, "uuid": "u2", "message": {"role": "assistant", "content": [
                     {"type": "text", "text": "Done."}]}}]
        store.keep_log("agents", "rnd", f"claude_code/main/{SESSION_UUID}.jsonl", "".join(json.dumps(x) + "\n" for x in lines).encode(), consent="train")
        for agent in ("agent-a1", "agent-b2"):  # its subagents: their own runs (their own lines), under its session id
            own = [{**x, "uuid": f"{agent}-{x['uuid']}"} for x in lines]
            store.keep_log("agents", "rnd", f"claude_code/main/{SESSION_UUID}/subagents/{agent}.jsonl",
                           "".join(json.dumps(x) + "\n" for x in own).encode(), consent="train")
        shapers.shape(self.layout)
        curators.curate(self.layout, run="r1")
        from dclab_rnd.store.sets import llm

        self.report = llm.build(self.layout)

    def base(self, folder, version):
        return self.layout.sets("m5_solver_llm", *folder.split("/"), version)

    def rows(self, folder, version):
        base = self.base(folder, version)
        return {json.loads(line)["id"]: json.loads(line) for s in ("train", "dev", "sealed", "sealed.pending") if (base / f"{s}.jsonl").exists()
                for line in (base / f"{s}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}

    def placed(self):
        return {**self.rows("trajectories/studio", "v2"), **self.rows("trajectories/silver_frontier", "v1")}

    def test_an_intern_file_is_an_intern_session(self):
        from dclab_rnd.store import shapers

        self.assertEqual(shapers._type("conversations", "intern/aaaaaaaaaaaa.json", {}), "intern_session")
        self.assertEqual(shapers._type("conversations", "intern/versions/aaaaaaaaaaaa.jsonl", {}), "intern_session")
        self.assertEqual(shapers._type("conversations", "intern/memory.json", {}), "snapshot")

    def test_a_claude_code_session_keeps_its_id(self):
        self.assertIn(f"m5_solver_llm:claude_code:{SESSION_UUID}", self.rows("trajectories/development", "v2"))

    def test_scored_checked_runs_are_placed_by_who_made_them(self):
        from dclab_core.datasets import split_of
        from dclab_core.store import SolverTrajectory

        studio, silver = self.rows("trajectories/studio", "v2"), self.rows("trajectories/silver_frontier", "v1")
        rules = SolverTrajectory.model_validate(studio["m5_solver_llm:intern:aaaaaaaaaaaa"])
        self.assertEqual((rules.metric, rules.score), ("roc_auc", 0.77))
        self.assertEqual(rules.problem["answerer"], "rules")
        self.assertEqual(rules.submission["truth"], "passed")
        self.assertFalse(rules.submission["results_cut"])
        self.assertEqual(rules.steps[0].result, DESCRIBE)
        claude = SolverTrajectory.model_validate(silver["m5_solver_llm:intern:bbbbbbbbbbbb"])
        self.assertEqual((claude.problem["answerer"], claude.weight), ("claude:claude-sonnet-5-5", 0.5))
        self.assertNotIn("m5_solver_llm:intern:bbbbbbbbbbbb", studio)  # never in the clean set
        self.assertEqual(rules.split, claude.split)  # one table, one split
        self.assertEqual(rules.split, split_of("table:sha-one"))
        hosted = SolverTrajectory.model_validate(silver["m5_solver_llm:intern:ffffffffffff"])  # scored from a PostgreSQL row
        self.assertEqual((hosted.problem["answerer"], hosted.score), ("frontier:mistral-large-latest", 0.81))
        manifest = json.loads((self.base("trajectories/silver_frontier", "v1") / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["consent"], "improve")  # not cleared for training
        self.assertEqual(json.loads((self.base("trajectories/studio", "v2") / "MANIFEST.json").read_text(encoding="utf-8"))["consent"], "train")

    def test_unscored_unchecked_and_failed_runs_are_placed_nowhere(self):
        placed = self.placed()
        for sid in ("cccccccccccc", "c3c3c3c3c3c3", "dddddddddddd", "d2d2d2d2d2d2"):
            self.assertNotIn(f"m5_solver_llm:intern:{sid}", placed, sid)
        kept_out = self.report["kept_out"]
        self.assertEqual(kept_out.get("m5: failed its planted truth"), 1)
        self.assertEqual(kept_out.get("m5: its planted truth was not judged"), 1)
        self.assertEqual(kept_out.get("m5: not scored on its holdout"), 3)  # no final, a final it never ran, a final made before it
        self.assertNotIn("m5_solver_llm:intern:a1a1a1a1a1a1", placed)

    def test_a_sealed_session_seals_its_table(self):
        studio = self.rows("trajectories/studio", "v2")
        self.assertEqual(studio["m5_solver_llm:intern:111111111111"]["split"], "sealed")
        self.assertEqual(studio["m5_solver_llm:intern:222222222222"]["split"], "sealed")
        train = (self.base("trajectories/studio", "v2") / "train.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("sha-shared", train)
        self.assertEqual(studio["m5_solver_llm:intern:333333333333"]["split"], "sealed")  # its table's sealed final was run from the screen

    def test_no_cell_reaches_a_placed_row(self):
        from dclab_core.store import SolverTrajectory

        self.assertNotIn("\"examples\"", json.dumps(self.placed()))
        old = SolverTrajectory.model_validate(self.rows("trajectories/studio", "v2")["m5_solver_llm:intern:eeeeeeeeeeee"])
        self.assertIsNone(old.steps[0].result)  # its summary quoted the cells; nothing is kept in its place
        self.assertTrue(old.submission["results_cut"])

    def test_a_verified_selfplay_solve_is_one_trajectory_of_its_decisions(self):
        from dclab_core.store import SolverTrajectory

        rows = self.rows("trajectories/selfplay", "v1")
        self.assertEqual(list(rows), ["m5_solver_llm:selfplay:binary:0"])  # seed 1 did not pass its checks
        t = SolverTrajectory.model_validate(rows["m5_solver_llm:selfplay:binary:0"])
        self.assertEqual([s.tool for s in t.steps], ["decide:target.column", "decide:feature.leakage_suspect"])
        self.assertEqual(t.steps[1].result["label"]["source"], "benchmark")
        self.assertEqual((t.steps[1].thought_source, t.score), ("from_record", 0.74))
        self.assertEqual(self.report["kept_out"].get("m5: a self-play solve that did not pass its checks"), 1)

    def test_a_decision_the_truth_contradicts_is_a_pair_even_from_a_failed_solve(self):
        from dclab_core.store import PreferencePair

        pairs = self.rows("preferences/selfplay", "v1")
        self.assertEqual(list(pairs), ["m5_solver_llm:selfplay:binary:1:pair:1"])
        pair = PreferencePair.model_validate(pairs["m5_solver_llm:selfplay:binary:1:pair:1"])
        self.assertEqual((pair.chosen.content, pair.rejected.content, pair.decided_by), ("false", "true", "benchmark"))

    def test_a_model_is_local_only_on_its_names_evidence(self):
        from dclab_rnd.store.sets.llm import answerer_of

        self.assertEqual(answerer_of({"mode": "llm", "model": "qwen2.5-coder:1.5b"}), "local:qwen2.5-coder:1.5b")
        self.assertEqual(answerer_of({"mode": "llm", "model": "deepseek-chat"}), "frontier:deepseek-chat")
        self.assertEqual(answerer_of({"mode": "llm", "model": "gpt-6-luna"}), "openai:gpt-6-luna")
        self.assertEqual(answerer_of({"mode": "llm", "model": None}), "frontier:unknown")
        for hosted in ("groq:llama-3.3-70b", "google:gemini-2.5-pro", "meta-llama/llama-3.1-8b-instruct:free", "Qwen/Qwen2.5-72B-Instruct:novita"):
            self.assertTrue(answerer_of({"mode": "llm", "model": hosted}).startswith("frontier:"), hosted)
        self.assertEqual(answerer_of({"mode": "llm", "model": "ollama/llama3.1:8b"}), "local:ollama/llama3.1:8b")
        self.assertEqual(answerer_of({"mode": "standard", "model": None}), "rules")

    def test_a_run_a_hosted_model_helped_is_not_clean_and_its_words_are_redacted(self):
        silver = self.rows("trajectories/silver_frontier", "v1")
        mixed = silver["m5_solver_llm:intern:444444444444"]
        self.assertTrue(mixed["problem"]["answerer"].startswith("mixed:rules+claude:"))
        self.assertNotIn("maria.lopez@acme.com", json.dumps(self.placed()))

    def test_each_subagent_is_its_own_trajectory_in_its_parents_split(self):
        dev = self.rows("trajectories/development", "v2")
        ids = [i for i in dev if SESSION_UUID in i]
        self.assertEqual(len(ids), 3)
        self.assertEqual(len({dev[i]["split"] for i in ids}), 1)

    def test_what_an_earlier_version_sealed_stays_sealed(self):
        from dclab_core.store import Provenance, SolverTrajectory, TrajectoryStep
        from dclab_rnd.store.sets.llm import keep_sealed

        old = self.base("trajectories/studio", "v1")
        old.mkdir(parents=True, exist_ok=True)
        prov = {"entry": "conversations:dev:old:v1", "sha256": "0" * 64, "source": "intern/x.json", "workspace": "dev", "consent": "train",
                "at": AT}
        (old / "sealed.jsonl").write_text(json.dumps({"id": "x", "provenance": prov}) + "\n", encoding="utf-8")
        t = SolverTrajectory(id="m5_solver_llm:intern:x", model_type="m5_solver_llm", split="train", provenance=Provenance(**prov),
                             problem_kind="binary", problem={}, steps=[TrajectoryStep(n=0, no_thought="-", result=None)])
        self.assertEqual(keep_sealed(self.layout, "m5_solver_llm", "trajectories/studio", "v2", [t])[0].split, "sealed")


@unittest.skipIf(MISSING, MISSING)
class SelfplayKeptTest(unittest.TestCase):
    def test_a_solves_ledger_is_kept_as_decisions_and_a_test_folder_never_reads_the_real_one(self):
        from dclab_rnd.store.keepers import standard

        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        runs, play = folder / "agent_runs", folder / "selfplay"
        (runs / "experience").mkdir(parents=True)
        day = play / "ledgers" / "binary" / "0" / "decisions" / "2026" / "10"
        day.mkdir(parents=True)
        (day / "21.jsonl").write_text("{}\n", encoding="utf-8")
        (play / "done.jsonl").write_text("{}\n", encoding="utf-8")
        items = {(i.ws.name, i.rel, i.kind) for i in standard.inventory({"agent_runs": runs, "selfplay": play})}
        self.assertIn(("selfplay__ledgers__binary__0", "ledger/decisions/2026/10/21.jsonl", "decisions"), items)
        self.assertIn(("selfplay", "done.jsonl", "other"), items)
        self.assertFalse(any(n.startswith("selfplay") for n, _r, _k in
                             {(i.ws.name, i.rel, i.kind) for i in standard.inventory({"agent_runs": runs})}))


@unittest.skipIf(MISSING, MISSING)
class SolverRunsTest(unittest.TestCase):
    def test_the_runner_names_the_outcome_as_the_builder_reads_it(self):
        from dclab_rnd.store.sets import llm
        from dclab_rnd.twin import solver_runs

        self.assertEqual(llm.OUTCOME_NAMED, solver_runs.OUTCOME_NAMED)
        self.assertIn(llm.OUTCOME_NAMED, solver_runs.TASK.format(target="Churn"))


@unittest.skipIf(MISSING, MISSING)
class InternKeepsWholeResultsTest(unittest.TestCase):
    def test_a_step_keeps_its_whole_result_without_cells(self):
        from dclab_rnd.intern import loop

        result = {"columns": [{"name": "a", "examples": ["1", "2"]}], "score": float("nan"), "note": "x" * 2000}
        step = loop.step_record(1, "describe_data", {"project_id": "p"}, result, 0.1)
        self.assertEqual(step["result"], {"columns": [{"name": "a"}], "score": None, "note": "x" * 2000})  # whole, no cell, no NaN
        self.assertTrue(step["summary"].endswith("…"))
        self.assertNotIn("examples", step["summary"])
        json.dumps(step, allow_nan=False)  # readable through the API


if __name__ == "__main__":
    unittest.main()
