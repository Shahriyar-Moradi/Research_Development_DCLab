"""26.1: the training lake — its records, its rules, and the builders that fill it from every source."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from dclab_core import lake as L


def prov(group="dataset:a", split="train", origin="construction"):
    return L.Provenance(source="test", origin=origin, consent="ours", group=group, split=split)


def sample(i, **kw):
    return L.DecisionSample(sample_id=f"s{i}", decision_point="column.is_identifier", subject_kind="column", subject_ref="c",
                            features={"dtype": "integer"}, state_text="{}", label=True, provenance=prov(**kw))


def example(i, **kw):
    return L.TrainingExample(example_id=f"e{i}", family="understand_problem", messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": f"y{i}"}],
                             verification=L.Verification(kind="none", passed=False), provenance=prov(**kw))


class LakeRecordTests(unittest.TestCase):
    def test_a_clean_set_has_no_problem(self):
        self.assertEqual(L.gold_problems([sample(1), sample(2, group="dataset:b", split="sealed")]), [])

    def test_the_rules_every_gold_set_passes(self):
        problems = L.gold_problems([sample(1), sample(2, split="sealed")])  # one table in two splits
        self.assertTrue(any("dataset:a" in p for p in problems))
        problems = L.gold_problems([example(1, origin="teacher_claude", split="train", group="g1")])  # Claude as a target
        self.assertTrue(any("teacher_claude" in p for p in problems))
        self.assertEqual(L.gold_problems([example(1, origin="teacher_claude", split="eval_only", group="g1")]), [])
        self.assertTrue(L.gold_problems([example(1, origin="research_record", split="eval_only", group="g1")]))  # eval_only is for teachers
        bad = L.DecisionSample(sample_id="s9", decision_point="p", subject_kind="column", subject_ref="c", features={"holdout_values": [1]},
                               state_text="{}", label=1, provenance=prov())
        self.assertTrue(any("holdout" in p for p in L.gold_problems([bad])))
        self.assertTrue(any("appears 2 times" in p for p in L.gold_problems([sample(1), sample(1)])))

    def test_the_split_is_the_groups(self):
        self.assertEqual({L.split_of("dataset:x") for _ in range(3)}, {L.split_of("dataset:x")})
        self.assertEqual(set(Counter(L.split_of(f"g{i}") for i in range(2000))), {"train", "dev", "sealed"})

    def test_the_contracts_are_registered_and_versioned(self):
        from dclab_core.contracts import CONTRACTS

        for name in ("decision_sample", "training_example", "gold_manifest"):
            self.assertIn(name, CONTRACTS)
        self.assertEqual(L.DecisionSample.read(sample(1).model_dump_json()).sample_id, "s1")


class LakeBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def decision_row(self, record, group, split, value, kind="column"):
        return {"record": record, "point": "column.is_identifier", "subject": {"kind": kind, "key": "id"}, "state": {"dtype": "integer", "uniqueness": "all", "column": group},
                "options": ["yes", "no"], "answers": [{"by": "rule", "value": True}], "label": {"value": value, "source": "benchmark"}, "group": group, "split": split}

    def build(self, homes, sft=None, dev=None):
        from dclab_rnd import lake as B

        with mock.patch.object(B, "GENERATED", self.tmp / "generated"), mock.patch.object(B, "GENERATED_IMAGES", self.tmp / "generated_images"), \
                mock.patch.object(B, "SFT_V3", sft or self.tmp / "none"), \
                mock.patch.object(B, "SFT_DEV", dev or self.tmp / "none"), mock.patch("dclab_rnd.lake.openml.CACHE", self.tmp / "openml"), \
                mock.patch("dclab_rnd.lake.selfplay.OUT", self.tmp / "selfplay"):
            return B.build(self.tmp / "lake", homes, None)

    def workspace(self, train=True, use="train", stale=False):
        from dclab_rnd import consent

        home = self.tmp / f"ws_{train}_{use}_{stale}"
        home.mkdir()
        if train:
            consent.set_workspace_consent(home, "train", True)
        twin = home / "datasets" / "twin" / "twin.teacher.technical"
        twin.mkdir(parents=True)
        (twin / "manifest.json").write_text(json.dumps({"use": use}))
        (twin / "rows.jsonl").write_text(json.dumps({"point": "column.is_identifier", "teacher": {"model": "claude", "value": True},
                                                      "group": "draft:d1", "split": "train", "sources": [["step", "s1"]]}) + "\n")
        dec = home / "datasets" / "decisions" / "workflow.next_question"
        dec.mkdir(parents=True)
        (dec / "manifest.json").write_text(json.dumps({"use": use}))
        (dec / "rows.jsonl").write_text(json.dumps({**self.decision_row("w1", "project:p1", "train", "target", "draft"), "point": "workflow.next_question"}) + "\n")
        if stale:
            (home / "datasets" / ".stale").write_text("now")
        return home

    def test_generated_sets_keep_their_sealed_partition_and_learn_nothing_of_it(self):
        gen = self.tmp / "generated" / "column.is_identifier"
        gen.mkdir(parents=True)
        rows = [self.decision_row("r1", "dataset:a", "train", True), self.decision_row("r2", "dataset:b", "dev", False),
                self.decision_row("r3", "dataset:c", "train", None)]
        same = {**self.decision_row("r5", "dataset:e", "train", True), "state": {"dtype": "integer", "uniqueness": "all", "column": "dataset:d"}}
        (gen / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + [same]))
        (gen / "sealed.jsonl").write_text(json.dumps(self.decision_row("r4", "dataset:d", "train", True)) + "\n")
        summary = self.build([])
        ident = next(s for s in summary["sets"] if s["name"] == "decisions/column.is_identifier")
        self.assertEqual(ident["by_split"], {"train": 1, "dev": 1, "sealed": 1})  # the unlabelled row and the sealed row's twin gone
        self.assertEqual(ident["dropped"]["no usable label"], 1)
        self.assertEqual(ident["dropped"]["train row identical to an evaluation row"], 1)  # decontaminated, whatever its group
        sealed = [json.loads(x) for x in (self.tmp / "lake" / "gold" / "decisions" / "column.is_identifier" / "v1" / "sealed.jsonl").read_text().splitlines()]
        self.assertTrue(all(x["provenance"]["from_sealed"] for x in sealed))

    def test_a_workspace_is_read_only_under_train_consent_and_claude_only_as_evaluation(self):
        off, improve, stale, on = self.workspace(train=False), self.workspace(use="improve"), self.workspace(stale=True), self.workspace()
        summary = self.build([off, improve, stale, on])
        self.assertIn("its train consent is off", summary["workspaces_skipped"])
        self.assertTrue(any("stale" in w for w in summary["workspaces_skipped"]))
        names = {s["name"]: s for s in summary["sets"]}
        self.assertEqual(names["eval/twin.teacher.technical"]["rows"], 1)  # only the train-consented workspace's
        self.assertEqual(names["eval/twin.teacher.technical"]["by_split"], {"eval_only": 1})
        self.assertEqual(names["eval/twin.teacher.technical"]["by_origin"], {"teacher_claude": 1})
        workflow = names["decisions/workflow.next_question"]
        self.assertEqual(workflow["rows"], 1)
        improve_drops = next(v for k, v in summary["dropped_by_source"].items() if "improve" in k and "workflow" in k)
        self.assertTrue(any("not train" in k for k in improve_drops))  # the improve-only set counted under its own source, not read
        row = json.loads((self.tmp / "lake" / "gold" / "decisions" / "workflow.next_question" / "v1" / "train.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["subject_kind"], "problem")  # a draft's decision, never mislabelled a column

    def test_the_build_history_is_evaluation_only_and_a_rebuild_leaves_nothing_withdrawn(self):
        dev = self.tmp / "dev"
        dev.mkdir()
        (dev / "train.chat.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}) + "\n")
        on = self.workspace()
        self.build([on], dev=dev)
        self.assertEqual(json.loads((self.tmp / "lake" / "gold" / "solver" / "engineering" / "v1" / "manifest.json").read_text())["by_split"], {"eval_only": 1})
        from dclab_rnd import consent

        consent.set_workspace_consent(on, "train", False)  # withdrawn
        self.build([on], dev=dev)
        self.assertFalse((self.tmp / "lake" / "gold" / "eval").exists())  # the earlier build's Claude rows are gone
        self.assertFalse((self.tmp / "lake" / "gold" / "decisions" / "workflow.next_question").exists())

    def test_a_sealed_row_outside_the_sealed_split_is_refused(self):
        row = sample(1, split="train")
        bad = row.model_copy(update={"provenance": row.provenance.model_copy(update={"from_sealed": True})})
        self.assertTrue(any("sealed partition" in p for p in L.gold_problems([bad])))


class PublicAndSelfplayTests(unittest.TestCase):
    def test_an_openml_table_becomes_samples_with_its_licence_and_the_outcome_never_in_a_learnable_place(self):
        from dclab_rnd.lake import openml as O

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "datasets").mkdir()
            state = lambda name, dtype: {"column": name, "name_tokens": [name], "dtype": dtype, "uniqueness": "all", "nulls": "none", "cardinality": "high"}  # noqa: E731
            rec = {"did": 7, "name": "t", "licence": "CC0", "target": "zz_class", "row_id": "id", "ignore": "", "rows": 500, "columns": [
                {"name": "id", "openml_type": "numeric", "is_row_identifier": True, "is_target": False, "is_ignore": False, "state": state("id", "integer")},
                {"name": "age", "openml_type": "numeric", "is_row_identifier": False, "is_target": False, "is_ignore": False, "state": state("age", "integer")},
                {"name": "zz_class", "openml_type": "nominal", "is_row_identifier": False, "is_target": True, "is_ignore": False, "distinct_nominal": 2,
                 "state": state("zz_class", "string")}]}
            (cache / "datasets" / "7.json").write_text(json.dumps(rec))
            (cache / "datasets" / "8.json").write_text(json.dumps({"did": 8, "skipped": "licence 'unknown' not known to allow reuse"}))
            with mock.patch.object(O, "CACHE", cache):
                got = O.samples(Counter())
        by = {(x.decision_point, x.subject_ref): x for x in got}
        self.assertIs(by[("column.is_identifier", "id")].label, True)
        self.assertEqual(by[("column.semantic_role", "id")].label, "identifier")
        self.assertEqual(by[("column.semantic_role", "age")].label, "numeric")
        self.assertEqual(by[("task.framing", "openml:7")].label, "binary")
        target = by[("target.column", "openml:7")]
        self.assertEqual(target.options, sorted(target.options))  # never last by construction
        self.assertTrue(all(x.provenance.licence == "CC0" and x.provenance.origin == "public_catalog" for x in got))
        self.assertEqual(len({x.provenance.group for x in got}), 1)  # one table, one split
        self.assertEqual(L.gold_problems(got), [])

    def test_a_verified_solve_teaches_and_its_masked_twin_names_nothing(self):
        from dclab_rnd import kinds
        from dclab_rnd.lake import selfplay as P

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(P, "OUT", Path(tmp)), mock.patch.dict("os.environ", {"DCLAB_DECISION_SHADOW": "0"}):
            done = P.solve(["binary"], seeds=1)
            got = P.examples()
        self.assertEqual((done["problems"], done["rejected"]), (1, 0))
        self.assertTrue(all("answer_format" in json.loads(e.messages[1]["content"]) for e in got))  # a model asked cold sees the contract too
        self.assertEqual({e.family for e in got}, {"problem_type", "leakage_judgment", "split_design", "algorithm_choice", "read_results", "solution_design"})
        self.assertEqual(L.gold_problems(got), [])
        case = next(k for k in kinds.all_kinds() if k.key == "binary").case(0)
        names = [c for c in case.table.columns if len(c) > 3]
        masked = [e for e in got if e.family == "leakage_judgment" and "col_01" in e.messages[1]["content"]]
        self.assertTrue(masked)
        for e in masked:
            text = e.messages[1]["content"] + e.messages[2]["content"]
            self.assertFalse([n for n in names if n in text], "a column name in the masked variant")
        leak = json.loads([e for e in got if e.family == "leakage_judgment" and "col_01" not in e.messages[1]["content"]][0].messages[2]["content"])
        self.assertIn(case.facts["leak"], [f["column"] for f in leak["forbidden"]])  # verified: the planted leak is kept out


    def test_an_image_problem_is_solved_in_self_play_and_its_pixels_are_one_state(self):
        from dclab_rnd.lake import selfplay as P

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(P, "OUT", Path(tmp)), mock.patch.dict("os.environ", {"DCLAB_DECISION_SHADOW": "0"}):
            done = P.solve(["embedded_images"], seeds=1)
            got = P.examples()
            ledgers = list((Path(tmp) / "ledgers" / "embedded_images" / "0").rglob("*.jsonl"))
            written = [json.loads(x) for p in ledgers for x in p.read_text().splitlines() if x.strip()]
        self.assertEqual((done["problems"], done["rejected"]), (1, 0))
        self.assertTrue(ledgers)  # the image solve's decisions are recorded in its own ledger
        image = [r for r in written if "subject" in r and str(r.get("point", "")).startswith("image.")]  # 26.8: the image points, from the engine
        self.assertEqual({r["point"] for r in image}, {"image.near_copy", "image.label_suspect", "image.flip_safe"})
        self.assertTrue(all(r["subject"]["key"].split(":", 1)[1].startswith(("row:", "rows:", "table")) for r in image))
        flip = next(r for r in image if r["point"] == "image.flip_safe")
        self.assertEqual(flip["value_used"], True)  # circles and squares: a mirror keeps the label
        self.assertEqual({e.provenance.group for e in got}, {"selfplay:embedded_images:0"})
        kind = json.loads(next(e for e in got if e.family == "problem_type").messages[2]["content"])
        self.assertEqual(kind["kind"], "embedded")  # an engine kind, not the variant's name
        given = json.loads(next(e for e in got if e.family == "split_design").messages[1]["content"])
        pixels = [c for c in given["columns"] if c.get("kind") == "pixels"]
        self.assertEqual((len(pixels), pixels[0]["count"], pixels[0]["image"]), (1, 1024, "32x32"))
        self.assertLess(len(given["columns"]), 10)
        split = json.loads(next(e for e in got if e.family == "split_design").messages[2]["content"])
        self.assertEqual(split["group_column"], "scene")  # whole scenes on one side


class ExportTests(unittest.TestCase):
    def test_the_training_files_hold_no_sealed_or_evaluation_only_row(self):
        from dclab_rnd.lake.train import export

        with tempfile.TemporaryDirectory() as tmp:
            lake = Path(tmp) / "lake"
            gold = lake / "gold" / "decisions" / "column.is_identifier" / "v1"
            gold.mkdir(parents=True)
            rows = {"train": [sample(1, group="g1"), sample(2, group="g2")], "dev": [sample(3, group="g3", split="dev")],
                    "sealed": [sample(4, group="g4", split="sealed")]}
            for split, items in rows.items():
                (gold / f"{split}.jsonl").write_text("".join(i.model_dump_json() + "\n" for i in items))
            solver = lake / "gold" / "solver" / "understand_problem" / "v1"
            solver.mkdir(parents=True)
            (solver / "eval_only.jsonl").write_text(example(9, origin="teacher_claude", split="eval_only", group="g9").model_dump_json() + "\n")
            (solver / "train.jsonl").write_text(example(8, origin="verified_synthetic", group="g8").model_dump_json() + "\n")
            m = export("decisions", lake, Path(tmp) / "out_d", {"column.is_identifier": 2.0})
            self.assertEqual((m["train"], m["val"]), (4, 1))  # weight 2 doubles train only
            self.assertEqual(m["refused"]["sealed (never exported)"], 1)
            text = (Path(tmp) / "out_d" / "train.chat.jsonl").read_text()
            self.assertNotIn("s4", text)
            first = json.loads(text.splitlines()[0])["messages"]
            self.assertEqual([x["role"] for x in first], ["system", "user", "assistant"])
            self.assertTrue((Path(tmp) / "out_d" / "valid.jsonl").exists())  # the names mlx_lm reads
            m = export("solver", lake, Path(tmp) / "out_s")
            self.assertEqual((m["train"], m["refused"]["eval_only (never exported)"]), (1, 1))  # Claude's answer never a target
            self.assertNotIn("y9", (Path(tmp) / "out_s" / "train.chat.jsonl").read_text())


class PositiveClassTests(unittest.TestCase):
    def test_the_flagged_class_is_the_event_not_the_alphabetical_last(self):
        """Live (2026-10-16): fake_return against genuine_or_no_return made the genuine returns the positive class."""
        from dclab_rnd.studio.data import _positive_label

        self.assertEqual(_positive_label(["fake_return", "genuine_or_no_return"]), "fake_return")
        self.assertEqual(_positive_label(["a", "b"], {"a": 900, "b": 100}), "b")  # the rarer, when the names say nothing
        self.assertEqual(_positive_label(["no", "yes"]), "yes")
        self.assertEqual(_positive_label([0, 1]), 1)


class SolverBenchmarkTests(unittest.TestCase):
    def test_answers_are_scored_against_the_planted_truth_never_against_dclabs_answer(self):
        from dclab_rnd.lake import benchmark as B

        row = {"family": "leakage_judgment", "provenance": {"group": "selfplay:binary:0"},
               "messages": [{"role": "system", "content": ""}, {"role": "user", "content": "{}"}, {"role": "assistant", "content": "{}"}]}
        caught = B.score(row, {"forbidden": [{"column": "final_refund_amount"}], "identifiers": ["subscriber_id"]})
        self.assertEqual((caught["right"], caught["extra"]), (True, 0))
        missed = B.score(row, {"forbidden": [], "identifiers": ["subscriber_id"]})
        self.assertEqual((missed["right"], missed["leak_caught"]), (False, False))
        self.assertEqual(B.score(row, None), {"family": "leakage_judgment", "answered": False, "right": False})
        split = {**row, "family": "split_design"}
        self.assertTrue(B.score(split, {"time_column": "signup_time"})["right"])
        self.assertFalse(B.score({**row, "family": "problem_type"}, {"task": "regression"})["right"])
        with self.assertRaises(ValueError):
            B.run("rules", "train", store=False)  # never the training split
        self.assertEqual(B.wilson(0, 0), [None, None])


class LearnedAutomlTests(unittest.TestCase):
    def test_a_meta_model_that_reads_the_table_beats_a_fixed_family_where_the_winner_depends_on_it(self):
        from dclab_rnd.lake import automl as A

        rows = []
        for kind in ("k1", "k2", "k3"):
            for i in range(12):
                small = i % 2 == 0  # small tables: the linear model wins; large ones: boosting
                screen = [{"model": "logistic_regression", "mean": 0.8 if small else 0.6}, {"model": "xgboost", "mean": 0.65 if small else 0.82},
                          {"model": "random_forest", "mean": 0.7}]
                rows.append({"kind": kind, "task": "binary", "rows": 500 if small else 200000, "columns": 10, "usable_features": 9,
                             "dtypes": {"float": 10}, "cardinality": {"high": 10}, "nulls": {"none": 10}, "target_summary": {"positive_rate_train": 0.2},
                             "screen": screen})
        out = A.evaluate(rows)
        self.assertEqual(out["problems"], 36)
        self.assertEqual(out["pickers"]["meta_model"]["mean_regret"], 0.0)  # learned which family wins, on kinds it never saw
        self.assertGreater(out["pickers"]["always_xgboost"]["mean_regret"], 0.3)
        self.assertEqual(A.regret(rows[0], "not_in_screen"), 1.0)  # a family the screen did not try counts as the worst


class SelfplayLedgerTests(unittest.TestCase):
    def test_every_decision_of_a_solve_is_recorded_in_its_own_ledger(self):
        from dclab_rnd import ledger
        from dclab_rnd.lake import selfplay

        tmp = Path(tempfile.mkdtemp())

        class Engine:
            @staticmethod
            def engine_run(case, key):
                ledger.decide("split.strategy", ("project", "p1"), answers=[ledger.answer("rule", "rule:split", "time")], value_used="time",
                              state={"has_time": True}, options=["time", "random"])
                return "ran"

        before = ledger._INSTALLED.get("ledger")
        with mock.patch.object(selfplay, "OUT", tmp):
            self.assertEqual(selfplay._run_recorded(Engine, None, "binary", "selfplay:binary:7"), "ran")
        self.assertIs(ledger._INSTALLED.get("ledger"), before)  # the solve's ledger is put back afterwards
        book = ledger.FileLedger(tmp / "ledgers" / "binary" / "7", "selfplay")
        self.assertEqual([r.point for r in book.records()], ["split.strategy"])


class SelfplayForgetTests(unittest.TestCase):
    def test_forgetting_a_kind_takes_out_only_its_solves(self):
        from dclab_rnd.lake import selfplay

        tmp = Path(tempfile.mkdtemp())
        (tmp / "done.jsonl").write_text(json.dumps({"group": "selfplay:embedded:0"}) + "\n" + json.dumps({"group": "selfplay:binary:0"}) + "\n")
        (tmp / "examples.jsonl").write_text(json.dumps({"provenance": {"group": "selfplay:embedded:0"}}) + "\n")
        (tmp / "ledgers" / "embedded" / "0").mkdir(parents=True)
        with mock.patch.object(selfplay, "OUT", tmp):
            got = selfplay.forget(["embedded"])
            with self.assertRaises(ValueError):
                selfplay.forget([])
        self.assertEqual((got["done.jsonl"], got["examples.jsonl"]), (1, 1))
        self.assertEqual([json.loads(x)["group"] for x in (tmp / "done.jsonl").read_text().splitlines()], ["selfplay:binary:0"])
        self.assertFalse((tmp / "ledgers" / "embedded").exists())


class SessionMappingTests(unittest.TestCase):
    """27.2: a workspace's session sets become training examples, marked by who made their moves."""

    def item(self, row, builder):
        return {"row": {"group": "project:p1", "split": "train", **row}, "from_sealed": False, "source": f"workspace:w/{builder}/x"}

    def test_each_set_is_mapped_and_claudes_moves_are_evaluation_only(self):
        from dclab_rnd import lake as B

        run = B.Run()
        convo = [{"role": "user", "content": "Which customers leave?"}, {"role": "assistant", "content": "What do you predict?"}]
        other = {
            "conversations/home": [self.item({"messages": convo, "models": {}}, "conversations"),
                                   self.item({"messages": convo + [{"role": "assistant", "content": "x"}], "models": {"home_agent": "claude-sonnet-5-5"}}, "conversations"),
                                   self.item({"messages": convo, "models": {"home_agent": "qwen2.5"}}, "conversations")],
            "trajectories/agents": [self.item({"agent": "intern", "models": {}, "steps": [
                {"state": {"stage": "data"}, "tool": "run_stage", "verdict": "allowed"},
                {"state": {"stage": "final"}, "tool": "read_holdout", "verdict": "refused"}]}, "trajectories")],
            "workflows/projects": [self.item({"problem_card": {"family": "classification"}, "data_card": {}, "solution": {"target": "churn"},
                                              "workflow": {"source": "rules"}, "pipeline": [{"step": "data", "run": "r1"}]}, "workflows")],
        }
        got = B.session_examples(other, run)
        by = {(e.family, e.provenance.origin, e.provenance.split) for e in got}
        self.assertIn(("understand_problem", "rule_prod", "train"), by)
        self.assertIn(("understand_problem", "teacher_claude", "eval_only"), by)
        self.assertIn(("agent_step", "rule_prod", "train"), by)
        self.assertIn(("solution_design", "rule_prod", "train"), by)
        self.assertEqual(sum(1 for e in got if e.family == "agent_step"), 1)  # the refused move is not taught
        self.assertEqual(run.dropped["workspace:w/conversations/x"]["written by a model whose terms are not known"], 1)
        from dclab_core import lake as L

        self.assertEqual(L.gold_problems(got), [])


if __name__ == "__main__":
    unittest.main()
