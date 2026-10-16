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
        return {"record": record, "point": "column.is_identifier", "subject": {"kind": kind, "key": "id"}, "state": {"dtype": "integer", "uniqueness": "all", "g": group},
                "options": ["yes", "no"], "answers": [{"by": "rule", "value": True}], "label": {"value": value, "source": "benchmark"}, "group": group, "split": split}

    def build(self, homes, sft=None, dev=None):
        from dclab_rnd import lake as B

        with mock.patch.object(B, "GENERATED", self.tmp / "generated"), mock.patch.object(B, "SFT_V3", sft or self.tmp / "none"), \
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
        same = {**self.decision_row("r5", "dataset:e", "train", True), "state": {"dtype": "integer", "uniqueness": "all", "g": "dataset:d"}}
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
        self.assertEqual({e.family for e in got}, {"problem_type", "leakage_judgment", "split_design", "algorithm_choice", "read_results"})
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


if __name__ == "__main__":
    unittest.main()
