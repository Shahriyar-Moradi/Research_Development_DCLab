"""Package 29.2: homework for table-level choices (benchgen v2): the outcome's kind, the split, a list of known cases, each
column's cleaning step; planted truths with traps; the rules' answers beside them; sets with a sealed partition by table."""

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
except ImportError as error:  # the agent environment has no numpy: only that skips (a broken module of the package fails)
    MISSING = str(error)
else:
    MISSING = ""
if not MISSING:
    from dclab_core import benchgen2 as g
    from dclab_core import ledger
    from dclab_rnd import decisions_gen as G


@unittest.skipIf(MISSING, MISSING)
class GeneratorTests(unittest.TestCase):
    def test_the_same_seed_draws_the_same_table_and_every_kind_is_drawn(self):
        a, b = g.task_table(7), g.task_table(7)
        self.assertTrue(a[0].equals(b[0]))
        self.assertEqual({g.task_table(s)[2]["kind"] for s in range(len(g.TASK_KINDS))}, set(g.TASK_KINDS))
        self.assertEqual({g.split_table(s)[1]["split"] for s in range(8)}, set(g.SPLITS))

    def test_the_traps_are_planted_as_their_truth_says(self):
        counts = next(g.task_table(s) for s in range(40) if g.task_table(s)[2]["kind"] == "regression_small_counts")
        frame, target, facts = counts
        self.assertEqual(facts["task"], "regression")
        self.assertLessEqual(frame[target].nunique(), 20)  # few whole numbers, still a number to predict
        many = next(g.task_table(s) for s in range(40) if g.task_table(s)[2]["kind"] == "multiclass_many_codes")
        self.assertGreater(many[0][many[1]].nunique(), 20)  # many codes, still classes
        dirty = [g.dirty_table(s) for s in range(30)]
        country = [t[c] for f, t in dirty for c in f.columns if set(f[c].astype(str)) <= {"NA", "ZA", "BW", "ZM", "KE", "NG"}]
        self.assertTrue(country and all(step == "none" for step in country))  # "NA" is Namibia here, whatever the column's name
        cases = [g.cases_table(s) for s in range(12)]
        self.assertEqual({f["variant"] for _, _, f in cases if f["cases"]}, {"cases", "cases_no_status", "cases_plain_file"})


@unittest.skipIf(MISSING, MISSING)
class DecisionTests(unittest.TestCase):
    def test_every_record_is_valid_holds_no_value_and_the_rules_are_right_and_wrong(self):
        records, labels = [], []
        for seed in range(40):
            r, l = G.decide_v2(seed)
            records += r
            labels += l
        self.assertEqual({r.point for r in records}, {"task.framing", "split.strategy", "data.cases_table", "data.cleaning_step"})
        for r in records:
            self.assertIn(r.point, ledger.POINTS)
            self.assertEqual(ledger.check_state(r.state), [], (r.point, r.state))
        self.assertEqual(len({r.id for r in records}), len(records))
        truth = {x.record: x.value for x in labels}
        for point in {r.point for r in records}:
            mine = [r for r in records if r.point == point]
            right = sum(r.value_used == truth[r.id] for r in mine)
            self.assertGreater(right, 0, point)
            self.assertLess(right, len(mine), point)  # the traps catch the rule somewhere: there is something to learn
        small = [r for r in records if r.point == "task.framing" and truth[r.id] == "regression" and r.state.get("classes", 99) <= 20]
        self.assertTrue(small and all(r.value_used == "multiclass" for r in small))  # the rule reads few whole numbers as classes

    def test_no_name_says_the_truth_and_every_table_has_a_hidden_twin_in_its_group(self):  # the review of 29.2
        from collections import defaultdict

        records, labels = [], []
        for seed in range(40):
            r, l = G.decide_v2(seed)
            records += r
            labels += l
        truth = {x.record: x.value for x in labels}
        for point in ("task.framing", "data.cleaning_step"):
            by_name = defaultdict(set)
            for r in records:
                if r.point == point and not str(r.state.get("column", "")).startswith("f"):
                    by_name[r.state.get("column")].add(str(truth[r.id]))
            self.assertTrue(any(len(v) > 1 for v in by_name.values()), point)  # one name, several truths
        tables = Counter(r.dataset for r in records if r.point == "task.framing")
        self.assertEqual(set(tables.values()), {2})  # a table and its hidden twin, one group
        hidden = [r for r in records if r.point == "task.framing" and str(r.state.get("column", "")).startswith("f")]
        self.assertEqual(len(hidden), 40)

    def test_the_states_are_production_s_and_the_split_rule_is_the_products(self):
        from dclab_rnd.kinds.base import default_split
        from dclab_rnd.ledger import states
        from dclab_rnd.studio.data import profile_table
        from dclab_rnd.studio.solution import propose

        frame, target, _ = g.task_table(5)
        r, _ = G.decide_v2(5)
        named = next(x for x in r if x.point == "task.framing" and x.state.get("column") == target)
        self.assertEqual(named.state, states.task_state(profile_table(frame), target))
        frame, _ = g.split_table(7)
        profile = profile_table(frame)
        p = propose(frame, profile, "outcome")
        rule = default_split({"time_column": (p.get("time_candidates") or [None])[0], "group_column": (p.get("group_candidates") or [None])[0]})
        r, _ = G.decide_v2(7)
        split = next(x for x in r if x.point == "split.strategy" and x.subject.key == "split-s7")
        self.assertEqual(split.value_used, rule)

    def test_a_rare_binary_table_always_holds_positives(self):
        for seed in range(0, 400, len(g.TASK_KINDS)):
            frame, target, facts = g.task_table(seed + g.TASK_KINDS.index("binary_rare"))
            self.assertEqual(facts["kind"], "binary_rare")
            self.assertGreaterEqual(int((frame[target] == "yes").sum()), 3)

    def test_the_sets_are_built_with_a_sealed_partition_by_table_and_rebuild_the_same(self):
        one, two = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        a = G.build_v2(seeds=30, out=one)
        b = G.build_v2(seeds=30, out=two)
        self.assertEqual(set(a["sets"]), {"task.framing", "split.strategy", "data.cases_table", "data.cleaning_step"})
        for point, s in a["sets"].items():
            self.assertEqual(s["sealed_fingerprint"], b["sets"][point]["sealed_fingerprint"])
            rows = [json.loads(x) for x in (one / "decisions" / point / "rows.jsonl").read_text().splitlines()]
            sealed = [json.loads(x) for x in (one / "decisions" / point / "sealed.jsonl").read_text().splitlines()]
            self.assertFalse({r["group"] for r in rows} & {r["group"] for r in sealed})
            self.assertTrue(all(r["label"]["source"] == "benchmark" for r in rows + sealed))
        self.assertGreater(Counter(json.loads(x)["split"] for x in (one / "decisions" / "data.cleaning_step" / "rows.jsonl").read_text().splitlines())["dev"], 0)


if __name__ == "__main__":
    unittest.main()
