"""Package 30.10: the recipe registry. A problem is one kind plus modifiers; every part fills the same contract, has a
version and a certification level; a resolved recipe names the part and the reason behind each choice. The kinds (sixteen since 30.16)
become parts through an adapter; an engine run records its recipe, and the recipe's split and metric are the run's own."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import recipes as R  # noqa: E402

try:
    from dclab_rnd.kinds.binary import Binary
except ImportError:  # the agent environment
    Binary = object


class Stricter(Binary):
    """A binary kind whose rare-class cut-off moved: a change inside a method."""

    def default_metric(self, solution, positive_rate, pack=None):
        return "average_precision" if positive_rate is not None and positive_rate < 0.02 else "roc_auc"


def part(name, part_type="modifier", **fields):
    return R.RecipePart(name=name, part_type=part_type, version="1", level="experimental", **fields)


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.binary = part("binary", "kind", split="stratified", primary_metric="roc_auc", guard_metrics=["average_precision"],
                           baseline="logistic regression", model_families=["logistic", "lightgbm"], leakage_checks=["PIT-001"])

    def test_with_no_modifier_every_choice_is_the_kinds_and_names_it(self):
        r = R.resolve(self.binary, [])
        self.assertEqual((r.value("split"), r.value("primary_metric")), ("stratified", "roc_auc"))
        self.assertTrue(all(c.part == "binary@1" for c in r.choices))
        self.assertIn("shared rule", r.choice("calibration").reason)  # no part sets one: said so, never called the kind's default
        with self.assertRaises(KeyError):
            r.choice("colour")

    def test_splits_combine_and_the_most_specific_metric_wins(self):
        time = part("time_order", split="time", leakage_checks=["features only from before each row's cutoff"])
        groups = part("groups", split="group")
        rare = part("rare_positives", primary_metric="average_precision", guard_metrics=["recall_at_budget"])
        capacity = part("review_capacity", primary_metric="precision_at_k", threshold_rule="the K-th score of a period")
        r = R.resolve(self.binary, [rare, time, groups, capacity])
        self.assertEqual(r.value("split"), "time_group")
        self.assertIn("time_order@1", r.choice("split").part)
        self.assertIn("groups@1", r.choice("split").part)
        self.assertEqual((r.value("primary_metric"), r.choice("primary_metric").part), ("precision_at_k", "review_capacity@1"))
        self.assertEqual(r.value("threshold_rule"), "the K-th score of a period")
        self.assertEqual(r.value("leakage_checks"), ["PIT-001", "features only from before each row's cutoff"])  # lists add up, in order
        self.assertEqual(r.value("guard_metrics"), ["average_precision", "recall_at_budget"])

    def test_a_kinds_time_order_is_never_dropped_by_a_modifiers_split(self):
        forward = part("forecasting", "kind", split="time", primary_metric="mase")
        r = R.resolve(forward, [part("groups", split="group")])
        self.assertEqual((r.value("split"), r.choice("split").part.split(", ")[0]), ("time_group", "forecasting@1"))
        r = R.resolve(part("series", "kind", split="rolling_origin"), [part("groups", split="group")])
        self.assertIsNone(r.value("split"))  # a rolling origin and groups do not combine: asked, never one of them dropped
        self.assertEqual(r.ask, ["split"])

    def test_modifiers_of_equal_rank_that_disagree_are_asked(self):
        a, b = part("noisy_labels", calibration="isotonic"), part("text_columns", calibration="sigmoid")
        for order in ([a, b], [b, a]):
            r = R.resolve(self.binary, order)
            self.assertIsNone(r.value("calibration"))
            self.assertIn("calibration", r.ask)

    def test_the_person_wins_and_what_they_state_is_checked(self):
        r = R.resolve(self.binary, [part("time_order", split="time")], stated={"primary_metric": "f1", "split": "group",
                                                                               "model_families": ["logistic"]})
        self.assertEqual((r.value("primary_metric"), r.choice("primary_metric").part), ("f1", "person"))
        self.assertEqual((r.value("split"), r.value("model_families")), ("group", ["logistic"]))
        for stated in ({"split": "temporal"}, {"colour": "red"}, {"primary_metric": ["a", "b"]}, {"leakage_checks": "PIT-001"}):
            with self.assertRaises(ValueError, msg=stated):
                R.resolve(self.binary, [], stated=stated)
        with self.assertRaises(ValueError):
            R.resolve(self.binary, [part("groups"), part("groups")])  # one modifier, named once

    def test_long_names_and_many_modifiers_fit_the_contract(self):
        mods = [part(f"modifier_number_{i:02d}_with_a_long_name", report_items=[f"item {i}"]) for i in range(10)]
        r = R.resolve(part("k" * 60, "kind", report_items=["kind item"]), mods)
        self.assertEqual(len(r.value("report_items")), 11)

    def test_the_contract_refuses_a_modifier_that_is_a_kind_and_an_unknown_level(self):
        with self.assertRaises(ValueError):
            R.resolve(part("binary", "kind"), [part("regression", "kind")])
        with self.assertRaises(ValueError):
            R.RecipePart(name="x", part_type="kind", version="1", level="production")


class KindsAsPartsTests(unittest.TestCase):
    def test_every_kind_is_a_part_with_a_version_its_evidence_and_its_true_defaults(self):
        from dclab_rnd import kinds, recipes

        parts = {p.name: p for p in recipes.kind_parts()}
        self.assertEqual(set(parts), {k.key for k in kinds.all_kinds()})
        for kind in kinds.all_kinds():
            p = parts[kind.key]
            self.assertEqual(p.part_type, "kind")
            self.assertIn(p.level, ("experimental", "certified"), kind.key)  # from its certification record (level_of)
            self.assertEqual(p.primary_metric, kind.default_metric({"task": kind.base_task}, None), kind.key)
            self.assertNotIn(p.primary_metric, p.guard_metrics, kind.key)
            self.assertTrue(all(kind.export.get(n) is True for n in p.artifact_needs), kind.key)
            if kind.own_machine() is not None:
                self.assertIsNone(p.split, kind.key)  # its machine splits; the part does not pretend
        self.assertEqual((parts["regression"].split, parts["places"].split, parts["binary"].split), ("random", "spatial", "stratified"))
        self.assertEqual({p.name: p.version for p in recipes.kind_parts()}, {n: p.version for n, p in parts.items()})  # stable

    def test_the_level_comes_from_the_certification_record_and_holds_only_for_its_version(self):  # the owner, 2026-10-20
        import json
        import tempfile
        from pathlib import Path

        from dclab_rnd import kinds, recipes

        folder = Path(tempfile.mkdtemp())
        current = recipes.part_of(kinds.get("binary"), found={}).version
        (folder / "CRT-001_binary.json").write_text(json.dumps({"id": "CRT-001", "parts": {"binary": {"level": "certified"}}}))
        found = recipes.verdicts(folder)
        self.assertEqual(recipes.part_of(kinds.get("binary"), found=found).level, "certified")  # an old record kept no version: trusted, said so
        self.assertIn("not recorded", recipes.part_of(kinds.get("binary"), found=found).certification[0])
        (folder / "CRT-002_binary.json").write_text(json.dumps({"id": "CRT-002", "parts": {"binary": {"level": "certified"}}, "versions": {"binary": "0000000000"}}))
        changed = recipes.part_of(kinds.get("binary"), found=recipes.verdicts(folder))
        self.assertEqual(changed.level, "experimental")  # the kind changed since the record judged it
        self.assertIn("certify it again", changed.certification[0])
        (folder / "CRT-003_binary.json").write_text(json.dumps({"id": "CRT-003", "parts": {"binary": {"level": "certified"}}, "versions": {"binary": current}}))
        self.assertEqual(recipes.part_of(kinds.get("binary"), found=recipes.verdicts(folder)).level, "certified")
        (folder / "CRT-004_binary.json").write_text(json.dumps({"id": "CRT-004", "parts": {"binary": {"level": "product_default"}}, "versions": {"binary": current}}))
        self.assertEqual(recipes.part_of(kinds.get("binary"), found=recipes.verdicts(folder)).level, "experimental")  # never granted by a record
        self.assertEqual({p.name for p in recipes.modifier_parts()}, set(__import__("dclab_core.modifiers", fromlist=["PARTS"]).PARTS))

    def test_every_modifier_has_the_rule_that_keeps_it_in_a_run(self):  # a modifier without one would be dropped without a word
        from dclab_core import modifiers as M
        from dclab_rnd import recipes

        self.assertEqual(set(recipes.KEEP), set(M.PARTS))

    def test_a_kind_whose_behaviour_changes_gets_a_new_version(self):
        from dclab_rnd import kinds, recipes

        self.assertNotEqual(recipes.part_of(Stricter()).version, recipes.part_of(kinds.get("binary")).version)

    def test_a_recipe_that_cannot_resolve_is_recorded_never_raised(self):
        from dclab_rnd import kinds, recipes

        class Broken(type(kinds.get("binary"))):
            def split(self, solution):
                raise RuntimeError("no split")

        out = recipes.record_for(Broken(), {"task": "binary"}, 0.2)
        self.assertIn("RuntimeError", out["error"])


class EngineRecordTests(unittest.TestCase):
    """The recipe a run records is the run's own: its split and metric on every engine kind tried, and a machine kind's."""

    def check(self, key):
        from dclab_rnd import kinds
        from dclab_rnd.kinds import acceptance

        kind = kinds.get(key)
        done = acceptance.engine_run(kind.case(0), kind.key)
        recipe = done.evidence("data")["recipe"]
        self.assertNotIn("error", recipe, key)
        self.assertTrue(recipe["kind"].startswith(f"{key}@"), key)
        by = {c["field"]: c for c in recipe["choices"]}
        self.assertTrue(all(c["part"] for c in recipe["choices"]), key)
        return done, by

    def test_engine_kinds_record_the_split_and_metric_they_ran(self):
        from dclab_rnd.extensions import available

        fraud = ("rare_event",) if available("dclab_core.replicas.delivery_fraud") else ()  # its replica is not in the open edition
        for key in ("binary", "regression", "places", *fraud):
            done, by = self.check(key)
            self.assertEqual(by["primary_metric"]["value"], done.records["data"]["primary_metric"], key)
            self.assertEqual(by["split"]["value"], done.evidence("data")["split_strategy"], key)

    def test_a_kind_with_its_own_machine_records_its_recipe(self):
        done, by = self.check("segmentation")
        self.assertEqual((by["split"]["part"], by["primary_metric"]["part"]), ("machine", "machine"))
        self.assertIsNone(by["split"]["value"])
        self.assertTrue(by["primary_metric"]["value"])


if __name__ == "__main__":
    unittest.main()
