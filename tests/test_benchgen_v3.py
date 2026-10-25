"""benchgen v3 (S7, 2026-10-25): no layout names a truth. The two shortcuts the reviews found in v2 are tested for
directly: a column count that says whether an entity is there, and unique plain digits that always meant an identifier."""

import sys
import unittest
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import benchgen3 as g  # noqa: E402

SEEDS = range(300)


def best_rule_accuracy(xs, ys) -> float:
    """The best a lookup from one layout number to the truth can do (the shortcut's ceiling)."""
    by = {}
    for x, y in zip(xs, ys):
        by.setdefault(x, Counter())[y] += 1
    return sum(c.most_common(1)[0][1] for c in by.values()) / len(ys)


class LayoutTest(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")

    def test_a_split_tables_width_never_says_its_truths(self):
        widths, entity, time = [], [], []
        for seed in SEEDS:
            frame, facts = g.split_table(seed)
            widths.append(frame.shape[1])
            entity.append(facts["has_entity"])
            time.append(facts["has_row_time"])
        for truth in (entity, time):
            base = max(Counter(truth).values()) / len(truth)
            self.assertLess(best_rule_accuracy(widths, truth) - base, 0.12)  # v2: a width named the entity on every table
        self.assertGreaterEqual(len(set(widths)), 6)

    def test_unique_plain_digits_are_numbers_on_some_tables_and_identifiers_on_others(self):
        steps = Counter()
        for seed in SEEDS:
            frame, truth = g.dirty_table(seed)
            for name, step in truth.items():
                values = frame[name].astype(str)
                if values.str.fullmatch(r"\d+(\.\d+)?").all() and values.nunique() > 0.9 * len(values):
                    steps[step] += 1
        self.assertGreater(steps["numbers"], 20)  # v2: never; every such column was an identifier
        self.assertGreater(steps["none"], 20)

    def test_a_craft_tables_layout_never_says_its_structure(self):
        widths, structures, tasks = [], [], Counter()
        for seed in SEEDS:
            frame, target, facts = g.craft_table(seed, rows=300)
            self.assertIn(target, frame.columns)
            widths.append(frame.shape[1])
            structures.append(facts["structure"])
            tasks[(facts["structure"], facts["task"])] += 1
        base = max(Counter(structures).values()) / len(structures)
        self.assertLess(best_rule_accuracy(widths, structures) - base, 0.15)
        self.assertEqual(set(structures), set(g.STRUCTURES))
        self.assertGreaterEqual(len(tasks), 20)  # structures and tasks drawn apart

    def test_no_field_of_the_table_at_a_glance_says_the_structure(self):  # the review of S7: a big category or a date named it
        from dclab_rnd.ledger.states import table_state
        from dclab_rnd.studio.data import profile_table

        fields = {"high_card_columns": [], "date_columns": [], "missing_columns": [], "category_columns": [], "numeric_columns": []}
        structures = []
        for seed in range(160):
            frame, target, facts = g.craft_table(seed, rows=300)
            state = table_state(profile_table(frame), target, facts["task"])
            for name in fields:
                fields[name].append(min(int(state.get(name, 0)), 4))
            structures.append(facts["structure"])
        for name, values in fields.items():
            for structure in ("high_card", "seasonal", "missing_informative"):
                truth = [s == structure for s in structures]
                base = max(Counter(truth).values()) / len(truth)
                self.assertLess(best_rule_accuracy(values, truth) - base, 0.12, f"{name} names {structure}")

    def test_every_structure_plants_what_it_says(self):
        for structure in g.STRUCTURES:
            frame, target, facts = g.craft_table(7, rows=800, structure=structure, task="binary")
            self.assertEqual(facts["structure"], structure)
            y = frame[target]
            self.assertTrue(set(np.unique(y)) <= {0, 1})
            if structure == "high_card":
                self.assertTrue(any(frame[c].dtype == object and frame[c].nunique() > 100 for c in frame.columns))
            if structure == "missing_informative":
                missing = frame.drop(columns=[target]).isna()
                rates = [float(y[missing[c]].mean()) for c in missing.columns if missing[c].sum() > 50]
                self.assertTrue(any(abs(r - float(y.mean())) > 0.08 for r in rates))  # who leaves it empty differs
        frame, target, _ = g.craft_table(3, rows=500, structure="skewed", task="regression")
        values = frame[target].to_numpy()
        self.assertGreater(float(np.mean(values)), float(np.median(values)))  # a long right tail

    def test_the_same_seed_draws_the_same_table(self):
        a, _, fa = g.craft_table(11)
        b, _, fb = g.craft_table(11)
        self.assertTrue(a.equals(b) and fa == fb)


if __name__ == "__main__":
    unittest.main()
