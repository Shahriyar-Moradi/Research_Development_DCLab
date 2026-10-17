"""Package 26.6b (data): invented tables from random priors, the same table from the same seed, every prior's shape
present, written in shards that read back whole, each table's split from its seed and its signal checked."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401
except ImportError as error:  # the agent environment has no pandas
    MISSING = str(error)
else:
    MISSING = ""
    from dclab_core import priors
    from dclab_core.lake import split_of
    from dclab_rnd.lake import priors as corpus


@unittest.skipIf(MISSING, MISSING)
class PriorsTest(unittest.TestCase):
    def test_a_seed_always_makes_the_same_table_and_each_prior_its_own(self):
        a, b = priors.sample(11), priors.sample(11)
        self.assertTrue(np.array_equal(a.X, b.X, equal_nan=True) and np.array_equal(a.y, b.y))
        for prior in priors.PRIORS:
            t = priors.sample(3, prior)
            self.assertEqual(t.meta["prior"], prior)
            self.assertEqual(t.X.shape, (t.meta["rows"], t.meta["features"]))
            self.assertEqual(len(t.categorical), t.meta["features"])
            self.assertTrue(2 <= t.meta["classes"] <= 5 and t.y.max() < t.meta["classes"])
            self.assertEqual(t.X.dtype, np.float32)
        with self.assertRaises(ValueError):
            priors.sample(0, "images")

    def test_categories_are_codes_and_missing_cells_are_where_the_meta_says(self):
        tables = [priors.sample(s) for s in range(40)]
        with_cat = [t for t in tables if t.categorical.any()]
        self.assertTrue(with_cat)
        for t in with_cat:
            col = t.X[:, np.where(t.categorical)[0][0]]
            col = col[~np.isnan(col)]
            self.assertTrue(np.all(col == np.round(col)) and col.min() >= 0)
        self.assertTrue(any(t.meta["missing_cells"] for t in tables))
        self.assertTrue(all(int(np.isnan(t.X).sum()) == t.meta["missing_cells"] for t in tables))

    def test_most_tables_carry_signal_a_quick_model_finds(self):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        found = [priors.learnable(priors.sample(s, rows=(200, 400)))["signal"] for s in range(12)]
        self.assertGreaterEqual(sum(found), 8)

    def test_a_table_with_a_class_too_small_to_split_is_not_checked_rather_than_called_empty(self):  # reviewer finding 7
        t = priors.sample(570)
        counts = np.bincount(t.y)
        self.assertTrue(counts[counts > 0].min() < 2 or (counts == 0).any())
        got = priors.learnable(t)
        if counts[counts > 0].min() < 2:
            self.assertIsNone(got["signal"])
        else:  # an empty class code in the middle is not a class: the table is checked on the classes it has
            self.assertIsNotNone(got["signal"])

    def test_the_outcomes_effects_shown_are_recorded_as_positions(self):
        found = [priors.sample(s, "scm") for s in range(30)]
        with_effects = [t for t in found if t.meta["params"]["effects_shown"]]
        self.assertTrue(with_effects)
        for t in with_effects:
            self.assertTrue(all(0 <= i < t.meta["features"] for i in t.meta["params"]["effects_shown"]))

    def test_the_corpus_is_written_in_shards_that_read_back_whole_with_their_split(self):
        lake = Path(tempfile.mkdtemp())
        doc = corpus.build(tables=9, start=100, shard=3, lake=lake, check=False)
        self.assertEqual(doc["tables"], 9)
        seen = []
        for shard in sorted(corpus.folder(lake).rglob("shard-*.npz")):
            for t in corpus.read(shard):
                same = priors.sample(t["meta"]["seed"])
                self.assertTrue(np.array_equal(t["X"], same.X, equal_nan=True) and np.array_equal(t["y"], same.y))
                self.assertEqual(t["meta"]["split"], split_of(f"prior:{t['meta']['seed']}"))
                self.assertEqual(shard.parent.name, t["meta"]["split"])
                seen.append(t["meta"]["seed"])
        self.assertEqual(sorted(seen), list(range(100, 109)))
        again = corpus.build(tables=9, start=100, shard=3, lake=lake, check=False)  # the same seeds are never added twice
        self.assertEqual((again["tables"], again["added"]["skipped_as_kept"]), (9, 9))
        more = corpus.build(tables=6, start=105, shard=2, lake=lake, check=False)  # overlapping, another shard size: only new seeds
        self.assertEqual((more["tables"], more["added"]["skipped_as_kept"]), (11, 4))
        self.assertIn("no real table", json.loads((corpus.folder(lake) / "MANIFEST.json").read_text())["rule"])


if __name__ == "__main__":
    unittest.main()
