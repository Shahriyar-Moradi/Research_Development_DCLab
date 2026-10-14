"""Package 22.1: generated decision benchmarks. Planted columns of every kind, their names hidden in a twin table, the
rules' decisions recorded with benchmark labels; a table and its twin share a group, so they never sit in two
splits; a state holds bands, never a value; the same seeds build the same sealed partition."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core import benchgen as bg
    from dclab_core.ledger import check_state
    from dclab_core.replicas import subscription_churn
    from dclab_rnd import decisions_gen
except ImportError as error:  # the agent environment has no numpy
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class PlantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = subscription_churn.generate(seed=0).subscribers
        cls.frame, cls.planted = bg.plant(cls.base, "churned", seed=3, rows=400)

    def test_every_kind_is_planted_and_its_twin_hides_every_name(self):
        self.assertEqual(sorted(self.planted.values()), sorted(bg.KINDS))
        hidden, planted, target = bg.obfuscate(self.frame, self.planted, "churned")
        self.assertTrue(all(c.startswith("f") and c[1:].isdigit() for c in hidden.columns))
        self.assertEqual(sorted(planted.values()), sorted(bg.KINDS))
        self.assertIn(target, hidden.columns)

    def test_decisions_carry_bands_and_their_benchmark_label(self):
        records, labels = bg.decide(self.frame, self.planted, "churned", "churn-s3", "2026-10-14T00:00:00+00:00")
        self.assertEqual(len(records), len(labels))
        for r in records:
            self.assertEqual(check_state(r.state), [], r.state)  # bands and names only: never a table's value
            self.assertEqual(r.dataset, "churn-s3")
        truth = {l.record: (l.value, l.source) for l in labels}
        kind = {c: k for c, k in self.planted.items()}
        for r in records:
            self.assertEqual(truth[r.id], (bg.TRUTH[kind[r.subject.key]][r.point], "benchmark"))

    def test_each_kind_is_what_its_truth_says_read_from_the_data(self):
        """Independent of TRUTH: each planted column checked by its values."""
        import numpy as np
        import pandas as pd

        f, kind_of = self.frame, {k: c for c, k in self.planted.items()}
        col = lambda k: f[kind_of[k]]  # noqa: E731
        y = (f["churned"] == "yes").astype(float) if f["churned"].dtype == object else f["churned"].astype(float)
        self.assertTrue(col("identifier").is_unique)
        self.assertLess(col("entity").nunique(), len(f) / 2)  # a key that recurs
        self.assertTrue(3 <= col("code").nunique() <= 8 and pd.api.types.is_numeric_dtype(col("code")))
        self.assertTrue(12 <= col("wide_code").nunique() <= 18)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(col("date")) and col("date").is_monotonic_increasing)
        self.assertFalse(col("other_date").is_monotonic_increasing)
        self.assertGreaterEqual(col("text").astype(str).str.split().str.len().min(), 8)
        rare = y == y.value_counts().idxmin()
        self.assertTrue(((col("after_outcome").notna()) == rare).all())  # filled exactly when the outcome is the rare one
        self.assertTrue(np.allclose(col("copy").astype(float), y))
        self.assertTrue(np.allclose(col("sum") + col("part"), y, atol=1e-5))  # a sum identity: the two parts give the outcome
        self.assertLess(abs(np.corrcoef(col("part"), y)[0, 1]), 0.3)  # the part alone says little
        self.assertTrue(0.2 < abs(np.corrcoef(col("signal"), y)[0, 1]) < 0.9)  # a real input, not a copy
        self.assertTrue(col("gappy").isna().any() and abs(col("gappy").isna().astype(float).corr(y)) < 0.3)  # gaps at random

    def test_a_hidden_name_never_says_the_kind(self):
        names = {}
        for seed in range(6):
            frame, planted = bg.plant(self.base, "churned", seed=seed, rows=200)
            _, hidden, _ = bg.obfuscate(frame, planted, "churned", seed=seed)
            for column, kind in hidden.items():
                names.setdefault(kind, set()).add(column)
        self.assertTrue(all(len(v) > 1 for v in names.values()), names)  # the same kind under different hidden names

    def test_a_twin_shares_its_tables_group(self):
        hidden, planted, target = bg.obfuscate(self.frame, self.planted, "churned")
        records, _ = bg.decide(hidden, planted, target, "churn-s3-ob", "2026-10-14T00:00:00+00:00", group="churn-s3")
        self.assertEqual({r.dataset for r in records}, {"churn-s3"})


@unittest.skipIf(MISSING, MISSING)
class BuildTest(unittest.TestCase):
    def test_a_small_build_and_its_sealed_partition_does_not_move(self):
        out1, out2 = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        first = decisions_gen.build(seeds=2, rows=300, out=out1)
        again = decisions_gen.build(seeds=2, rows=300, out=out2)
        self.assertEqual({p: s["sealed_fingerprint"] for p, s in first["sets"].items()},
                         {p: s["sealed_fingerprint"] for p, s in again["sets"].items()})
        point = first["sets"]["column.is_identifier"]
        self.assertGreater(point["decisions"], 0)
        rows = [json.loads(x) for f in ("rows.jsonl", "sealed.jsonl") for x in (out1 / "decisions" / "column.is_identifier" / f).read_text().splitlines()]
        groups = {}
        for r in rows:
            groups.setdefault(r["group"], set()).add(r["split"])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))  # a group sits in one split, twins included
        self.assertTrue(all("-ob" not in g for g in groups))

    def test_no_single_band_gives_a_label_away(self):
        out = Path(tempfile.mkdtemp())
        decisions_gen.build(seeds=3, rows=300, out=out)
        docs = [json.loads(x) for x in (out / "records.jsonl").read_text().splitlines()]
        for point in ("feature.leakage_suspect", "column.is_identifier", "column.category_code", "column.time"):
            rows = [(d["state"], d["labels"][0]["value"]) for d in docs if d["point"] == point]
            for field in ("dtype", "uniqueness", "nulls", "cardinality", "value_pattern"):
                by = {}
                for state, label in rows:
                    by.setdefault(state.get(field), []).append(label)
                best = sum(max(v.count(True), v.count(False)) for v in by.values()) / len(rows)  # the best rule on this field alone
                with self.subTest(point=point, field=field):
                    self.assertLess(best, 0.97)

    def test_the_committed_manifest_meets_the_done_when(self):
        manifest = json.loads((decisions_gen.OUT / "MANIFEST.json").read_text())
        for point in ("column.is_identifier", "column.semantic_role", "feature.leakage_suspect",  # the Jev purposes
                      "column.category_code", "column.time", "column.group"):  # the studio's points with a planted truth
            self.assertGreaterEqual(manifest["sets"][point]["decisions"], 2000, point)
            self.assertGreaterEqual(manifest["sets"][point]["tables"], 100, point)
            self.assertGreater(manifest["sets"][point]["splits"].get("sealed", 0), 0)


if __name__ == "__main__":
    unittest.main()
