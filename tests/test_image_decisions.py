"""26.8: the image decision points (a near copy across sources, a suspect label, whether a mirror flip keeps the label),
their rules and truths on the shapes replica with its planted traps, and their decision sets."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_core import image_decisions as idc  # noqa: E402
from dclab_core import ledger  # noqa: E402
from dclab_core.replicas import shapes_and_notes as replica  # noqa: E402


def fingerprint(r):
    return hashlib.sha256(np.stack(r.items).tobytes() + r.table.to_csv(index=False).encode()).hexdigest()[:16]


class ReplicaTraps(unittest.TestCase):
    def test_the_defaults_draw_the_same_images_as_before(self):
        self.assertEqual(fingerprint(replica.shapes(0)), "ad45e1fc1c94e4f0")
        self.assertEqual(fingerprint(replica.shapes(3)), "015c0b92d33a21f7")

    def test_the_planted_traps_are_in_the_facts(self):
        r = replica.shapes(1, reshot=6, mislabel=0.05)
        self.assertEqual(len(r.facts["reshot_scenes"]), 6)
        scenes = set(r.table["scene"])
        self.assertTrue(all(a in scenes and b in scenes and a != b for a, b in r.facts["reshot_scenes"]))
        wrong = r.facts["mislabelled_items"]
        self.assertGreater(len(wrong), 0)
        self.assertTrue(r.facts["flip_keeps_label"])
        arrows = replica.shapes(1, arrows=True)
        self.assertEqual(set(arrows.table["label"]), {"left", "right"})
        self.assertFalse(arrows.facts["flip_keeps_label"])


class Rules(unittest.TestCase):
    def setUp(self):
        self.r = replica.shapes(2, reshot=8, mislabel=0.06)

    def test_near_copies_across_sources_are_found_and_look_alike_scenes_are_not_all_called_copies(self):
        records, labels = idc.decide(self.r, table="shapes-s2", at="2026-10-17T00:00:00+00:00")
        near = [(rec, lab) for rec, lab in zip(records, labels) if rec.point == "image.near_copy"]
        self.assertGreater(sum(lab.value for _, lab in near), 0)
        hits = sum(rec.value_used and lab.value for rec, lab in near)
        self.assertGreaterEqual(hits / sum(lab.value for _, lab in near), 0.8)
        self.assertGreater(sum(not lab.value for _, lab in near), 0)  # negatives, the look-alike ones among them

    def test_a_mislabelled_shot_is_suspected_from_its_near_copies(self):
        records, labels = idc.decide(self.r, table="shapes-s2", at="2026-10-17T00:00:00+00:00")
        sus = [(rec, lab) for rec, lab in zip(records, labels) if rec.point == "image.label_suspect"]
        planted = sum(lab.value for _, lab in sus)
        self.assertGreater(planted, 0)
        caught = sum(rec.value_used and lab.value for rec, lab in sus)
        false = sum(rec.value_used and not lab.value for rec, lab in sus)
        self.assertGreaterEqual(caught / planted, 0.8)
        self.assertLessEqual(false, planted)

    def test_a_flip_is_safe_for_shapes_and_not_for_arrows(self):
        for r, truth in ((replica.shapes(4), True), (replica.shapes(4, arrows=True), False)):
            records, labels = idc.decide(r, table="t", at="2026-10-17T00:00:00+00:00")
            (rec, lab), = [(a, b) for a, b in zip(records, labels) if a.point == "image.flip_safe"]
            self.assertEqual(lab.value, truth)
            self.assertEqual(rec.value_used, truth)

    def test_every_record_is_a_ledger_record_whose_state_holds_no_pixel(self):
        records, labels = idc.decide(self.r, table="shapes-s2", at="2026-10-17T00:00:00+00:00")
        self.assertEqual({r.point for r in records}, set(idc.POINTS))
        for rec in records:
            self.assertIn(rec.point, ledger.POINTS)
            self.assertEqual(ledger.check_state(rec.state), [])
            self.assertLessEqual(len(json.dumps(rec.state)), 400)
        self.assertEqual(len({r.id for r in records}), len(records))
        self.assertTrue(all(lab.source == "benchmark" for lab in labels))

    def test_the_rules_count_totals_before_any_cap(self):
        totals = {}
        out = idc.rule_decisions(self.r.items, self.r.table["scene"], self.r.table["label"], totals=totals)
        self.assertEqual(totals["labels_suspect"], sum(d["point"] == "image.label_suspect" and d["value"] for d in out))
        self.assertGreaterEqual(totals["near_copies_across_sources"], sum(d["point"] == "image.near_copy" and d["value"] for d in out))
        self.assertLessEqual(sum(d["point"] == "image.near_copy" for d in out), idc.MAX_PAIRS)


class Sets(unittest.TestCase):
    def test_the_image_sets_are_built_with_a_sealed_partition_and_the_same_seeds_build_the_same_files(self):
        from dclab_rnd import decisions_gen

        one, two = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        a = decisions_gen.build_images(seeds=6, out=one)
        b = decisions_gen.build_images(seeds=6, out=two)
        self.assertEqual(set(a["sets"]), set(idc.POINTS))
        for point, s in a["sets"].items():
            self.assertEqual(s["sealed_fingerprint"], b["sets"][point]["sealed_fingerprint"])
            rows = [json.loads(x) for x in (one / "decisions" / point / "rows.jsonl").read_text().splitlines()]
            sealed = [json.loads(x) for x in (one / "decisions" / point / "sealed.jsonl").read_text().splitlines()]
            self.assertFalse({r["group"] for r in rows} & {r["group"] for r in sealed})  # a table never on both sides
            self.assertTrue(all(r["label"]["source"] == "benchmark" for r in rows + sealed))
        self.assertGreater(sum(s["splits"].get("sealed", 0) for s in a["sets"].values()), 0)


if __name__ == "__main__":
    unittest.main()
