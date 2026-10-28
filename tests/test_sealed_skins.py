"""31.3: the sealed variants have domains of their own, written by an independent agent; practice never meets their names.

Only column names are read from the sealed variants here: nothing is solved, graded or stored, so the exam's one read
is not spent.
"""
import hashlib
import json
import unittest
from dataclasses import fields

from dclab_core import variants as V


def names_of(skins):
    return {getattr(s, f.name) for s in skins for f in fields(V.Skin) if f.name not in ("prefix", "row")}


class SealedSkinsTests(unittest.TestCase):
    def test_the_sealed_skins_are_pinned(self):
        canon = json.dumps([[getattr(s, f.name) for f in fields(V.Skin)] for s in V.SEALED_SKINS], separators=(",", ":"))
        self.assertEqual(hashlib.sha256(canon.encode()).hexdigest(), V.SEALED_SKINS_SHA256, "a change to the sealed skins is a new sealed set")
        self.assertEqual(len(V.SEALED_SKINS), 8)

    def test_no_name_is_shared_between_practice_and_the_exam(self):
        self.assertEqual(names_of(V.SKINS) & names_of(V.SEALED_SKINS), set())
        self.assertEqual({s.row for s in V.SKINS} & {s.row for s in V.SEALED_SKINS}, set())

    def test_each_partition_draws_from_its_own_domains(self):
        practice, exam = names_of(V.SKINS) | set(V.SHARED), names_of(V.SEALED_SKINS) | set(V.SHARED)
        for partition in ("learning", "check"):
            for seed in V.seeds("groups", partition, 20):
                columns = set(V.generate("groups", seed).table.columns)
                self.assertLessEqual(columns, practice, f"{partition} seed {seed}")
        for seed in V.seeds("groups", "sealed", 20):
            variant = V.generate("groups", seed)
            self.assertLessEqual(set(variant.table.columns), exam, f"sealed seed {seed}")
            self.assertEqual(variant.key.knobs.get("skins"), "sealed_v1")

    def test_no_lesson_may_name_a_sealed_column(self):
        self.assertLessEqual(names_of(V.SEALED_SKINS), set(V.family_names("groups")))


if __name__ == "__main__":
    unittest.main()
