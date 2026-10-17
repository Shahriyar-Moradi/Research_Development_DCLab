"""Package 26.8: images in the engine, as rows of pixel columns, with their traps caught (a scene's shots on one side)."""

import unittest

try:
    import pandas  # noqa: F401
    import sklearn  # noqa: F401

    HAVE_ML = True
except ImportError:  # .venv-agent
    HAVE_ML = False


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class ImagesInTheEngineTests(unittest.TestCase):
    def test_the_shapes_run_to_their_holdout_with_every_scene_on_one_side(self):
        from dclab_rnd import kinds
        from dclab_rnd.kinds.acceptance import engine_run

        kind = kinds.get("embedded")
        checks = kind.accept_image_engine(engine_run(kind.image_case(0), "embedded"))
        self.assertEqual([c["decision"] for c in checks if not c["ok"]], [])
        self.assertEqual(len(checks), 5)

    def test_the_trap_is_caught_on_other_seeds_and_the_holdout_labels_reach_the_final_stage_only(self):
        import dataclasses

        from dclab_rnd import kinds
        from dclab_rnd.kinds.acceptance import engine_run
        from tests.test_engine_machines import before_final, run_case

        kind = kinds.get("embedded")
        for seed in (2, 3):
            self.assertEqual([c["decision"] for c in kind.accept_image_engine(engine_run(kind.image_case(seed), "embedded")) if not c["ok"]], [], seed)
        case = kind.image_case(0)
        _, _, records = run_case("embedded", case)
        held = set(records["final"]["evidence"]["holdout_row_positions"])
        table = case.table.copy()
        flip = table.index.isin(sorted(held))
        table.loc[flip, "label"] = table.loc[flip, "label"].map({"circle": "square", "square": "circle"})
        _, _, other = run_case("embedded", dataclasses.replace(case, table=table))
        first, second = before_final(records), before_final(other)
        for stage in first:
            self.assertEqual(first[stage], second[stage], f"the {stage} stage read the holdout's labels")

    def test_the_exported_code_reproduces_the_image_holdout(self):
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        from dclab_rnd import kinds
        from dclab_rnd.studio import code
        from tests.test_engine_machines import run_case

        store, pid, records = run_case("embedded", kinds.get("embedded").image_case(0))
        folder = Path(tempfile.mkdtemp()) / "code"
        code.export(store, pid, folder)
        out = subprocess.run([sys.executable, "-I", str(folder / "pipeline.py")], capture_output=True, text=True, cwd=tempfile.mkdtemp(), timeout=900)
        self.assertEqual(out.returncode, 0, out.stderr[-1500:])
        result = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertTrue(result["same"], result)

    def test_pixels_are_read_in_pixel_order_and_a_name_without_a_number_is_refused(self):
        from dclab_core import embeddings

        names = [f"pixel{k}" for k in range(16)]
        self.assertEqual(embeddings.pixel_columns(sorted(names), "pixel"), names)  # pixel2 before pixel10, whatever the table's order
        with self.assertRaises(ValueError):
            embeddings.pixel_columns(["pixel0", "pixel_mean"], "pixel")

    def machine(self, table, task="binary", **fields):
        from dclab_rnd import kinds
        from dclab_rnd.kinds.embedded import EmbeddedMachine
        from dclab_rnd.studio.solution import Solution

        sol = Solution(target="label", task=task, prediction_moment="when a photo arrives", kind="embedded", identifiers=["item"],
                       kind_fields={"source_column": "scene", "item_kind": "image", "pixel_prefix": "px_", **fields},
                       **({"positive_label": "square"} if task == "binary" else {}))
        return EmbeddedMachine(kinds.get("embedded"), sol, table, {"id": "p"})

    def test_the_image_points_read_training_images_only(self):  # 26.8, the review's F6
        import numpy as np

        from dclab_core.replicas import shapes_and_notes

        table = shapes_and_notes.shapes_table(0, scenes=40).table
        m = self.machine(table)
        got = m._image_decisions(0.98)
        training = {int(i) for i in m.index[~m.test]}
        keys = [int(k) for d in got if d["point"] != "image.flip_safe" for k in d["key"].split(":", 1)[1].split("-")]
        self.assertTrue(keys and set(keys) <= training)
        changed = table.copy()
        held = m.index[m.test]
        pixels = [c for c in table.columns if c.startswith("px_")]
        changed.loc[held, pixels] = np.random.default_rng(1).uniform(0, 1, (len(held), len(pixels))).round(4)
        changed.loc[held, "label"] = "circle"  # a held-out label read anywhere would move an answer
        self.assertEqual(self.machine(changed)._image_decisions(0.98), got)
        self.assertGreaterEqual(m.image_totals["near_copies_across_sources"],
                                sum(d["point"] == "image.near_copy" and d["value"] for d in got))  # totals before the caps

    def test_a_number_label_is_asked_only_about_near_copies(self):  # the review's F2
        from dclab_core.replicas import shapes_and_notes

        table = shapes_and_notes.shapes_table(0, scenes=30).table
        table["label"] = (table["item"] % 7).astype(float)
        got = self.machine(table, task="regression")._image_decisions(0.98)
        self.assertEqual({d["point"] for d in got}, {"image.near_copy"})

    def test_a_forbidden_pixel_is_refused(self):
        from dclab_rnd.studio.solution import Solution

        with self.assertRaises(ValueError):
            Solution(target="label", task="binary", prediction_moment="when a photo arrives", kind="embedded",
                     kind_fields={"source_column": "scene", "item_kind": "image", "pixel_prefix": "px_"},
                     forbidden=[{"column": "px_0042", "reason": "a watermark"}])

    def test_the_image_table_is_the_replicas_images_pixel_by_pixel(self):
        import numpy as np

        from dclab_core.replicas import shapes_and_notes

        r = shapes_and_notes.shapes_table(0, scenes=4, shots=2, size=24)
        pixels = [c for c in r.table.columns if c.startswith("px_")]
        self.assertEqual((len(r.table), len(pixels)), (8, 576))
        np.testing.assert_allclose(r.table[pixels].to_numpy()[3].reshape(24, 24), r.items[3], atol=1e-4)

    def test_images_need_their_pixel_columns_named(self):
        from dclab_rnd.kinds.embedded import Embedded
        from dclab_rnd.studio.solution import Solution

        with self.assertRaises(ValueError) as refused:  # refused when the solution is saved, before any stage runs
            Solution(target="label", task="binary", prediction_moment="when a photo arrives", kind="embedded",
                     kind_fields={"source_column": "scene", "item_kind": "image"})
        self.assertIn("pixel_prefix", str(refused.exception))
        self.assertTrue(Embedded().runs_in_engine)


if __name__ == "__main__":
    unittest.main()
