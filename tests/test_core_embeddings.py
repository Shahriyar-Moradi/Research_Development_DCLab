"""Package 16.12: images and long text through encoders. Vectors are cached by content (a second pass encodes nothing,
another encoder never reads them), near copies across a split are counted (many across random folds, few across folds by
source), copied texts are found across authors, a source stays on one side, and the encoders that need no download
say they are baselines; the pretrained image encoder never downloads its weights."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import embeddings  # noqa: E402
from dclab_core.replicas import shapes_and_notes  # noqa: E402


class Counting:
    pretrained = False

    def __init__(self, name="counting"):
        self.name, self.calls = name, 0

    def encode(self, items):
        self.calls += len(items)
        return np.array([[len(str(i)), sum(map(ord, str(i))) % 97] for i in items], dtype=np.float32)


class CacheTest(unittest.TestCase):
    def test_the_same_content_is_encoded_once_per_encoder(self):
        cache = embeddings.Cache(tempfile.mkdtemp())
        enc = Counting()
        X, first = embeddings.encode(["a", "b", "a"], enc, cache)
        self.assertEqual((first["distinct_contents"], first["encoded_now"], enc.calls), (2, 2, 2))
        np.testing.assert_array_equal(X[0], X[2])
        _, again = embeddings.encode(["b", "a"], enc, cache)
        self.assertEqual((again["encoded_now"], again["from_cache"], enc.calls), (0, 2, 2))
        other = Counting("another")
        embeddings.encode(["a"], other, cache)
        self.assertEqual(other.calls, 1)  # another encoder never reads this one's vectors

    def test_content_is_the_key_whatever_its_form(self):
        from PIL import Image

        rng = np.random.default_rng(0)
        pics = [Image.fromarray((rng.uniform(size=(8, 8)) * 255).astype(np.uint8)) for _ in range(20)]
        self.assertEqual(len({embeddings.content_key(p) for p in pics}), 20)  # never keyed by a printed form
        self.assertNotEqual(embeddings.content_key(b"abc"), embeddings.content_key("abc"))
        f = Path(tempfile.mkdtemp()) / "a.txt"
        f.write_text("one")
        first = embeddings.content_key(f)
        f.write_text("two")
        self.assertNotEqual(first, embeddings.content_key(f))  # a replaced file is a new content
        with self.assertRaises(TypeError):
            embeddings.content_key(object())

    def test_vectors_of_another_width_under_one_name_are_refused(self):
        cache = embeddings.Cache(tempfile.mkdtemp())
        cache.put("counting", embeddings.content_key("a"), np.zeros(5))
        with self.assertRaises(ValueError):
            embeddings.encode(["a", "b"], Counting(), cache)

    def test_near_copies_across_sides_by_hand(self):
        X = np.array([[1.0, 0.0], [0.999, 0.01], [0.0, 1.0], [0.01, 1.0]])
        found = embeddings.near_duplicates(X, ["a", "b", "a", "a"], threshold=0.99)
        self.assertEqual(found["pairs_across_sides"], 1)  # rows 0 and 1 are near and on different sides; 2 and 3 share a side
        self.assertEqual(found["pairs"][0][:2], (0, 1))


class ReplicaTest(unittest.TestCase):
    def test_image_shots_of_one_scene_are_near_copies_a_random_split_spreads(self):
        im = shapes_and_notes.shapes(0)
        enc = embeddings.RandomFeatures()
        X, record = embeddings.encode(im.items, enc)
        self.assertFalse(record["pretrained"])
        cal = embeddings.calibrate(im.items, enc, embeddings.perturb_image)
        self.assertTrue(cal["separable"])
        r = embeddings.evaluate(X, (im.table["label"] == "square").astype(int).to_numpy(), im.table["scene"], threshold=cal["threshold"])
        self.assertEqual(r["sources_on_both_sides"], 0)
        self.assertFalse(set(r["train_rows"]) & set(r["holdout_rows"]))
        self.assertEqual(r["fitted_rows"], len(r["train_rows"]))  # the holdout's rows are never fitted
        self.assertEqual(r["near_duplicates_across_holdout"]["pairs_across_sides"], 0)
        self.assertGreater(r["near_duplicates_across_random_folds"]["pairs_across_sides"], 5 * max(r["near_duplicates_across_folds"]["pairs_across_sides"], 1))
        self.assertLess(r["grouped_folds"], r["random_folds"])  # near copies flatter random folds
        y = (im.table["label"] == "square").astype(int).to_numpy()
        y[r["holdout_rows"]] = 1 - y[r["holdout_rows"]]  # other labels on the holdout: the model must not move
        again = embeddings.evaluate(X, y, im.table["scene"], threshold=cal["threshold"])
        self.assertEqual(again["holdout_predictions"], r["holdout_predictions"])
        self.assertGreaterEqual(r["grouped_folds"], 0.9)

    def test_texts_pasted_by_another_author_are_found(self):
        nt = shapes_and_notes.notes(0)
        enc = embeddings.HashingText()
        X, _ = embeddings.encode(nt.items, enc)
        threshold = embeddings.calibrate(nt.items, enc, embeddings.perturb_text)["threshold"]
        found = {tuple(sorted(p[:2])) for p in embeddings.near_duplicates(X, nt.table["author"].to_numpy(), threshold=threshold)["pairs"]}
        copied = {tuple(sorted(p)) for p in nt.facts["copied_pairs"]}
        self.assertEqual(found, copied)  # every copy, and nothing else
        with self.assertRaises(ValueError):
            embeddings.evaluate(X[:20], nt.table["urgent"].to_numpy()[:20], ["one"] * 20)  # one source: nothing to hold apart


class KindTest(unittest.TestCase):
    def setUp(self):
        try:
            from dclab_rnd import kinds
        except ImportError as error:
            self.skipTest(str(error))
        self.kinds = kinds

    def test_the_acceptance_holds_on_several_seeds_and_says_no_pretrained_encoder_was_measured(self):
        kind = self.kinds.get("embedded")
        for seed in (0, 1, 2):
            case = kind.case(seed)
            run = kind.solve(type(case)(**{**case.__dict__, "facts": {}}))
            decisions = kind.accept({**run, "facts": case.facts})
            self.assertTrue(all(c["ok"] for c in decisions[:-1]), (seed, decisions))
            self.assertFalse(decisions[-1]["ok"])  # baselines only: the open item, listed as missing
        uncached = {**run, "facts": case.facts, "images": {**run["images"], "again": {**run["images"]["again"], "encoded_now": 5}}}
        self.assertFalse(kind.accept(uncached)[0]["ok"])

    def test_detected_only_when_the_items_are_the_input(self):
        self.assertEqual(self.kinds.detect("Classify the photos of parts as defective or not")["key"], "embedded")
        for other in ("Forecast daily demand per store; we also have photos of the shelves",
                      "How long until a customer churns, given photos of their store; some are still active (censored)",
                      "Predict which customers churn; we keep scans of their signed contracts"):
            self.assertNotEqual(self.kinds.detect(other)["key"], "embedded", other)

    def test_the_image_encoder_never_downloads_and_the_engine_refuses_the_kind(self):
        from dclab_rnd.encoders import TorchvisionImage
        from dclab_rnd.studio.solution import Solution

        with self.assertRaises(ValueError) as refused:
            TorchvisionImage(Path(tempfile.mkdtemp()) / "missing.pt")
        self.assertIn("never downloads", str(refused.exception))
        # torch runs in a process of its own: loaded beside another OpenMP library (an earlier test's boosters) it can crash
        import subprocess
        import textwrap

        script = textwrap.dedent(f"""
            import sys, tempfile
            from pathlib import Path
            import numpy as np
            sys.path.insert(0, {str(ROOT)!r})
            try:
                import torch, torchvision
            except ImportError:
                print("skip"); sys.exit(0)
            from dclab_rnd.encoders import TorchvisionImage
            weights = Path(tempfile.mkdtemp()) / "weights.pt"
            torch.manual_seed(0)
            torch.save(torchvision.models.resnet18(weights=None).state_dict(), weights)  # random weights, written here
            enc = TorchvisionImage(weights, size=64)
            photo = (np.random.default_rng(0).uniform(size=(16, 16, 3)) * 255).astype(np.uint8)
            same = np.allclose(enc.encode([photo])[0], enc.encode([photo.astype(np.float32) / 255])[0], rtol=1e-4, atol=1e-5)
            print("ok" if same and "-64" in enc.name else "different")
        """)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
        self.assertIn(out.stdout.strip(), ("ok", "skip"), out.stderr[-500:])
        with self.assertRaises(ValueError):
            Solution(target="label", task="binary", prediction_moment="When the photo is taken", kind="embedded",
                     kind_fields={"source_column": "scene"})


if __name__ == "__main__":
    unittest.main()
