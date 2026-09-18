from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dclab_rnd.provenance import capture_provenance, file_sha256


class ProvenanceTests(unittest.TestCase):
    def test_hash_and_capture_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.csv"
            data.write_text("x,y\n1,0\n")
            first = file_sha256(data)
            second = capture_provenance(root, data_paths=[data], random_state=42)
            self.assertEqual(first, second["data_sha256"]["data.csv"])
            self.assertEqual(42, second["random_state"])
            self.assertIn("python", second)


if __name__ == "__main__":
    unittest.main()
