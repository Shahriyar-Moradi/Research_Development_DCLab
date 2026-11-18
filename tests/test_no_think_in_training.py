"""The owner's rule (2026-10-28, again 2026-11-18): reasoning comes only from the model's API thinking summary. No code asks
a model to write its reasoning inside tags, and no export puts a trace written by code where the model's thinking goes."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FORBIDDEN = ("inside <think>", "Think it through first", "from dclab_core.reasoning import think")


class NoThinkTagsTests(unittest.TestCase):
    def test_no_code_asks_for_think_tags_or_wraps_a_trace_in_them(self):
        hits = []
        for base in ("dclab_rnd", "dclab_core"):
            for path in sorted((ROOT / base).rglob("*.py")):
                text = path.read_text(encoding="utf-8")
                hits += [f"{path.relative_to(ROOT)}: {phrase}" for phrase in FORBIDDEN if phrase in text]
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
