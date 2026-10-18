"""Package 30.5: the business-examples knowledge base: built from what the repository knows (tracks, replicas, the owner's
solution when beside it) and the written examples, each with its source; a person's words find the closest."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core.business_examples import BusinessExample, search  # noqa: E402
from dclab_rnd import business_examples as kb  # noqa: E402


class KnowledgeBaseTests(unittest.TestCase):
    def test_every_example_states_its_source_and_written_ones_cite_no_run(self):
        out = Path(tempfile.mkdtemp())
        manifest = kb.build(out)
        examples = kb.load(out)
        self.assertEqual(len(examples), manifest["examples"])
        self.assertGreaterEqual(manifest["by_source"].get("replica", 0), 10)  # every replica with its planted traps
        self.assertGreaterEqual(manifest["by_source"].get("written", 0), 10)
        for e in examples:
            if e.source.kind == "written":
                self.assertEqual(e.results, [])  # a written example is no record of a run
            if e.source.kind == "replica":
                self.assertTrue(e.pitfalls, e.id)  # the traps a careful data scientist must see
        self.assertEqual(json.loads((out / "MANIFEST.json").read_text())["examples"], len(examples))

    def test_a_persons_words_find_the_closest_example(self):
        examples = kb.from_replicas() + kb.from_written()
        found = [e.id for e, _ in search(examples, "find the fake orders our bikers make, we have a few confirmed cases", 2)]
        self.assertTrue(any("fraud" in i for i in found), found)
        found = [e.id for e, _ in search(examples, "how many units will each store sell next week", 2)]
        self.assertTrue(any("demand" in i for i in found), found)
        self.assertEqual(search(examples, "zzzz qqqq"), [])

    def test_the_contract_refuses_an_unknown_source(self):
        with self.assertRaises(ValueError):
            BusinessExample(id="x", title="t", problem="p", source={"kind": "rumour", "ref": "r"})


if __name__ == "__main__":
    unittest.main()
