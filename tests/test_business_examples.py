"""Package 30.5: the business-examples knowledge base: built from what the repository knows (tracks, replicas, the owner's
solution when beside it) and the written examples, each with its source; a person's words find the closest."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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



class WhatAModelMayReadTests(unittest.TestCase):
    """30.7: an example says whether a model may read it; the owner's fraud solution is read by the rules only."""

    def test_a_stored_v1_example_reads_as_v2_and_v2_adds_the_pattern_practice_and_recipe(self):
        v1 = {"schema_version": "dclab_core.business_example.v1", "id": "x", "title": "t", "problem": "p",
              "source": {"kind": "written", "ref": "public knowledge"}}
        e = BusinessExample.read(v1)
        self.assertEqual((e.schema_version, e.visibility), ("dclab_core.business_example.v2", "model"))
        e = BusinessExample(id="p1", title="Fraud with no labels", problem="p", source={"kind": "owner_document", "ref": "docs/sources/x.pdf"},
                            pattern={"general_problem": "huge volumes, a few bad ones, no list", "why_hard": ["unusual is not fraud"],
                                     "elsewhere": [{"industry": "banking", "case": "card fraud"}]},
                            practice=[{"question": "Find the first 500 cases worth reviewing", "names": ["precision at K", "time split"]}],
                            recipe={"kind": "anomaly", "modifiers": ["no_target", "review_capacity"], "capability": "built"})
        self.assertEqual(e.pattern.elsewhere[0].industry, "banking")
        with self.assertRaises(ValueError):
            BusinessExample(id="x", title="t", problem="p", source={"kind": "written", "ref": "r"}, visibility="everyone")

    def test_the_owners_fraud_solution_is_for_the_rules_only_and_never_in_a_models_context(self):
        readme = Path(tempfile.mkdtemp()) / "README.md"
        readme.write_text("* **Labels.** a sample of the cases found.\n* **Choice.** PU bagging.\n", encoding="utf-8")
        with mock.patch.object(kb, "SNAP", readme):
            owner = kb.from_owner_fraud()
        self.assertEqual([e.visibility for e in owner], ["rules_only"])
        examples = owner + kb.from_written()
        self.assertIn(owner[0].id, [e.id for e, _ in search(examples, "fraud bikers rides positive-unlabelled", 3)])  # the rules see it
        shown = json.dumps(kb.context_for_model(examples, "fraud bikers rides positive-unlabelled", 5))
        self.assertNotIn(owner[0].id, shown)
        self.assertNotIn("PU bagging", shown)
        self.assertNotIn("Snap_fruad", shown)  # no example's source path either: the model reads the pattern, not where it is kept

    def test_an_owners_solution_stored_as_v1_stays_for_the_rules_only_and_none_may_be_made_visible(self):
        v1 = {"schema_version": "dclab_core.business_example.v1", "id": "owner:x", "title": "t", "problem": "p",
              "pitfalls": ["joined on the wrong key"], "source": {"kind": "owner_solution", "ref": "elsewhere/README.md"}}
        e = BusinessExample.read(v1)
        self.assertEqual(e.visibility, "rules_only")
        self.assertEqual(kb.context_for_model([e], "joined on the wrong key", 3), [])
        with self.assertRaises(ValueError):
            BusinessExample(id="owner:y", title="t", problem="p", source={"kind": "owner_solution", "ref": "r"})

    def test_a_models_context_holds_the_general_pattern_never_the_conversation_or_where_it_is_kept(self):
        e = BusinessExample(id="p:x", title="Fraud with no labels", problem="Fake returns on copied bills",
                            source={"kind": "owner_document", "ref": "docs/sources/private_folder/notes.pdf"},
                            pattern={"said": "Ada Example told us the store closes at 11 p.m.", "general_problem": "a few bad events, no list",
                                     "how_to_measure": ["calibration curve on confirmed cases"]},
                            practice=[{"question": "How do you find the first 500 cases worth reviewing?", "names": ["precision at K"]}])
        shown = json.dumps(kb.context_for_model([e], "fake returns copied bills", 3))
        self.assertIn("a few bad events, no list", shown)
        for hidden in ("Ada Example", "private_folder", "store closes"):
            self.assertNotIn(hidden, shown)
        self.assertTrue(search([e], "calibration curve"))  # how it is measured is searchable
        self.assertEqual(search([e], "first 500 worth reviewing"), [])  # a practice question is never its own answer

    def test_without_the_owners_folder_a_rebuild_keeps_the_committed_owner_example(self):
        with mock.patch.object(kb, "SNAP", Path(tempfile.mkdtemp()) / "missing.md"):
            kept = kb.from_owner_fraud()
        committed = [e for e in kb.load() if e.source.kind == "owner_solution"]
        self.assertEqual([e.id for e in kept], [e.id for e in committed])
        self.assertTrue(all(e.visibility == "rules_only" for e in kept))

    def test_the_committed_knowledge_base_is_what_the_code_builds(self):
        out = Path(tempfile.mkdtemp())
        kb.build(out)
        def lines(folder):
            return (folder / "examples.jsonl").read_text(encoding="utf-8").splitlines()  # without the owner's folder, the committed owner line is kept
        self.assertEqual(lines(out), lines(kb.OUT), "run: make business-examples")


if __name__ == "__main__":
    unittest.main()
