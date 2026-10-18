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



class PatternCardTests(unittest.TestCase):
    """30.8: the owner's eleven problem patterns from their retail interview, each with its recipe and practice questions."""

    # SHA-256 (first 16 hex) of the names, brands and internal tools of the interview that must never appear in the
    # knowledge base: the tokens themselves are not written here (the review of 30.8)
    FORBIDDEN = {"94574690588bc3af", "f4d25158ccdd82c2", "7722aba40e28d98f", "630471bc17c69d0b", "860731b8f5d6db7c", "646b1045ad1e5259",
                 "6543723a48ab585d", "2d7f45d7b98b427f", "8bd8f9ff7afbfeb4", "77dc08c272fd3be2", "48017af3b98b07bc", "6bd455f33de782ba",
                 "b15df7b65fa45a57"}
    INDUSTRIES = {"banking", "insurance", "telecom", "healthcare", "e-commerce", "logistics", "travel"}

    @classmethod
    def setUpClass(cls):
        cls.patterns = kb.from_owner_documents()

    def test_eleven_patterns_each_linking_a_recipe_or_saying_it_is_knowledge(self):
        from dclab_rnd import kinds

        self.assertEqual(len(self.patterns), 11)
        plan = (ROOT / "docs" / "guides" / "PHASE8_KNOWLEDGE_CARDS_RECIPES_PLAN.md").read_text(encoding="utf-8")
        built = {k.key for k in kinds.all_kinds()}
        replicas = {p.stem for p in (ROOT / "dclab_core" / "replicas").glob("*.py") if p.stem != "__init__"}
        for e in self.patterns:
            self.assertEqual((e.source.kind, e.visibility), ("owner_document", "model"), e.id)
            self.assertTrue(e.pattern.general_problem and e.pattern.how_to_solve and e.pattern.how_to_measure and e.pitfalls, e.id)
            self.assertTrue((ROOT / e.source.ref.split("#")[0]).is_file(), e.source.ref)  # the owner's document is in the repository
            link = e.recipe
            if link.capability == "built":
                self.assertIn(link.kind, built, e.id)
            elif link.capability == "planned":
                self.assertIn(f"| {link.package} |", plan, e.id)  # a planned capability names the package that builds it
                self.assertTrue(link.kind not in built or link.modifiers, e.id)  # what is planned: a new kind, or the modifiers on a built one
            else:
                self.assertEqual((link.kind, link.package), (None, None), e.id)
            self.assertEqual(set(e.replicas) - replicas, set(), e.id)  # every replica it names exists
        self.assertEqual(sum(len(e.practice) for e in self.patterns), 10)

    def test_the_cross_industry_table_is_whole(self):
        by_id = {e.id: e for e in self.patterns}
        for pid in ("pattern:01_fraud_no_labels", "pattern:04_near_duplicates", "pattern:06_why_did_it_change", "pattern:08_department_agents"):
            self.assertLessEqual(self.INDUSTRIES, {x.industry for x in by_id[pid].pattern.elsewhere}, pid)
        self.assertEqual({e.id for e in self.patterns if not e.industry},  # the owner: these are not tied to an industry
                         {"pattern:05_drift_in_production", "pattern:07_secure_llm_platform", "pattern:09_simplest_tool",
                          "pattern:10_data_foundation", "pattern:11_many_projects"})

    def test_every_practice_question_carries_the_owners_rubric(self):
        for e in self.patterns:
            for q in e.practice:
                self.assertEqual(q.names[:3], ["the metric", "the validation split", "the first simple baseline before any model"], q.question)

    def test_the_search_finds_each_practice_questions_own_pattern(self):
        from dclab_core.business_examples import practice_recall

        found = practice_recall(kb.from_tracks() + kb.from_replicas() + self.patterns + kb.from_written(), top=3)
        self.assertEqual(found["found"], found["questions"], [r for r in found["rows"] if not r["found"]])

    def test_no_name_brand_or_internal_tool_and_no_conversation_reaches_a_model(self):
        import hashlib
        import re

        text = (ROOT / "dclab_rnd" / "knowledge" / "retail_interview_patterns.json").read_text(encoding="utf-8").lower()
        words = re.findall(r"[a-z]+", text)
        grams = set(words) | {f"{a} {b}" for a, b in zip(words, words[1:])}
        hashed = {hashlib.sha256(g.encode()).hexdigest()[:16] for g in grams}
        self.assertEqual(hashed & self.FORBIDDEN, set())
        shown = json.dumps(kb.context_for_model(self.patterns, "fraud returns labels drift agents data platform projects tools", 11))
        for e in self.patterns:
            self.assertNotIn(e.pattern.said, shown, e.id)  # what was said stays out of a model's context


if __name__ == "__main__":
    unittest.main()
