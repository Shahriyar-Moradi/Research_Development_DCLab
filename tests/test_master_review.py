import unittest
from dclab_rnd.master_review import build_pack, rules, workflow_blocks, render_markdown

class MasterReviewTests(unittest.TestCase):
    def test_evidence_pack_preserves_protocol_boundaries(self):
        pack=build_pack()
        self.assertEqual(len(pack["campaign"]),10)
        self.assertEqual(pack["churn"]["completed"],15)
        self.assertEqual(pack["scope"]["hyperack_historical_experiments"],83)
        self.assertFalse(pack["churn"]["production_approved"])
        self.assertTrue(any("must not be pooled" in x for x in pack["boundaries"]))

    def test_rules_are_cited_and_workflows_are_complete(self):
        ruleset=rules();blocks=workflow_blocks()
        self.assertEqual(len(ruleset),22);self.assertEqual(len(blocks),10)
        self.assertEqual(len({r["rule_id"] for r in ruleset}),22)
        self.assertTrue(all(r["evidence"] and r["production_approved"] is False for r in ruleset))
        guide=render_markdown(build_pack(),ruleset,blocks)
        self.assertIn("Leakage: the practical taxonomy",guide)
        self.assertIn("Future-LLM training format",guide)
        self.assertIn("DCLAB-R22",guide)

if __name__=="__main__":unittest.main()
