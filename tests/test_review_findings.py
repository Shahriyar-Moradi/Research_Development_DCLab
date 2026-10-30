"""2026-10-30 (the owner: the reviewer's findings, each confirmed or declined, are a judge's lessons): every reviewer finding
of the development records as a judged claim in m7_judges/review_findings; a declined finding scores 0 and never becomes a
"fix this defect" example; open findings and other sources are left out; one package stays in one split."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def record(findings):
    return {"package": "pkg-x", "title": "A change", "date": "2026-10-30", "request": "Do the thing well.", "spec": "s", "plan": ["p"],
            "decisions": [], "change": "The change made.", "findings": findings, "measurements": [], "outcome": "The change is done and tested.", "lessons": [],
            "risks": [], "commits": []}


def finding(fid, verdict="confirmed", test="tests/test_x.py T.test_y", source="reviewer"):
    return {"id": fid, "source": source, "severity": "behaviour", "verdict": verdict, "defect": "A defect that matters here",
            "scenario": "A scenario that shows it", "cause": "c", "fix": "A fix or the reason it was not taken", "test": test}


class ReviewFindingsTests(unittest.TestCase):
    def test_each_reviewer_finding_is_judged_by_its_test_or_its_verdict(self):
        from dclab_rnd.store.sets.others import review_findings

        root = Path(tempfile.mkdtemp())
        folder = root / "research/llm-fine-tuning/experiments/dev_trajectories/records"
        folder.mkdir(parents=True)
        (folder / "pkg-x.json").write_text(json.dumps(record([
            finding("F1"), finding("F2", test="(code)"), finding("F3", verdict="declined", test="none"), finding("F1"),
            finding("F4", verdict="open"), finding("F5", source="test")])))
        out = {x.id.rsplit(":", 1)[1]: x for x in review_findings(root)}
        self.assertEqual(sorted(out), ["F1", "F1.3", "F2", "F3"])  # a reused id gets its own
        self.assertEqual((out["F1"].score, out["F1"].evidence, out["F1"].judge, out["F1"].weight), (1.0, "test", "tests", 0.5))
        self.assertEqual((out["F2"].evidence, out["F2"].weight), ("model", 0.3))
        self.assertEqual(out["F3"].score, 0.0)  # a plausible claim that was wrong
        self.assertNotIn("fix", json.dumps(out["F1"].candidate))  # the claim, never its fix
        self.assertEqual(len({x.split for x in out.values()}), 1)  # one package, one split

    def test_a_declined_finding_is_never_a_fix_this_defect_example(self):
        from dclab_rnd import devlog

        t = devlog.Trajectory.model_validate(record([finding("F1"), finding("F3", verdict="declined", test="none")]))
        fixes = [e for e in devlog.examples(t) if e["task"] == "fix_finding"]
        self.assertEqual(len(fixes), 1)


if __name__ == "__main__":
    unittest.main()
