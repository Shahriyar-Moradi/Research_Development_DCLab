"""2026-11-02: the fresh sealed set sealed_v3 (12 people written by an independent writer shown the tables): registered, pinned, written to
fit its tables, and in the order the brief asks for. These tests read its STRUCTURE only (counts, order, claims), never its stories."""
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval import conversation_exam as X  # noqa: E402
from dclab_rnd.agent_eval.conversation_exam import gate as G  # noqa: E402


class SealedV3Tests(unittest.TestCase):
    def people(self):
        return json.loads(X.SPLITS["sealed_v3"].read_text(encoding="utf-8"))

    def test_it_is_registered_pinned_and_compared_with_claude_alone_only(self):
        self.assertIn("sealed_v3", X.SPLITS)
        self.assertEqual(set(X.SEALED_ARMS["sealed_v3"]), {"plain", "case"})
        self.assertEqual(G.gate_for("sealed_v3"), G.GATE_V4)
        self.assertIn("sealed_v3", G.GATES)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(X.main(["--split", "sealed_v3", "--arms", "plain,case"]), 0)  # the plan only; nothing is sent
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            self.assertEqual(X.main(["--split", "sealed_v3", "--arms", "plain,case", "--only", "s3-01"]), 2)  # a sealed set is run whole
        self.assertIn("whole", err.getvalue())

    def test_an_edit_after_it_was_pinned_is_refused(self):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        copy = folder / "people_sealed_v3.json"
        copy.write_bytes(X.SPLITS["sealed_v3"].read_bytes() + b" ")
        err = io.StringIO()
        with mock.patch.dict(X.SPLITS, {"sealed_v3": copy}), redirect_stderr(err), redirect_stdout(io.StringIO()):
            self.assertEqual(X.main(["--split", "sealed_v3", "--arms", "plain,case"]), 2)
        self.assertIn("fingerprint", err.getvalue())

    def test_every_person_passes_the_checks_and_every_table_story_carries_claims(self):
        people = self.people()
        self.assertEqual([p["id"] for p in people], [f"s3-{i:02d}" for i in range(1, 13)])
        self.assertIn("sealed_v3", X.REQUIRE_CLAIMS)
        self.assertEqual({p["id"]: X.check_person(p, require_claims=True) for p in people if X.check_person(p, require_claims=True)}, {})
        self.assertTrue(all(p.get("table_claims") for p in people if p.get("upload")))
        self.assertEqual(len({p["opening"].strip().casefold() for p in people}), 12)

    def test_the_hard_kinds_come_first_and_the_minimums_of_the_brief_are_met(self):
        people = self.people()
        advice = [p for p in people if p["route"] == "advice" and not p.get("upload")]
        clear = [p for p in people if p.get("trap") == "clear_request"]
        self.assertGreaterEqual(len(advice), 3)
        self.assertGreaterEqual(len(clear), 3)
        self.assertTrue(all(p.get("upload") for p in clear))  # a clear request is about a table that holds it
        self.assertEqual([p["route"] for p in people[:3]], ["advice"] * 3)  # a run that stops early has played the costly ones
        self.assertTrue({p["id"] for p in clear} <= {p["id"] for p in people[3:7]})
        traps = {p["trap"] for p in people}
        self.assertTrue({"real_prediction", "report_not_model", "missing_column", "vague_goal", "silent_drop", "no_outcome"} <= traps)
        for key, wanted in (("level", set(X.LEVELS)), ("style", set(X.STYLES))):
            self.assertEqual({p[key] for p in people}, wanted)


class LabelCountTests(unittest.TestCase):
    """The review of the registration: gate v3 compared an agreement number with its bar and never counted the papers it came from."""

    def rows(self, arm, understanding, trap_pass, fit):
        return [{"id": f"s{i}", "arm": arm, "understanding": understanding, "options_fit": fit, "route_reached": True, "safety_failures": 0,
                 "trap": "vague_goal", "trap_pass": trap_pass, "unjudged": False} for i in range(8)]

    def scores(self):
        return {"case": self.rows("case", 0.9, True, 5), "plain": self.rows("plain", 0.5, False, 3)}

    def test_v4_waits_when_too_few_papers_were_labelled_and_decides_when_enough(self):
        gate = G.gate_for("sealed_v3")
        self.assertEqual(G.verdict(self.scores(), gate=gate)["status"], "pending_owner_labels")  # no labels yet
        few = G.verdict(self.scores(), agreement=0.95, gate=gate, labelled=1)  # one paper cannot carry the verdict
        self.assertEqual((few["status"], few["labelled"], few["labels_needed"]), ("pending_owner_labels", 1, 6))
        self.assertEqual(G.verdict(self.scores(), agreement=0.95, gate=gate, labelled=6)["status"], "pass")
        self.assertEqual(G.verdict(self.scores(), agreement=0.5, gate=gate, labelled=6)["status"], "fail")
        with self.assertRaises(ValueError):
            G.verdict(self.scores(), agreement=0.95, gate=gate)  # an agreement with no count is refused

    def test_the_gates_of_the_sets_already_run_are_unchanged(self):
        for split in ("sealed_v1", "sealed_v2"):
            self.assertNotIn("enforce_count", G.gate_for(split)["judge_owner_agreement"])
        v = G.verdict(self.scores(), agreement=0.95, gate=G.gate_for("sealed_v2"))  # no count asked, none needed, as when it was run
        self.assertEqual(v["status"], "pass")
        self.assertNotIn("labelled", v)
        self.assertEqual(G.GATE_V4["understanding"], G.GATE_V3["understanding"])  # the thresholds are v3's
        self.assertEqual(G.GATE_V4["options_fit"], G.GATE_V3["options_fit"])


if __name__ == "__main__":
    unittest.main()
