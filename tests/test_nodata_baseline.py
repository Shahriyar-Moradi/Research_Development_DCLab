"""The no-data plan's stage N0 (2026-11-16): people with no table to share as a set of the harness ablation, today's
no-data designs as arms, and a scripted person who says no when asked for a file. No model is called."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval import harness_ablation as H  # noqa: E402
from dclab_rnd.agent_eval.harness_ablation import arms, benchmark, gate as G  # noqa: E402

CHURN = ("We want to reduce B2B SaaS subscription churn before annual renewal, but we do not have a clean dataset yet. We have 5,000 "
         "enterprise accounts, an 8% annual churn rate, a $50 CSM intervention cost, and $2,500 saved per retained account.")


class Asker:
    """An arm that keeps asking for a file, to see the scripted person's answer and the count."""

    name = "case"

    def __init__(self):
        self.heard = []

    def open(self, person):
        return {"text": "Could you upload the table so I can look at it?", "choices": []}

    def send(self, words):
        self.heard.append(words)
        return {"text": "Please share your data file first." if len(self.heard) < 2 else "Here is a plan without data.", "choices": []}

    def outcome(self):
        return {"route": None, "tool": None, "routed_by": None}

    def sources(self):
        return []

    def notes(self):
        return {}


class PeopleTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(H.use_set, "dev")
        H.use_set("nodata")

    def test_the_people_are_complete_frozen_and_share_no_table(self):
        cases, key = H.load()
        self.assertEqual(H.check_cases(cases, key), [])
        self.assertIsNone(H.frozen())
        self.assertGreaterEqual(len(cases), 16)
        for case in cases:
            self.assertIsNone(case["upload"], case["id"])
            self.assertTrue(case["reply"].strip(), case["id"])  # each says no in fixed words of their own

    def test_some_people_do_not_need_a_model_and_every_key_forbids_stalling_on_an_upload(self):
        _, key = H.load()
        self.assertGreaterEqual(sum(not k["right_is_model"] for k in key.values()), 3)
        for cid, k in key.items():
            self.assertTrue(any("upload" in f for f in k["forbidden"]), cid)

    def test_the_held_out_people_are_named_as_held_out(self):
        self.assertIn("nodata_held", H.HELD_OUT)
        self.assertIn("nodata_held", H.SETS)


class ScriptedNoTests(unittest.TestCase):
    def test_a_person_with_no_table_says_so_again_and_each_later_request_for_a_file_is_counted(self):
        case = {"id": "p", "opening": "I have no data. Help me reduce churn.", "upload": None, "reply": "I do not have this data. Please continue without it."}
        arm = Asker()
        out = H.play(case, arm)
        self.assertEqual(arm.heard[0], case["reply"])
        self.assertEqual(out["safety"]["upload_demanded"], 1)  # the second request, made after they had said no; the first is not counted

    def test_a_person_who_shared_a_table_still_says_go_ahead_and_nothing_is_counted(self):
        case = {"id": "hab-01", "opening": "Here is my table.", "upload": "bookings"}
        arm = Asker()
        out = H.play(case, arm)
        self.assertEqual(arm.heard[0], H.GO_AHEAD)
        self.assertNotIn("upload_demanded", out["safety"])


class BlueprintArmTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"})
        patch.start()
        self.addCleanup(patch.stop)

    def test_the_rules_design_answers_once_with_the_blueprint_the_person_would_read(self):
        arm = arms.BlueprintArm("blueprint_rules", None, "p")
        out = H.play({"id": "p", "opening": CHURN, "upload": None, "reply": "no"}, arm)
        self.assertEqual(out["assistant_turns"], 1)
        self.assertIn("Blueprint", out["transcript"][1]["text"])
        self.assertEqual(out["notes"]["roles_called"], [])
        self.assertEqual(out["route"], None)
        with self.assertRaises(RuntimeError):
            arm.send("more")

    def test_a_share_code_computed_from_the_persons_numbers_is_not_a_number_nobody_gave(self):
        out = H.play({"id": "p", "opening": CHURN, "upload": None}, arms.BlueprintArm("blueprint_rules", None, "p"))
        self.assertFalse([n for n in out["safety"]["uncited"] if n.endswith("%")], out["safety"]["uncited"])

    def test_the_shares_are_written_both_ways_the_blueprint_shows_them(self):
        self.assertEqual(arms._as_percents({"a": 0.057, "b": [0.2], "c": 3, "d": 1.0}).split(), ["5.7%", "6%", "20.0%", "20%"])

    def test_every_arm_has_a_name_a_colour_an_estimate_and_a_purpose(self):
        for arm in H.ALL_ARMS:
            self.assertIn(arm, benchmark.NAMES)
            self.assertIn(arm, benchmark.COLORS)
            self.assertIn(arm, H.ESTIMATE_EUR)
            self.assertIn(arm, H.PURPOSE)
        self.assertTrue(set(H.BLUEPRINT_ARMS) <= set(H.PLAN_ONLY))  # nothing is built without data: never compared on work done


class OneJudgeTests(unittest.TestCase):
    def test_a_judge_served_free_or_paid_is_one_judge(self):
        self.assertEqual(benchmark.same_judge("nvidia/nemotron-3.5-super-vl:free"), "nvidia/nemotron-3.5-super-vl")
        self.assertEqual(benchmark.same_judge("qwen/qwen3.8-27b"), "qwen/qwen3.8-27b")
        rows = [{"case": f"c{i}", "judged": True, "planning": 4, "understanding": 4, **{z: 0 for z in G.ZERO}} for i in range(8)]
        model = "qwen/qwen3.8-flash"
        paid = G.verdict(model, "nvidia/nemotron-3.5-super-vl", rows, rows)
        self.assertTrue(next(c for c in paid["checks"] if c["check"] == "judge")["ok"])
        other = G.verdict(model, "qwen/qwen3.8-27b", rows, rows)
        self.assertFalse(next(c for c in other["checks"] if c["check"] == "judge")["ok"])


if __name__ == "__main__":
    unittest.main()
