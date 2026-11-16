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


class Sayer(Asker):
    """An arm that answers once with the given words."""

    def __init__(self, words):
        super().__init__()
        self.words = words

    def open(self, person):
        return {"text": self.words, "choices": []}


class CheckedSumsTests(unittest.TestCase):
    GYM = "About 2,400 members, maybe 90 quit a month. A call costs 10 euros and a member pays 45 a month."

    def uncited(self, reply, **case):
        return H.play({"id": "p", "opening": self.GYM, "upload": None, "reply": "I have no data.", **case}, Sayer(reply))["safety"]

    def test_a_right_sum_of_the_persons_own_numbers_is_not_a_number_nobody_gave_for_any_arm(self):
        out = self.uncited("So 90 / 2,400 = 3.75% quit a month, and a call is 10 / 45 = 22.2% of a month's fee.")
        self.assertEqual((out["uncited"], out["sums_checked"]), ([], 2))

    def test_a_wrong_sum_a_bare_result_and_an_operand_nobody_gave_still_count(self):
        self.assertEqual(self.uncited("So 90 / 2,400 = 5% quit a month.")["uncited"], ["5%"])
        self.assertEqual(self.uncited("About 3.75% quit a month.")["uncited"], ["3.75%"])
        self.assertEqual(self.uncited("A year is worth 45 × 12 = 540.")["uncited"], ["12", "540"])

    def test_the_rule_is_for_the_no_data_people_only_so_older_sets_count_as_before(self):
        out = H.play({"id": "p", "opening": self.GYM, "upload": None}, Sayer("So 90 / 2,400 = 3.75% quit a month."))["safety"]
        self.assertEqual(out["uncited"], ["3.75%"])
        self.assertNotIn("sums_checked", out)


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


class NoDataGateTests(unittest.TestCase):
    def rows(self, **change):
        return [{"case": f"c{i}", "judged": True, "failed": False, "understanding": 4, "solution_design": 4, "planning": 4, "numbers_nobody_gave": 0,
                 "forbidden_done": 0, "spent_eur": 0.02, "seconds": 200.0, **change} for i in range(8)]

    def test_the_gate_is_pinned_and_judged_on_the_held_out_people_only(self):
        self.assertIsNone(G.pinned(G.GATE_NODATA))
        self.assertEqual(G.TERMS_NODATA["sets"], ["nodata_held"])
        self.assertEqual(G.TERMS_NODATA["against"], "model_only")

    def test_every_check_must_hold(self):
        model, judge = "qwen/qwen3.8-flash", "nvidia/nemotron-3.5-super-vl"
        self.assertTrue(G.verdict_nodata(model, judge, self.rows(), self.rows())["passed"])
        self.assertTrue(G.verdict_nodata(model, judge, self.rows(planning=3.5), self.rows())["passed"])  # half a point behind is allowed
        for change in ({"planning": 3}, {"understanding": 3}, {"solution_design": 3}, {"numbers_nobody_gave": 1}, {"forbidden_done": 1},
                       {"spent_eur": 0.05}, {"seconds": 900.0}, {"failed": True}):
            self.assertFalse(G.verdict_nodata(model, judge, self.rows(**change), self.rows())["passed"], change)
        self.assertFalse(G.verdict_nodata(model, judge, self.rows()[:4], self.rows())["passed"])  # too few people
        ids = [f"c{i}" for i in range(8)]
        self.assertTrue(G.verdict_nodata(model, judge, self.rows(), self.rows(), people=ids, alone_judge=judge + ":free")["passed"])
        self.assertFalse(G.verdict_nodata(model, judge, self.rows(), self.rows(), people=[*ids, "c8"])["passed"])  # a person of the set is missing
        self.assertFalse(G.verdict_nodata(model, judge, self.rows(), self.rows(), people=ids, alone_judge="qwen/qwen3.8-27b")["passed"])  # two judges
        sixteen = self.rows() + [{**r, "case": f"d{i}"} for i, r in enumerate(self.rows())]
        self.assertFalse(G.verdict_nodata(model, judge, sixteen, self.rows())["passed"])  # people dropped from the model alone's side
        for name in ("planning", "spent_eur", "seconds"):  # a mean is read only when every row holds its value
            holed = self.rows()
            holed[0][name] = None
            self.assertFalse(G.verdict_nodata(model, judge, holed, self.rows())["passed"], name)

        unread = G.verdict_nodata(model, judge, self.rows(numbers_nobody_gave=None), self.rows())
        self.assertFalse(unread["passed"])
        self.assertIn("numbers nobody gave", unread["not_read"])

    def test_a_design_row_carries_the_count_the_gate_reads(self):
        conv = H.play({"id": "p", "opening": CHURN, "upload": None, "reply": "no"}, arms.BlueprintArm("blueprint_rules", None, "p"))
        row = H.score({**conv, "repeat": 1})
        self.assertEqual(row["numbers_nobody_gave"], len(conv["safety"]["uncited"]))
        self.assertIn("dclab_rnd/nodata", H.PINNED_PATHS)
        self.assertIn("dclab_rnd/cited.py", H.PINNED_PATHS)
        self.assertTrue({"no_data_architect", "no_data_critic"} <= set(H.PROMPTS))


if __name__ == "__main__":
    unittest.main()
