"""35.8 v2: the treated cases come from the intern's own mistakes in a memory-off pilot, the repeats are sized by a
simulation before the plan is frozen, the plan is frozen once, and the main run refuses a missing, changed or
unreachable plan; every session is kept whole with its thinking."""

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval import memory_value as v1  # noqa: E402
from dclab_rnd.agent_eval import memory_value_v2 as v2  # noqa: E402
from dclab_rnd.models.gateway import Gateway  # noqa: E402
from dclab_rnd.models.usage import FileUsage  # noqa: E402

KEYED = {"OPENAI_API_KEY": "k", "DCLAB_NO_LIVE_MODELS": "1", **{f"DCLAB_TIER_{t}_{n}": "" for t in ("STRONG", "STANDARD", "CHEAP") for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")}}
THOUGHT = "I check the brief's corrections before I forbid a column."


def reply(text="", calls=()):
    calls = [{"id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(calls)]
    return {"content": text, "tool_calls": calls, "usage": {"input_tokens": 100, "output_tokens": 20}, "thinking": [{"type": "thinking", "thinking": THOUGHT}],
            "assistant_message": {"role": "assistant", "content": text, "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]}}


class Follower:
    """A scripted model: accepts the proposal's forbidden columns, except a column its brief's correction names."""

    def complete(self, messages, tools=None, max_tokens=None, **_):
        task = messages[1]["content"]
        pid = re.search(r"existing project ([0-9a-f]{12})", task).group(1)
        last = messages[-1]
        if last["role"] != "tool":
            return reply(calls=[("propose_solution", {"project_id": pid, "target": "target"})])
        result = json.loads(last["content"])
        if "forbidden" in result and "task" in result:
            corrected = " ".join(re.findall(r"\(correction\)([^\n]*)", task))
            forbid = [{"column": f["column"], "reason": f["reason"]} for f in result["forbidden"] if f["column"] not in corrected]
            return reply(calls=[("set_solution", {"project_id": pid, "target": "target", "task": result["task"], "forbidden": forbid,
                                                  "identifiers": [i for i in result.get("identifiers") or [] if i not in corrected],
                                                  "prediction_moment": "At the monthly snapshot."})])
        return reply(calls=[("finish", {"report": "Solution saved."})])


def gateway():
    return Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=lambda tier, purpose: Follower())


FULL = v2.manifest()


def small(**over):
    m = json.loads(json.dumps(FULL))
    m["pool"] = [s for s in m["pool"] if s["case"] in ("AE-16", "MEM-10", "MEM-03")]
    m["controls"] = [s for s in m["controls"] if s["case"] == "AE-01"]
    m["pilot"]["repeats"] = 1
    m["selection"].update(min_scored=1)
    m["sizing"].update(min_cases=1, repeats_options=[1], min_power=0.0, draws=50)
    m["gate"].update(min_pairs_per_case=1)
    for k, v in over.items():
        m[k] = {**m[k], **v}
    return m


class PoolTest(unittest.TestCase):
    def test_every_candidate_is_the_case_it_was_written_for(self):
        for s in (*FULL["pool"], *FULL["controls"]):
            self.assertEqual(v2.case_of(s["case"]).fingerprint(), s["fingerprint"], s["case"])

    def test_a_correction_only_ever_keeps_an_honest_column(self):
        for s in FULL["pool"]:
            case = v2.case_of(s["case"])
            self.assertEqual(s["expect"], "keep")
            self.assertTrue(set(s["columns"]) <= set(case.innocent), s["case"])
            self.assertFalse(set(s["columns"]) & set(case.leaks), s["case"])

    def test_the_pilot_is_memory_off_only(self):
        plan = v2.pilot_schedule()
        self.assertEqual(len(plan), len(FULL["pool"]) * FULL["pilot"]["repeats"])
        self.assertEqual({i["arm"] for i in plan}, {"off"})


def prow(case, r, against, ran=True, column=None):
    column = column or v2.scenario_of(case, FULL)["columns"][0]
    return {"id": f"P-{case}-r{r}-off", "case": case, "repeat": r, "arm": "off", "ran": ran, "against_memory": against,
            "against": [column] if against else []}


class ChooseTest(unittest.TestCase):
    def test_wrong_two_times_in_three_is_treated_the_most_often_wrong_first(self):
        rows = {}
        for r in range(3):
            for c, wrong in (("MEM-01", r < 2), ("MEM-02", True), ("MEM-03", r < 1)):
                rows[f"{c}-r{r}"] = prow(c, r, wrong)
        rows["MEM-04-r0"] = prow("MEM-04", 0, True)  # only one scored: too few to choose
        rows["MEM-04-r1"] = prow("MEM-04", 1, True, ran=False)
        for r in range(3):  # wrong on a base column only: its correction does not address that, so it is not chosen
            rows[f"MEM-05-r{r}"] = prow("MEM-05", r, True, column="tenure_months")
        self.assertEqual(v2.select(rows)["treated"], ["MEM-02", "MEM-01"])

    def test_too_few_cases_can_never_pass_and_nothing_runs(self):
        choice = {"tally": {c: {"scored": 3, "against": 3} for c in ("A", "B", "C")}, "treated": ["A", "B", "C"]}
        sized = v2.size(choice)
        self.assertFalse(sized["runnable"])
        self.assertIn("at least 5", sized["why_not"])

    def test_enough_mistakes_give_power_and_the_fewest_repeats(self):
        names = [f"C{i}" for i in range(8)]
        choice = {"tally": {c: {"scored": 3, "against": 3} for c in names}, "treated": names}
        sized = v2.size(choice)
        self.assertTrue(sized["runnable"])
        self.assertEqual(sized["repeats"], 3)
        self.assertGreaterEqual(sized["power_by_repeats"][3], 0.8)
        weak = {"tally": {c: {"scored": 3, "against": 2} for c in names[:5]}, "treated": names[:5]}
        self.assertLess(v2.size(weak)["power_by_repeats"][3], v2.size(choice)["power_by_repeats"][3])

    def test_the_fast_sign_flip_agrees_with_v1(self):
        import itertools

        import numpy as np

        means = np.array([[-1, -1, -1, -1, -1, 0, 0, 0], [-1, -0.5, 0, 0.5, -1, 0, -1, 0]], dtype=float)
        signs = np.array(list(itertools.product((1, -1), repeat=8)), dtype=float)
        fast = v2._sign_flip_p_many(means, signs)
        self.assertAlmostEqual(fast[0], v1.sign_flip_p(list(means[0])))
        self.assertAlmostEqual(fast[1], v1.sign_flip_p(list(means[1])), places=4)


class FlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.m = small()
        for p in (mock.patch.object(v2, "manifest", lambda: self.m), mock.patch.object(v2, "FROZEN", self.tmp / "frozen.json"), mock.patch.object(v2, "RUNS", self.tmp / "runs"),  # never the real pilots
                  mock.patch.dict(os.environ, KEYED)):
            p.start()
            self.addCleanup(p.stop)

    def test_pilot_freeze_run_and_judge_with_thinking_kept(self):
        pilot = self.tmp / "PILOT-t"
        out = v2.run_pilot(gateway(), pilot, None, 200, 8, log=lambda *a: None)
        self.assertEqual((len(out["ran"]), out["left"]), (3, 0))
        rows = v1.done(pilot)
        self.assertEqual(rows["P-AE-16-r0-off"]["against"], ["post_signup_survey"])
        self.assertEqual(rows["P-MEM-03-r0-off"]["against"], [])  # the rules leave it, so this model keeps it: not a mistake
        f = v2.freeze(pilot, at="2026-10-28T00:00:00+00:00")
        self.assertEqual(f["treated"], ["MEM-10", "AE-16"])  # pool order between equals
        with self.assertRaises(SystemExit):
            v2.freeze(pilot)  # frozen once
        main = self.tmp / "MV2-t"
        out = v2.run_main(gateway(), main, None, 200, 8, log=lambda *a: None)
        self.assertEqual(out["left"], 0)
        rows = v1.done(main)
        on, off = rows["MEM-10-r0-on"], rows["MEM-10-r0-off"]
        self.assertEqual((on["against"], off["against"]), ([], ["usage_first_week_after_signup"]))
        self.assertTrue(on["leak_caught"] and off["leak_caught"])  # memory kept the honest column, never the leak beside it
        steps = on["session"]["steps"]
        self.assertTrue(steps and all(s.get("thought") == THOUGHT and s.get("thought_source") == "recorded" for s in steps))
        s = v2.summarize(main)
        self.assertEqual(s["arms"]["treated"]["on"]["against_memory"]["share"], 0.0)
        self.assertEqual(s["arms"]["treated"]["off"]["against_memory"]["share"], 1.0)
        self.assertEqual(s["frozen"]["treated"], ["MEM-10", "AE-16"])
        results = self.tmp / "results"
        with mock.patch.object(v2, "RESULTS", results):
            path = v2.store(main, s)
        self.assertTrue(path.name.startswith("MV2-001"))
        self.assertEqual(json.loads(path.read_text())["frozen"]["treated"], ["MEM-10", "AE-16"])

    def test_the_main_run_refuses_a_missing_changed_or_unreachable_plan(self):
        with self.assertRaises(SystemExit):
            v2.run_main(gateway(), self.tmp / "MV2-a", None, 200, 8, log=lambda *a: None)  # nothing frozen
        v2.FROZEN.write_text(json.dumps({"manifest_sha256": "0" * 64, "runnable": True, "treated": [], "repeats": 1}))
        with self.assertRaises(SystemExit):
            v2.run_main(gateway(), self.tmp / "MV2-b", None, 200, 8, log=lambda *a: None)  # the manifest moved
        v2.FROZEN.write_text(json.dumps({"manifest_sha256": v2.manifest_sha(), "runnable": False, "why_not": "too few", "treated": [], "repeats": None}))
        with self.assertRaises(SystemExit):
            v2.run_main(gateway(), self.tmp / "MV2-c", None, 200, 8, log=lambda *a: None)  # the gate cannot be reached

    def test_freezing_refuses_another_manifest_a_second_pilot_and_the_main_run_another_model(self):
        pilot = self.tmp / "PILOT-a"
        v2.run_pilot(gateway(), pilot, None, 200, 8, log=lambda *a: None)
        meta = json.loads((pilot / "run.json").read_text())
        (pilot / "run.json").write_text(json.dumps({**meta, "manifest_sha256": "0" * 64}))  # rules changed after the pilot
        with self.assertRaises(SystemExit):
            v2.freeze(pilot)
        (pilot / "run.json").write_text(json.dumps(meta))
        other = self.tmp / "try2"  # any name: its run.json says it is a pilot
        other.mkdir()
        (other / "P-AE-16-r0-off.json").write_text(json.dumps(prow("AE-16", 0, False)))
        (other / "run.json").write_text(json.dumps({"campaign": v2.CAMPAIGN + ":pilot"}))
        with self.assertRaises(SystemExit):
            v2.freeze(pilot)  # never the best of several pilots
        (other / "P-AE-16-r0-off.json").unlink()
        f = v2.freeze(pilot)
        self.assertIn("prompts", f["pilot_setup"])
        v2.FROZEN.write_text(json.dumps({**f, "pilot_setup": {**f["pilot_setup"], "model": "another-model"}}))
        with self.assertRaises(SystemExit):
            v2.run_main(gateway(), self.tmp / "MV2-m", None, 200, 8, log=lambda *a: None)
        self.assertFalse((self.tmp / "MV2-m").exists())  # refused before anything was sent or written

    def test_freezing_waits_for_the_whole_pilot(self):
        pilot = self.tmp / "PILOT-part"
        pilot.mkdir()
        (pilot / "P-AE-16-r0-off.json").write_text(json.dumps(prow("AE-16", 0, True)))
        with self.assertRaises(SystemExit):
            v2.freeze(pilot)
        self.assertFalse(v2.FROZEN.exists())

    def write_run(self, f, lose=lambda i: False):
        v2.FROZEN.write_text(json.dumps(f))
        folder = self.tmp / "MV2-g"
        folder.mkdir()
        (folder / "run.json").write_text(json.dumps({"campaign": v2.CAMPAIGN, "frozen_sha256": v2.frozen_sha(), "manifest_sha256": v2.manifest_sha()}))
        for i in v2.main_schedule(f):
            leak = None if i["case"] in ("AE-16",) else not lose(i)
            r = {"id": i["id"], "case": i["case"], "repeat": i["repeat"], "arm": i["arm"], "ran": True, "solution_saved": True,
                 "against_memory": i["arm"] == "off" and i["role"] == "treated", "false_alarm": False, "tokens": [100, 10], "eur": 0.01,
                 "leak_caught": leak}
            (folder / f"{i['id']}.json").write_text(json.dumps(r))
        return folder

    def test_one_case_losing_its_leak_fails_even_when_the_pooled_guard_passes(self):
        self.m["controls"] = FULL["controls"]  # 4 controls: one lost session in 15 is hidden by the pooled share
        f = {"treated": ["MEM-10", "AE-16"], "repeats": 3, "manifest_sha256": v2.manifest_sha(), "runnable": True, "power_by_repeats": {}, "pilot_run": "p"}
        folder = self.write_run(f, lose=lambda i: i["arm"] == "on" and i["case"] == "MEM-10" and i["repeat"] == 0)
        s = v2.summarize(folder)
        self.assertTrue(s["checks"]["controls_keep_their_traps"])
        self.assertFalse(s["checks"]["treated_leaks_kept_each_case"])
        self.assertEqual(s["verdict"], "memory_does_not_help")

    def test_a_plan_edited_after_the_run_is_refused_at_the_verdict(self):
        f = {"treated": ["MEM-10", "AE-16"], "repeats": 1, "manifest_sha256": v2.manifest_sha(), "runnable": True, "power_by_repeats": {}, "pilot_run": "p"}
        folder = self.write_run(f)
        self.assertIn("verdict", v2.summarize(folder))
        v2.FROZEN.write_text(json.dumps({**f, "treated": ["MEM-10"]}))  # a case dropped after its result was seen
        with self.assertRaises(SystemExit):
            v2.summarize(folder)

if __name__ == "__main__":
    unittest.main()
