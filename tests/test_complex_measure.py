"""Package 24.2: the map roles' measurement. Twenty-eight written problems (14 dev, 14 sealed); the sealed set is checked by its fingerprint, read
only by its one run, and never run without the owner's yes and a euro cap; a case is scored on its size and the F1 of its families; the gate
map_gate_v1 was fixed before any run and fails on a size regression, on too many cases made worse, or on a run in which a request failed."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core.agents import ScriptedAdvisor
    from dclab_core.bands import FAMILIES
    from dclab_rnd.agent_eval import complex as cx
    from dclab_rnd.agent_eval.complex import __main__ as cli
except ImportError as error:  # pydantic is not in every environment
    MISSING = str(error)
else:
    MISSING = ""


def reply_for(case):
    """What a careful model would answer for a case: every part on the person's own words, one sub-problem per gold family, one sequence."""
    words = case["words"].replace("\n", " ")
    quote = words[:200]
    families = case["gold"]["families"]
    return {"parts": [{"title": f"Part {i}", "quote": quote} for i in range(1, len(families) + 1)],
            "sub_problems": [{"part": i, "family": f, "labels": "unknown", "predict_rank": i, "quote": quote} for i, f in enumerate(families, start=1)],
            "composition": {"method": "sequence", "steps": [{"kind": "sequence", "inputs": [f"sp{i}" for i in range(1, len(families) + 1)], "output": "all"}] if len(families) > 1 else []}}


@unittest.skipIf(MISSING, MISSING)
class SetsTest(unittest.TestCase):
    def test_both_sets_are_well_formed_and_the_same_size(self):
        for split in ("dev", "sealed"):
            cases = cx.load(split)
            self.assertEqual(len(cases), 14, split)
            self.assertEqual(len({c["id"] for c in cases}), 14, split)
            for case in cases:
                gold = case["gold"]
                self.assertTrue(set(gold["families"]) <= set(FAMILIES), case["id"])
                self.assertEqual(gold["size"], "single" if len(gold["families"]) == 1 else "complex", case["id"])
                self.assertTrue(case["words"].strip(), case["id"])

    def test_each_set_has_complex_problems_and_single_ones_that_only_share_words(self):
        for split in ("dev", "sealed"):
            sizes = [c["gold"]["size"] for c in cx.load(split)]
            self.assertEqual((sizes.count("complex"), sizes.count("single")), (8, 6), split)
            self.assertTrue(any(c.get("brief") for c in cx.load(split)), split)  # a brief with its own parts

    def test_the_sealed_file_is_the_one_that_was_sealed_and_an_edit_is_refused(self):
        self.assertEqual(cx.fingerprint(cx.SPLITS["sealed"]), cx.recorded_fingerprint())
        with tempfile.TemporaryDirectory() as tmp:
            edited = Path(tmp) / "problems_sealed.json"
            edited.write_text(cx.SPLITS["sealed"].read_text(encoding="utf-8").replace("Group our customers", "Cluster our customers"), encoding="utf-8")
            with mock.patch.dict(cx.SPLITS, {"sealed": edited}):
                with self.assertRaises(ValueError) as caught:
                    cx.load("sealed")
        self.assertIn("never edited", str(caught.exception))

    def test_nothing_but_the_measurement_reads_the_sealed_file(self):
        readers = []
        for base in ("dclab_rnd", "dclab_core", "scripts", "tests"):
            for path in (ROOT / base).rglob("*.py"):
                if path.name == "test_complex_measure.py":
                    continue
                if "problems_sealed" in path.read_text(encoding="utf-8", errors="ignore"):
                    readers.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(readers, ["dclab_rnd/agent_eval/complex/__init__.py"])

    def test_the_sealed_set_is_run_at_most_once(self):
        self.assertLessEqual(len(cx.sealed_runs()), 1)


@unittest.skipIf(MISSING, MISSING)
class ScoreTest(unittest.TestCase):
    def test_f1_of_two_multisets_of_families(self):
        self.assertEqual(cx.families_f1(["anomaly", "ranking"], ["ranking", "anomaly"]), 1.0)
        self.assertEqual(cx.families_f1(["anomaly"], ["ranking"]), 0.0)
        self.assertEqual(cx.families_f1(["classification", "classification"], ["classification"]), 0.6667)  # a repeated family counts as many
        self.assertEqual(cx.families_f1([], []), 1.0)
        self.assertEqual(cx.families_f1([], ["anomaly"]), 0.0)

    def test_a_case_is_scored_on_its_size_and_its_families(self):
        gold = {"size": "complex", "families": ["anomaly", "ranking"]}
        self.assertEqual(cx.score(["anomaly", "ranking"], gold), {"size": "complex", "size_ok": True, "f1": 1.0, "parts": 2})
        self.assertFalse(cx.score(["anomaly"], gold)["size_ok"])
        self.assertTrue(cx.score(["anomaly"], {"size": "single", "families": ["anomaly"]})["size_ok"])

    def test_the_rules_alone_read_the_dev_set_the_same_way_every_time(self):
        first = cx.measure(cx.load("dev"), None)
        again = cx.measure(cx.load("dev"), None)
        self.assertEqual(first["summary"], again["summary"])
        self.assertEqual(list(first["summary"]), ["rules"])
        self.assertEqual((first["requests"], first["failures"]), (0, []))
        self.assertEqual(first["summary"]["rules"]["cases"], 14)


@unittest.skipIf(MISSING, MISSING)
class ArmsTest(unittest.TestCase):
    def advisor(self, cases):
        return ScriptedAdvisor({"decomposer": lambda evidence: next(reply_for(c) for c in cases if c["words"].replace("\n", " ")[:1500] == evidence["words"].replace("\n", " ")[:1500])})

    def test_a_careful_model_scores_perfectly_through_the_role_and_through_the_plain_arm(self):
        cases = cx.load("dev")
        out = cx.measure(cases, self.advisor(cases))
        for arm in ("roles", "plain"):
            self.assertEqual((out["summary"][arm]["mean_f1"], out["summary"][arm]["size_right"], out["summary"][arm]["exact"]), (1.0, 14, 14), arm)
        self.assertTrue(all(r["sources"]["roles"] == "advisor" for r in out["rows"]))
        self.assertEqual(out["requests"], 14)  # one a case: the role and the plain arm read the same reply
        self.assertEqual(cx.verdict(out, "dev")["status"], "pass (dev: the gate is read on the sealed set)")
        self.assertEqual(cx.verdict(out, "dev")["plain_mean_f1"], 1.0)  # shown beside the gate, never part of it

    def test_the_plain_arm_takes_the_same_reply_the_role_read_and_costs_no_request(self):
        cases = cx.load("dev")
        out = cx.measure(cases, self.advisor(cases), plain=False)
        self.assertEqual((out["requests"], sorted(out["summary"])), (14, ["roles", "rules"]))  # left out of the scores, never sent twice

    def test_an_ungrounded_reply_is_taken_as_it_comes_by_the_plain_arm_and_refused_by_the_role(self):
        cases = cx.load("dev")
        loose = {"parts": [{"title": "Invented", "quote": "words nobody wrote"}], "sub_problems": [{"part": 1, "family": "anomaly", "quote": "words nobody wrote"}] * 3}
        out = cx.measure(cases, ScriptedAdvisor({"decomposer": loose}))
        self.assertEqual({tuple(r["values"]["plain"]) for r in out["rows"]}, {("anomaly", "anomaly", "anomaly")})  # no check: what it said
        self.assertEqual({r["sources"]["roles"] for r in out["rows"]}, {"fallback"})  # the role refused it, so the rules' map stands

    def test_an_ungrounded_model_is_refused_by_the_role_and_scores_as_the_rules(self):
        cases = cx.load("dev")
        invented = {"parts": [{"title": "Invented", "quote": "words nobody wrote"}], "sub_problems": [{"part": 1, "family": "anomaly", "quote": "words nobody wrote"}]}
        out = cx.measure(cases, ScriptedAdvisor({"decomposer": invented}), plain=False)
        self.assertEqual(out["summary"]["roles"]["mean_f1"], out["summary"]["rules"]["mean_f1"])  # refused whole: the rules stand
        self.assertEqual({r["sources"]["roles"] for r in out["rows"]}, {"fallback"})
        self.assertEqual(out["summary"]["roles"]["worse_than_rules"], [])

    def test_a_failed_request_is_counted_and_the_run_is_not_scored(self):
        class Fails:
            def ask(self, purpose, evidence):
                raise RuntimeError("provider down")

        out = cx.measure(cx.load("dev"), Fails())
        self.assertTrue(out["failures"])
        self.assertEqual(cx.verdict(out, "dev")["status"], "not scored")


@unittest.skipIf(MISSING, MISSING)
class GateTest(unittest.TestCase):
    def result(self, roles_f1=1.0, rules_f1=0.5, regressions=(), worse=(), failures=()):
        return {"summary": {"rules": {"cases": 14, "mean_f1": rules_f1, "size_right": 10, "exact": 7},
                            "roles": {"cases": 14, "mean_f1": roles_f1, "size_right": 13, "exact": 11, "size_regressions": list(regressions), "worse_than_rules": list(worse),
                                      "better_than_rules": []}}, "failures": list(failures)}

    def test_the_gate_passes_on_the_sealed_set_when_all_three_hold(self):
        self.assertEqual(cx.verdict(self.result(), "sealed")["status"], "pass")

    def test_each_check_can_fail_the_gate(self):
        self.assertEqual(cx.verdict(self.result(roles_f1=0.5), "sealed")["checks"]["better_f1"], False)  # not better than the rules
        self.assertEqual(cx.verdict(self.result(regressions=["CS-09"]), "sealed")["checks"]["no_size_regression"], False)
        self.assertEqual(cx.verdict(self.result(worse=["a", "b", "c"]), "sealed")["checks"]["few_worse"], False)
        self.assertEqual(cx.verdict(self.result(worse=["a", "b"]), "sealed")["checks"]["few_worse"], True)  # two is the limit
        for bad in (self.result(roles_f1=0.4), self.result(regressions=["x"]), self.result(worse=["a", "b", "c"])):
            self.assertEqual(cx.verdict(bad, "sealed")["status"], "fail")

    def test_on_dev_the_verdict_says_the_gate_is_read_on_the_sealed_set(self):
        self.assertIn("the gate is read on the sealed set", cx.verdict(self.result(), "dev")["status"])

    def test_a_failed_request_or_no_model_is_not_scored(self):
        self.assertEqual(cx.verdict(self.result(failures=["x"]), "sealed")["status"], "not scored")
        self.assertEqual(cx.verdict({"summary": {"rules": {}}, "failures": []}, "sealed")["status"], "not scored")

    def test_the_gate_was_fixed_before_the_runs_and_names_the_sealed_set(self):
        self.assertEqual((cx.GATE["version"], cx.GATE["split"]), ("map_gate_v1", "sealed"))


@unittest.skipIf(MISSING, MISSING)
class CommandLineTest(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict("os.environ", {"DCLAB_NO_LIVE_MODELS": "1"}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_without_yes_nothing_is_sent_and_the_plan_is_printed(self):
        code, out, _ = self.run_cli("--split", "dev")
        self.assertEqual(code, 0)
        self.assertIn("nothing was sent", out)
        self.assertIn("rules alone on dev", out)
        self.assertNotIn("rules alone on sealed", out)  # a sealed set's scores are read only in its one run

    def test_the_sealed_set_is_run_once_a_second_run_is_refused_before_anything_is_sent(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "MAP-001_sealed.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(cx, "RESULTS", Path(tmp)), mock.patch.object(cli, "sealed_runs", lambda: sorted(Path(tmp).glob("MAP-*_sealed.json"))):
                code, _, err = self.run_cli("--split", "sealed", "--yes", "--cap-eur", "0.5")
        self.assertEqual(code, 2)
        self.assertIn("run once", err)

    def test_a_run_with_no_model_is_refused(self):
        code, _, err = self.run_cli("--split", "dev", "--yes", "--cap-eur", "0.5")
        self.assertEqual(code, 2)
        self.assertIn("no model serves decomposer", err)

    def test_the_dev_run_that_was_made_is_stored_with_its_plan_cost_and_verdict(self):
        stored = sorted(cx.RESULTS.glob("MAP-*_dev.json"))
        self.assertTrue(stored)
        record = json.loads(stored[0].read_text(encoding="utf-8"))
        self.assertEqual((record["campaign_id"], record["split"], record["gate"]["version"]), ("map_roles_v1", "dev", "map_gate_v1"))
        self.assertLessEqual(record["spent_eur"], 0.5)
        self.assertLessEqual(record["plan"]["max_eur"], 0.5)
        self.assertIn("dev: the gate is read on the sealed set", record["verdict"]["status"])


if __name__ == "__main__":
    unittest.main()
