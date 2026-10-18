"""Package 28.6, the debate: two solution designers (the planner and a challenger who reads the first design) and a judge
who rules on what they dispute; every kept change passes the plan's code checks again; with no judge, or a refused
ruling, the first design stands (never worse than the single designer for want of a model); the record holds every seat."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd
except ImportError as error:  # the agent environment has no pandas
    MISSING = str(error)
else:
    MISSING = ""

GROUP = {"key": "split.strategy", "choice": {"kind": "group", "group_column": "customer_code", "holdout": "locked first, used once, after the choice"},
         "reason": "the same customer appears on many rows: their rows must stay on one side", "cites": ["F-keys"]}


@unittest.skipIf(MISSING, MISSING)
class DebateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_core.analyse import analyse
        from dclab_core.understand import understand

        rng = np.random.default_rng(0)
        n = 1200
        cust = rng.integers(0, 60, n)
        frame = pd.DataFrame({"customer_code": [f"C{c}" for c in cust], "amount": rng.gamma(2, 20, n).round(2),
                              "visits": rng.integers(1, 9, n), "churned": (rng.random(n) < 0.2 + 0.3 * (cust % 3 == 0)).astype(int)})
        cls.brief = understand("Predict which customers churn next month, when we score them each Monday.")
        cls.findings = analyse({"table": frame}, target="churned", task="binary", brief=cls.brief)

    def run_debate(self, a, b, judge):
        from dclab_core.agents import ScriptedAdvisor
        from dclab_core.debate import debate

        seat = lambda script: ScriptedAdvisor(script) if script is not None else None  # noqa: E731
        return debate(seat({"solution_planner": a}), seat({"solution_challenger": b}), seat({"debate_judge": judge}), self.brief, self.findings)

    def test_a_change_only_the_challenger_makes_is_kept_when_the_judge_keeps_it(self):
        out = self.run_debate({"decisions": []}, {"decisions": [GROUP]},
                              {"rulings": [{"key": "split.strategy", "keep": "b", "reason": "the customer repeats across rows (F-keys)", "cites": ["F-keys"]}]})
        split = out["plan"].decision("split.strategy").choice
        self.assertEqual((split["kind"], split["group_column"]), ("group", "customer_code"))
        rec = out["debate"]
        self.assertEqual(([c.key for c in rec.b], rec.agreed, [(r.key, r.keep, r.by) for r in rec.rulings]),
                         (["split.strategy"], [], [("split.strategy", "b", "judge")]))
        self.assertEqual(rec.sources["judge"], "advisor")
        self.assertEqual(out["single"]["kept"], [])  # the single designer's own plan, the baseline: unchanged
        from dclab_core.debate import Debate

        self.assertEqual(Debate.read(rec.model_dump_json()), rec)

    def test_with_no_judge_or_a_refused_ruling_the_first_design_stands(self):
        for judge in (None, {"rulings": [{"key": "split.strategy", "keep": "b", "reason": "trust me on this one", "cites": ["F-nothing"]}]}):
            out = self.run_debate({"decisions": []}, {"decisions": [GROUP]}, judge)
            self.assertEqual(out["plan"].decision("split.strategy").choice["kind"], "stratified")  # the rules' (the first designer kept it)
            self.assertEqual([(r.keep, r.by) for r in out["debate"].rulings], [("neither", "fallback")])

    def test_a_change_both_designers_make_is_kept_without_the_judge(self):
        from dclab_core.agents import ScriptedAdvisor
        from dclab_core.debate import debate

        judge = ScriptedAdvisor({"debate_judge": {"rulings": []}})
        out = debate(ScriptedAdvisor({"solution_planner": {"decisions": [GROUP]}}), ScriptedAdvisor({"solution_challenger": {"decisions": [GROUP]}}),
                     judge, self.brief, self.findings)
        self.assertEqual(out["debate"].agreed, ["split.strategy"])
        self.assertEqual(out["plan"].decision("split.strategy").choice["kind"], "group")
        self.assertEqual(judge.requests, [])  # nothing disputed: the judge is never asked (no spend)

    def test_the_challenger_sees_the_first_design_and_a_dispute_with_it_goes_to_the_judge(self):
        from dclab_core.agents import ScriptedAdvisor
        from dclab_core.debate import debate

        challenger = ScriptedAdvisor({"solution_challenger": {"decisions": [
            {"key": "split.strategy", "choice": {"kind": "random"}, "reason": "a random split is simpler for everyone", "cites": ["F-keys"]}]}})
        out = debate(ScriptedAdvisor({"solution_planner": {"decisions": [GROUP]}}), challenger, None, self.brief, self.findings)
        self.assertEqual(challenger.requests[0][1]["first_design"][0]["key"], "split.strategy")
        self.assertEqual(out["debate"].rulings[0].key, "split.strategy")  # the two designs differ: a dispute, not a silent overwrite
        self.assertEqual(out["plan"].decision("split.strategy").choice["kind"], "group")  # no judge: the first design stands

    def test_no_seat_can_undo_a_safety_decision_of_the_rules(self):
        from dclab_core.agents import ScriptedAdvisor
        from dclab_core.debate import debate
        from dclab_core.plan import plan

        base = plan(self.brief, self.findings)
        leak = {"key": "features.forbidden", "choice": [], "reason": "nothing needs keeping out of this table at all", "cites": ["F-keys"]}
        kept_out = base.decision("features.forbidden")
        if not (kept_out and kept_out.choice):  # this table keeps nothing out: the guard is shown on a plan that does
            from dclab_core.plan import override

            base = override(base, "features.forbidden", [{"column": "visits", "reason": "written after the moment"}], "kept out by the rules for this test", by="rule")
        from unittest import mock

        with mock.patch("dclab_core.plan.plan", return_value=base):
            out = debate(ScriptedAdvisor({"solution_planner": {"decisions": [leak]}}), ScriptedAdvisor({"solution_challenger": {"decisions": [leak]}}),
                         None, self.brief, self.findings)
        self.assertEqual([c["column"] for c in out["plan"].decision("features.forbidden").choice], ["visits"])  # never brought back
        self.assertEqual(out["debate"].agreed, [])  # both seats' removals were refused before they could agree

    def test_with_no_model_at_all_the_debate_is_the_rules_plan(self):
        from dclab_core.debate import debate
        from dclab_core.plan import plan

        out = debate(None, None, None, self.brief, self.findings)
        self.assertEqual(out["plan"].model_dump(), plan(self.brief, self.findings).model_dump())
        self.assertEqual(out["debate"].sources["judge"], "no dispute")


class ShapeTests(unittest.TestCase):
    def test_the_records_size_is_the_planners_decision_keys(self):  # debate.py reads plan only when used (the import cycle)
        from dclab_core.debate import MAX_CHANGES
        from dclab_core.plan import PLANNABLE

        self.assertEqual(MAX_CHANGES, len(PLANNABLE))


@unittest.skipIf(MISSING, MISSING)
class JourneyTests(unittest.TestCase):
    def test_a_judged_change_reaches_the_saved_solution_and_is_the_debates(self):
        import json as _json

        from dclab_core.agents import ScriptedAdvisor
        from dclab_rnd.agent_eval import journeys

        person = next(j for j in journeys.load("dev") if j["id"] == "JD-02")

        def challenger(evidence):  # keeps out a column the rules let in, citing the first measured finding
            ids = [f.split(" ")[0] for f in evidence["findings"]]
            kept = _json.loads(evidence["rule_plan"]["features.forbidden"]) if isinstance(evidence["rule_plan"]["features.forbidden"], str) else evidence["rule_plan"]["features.forbidden"]
            kept = [k if isinstance(k, dict) else {"column": k, "reason": "kept out by the rules"} for k in kept]  # restated in the rules' shape
            return {"decisions": [{"key": "features.forbidden", "choice": [*kept, {"column": "tenure_months", "reason": "a second look: kept out to be safe"}],
                                   "reason": "the challenger keeps this column out until its timing is confirmed", "cites": ids[:1]}]}
        ruling = lambda evidence: {"rulings": [{"key": d["key"], "keep": "b", "reason": "the second design is the safer one here",  # noqa: E731
                                                "cites": [evidence["findings"][0].split(" ")[0]]} for d in evidence["disputes"]]}
        live = {"designer": ScriptedAdvisor({"solution_planner": {"decisions": []}}), "challenger": ScriptedAdvisor({"solution_challenger": challenger}),
                "judge": ScriptedAdvisor({"debate_judge": ruling})}
        row = journeys.play(person, 0, None, live)  # JD-02's table holds tenure_months, a column the rules let in
        self.assertEqual(row["debate"]["sources"]["judge"], "advisor")
        self.assertEqual(row["debate"]["changed"], ["forbidden"])  # what reaches the solution, not the plan's wording
        self.assertEqual(row["refused_plans"], 0)


@unittest.skipIf(MISSING, MISSING)
class RunnerTests(unittest.TestCase):
    def test_the_debates_run_shares_the_designer_and_stores_only_a_whole_clean_run(self):
        import io
        import os
        from contextlib import redirect_stdout
        from unittest import mock

        from dclab_rnd.agent_eval import journeys
        from dclab_rnd.agent_eval.journeys import __main__ as cli

        seat = lambda spent: mock.MagicMock(failures=0, spent_eur=mock.MagicMock(return_value=spent), report=mock.MagicMock(return_value=[]))  # noqa: E731
        seats = (seat(0.3), seat(0.02), seat(0.01))

        def played(j, seed, routing, live, keep):
            self.assertIs(live["designer"], seats[0])  # one designer for both arms
            return {"id": j["id"], "kind": j["kind"], "family": j["family"], "parts": dict.fromkeys(journeys.PARTS, True), "right": True,
                    "why": [], "questions": 3, "asked": ["target"], "refused_plans": 0}
        for argv, stored in ((["--only", "JD-01"], 0), ([], 1)):
            with mock.patch.dict(os.environ, {}), mock.patch.object(cli, "_debate_seats", return_value=seats), mock.patch("dotenv.load_dotenv"), \
                    mock.patch.object(journeys, "play", side_effect=played), mock.patch("dclab_rnd.orchestra.patterns.store") as store, \
                    redirect_stdout(io.StringIO()):
                cli.main(["--pattern", "debate", "--yes", "--cap-eur", "1.0", *argv])
            self.assertEqual(store.call_count, stored, argv)
        against = store.call_args.args[3]
        self.assertEqual(against["spent_eur"], {"baseline": 0.3, "arm": 0.33})  # the designer's answer counted in both arms, once

    def test_a_shared_designer_is_asked_once_for_the_same_evidence(self):
        from unittest import mock

        from dclab_rnd.agent_eval.journeys.__main__ import _Shared

        inner = mock.MagicMock()
        inner.ask.return_value = {"decisions": []}
        shared = _Shared(inner)
        shared.ask("solution_planner", {"a": 1})
        shared.ask("solution_planner", {"a": 1})
        shared.ask("solution_planner", {"a": 2})
        self.assertEqual(inner.ask.call_count, 2)


if __name__ == "__main__":
    unittest.main()
