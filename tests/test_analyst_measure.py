"""Package 18.2: the problem analyst's measurement (dclab_rnd/agent_eval/analyst) and its gateway advisor.

No model is called: advisors here are scripted. The live run needs the owner's yes and a euro cap.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pydantic  # noqa: F401
except ImportError as error:
    PYDANTIC_MISSING = str(error)
else:
    PYDANTIC_MISSING = ""


class Scripted:
    """Answers by case: the description's first words pick the answer."""

    def __init__(self, answers, fail=False):
        self.answers, self.fail = answers, fail

    def ask(self, purpose, evidence):
        if self.fail:
            raise TimeoutError("down")
        text = " ".join(evidence["description_paragraphs"])
        return next((a for start, a in self.answers.items() if text.startswith(start)), {})


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class AnalystMeasureTest(unittest.TestCase):
    def test_the_sets_are_complete_and_the_sealed_one_is_sealed(self):
        from dclab_rnd.agent_eval import analyst

        for split in ("dev", "sealed"):
            cases = analyst.load(split)
            self.assertEqual(len(cases), 12)
            for case in cases:
                self.assertEqual(set(case["gold"]), set(analyst.FIELDS), case["id"])
                columns = {c for cols in case["files"].values() for c in cols}
                target = case["gold"]["target_column"]
                self.assertTrue(target is None or target in columns, case["id"])
        self.assertEqual(len({c["gold"]["task"] for c in analyst.load("sealed")} - {None}), 3)  # every kind
        with tempfile.TemporaryDirectory() as tmp:
            edited = Path(tmp) / "briefs_sealed.json"
            rows = json.loads(analyst.SPLITS["sealed"].read_text(encoding="utf-8"))
            rows[0]["gold"]["task"] = "regression"
            edited.write_text(json.dumps(rows), encoding="utf-8")
            with mock.patch.dict(analyst.SPLITS, {"sealed": edited}), self.assertRaises(ValueError):
                analyst.load("sealed")

    def test_scoring_rewards_abstaining_and_punishes_a_moment_inside_the_outcome(self):
        from dclab_rnd.agent_eval.analyst import score

        moment = {"any": ["loads"], "none": ["checkout", "paid"]}
        self.assertEqual(score("when the cart page loads", moment), "right")
        self.assertEqual(score("at checkout", moment), "wrong")
        self.assertEqual(score(None, moment), "missed")
        self.assertEqual(score(None, None), "abstained")
        self.assertEqual(score("anything", None), "wrong")
        self.assertEqual(score("binary", "binary"), "right")
        self.assertEqual(score("6 pm each day", {"any": ["6 pm"]}), "right")
        self.assertEqual(score("pm", {"any": ["mae"]}), "wrong")  # short words match whole
        self.assertEqual(score("customers sees it", {"any": ["see"]}), "right")  # or as a plural
        self.assertEqual(score("At checkout, before delivery", {"any": ["checkout"], "none": ["delivered", "delivery"]}, "prediction_moment"), "right")
        self.assertEqual(score("whether the loan is not repaid", {"any": ["repaid"]}, "goal"), "right")  # only a moment drops "not yet" clauses
        review = {"any": ["review"], "none": ["exit", "next review"]}
        self.assertEqual(score("before the next review", review, "prediction_moment"), "wrong")  # the end of the outcome window, not a moment
        self.assertEqual(score("At each quarterly review, before the next review", review, "prediction_moment"), "right")
        self.assertEqual(score("`Binary`", "binary"), "right")  # formatting is not judgement

    def test_an_accepted_wrong_proposal_fails_the_gate(self):
        from dclab_rnd.agent_eval.analyst import load, measure, verdict

        cases = [c for c in load("dev") if c["id"] in ("d09", "d04")]
        advisor = Scripted({
            "An online shop": {"prediction_moment": {"value": "at checkout", "quote": "a customer will pick at checkout"}},
            "Customers see": {"task": {"value": "regression", "quote": "Estimate how many minutes each order will take"}},
        })
        result = measure(cases, advisor)
        analyst = result["summary"]["analyst"]
        self.assertIsNone(result["rows"][1]["values"]["rules"]["prediction_moment"])  # the rules leave it unknown: the error is new
        self.assertIn("d09.prediction_moment", analyst["new_wrong"])
        self.assertFalse(verdict(result, "sealed")["checks"]["no_new_wrong"])
        self.assertEqual(analyst["rules_changed"], [])
        self.assertIn("plain", result["summary"])
        self.assertEqual(result["requests"], 2 * len(cases) - sum(  # one analyst request per case with unknown fields, one plain each
            1 for r in result["rows"] if all(v is not None for v in r["values"]["rules"].values())))

    def test_a_correct_analyst_passes_the_gate(self):
        from dclab_rnd.agent_eval.analyst import load, measure, verdict

        cases = [c for c in load("dev") if c["id"] in ("d09", "d04")]
        advisor = Scripted({
            "An online shop": {"prediction_moment": {"value": "when the cart page loads, before checkout",
                                                     "quote": "The prediction is made when the cart page loads"}},
            "Customers see": {"prediction_moment": {"value": "the moment they check out", "quote": "the moment they check out"}},
        })
        result = measure(cases, advisor)
        self.assertEqual(result["summary"]["analyst"]["new_wrong"], [])
        self.assertEqual(verdict(result, "sealed")["status"], "pass")

    def test_a_failed_request_is_never_scored_and_the_rules_alone_are_not_a_verdict(self):
        from dclab_rnd.agent_eval.analyst import load, measure, verdict

        cases = load("dev")[:2]
        failed = measure(cases, Scripted({}, fail=True))
        self.assertTrue(failed["failures"])
        self.assertEqual(verdict(failed, "sealed")["status"], "not scored")
        self.assertEqual(verdict(measure(cases, None), "sealed")["status"], "not scored")

    def test_the_sealed_set_runs_once_and_nothing_is_sent_without_yes(self):
        from dclab_rnd.agent_eval.analyst import __main__ as cli

        with mock.patch("dotenv.load_dotenv"), mock.patch.object(cli, "sealed_runs", return_value=[Path("ANA-001_sealed.json")]), \
                mock.patch("dclab_rnd.models.gateway.for_workspace") as gateway:
            self.assertEqual(cli.main(["--split", "sealed", "--yes", "--cap-eur", "1"]), 2)
        gateway.assert_not_called()
        with mock.patch("dotenv.load_dotenv"), mock.patch.object(cli, "plan", return_value={"available": False}), \
                mock.patch("dclab_rnd.models.gateway.for_workspace"), \
                mock.patch.object(cli, "measure", wraps=cli.measure) as measured:
            self.assertEqual(cli.main(["--split", "dev"]), 0)
        measured.assert_called_once()  # the rules alone; the plan is printed, nothing sent


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class AnalystCliTest(unittest.TestCase):
    """The refusals before anything is sent, and a run cut short that never uses up the sealed set."""

    def run_cli(self, plan, argv, advisor_replies=None):
        from dclab_rnd.agent_eval.analyst import __main__ as cli

        tmp = Path(tempfile.mkdtemp())
        with mock.patch("dotenv.load_dotenv"), mock.patch("dclab_rnd.models.gateway.for_workspace"), \
                mock.patch.object(cli, "plan", return_value=plan), mock.patch.object(cli, "RESULTS", tmp), \
                mock.patch("dclab_rnd.agent_eval.analyst.RESULTS", tmp), \
                mock.patch("dclab_rnd.core_advisor.GatewayAdvisor.ask", side_effect=advisor_replies), \
                mock.patch("dclab_rnd.core_advisor.GatewayAdvisor.spent_eur", return_value=0.001):
            return cli.main(argv), tmp

    def test_it_refuses_before_sending(self):
        priced = {"available": True, "local": False, "max_eur": 0.02}
        cases = {"no model": ({**priced, "available": False}, ["--cap-eur", "1"]),
                 "unpriced": ({**priced, "max_eur": None}, ["--cap-eur", "1"]),
                 "no cap": (priced, []), "zero cap": (priced, ["--cap-eur", "0"]),
                 "above the cap": (priced, ["--cap-eur", "0.01"])}
        for name, (plan, extra) in cases.items():
            with self.subTest(case=name):
                code, folder = self.run_cli(plan, ["--split", "dev", "--yes", *extra], advisor_replies=AssertionError("sent"))
                self.assertEqual(code, 2)
                self.assertEqual(list(folder.iterdir()), [])

    def test_a_run_cut_short_is_stored_unscored_and_the_sealed_set_stays_unrun(self):
        from dclab_rnd.agent_eval import analyst

        code, folder = self.run_cli({"available": True, "local": False, "max_eur": 0.02}, ["--split", "sealed", "--yes", "--cap-eur", "1"],
                                    advisor_replies=RuntimeError("the reply was cut at the token limit"))
        self.assertEqual(code, 0)
        [stored] = list(folder.iterdir())
        self.assertTrue(stored.name.endswith("_sealed_unscored.json"))
        record = json.loads(stored.read_text())
        self.assertEqual(record["verdict"]["status"], "not scored")
        self.assertEqual((record["experiment_id"], record["status"]), (record["id"], "failed"))  # the campaign record's keys
        self.assertNotIn("rows", record)
        self.assertEqual(record["requests"], 1)  # stopped at the first failure
        with mock.patch.object(analyst, "RESULTS", folder):
            self.assertEqual(analyst.sealed_runs(), [])


class GatewayAdvisorTest(unittest.TestCase):
    def test_it_sends_the_pinned_prompt_caps_the_run_and_reads_one_json_object(self):
        from dclab_rnd.core_advisor import GatewayAdvisor

        sent = []

        class Client:
            run_limit_eur, spent_eur = None, 0.0

            def complete(self, messages, tools, max_tokens=None):
                sent.append(messages)
                self.spent_eur += 0.01
                return {"content": 'Here it is:\n```json\n{"unit": {"value": "ride", "quote": "one row per ride"}}\n```'}

        client = Client()
        gateway = mock.Mock()
        gateway.client.return_value = client
        advisor = GatewayAdvisor(gateway, cap_eur=0.5)
        self.assertEqual(advisor.ask("problem_analyst", {"description_paragraphs": ["x"]}), {"unit": {"value": "ride", "quote": "one row per ride"}})
        advisor.ask("problem_analyst", {"description_paragraphs": ["y"]})
        gateway.client.assert_called_once()  # one client, so one cap, for the whole run
        self.assertEqual(client.run_limit_eur, 0.5)
        self.assertAlmostEqual(advisor.spent_eur(), 0.02)
        from dclab_core import prompts

        self.assertEqual(sent[0][0], {"role": "system", "content": prompts.text("problem_analyst")})
        client.complete = lambda *a, **k: {"content": "{}"}
        self.assertEqual(advisor.ask("problem_analyst", {}), {})  # an empty object is a real answer: nothing supported
        client.complete = lambda *a, **k: {"content": "", "finish_reason": "length"}
        with self.assertRaises(RuntimeError):
            advisor.ask("problem_analyst", {})  # cut at the token limit: a failure, never an abstention
        client.complete = lambda *a, **k: {"content": "I cannot tell."}
        with self.assertRaises(ValueError):
            advisor.ask("problem_analyst", {})
        client.complete = lambda *a, **k: {"content": '{"a": 1} and {"b": 2}'}
        self.assertEqual(advisor.ask("problem_analyst", {}), {"a": 1})
        gateway.client.return_value = None
        self.assertIsNone(GatewayAdvisor(gateway).ask("problem_analyst", {}))  # no model here: the rules stand


if __name__ == "__main__":
    unittest.main()
