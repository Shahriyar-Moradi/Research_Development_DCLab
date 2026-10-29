"""Package 28.6, the second opinion: two readers of the card's family (the conversation reader and the card's rules), a
judge that keeps what they agree on and asks the person, once, where they differ; the person's choice is a family they
stated, no answer keeps the card; every verdict a decision; the switch on only where the pattern's result passed; the
gate fixed before the run, judged on the two arms."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import second_opinion as so  # noqa: E402

try:
    import pandas as pd  # noqa: F401
except ImportError as error:  # the agent environment has no pandas
    MISSING = str(error)
else:
    MISSING = ""

# JS5-08's words (the spent fifth sealed set, read as dev material in 28.6): a number to estimate, read by the rules as time to event
WORDS = ("We supply couriers to restaurants that don't have their own. The restaurants keep asking how long until the food reaches "
         "their customer, and our dispatchers just say half an hour to everyone. Rain makes it all worse, obviously.")


class JudgeTests(unittest.TestCase):
    def test_two_readings_that_differ_are_asked_once_with_both_as_the_choices(self):
        o = so.judge_family("regression", "time_to_event")
        self.assertEqual((o.verdict, o.choices), ("ask", ["regression", "time_to_event"]))
        self.assertIn("a number for each row", o.question)
        self.assertIn("how long until something happens", o.question)
        self.assertEqual(so.Opinion.read(o.model_dump_json()), o)

    def test_agreement_or_one_reader_keeps_the_card_and_asks_nothing(self):
        self.assertEqual(so.judge_family("regression", "regression").verdict, "agree")
        self.assertEqual(so.judge_family(None, "time_to_event").verdict, "one_reader")
        self.assertEqual(so.judge_family("regression", None).verdict, "one_reader")
        self.assertEqual(so.judge_family("a number", "regression").verdict, "one_reader")  # not a family: no reading

    def test_an_answer_chooses_only_when_it_names_or_means_one_of_the_two(self):  # the review of 28.6, finding 8
        c = ["regression", "time_to_event"]
        self.assertEqual(so.read_choice("Predict the minutes, a number.", c), "regression")  # JS5-08's words: regression's meaning
        self.assertIsNone(so.read_choice("How long until the food reaches the customer, in minutes.", c, "time_to_event"))  # cues of both: the card stands
        self.assertEqual(so.read_choice("How long until the food reaches the customer.", c, "time_to_event"), "time_to_event")
        self.assertIsNone(so.read_choice("A yes or no for each order.", ["rare_event", "classification"]))  # a shape fits both
        self.assertIsNone(so.read_choice("Predict the minutes, a number.", c, "time_to_event"))  # two readings that differ choose nothing
        self.assertIsNone(so.read_choice("I am not sure", c))
        self.assertEqual(so.choices_of([so.option_label(f) for f in c]), c)

    @unittest.skipIf(MISSING, MISSING)
    def test_no_written_persons_answer_picks_the_wrong_family_of_any_pair(self):  # the second review of 28.6, finding 6
        from dclab_core.card import FAMILIES
        from dclab_core.conversation import read_answer
        from dclab_rnd.agent_eval import journeys

        wrong, chosen = [], 0
        for split in ("dev", "sealed_v5"):  # the fifth sealed set is spent: dev material
            for j in journeys.load(split):
                truth = j["family"] if isinstance(j["family"], list) else [j["family"]]
                said = (j.get("answers") or {}).get("family")
                if not said:
                    continue
                r = read_answer("family", said)
                stated = r.get("family") if r.get("family") and r.get("family_shape_only") != "yes" else None
                for t in truth:
                    for other in (f for f in FAMILIES if f not in truth):
                        for pair in ([t, other], [other, t]):
                            got = so.read_choice(said, pair, stated)
                            chosen += got is not None
                            if got == other:
                                wrong.append((j["id"], pair))
        self.assertEqual(wrong, [])
        self.assertGreater(chosen, 300)  # and it does choose: it is not silent

    def test_both_modules_import_first_in_a_fresh_interpreter(self):  # the review of 28.6, finding 1: an import cycle
        import subprocess

        for module in ("dclab_core.card", "dclab_core.conversation", "dclab_core.plan", "dclab_core.debate", "dclab_core.second_opinion",
                       "dclab_core.understand", "dclab_core.contracts"):
            done = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r}); import {module}"], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, f"{module}: {done.stderr[-400:]}")

    def test_an_answer_names_a_choice_by_its_label_or_name_only_when_one_matches(self):
        choices = ["regression", "time_to_event"]
        self.assertEqual(so.from_label(so.option_label("time_to_event"), choices), "time_to_event")
        self.assertEqual(so.from_label("Regression", choices), "regression")
        self.assertIsNone(so.from_label("a number", choices))  # the host reads words that name neither


@unittest.skipIf(MISSING, MISSING)
class HomeAgentTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import pandas as pd

        from dclab_rnd import experience, ledger
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore
        from tests.test_coordinator import upload

        self.home = Path(tempfile.mkdtemp())
        experience.install(experience.FileLog(self.home / "experience", "so"))
        self.addCleanup(experience.install, None)
        self.book = ledger.FileLedger(self.home / "ledger", "w")
        ledger.install(self.book)
        self.addCleanup(ledger.install, None)
        self.store = DraftStore(self.home / "drafts")
        rng = np.random.default_rng(0)
        n = 400
        self.table = pd.DataFrame({"order_id": [f"O{i}" for i in range(n)], "distance_km": rng.gamma(2, 2, n).round(2),
                                   "couriers_online": rng.integers(3, 30, n), "actual_minutes": rng.normal(32, 8, n).round(1)})
        self.upload = upload
        self.HomeAgent = HomeAgent

    def draft(self, second_opinion=True):
        agent = self.HomeAgent(self.store, None, second_opinion=second_opinion)
        draft = self.store.create(WORDS)["id"]
        from dclab_rnd.draft import intent as intents
        self.store.update(draft, lambda x: x.update(intent=intents.chosen("prediction")))  # the person chose a prediction (unread is asked)
        agent.start(draft)
        self.upload(self.store, agent, draft, "dispatch.csv", self.table)
        # where the reader left the card on JS5-08: the outcome named, the moment said, the reader's family a number

        def put(d):
            d["understanding"].update(target="actual_minutes", prediction_moment="When the order comes in, before a courier is picked",
                                      action="Send the restaurant the estimate", family="regression", family_by="reader", family_shape_only="yes")
            for q in d["questions"]:
                q["answered"] = q.get("answered") or "(answered)"
        self.store.update(draft, put)
        return agent, draft

    def test_the_rules_alone_read_time_to_event_here(self):  # the precondition: the reader's number gives way to the rules today
        from dclab_rnd.draft.card import card_of

        _, draft = self.draft(second_opinion=False)
        self.assertEqual(card_of(self.store.get(draft)).family.value, "time_to_event")

    def test_the_judge_asks_once_and_the_persons_choice_is_the_family_they_stated(self):
        from dclab_rnd.draft.card import card_of

        agent, draft = self.draft()
        agent.next_turn(draft)
        q = agent.pending(self.store.get(draft))
        self.assertEqual((q["field"], q["source"]), ("family", "second_opinion"))
        self.assertEqual(len(q["options"]), 2)
        agent.reply(draft, "Predict the minutes, a number.")  # JS5-08's own answer to a family question: neither label, read as a choice
        d = self.store.get(draft)
        self.assertEqual((d["understanding"]["family"], d["understanding"]["family_by"], d["understanding"]["family_shape_only"]),
                         ("regression", "person", "no"))
        self.assertEqual(card_of(d).family.value, "regression")  # the person's choice, never refined back by the rules
        agent.next_turn(draft)
        self.assertEqual(sum(1 for q in self.store.get(draft)["questions"] if q.get("source") == "second_opinion"), 1)  # once
        rec = next(r for r in self.book.records() if r.point == "problem.family")
        self.assertEqual({a.by: a.value for a in rec.answers if a.by != "classifier"},  # a shadow classifier may answer beside them
                         {"model_role": "regression", "rule": "time_to_event", "person": "regression"})
        self.assertEqual(rec.value_used, "regression")

    def test_no_answer_keeps_the_card_and_a_walk_goes_on(self):
        from dclab_rnd.draft.card import card_of
        from dclab_rnd.draft.coordinator import Coordinator
        from dclab_rnd.studio import ProjectStore, team

        agent, draft = self.draft()
        agent.next_turn(draft)
        person = team.Person(answer=lambda field, question: None, approve=lambda gate, summary: True, who="p")
        out = Coordinator(self.store, ProjectStore(self.home / "projects"), agent)._answer(draft, person)
        self.assertEqual(out, {"declined": "family", "source": "second_opinion"})  # declined, never a stop
        d = self.store.get(draft)
        self.assertEqual(card_of(d).family.value, "time_to_event")  # the card as it was
        self.assertIsNone(agent.pending(d))
        rec = next(r for r in self.book.records() if r.point == "problem.family")
        self.assertEqual(rec.value_used, "time_to_event")

    def test_a_kind_the_person_stated_is_never_put_to_them_again(self):  # the review of 28.6, finding 9
        agent, draft = self.draft()
        self.store.update(draft, lambda d: d.update(solution={"kind": "time_to_event"}))
        self.assertIsNone(agent.second_opinion_question(self.store.get(draft)))

    def test_a_shape_answer_keeps_the_card_and_the_choices_are_what_the_person_was_shown(self):  # findings 8 and 11
        from dclab_rnd.draft.card import card_of

        agent, draft = self.draft()
        agent.next_turn(draft)
        q = agent.pending(self.store.get(draft))
        self.assertEqual(q["opinion"]["choices"], ["regression", "time_to_event"])  # the verdict travels with the question
        agent.reply(draft, "A number, and how long until it happens")  # the words of both meanings: no choice
        d = self.store.get(draft)
        self.assertEqual(card_of(d).family.value, "time_to_event")  # declined: the card as it was
        self.assertTrue(next(x for x in d["questions"] if x.get("source") == "second_opinion")["declined"])

    def test_with_a_model_serving_home_the_choice_is_still_read_by_code(self):  # finding 10: the product's path
        import json as _json

        class Model:  # a step of a read-only tool, then a reply: the model's turn never reaches take_answer
            def complete(self, messages, tools=None, max_tokens=None, **_):
                if messages[-1]["role"] != "tool":
                    calls = [{"id": "c0", "name": "get_profile", "arguments": {}}]
                    return {"content": "", "tool_calls": calls, "usage": {"input_tokens": 10, "output_tokens": 5},
                            "assistant_message": {"role": "assistant", "content": "", "tool_calls": [
                                {"id": "c0", "type": "function", "function": {"name": "get_profile", "arguments": _json.dumps({})}}]}}
                return {"content": "Noted.", "tool_calls": [], "usage": {"input_tokens": 10, "output_tokens": 5},
                        "assistant_message": {"role": "assistant", "content": "Noted."}}

        agent, draft = self.draft()
        agent.next_turn(draft)
        served = self.HomeAgent(self.store, Model(), second_opinion=True)
        served.reply(draft, "Regression")
        d = self.store.get(draft)
        self.assertEqual((d["understanding"]["family"], d["understanding"]["family_by"]), ("regression", "person"))
        self.assertTrue(any(r.point == "problem.family" for r in self.book.records()))

    def test_off_nothing_is_asked_and_a_family_the_person_answered_is_never_asked_again(self):
        agent, draft = self.draft(second_opinion=False)
        agent.next_turn(draft)
        self.assertFalse(any(q.get("source") == "second_opinion" for q in self.store.get(draft)["questions"]))
        agent, draft = self.draft()
        self.store.update(draft, lambda d: d["questions"].append({"id": "qf", "field": "family", "text": "Which kind?", "options": [],
                                                                    "why": "", "answered": "a number"}))
        self.assertIsNone(agent.second_opinion_question(self.store.get(draft)))


class SwitchAndGateTests(unittest.TestCase):
    def test_a_pattern_switch_turns_on_only_where_its_result_passed(self):
        from unittest import mock

        from dclab_rnd.orchestra import patterns, settings

        home, results = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        with mock.patch.object(patterns, "RESULTS", results):
            with self.assertRaises(ValueError):
                settings.set_settings(home, {"second_opinion": True})
            (results / "ORC-001_second_opinion.json").write_text(json.dumps({"pattern": "second_opinion", "status": "completed", "verdict": "fail"}))
            with self.assertRaises(ValueError):
                settings.set_settings(home, {"second_opinion": True})
            (results / "ORC-002_second_opinion.json").write_text(json.dumps({"pattern": "second_opinion", "status": "completed", "verdict": "pass"}))
            self.assertTrue(settings.set_settings(home, {"second_opinion": True})["second_opinion"])
            self.assertFalse(settings.settings(home)["debate"])
            settings.set_settings(home, {"second_opinion": False})  # off is always allowed

    def test_the_product_serves_a_pattern_only_while_its_switch_is_on_and_its_result_passes(self):
        from types import SimpleNamespace
        from unittest import mock

        from dclab_rnd.draft.work import DraftWork
        from dclab_rnd.orchestra import patterns

        home, results = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        work = DraftWork(None, None, jobs=SimpleNamespace(store=SimpleNamespace(home=home)))
        (home / "team.json").write_text(json.dumps({"second_opinion": True}))  # switched on by hand, with no result behind it
        with mock.patch.object(patterns, "RESULTS", results):
            self.assertFalse(work.pattern_on("second_opinion"))
            (results / "ORC-001_second_opinion.json").write_text(json.dumps({"pattern": "second_opinion", "status": "completed", "verdict": "pass"}))
            self.assertTrue(work.pattern_on("second_opinion"))
            (results / "ORC-002_second_opinion.json").write_text(json.dumps({"pattern": "second_opinion", "status": "completed", "verdict": "fail"}))
            self.assertFalse(work.pattern_on("second_opinion"))  # the newest result failed: it stops serving
        self.assertFalse(DraftWork(None, None).pattern_on("second_opinion"))  # no workspace known: off

    @unittest.skipIf(MISSING, MISSING)
    def test_the_gate_wants_a_gain_on_its_field_and_nothing_lost_elsewhere(self):
        from dclab_rnd.agent_eval.journeys import pattern_verdict

        def row(i, card=True, holdout=True, questions=4, asked=("target", "prediction_moment"), judged=False):
            return {"id": f"J{i}", "parts": {"card": card, "workflow": True, "pipeline": True, "holdout": holdout},
                    "questions": questions, "asked": list(asked), "refused_moves": 0,
                    **({"second_opinion": {"asked": 1, "declined": 0, "chose": 1}} if judged else {})}
        base = [row(1, card=False), row(2)]
        better = [row(1, questions=5, judged=True), row(2)]
        noise = [row(1, questions=5), row(2)]  # a card gained where the judge never asked: the reader's noise
        unanswered = [{**row(1, questions=5), "second_opinion": {"asked": 1, "declined": 0, "chose": 0}}, row(2)]  # asked, never chosen
        self.assertFalse(pattern_verdict("second_opinion", base, unanswered, {"baseline": 0.10, "arm": 0.12}, 0)["passed"])
        self.assertFalse(pattern_verdict("second_opinion", base, noise, {"baseline": 0.10, "arm": 0.12}, 0)["passed"])
        broke = [{**row(1, card=False), "why": ["broke: RuntimeError: x"]}, row(2)]
        self.assertFalse(pattern_verdict("second_opinion", broke, better, {"baseline": 0.10, "arm": 0.12}, 0)["passed"])  # a broken journey
        self.assertFalse(pattern_verdict("second_opinion", base, better, {"baseline": 0.0, "arm": 0.0}, 0)["passed"])  # no priced baseline
        v = pattern_verdict("second_opinion", base, better, {"baseline": 0.10, "arm": 0.12}, 0)
        self.assertTrue(v["passed"], v)
        self.assertEqual(v["gained"], ["J1"])
        self.assertFalse(pattern_verdict("second_opinion", base, base, {"baseline": 0.1, "arm": 0.1}, 0)["passed"])  # no gain
        self.assertFalse(pattern_verdict("second_opinion", base, [row(1, questions=5), row(2, holdout=False)], {"baseline": 0.1, "arm": 0.1}, 0)["passed"])
        self.assertFalse(pattern_verdict("second_opinion", base, better, {"baseline": 0.10, "arm": 0.2}, 0)["passed"])  # over 1.5 times the cost
        self.assertFalse(pattern_verdict("second_opinion", base, better, {"baseline": 0.1, "arm": 0.1}, 1)["passed"])  # a request failed
        twice = [row(1, asked=("chat→family", "family"), judged=True), row(2)]  # asked at a chat stop, then by the judge: twice
        self.assertFalse(pattern_verdict("second_opinion", base, twice, {"baseline": 0.1, "arm": 0.1}, 0)["passed"])
        with self.assertRaises(ValueError):
            pattern_verdict("second_opinion", base, list(reversed(better)), {}, 0)

    @unittest.skipIf(MISSING, MISSING)
    def test_a_subset_or_a_run_with_a_broken_journey_is_never_stored(self):  # the review of 28.6, findings 3 and 4
        import io
        import os
        from contextlib import redirect_stdout
        from unittest import mock

        from dclab_rnd.agent_eval import journeys
        from dclab_rnd.agent_eval.journeys import __main__ as cli

        reader = mock.MagicMock(failures=0)
        reader.advisor.spent_eur.return_value = 0.01
        reader.advisor.report.return_value = []

        def played(j, seed, routing, live, keep, broken=()):
            if j["id"] in broken:
                raise RuntimeError("the engine broke")
            return {"id": j["id"], "kind": j["kind"], "family": j["family"], "parts": dict.fromkeys(journeys.PARTS, True), "right": True,
                    "why": [], "questions": 3, "asked": ["target"]}
        for argv, broken, stored in ((["--only", "JS5-08"], (), 0), ([], ("JD-01",), 0), ([], (), 1)):  # the last: the whole material, stored
            with mock.patch.dict(os.environ, {}), mock.patch("dclab_rnd.draft.reader.reader", return_value=reader), \
                    mock.patch("dclab_rnd.models.gateway.for_workspace"), mock.patch("dotenv.load_dotenv"), \
                    mock.patch.object(journeys, "play", side_effect=lambda *a, broken=broken: played(*a, broken=broken)), \
                    mock.patch("dclab_rnd.orchestra.patterns.store") as store, redirect_stdout(io.StringIO()):
                cli.main(["--pattern", "second_opinion", "--yes", "--cap-eur", "0.1", *argv])
            self.assertEqual(store.call_count, stored, (argv, broken))
            if stored:
                self.assertEqual(len(store.call_args.args[2]["arm"]), 45)  # every person of the material, in both arms

    @unittest.skipIf(MISSING, MISSING)
    def test_the_runner_prints_its_plan_and_sends_nothing_without_yes(self):
        import io
        from contextlib import redirect_stdout

        from dclab_rnd.agent_eval.journeys import __main__ as cli

        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["--pattern", "second_opinion", "--cap-eur", "1.5"])
        self.assertEqual(code, 0)
        self.assertIn("Plan: 45 people", out.getvalue())
        self.assertIn("second_opinion_gate_v1", out.getvalue())


if __name__ == "__main__":
    unittest.main()
