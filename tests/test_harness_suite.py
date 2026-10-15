"""Package 21.5, harness evaluation from real sessions: each case graded step by step (the right questions in order, each
once; the coordinator's steps; the hand-offs; no failure flag back); a flagged session a reviewer accepted replayed on
a table invented from its data card with the person's own answers; the owner's morning a permanent case; and no model
serves a purpose, no decision model acts, and no prompt or gate changes until the suite passed on it. With what the
morning's second half found: "anomaly" said beside a file of known cases is asked, past the question cap."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd

    from dclab_rnd.agent_eval import journeys, regression
except ImportError as error:  # pandas is not in every environment
    MISSING = str(error)
else:
    MISSING = ""

MORNING = {"questions": ["target", "labels"], "steps": ["bring_table", "build"],
           "handoffs": [{"role": "data_investigator", "allowed": False, "refused_columns": ["driver_rating"]}], "no_flags": ["error"]}


def row(**over):
    base = {"asked": ["target", "prediction_moment", "labels"], "coordinator": ["answer", "bring_table", "analyse", "build"],
            "handoffs": [{"role": "data_investigator", "allowed": False, "refused_columns": ["driver_rating"], "source": "rules"}], "flags": []}
    return {**base, **over}


@unittest.skipIf(MISSING, MISSING)
class GradeTest(unittest.TestCase):
    def failed(self, played, expect=MORNING):
        return [c["step"] for c in regression.grade({"expect": expect}, played) if not c["ok"]]

    def test_a_trace_that_follows_the_case_passes_every_step(self):
        self.assertEqual(self.failed(row()), [])

    def test_each_step_that_goes_wrong_is_named(self):
        self.assertEqual(self.failed(row(asked=["labels", "target"])), ["the right questions, in order"])
        self.assertEqual(self.failed(row(asked=["target", "labels", "target"])), ["each question asked once"])
        self.assertEqual(self.failed(row(coordinator=["answer", "build", "bring_table"])),
                         ["no build before the analysis", "the coordinator's steps, in order"])
        self.assertEqual(self.failed(row(handoffs=[])), ["the right hand-offs"])
        self.assertEqual(self.failed(row(flags=["error"])), ["no failure flag comes back"])

    def test_the_owners_morning_is_a_permanent_case(self):
        cases = {c["id"]: c for c in regression.load()}
        morning = cases["RG-01"]
        self.assertEqual((morning["words"], morning["family"]), ("I want build anomaly detection", "rare_event"))
        self.assertEqual(morning["expect"]["handoffs"][0]["refused_columns"], ["driver_rating"])
        self.assertIn("built_with_question_open", morning["expect"]["no_flags"])


@unittest.skipIf(MISSING, MISSING)
class ReviewFixesTest(unittest.TestCase):
    """The review's findings, each pinned."""

    def test_an_exact_trace_refuses_an_extra_question_or_step_and_a_build_never_precedes_the_analysis(self):
        exact = {"questions": ["target", "labels"], "steps": ["answer", "analyse", "build"], "exact": True}
        ok = row(asked=["target", "labels"], coordinator=["answer", "analyse", "build"])
        self.assertEqual([c["step"] for c in regression.grade({"expect": exact}, ok) if not c["ok"]], [])
        extra = row(asked=["family", "target", "labels"], coordinator=["answer", "analyse", "build"])
        self.assertEqual([c["step"] for c in regression.grade({"expect": exact}, extra) if not c["ok"]], ["the right questions, in order"])
        early = row(coordinator=["bring_table", "build", "analyse", "build"])
        self.assertIn("no build before the analysis", [c["step"] for c in regression.grade({"expect": MORNING}, early) if not c["ok"]])

    def test_what_the_suite_ran_on_includes_its_cases_and_every_gate(self):
        keys = set(regression.fingerprints())
        self.assertTrue({"cases:regression", "cases:journeys", "gate:roles", "gate:team", "gate:decision_routing"} <= keys, keys)
        run = {"fingerprints": regression.fingerprints()}
        weaker = [{**c, "expect": {}} for c in regression.load()]
        with mock.patch.object(regression, "load", return_value=weaker):
            self.assertEqual(regression.stale(run), ["cases:regression"])

    def test_the_models_a_run_used_are_those_its_plays_asked(self):
        rows = [{"models": {"home_agent": "m-1"}}, {"models": {"home_agent": "m-1", "critic": "m-2"}}, {"models": {"critic": "m-3"}}]
        self.assertEqual(regression.exercised(rows), {"critic": None, "home_agent": "m-1"})  # two models on one purpose: none passed it
        self.assertEqual(regression.exercised([{"models": {}}]), {})

    def test_in_a_live_run_the_roles_ask_their_models(self):
        from dclab_rnd.models import gateway

        seen = []

        class Advisor:
            def ask(self, purpose, evidence):
                seen.append(gateway._PRIVATE.get())
                return None

        case = {**journeys.load("dev")[0], "advisor": None}
        journeys.play(case, live={"client": None, "advisor": Advisor()})
        self.assertTrue(seen and not any(seen), seen)  # never private: a served model may answer
        played = journeys.play(journeys.load("dev")[1])
        self.assertEqual((played["models"], played["spent_eur"]), ({}, 0.0))  # the rules alone

    def test_a_prompt_changed_since_the_last_live_run_is_listed(self):
        now = regression.fingerprints()
        old = {**now, **{k: "changed" for k in now if k.startswith("prompt:")}}
        with mock.patch.object(regression, "runs", return_value=[{"live": True, "fingerprints": old}, {"live": False, "fingerprints": now}]):
            self.assertTrue(regression.prompts_unmeasured())
        with mock.patch.object(regression, "runs", return_value=[{"live": False, "fingerprints": old}]):
            self.assertEqual(regression.prompts_unmeasured(), [])  # no model measured here: nothing to compare with

    def test_the_harness_gate_runs_in_every_package_gate(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import affected_tests

        self.assertIn("test_harness_gate.py", affected_tests.ALWAYS)

    def test_anomaly_confirmed_beside_a_label_file_no_longer_blocks(self):
        from dclab_core.card import problem_card
        from dclab_core.contracts import LabelSourceFact
        from dclab_core.datacard import card_of
        from dclab_core.meet import check
        from dclab_core.understand import understand

        orders = pd.DataFrame({"order_id": range(200), "fare": [1.0] * 200})
        data = card_of({"orders.csv": orders, "labels": pd.DataFrame({"order_id": range(0, 200, 10)})}, roles={"labels": "labels"})
        words = "I want build anomaly detection"
        brief = understand(words, {"orders.csv": list(orders.columns)})
        brief = brief.model_copy(update={"label_source": LabelSourceFact(value="none", status="stated", quote="Nothing is labelled", source="person")})
        found = [m for m in check(problem_card(brief, words=words, family="anomaly"), data, joined=True).mismatches if m.code == "labels_contradict"]
        self.assertTrue(found and not found[0].blocking)

    def test_a_labels_answer_naming_a_column_stays_a_column(self):
        from dclab_core.conversation import read_answer

        for said in ("A column marks the ones the team flagged", "The label column holds the cases we confirmed",
                     "Yes, the is_fraud column: the frauds the team caught"):
            self.assertEqual(read_answer("labels", said), {"labels": "column"}, said)

    def test_a_sessions_label_file_is_replayed_as_its_own_file(self):
        from dclab_core.datacard import card_of

        orders = pd.DataFrame({"order_id": range(300), "fare": range(300)})
        card = card_of({"orders.csv": orders, "labels": pd.DataFrame({"order_id": range(0, 300, 10)})}, roles={"labels": "labels"})
        replica = journeys._from_card(card.model_dump(mode="json"), 0)
        self.assertIsNotNone(replica.labels)
        self.assertNotIn("labels", replica.tables)

    def test_a_sessions_row_keeps_no_column_name_in_the_committed_result(self):
        kept = regression._kept({"id": "FC-1", "source": "session", "checks": [{"step": "s", "ok": False, "detail": "modelled: ['secret_col']"}],
                                 "why": ["holdout: ['secret_col']"], "handoffs": [{"role": "r", "allowed": False, "refused_columns": ["secret_col"]}]})
        self.assertNotIn("secret_col", json.dumps(kept))

    def test_runs_are_read_in_number_order_and_a_measurement_on_other_journeys_is_not_this_one(self):
        self.assertEqual(sorted([Path("HRN-1000.json"), Path("HRN-999.json")], key=regression._number)[-1].name, "HRN-1000.json")
        with mock.patch.object(journeys, "fingerprint", return_value="another set"):
            from dclab_rnd.ledger import routing

            name = next(n for n, i in routing.passed().items() if i["point"] == "column.semantic_role")
            self.assertIsNone(journeys.workflow_measured(name, "column.semantic_role", 3))


@unittest.skipIf(MISSING, MISSING)
class SessionCaseTest(unittest.TestCase):
    def test_an_accepted_flagged_session_is_replayed_on_a_table_from_its_card(self):
        from dclab_core.datacard import card_of
        from dclab_core.replicas import subscription_churn

        table = subscription_churn.generate(0).subscribers
        card = card_of({"subscribers.csv": table}).model_dump(mode="json")
        home = Path(tempfile.mkdtemp())
        folder = home / regression.SESSION_CASES
        folder.parent.mkdir(parents=True)
        rows = [{"id": "FC-abc", "problem": "Which subscribers will cancel at renewal?", "detectors": ["built_with_question_open"],
                 "answers": {"target": "churned", "prediction_moment": "At renewal, before the subscriber decides; refunds come only after",
                             "action": "We call the ones likely to leave"},
                 "data_card": card},
                {"id": "FC-nocard", "problem": "x", "detectors": ["error"], "answers": {}, "data_card": None}]
        folder.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        cases = regression.session_cases(home)
        self.assertEqual([c["id"] for c in cases], ["FC-abc"])  # without its card a session cannot be replayed
        played = journeys.play(cases[0])
        self.assertTrue(played["right"], played["why"])
        self.assertEqual([c["step"] for c in regression.grade(cases[0], played) if not c["ok"]], [])

    def test_a_candidate_keeps_the_persons_answers(self):
        from dclab_core.datasets import failures

        rows, _ = failures([{"visit": "v1", "flags": [{"detector": "error", "severity": "high", "message": "m"}], "problem": "p",
                             "answers": {"target": "churned"}}], workspace="w")
        self.assertEqual(rows[0]["answers"], {"target": "churned"})


@unittest.skipIf(MISSING, MISSING)
class ChangeGateTest(unittest.TestCase):
    def test_a_gate_or_a_prompt_changed_since_the_run_makes_it_stale(self):
        run = {"fingerprints": regression.fingerprints()}
        self.assertEqual(regression.stale(run), [])
        with mock.patch.dict(journeys.GATE, {"suite": "something looser"}):
            self.assertEqual(regression.stale(run), ["gate:journeys"])
        self.assertTrue(regression.stale({"fingerprints": {}}))  # a run with nothing recorded is stale

    def test_no_model_serves_a_purpose_before_the_suite_passed_with_it(self):
        from dclab_rnd.models import routing

        store = routing.FileRouting(Path(tempfile.mkdtemp()) / "model_routing.json")
        public = {"model": "m-1", "key_configured": True, "endpoint": "e", "local": False}
        with mock.patch("dclab_rnd.models.settings.public", return_value=public), mock.patch.object(regression, "passed_with", return_value=None):
            with self.assertRaises(routing.RoutingError) as caught:
                routing.change(store, "critic", "serve", "strong", "reviewer", "it is better")
            self.assertIn("harness suite", str(caught.exception))
            doc = routing.change(store, "critic", "serve", "strong", "owner", "accepted untested", override=True)
            self.assertTrue(doc["history"][-1]["evidence"]["override"])
        with mock.patch("dclab_rnd.models.settings.public", return_value=public), \
                mock.patch.object(regression, "passed_with", return_value={"id": "HRN-009"}):
            doc = routing.change(store, "reporter", "serve", "strong", "reviewer", "it passed the suite")
        self.assertEqual(doc["history"][-1]["evidence"]["suite"], "HRN-009")

    def test_a_suite_run_counts_for_a_model_only_on_what_is_here(self):
        good = {"id": "HRN-1", "verdict": "pass: 17 of 17", "models": {"critic": "m-1"}, "fingerprints": regression.fingerprints()}
        old = {**good, "id": "HRN-0", "fingerprints": {}}
        with mock.patch.object(regression, "runs", return_value=[good]):
            self.assertEqual(regression.passed_with("critic", "m-1")["id"], "HRN-1")
            self.assertIsNone(regression.passed_with("critic", "m-2"))
        with mock.patch.object(regression, "runs", return_value=[old]):
            self.assertIsNone(regression.passed_with("critic", "m-1"))


@unittest.skipIf(MISSING, MISSING)
class MorningSecondHalfTest(unittest.TestCase):
    def test_anomaly_said_beside_a_file_of_known_cases_is_asked(self):
        from dclab_core.card import problem_card
        from dclab_core.datacard import card_of
        from dclab_core.meet import check
        from dclab_core.understand import understand

        orders = pd.DataFrame({"order_id": range(200), "fare": [1.0] * 200})
        cases = pd.DataFrame({"order_id": range(0, 200, 10)})
        data = card_of({"orders.csv": orders, "labels": cases}, roles={"labels": "labels"})
        words = "I want build anomaly detection"
        for stated, blocking in (("anomaly", True), (None, None)):
            card = problem_card(understand(words, {"orders.csv": list(orders.columns)}), words=words, data={"label_file": True}, family=stated)
            found = [m for m in check(card, data, joined=True).mismatches if m.code == "labels_contradict"]
            if blocking is None:
                self.assertEqual(found, [], card.family)  # no anomaly said: the file settles a rare event
            else:
                self.assertTrue(found and found[0].blocking, [m.code for m in check(card, data, joined=True).mismatches])

    def test_the_cases_we_found_are_known_cases(self):
        from dclab_core.conversation import read_answer

        self.assertEqual(read_answer("labels", "A sample of the cases we found"), {"labels": "file"})
        self.assertEqual(read_answer("labels", "the frauds the team caught"), {"labels": "file"})

    def test_a_contradiction_is_asked_past_the_question_cap(self):
        from dclab_rnd.draft.chat import MAX_QUESTIONS, HomeAgent
        from dclab_rnd.draft.store import DraftStore

        store = DraftStore(Path(tempfile.mkdtemp()))
        agent = HomeAgent(store, None)
        d = store.create("x")["id"]
        store.update(d, lambda x: x.update(questions=[{"id": f"q{i}", "field": f"f{i}", "text": "t", "options": [], "answered": "a"}
                                                      for i in range(MAX_QUESTIONS)]))
        self.assertIsNone(agent.ask(d, "budget", "How many?"))
        self.assertIsNotNone(agent.ask(d, "labels_check", "Learn from the known cases?"))


if __name__ == "__main__":
    unittest.main()
