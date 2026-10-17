"""Package 20.10, the acceptance suite of conversations: a written person is carried from their first sentence to the
holdout number (the coordinator, the build, the Solution page, the engine) and scored on the card, the workflow, the
pipeline and the holdout against the replica's planted truth; a family the engine does not run stops at the build with
its reason; the sealed people are never edited; and the gate is fixed. With the defects the suite found on dev, each
pinned: a card's family as a word in the pipeline, a yes or no outcome cleaned into booleans, the families the words
named nowhere, an outcome named inside a sentence, an outcome computed from related tables, blocks of places under any
pack, a family the engine does not run built as a model of a column, and a rare event's review budget never asked."""

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

    from dclab_core.card import problem_card
    from dclab_core.understand import understand
    from dclab_rnd import kinds
    from dclab_rnd.agent_eval import journeys
except ImportError as error:  # pandas is not in every environment
    MISSING = str(error)
else:
    MISSING = ""


def journey(jid):
    return next(j for j in journeys.load("dev") if j["id"] == jid)


@unittest.skipIf(MISSING, MISSING)
class JourneyTest(unittest.TestCase):
    def test_a_written_person_is_carried_to_the_holdout_number(self):
        row = journeys.play(journey("JD-02"))
        self.assertTrue(row["right"], row["why"])
        self.assertEqual(row["parts"], {"card": True, "workflow": True, "pipeline": True, "holdout": True})
        self.assertEqual((row["got"]["family"], row["got"]["target"]), ("classification", "churned"))
        self.assertGreater(row["holdout"]["roc_auc"], 0.65)
        self.assertEqual(row["refused_moves"], 0)
        self.assertLessEqual(row["questions"], 5)

    def test_a_kept_play_leaves_its_whole_record(self):  # 26.5: a live run's journeys are kept, not lost with their temporary folder
        import tempfile

        keep = Path(tempfile.mkdtemp()) / "arm" / "JD-02"
        row = journeys.play({**journey("JD-02"), "answers": {}}, keep=keep)
        self.assertTrue((keep / "row.json").is_file() and (keep / "events.jsonl").is_file())
        self.assertEqual(json.loads((keep / "row.json").read_text())["id"], row["id"])
        self.assertTrue(list((keep / "workspace" / "drafts").rglob("*.json")))  # the drafts with every turn of the conversation

    def test_a_walk_stopped_on_an_untagged_question_is_answered_as_in_the_chat_and_goes_on(self):  # 26.5 live pass 2
        from dclab_rnd.draft import coordinator

        calls = []
        stop = {"state": "problem", "next": "stop", "says": "The problem card is not ready: Which column holds the outcome?", "steps": [],
                "blockers": [], "project_id": None}
        later = {**stop, "says": "No progress at problem: nothing more"}

        def run(self, draft_id, person):
            calls.append(draft_id)
            return stop if len(calls) == 1 else later

        with mock.patch.object(coordinator.Coordinator, "run", run):
            row = journeys.play(journey("JD-03"))
        self.assertEqual(len(calls), 2)  # walked again after the person's answer
        self.assertIn("chat→target", row["asked"])
        self.assertFalse(row["right"])

    def test_a_person_who_answers_nothing_is_never_built_and_the_row_says_why(self):
        row = journeys.play({**journey("JD-02"), "answers": {}})
        self.assertFalse(row["right"])
        self.assertTrue(any(w.startswith("not built") for w in row["why"]), row["why"])

    def test_a_family_the_engine_does_not_run_stops_at_the_build_with_its_reason(self):
        with mock.patch.object(kinds.get("uplift"), "runs_in_engine", False):  # as if the engine did not run uplift (16.15 B: it does)
            row = journeys.play({**journey("JD-14"), "built": False})
            self.assertTrue(row["right"], row["why"])
            self.assertIn("not run by the engine", row["said"])
            built = journeys.play(journey("JD-14"))  # expected built, it is not: wrong, and said
            self.assertFalse(built["right"])
        silent = journeys.play({**journey("JD-13"), "words": "Help me with our customer table", "answers": {}, "built": False})  # the review's R2: any other stop is not the refusal
        self.assertFalse(silent["right"], silent)
        self.assertFalse(silent["parts"]["holdout"], silent)  # not for its card alone: the stop was no refusal of the family

    def test_a_move_the_validator_refused_is_counted(self):
        from dclab_rnd.kinds import acceptance
        from dclab_rnd.studio import graph

        store, pid = acceptance.build(kinds.get("binary").case(0), Path(tempfile.mkdtemp()))
        self.assertEqual(journeys.refused(store, pid), [])
        verdict = graph.check(store.get(pid), "approve_gate", "agent", stage="final")  # an agent may not approve a gate
        self.assertFalse(verdict.allowed)
        graph.log(store, pid, verdict)
        self.assertEqual([t["move"] for t in journeys.refused(store, pid)], ["approve_gate"])

    def test_the_sealed_people_are_never_edited(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "journeys_sealed.json").write_text("[]", encoding="utf-8")
        (tmp / "sealed.sha256").write_text("0" * 64, encoding="utf-8")
        with mock.patch.object(journeys, "HERE", tmp), self.assertRaises(ValueError):
            journeys.load("sealed")

    def test_the_reader_commands_refuse_what_would_spend_or_store_a_set_wrongly(self):
        from dclab_rnd.agent_eval.journeys import __main__ as cli

        for argv in (["--split", "sealed_v4"], ["--split", "sealed_v4", "--spent", "--reader"]):  # no reader arm; a set never run
            with mock.patch.object(sys, "argv", ["journeys", *argv]), mock.patch.object(cli, "sealed_runs", return_value=[]):
                self.assertEqual(cli.main(), 2, argv)
        self.assertIn("reader", journeys.GATES["sealed_v4"])  # the reader's rule, fixed before the run
        with mock.patch.object(journeys, "RESULTS", Path(tempfile.mkdtemp())):
            self.assertFalse(journeys.reader_passed())

    def test_the_fourth_sealed_set_is_frozen_with_its_own_gate_and_seed(self):
        people = journeys.load("sealed_v4")
        self.assertEqual(len(people), 30)
        self.assertEqual({j["kind"] for j in people}, {k.key for k in kinds.all_kinds()})
        self.assertEqual((journeys.GATES["sealed_v4"]["seed"], journeys.GATES["sealed_v4"]["split"]), (4, "sealed_v4"))

    def test_the_third_sealed_set_is_frozen_with_its_own_gate_and_seed(self):
        people = journeys.load("sealed_v3")
        self.assertEqual(len(people), 30)
        self.assertEqual({j["kind"] for j in people}, {k.key for k in kinds.all_kinds()})
        self.assertEqual((journeys.GATES["sealed_v3"]["seed"], journeys.GATES["sealed_v3"]["split"]), (3, "sealed_v3"))

    def test_the_second_sealed_set_is_frozen_with_its_own_gate_and_seed(self):
        people = journeys.load("sealed_v2")  # refused if a byte changed since it was pinned
        by_kind = {}
        for j in people:
            by_kind[j["kind"]] = by_kind.get(j["kind"], 0) + 1
        self.assertEqual(set(by_kind.values()), {2})
        self.assertEqual(set(by_kind), {k.key for k in kinds.all_kinds()})
        self.assertTrue(all("built" not in j for j in people))  # every kind runs in the engine
        gate = journeys.GATES["sealed_v2"]
        self.assertEqual((gate["seed"], gate["split"]), (2, "sealed_v2"))
        self.assertEqual({k: v for k, v in gate.items() if k in ("journey", "suite")}, {k: journeys.GATE[k] for k in ("journey", "suite")})

    def test_the_gate(self):
        from dclab_core.card import FAMILIES

        def rows(right, n, refused=0, families=tuple(FAMILIES)):
            return [{"right": i < right, "refused_moves": refused if i == 0 else 0, "family": families[i % len(families)]} for i in range(n)]

        self.assertTrue(journeys.verdict(rows(26, 30)).startswith("pass"))
        self.assertTrue(journeys.verdict(rows(25, 30)).startswith("fail"))  # below 85%
        self.assertTrue(journeys.verdict(rows(30, 30, refused=1)).startswith("fail"))  # a move the validator refused
        self.assertTrue(journeys.verdict(rows(5, 5)).startswith("fail"))  # all right, but too few to bound it above 0.6
        self.assertTrue(journeys.verdict([]).startswith("fail"))
        uncovered = journeys.verdict(rows(30, 30, families=("regression",)))
        self.assertTrue(uncovered.startswith("fail") and "no journey for" in uncovered, uncovered)  # every family listed, each with a journey
        listed = {f["family"]: f for f in journeys.families(rows(1, 2, families=("regression",)))}
        self.assertEqual(set(listed), set(FAMILIES))
        self.assertEqual(listed["regression"], {"family": "regression", "journeys": 2, "right": 1, "interval": [0.095, 0.905]})
        self.assertEqual(listed["uplift"]["interval"], [0.0, 1.0])
        self.assertEqual(json.loads(json.dumps(journeys.GATE))["version"], "journeys_gate_v1")


@unittest.skipIf(MISSING, MISSING)
class FoundOnDevTest(unittest.TestCase):
    """The defects the suite found, each pinned where it lives."""

    def test_a_cleaned_yes_or_no_outcome_is_read_and_a_label_in_no_row_is_refused(self):
        kind = kinds.get("binary")
        frame = pd.DataFrame({"churned": [True, False, True, False]})
        y, _ = kind.target(frame, {"target": "churned", "positive_label": "yes"})
        self.assertEqual(y.tolist(), [1, 0, 1, 0])
        with self.assertRaises(ValueError) as caught:
            kind.target(pd.DataFrame({"churned": ["a", "b"]}), {"target": "churned", "positive_label": "yes"})
        self.assertIn("none of the rows", str(caught.exception))

    def test_the_families_the_words_named_nowhere(self):
        def family(words, columns):
            return problem_card(understand(words, {"t": columns}), words=words).family.value

        self.assertEqual(family("How long will each new subscriber stay before cancelling?", ["plan", "days_subscribed"]), "time_to_event")
        self.assertEqual(family("Raise an alert before a pump fails, from its sensor readings", ["pump", "time", "pressure"]), "streams")
        self.assertEqual(family("Send the coupon only to people it would make buy", ["age", "bought", "got_coupon"]), "uplift")
        self.assertEqual(family("Which accounts go quiet, from their logins, payments and tickets?", ["account_id", "plan"]), "entity_tables")
        self.assertEqual(family("Flag the urgent emails from their text", ["email_id", "body", "urgent"]), "text_or_images")
        self.assertNotEqual(family("Which listings get sold, with their photos counted?", ["listing_id", "n_photos", "sold"]), "text_or_images")

    def test_an_outcome_in_words_counts_with_a_table_only_when_related_tables_compute_it(self):
        def card(words, family):
            brief = understand(words, {"t": ["customer_id", "plan"]})
            brief = brief.model_copy(update={"target_definition": brief.target_definition.model_copy(
                update={"value": "no order in the next 30 days", "status": "stated"})})
            return problem_card(brief, words=words, data={"columns": True}, family=family)

        self.assertEqual(card("Which customers stop ordering?", "entity_tables").outcome, "no order in the next 30 days")
        self.assertIsNone(card("Which customers stop ordering?", "classification").outcome)  # a table's outcome is a column

    def test_an_outcome_named_inside_a_sentence_is_that_column_and_a_word_is_not(self):
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore

        store = DraftStore(Path(tempfile.mkdtemp()))
        agent = HomeAgent(store, None)
        for said, expected in (("days_subscribed, with cancelled saying whether they left", "days_subscribed"),
                               ("duration days_subscribed, event flag cancelled", "days_subscribed"),  # 20.10 v2: a word before it
                               ("we only know it once cancelled is set", None),
                               ("Flag refunded is only set after delivery, we want is_late", None),  # mentioned, set later: never the outcome
                               ("the target days_subscribed is not known at the moment", None),
                               ("A failure of the machine in the next hours", None), ("machine", None)):
            d = store.create("x")["id"]
            store.update(d, lambda x: x.update(analysis={"profile": {"columns": [{"name": n} for n in ("machine", "days_subscribed", "cancelled", "refunded", "is_late")]}}))
            agent.read_card_answer(d, "target", said)
            self.assertEqual((store.get(d).get("understanding") or {}).get("target"), expected, said)

    def test_a_named_outcome_column_beside_another_familys_words_is_asked_never_refused(self):
        from dclab_rnd.draft import api

        for words, column in (("Predict churn so we know who to target with retention calls", "churned"),
                              ("Will a customer click the recommended products in the email?", "clicked"),
                              ("Which customers will churn, from their tenure, plan and monthly charges?", "churned"),
                              ("Predict whether a listing sells from its photos and price", "sold")):
            brief = understand(words, {"t": [column, "plan", "price"]})
            brief = brief.model_copy(update={"target_column": brief.target_column.model_copy(update={"value": column, "status": "stated"}),
                                             "task": brief.task.model_copy(update={"value": "binary", "status": "inferred"})})
            card = problem_card(brief, words=words, data={"columns": True, "tables": 1})
            self.assertIsNone(card.family.value, words)
            self.assertIn("classification", card.in_play, words)
            self.assertEqual([q.field for q in card.questions if q.blocks][:1], ["family"], words)
        store = mock.Mock()
        store.get.return_value = {"problem": "Predict churn so we know who to target with retention calls", "assets": [],
                                  "understanding": {"target": "churned"}, "solution": {"target": "churned", "task": "binary"}}
        self.assertNotIn("family not built yet", [r.rule for r in api.build_blockers(store, "d1")])

    def test_an_outcome_in_words_on_one_table_says_the_table_it_needs(self):
        words = "Predict whether each machine fails next week from its sensor readings"
        brief = understand(words, {"t": ["machine", "time", "pressure"]})
        brief = brief.model_copy(update={"target_definition": brief.target_definition.model_copy(update={"value": "the ones that fail", "status": "stated"})})
        card = problem_card(brief, words=words, data={"columns": True, "tables": 1})
        self.assertEqual(card.family.value, "streams")
        self.assertIn("Attach the table whose events decide the outcome", " ".join(q.text for q in card.questions))

    def test_a_column_only_mentioned_is_never_the_outcome(self):
        from dclab_rnd.draft.chat import HomeAgent
        from dclab_rnd.draft.store import DraftStore

        store = DraftStore(Path(tempfile.mkdtemp()))
        agent = HomeAgent(store, None)
        for said in ("Whether they leave; we only know it once final_refund_amount is paid", "Whether they cancel before plan_end_date"):
            d = store.create("x")["id"]
            store.update(d, lambda x: x.update(analysis={"profile": {"columns": [{"name": n} for n in ("churned", "final_refund_amount", "plan_end_date")]}}))
            agent.read_card_answer(d, "target", said)
            self.assertIsNone((store.get(d).get("understanding") or {}).get("target"), said)

    def test_the_exports_read_a_yes_label_as_the_engine_does(self):
        from dclab_rnd.studio import export

        profile = {"columns": [{"name": "churned", "dtype": "bool", "kind": "boolean"}]}
        self.assertIn("== 'True'", export._target_code("binary", {"target": "churned", "positive_label": "yes"}, profile))
        self.assertIn("== 'yes'", export._target_code("binary", {"target": "churned", "positive_label": "yes"}, {"columns": [{"name": "churned", "dtype": "object"}]}))

    def test_an_unrun_family_is_said_as_soon_as_it_is_known(self):
        from dclab_core import coordinator as machine

        s = machine.situation(machine.Facts(blocking_questions=[{"field": "outcome", "text": "Which column?"}], files_ready=1,
                                            build_refusals=[{"rule": machine.FAMILY_NOT_RUN, "message": "Ranking is not run by the engine yet"}]))
        self.assertEqual((s.state, s.next, s.blockers[0].code), ("problem", "stop", machine.FAMILY_NOT_RUN))

    def test_the_solution_page_saves_a_label_files_target(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app
        from dclab_rnd.kinds import acceptance

        client = TestClient(create_app(Path(tempfile.mkdtemp())))  # the server's own store, on files or PostgreSQL
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        store = client.app.state.services.projects
        _, pid = acceptance.build(kinds.get("rare_event").case(0), None, store=store)
        solution = store.get(pid)["solution"]
        self.assertNotIn(solution["target"], store.get(pid)["data"]["columns"])  # the target is the label file's
        saved = client.put(f"/api/projects/{pid}/solution", json={**solution, "notes": "checked"}, headers=h)
        self.assertEqual(saved.status_code, 200, saved.text)  # refused before: the label's target is no column of the table

    def test_blocks_of_places_run_as_the_places_kind_under_any_pack(self):
        spatial = {"task": "regression", "spatial": {"lat_column": "lat", "lon_column": "lon", "block_km": 5.0}}
        for pack in (None, "tabular", "maps"):
            self.assertEqual(kinds.resolve(spatial, pack).key, "places", pack)
        self.assertEqual(kinds.resolve({"task": "regression"}, "tabular").key, "regression")

    def test_a_family_the_engine_does_not_run_is_refused_at_the_build(self):
        from dclab_rnd.draft import api

        store = mock.Mock()
        store.get.return_value = {"problem": "Group our customers by how they buy", "assets": [], "understanding": {}}
        self.assertNotIn("family not built yet", [r.rule for r in api.build_blockers(store, "d1")])  # 16.15 B: the engine runs it
        with mock.patch.object(kinds.get("segmentation"), "runs_in_engine", False):
            refused = api.build_blockers(store, "d1")
        self.assertIn("family not built yet", [r.rule for r in refused])

    def test_home_asks_a_rare_events_review_budget(self):
        from dclab_rnd.draft.chat import MAX_QUESTIONS

        self.assertGreaterEqual(MAX_QUESTIONS, 5)  # four stopped before the budget: the operating point was never set
        self.assertLessEqual(MAX_QUESTIONS, 5)  # the conversation gate's limit



class WrittenPersonMeaningTests(unittest.TestCase):
    """26.5: a written person answers a model's own question by what it asks, the rules' by its field as before."""

    def test_a_question_tagged_with_its_field_is_answered_as_before(self):
        from dclab_rnd.agent_eval.journeys import _Person

        p = _Person({"target": "The churners", "prediction_moment": "At signup"})
        self.assertEqual(p._answer("target", "anything"), "The churners")
        self.assertEqual(p.asked, ["target"])

    def test_a_models_question_in_its_own_words_is_answered_by_its_meaning(self):
        from dclab_rnd.agent_eval.journeys import _Person

        p = _Person({"prediction_moment": "Each morning, before the rides are paid", "budget": "About 80 a day",
                     "action": "The audit team reviews them", "target": "The confirmed frauds"})
        self.assertEqual(p._answer("notes", "When is each prediction made, and what is known then?"), "Each morning, before the rides are paid")
        self.assertEqual(p._answer("notes", "About how many rides can the team review each morning?"), "About 80 a day")
        self.assertEqual(p._answer("costs", "What does a misrouted case cost you?"), "The audit team reviews them")
        self.assertEqual(p._answer("notes", "What exactly do you want to predict?"), "The confirmed frauds")
        self.assertIn("notes→prediction_moment", p.asked)
        self.assertIsNone(p._answer("notes", "What is your favourite colour?"))  # nothing it holds: no answer, as a person who does not know


if __name__ == "__main__":
    unittest.main()
