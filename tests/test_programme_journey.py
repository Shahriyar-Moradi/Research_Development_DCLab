"""Packages 24.6 and 24.8: the written people of a problem with several parts, from the first sentence to the one read of the holdout (or to the reason it stays closed),
the gate fixed before the sealed set was run, the two arms and the once-only sealed run."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd.agent_eval.journeys import programme as journeys
except ImportError as error:  # the agent environment lacks pandas
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class DevJourneysTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.people = {j["id"]: j for j in journeys.load("dev")}
        cls.rows = {i: journeys.play(cls.people[i], seed=0) for i in ("JP-01", "JP-03", "JP-04", "JP-05", "JP-06")}

    def test_the_dev_set_is_written_in_the_format_the_sealed_set_has(self):
        self.assertEqual(sorted(self.people), ["JP-01", "JP-02", "JP-03", "JP-04", "JP-05", "JP-06"])
        sealed = journeys.load("sealed")
        for j in [*self.people.values(), *sealed]:
            self.assertTrue({"id", "programme", "words", "answers", "brings", "when_blocked", "expect"} <= set(j), j["id"])
            self.assertTrue({"size", "families", "approaches", "composes"} <= set(j["expect"]), j["id"])
            self.assertEqual(len(j["expect"]["families"]), len(j["expect"]["approaches"]), j["id"])

    def test_a_person_with_all_the_files_is_carried_from_the_first_sentence_to_the_holdout_number(self):
        row = self.rows["JP-01"]
        self.assertTrue(row["right"], row["why"])
        self.assertEqual(set(row["parts"]), set(journeys.PARTS))
        for step in ("draw_map", "confirm_map"):
            self.assertIn(step, row["steps"])
        self.assertEqual((row["refused_moves"], row["holdout_uses"]), (0, 1))
        h = row["holdout"]
        self.assertGreater(h["low"], 0)
        self.assertEqual(h["leaves"], 3)

    def test_a_person_without_the_accounts_drops_that_part_and_the_rest_is_composed(self):
        row = self.rows["JP-03"]
        self.assertTrue(row["right"], row["why"])
        self.assertEqual(row["got"]["approaches"], ["anomaly", "rare_event", None, "review_budgeted"], "the rings have no approach without the accounts")
        self.assertEqual(row["holdout"]["leaves"], 2)
        self.assertGreaterEqual(row["holdout"]["difference"], 0, "never worse than its best leaf")
        self.assertEqual(row["holdout_uses"], 1)

    def test_a_person_who_stops_at_a_blocked_part_leaves_the_holdout_closed_and_unused(self):
        row = self.rows["JP-04"]
        self.assertTrue(row["right"], row["why"])
        self.assertEqual(row["holdout_uses"], 0)
        self.assertNotIn("holdout", row, "nothing was scored on the holdout")

    def test_with_no_confirmed_case_there_is_nothing_to_judge_a_list_on(self):
        row = self.rows["JP-05"]
        self.assertTrue(row["right"], row["why"])
        self.assertEqual(row["got"]["families"], ["anomaly", "entity_tables", "ranking"], "no classifier without a case")
        self.assertEqual(row["holdout_uses"], 0)

    def test_one_problem_is_read_as_one_and_routed_by_the_family_asked_for(self):
        row = self.rows["JP-06"]
        self.assertTrue(row["right"], row["why"])
        self.assertEqual((row["got"]["size"], row["got"]["families"], row["got"]["approaches"]), ("single", ["anomaly"], ["anomaly"]))
        self.assertEqual((row["parts"]["pipeline"], row["parts"]["holdout"]), (None, None), "a single problem has no programme: those parts do not apply")


@unittest.skipIf(MISSING, MISSING)
class ScoringTest(unittest.TestCase):
    def expect(self, families, approaches):
        return {"families": families, "approaches": approaches}

    def test_every_part_the_expert_expects_must_be_drawn_with_the_experts_approach(self):
        e = self.expect(["rare_event", "entity_tables", "ranking"], ["rare_event", "entity_tables", "review_budgeted"])
        ok, why = journeys._workflow_ok(e, ["rare_event", "entity_tables", "ranking"], ["rare_event", "entity_tables", "review_budgeted"])
        self.assertTrue(ok, why)
        ok, why = journeys._workflow_ok(e, ["rare_event", "entity_tables", "ranking"], ["rare_event", "anomaly", "review_budgeted"])
        self.assertFalse(ok)
        self.assertIn("not drawn with that approach", why)
        self.assertFalse(journeys._workflow_ok(e, ["rare_event", "ranking"], ["rare_event", "review_budgeted"])[0], "a missing part")

    def test_one_part_beyond_the_experts_is_tolerated_and_two_are_not(self):
        e = self.expect(["rare_event", "ranking"], ["rare_event", "review_budgeted"])
        self.assertTrue(journeys._workflow_ok(e, ["anomaly", "rare_event", "ranking"], ["anomaly", "rare_event", "review_budgeted"])[0])
        ok, why = journeys._workflow_ok(e, ["anomaly", "entity_tables", "rare_event", "ranking"], ["anomaly", None, "rare_event", "review_budgeted"])
        self.assertFalse(ok)
        self.assertIn("beyond the expert's", why)

    def test_a_part_the_files_cannot_solve_has_no_approach_and_a_wrong_one_is_wrong(self):
        e = self.expect(["entity_tables"], [None])
        self.assertTrue(journeys._workflow_ok(e, ["entity_tables"], [None])[0])
        self.assertFalse(journeys._workflow_ok(e, ["entity_tables"], ["entity_tables"])[0])

    def test_the_suite_bar_is_nine_of_ten_and_no_double_read_and_no_refused_move(self):
        def rows(right, n=10, **extra):
            return [{"right": i < right, "parts": {}, **extra} for i in range(n)]

        self.assertTrue(journeys.verdict(rows(9)).startswith("pass"))
        self.assertTrue(journeys.verdict(rows(8)).startswith("fail"))
        self.assertTrue(journeys.verdict(rows(10, refused_moves=1)).startswith("fail"))
        self.assertTrue(journeys.verdict(rows(10, holdout_uses=2)).startswith("fail"))
        self.assertEqual(journeys.wilson(9, 10)[0], 0.596)
        self.assertGreaterEqual(0.596, journeys.MIN_LOW)

    def test_a_part_that_does_not_apply_is_not_scored(self):
        row = {"parts": {"card": True, "workflow": True, "pipeline": None, "holdout": None}}
        self.assertTrue(journeys._end(row, 0.0)["right"])
        row = {"parts": {"card": True, "workflow": False, "pipeline": None, "holdout": None}}
        self.assertFalse(journeys._end(row, 0.0)["right"])
        self.assertFalse(journeys._end({"parts": {"card": None, "workflow": None, "pipeline": None, "holdout": None}}, 0.0)["right"])

    def test_the_roles_must_be_right_on_as_many_journeys_and_parts_as_the_rules_with_no_failed_request(self):
        def arm(*rights, parts=None):
            return [{"id": f"J{i}", "right": r, "parts": parts or {p: r for p in journeys.PARTS}} for i, r in enumerate(rights)]

        same = journeys.compare({"rules": arm(*[True] * 9, False), "roles": arm(*[True] * 9, False)})
        self.assertTrue(same["passed"])
        worse = journeys.compare({"rules": arm(*[True] * 9, False), "roles": arm(*[True] * 8, False, False)})
        self.assertFalse(worse["passed"])
        self.assertEqual(worse["changed"], [{"id": "J8", "rules": True, "roles": False}])
        better = journeys.compare({"rules": arm(*[True] * 8, False, False), "roles": arm(*[True] * 9, False)})
        self.assertTrue(better["passed"], "right on more journeys than the rules and on the suite bar")
        failed = journeys.compare({"rules": arm(*[True] * 10), "roles": arm(*[True] * 10)}, failures=1)
        self.assertFalse(failed["passed"], "a failed request: the rules answered for a role")
        below_bar = journeys.compare({"rules": arm(*[True] * 7, False, False, False), "roles": arm(*[True] * 7, False, False, False)})
        self.assertFalse(below_bar["passed"], "as good as the rules is not enough when the rules are below the bar")


@unittest.skipIf(MISSING, MISSING)
class SealedTest(unittest.TestCase):
    def test_the_sealed_set_is_the_independent_authors_file_unedited(self):
        self.assertEqual(journeys.fingerprint(journeys.SPLITS["sealed"]), journeys.SEALED_HASH.read_text().strip())
        sealed = journeys.load("sealed")
        self.assertEqual([j["id"] for j in sealed], [f"JS-{i:02d}" for i in range(1, 11)])
        self.assertEqual(journeys.GATE["split"], "sealed")
        self.assertNotIn(journeys.GATE["seed"], (0, 10, 11, 12, 13, 14, 15), "not the dev seed, nor a calibration seed")

    def test_an_edited_sealed_set_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            hashed = Path(tmp) / "sealed.sha256"
            hashed.write_text("0" * 64)
            with mock.patch.object(journeys, "SEALED_HASH", hashed):
                with self.assertRaises(ValueError) as cm:
                    journeys.load("sealed")
        self.assertIn("never edited", str(cm.exception))

    def test_the_gate_was_fixed_in_code_with_the_thresholds_the_verdict_uses(self):
        self.assertEqual((journeys.MIN_SHARE, journeys.MIN_LOW), (0.85, 0.55))
        self.assertIn("85%", journeys.GATE["suite"])
        self.assertIn("0.55", journeys.GATE["suite"])

    def test_a_second_sealed_run_is_refused_and_the_sealed_set_is_run_whole_with_both_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            (results / "JPR-001_sealed.json").write_text("{}")
            with mock.patch.object(journeys, "RESULTS", results):
                self.assertEqual(journeys.main(["--split", "sealed", "--yes", "--cap-eur", "0.5"]), 2)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(journeys, "RESULTS", Path(tmp)):
                self.assertEqual(journeys.main(["--split", "sealed", "--only", "JS-01"]), 2)
                self.assertEqual(journeys.main(["--split", "sealed", "--no-store"]), 2)
                self.assertEqual(journeys.main(["--split", "sealed", "--seed", "0"]), 2)

    def test_without_yes_the_sealed_run_only_prints_its_plan_and_sends_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(journeys, "RESULTS", Path(tmp)), mock.patch.object(journeys, "measure") as measured:
            self.assertEqual(journeys.main(["--split", "sealed"]), 0)
            measured.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_a_live_arm_needs_a_cap_and_a_run_with_a_failed_request_is_not_stored(self):
        rows = [{"id": "J", "right": True, "why": [], "parts": {p: True for p in journeys.PARTS}}]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(journeys, "RESULTS", Path(tmp)), mock.patch.object(journeys, "measure", return_value=rows):
            measured = journeys.measure
            self.assertEqual(journeys.main(["--split", "dev", "--yes"]), 2, "a remote model needs a euro cap above zero")
            self.assertEqual(journeys.main(["--split", "dev", "--yes", "--cap-eur", "0"]), 2)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_a_roles_arm_that_no_model_answered_is_refused_not_scored_against_the_rules(self):
        """Every ask returned nothing (a private scope, no key): the arm would be the rules again and 'as good as the rules' would prove nothing."""
        rows = [{"id": "J", "right": True, "why": [], "parts": {p: True for p in journeys.PARTS}}]

        class Gate:
            shadows = None

            def available(self, purpose):
                return True

            def route(self, purpose):
                return ("standard",)

            def client(self, purpose, **kw):
                return None  # no model serves it here: the advisor answers nothing

        def measured(split, seed, live=None, only=None):
            if live is not None:
                live["map_advisor"].ask("decomposer", {"words": "x"})
            return rows

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(journeys, "RESULTS", Path(tmp)), mock.patch.object(journeys, "measure", side_effect=measured), \
                mock.patch("dclab_rnd.models.gateway.for_workspace", return_value=Gate()), mock.patch("dclab_rnd.models.prices.estimate", return_value=0.001):
            self.assertEqual(journeys.main(["--split", "dev", "--yes", "--cap-eur", "0.5"]), 1)
            self.assertEqual(list(Path(tmp).iterdir()), [], "nothing is stored")

    def test_the_roles_arm_leaves_the_private_scope_that_would_forbid_its_requests(self):
        import inspect

        source = inspect.getsource(journeys.play)
        self.assertIn("if advisor is None else contextlib.nullcontext()", source)

    def test_the_roles_passed_only_from_a_stored_sealed_roles_arm_of_this_exact_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            with mock.patch.object(journeys, "RESULTS", results):
                self.assertFalse(journeys.roles_passed())
                good = {"arm": "roles", "journeys": journeys.fingerprint(journeys.SPLITS["sealed"]), "roles_against_rules": {"passed": True}}
                (results / "JPR-002_sealed.json").write_text(json.dumps(good))
                self.assertTrue(journeys.roles_passed())
                (results / "JPR-002_sealed.json").write_text(json.dumps({**good, "journeys": "0" * 64}))
                self.assertFalse(journeys.roles_passed(), "measured on another set of people")
                (results / "JPR-002_sealed.json").write_text(json.dumps({**good, "roles_against_rules": {"passed": False}}))
                self.assertFalse(journeys.roles_passed())


if __name__ == "__main__":
    unittest.main()
