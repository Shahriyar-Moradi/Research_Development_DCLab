"""Package 24.3: the solution router and its catalogue. For each sub-problem the approaches of the catalogue for its family, ranked, each need judged
met, unmet or unknown from the data card and the map; the chosen one is the first with no need refuted; a single problem keeps the kind the host's
detection found. On every replica the first candidate is the kind its acceptance runs, and the owner's fraud problem routes to a screen with no labels,
a rare event with partial labels, entity tables and a budgeted review. A model's choice is refused unless it is a candidate the data does not refute."""

import json
import sys
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core import datacard as dc, problem_map as pm, solutions as sol
    from dclab_core.agents import ScriptedAdvisor
    from dclab_core.bands import FAMILIES
    from dclab_core.contracts import CONTRACTS
except ImportError as error:  # pydantic and pandas are not in every environment
    MISSING = str(error)
else:
    MISSING = ""

FRAUD = ("Find the fake rides and fake orders our bikers and customers make. We have a few confirmed cases, nothing else is labelled, "
         "and the team can review 200 a day.")
CHURN = "I want to know which subscribers will cancel at their renewal."
# the family and label situation of the problem each kind's replica poses, and whether its outcome is named
KIND_PROBLEM = {"binary": ("classification", "column"), "multiclass": ("classification", "column"), "text_tabular": ("classification", "column"),
                "regression": ("regression", "column"), "places": ("regression", "column"), "rare_event": ("rare_event", "file"), "anomaly": ("anomaly", "none"),
                "forecasting": ("forecasting", "column"), "time_to_event": ("time_to_event", "column"), "ranking": ("ranking", "none"),
                "segmentation": ("segmentation", "none"), "uplift": ("uplift", "column"), "entity_tables": ("entity_tables", "none"), "streams": ("streams", "none"),
                "embedded": ("text_or_images", "column")}


def single(family, labels="column"):
    sub = pm.SubProblem(id="sp1", part="part1", title="The problem", family=family, labels=labels)
    return sub, pm.ProblemMap(size="single", parts=[pm.Part(id="part1", title="The problem")], sub_problems=[sub])


@unittest.skipIf(MISSING, MISSING)
class CatalogueTest(unittest.TestCase):
    def test_every_family_has_an_approach_and_every_approach_a_known_need(self):
        served = {f for a in sol.CATALOGUE for f in a.families}
        self.assertEqual(served, set(FAMILIES))
        for a in sol.CATALOGUE:
            self.assertTrue(set(a.needs) <= set(sol.PRECONDITIONS), a.id)
            self.assertEqual(sol.BY_ID[a.id], a)
        self.assertEqual(len({a.id for a in sol.CATALOGUE}), len(sol.CATALOGUE))

    def test_the_catalogue_is_the_studios_kinds_by_key_and_track_with_one_composition(self):
        from dclab_rnd import kinds

        keys = {k.key for k in kinds.all_kinds()}
        compositions = {a.id for a in sol.CATALOGUE if a.engine == "composition"}
        self.assertEqual(compositions, {"review_budgeted"})
        for a in sol.CATALOGUE:
            if a.engine == "engine":
                self.assertIn(a.id, keys, a.id)
                self.assertEqual(a.evidence_track, kinds.get(a.id).evidence_track, a.id)  # the track that measured it, as the kind says

    def test_every_candidate_names_a_track_or_is_a_composition(self):
        for a in sol.CATALOGUE:
            self.assertTrue(a.evidence_track or a.engine == "composition", a.id)

    def test_route_is_a_versioned_contract_that_reads_back(self):
        sub, m = single("classification")
        r = sol.route(sub, None, m)
        self.assertEqual(sol.Route.read(r.model_dump_json()), r)
        self.assertIn("route", CONTRACTS)


@unittest.skipIf(MISSING, MISSING)
class CheckTest(unittest.TestCase):
    def table(self, **columns):
        import pandas as pd

        n = 60
        frame = pd.DataFrame({name: values(n) for name, values in columns.items()})
        return dc.card_of({"t": frame}, name="t", target="y" if "y" in frame else None)

    def test_a_need_is_unknown_until_the_data_or_the_words_say(self):
        sub, _ = single("classification", labels="unknown")
        self.assertEqual(sol.check("labels", sub)[0], None)
        self.assertEqual(sol.check("text_column", sub)[0], None)  # no table yet
        self.assertEqual(sol.check("binary_target", sub)[0], None)
        self.assertEqual(sol.check("labels", single("classification", "none")[0])[0], False)
        self.assertEqual(sol.check("labels", single("classification", "file")[0])[0], True)

    def test_a_target_of_two_values_is_binary_and_of_four_is_many_classes(self):
        sub, _ = single("classification")
        two = self.table(x=lambda n: list(range(n)), y=lambda n: [i % 2 for i in range(n)])
        four = self.table(x=lambda n: list(range(n)), y=lambda n: [i % 4 for i in range(n)])
        self.assertTrue(sol.check("binary_target", sub, two)[0] and not sol.check("many_classes", sub, two)[0])
        self.assertTrue(sol.check("many_classes", sub, four)[0] and not sol.check("binary_target", sub, four)[0])
        self.assertIn("4 values", sol.check("many_classes", sub, four)[1])

    def test_a_check_says_what_it_found_in_names_and_counts_never_a_value(self):
        sub, _ = single("rare_event", "file")
        data = self.table(x=lambda n: list(range(n)), y=lambda n: [i % 2 for i in range(n)])
        for code in ("binary_target", "time_column", "entity_column", "several_tables", "place_columns", "text_column"):
            why = sol.check(code, sub, data)[1]
            self.assertLess(len(why), 120, code)
            self.assertNotRegex(why, r"\b\d{3,}\b")

    def test_an_unknown_precondition_is_refused(self):
        with self.assertRaises(ValueError):
            sol.check("telepathy", single("classification")[0])

    def test_a_treatment_and_a_duration_are_found_by_the_words_of_the_columns_names(self):
        sub, _ = single("uplift")
        for name in ("got_offer", "gotOffer", "Got Offer", "treated", "control_group"):
            data = self.table(**{name: lambda n: [i % 2 for i in range(n)], "y": lambda n: [i % 2 for i in range(n)]})
            self.assertTrue(sol.check("treatment_column", sub, data)[0], name)
        plain = self.table(x=lambda n: list(range(n)), y=lambda n: [i % 2 for i in range(n)])
        self.assertFalse(sol.check("treatment_column", sub, plain)[0])
        sub, _ = single("time_to_event")
        self.assertTrue(sol.check("duration_column", sub, self.table(days_subscribed=lambda n: list(range(n)), y=lambda n: [i % 2 for i in range(n)]))[0])


@unittest.skipIf(MISSING, MISSING)
class RouteTest(unittest.TestCase):
    def test_the_owners_fraud_problem_routes_its_parts_to_a_screen_a_rare_event_entity_tables_and_a_budgeted_review(self):
        m = pm.decompose(FRAUD)
        got = {sid: r.chosen for sid, r in sol.routes(m).items()}
        self.assertEqual(got, {"sp1": "anomaly", "sp2": "rare_event", "sp3": "entity_tables", "sp4": "review_budgeted"})

    def test_a_review_list_is_a_composition_and_not_a_ranking_of_items(self):
        r = sol.routes(pm.decompose(FRAUD))["sp4"]
        by = {c.approach: c for c in r.candidates}
        self.assertEqual(by["ranking"].blocking, ["not_budget_step"])
        self.assertEqual(by["review_budgeted"].blocking, [])
        sub, m = single("ranking", "none")
        self.assertEqual(sol.route(sub, None, m).chosen, "ranking")  # outside a budget step it is the ranking kind

    def test_the_alternatives_are_listed_with_what_blocks_them(self):
        sub, m = single("classification", "column")
        r = sol.route(sub, None, m)
        self.assertEqual(r.candidates[0].approach, r.chosen)  # nothing is refuted, so the first is the one chosen
        self.assertTrue(all(c.rank == i for i, c in enumerate(r.candidates, start=1)))
        self.assertGreaterEqual(len(r.candidates), 3)
        self.assertTrue(all(c.reasons or c.unknown for c in r.candidates))

    def test_a_refuted_approach_is_never_chosen_even_when_it_is_more_specific(self):
        import pandas as pd

        frame = pd.DataFrame({"x": range(80), "y": [i % 2 for i in range(80)]})
        data = dc.card_of({"t": frame}, name="t", target="y")
        sub, m = single("classification", "column")
        r = sol.route(sub, data, m)
        by = {c.approach: c for c in r.candidates}
        self.assertEqual(r.chosen, "binary")
        self.assertEqual(by["text_tabular"].blocking, ["text_column"])  # more specific, but no text here
        self.assertEqual(by["multiclass"].blocking, ["many_classes"])

    def test_every_approach_refuted_says_so_and_chooses_none(self):
        sub, m = single("uplift", "none")  # nothing is labelled: an uplift model has nothing to learn from
        r = sol.route(sub, None, m)
        self.assertIsNone(r.chosen)
        self.assertIn("refuted", r.reason)

    def test_unknown_needs_do_not_block_they_are_the_questions_still_to_settle(self):
        sub, m = single("classification", "unknown")
        r = sol.route(sub, None, m)
        self.assertIsNotNone(r.chosen)
        first = next(c for c in r.candidates if c.approach == r.chosen)
        self.assertIn("labels", first.unknown)
        self.assertIn("still to settle", r.reason)

    def test_a_single_problem_keeps_the_kind_the_detection_found_and_the_rest_are_still_listed(self):
        from dclab_rnd import kinds

        journeys = json.loads((ROOT / "dclab_rnd" / "agent_eval" / "journeys" / "journeys_dev.json").read_text(encoding="utf-8"))
        for journey in journeys:
            detected = kinds.detect(journey["words"])["key"]
            m = pm.decompose(journey["words"])
            sub = m.sub_problems[0]
            r = sol.route(sub, None, m, detected=detected)
            if detected in sol.BY_ID:
                self.assertEqual((r.chosen, r.source, r.single), (detected, "detected", True), journey["id"])
                self.assertIn(detected, [c.approach for c in r.candidates], journey["id"])

    def test_the_detection_of_a_kind_the_catalogue_does_not_know_is_not_used(self):
        sub, m = single("classification")
        r = sol.route(sub, None, m, detected="astrology")
        self.assertEqual(r.source, "rules")

    def test_a_complex_map_ignores_a_detected_kind(self):
        m = pm.decompose(FRAUD)
        self.assertEqual(sol.route(m.sub_problems[0], None, m, detected="binary").chosen, "anomaly")

    def test_the_route_names_the_map_it_was_made_for(self):
        m = pm.decompose(FRAUD)
        self.assertEqual({r.map for r in sol.routes(m).values()}, {m.fingerprint})

    def test_the_same_inputs_give_the_same_route(self):
        m = pm.decompose(FRAUD)
        self.assertEqual({k: v.model_dump() for k, v in sol.routes(m).items()}, {k: v.model_dump() for k, v in sol.routes(m).items()})


@unittest.skipIf(MISSING, MISSING)
class NeedsTest(unittest.TestCase):
    """What the review of 24.3 found: needs judged on what is known, never guessed."""

    def frame(self, **columns):
        import pandas as pd

        n = 80
        return pd.DataFrame({name: values(n) for name, values in columns.items()})

    def card(self, tables, target=None, roles=None):
        return dc.card_of(tables, name="t", target=target, roles=roles or {})

    def test_an_outcome_nobody_named_is_unknown_never_the_tables_first_candidate(self):
        import pandas as pd

        frame = pd.DataFrame({"has_attachment": [i % 2 for i in range(80)], "assigned_team": [i % 5 for i in range(80)], "x": range(80)})
        data = self.card({"t": frame})  # no target named: the card's candidates are not an answer
        sub, m = single("classification", "column")
        self.assertIsNone(sol.check("binary_target", sub, data, m)[0])
        self.assertIsNone(sol.check("many_classes", sub, data, m)[0])
        r = sol.route(sub, data, m)
        self.assertIn("still to settle", r.reason)
        self.assertTrue(all(not c.blocking for c in r.candidates if c.approach in ("binary", "multiclass")))
        named = self.card({"t": frame}, target="assigned_team")
        self.assertEqual(sol.route(sub, named, m).chosen, "multiclass")  # named, it settles the question

    def test_a_card_that_names_one_outcome_is_judged_only_for_the_part_that_owns_it(self):
        import pandas as pd

        frame = pd.DataFrame({"x": range(80), "y": [i % 2 for i in range(80)]})
        data = self.card({"t": frame}, target="y")
        m = pm.decompose("", parts=[{"title": "Part 1", "text": "Which customers will cancel?"}, {"title": "Part 2", "text": "Estimate the price of each flat."}])  # two parts that each learn from an outcome
        for s in m.sub_problems:
            self.assertIsNone(sol.check("binary_target", s, data, m)[0], s.id)
        one, single_map = single("classification", "column")
        self.assertTrue(sol.check("binary_target", one, data, single_map)[0])

    def test_a_sub_problems_own_outcome_words_name_its_column(self):
        import pandas as pd

        frame = pd.DataFrame({"x": range(80), "churned": [i % 2 for i in range(80)], "price": [float(i) for i in range(80)]})
        data = self.card({"t": frame})
        m = pm.decompose("", parts=[{"title": "Part 1", "text": "Which customers will cancel?"}, {"title": "Part 2", "text": "Estimate the price of each flat."}])
        first = m.sub_problems[0].model_copy(update={"outcome": "churned"})
        second = m.sub_problems[1].model_copy(update={"outcome": "the price of each flat"})
        self.assertTrue(sol.check("binary_target", first, data, m)[0])
        self.assertTrue(sol.check("number_target", second, data, m)[0])

    def test_a_number_is_met_for_a_float_unknown_for_a_few_whole_values_refuted_for_text(self):
        import pandas as pd

        sub, m = single("regression", "column")
        floats = self.card({"t": pd.DataFrame({"y": [i / 7 for i in range(80)]})}, target="y")
        count = self.card({"t": pd.DataFrame({"y": [i % 11 for i in range(80)]})}, target="y")
        text = self.card({"t": pd.DataFrame({"y": ["a", "b", "c", "d"] * 20})}, target="y")
        self.assertTrue(sol.check("number_target", sub, floats, m)[0])
        self.assertIsNone(sol.check("number_target", sub, count, m)[0])
        self.assertFalse(sol.check("number_target", sub, text, m)[0])
        self.assertIsNotNone(sol.route(sub, count, m).chosen)  # a count of eleven values does not refute a regression

    def test_two_columns_of_place_need_a_latitude_and_a_longitude_not_a_city_and_a_zip(self):
        sub, m = single("regression", "column")
        cityzip = self.card({"t": self.frame(city=lambda n: [str(i % 7) for i in range(n)], zip=lambda n: [str(10000 + i % 9) for i in range(n)], y=lambda n: [float(i) for i in range(n)])}, target="y")
        points = self.card({"t": self.frame(lat=lambda n: [i / 100 for i in range(n)], lon=lambda n: [i / 90 for i in range(n)], y=lambda n: [float(i) for i in range(n)])}, target="y")
        self.assertFalse(sol.check("place_columns", sub, cityzip, m)[0])
        self.assertTrue(sol.check("place_columns", sub, points, m)[0])
        self.assertEqual(sol.route(sub, points, m).chosen, "places")
        self.assertEqual(sol.route(sub, cityzip, m).chosen, "regression")

    def test_a_file_of_known_cases_is_not_a_table_of_events(self):
        sub, m = single("entity_tables", "none")
        events = self.frame(entity_id=lambda n: [i % 9 for i in range(n)], amount=lambda n: [float(i) for i in range(n)])
        main = self.frame(entity_id=lambda n: list(range(n)), x=lambda n: [float(i) for i in range(n)])
        with_cases = self.card({"main": main, "cases": events}, roles={"cases": "labels"})
        with_events = self.card({"main": main, "events": events}, roles={"events": "related"})
        self.assertFalse(sol.check("several_tables", sub, with_cases, m)[0])
        self.assertTrue(sol.check("several_tables", sub, with_events, m)[0])
        self.assertIsNone(sol.route(sub, with_cases, m).chosen)  # the rings part needs the table of events: said, not pretended

    def test_a_treatment_or_duration_nobody_found_by_name_is_unknown_not_refuted(self):
        sub, m = single("uplift", "column")
        data = self.card({"t": self.frame(ab_group=lambda n: [i % 2 for i in range(n)], y=lambda n: [i % 2 for i in range(n)])}, target="y")
        self.assertIsNone(sol.check("treatment_column", sub, data, m)[0])  # 'ab_group' is a treatment the name does not say
        self.assertIsNotNone(sol.route(sub, data, m).chosen)
        sub, m = single("time_to_event", "column")
        data = self.card({"t": self.frame(observed=lambda n: list(range(n)), y=lambda n: [i % 2 for i in range(n)])}, target="y")
        self.assertIsNone(sol.check("duration_column", sub, data, m)[0])
        self.assertIsNotNone(sol.route(sub, data, m).chosen)

    def test_a_data_card_with_no_purposes_leaves_time_entity_and_place_unknown(self):
        sub, m = single("forecasting", "column")
        data = self.card({"t": self.frame(day=lambda n: list(range(n)), store=lambda n: [i % 4 for i in range(n)], units=lambda n: [float(i) for i in range(n)])}, target="units")
        bare = data.model_copy(update={"tables": [t.model_copy(update={"columns": [c.model_copy(update={"purpose": None}) for c in t.columns]}) for t in data.tables]})
        for code in ("time_column", "entity_column", "place_columns"):
            self.assertIsNone(sol.check(code, sub, bare, m)[0], code)
        self.assertIsNotNone(sol.route(sub, bare, m).chosen)

    def test_a_rare_event_with_labels_not_yet_said_routes_to_the_rare_event_kind_not_the_screen(self):
        sub, m = single("rare_event", "unknown")
        r = sol.route(sub, None, m)
        self.assertEqual(r.chosen, "rare_event")
        self.assertIn("labels", r.candidates[0].unknown)
        self.assertEqual([c.approach for c in r.candidates], ["rare_event", "anomaly"])

    def test_the_kind_a_map_names_is_asked_first_among_those_not_refuted(self):
        sub = pm.SubProblem(id="sp1", part="part1", title="t", family="classification", kind="multiclass", labels="column")
        m = pm.ProblemMap(size="single", parts=[pm.Part(id="part1", title="t")], sub_problems=[sub])
        self.assertEqual(sol.route(sub, None, m).chosen, "multiclass")

    def test_the_streams_and_uplift_and_forecasting_needs_are_the_kinds_real_ones(self):
        needs = {a.id: set(a.needs) for a in sol.CATALOGUE}
        self.assertIn("several_tables", needs["streams"])  # the readings and the incidents
        self.assertIn("binary_target", needs["uplift"])
        self.assertIn("number_target", needs["forecasting"])


@unittest.skipIf(MISSING, MISSING)
class DetectedTest(unittest.TestCase):
    def test_a_detected_kind_the_data_refutes_is_still_chosen_and_the_refutation_is_said(self):
        import pandas as pd

        frame = pd.DataFrame({"x": range(80), "y": [i % 4 for i in range(80)]})
        data = dc.card_of({"t": frame}, name="t", target="y")
        sub, m = single("classification", "column")
        r = sol.route(sub, data, m, detected="binary")
        self.assertEqual((r.chosen, r.source), ("binary", "detected"))
        self.assertIn("refutes it: it needs binary_target", r.reason)
        self.assertEqual(next(c for c in r.candidates if c.approach == "binary").blocking, ["binary_target"])

    def test_a_detected_kind_of_another_family_is_chosen_for_a_single_map_the_rules_drew_and_the_difference_is_said(self):
        sub, m = single("classification", "column")
        r = sol.route(sub, None, m, detected="rare_event")
        self.assertEqual((r.chosen, r.source), ("rare_event", "detected"))
        self.assertIn("where the map reads classification", r.reason)

    def test_a_single_map_the_person_edited_says_its_own_family_and_ignores_the_detection(self):
        sub, m = single("entity_tables", "none")
        mine = pm.ProblemMap.model_validate({**m.model_dump(mode="json"), "fingerprint": "", "source": "person"})
        r = sol.route(sub, None, mine, detected="binary")
        self.assertEqual(r.source, "rules")
        self.assertIn("the person's own reading", r.reason)
        self.assertNotEqual(r.chosen, "binary")

    def test_every_kind_of_the_engine_is_in_the_catalogue_so_a_detection_is_never_silently_replaced(self):
        from dclab_rnd import kinds

        self.assertEqual(sorted(k.key for k in kinds.all_kinds() if k.key not in sol.BY_ID), [])


@unittest.skipIf(MISSING, MISSING)
class ReplicaTest(unittest.TestCase):
    """The done-when: on each replica the router's first candidate is the kind its acceptance runs."""

    def data_of(self, key):
        from dclab_rnd import kinds

        case = kinds.get(key).case(0)
        tables, roles = {"table": case.table}, {}
        if case.labels is not None:
            tables["cases"], roles["cases"] = case.labels[0], "labels"
        for name, frame in (case.tables or {}).items():
            tables[name], roles[name] = frame, "related"
        return dc.card_of(tables, name=key, roles=roles, target=case.target if case.target in case.table.columns else None)

    def test_every_kinds_replica_routes_first_to_that_kind(self):
        warnings.filterwarnings("ignore")
        from dclab_rnd import kinds

        self.assertEqual(set(KIND_PROBLEM), {k.key for k in kinds.all_kinds() if k.key in sol.BY_ID})
        for key, (family, labels) in KIND_PROBLEM.items():
            with self.subTest(kind=key):
                sub, m = single(family, labels)
                r = sol.route(sub, self.data_of(key), m)
                self.assertEqual(r.chosen, key, [(c.approach, c.blocking) for c in r.candidates])
                self.assertEqual(r.candidates[0].approach, key)
                self.assertEqual(r.candidates[0].blocking, [])

    def test_a_kinds_alternatives_say_what_blocks_them_where_the_data_refutes_them(self):
        warnings.filterwarnings("ignore")
        sub, m = single("classification", "column")
        r = sol.route(sub, self.data_of("binary"), m)
        by = {c.approach: c for c in r.candidates}
        self.assertEqual((by["text_tabular"].blocking, by["multiclass"].blocking), (["text_column"], ["many_classes"]))
        sub, m = single("regression", "column")
        self.assertEqual({c.approach: c.blocking for c in sol.route(sub, self.data_of("regression"), m).candidates}["places"], ["place_columns"])


@unittest.skipIf(MISSING, MISSING)
class RoleTest(unittest.TestCase):
    def setUp(self):
        self.m = pm.decompose(FRAUD)
        self.sub = self.m.sub_problems[1]  # the classifier on the confirmed cases: rare_event, with anomaly as the alternative

    def ask(self, reply, sub=None):
        return sol.route_with(sub or self.sub, None, self.m, advisor=ScriptedAdvisor({"solution_router": reply}))

    def test_a_choice_among_the_candidates_with_a_reason_is_taken(self):
        r, why = self.ask({"choice": "anomaly", "reason": "only a few cases are confirmed, so a screen may read them better"})
        self.assertEqual((r.chosen, r.source), ("anomaly", "advisor"))
        self.assertIn("a screen", r.reason)
        self.assertEqual(why, "accepted after validation")

    def test_agreeing_with_the_rules_is_noted(self):
        r, _ = self.ask({"choice": "rare_event", "reason": "labels exist for some cases"})
        self.assertEqual((r.chosen, r.source), ("rare_event", "advisor"))
        self.assertTrue(r.reason.startswith("the advisor agrees"))

    def test_an_approach_outside_the_catalogue_or_the_candidates_is_refused(self):
        for choice in ("deep_magic", "forecasting", "binary", None, 7):
            r, why = self.ask({"choice": choice, "reason": "because"})
            self.assertEqual((r.chosen, r.source, why), ("rare_event", "rules", "invalid"), choice)

    def test_an_approach_the_data_refutes_is_refused(self):
        sub, m = single("classification", "column")
        import pandas as pd

        frame = pd.DataFrame({"x": range(80), "y": [i % 2 for i in range(80)]})
        data = dc.card_of({"t": frame}, name="t", target="y")
        r, why = sol.route_with(sub, data, pm.ProblemMap.model_validate({**m.model_dump(mode="json"), "fingerprint": "", "size": "complex",
                                                                        "parts": [{"id": "part1", "title": "t"}, {"id": "part2", "title": "u"}],
                                                                        "sub_problems": [sub.model_dump(mode="json"), {**sub.model_dump(mode="json"), "id": "sp2", "part": "part2"}],
                                                                        "composition": {"method": "sequence", "steps": [{"kind": "sequence", "inputs": ["sp1", "sp2"], "output": "x"}]}}),
                                advisor=ScriptedAdvisor({"solution_router": {"choice": "multiclass", "reason": "there may be several classes"}}))
        self.assertEqual((r.chosen, why), ("binary", "invalid"))

    def test_a_choice_whose_needs_are_not_yet_known_is_refused_unless_it_is_the_rules_own(self):
        sub, m = single("classification", "unknown")
        advisor = ScriptedAdvisor({"solution_router": {"choice": "multiclass", "reason": "there may be several classes"}})
        r, why = sol.route_with(sub, None, pm.ProblemMap.model_validate({**m.model_dump(mode="json"), "fingerprint": "", "size": "complex", "parts": [{"id": "part1", "title": "t"}, {"id": "part2", "title": "u"}],
                                                                        "sub_problems": [sub.model_dump(mode="json"), {**sub.model_dump(mode="json"), "id": "sp2", "part": "part2"}],
                                                                        "composition": {"method": "sequence", "steps": [{"kind": "sequence", "inputs": ["sp1", "sp2"], "output": "x"}]}}), advisor=advisor)
        self.assertEqual((r.chosen, r.source, why), ("text_tabular", "rules", "invalid"))  # no data card: multiclass's needs are unknown, and it is not the rules' own choice

    def test_a_choice_without_its_reason_is_refused(self):
        r, why = self.ask({"choice": "anomaly", "reason": "  "})
        self.assertEqual((r.chosen, why), ("rare_event", "invalid"))

    def test_a_single_problem_is_never_asked_and_keeps_its_kind(self):
        m = pm.decompose(CHURN)
        advisor = ScriptedAdvisor({"solution_router": {"choice": "multiclass", "reason": "x"}})
        r, why = sol.route_with(m.sub_problems[0], None, m, detected="binary", advisor=advisor)
        self.assertEqual((r.chosen, r.source, advisor.requests), ("binary", "detected", []))
        self.assertIn("single", why)

    def test_a_failing_or_silent_advisor_leaves_the_rules(self):
        class Fails:
            def ask(self, purpose, evidence):
                raise RuntimeError("down")

        r, why = sol.route_with(self.sub, None, self.m, advisor=Fails())
        self.assertEqual((r.chosen, r.source, why), ("rare_event", "rules", "failed"))
        r, why = self.ask(None)
        self.assertEqual((r.source, why), ("rules", "the advisor abstained"))

    def test_the_evidence_is_names_and_lines_never_rows(self):
        advisor = ScriptedAdvisor({"solution_router": None})
        sol.route_with(self.sub, None, self.m, advisor=advisor)
        (purpose, evidence), = advisor.requests
        self.assertEqual(purpose, "solution_router")
        self.assertIn("rare_event", evidence["approaches"])
        self.assertTrue(all(isinstance(line, str) for line in evidence["candidates"]))
        self.assertIn("sp2", evidence["sub_problem"])

    def test_the_role_is_registered_with_its_prompt(self):
        from dclab_core import prompts
        from dclab_core.agents import roles

        self.assertIn("solution_router", roles())
        self.assertIn("solution_router", prompts.names())


if __name__ == "__main__":
    unittest.main()
