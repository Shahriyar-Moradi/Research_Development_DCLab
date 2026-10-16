"""Package 24.5: the programme. A solution map run leaf by leaf, the composition chosen on validation, the holdout opened only when every leaf ran and read once."""

from __future__ import annotations

import json
import unittest
from functools import lru_cache

import pandas as pd

from dclab_core import datacard, leaves, programme as pg, problem_map as pm, solution_map as sm_mod, solutions
from dclab_core.contracts import CONTRACTS
from dclab_core.replicas import fraud_programme as fp

WORDS = ("Find the fake rides and fake orders our bikers and customers make. We have a few confirmed cases, nothing else is labelled, "
         "and the team can review about 2% of the rides each day.")
SMALL = dict(n_orders=14_000, n_accounts=1_000, n_bikers=90, n_rings=6, n_markers=10)


@lru_cache(maxsize=2)
def _world(seed: int = 0):
    r = fp.generate(seed=seed, **SMALL)
    frame = r.rides.copy()
    m = pm.decompose(WORDS)
    card = datacard.card_of({"rides": frame, "accounts": r.accounts}, roles={"rides": "main", "accounts": "related"})
    sm = sm_mod.solution_map(m, solutions.routes(m, card), card)
    labels = leaves.confirmed_labels(frame, r.cases, "order_id")
    return r, frame, m, sm, labels


def _run(seed: int = 0, **kwargs):
    r, frame, m, sm, labels = _world(seed)
    args = dict(tables={"accounts": r.accounts}, labels=labels, entities=["account_id", "biker_id"], time="created_at", exclude=["order_id"], budget=0.02, seed=seed,
                bootstraps=60)
    args.update(kwargs)
    return pg.execute(sm, m, frame, **args), (r, frame, m, sm, labels)


class PlanTest(unittest.TestCase):
    def test_a_complex_map_has_a_leaf_per_block_and_the_budget_step_is_not_a_model(self):
        _, _, m, sm, _ = _world()
        p = pg.plan(sm, m, budget=0.02)
        self.assertEqual([l.block for l in p.leaves], [b.id for b in sm.blocks])
        by = {l.approach: l for l in p.leaves}
        self.assertEqual(by["review_budgeted"].status, "composition")
        self.assertEqual({by[a].status for a in ("anomaly", "rare_event", "entity_tables")}, {"planned"})
        self.assertEqual(p.fused, ["blk1", "blk2", "blk3"], "the composition combines the three scores, not the budget step")
        self.assertEqual(p.state, "planned")
        self.assertEqual(p.metric, "recall_at_budget")
        self.assertEqual(p.split, sm.holdout.split)
        self.assertEqual(pg.validate(p, sm), [])

    def test_a_single_problem_is_not_a_programme(self):
        m = pm.decompose("Which customers will cancel next month?")
        sm = sm_mod.solution_map(m, solutions.routes(m, None), None)
        self.assertEqual(m.size, "single")
        with self.assertRaises(ValueError) as cm:
            pg.plan(sm, m)
        self.assertIn("not a programme", str(cm.exception))

    def test_an_approach_with_no_runner_is_a_blocker_with_its_reason(self):
        _, _, m, sm, _ = _world()
        data = sm.model_dump()
        data["blocks"] = [{**b, "approach": "forecasting"} if b["id"] == "blk1" else b for b in data["blocks"]]
        changed = sm_mod.SolutionMap.model_validate({**data, "fingerprint": ""})
        p = pg.plan(changed, m)
        leaf = next(l for l in p.leaves if l.block == "blk1")
        self.assertEqual(leaf.status, "blocked")
        self.assertIn("no runner", leaf.why)
        self.assertTrue(any(b.startswith("blk1") for b in p.blockers))

    def test_without_a_budget_the_default_is_said(self):
        _, _, m, sm, _ = _world()
        p = pg.plan(sm, m)
        self.assertTrue(any("1% of the rows" in n for n in p.notes))

    def test_the_contract_is_registered_and_has_a_schema_version(self):
        self.assertIn("programme", CONTRACTS)
        self.assertEqual(pg.Programme.model_fields["schema_version"].default, "dclab_core.programme.v1")


class RunTest(unittest.TestCase):
    def test_a_programme_runs_from_words_to_one_read_of_the_holdout(self):
        run, (r, frame, m, sm, labels) = _run()
        p = run.programme
        self.assertEqual(p.state, "composed")
        self.assertEqual(pg.validate(p, sm), [])
        self.assertEqual(p.holdout.reads, 1)
        self.assertEqual(p.holdout.chosen, p.composition.chosen)
        self.assertEqual({l.block: l.status for l in p.leaves}, {"blk1": "built", "blk2": "built", "blk3": "built", "blk4": "composition"})
        self.assertEqual(set(run.scores), {"blk1", "blk2", "blk3"})
        self.assertEqual(p.periods["holdout"], len(run.scores["blk1"].scores["holdout"]))

    def test_every_leaf_predicts_the_holdout_and_none_scored_it(self):
        run, _ = _run()
        self.assertTrue(all(l.holdout == "predicted" for l in run.programme.leaves if l.status == "built"))
        self.assertTrue(all("holdout" not in l.validation for l in run.programme.leaves))

    def test_the_record_holds_names_counts_and_scores_never_a_row(self):
        run, (r, frame, *_rest) = _run()
        text = json.dumps(run.programme.model_dump(mode="json"))
        for value in list(frame["order_id"].astype(str)) + list(r.accounts["phone"]) + list(r.accounts["device_id"]):
            self.assertNotIn(value, text)

    def test_the_run_is_deterministic(self):
        a, _ = _run()
        b, _ = _run()
        self.assertEqual(a.programme.fingerprint, b.programme.fingerprint)

    def test_the_score_a_leaf_reads_is_from_its_own_protocol(self):
        run, _ = _run()
        clf = next(l for l in run.programme.leaves if l.approach == "rare_event")
        self.assertEqual(clf.reads, ["blk1"])

    def test_the_markdown_has_a_section_per_leaf_the_composition_and_the_holdout(self):
        run, _ = _run()
        text = pg.markdown(run.programme)
        for l in run.programme.leaves:
            self.assertIn(f"## {l.block}:", text)
        self.assertIn("## The composition", text)
        self.assertIn("## The holdout, read once", text)
        self.assertIn("Composed minus best leaf", text)

    def test_the_cluster_for_the_bootstrap_is_the_coarsest_entity(self):
        run, (r, frame, *_rest) = _run()
        self.assertEqual(run.programme.holdout.cluster_key, "biker_id")


class HoldoutRulesTest(unittest.TestCase):
    def _blocked(self):
        r, frame, m, sm, labels = _world()
        return pg.execute(sm, m, frame, tables={}, labels=labels, entities=["account_id", "biker_id"], time="created_at", exclude=["order_id"], budget=0.02, bootstraps=30), (r, frame, m, sm, labels)

    def test_a_leaf_that_cannot_run_blocks_the_programme_and_the_holdout_stays_closed(self):
        run, _ = self._blocked()
        p = run.programme
        self.assertEqual(p.state, "planned")
        self.assertIsNone(p.holdout)
        self.assertIsNone(p.composition)
        blocked = next(l for l in p.leaves if l.block == "blk3")
        self.assertEqual(blocked.status, "blocked")
        self.assertIn("table of accounts", blocked.why)
        self.assertTrue(p.blockers)
        self.assertTrue(any("blocked" in i for i in pg.open_issues(p)))

    def test_the_holdout_cannot_be_read_while_a_leaf_is_blocked(self):
        run, (_, _, _, _, labels) = self._blocked()
        with self.assertRaises(pg.HoldoutClosed) as cm:
            pg.read_holdout(run.programme, run.scores, labels)
        self.assertTrue(cm.exception.issues)

    def test_a_blocked_leaf_the_person_drops_lets_the_rest_be_composed(self):
        run, (r, frame, m, sm, labels) = self._blocked()
        p = pg.drop_leaf(run.programme, "blk3", "no table of accounts exists for this data")
        self.assertEqual(next(l for l in p.leaves if l.block == "blk3").status, "dropped")
        self.assertNotIn("blk3", p.fused)
        self.assertTrue(any("dropped by person" in n for n in p.notes))
        ctx = leaves.Context(frame=frame, parts=leaves.partition(frame, moment="created_at"), time="created_at", entities=["account_id", "biker_id"], labels=labels,
                             exclude=["order_id"], budget=0.02)
        after = pg.resume(pg.Execution(p, run.scores), sm, m, ctx, labels, bootstraps=30)
        self.assertEqual(after.programme.state, "composed")
        self.assertNotIn("blk3", after.programme.composition.candidates)

    def test_a_drop_is_carried_into_the_next_plan_of_the_same_solution_map_and_into_no_other(self):
        run, (r, frame, m, sm, labels) = self._blocked()
        dropped = pg.drop_leaf(run.programme, "blk3", "no table of accounts exists for this data")
        again = pg.replan(sm, m, dropped, budget=0.02)
        self.assertEqual(next(l for l in again.leaves if l.block == "blk3").status, "dropped")
        self.assertIn("no table of accounts", next(l for l in again.leaves if l.block == "blk3").why)
        other = sm_mod.SolutionMap.model_validate({**sm.model_dump(), "notes": ["changed"], "fingerprint": ""})
        self.assertEqual(next(l for l in pg.replan(other, m, dropped, budget=0.02).leaves if l.block == "blk3").status, "planned")

    def test_dropping_needs_a_reason_and_only_a_blocked_or_planned_leaf_is_dropped(self):
        run, _ = self._blocked()
        with self.assertRaises(ValueError):
            pg.drop_leaf(run.programme, "blk3", "  ")
        with self.assertRaises(ValueError):
            pg.drop_leaf(run.programme, "blk1", "I do not like it")  # it ran: that is a result, not a blocker
        with self.assertRaises(ValueError):
            pg.drop_leaf(run.programme, "blk9", "no such leaf")

    def test_a_second_read_is_refused(self):
        run, (_, _, _, _, labels) = _run()
        self.assertTrue(any("already read" in i for i in pg.open_issues(run.programme)))
        with self.assertRaises(pg.HoldoutClosed):
            pg.read_holdout(run.programme, run.scores, labels)

    def test_without_the_confirmed_cases_the_composition_is_not_judged_and_the_holdout_stays_closed(self):
        run, _ = _run(labels=None)
        self.assertNotEqual(run.programme.state, "composed")
        self.assertIsNone(run.programme.holdout)
        self.assertTrue(any("No confirmed case" in n for n in run.programme.notes))

    def test_the_holdout_cannot_open_before_the_composition_is_chosen(self):
        r, frame, m, sm, labels = _world()
        p = pg.plan(sm, m, budget=0.02)
        self.assertTrue(any("not chosen" in i for i in pg.open_issues(p)))


class ReportTest(unittest.TestCase):
    def test_the_report_has_a_section_per_leaf_and_every_number_is_traced_to_the_record(self):
        from dclab_core import report as R

        run, _ = _run()
        rep = pg.report(run.programme)
        titles = [s.title for s in rep.sections]
        for leaf in run.programme.leaves:
            self.assertTrue(any(t.startswith(f"{leaf.block}:") for t in titles), leaf.block)
        self.assertEqual(titles[-2:], ["The composition, chosen on validation", "The holdout, read once"])
        ctx = pg.programme_context(run.programme)
        self.assertEqual(R.check(rep, ctx), [])
        self.assertIn("programme.holdout.difference.low", rep.numbers)
        self.assertAlmostEqual(rep.numbers["programme.holdout.composed.value"], run.programme.holdout.composed.value)

    def test_a_number_that_the_record_does_not_hold_is_found_out(self):
        from dclab_core import report as R

        run, _ = _run()
        rep = pg.report(run.programme)
        ctx = pg.programme_context(run.programme)
        ctx["programme"]["holdout"]["rows"] += 7                                       # a count no other cited number shares
        self.assertTrue(any("does not match the record" in p for p in R.check(rep, ctx)))

    def test_a_programme_that_never_read_the_holdout_reports_what_blocks_it(self):
        r, frame, m, sm, labels = _world()
        run = pg.execute(sm, m, frame, tables={}, labels=labels, entities=["account_id", "biker_id"], time="created_at", exclude=["order_id"], budget=0.02, bootstraps=20)
        rep = pg.report(run.programme)
        self.assertNotIn("The holdout, read once", [s.title for s in rep.sections])
        self.assertIn("What blocks the programme", [s.title for s in rep.sections])


class PlanRulesTest(unittest.TestCase):
    def test_a_solution_map_of_another_problem_map_or_an_unsound_one_is_not_planned(self):
        _, _, m, sm, _ = _world()
        other = pm.decompose("Which customers will cancel next month, and which team should handle each ticket, then how urgent is it?")
        with self.assertRaises(ValueError) as cm:
            pg.plan(sm, other)
        self.assertIn("another problem map", str(cm.exception))
        data = sm.model_dump()
        data["steps"] = data["steps"][::-1]                                          # the composition's steps in another order than the problem map's
        broken = sm_mod.SolutionMap.model_validate({**data, "fingerprint": ""})
        with self.assertRaises(ValueError) as cm:
            pg.plan(broken, m)
        self.assertIn("not sound", str(cm.exception))


class ReadsTest(unittest.TestCase):
    def test_once_the_holdout_was_read_a_leaf_cannot_be_dropped_even_in_a_programme_that_read_nothing(self):
        run, (r, frame, m, sm, labels) = _run()
        first = run.programme
        self.assertEqual(first.reads_before, 0)
        again = pg.replan(sm, m, first, budget=0.02)
        self.assertEqual(again.reads_before, 1, "the second programme on the same split knows the holdout was read once")
        with self.assertRaises(ValueError) as cm:
            pg.drop_leaf(again, "blk3", "I would rather not")
        self.assertIn("already read", str(cm.exception))
        with self.assertRaises(ValueError):
            pg.drop_leaf(first, "blk3", "after the fact")

    def test_the_count_follows_the_split_design_not_the_routes(self):
        run, (r, frame, m, sm, labels) = _run()
        data = sm.model_dump()
        data["notes"] = ["a table was attached and the routes changed"]
        changed = sm_mod.SolutionMap.model_validate({**data, "fingerprint": ""})
        self.assertNotEqual(changed.fingerprint, sm.fingerprint)
        again = pg.replan(changed, m, run.programme, budget=0.02)
        self.assertEqual(again.reads_before, 1, "the holdout rows are the same rows: seeing them under new routes is a second look")
        moved = sm_mod.SolutionMap.model_validate({**data, "holdout": {**data["holdout"], "split": {**data["holdout"]["split"], "share": 0.3}},
                                                   "blocks": [{**b, "split": {**b["split"], "share": 0.3}} for b in data["blocks"]], "fingerprint": ""})
        self.assertEqual(pg.replan(moved, m, run.programme, budget=0.02).reads_before, 0, "another holdout is a holdout nobody has seen")

    def test_a_replay_is_not_a_second_read(self):
        run, (r, frame, m, sm, labels) = _run()
        self.assertEqual(pg.replan(sm, m, run.programme, budget=0.02, replaying=True).reads_before, 0)

    def test_the_report_says_when_it_is_not_the_first_look(self):
        run, (r, frame, m, sm, labels) = _run()
        data = run.programme.model_dump()
        data["reads_before"] = 1
        data["fingerprint"] = ""
        text = pg.markdown(pg.Programme.model_validate(data))
        self.assertIn("already read the holdout 1 time(s)", text)


class ClosedHoldoutTest(unittest.TestCase):
    def _ctx(self, labels, frame, sm):
        return leaves.Context(frame=frame, parts=leaves.partition(frame, moment="created_at"), time="created_at", entities=["account_id", "biker_id"], labels=labels,
                              tables={}, exclude=["order_id"], budget=0.02)

    def test_a_choice_on_too_few_confirmed_cases_keeps_the_holdout_closed_and_unspent(self):
        r, frame, m, sm, labels = _world()
        parts = leaves.partition(frame, moment="created_at")
        starved = labels.copy()
        starved.loc[parts.validation] = 0                                           # no case among the validation rows
        run = pg.execute(sm, m, frame, tables={"accounts": r.accounts}, labels=starved, entities=["account_id", "biker_id"], time="created_at", exclude=["order_id"],
                         budget=0.02, bootstraps=30)
        p = run.programme
        self.assertIsNone(p.holdout)
        self.assertEqual(p.state, "leaves_built")
        self.assertEqual(p.composition.positives, 0)
        self.assertTrue(any("holdout stays closed" in n and "coin toss" in n for n in p.notes))
        self.assertTrue(any("too precious" in i for i in pg.open_issues(p)))

    def test_a_holdout_with_too_few_confirmed_cases_is_not_read(self):
        r, frame, m, sm, labels = _world()
        parts = leaves.partition(frame, moment="created_at")
        starved = labels.copy()
        starved.loc[parts.holdout] = 0
        starved.loc[parts.holdout[:2]] = 1
        run = pg.execute(sm, m, frame, tables={"accounts": r.accounts}, labels=starved, entities=["account_id", "biker_id"], time="created_at", exclude=["order_id"],
                         budget=0.02, bootstraps=30)
        self.assertIsNone(run.programme.holdout)
        self.assertTrue(any("among the holdout rows" in n for n in run.programme.notes))
        self.assertEqual(run.programme.state, "leaves_built")


class OrderTest(unittest.TestCase):
    def test_a_reader_runs_after_the_leaf_it_reads_whatever_order_they_are_listed_in(self):
        from types import SimpleNamespace as NS

        subs = [NS(id="sp1", predict_rank=1, reads=[NS(sub_problem="sp2", what="score")]), NS(id="sp2", predict_rank=1, reads=[])]    # the reader is listed first
        m = NS(sub_problems=subs)
        sm = NS(blocks=[NS(id="blk1", sub_problem="sp1"), NS(id="blk2", sub_problem="sp2")])
        p = NS(leaves=[NS(block="blk1"), NS(block="blk2")])
        self.assertEqual(pg._order(p, m, sm), ["blk2", "blk1"])

    def test_a_reader_whose_source_has_not_run_is_blocked_not_run_without_it(self):
        r, frame, m, sm, labels = _world()
        p = pg.plan(sm, m, budget=0.02)
        parts = leaves.partition(frame, moment="created_at")
        ctx = leaves.Context(frame=frame, parts=parts, time="created_at", entities=["account_id", "biker_id"], labels=labels, tables={}, exclude=["order_id"], budget=0.02)
        started = pg.Execution(p.model_copy(update={"leaves": [l.model_copy(update={"status": "dropped", "why": "dropped by person: x"}) if l.block == "blk1" else l for l in p.leaves]}))
        built = pg.build_leaves(started.programme, sm, m, ctx)
        clf = next(l for l in built.programme.leaves if l.block == "blk2")
        self.assertEqual(clf.status, "blocked")
        self.assertIn("reads blk1", clf.why)


class ComposeLeavesTest(unittest.TestCase):
    def test_the_composition_is_chosen_on_the_validation_scores_and_their_labels_only(self):
        run, (r, frame, m, sm, labels) = _run()
        p = run.programme
        val_idx = run.scores["blk1"].scores["validation"].index
        self.assertEqual(p.composition.positives, int(labels.loc[val_idx].sum()), "the positives it was chosen on are the validation rows'")
        scrambled = {b: leaves.LeafScores(b, {**s.scores, "holdout": s.scores["holdout"].sample(frac=1.0, random_state=1).set_axis(s.scores["holdout"].index)}, s.record)
                     for b, s in run.scores.items()}
        again = pg.compose_leaves(_built(run.programme), scrambled, labels)
        self.assertEqual(again.composition.candidates, p.composition.candidates, "holdout scores cannot change what is chosen")
        swapped = {b: leaves.LeafScores(b, {**s.scores, "validation": s.scores["holdout"].iloc[: len(s.scores["validation"])].set_axis(s.scores["validation"].index)}, s.record)
                   for b, s in run.scores.items()}
        other = pg.compose_leaves(_built(run.programme), swapped, labels)
        self.assertNotEqual(other.composition.candidates, p.composition.candidates, "the validation scores are what it reads")


def _built(p):
    """The programme as it stood before the composition and the holdout: its leaves built."""
    data = p.model_dump()
    data.update(composition=None, holdout=None, state="leaves_built", fingerprint="")
    return pg.Programme.model_validate(data)


class ValidateTest(unittest.TestCase):
    def test_a_holdout_result_with_a_programme_that_is_not_composed_is_refused(self):
        run, (_, _, m, sm, _) = _run()
        data = run.programme.model_dump()
        data.update(state="leaves_built", fingerprint="")
        issues = pg.validate(pg.Programme.model_validate(data))
        self.assertTrue(any("not composed" in i for i in issues))

    def test_a_holdout_scored_with_another_composition_than_the_one_chosen_is_refused(self):
        run, _ = _run()
        data = run.programme.model_dump()
        data["holdout"]["chosen"] = "rank_mean" if data["holdout"]["chosen"] != "rank_mean" else "rank_max"
        data["fingerprint"] = ""
        self.assertTrue(any("another composition" in i for i in pg.validate(pg.Programme.model_validate(data))))

    def _tampered(self, edit, expect):
        run, (_, _, m, sm, _) = _run()
        data = run.programme.model_dump()
        edit(data)
        data["fingerprint"] = ""
        issues = pg.validate(pg.Programme.model_validate(data), sm)
        self.assertTrue(any(expect in i for i in issues), (expect, issues))

    def test_every_invariant_of_a_read_is_checked(self):
        cases = [(lambda d: d["holdout"].update(rows=d["holdout"]["rows"] + 1), "other rows"),
                 (lambda d: d["holdout"].update(metric="average_precision"), "another metric"),
                 (lambda d: d["holdout"].update(budget_rows=d["holdout"]["budget_rows"] + 5), "review budget"),
                 (lambda d: d["fused"].append("blk9"), "blk9"),
                 (lambda d: d["gates"].append(["blk1", "blk9"]), "blk9"),
                 (lambda d: d["holdout"]["leaves"].pop("blk2"), "does not score the leaves"),
                 (lambda d: d["composition"]["weights"].pop("blk2"), "built or dropped after"),
                 (lambda d: d["leaves"][1].update(status="blocked", why="x"), "while blk2 was blocked"),
                 (lambda d: d["composition"].update(positives=2), "too few confirmed cases"),
                 (lambda d: d["composition"].update(metric="average_precision"), "another metric")]
        for edit, expect in cases:
            with self.subTest(expect):
                self._tampered(edit, expect)

    def test_a_programme_answering_another_problem_map_is_refused(self):
        run, (_, _, m, sm, _) = _run()
        data = run.programme.model_dump()
        data["map"] = "0" * 16
        data["fingerprint"] = ""
        self.assertTrue(any("another problem map" in i for i in pg.validate(pg.Programme.model_validate(data), sm)))

    def test_two_reads_are_not_a_valid_result(self):
        run, _ = _run()
        data = run.programme.model_dump()
        data["holdout"]["reads"] = 2
        with self.assertRaises(Exception):
            pg.Programme.model_validate({**data, "fingerprint": ""})

    def test_a_programme_run_for_another_solution_map_is_refused(self):
        run, (_, _, m, sm, _) = _run()
        other = sm_mod.SolutionMap.model_validate({**sm.model_dump(), "notes": ["changed"], "fingerprint": ""})
        self.assertTrue(any("changed since" in i for i in pg.validate(run.programme, other)))

    def test_a_programme_is_never_edited_in_place(self):
        run, _ = _run()
        data = run.programme.model_dump()
        data["budget"] = 0.5
        with self.assertRaises(Exception) as cm:
            pg.Programme.model_validate(data)
        self.assertIn("fingerprint", str(cm.exception))

    def test_the_fingerprint_follows_a_leaf_choice_and_the_holdout_numbers(self):
        run, _ = _run()
        a = run.programme
        data = a.model_dump()
        data["leaves"][0]["chosen"] = "something_else"
        data["fingerprint"] = ""
        self.assertNotEqual(pg.Programme.model_validate(data).fingerprint, a.fingerprint)
        data = a.model_dump()
        data["holdout"]["composed"]["value"] = 0.123
        data["fingerprint"] = ""
        self.assertNotEqual(pg.Programme.model_validate(data).fingerprint, a.fingerprint)

    def test_the_document_round_trips_through_json(self):
        run, _ = _run()
        text = run.programme.model_dump_json()
        back = pg.Programme.read(text)
        self.assertEqual(back.fingerprint, run.programme.fingerprint)


class AdapterTest(unittest.TestCase):
    def test_each_leaf_that_runs_is_an_experiment_plan_with_the_one_split(self):
        from dclab_core.adapters import main_product_problem_map as adapter

        run, _ = _run()
        out = adapter.programme_experiments(run.programme)
        self.assertEqual(set(out["experiment_plans"]), {"blk1", "blk2", "blk3"}, "the budget step is not an experiment")
        for plan in out["experiment_plans"].values():
            self.assertEqual(plan["contract"], "ExperimentPlan")
            self.assertIsNone(plan["fields"]["split"], "a time-and-entity split has no PlanSplit in the main product")
            self.assertIn("unsupported", plan["notes"]["split"])
        self.assertIn("composition", out["not_mapped"])
        self.assertIn("holdout_read", out["not_mapped"])

    def test_the_conformance_gate_covers_the_contract(self):
        from dclab_core.adapters import conformance

        self.assertIn("programme", conformance.CONTRACT_ADAPTERS)
        self.assertEqual([p for p in conformance.problems() if "programme" in p], [])


if __name__ == "__main__":
    unittest.main()
