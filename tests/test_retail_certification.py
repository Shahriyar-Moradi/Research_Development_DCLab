"""Package 30.12: labels of unequal worth and a team's capacity, on the retail returns replica through the engine. The
rule-label trap is caught (noisy labels applied, the run judged on confirmed labels only, the late-evening rule behind
the rule's labels flagged, the verdict and the whole-log statistic kept out), and the queue's recall, estimated only from
the random audit, has an interval that covers the true recall on every seed; the proposal finds the label source alone."""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401

    HAVE_ML = True
except ImportError:  # the agent environment
    HAVE_ML = False


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class RetailTests(unittest.TestCase):
    def test_the_trap_is_caught_and_the_audits_interval_covers_the_true_recall_on_every_seed(self):
        from dclab_core import capacity as C
        from dclab_rnd.kinds import rare_event
        from dclab_rnd.kinds.acceptance import engine_run
        from dclab_rnd.studio import engine

        for seed in (0, 1, 2):
            case, replica = rare_event.KIND.retail_case(seed)
            seen = {}
            real = engine._capacity_inputs

            def grab(*a, **k):
                seen["inputs"] = real(*a, **k)
                return seen["inputs"]

            with mock.patch.object(engine, "_capacity_inputs", side_effect=grab):
                run = engine_run(case, "retail_returns")
            failed = [c for c in rare_event.KIND.accept_retail(run) if not c["ok"]]
            self.assertEqual(failed, [], seed)
            rows, scores, periods, _, _, _, _ = seen["inputs"]
            queued = C.queue(scores, periods, case.solution["capacity_k"])
            fraud = rows["return_id"].map(replica.truth.set_index("return_id")["fraud"]).to_numpy()
            true = float((queued & (fraud == 1)).sum() / (fraud == 1).sum())
            low, high = run.evidence("final")["capacity"]["audit_recall"]["recall_ci"]
            self.assertTrue(low <= true <= high, (seed, true, low, high))  # estimated from the audit alone, the truth inside

    def test_the_proposal_finds_the_label_source_and_never_forbids_it(self):
        from dclab_core.replicas import retail_returns as rr
        from dclab_rnd.studio.data import profile_table
        from dclab_rnd.studio.solution import propose

        table = rr.generate(seed=0).returns
        table = table.drop(columns=[c for c in table.columns if c.startswith("_")], errors="ignore")
        prop = propose(table, profile_table(table), "label", time_column="returned_at")
        found = {d["name"]: d for d in prop["modifiers"]}
        self.assertEqual(prop.get("label_source_column"), "label_source")
        self.assertTrue(found["noisy_labels"]["applies"] and found["rare_positives"]["applies"])  # rare on the audit's rate, not the rule's
        self.assertIn("random for review", found["rare_positives"]["reason"])
        self.assertFalse(found["review_capacity"]["applies"])
        self.assertTrue(found["review_capacity"]["asks_user"])  # K is always asked, never guessed
        self.assertNotIn("label_source", {f["column"] for f in prop["forbidden"]})


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class AdoptTests(unittest.TestCase):
    """Built as Home and the journeys build (the proposal and its plan adopted, nothing declared): the label source is
    never a feature, whether its noisy labels apply or only ask."""

    def _roles(self, table, statements):
        from dclab_rnd.kinds.acceptance import Case, engine_run

        run = engine_run(Case(table, "label", "binary", "When a return is made, before the team's daily review.", statements, quick=True))
        return run, run.records["data"].get("evidence", {}).get("column_roles") or {}

    def test_adopted_with_nothing_declared_the_source_is_the_labels_source(self):
        from dclab_core.replicas import retail_returns as rr

        table = rr.generate(seed=1).returns
        table = table.drop(columns=[c for c in table.columns if c.startswith("_")], errors="ignore")
        run, roles = self._roles(table, {"time_column": "returned_at"})  # no forbidden list: the plan's own is adopted
        self.assertEqual(run.solution.get("label_source_column"), "label_source")
        self.assertEqual(roles.get("label_source"), "identifier")
        self.assertIn("noisy_labels@1", run.records["data"]["evidence"]["recipe"]["modifiers"])

    def test_with_too_few_confirmed_labels_the_source_stays_out_of_the_model(self):
        import numpy as np

        from dclab_core.replicas import retail_returns as rr

        table = rr.generate(seed=1).returns
        table = table.drop(columns=[c for c in table.columns if c.startswith("_")], errors="ignore")
        confirmed = table.index[table["label_source"].isin(["expert", "expert_audit"])]
        drop = confirmed[10:]  # ten confirmed labels left: the noisy labels can only ask
        table.loc[drop, "label"], table.loc[drop, "label_source"], table.loc[drop, "review_status"] = np.nan, "", ""
        import tempfile

        from dclab_rnd.kinds.acceptance import Case, build

        store, project_id = build(Case(table, "label", "binary", "When a return is made, before the team's daily review.",
                                       {"time_column": "returned_at"}, quick=True), Path(tempfile.mkdtemp()))
        solution = store.get(project_id)["solution"]  # the saved solution, as Home and the journeys save it
        self.assertIsNone(solution.get("label_source_column"))
        kept_out = {f["column"] for f in solution["forbidden"]} | set(solution.get("identifiers") or [])
        self.assertIn("label_source", kept_out)  # never learned from: it says how each label was written


class CertifyTests(unittest.TestCase):
    def test_a_part_is_certified_only_by_runs_where_it_applies_and_every_criterion_measured(self):
        from dclab_core.recipes import CRITERIA, certify

        ok = {c: True for c in CRITERIA}
        runs = [{"name": "replica", "source": "replica", "applies": True, "criteria": ok},
                {"name": "a", "source": "real", "applies": True, "criteria": ok},
                {"name": "b", "source": "real", "applies": False, "criteria": ok}]  # its right absence: checked, never counted
        self.assertEqual(certify(runs)["level"], "experimental")
        self.assertEqual(certify(runs + [{"name": "c", "source": "real", "applies": True, "criteria": ok}])["level"], "certified")
        pending = [{**r, "criteria": {**ok, "artifact": None}} for r in runs]
        got = certify(pending + [{"name": "c", "source": "real", "applies": True, "criteria": {**ok, "artifact": None}}])
        self.assertEqual((got["level"], got["pending"]), ("experimental", ["artifact"]))

    def test_detection_is_judged_on_what_the_proposal_read_never_on_what_the_case_stated(self):
        from dclab_rnd import certify as K

        class Run:  # a finished run, as its records hold it
            records = {"data": {"evidence": {"recipe": {"modifiers": ["noisy_labels@1", "review_capacity@1"],
                                                        "detected": [{"name": "noisy_labels", "applies": False, "asks_user": True},
                                                                     {"name": "review_capacity", "applies": False, "asks_user": True}]},
                                             "column_roles": {"customerID": "identifier"}}}}

            def evidence(self, stage):
                return {"holdout_metrics": {"average_precision": 0.6}, "baselines": {"majority_class": {"average_precision_random_ranking": 0.2}}}

        entry = {"name": "x", "source": "real", "expect": {"noisy_labels": True, "review_capacity": "asks"}, "traps": {"identifiers": ["customerID"]}}
        got = K.judge(entry, Run())
        self.assertIs(got["criteria"]["noisy_labels"]["detection"], False)  # applied only because stated: not a detection
        self.assertIs(got["criteria"]["review_capacity"]["detection"], True)  # asked, as only a person can state K
        self.assertIsNone(got["criteria"]["noisy_labels"]["llm_off"])  # never compared with a model on: pending, not true


class SolutionTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
    def test_a_capacity_needs_its_period_and_time_and_the_label_source_is_never_a_feature(self):
        from dclab_rnd.studio.solution import Solution

        base = {"target": "label", "task": "binary", "prediction_moment": "When a return is made, before the review"}
        Solution(**base, time_column="returned_at", capacity_k=5, capacity_period="day", label_source_column="label_source")
        for bad in ({"capacity_k": 5}, {"capacity_k": 5, "time_column": "returned_at"}, {"label_source_column": "label"},
                    {"label_source_column": "src", "forbidden": [{"column": "src"}]}):
            with self.assertRaises(ValueError, msg=str(bad)):
                Solution(**{**base, **bad})


if __name__ == "__main__":
    unittest.main()
