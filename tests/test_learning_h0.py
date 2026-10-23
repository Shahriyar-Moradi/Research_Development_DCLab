"""H0 of the harness that learns: challenge packs, their keys, the grader with codes, the first variant family, the baseline.

A pack's public part and its key live apart and a solve never opens the key; the grader passes a right solve and fails
every planted way of gaming it (a leak used, validation beating the holdout, a random split over repeated entities,
everything kept out, a row key used), each with a code that names the kind of mistake and never a column; the family's
variants are drawn from their seeds, all different, each in one partition, and solvable by the key's statements; every
attempt is a run of the learning agent; a sealed partition is read once.
"""

import builtins
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)


def key(**over):
    from dclab_core.challenge_pack import ChallengeKey

    base = dict(pack="pk-000000000000", family="groups", seed=0, partition="check", leaks=["refund_settled"], group_column="household",
                row_key="order_id", decoys=["prior_refund_count"], known_before=["basket_value", "loyalty_tier", "item_count"], knobs={})
    return ChallengeKey(**{**base, **over})


def right(**over):
    """A right solve of ``key()``: the leak and the row key kept out, the decoy used, grouped by the entity, an honest estimate."""
    from dclab_core.challenge_pack import Observed

    base = dict(kept_out=["order_id", "refund_settled"], group_column="household", split="group",
                leaned_on=["basket_value", "prior_refund_count", "loyalty_tier=gold"], metric="roc_auc", validation=0.71,
                holdout=0.70, holdout_interval=[0.66, 0.74], gap=-0.01, baseline=0.5, higher_is_better=True)
    return Observed(**{**base, **over})


class GraderTests(unittest.TestCase):
    def test_a_right_solve_passes_with_every_check(self):
        from dclab_core.challenge_pack import grade

        g = grade(key(), right())
        self.assertEqual((g.verdict, g.codes, g.decision_score, g.traps_caught, g.traps), ("pass", [], 1.0, 3, 3))
        self.assertEqual({c.trap for c in g.checks}, {"after_outcome", "repeated_entity", "row_key", "decoy", "known_before", "honest_estimate", "beats_baseline"})

    def test_every_gaming_trace_fails_with_its_code(self):
        from dclab_core.challenge_pack import grade

        traces = {  # each a way to look good without solving the problem
            "the leak used, the model leaning on it": (right(kept_out=["order_id"], leaned_on=["refund_settled", "basket_value"], validation=0.97,
                                                              holdout=0.96, holdout_interval=[0.95, 0.97]), "LEAK_LEANED:after_outcome"),
            "the leak among the inputs": (right(kept_out=["order_id"]), "LEAK_KEPT:after_outcome"),
            "validation won, the holdout lost": (right(validation=0.86, holdout=0.70, holdout_interval=[0.66, 0.74]),
                                                "GAP_OPTIMISTIC:validation_over_holdout"),
            "a random split over repeated entities": (right(group_column=None, split="stratified"), "SPLIT_WRONG:repeated_entity"),
            "grouped by the wrong column": (right(group_column="loyalty_tier"), "SPLIT_WRONG:repeated_entity"),
            "everything kept out": (right(kept_out=["order_id", "refund_settled", "prior_refund_count", "basket_value"], holdout=0.5,
                                          holdout_interval=[0.46, 0.54], validation=0.5), "FALSE_ALARM:known_before"),
            "a row key used": (right(kept_out=["refund_settled"]), "IDENTIFIER_USED:row_key"),
        }
        for name, (observed, code) in traces.items():
            g = grade(key(), observed)
            self.assertEqual(g.verdict, "fail", name)
            self.assertIn(code, g.codes, name)
        self.assertIn("NO_SIGNAL:not_above_baseline", grade(key(), traces["everything kept out"][0]).codes)

    def test_a_solve_with_no_model_fails_the_guards_and_blocking_honest_columns_fails_without_a_decoy(self):
        from dclab_core.challenge_pack import Observed, grade

        no_model = Observed(kept_out=["order_id", "refund_settled"], group_column="household", split="group")
        g = grade(key(), no_model)
        self.assertEqual(g.verdict, "fail")
        self.assertTrue({"GAP_OPTIMISTIC:validation_over_holdout", "NO_SIGNAL:not_above_baseline"} <= set(g.codes))
        no_baseline = grade(key(), right(baseline=None, metric="log_loss"))
        self.assertIn("NO_SIGNAL:not_above_baseline", no_baseline.codes)
        over = grade(key(decoys=[]), right(kept_out=["order_id", "refund_settled", "basket_value", "loyalty_tier"], holdout=0.55,
                                           holdout_interval=[0.5, 0.6], validation=0.56))
        self.assertEqual(over.verdict, "fail")
        self.assertIn("FALSE_ALARM:known_before", over.codes)

    def test_names_in_finds_a_column_as_a_whole_word_with_underscores_or_spaces(self):
        from dclab_core.challenge_pack import names_in

        k = key(group_column="account", known_before=["account_age_days"])
        self.assertEqual(names_in("Group the folds by account.", k, []), ["account"])
        self.assertEqual(names_in("Keep refund settled out of the model", k, []), ["refund_settled"])
        self.assertEqual(names_in("Take this into account_age_days", k, []), ["account_age_days"])
        self.assertEqual(names_in("Group the folds by the entity whose rows repeat.", k, ["basket_value"]), [])

    def test_an_honest_gap_inside_the_holdouts_interval_passes_and_a_loss_metric_reads_the_other_side(self):
        from dclab_core.challenge_pack import grade

        self.assertEqual(grade(key(), right(validation=0.73, holdout=0.70, holdout_interval=[0.66, 0.74])).verdict, "pass")
        mae = dict(metric="mae", higher_is_better=False, baseline=15.0)
        self.assertEqual(grade(key(), right(validation=11.6, holdout=12.2, holdout_interval=[11.0, 13.4], **mae)).verdict, "pass")
        self.assertIn("GAP_OPTIMISTIC:validation_over_holdout",
                      grade(key(), right(validation=9.0, holdout=12.2, holdout_interval=[11.0, 13.4], **mae)).codes)
        self.assertIn("NO_SIGNAL:not_above_baseline", grade(key(), right(validation=15.5, holdout=15.2, holdout_interval=[14.0, 16.0], **mae)).codes)
        # the calibration's worst honest solve (a regression on 57 groups: 2.7 noise units) passes; past 3.5 is optimistic
        honest = right(validation=10.705, validation_std=0.922, folds=3, holdout=13.142, holdout_interval=[11.621, 14.469], **mae)
        self.assertEqual(grade(key(), honest).verdict, "pass")
        self.assertIn("GAP_OPTIMISTIC:validation_over_holdout", grade(key(), honest.model_copy(update={"validation": 9.6})).codes)

    def test_no_code_names_a_column_of_any_skin(self):
        import re

        from dclab_core.challenge_pack import CODES
        from dclab_core.variants import SKINS

        columns = ("entity_id", "entity_plain", "row_key", "binary", "regression", "leak_worded", "leak_plain", "decoy", "attribute",
                   "amount", "channel", "created")
        names = {getattr(s, f) for s in SKINS for f in columns} | {"account_age_days", "item_count"}
        for code, meaning in CODES.items():
            for name in names:  # a whole word: a column's name inside a code or its meaning
                self.assertIsNone(re.search(rf"(?<![a-z_]){re.escape(name.lower())}(?![a-z_])", f"{code} {meaning}".lower()), (code, name))

    def test_the_contracts_read_and_refuse_another_version(self):
        from dclab_core.challenge_pack import ChallengeKey, Grade, grade
        from dclab_core.contracts import CONTRACTS

        self.assertTrue({"challenge_pack", "challenge_key", "grade"} <= set(CONTRACTS))
        g = grade(key(), right())
        self.assertEqual(Grade.read(g.model_dump_json()), g)
        with self.assertRaises(ValueError):
            ChallengeKey.read({**key().model_dump(), "schema_version": "dclab_core.challenge_key.v0"})


class VariantTests(unittest.TestCase):
    def test_a_seed_gives_the_same_table_and_a_partition_of_its_own(self):
        from dclab_core.challenge_pack import partition_of
        from dclab_core.variants import generate, seeds

        import hashlib

        a, b = generate("groups", 7, pack="pk-a"), generate("groups", 7, pack="pk-a")
        self.assertEqual((a.pack, a.csv), (b.pack, b.csv))
        self.assertEqual(a.pack.table_sha256, hashlib.sha256(a.csv).hexdigest())  # of the file's exact bytes
        self.assertNotEqual(generate("groups", 7).pack.id, generate("groups", 7).pack.id)  # an opaque token each time
        self.assertFalse({"seed", "partition"} & set(a.pack.model_dump()))  # nothing public carries the seed
        self.assertEqual((a.key.seed, a.key.partition), (7, partition_of("groups", 7)))
        self.assertFalse(any("00007" in str(v) for v in a.table[a.key.row_key].head(50)))
        parts = {p: set(seeds("groups", p, 40)) for p in ("learning", "check", "sealed")}
        self.assertFalse(parts["learning"] & parts["check"] or parts["learning"] & parts["sealed"] or parts["check"] & parts["sealed"])
        self.assertTrue(all(partition_of("groups", s) == p for p, ss in parts.items() for s in ss))
        shares = [partition_of("groups", s) for s in range(2000)]
        self.assertTrue(0.45 < shares.count("learning") / 2000 < 0.55 and 0.15 < shares.count("sealed") / 2000 < 0.25)

    def test_twenty_variants_are_all_different_and_their_keys_name_real_columns(self):
        from dclab_core.variants import generate

        made = [generate("groups", s) for s in range(20)]
        self.assertEqual(len({v.pack.table_sha256 for v in made}), 20)
        knobs = {json.dumps({k: v.key.knobs[k] for k in ("skin", "task", "key_name", "leak_name", "decoy", "timed")}, sort_keys=True) for v in made}
        self.assertGreaterEqual(len(knobs), 8)  # the knobs vary, not only the rows
        for v in made:
            planted = [*v.key.leaks, *v.key.decoys, v.key.group_column, v.key.row_key, *([v.key.time_column] if v.key.time_column else [])]
            self.assertTrue(set(planted) <= set(v.table.columns), v.pack.id)
            self.assertFalse(set(v.key.known_before) & {*planted, v.pack.brief.target}, v.pack.id)
            self.assertTrue(v.key.known_before, v.pack.id)
            self.assertNotIn(v.pack.brief.target, planted)
            for name in planted:  # the brief never gives a trap away
                self.assertNotIn(name, f"{v.pack.brief.request} {v.pack.brief.moment}")


@unittest.skipIf(MISSING, MISSING)
class SolveTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        env = mock.patch.dict(os.environ, {"DCLAB_DECISION_SHADOW": "0"})
        env.start()
        self.addCleanup(env.stop)

    def test_a_solve_never_opens_a_key_and_the_reference_solves_every_variant_it_is_given(self):
        from dclab_core.challenge_pack import grade
        from dclab_core.variants import reference
        from dclab_rnd.learning import grading, packs

        keys = (self.root / "keys").resolve()
        opened: list[str] = []
        real_open, real_io_open = builtins.open, io.open

        def watch(file, *args, **kwargs):
            if isinstance(file, (str, os.PathLike)) and str(Path(file).resolve()).startswith(str(keys)):
                opened.append(str(file))
            return real_open(file, *args, **kwargs)

        for seed in range(20):  # every one of 20 variants solvable by its key's statements
            v = packs.materialize(self.root, "groups", seed)
            folder = packs.public_dir(self.root, "groups", v.pack.id)
            self.assertFalse(any(folder.rglob("*key*")))
            k = packs.read_key(self.root, "groups", v.pack.id)
            with mock.patch("builtins.open", watch), mock.patch("io.open", watch):
                run = grading.solve(folder, reference(k))
            self.assertEqual(opened, [], "a solve opened a key")
            self.assertEqual(grade(k, grading.observe(run)).verdict, "pass", v.pack.id)
        self.assertIs(builtins.open, real_open)
        self.assertIs(io.open, real_io_open)

    def test_an_attempt_is_a_run_of_the_learning_agent_with_a_step_per_arm(self):
        from dclab_rnd.learning import scoreboard

        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, True)
        log = experience.FileLog(home / "experience", "wtest")
        experience.install(log)
        self.addCleanup(experience.install, None)
        from dclab_rnd import ledger
        from dclab_rnd.learning import grading

        seen = []
        real_solve = grading.solve

        def watched(folder, statements=None):  # where each arm runs: the reference with no ledger and nothing recorded
            seen.append((statements is not None, ledger._INSTALLED.get("ledger") is None, experience._QUIET.get()))
            return real_solve(folder, statements)

        book = ledger.FileLedger(home / "ledger", "wtest")
        ledger.install(book)
        self.addCleanup(ledger.install, None)
        with mock.patch.object(scoreboard, "solve", watched):
            row = scoreboard.attempt(self.root, "groups", 2)
        self.assertEqual(seen, [(False, False, False), (True, True, True)])
        self.assertIs(ledger._INSTALLED.get("ledger"), book)  # put back after the reference arm
        self.assertTrue(row["solvable"])
        self.assertEqual(row["arms"]["rules"]["codes"], ["SPLIT_WRONG:repeated_entity"])  # a plain group key: the rules miss it
        events = log.read()
        steps = [e for e in events if e.kind == "agent.step" and e.payload.get("agent") == "learning"]
        self.assertEqual([s.payload.get("tool") for s in steps], ["solve:rules", "solve:reference"])
        runs = [e for e in events if e.kind == "run.recorded" and e.payload.get("name") == "learning"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].payload["params"]["pack"], row["pack"])
        calls = [e for e in events if e.kind == "tool.call" and e.ids.run == steps[0].ids.run]
        self.assertEqual([json.loads(json.dumps(c.payload.get("result"))) for c in calls][0]["codes"] if isinstance(calls[0].payload.get("result"), dict)
                         else json.loads(calls[0].payload["result"])["codes"], ["SPLIT_WRONG:repeated_entity"])  # the codes, whole
        k = scoreboard.read_key(self.root, "groups", row["pack"])
        text = json.dumps([c.payload for c in calls])
        for name in [*k.leaks, *k.decoys, k.group_column, k.row_key]:  # a learner reading the record finds no column's name
            self.assertNotIn(name, text)


@unittest.skipIf(MISSING, MISSING)
class BaselineTests(unittest.TestCase):
    def test_the_baseline_summarises_and_a_sealed_partition_is_read_once(self):
        from dclab_rnd.learning import scoreboard

        from dclab_core import variants

        results = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, results, True)
        learning = variants.seeds("groups", "learning", 2)  # the rule under test, on learning seeds: a test never reads a sealed variant
        with mock.patch.object(scoreboard, "RESULTS", results), mock.patch.dict(os.environ, {"DCLAB_DECISION_SHADOW": "0"}), \
                mock.patch.object(scoreboard.V, "seeds", lambda family, split, n: learning[:n]):
            with self.assertRaises(scoreboard.SealedSpent):  # never read without being stored
                scoreboard.baseline("groups", "sealed", 2, write=False, log=lambda *_: None)
            out = scoreboard.baseline("groups", "sealed", 2, write=True, log=lambda *_: None)
            self.assertTrue((results / ".sealed_groups").exists())
            self.assertTrue(out["verdict"].startswith("ok"), out["verdict"])
            saved = json.loads((results / f"{out['id']}_groups_sealed.json").read_text(encoding="utf-8"))
            self.assertEqual((saved["experiment_id"], saved["status"], saved["gate"]["version"]), (out["id"], "completed", "learning_gate_v1"))
            self.assertEqual(saved["summary"]["solvable"], 2)
            self.assertEqual(set(saved["fingerprints"]), {"generator", "grader"})
            with self.assertRaises(scoreboard.SealedSpent):
                scoreboard.baseline("groups", "sealed", 2, write=True, log=lambda *_: None)
            (results / f"{out['id']}_groups_sealed.json").unlink()  # a crash leaves no result: the marker still refuses
            with self.assertRaises(scoreboard.SealedSpent):
                scoreboard.baseline("groups", "sealed", 2, write=True, log=lambda *_: None)

    def test_an_attempt_that_breaks_is_a_row_with_its_reason(self):
        from dclab_rnd.learning import scoreboard

        with mock.patch.object(scoreboard, "attempt", side_effect=RuntimeError("the engine broke on purpose")):
            out = scoreboard.baseline("groups", "learning", 2, write=False, log=lambda *_: None)
        self.assertTrue(out["verdict"].startswith("fail"))
        self.assertEqual(len(out["summary"]["broke"]), 2)
        self.assertIn("the engine broke on purpose", out["rows"][0]["error"])

    def test_the_summary_counts_codes_and_knobs_and_drops_an_unsolvable_variant(self):
        from dclab_rnd.learning.scoreboard import bootstrap_mean, summarize

        def row(pack, verdict, codes, solvable=True, key_name="id"):
            arm = {"verdict": verdict, "codes": codes, "decision_score": 1.0 if verdict == "pass" else 0.5, "traps_caught": 3 if verdict == "pass" else 2, "traps": 3}
            return {"pack": pack, "solvable": solvable, "knobs": {"task": "binary", "key_name": key_name, "leak_name": "plain", "decoy": False, "timed": False},
                    "arms": {"rules": arm, "reference": {"verdict": "pass" if solvable else "fail"}}}

        s = summarize([row("a", "pass", []), row("b", "fail", ["SPLIT_WRONG:repeated_entity"], key_name="plain"),
                       row("c", "fail", ["FALSE_ALARM:known_before"], solvable=False)])
        self.assertEqual((s["variants"], s["solvable"], s["unsolvable"], s["passed"], s["pass_rate"]), (3, 2, ["c"], 1, 0.5))
        self.assertEqual(s["codes"], {"SPLIT_WRONG:repeated_entity": 1})
        self.assertEqual(s["by_knob"]["key_name"], {"id": {"passed": 1, "of": 1}, "plain": {"passed": 0, "of": 1}})
        low, high = bootstrap_mean([1.0, 0.5, 0.5, 1.0])
        self.assertTrue(0.5 <= low <= 0.75 <= high <= 1.0)


if __name__ == "__main__":
    unittest.main()
