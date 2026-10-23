"""Self-play on every kind (27.3 widened): the kinds whose case states no outcome column (their outcome lives in the kind's
fields, or there is none) and the kind the engine does not run (metric_change, a library) are solved, verified by their own
acceptance, and recorded; the image variant's fresh-draw label looks its own kind up."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ENGINE_KINDS = ("entity_tables", "matching", "ranking", "segmentation", "streams", "anomaly")


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []


class SelfplayKindsTests(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.lake import selfplay as P

        self.P = P
        self.tmp = Path(tempfile.mkdtemp())
        patches = [mock.patch.object(P, "OUT", self.tmp), mock.patch.dict("os.environ", {"DCLAB_DECISION_SHADOW": "0"})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def solve_one(self, key: str) -> dict:
        done = self.P.solve([key], seeds=1, log=lambda *_: None)
        self.assertEqual((done["problems"], done["rejected"]), (1, 0), lines(self.tmp / "done.jsonl"))
        return done

    def check_engine_kind(self, key: str) -> None:
        self.solve_one(key)
        group = f"selfplay:{key}:0"
        self.assertEqual(lines(self.tmp / "done.jsonl"), [{"group": group, "verified": True, "examples": 0,
                                                            "seconds": lines(self.tmp / "done.jsonl")[0]["seconds"]}])
        docs = self.P.decision_docs(self.tmp)  # the ledger is complete: the store and the decision sets read it
        self.assertTrue(docs and {g for g, _ in docs} == {group})
        # no table-shaped example: its families assume one outcome column and the person's forbidden list
        self.assertEqual(self.P.examples(), [])
        self.assertFalse((self.tmp / "screens.jsonl").exists())

    def test_entity_tables(self):
        self.check_engine_kind("entity_tables")

    def test_matching(self):
        self.check_engine_kind("matching")

    def test_ranking(self):
        self.check_engine_kind("ranking")

    def test_segmentation(self):
        self.check_engine_kind("segmentation")

    def test_streams(self):
        self.check_engine_kind("streams")

    def test_anomaly(self):
        self.check_engine_kind("anomaly")

    def test_metric_change_is_solved_by_its_library_and_kept_whole(self):
        self.solve_one("metric_change")
        solves = lines(self.tmp / "library_solves.jsonl")
        self.assertEqual([(s["group"], s["verified"]) for s in solves], [("selfplay:metric_change:0", True)])
        answers = solves[0]["solve"]["answers"]
        self.assertTrue(answers and all({"truth", "answer", "said"} <= set(a) for a in answers))
        self.assertTrue(all(isinstance(a["answer"], dict) and "status" in a["answer"] for a in answers))  # the answer whole, not its text
        self.assertTrue(all(c["ok"] for c in solves[0]["checks"]))
        self.assertFalse((self.tmp / "ledgers" / "metric_change" / "0").exists())  # its path records no decision: no empty workspace

    def test_a_failed_solve_is_logged_and_its_ledger_never_read(self):
        from dclab_rnd import kinds

        kind = kinds.get("ranking")
        recorded = self.P._run_recorded

        def planted(*args, **kwargs):  # a stage record quoting rows under a cell key, beside a key that stays
            run = recorded(*args, **kwargs)
            run.records.setdefault("data", {})["examples"] = ["58", "44", "33"]
            run.records["data"]["usable_feature_count"] = 7
            return run

        with mock.patch.object(type(kind), "accept_engine", lambda self, run: [{"decision": "planted", "ok": False, "detail": ""}]), \
                mock.patch.object(self.P, "_run_recorded", planted):
            done = self.P.solve(["ranking"], seeds=1, log=lambda *_: None)
        self.assertEqual((done["problems"], done["rejected"]), (0, 1))
        self.assertEqual(lines(self.tmp / "done.jsonl")[0], {"group": "selfplay:ranking:0", "verified": False, "missed": ["planted"]})
        self.assertEqual(self.P.decision_docs(self.tmp), [])
        # D2 / P3: the failed solve is kept whole beside the verified ones: the keys that quote rows removed, never the table itself
        kept = lines(self.tmp / "failed_solves.jsonl")
        self.assertEqual([(d["group"], d["verified"], d["missed"], d["reason"]) for d in kept], [("selfplay:ranking:0", False, ["planted"], None)])
        self.assertEqual(kept[0]["checks"], [{"decision": "planted", "ok": False, "detail": ""}])
        self.assertEqual(set(kept[0]["solve"]), {"solution", "records", "labels"})
        self.assertNotIn("examples", kept[0]["solve"]["records"]["data"])
        self.assertEqual(kept[0]["solve"]["records"]["data"]["usable_feature_count"], 7)
        self.assertNotIn('"table"', json.dumps(kept[0]))

    def test_a_solve_whose_checks_break_is_kept_with_its_whole_error(self):
        from dclab_rnd import kinds

        kind = kinds.get("ranking")

        def broken(self, run):
            raise RuntimeError("the acceptance read a column that is not there: " + "x" * 300)

        with mock.patch.object(type(kind), "accept_engine", broken):
            done = self.P.solve(["ranking"], seeds=1, log=lambda *_: None)
        self.assertEqual(done["rejected"], 1)
        kept = lines(self.tmp / "failed_solves.jsonl")
        self.assertEqual(len(kept[0]["reason"]), len("RuntimeError: the acceptance read a column that is not there: ") + 300)  # whole, never cut
        self.assertIn("RuntimeError", kept[0]["traceback"])
        self.assertLess(len(lines(self.tmp / "done.jsonl")[0]["reason"]), 220)  # the log line keeps its short form

    def test_forget_takes_out_a_failed_solve_too(self):
        from dclab_rnd import kinds

        kind = kinds.get("ranking")
        with mock.patch.object(type(kind), "accept_engine", lambda self, run: [{"decision": "planted", "ok": False, "detail": ""}]):
            self.P.solve(["ranking"], seeds=1, log=lambda *_: None)
        self.assertEqual(len(lines(self.tmp / "failed_solves.jsonl")), 1)
        self.P.forget(["ranking"])
        self.assertEqual(lines(self.tmp / "failed_solves.jsonl"), [])

    def test_a_library_solve_that_breaks_is_logged(self):
        from dclab_rnd import kinds

        kind = kinds.get("metric_change")
        with mock.patch.object(type(kind), "solve", side_effect=RuntimeError("no data")):
            done = self.P.solve(["metric_change"], seeds=1, log=lambda *_: None)
        self.assertEqual(done["rejected"], 1)
        self.assertEqual(lines(self.tmp / "done.jsonl")[0], {"group": "selfplay:metric_change:0", "verified": False, "reason": "RuntimeError: no data"})

    def test_backfill_writes_no_example_for_a_kind_with_no_outcome_column_and_skips_a_library_kind(self):
        (self.tmp / "done.jsonl").write_text("".join(json.dumps({"group": g, "verified": True}) + "\n"
                                                     for g in ("selfplay:ranking:0", "selfplay:metric_change:0")), encoding="utf-8")
        got = self.P.backfill(log=lambda *_: None)
        self.assertEqual(got, {"recorded_again": 1})
        self.assertEqual([m["group"] for m in lines(self.tmp / "backfill_done.jsonl")], ["selfplay:ranking:0"])
        self.assertTrue(lines(self.tmp / "backfill_done.jsonl")[0]["verified"])
        self.assertEqual(self.P.examples(), [])

    def test_forget_takes_out_a_library_solve(self):
        (self.tmp / "library_solves.jsonl").write_text(json.dumps({"group": "selfplay:metric_change:0"}) + "\n"
                                                       + json.dumps({"group": "selfplay:binary:0"}) + "\n", encoding="utf-8")
        self.P.forget(["metric_change"])
        self.assertEqual([x["group"] for x in lines(self.tmp / "library_solves.jsonl")], ["selfplay:binary:0"])

    def test_a_trajectory_of_a_kind_with_no_outcome_never_says_one_was_named(self):
        from dclab_rnd.store.sets import llm

        for kind in ("segmentation", "anomaly"):
            p = llm.selfplay_problem(kind, "0", f"selfplay:{kind}:0")
            self.assertFalse(p["outcome_named"])
            self.assertIn("no outcome to learn", p["request"])
        p = llm.selfplay_problem("ranking", "0", "selfplay:ranking:0")  # its outcome is the relevant event it names
        self.assertTrue(p["outcome_named"])
        self.assertEqual(p["request"], "Solve the invented ranking problem (seed 0): its outcome and its prediction moment are stated.")

    def check_machine_leaks_taught(self, key: str, leaks: list[str], identifier: str | None) -> None:
        """A kind with its own machine keeps its planted leak out in its leakage stage, not in the Solution: the examples
        must say so (they said ``forbidden: []`` on every forecasting solve)."""
        self.solve_one(key)
        got = self.P.examples()
        plain = [e for e in got if "col_01" not in e.messages[1]["content"]]
        leakage = json.loads(next(e for e in plain if e.family == "leakage_judgment").messages[2]["content"])
        forbidden = {f["column"]: f["reason"] for f in leakage["forbidden"]}
        self.assertLessEqual(set(leaks), set(forbidden), leakage)
        self.assertTrue(all(forbidden[c] for c in leaks))  # each with the machine's reason
        if identifier:
            self.assertNotIn(identifier, forbidden)  # an identifier is not a leak
            self.assertIn(identifier, leakage["identifiers"])
        workflow = json.loads(next(e for e in plain if e.family == "solution_design").messages[2]["content"])["workflow"]
        step = next(s["decision"] for s in workflow if s["step"] == "leakage")
        self.assertLessEqual(set(leaks), set(step["forbidden"]))
        masked = [e for e in got if e.family == "leakage_judgment" and "col_01" in e.messages[1]["content"]]
        self.assertTrue(masked)
        for e in masked:  # the masked twin names no column, not even in the machine's reasons
            self.assertFalse([c for c in leaks if c in e.messages[2]["content"]])

    def test_forecasting_examples_keep_its_planted_leak_out(self):
        self.check_machine_leaks_taught("forecasting", ["rain_mm"], None)

    def test_time_to_event_examples_keep_both_planted_leaks_out_and_its_identifier_apart(self):
        self.check_machine_leaks_taught("time_to_event", ["months_billed", "tickets_so_far"], "subscriber_id")

    def test_the_benchmark_reads_an_image_solves_truth(self):
        from dclab_rnd.lake import benchmark

        row = {"provenance": {"group": "selfplay:embedded_images:0"}, "messages": [{}, {"content": "col_01"}]}
        got = benchmark.truth(row)  # it raised StopIteration: the registry has no embedded_images
        self.assertEqual(got["task"], "binary")

    def test_the_image_variant_draws_its_fresh_case_from_its_own_kind(self):
        """The label step looked the variant up by its own key (``embedded_images``), which the registry does not know."""
        from dclab_rnd import ledger
        from dclab_rnd.lake import replicate

        seen = []

        def fresh(kind, seed, run):
            seen.append((kind.key, kind.case(seed + replicate.OFFSET).target))
            return None

        book = ledger.FileLedger(self.tmp / "book", "selfplay")
        out = io.StringIO()
        with mock.patch.object(replicate, "tuning_on_fresh_draw", fresh), redirect_stdout(out):
            self.P._label_tuning(book, "embedded_images", 0, None)
        self.assertEqual(seen, [("embedded_images", "label")])  # the image case, drawn again
        self.assertNotIn("no fresh-draw label", out.getvalue())


if __name__ == "__main__":
    unittest.main()
