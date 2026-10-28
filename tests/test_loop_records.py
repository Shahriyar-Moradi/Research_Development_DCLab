"""31.17: the learning loop's own records, joined from what each seat wrote and replayed from the records alone."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from dclab_core import loop_record as L

PACK, RUN, BULLET = "pk-aaaaaaaaaaaa", "learning-111111111111", "B-0123456789"


def write(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc) + "\n", encoding="utf-8")


def grade(verdict="fail", codes=("SPLIT_WRONG:repeated_entity",), score=0.5):
    return {"grader": "grader_v1", "verdict": verdict, "codes": list(codes), "decision_score": score}


class Workspace:
    """A small loop on disk: a draft attempt kept in its folder, the lesson drawn from it, a probe that read it."""

    def __init__(self, root: Path):
        self.home, self.results = root / "home", root / "results"
        draft = self.home / "learning_attempts" / "2026-10-29" / f"010000-{PACK}-draft"
        write(draft / "attempt.json", {"partition": "learning", "run": RUN, "key_sha256": "ab" * 32})
        write(draft / "pack.json", {"family": "groups", "id": PACK})
        write(draft / "grade.json", {"rules": {"grade": grade()}, "reference": {"grade": grade("pass", (), 1.0)}})
        write(self.home / "learning" / "drafts.jsonl", {"family": "groups", "pack": PACK, "run": RUN, "codes": ["SPLIT_WRONG:repeated_entity"],
                                                        "step": "solution", "filed": [BULLET], "refused": [], "why": "", "at": "2026-10-29T01:00:05+00:00"})
        self.log = [{"op": "add", "bullet": {"id": BULLET, "evidence": [{"pack": PACK, "run": RUN}]}, "by": "rule:reflect_v1", "at": "2026-10-29T01:00:05+00:00"},
                    {"op": "count", "target": BULLET, "counts": {"shown": 1, "helpful": 1}, "by": "probe", "at": "2026-10-29T02:00:00+00:00"}]
        path = self.home / "playbook" / "groups" / "main.deltas.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(d) + "\n" for d in self.log), encoding="utf-8")
        probe = self.home / "learning_attempts" / "2026-10-29" / f"013000-pk-bbbbbbbbbbbb-probe-0123456789-with-r0"
        write(probe / "probe_row.json", {"model": "qwen3.5:4b-16k", "arm": "with"})
        write(probe / "grade.json", {"grade": grade("pass", (), 1.0)})
        write(probe / "intern" / "s1.json", {"model": "qwen3.5:4b-16k", "playbook": {"lineage": "main", "version": 1, "shown": [BULLET],
                                                                                     "used": [BULLET], "sha256": "1052bc8457356c2b"}})
        write(probe / "probe_counted.json", {"bullet": BULLET, "gain": 1.0, "at": "2026-10-29T02:00:00+00:00"})
        self.results.mkdir(parents=True, exist_ok=True)


class LoopRecordsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.ws = Workspace(self.root)

    def build(self):
        from dclab_rnd.learning.loop_records import build

        return build(self.ws.home, self.ws.results)

    def test_one_turn_of_the_loop_replays_from_its_records_alone(self):
        records, changes = self.build()
        self.assertEqual(L.replay(records, changes), [])
        by_kind = {r.attempt.kind + ":" + (r.attempt.arm or "-"): r for r in records}
        draft = by_kind["draft:rules"]
        self.assertEqual(draft.reflection.filed, [BULLET])  # the lesson drawn from the attempt
        self.assertEqual([(d.op, d.bullet, d.index) for d in draft.deltas], [("add", BULLET, 0)])  # the playbook's version 1
        probe = by_kind["probe:with"]
        self.assertEqual((probe.attempt.reads.version, probe.attempt.reads.shown), (1, [BULLET]))  # the next attempt read it
        self.assertEqual([(d.op, d.index) for d in probe.deltas], [("count", 1)])  # and its tally is the next change
        self.assertEqual(probe.attempt.producer, "model:qwen3.5:4b-16k")
        self.assertFalse(by_kind["draft:reference"].training_allowed)  # the reference arm states the key's answers

    def test_a_broken_chain_is_found(self):
        records, changes = self.build()
        probe = next(r for r in records if r.attempt.kind == "probe")
        early = probe.model_copy(update={"attempt": probe.attempt.model_copy(update={"reads": probe.attempt.reads.model_copy(update={"version": 0})})})
        self.assertTrue(any("not in the playbook at version 0" in p for p in L.replay([early], changes)))
        later = probe.model_copy(update={"attempt": probe.attempt.model_copy(update={"reads": probe.attempt.reads.model_copy(update={"version": 9})})})
        self.assertTrue(any("beyond the 2 changes" in p for p in L.replay([later], changes)))
        without_draft = [r for r in records if r.attempt.kind != "draft"]
        self.assertTrue(any("which no record holds" in p for p in L.replay(without_draft, changes)))  # a lesson with no attempt behind it

    def test_an_exams_attempt_never_opens_to_training(self):
        pack = L.PackRef(id=PACK, family="groups", split="sealed")
        attempt = L.AttemptRef(id="x", kind="baseline", producer="rule:dclab", arm="rules")
        with self.assertRaises(ValueError):
            L.LoopRecord(id="a", family="groups", pack=pack, attempt=attempt, producer="rule:dclab", consent="c")  # not marked an exam
        with self.assertRaises(ValueError):
            L.LoopRecord(id="a", family="groups", pack=pack, attempt=attempt, producer="rule:dclab", consent="c", exam=True)  # open to training
        sealed = L.LoopRecord(id="a", family="groups", pack=pack, attempt=attempt, producer="rule:dclab", consent="c", exam=True, training_allowed=False)
        records, _ = self.build()
        self.assertNotIn(sealed, L.training(records + [sealed]))

    def test_a_scoreboard_row_is_recounted_from_its_records(self):
        rows = [{"pack": "pk-cccccccccccc", "run": "r1", "arms": {"rules": {"verdict": "pass", "codes": [], "decision_score": 1.0},
                                                                    "reference": {"verdict": "pass", "codes": [], "decision_score": 1.0}}},
                {"pack": "pk-dddddddddddd", "run": "r2", "arms": {"rules": {"verdict": "fail", "codes": ["X:y"], "decision_score": 0.5},
                                                                    "reference": {"verdict": "pass", "codes": [], "decision_score": 1.0}}},
                {"pack": "pk-eeeeeeeeeeee", "run": "r3", "arms": {"rules": {"verdict": "fail", "codes": ["X:y"], "decision_score": 0.0},
                                                                    "reference": {"verdict": "fail", "codes": ["X:y"], "decision_score": 0.5}}}]
        write(self.ws.results / "LRN-009_groups_check.json", {"id": "LRN-009", "kind": "learning_baseline", "family": "groups", "split": "check",
                                                              "at": "2026-10-29T03:00:00+00:00", "rows": rows,
                                                              "summary": {"solvable": 2, "passed": 1, "decision_score": 0.75}})
        records, changes = self.build()
        got = L.row(records, "LRN-009")
        self.assertEqual((got["solvable"], got["passed"], got["decision_score"]), (2, 1, 0.75))  # the unsolvable variant left out, as stored
        self.assertEqual(len(got["records"]), 2)

    def solved(self, partition):
        """A real attempt() on a variant of ``partition``, with the engine stubbed: the rules arm keeps nothing out, the
        reference arm keeps out exactly the key's planted columns (as the key's statements make the engine do)."""
        from unittest import mock

        from dclab_core import challenge_pack as cp
        from dclab_core import variants as V

        from dclab_rnd.learning import scoreboard
        from dclab_rnd.learning.packs import read_key

        work = self.root / f"packs-{partition}"
        seed = V.seeds("groups", partition, 1)[0]
        calls = []

        def solve(folder, statements=None):
            calls.append(statements)
            return statements

        def grade_run(key, run):
            if run:  # the reference arm: the key's answers
                observed = cp.Observed(kept_out=[*key.leaks, *([key.row_key] if key.row_key else [])], group_column=key.group_column, split="group")
            else:
                observed = cp.Observed()
            return cp.grade(key, observed), observed

        with mock.patch.object(scoreboard, "solve", solve), mock.patch.object(scoreboard, "grade_run", grade_run):
            row = scoreboard.attempt(work, "groups", seed, keep=self.ws.home)
        return row, read_key(work, "groups", row["pack"])

    def test_a_kept_learning_attempt_holds_the_keys_hash_never_the_keys_answers(self):
        row, key = self.solved("learning")
        folder = self.ws.home / row["kept"]
        planted = [*key.leaks, key.row_key, key.group_column]
        graded = json.loads((folder / "grade.json").read_text())
        self.assertNotIn("observed", graded["reference"])  # the reference arm's observations are the key's answers
        for name in ("grade.json", "attempt.json"):
            text = (folder / name).read_text(encoding="utf-8")
            for column in filter(None, planted):
                self.assertNotIn(f'"{column}"', text, f"{name} names the planted column {column}")
        self.assertIn("key_sha256", json.loads((folder / "attempt.json").read_text()))
        records, changes = self.build()
        mine = [r for r in records if r.pack.id == row["pack"]]
        self.assertEqual({(r.attempt.arm, r.pack.split, r.training_allowed) for r in mine}, {("rules", "learning", True), ("reference", "learning", False)})
        self.assertEqual(L.replay(records, changes), [])

    def test_a_check_or_sealed_attempt_keeps_only_its_row_and_never_opens_to_training(self):
        row, _ = self.solved("check")
        self.assertNotIn("kept", row)
        self.assertFalse(any(p.name.endswith("-baseline") for p in (self.ws.home / "learning_attempts").glob("*/*")))
        rows = [{"pack": row["pack"], "run": "r1", "arms": {"rules": {"verdict": "pass", "codes": [], "decision_score": 1.0},
                                                            "reference": {"verdict": "pass", "codes": [], "decision_score": 1.0}}}]
        for split, rid in (("check", "LRN-010"), ("sealed", "LRN-011")):
            write(self.ws.results / f"{rid}_groups_{split}.json", {"id": rid, "kind": "learning_baseline", "family": "groups", "split": split, "rows": rows})
        records, _ = self.build()
        theirs = [r for r in records if r.result in ("LRN-010", "LRN-011")]
        self.assertEqual(len(theirs), 4)
        self.assertFalse(any(r.training_allowed for r in theirs))
        self.assertTrue(all(r.exam for r in theirs if r.pack.split == "sealed"))
        self.assertEqual([r for r in L.training(records) if r.pack.split != "learning"], [])
        with self.assertRaises(ValueError):
            L.LoopRecord(id="c", family="groups", pack=L.PackRef(id=PACK, family="groups", split="check"),
                         attempt=L.AttemptRef(id="x", kind="baseline", producer="rule:dclab"), producer="rule:dclab", consent="c")

    def test_a_folder_that_does_not_say_its_partition_is_an_exams(self):
        folder = self.ws.home / "learning_attempts" / "2026-10-29" / "020000-pk-ffffffffffff-baseline"
        write(folder / "attempt.json", {"run": "r9"})
        write(folder / "grade.json", {"rules": {"grade": grade()}})
        records, _ = self.build()
        mine = next(r for r in records if r.pack.id == "pk-ffffffffffff")
        self.assertEqual((mine.pack.split, mine.exam, mine.training_allowed), ("sealed", True, False))

    def test_a_probes_tally_joins_the_change_it_names_or_the_gap_is_reported(self):
        probe = next((self.ws.home / "learning_attempts" / "2026-10-29").glob("*-probe-*"))
        write(probe / "probe_counted.json", {"bullet": BULLET, "gain": 1.0, "at": "2026-10-29T02:00:01+00:00", "lineage": "main", "delta": 1})
        records, changes = self.build()  # a second ticked between the two clocks: the index still joins them
        mine = next(r for r in records if r.attempt.kind == "probe")
        self.assertEqual([(d.op, d.index) for d in mine.deltas], [("count", 1)])
        self.assertEqual(L.replay(records, changes), [])
        lost = mine.model_copy(update={"deltas": []})
        self.assertTrue(any("no count delta" in p for p in L.replay([lost], changes)))

    def test_the_records_land_in_the_lake_with_their_manifest(self):
        from dclab_rnd.learning.loop_records import write as land

        records, changes = self.build()
        lake = self.root / "lake"
        first = land(records, changes, lake)[0].read_bytes()
        manifest = json.loads((lake / "bronze" / "loop" / "groups" / "MANIFEST.json").read_text())
        self.assertEqual((manifest["records"], manifest["problems"], manifest["exam"]), (len(records), [], 0))
        self.assertEqual(land(*self.build(), lake)[0].read_bytes(), first)  # the same sources give the same file


if __name__ == "__main__":
    unittest.main()
