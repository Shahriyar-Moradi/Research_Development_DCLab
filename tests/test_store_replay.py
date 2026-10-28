"""36.5: replay rounds. A round is its new rows plus a fixed share of every earlier round's new rows, drawn stratum by
stratum; a round rebuilt from its own files is byte-identical; nothing earlier is changed; dev and sealed rows never enter."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_core import replay as R  # noqa: E402
from dclab_core.contracts import CONTRACTS  # noqa: E402
from dclab_rnd.store import replay  # noqa: E402
from dclab_rnd.store.layout import Layout  # noqa: E402

MT = "m4_workflow_llm"


def row(i, track="analysis", text="Sales by month?"):
    return {"id": f"{MT}:r{i}", "model_type": MT, "check": {"track": track}, "messages": [{"role": "user", "content": text}]}


class PickTest(unittest.TestCase):
    def test_the_draw_is_deterministic_and_holds_the_share_in_every_stratum(self):
        rows = [(f"a{i}", "s1") for i in range(10)] + [(f"b{i}", "s2") for i in range(3)] + [("c0", "s3")]
        first = R.pick(rows, 0.3, R.SALT, 2)
        self.assertEqual(first, R.pick(list(reversed(rows)), 0.3, R.SALT, 2))
        by = {s: sum(1 for i, k in rows if k == s and i in first) for s in ("s1", "s2", "s3")}
        self.assertEqual(by, {"s1": 3, "s2": 1, "s3": 1})  # ceil of the share: a stratum of one comes back too

    def test_the_next_round_draws_another_share(self):
        rows = [(f"a{i}", "s1") for i in range(40)]
        self.assertNotEqual(R.pick(rows, 0.3, R.SALT, 2), R.pick(rows, 0.3, R.SALT, 3))

    def test_the_contract_is_registered(self):
        self.assertIn("replay_round", CONTRACTS)


class RoundTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.layout = Layout(self.root)

    def put(self, folder, train, dev=()):
        base = self.layout.sets(MT, folder, "v1")
        base.mkdir(parents=True, exist_ok=True)
        (base / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train), encoding="utf-8")
        (base / "dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dev))
        (base / "MANIFEST.json").write_text(json.dumps({"name": folder, "version": "v1"}))

    def test_the_first_round_is_all_new_and_the_next_replays_a_share(self):
        self.put("home_analysis", [row(i) for i in range(10)], dev=[row(99)])
        first = replay.build(self.layout, MT)
        self.assertEqual((first.round, first.new, first.train, first.replayed), (1, 10, 10, []))
        self.put("home_analysis", [row(i) for i in range(14)])  # four rows arrive
        second = replay.build(self.layout, MT)
        self.assertEqual((second.round, second.new), (2, 4))
        self.assertEqual([(d.round, d.available, d.drawn) for d in second.replayed], [(1, 10, 3)])
        ids = [json.loads(x)["id"] for x in (self.layout.rounds(MT, "round_002") / "train.jsonl").read_text().splitlines()]
        self.assertEqual(len(ids), len(set(ids)))  # no row twice
        self.assertEqual(len(ids), 7)
        self.assertNotIn(f"{MT}:r99", ids)  # a dev row never enters a round

    def test_a_round_rebuilt_from_its_files_is_byte_identical(self):
        self.put("home_analysis", [row(i) for i in range(10)])
        replay.build(self.layout, MT)
        self.put("home_analysis", [row(i) for i in range(14)])
        replay.build(self.layout, MT)
        self.assertTrue(replay.rebuild(self.layout, MT, 2)["identical"])

    def test_a_changed_earlier_round_is_refused(self):
        self.put("home_analysis", [row(i) for i in range(10)])
        replay.build(self.layout, MT)
        new = self.layout.rounds(MT, "round_001") / "new.jsonl"
        new.write_text(new.read_text().replace("Sales", "Costs"))
        self.put("home_analysis", [row(i) for i in range(14)])
        with self.assertRaises(ValueError):
            replay.build(self.layout, MT)

    def test_a_changed_row_is_new_again_and_never_twice_in_a_round(self):
        self.put("home_analysis", [row(i) for i in range(10)])
        replay.build(self.layout, MT)
        self.put("home_analysis", [row(i, text="Sales by week?") for i in range(10)])
        second = replay.build(self.layout, MT)
        self.assertEqual(second.new, 10)
        ids = [json.loads(x)["id"] for x in (self.layout.rounds(MT, "round_002") / "train.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(ids), sorted(set(ids)))

    def test_a_rare_stratum_comes_back_every_round(self):
        self.put("home_analysis", [row(i) for i in range(10)] + [row(50, text="فروش ماهانه؟")])
        replay.build(self.layout, MT)
        self.put("home_analysis", [row(i) for i in range(10)] + [row(50, text="فروش ماهانه؟"), row(60)])
        second = replay.build(self.layout, MT)
        draws = second.replayed[0].by_stratum
        persian = next(k for k in draws if k.endswith("|fa"))
        self.assertEqual(draws[persian], [1, 1])

    def test_a_rounds_train_rows_are_the_sets_lines_byte_for_byte(self):
        self.put("home_analysis", [row(i) for i in range(5)])
        replay.build(self.layout, MT)
        written = set((self.layout.sets(MT, "home_analysis", "v1") / "train.jsonl").read_text().splitlines())
        kept = (self.layout.rounds(MT, "round_001") / "train.jsonl").read_text().splitlines()
        self.assertTrue(kept and set(kept) <= written)  # no field added, nothing reordered inside a row

    def test_a_row_holding_a_unicode_line_separator_stays_one_row(self):
        self.put("home_analysis", [row(1, text="first\u2028second"), row(2)])
        first = replay.build(self.layout, MT)
        self.assertEqual((first.new, first.train), (2, 2))

    def test_the_plan_writes_nothing(self):
        self.put("home_analysis", [row(i) for i in range(10)])
        plan = replay.plan(self.layout, MT)
        self.assertEqual((plan["round"], plan["new"]), (1, 10))
        self.assertFalse(self.layout.rounds(MT).exists())


class ReviewTest(RoundTest):
    """The review's scenarios: a corrected row never comes back old, a row no longer allowed never comes back, the same id
    in two sets is two rows, a missing or changed round is refused, a rebuild never reads the current sets."""

    def lines(self, n):
        return (self.layout.rounds(MT, f"round_{n:03d}") / "train.jsonl").read_text().splitlines()

    def test_a_corrected_row_never_comes_back_in_its_old_form(self):
        self.put("home_analysis", [row(i, text="WRONG") for i in range(4)])
        replay.build(self.layout, MT)
        self.put("home_analysis", [row(i, text="FIXED") for i in range(4)])
        replay.build(self.layout, MT)
        third = replay.build(self.layout, MT)
        self.assertFalse([x for x in self.lines(3) if "WRONG" in x])
        self.assertEqual(third.train, third.new + sum(d.drawn for d in third.replayed))

    def test_a_row_no_longer_in_the_train_split_is_withheld(self):
        self.put("home_analysis", [row(i) for i in range(4)])
        replay.build(self.layout, MT)
        self.put("home_analysis", [row(i) for i in range(1, 4)])  # row 0 withdrawn (consent, curation, an exam)
        second = replay.build(self.layout, MT)
        self.assertEqual(second.withheld, 1)
        self.assertFalse([x for x in self.lines(2) if '"m4_workflow_llm:r0"' in x])

    def test_the_same_id_in_two_sets_is_two_rows(self):
        self.put("home_analysis", [row(1, text="one")])
        self.put("reasoning", [row(1, text="other")])
        first = replay.build(self.layout, MT)
        self.assertEqual((first.new, first.train), (2, 2))

    def test_a_missing_earlier_round_is_refused(self):
        self.put("home_analysis", [row(i) for i in range(4)])
        replay.build(self.layout, MT)
        replay.build(self.layout, MT)
        shutil.rmtree(self.layout.rounds(MT, "round_001"))
        with self.assertRaises(ValueError):
            replay.plan(self.layout, MT)

    def test_a_rebuild_never_reads_the_current_sets_and_sees_an_edited_train_file(self):
        self.put("home_analysis", [row(i) for i in range(10)])
        replay.build(self.layout, MT)
        replay.build(self.layout, MT)
        self.put("home_analysis", [row(i, text="something else") for i in range(30)])  # the sets move on
        self.assertTrue(replay.rebuild(self.layout, MT, 2)["identical"])
        train = self.layout.rounds(MT, "round_002") / "train.jsonl"
        train.write_text(train.read_text() + train.read_text().splitlines()[0] + "\n")
        self.assertFalse(replay.rebuild(self.layout, MT, 2)["identical"])

    def test_a_half_written_build_blocks_with_a_clear_message_and_is_never_read(self):
        self.put("home_analysis", [row(i) for i in range(4)])
        self.layout.rounds(MT, "round_001.part").mkdir(parents=True)
        self.assertEqual(replay.plan(self.layout, MT)["round"], 1)  # a .part folder is no round
        with self.assertRaises(ValueError):
            replay.build(self.layout, MT)


if __name__ == "__main__":
    unittest.main()
