"""One split per public table, everywhere (S8, 2026-10-25): the store split real tables by a hash of their draft or
project, so tables the lake keeps for measuring (and sealing) sat in its training sets. A public table's group now takes
the lake's split however a record names it, and the screens' records name their table."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import datasets as D  # noqa: E402
from dclab_core import lake  # noqa: E402


class SplitTest(unittest.TestCase):
    def test_one_table_has_the_lakes_split_on_every_path(self):
        seen = set()
        for did in range(1, 400):
            lake_split = lake.split_of(f"openml:{did}")
            for group in (f"openml:{did}", f"dataset:openml:{did}", f"draft:openml-{did}"):
                self.assertEqual(D.split_of(group), lake_split, group)
                self.assertEqual(D.table_key(group), f"openml:{did}")
            seen.add(lake_split)
        self.assertEqual(seen, {"train", "dev", "sealed"})

    def test_the_id_is_read_strictly(self):
        for group in ("openml:02", "openml:2a", "draft:openml-2-copy", "project:openml:2", "dataset:openml:", "openml:-2",
                      "dataset:openml:2 ", "xopenml:2", "draft:openml:2:3", "openml:12345678901"):
            self.assertIsNone(D.table_key(group), group)

    def test_any_other_group_keeps_its_own_hash(self):
        for group in ("project:abc123", "draft:dr42", "dataset:craft:5", "task-c1-s3", "openml:02"):
            h = int(hashlib.sha256(f"{D.SALT}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            want = next((name for name, bound in D.SPLITS if h < bound), "train")
            self.assertEqual(D.split_of(group), want, group)
        self.assertNotEqual(D.split_of("dataset:openml:7", salt="another"), None)  # a caller's own salt still hashes


class TwinTest(unittest.TestCase):
    """The review of S8 A1 (F3): OpenML holds one table under several ids, and the hash split the copies apart."""

    def test_copies_are_found_by_rows_columns_and_qualities_and_only_ever_move_to_dev(self):
        from dclab_rnd import table_twins

        ids = {s: [d for d in range(1, 2000) if lake.base_split_of(f"openml:{d}") == s][:3] for s in ("train", "dev", "sealed")}
        t1, t2, t3 = ids["train"]
        (s1, s2, _), (d1, _, _) = ids["sealed"], ids["dev"]
        folder = Path(tempfile.mkdtemp())
        tables = {t1: (100, ["a", "B", "y"]), s1: (100, ["b", "a", "y"]),  # glass and its copy: one sealed, one train
                  t2: (50, ["x", "z"]), d1: (50, ["x", "z"]),  # a train copy of a dev table
                  t3: (70, ["p", "q"]), s2: (70, ["p", "q"])}  # the same shape, other qualities: not a twin
        for did, (rows, cols) in tables.items():
            (folder / f"{did}.json").write_text(json.dumps({"did": did, "rows": rows, "columns": {c: {} for c in cols}}))
        qualities = {did: (("NumberOfMissingValues", "0"),) for did in tables}
        qualities[s2] = (("NumberOfMissingValues", "9"),)
        doc = table_twins.build([folder], qualities)
        self.assertEqual(set(doc["held"]), {f"openml:{t1}", f"openml:{t2}"})
        self.assertEqual(set(doc["sealed_with_twins"]), {f"openml:{s1}"})
        with mock.patch.object(lake, "twins", return_value=doc):
            self.assertEqual([lake.split_of(f"openml:{d}") for d in (t1, s1, t2, d1, t3, s2)], ["dev", "dev", "dev", "dev", "train", "sealed"])
            self.assertEqual(D.split_of(f"draft:openml-{t1}"), "dev")  # the store reads the same rule

    def test_the_committed_twins_hold_the_reviews_cases(self):
        doc = lake.twins()
        self.assertIn("openml:952", doc["held"])  # prnn_fglass, a copy of glass (41, sealed for the roles exam)
        self.assertIn("openml:41", doc["sealed_with_twins"])
        self.assertEqual((lake.split_of("openml:952"), lake.split_of("openml:41")), ("dev", "dev"))


class LedgerTest(unittest.TestCase):
    def test_a_decision_inside_a_dataset_scope_names_its_table(self):
        from dclab_rnd import ledger

        home = Path(tempfile.mkdtemp())
        book = ledger.FileLedger(home / "ledger", "test")
        before = ledger._INSTALLED.get("ledger")
        ledger.install(book)
        try:
            with mock.patch.dict(os.environ, {"DCLAB_DECISION_SHADOW": "0"}):
                with ledger.dataset_scope("openml:7"):
                    inside = ledger.decide("model.family", ("project", "p1"), answers=[ledger.answer("rule", "r", "a")], value_used="a", project="p1")
                outside = ledger.decide("model.family", ("project", "p1"), answers=[ledger.answer("rule", "r", "a")], value_used="a", project="p1")
                named = None
                with ledger.dataset_scope("openml:7"):
                    named = ledger.decide("model.family", ("project", "p1"), answers=[ledger.answer("rule", "r", "a")], value_used="a",
                                          project="p1", dataset="openml:9")
        finally:
            ledger.install(before)
        self.assertEqual((inside.dataset, outside.dataset, named.dataset), ("openml:7", None, "openml:9"))  # a caller's own name wins
        self.assertEqual(D.split_of(D.group_of(project=inside.project, dataset=inside.dataset)), lake.split_of("openml:7"))


class StoreTest(unittest.TestCase):
    """The review of S8 A1: proved through the store's own keeper paths and shapers, not only the split function."""

    def test_a_lake_sealed_tables_records_are_sealed_in_the_store_however_named(self):
        from dclab_rnd.store import Layout, Store, shapers
        from dclab_rnd.store.keepers.standard import classify

        sealed = next(d for d in range(1, 400) if lake.split_of(f"openml:{d}") == "sealed" and D.split_of(f"project:p{d}") == "train")
        train = next(d for d in range(1, 400) if lake.split_of(f"openml:{d}") == "train" and D.split_of(f"draft:x{d}") == "sealed")
        layout = Layout(Path(tempfile.mkdtemp()) / "store")
        store = Store(layout)
        at = "2026-10-25T10:00:00+00:00"
        docs = [{"id": f"dr{n:020d}", "point": "model.family", "at": at, "state": {"rows": "small"}, "used": "rule", "value_used": "a",
                 "answers": [{"by": "rule", "value": "a"}], **ids} for n, ids in enumerate([
                     {"project": f"p{sealed}", "dataset": f"openml:{sealed}"},  # a screen's record: a random project, the table named
                     {"draft": f"openml-{sealed}"},  # a shadow run's record: the table as its draft
                     {"draft": f"openml-{train}"}])]
        for d in docs:
            d["schema_version"] = "dclab_core.decision_record.v2"
        store.keep_log("decisions", "real_screens__dev__ledgers__7", "ledger/decisions/2026/10/25.jsonl",
                       "".join(json.dumps(d) + "\n" for d in docs).encode())
        shapers.shape(layout)
        got = {r.body.get("id"): r.split for r in shapers.records(layout, "decisions") if r.type == "decision"}
        self.assertEqual([got[d["id"]] for d in docs], ["sealed", "sealed", "train"])
        self.assertEqual(classify("ledger/decisions/2026/10/25.jsonl")[0], "decisions")  # where the screens now keep their ledger


class ScreenTest(unittest.TestCase):
    def test_a_screens_records_name_their_table_and_sit_in_its_lake_split(self):
        warnings.filterwarnings("ignore")
        from dclab_core import benchgen3

        from dclab_rnd import real_screen as R

        frame, target, _ = benchgen3.craft_table(5, rows=300, structure="linear", task="binary")
        home = Path(tempfile.mkdtemp())
        did = next(d for d in range(1, 400) if lake.split_of(f"openml:{d}") == "dev")
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1", "DCLAB_DECISION_SHADOW": "0"}):
            got = R.screen_one({"did": did, "target": target}, home, download=lambda d: frame)
        self.assertEqual(got["split"], "dev")
        datasets = set()
        for path in (home / "dev" / "ledgers" / str(did) / "ledger").rglob("decisions/**/*.jsonl"):
            datasets |= {json.loads(line).get("dataset") for line in path.read_text().splitlines() if line.strip()}
        self.assertEqual(datasets, {f"openml:{did}"})  # every decision of the run, none without its table
        self.assertEqual((R.dataset_of("craft12"), R.split_for(did)), ("craft:12", "dev"))


if __name__ == "__main__":
    unittest.main()
