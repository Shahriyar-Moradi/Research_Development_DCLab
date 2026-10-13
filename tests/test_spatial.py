"""Package 16.10: places. Nearby rows with alike targets are found (Moran's I) with the distance they stay alike, the
plan then splits by blocks of places (and refuses a random split), the engine derives the blocks and keeps each one on
one side, a random split's optimism against blocks is measured on the training rows, and the maps pack runs as the
places kind."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import spatial  # noqa: E402
from dclab_core.replicas import city_prices  # noqa: E402


class SpatialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flats = city_prices.generate(0, n=2000).flats
        cls.logp = np.log(cls.flats["price"].to_numpy())

    def test_blocks_and_neighbourhood_counts_by_hand(self):
        lat = np.array([35.7, 35.7, 35.7 + 3 / 111.32])  # the third 3 km north
        lon = np.array([51.4, 51.4 + 0.5 / (111.32 * np.cos(np.radians(35.7))), 51.4])  # the second 0.5 km east
        b = spatial.blocks(lat, lon, 2.0, centre=(35.7, 51.4))
        self.assertEqual(b[0], b[1])
        self.assertNotEqual(b[0], b[2])
        np.testing.assert_array_equal(spatial.neighbourhood_counts(lat, lon, lat, lon, 1.0, same_rows=True), [1, 1, 0])  # never itself
        np.testing.assert_array_equal(spatial.neighbourhood_counts(lat, lon, lat[:1], lon[:1], 1.0), [2])
        self.assertEqual(spatial.blocks([np.nan], [51.4], 2.0, centre=(35.7, 51.4))[0], "none")

    def test_nearby_prices_are_alike_and_shuffled_ones_are_not(self):
        f = self.flats
        alike = spatial.morans_i(self.logp, f["lat"], f["lon"], permutations=99)
        self.assertGreater(alike["i"], 0.2)
        self.assertLessEqual(alike["p_value"], 0.01)
        shuffled = spatial.morans_i(np.random.default_rng(0).permutation(self.logp), f["lat"], f["lon"], permutations=99)
        self.assertLess(abs(shuffled["i"]), 0.05)
        reach = spatial.correlation_range(self.logp, f["lat"], f["lon"])
        self.assertGreater(reach["bands"][0]["correlation"], 0.3)  # alike nearby
        at = next(b for b in reach["bands"] if b["to_km"] == reach["range_km"])
        self.assertLess(at["correlation"], 0.1)  # the range is where they stop being alike

    def test_a_random_split_flatters_where_places_cluster_and_not_where_they_do_not(self):
        f = self.flats
        X = f[["lat", "lon", "size_m2", "rooms", "age_years"]]
        gap = spatial.split_optimism(X, self.logp, f["lat"], f["lon"], block_km=8, task="regression")
        self.assertGreater(gap["optimism"], 0.05)
        rng = np.random.default_rng(1)
        y = 0.9 * np.log(f["size_m2"]) + rng.normal(0, 0.1, len(f))  # no place effect at all
        flat = spatial.split_optimism(X, y, f["lat"], f["lon"], block_km=8, task="regression")
        self.assertLess(abs(flat["optimism"]), 0.03)
        with self.assertRaises(ValueError):
            spatial.split_optimism(X, self.logp, f["lat"], f["lon"], block_km=500, task="regression")  # one block: no folds


class PlanTest(unittest.TestCase):
    def test_the_plan_splits_by_blocks_and_refuses_a_random_split(self):
        from dclab_core.analyse import analyse
        from dclab_core.plan import plan, validate

        flats = city_prices.generate(1, n=1500).flats
        findings = analyse({"flats": flats}, target="price", task="regression", train_index=flats.index[:1200])
        places = next(f for f in findings.findings if f.id == "F-places")
        self.assertEqual(places.severity, "warning")
        self.assertGreater(places.evidence["optimism"], 0.02)
        p = plan(None, findings)
        split = p.decision("split.strategy")
        self.assertEqual(split.choice["kind"], "spatial_block")
        self.assertEqual((split.choice["lat_column"], split.choice["lon_column"]), ("lat", "lon"))
        self.assertIn("F-places", split.cites)
        random = p.model_copy(update={"decisions": [d.model_copy(update={"choice": {"kind": "random"}}) if d.key == "split.strategy" else d
                                                    for d in p.decisions]})
        self.assertTrue(any("F-places" in problem for problem in validate(random, findings)))
        no_place = flats.assign(price=np.random.default_rng(0).permutation(flats["price"].to_numpy()))
        quiet = analyse({"flats": no_place}, target="price", task="regression", train_index=no_place.index[:1200])
        self.assertNotEqual(next(f for f in quiet.findings if f.id == "F-places").severity, "warning")
        self.assertNotEqual(plan(None, quiet).decision("split.strategy").choice["kind"], "spatial_block")

    def test_places_are_read_on_training_rows_and_time_comes_first(self):
        from dclab_core.analyse import analyse
        from dclab_core.plan import _guard, plan

        flats = city_prices.generate(1, n=1500).flats
        train = flats.index[:1200]
        one = next(f for f in analyse({"flats": flats}, target="price", task="regression", train_index=train).findings if f.id == "F-places")
        later = flats.copy()
        later.loc[1200:, "price"] = np.random.default_rng(2).permutation(later.loc[1200:, "price"].to_numpy())  # the rows the plan may not read
        two = next(f for f in analyse({"flats": later}, target="price", task="regression", train_index=train).findings if f.id == "F-places")
        self.assertEqual(one.evidence, two.evidence)
        dated = flats.assign(listed_at=pd.date_range("2024-01-01", periods=len(flats), freq="12h").strftime("%Y-%m-%d"))
        dated["price"] = dated["price"] * np.linspace(1.0, 1.8, len(dated))  # prices rise over two years
        p = plan(None, analyse({"flats": dated}, target="price", task="regression", train_index=train))
        split = p.decision("split.strategy")
        self.assertEqual(split.choice["kind"], "time")  # DCLAB-R02 first; the blocks are the stated risk
        self.assertIn("spatial_block", [a.choice for a in split.alternatives])
        findings = analyse({"flats": flats}, target="price", task="regression", train_index=train)
        blocks = plan(None, findings)
        self.assertIn("random split", _guard("split.strategy", {"kind": "random"}, blocks, findings))  # an advisor may not drop the blocks
        self.assertIsNone(_guard("split.strategy", blocks.decision("split.strategy").choice, blocks, findings))

    def test_coordinates_not_in_degrees_give_no_blocks_and_the_antimeridian_is_near(self):
        from dclab_core.analyse import analyse

        flats = city_prices.generate(1, n=600).flats
        metres = flats.assign(lat=flats["lat"] * 111_000, lon=flats["lon"] * 90_000)
        f = next(f for f in analyse({"flats": metres}, target="price", task="regression", train_index=metres.index[:500]).findings if f.id == "F-places")
        self.assertEqual(f.severity, "info")
        self.assertFalse(f.evidence["degrees"])
        xy = spatial.project([0.0, 0.0], [179.95, -179.95])
        self.assertLess(np.linalg.norm(xy[0] - xy[1]), 20)  # about 11 km, not 40,000

    def test_the_solution_takes_blocks_from_the_plan_and_keeps_one_split(self):
        try:
            from dclab_rnd.core_adapter import solution_fields_from_plan
            from dclab_rnd.studio.solution import Solution
        except ImportError as error:
            self.skipTest(str(error))
        from dclab_core.analyse import analyse
        from dclab_core.plan import plan

        flats = city_prices.generate(1, n=1500).flats
        p = plan(None, analyse({"flats": flats}, target="price", task="regression", train_index=flats.index[:1200]))
        fields = solution_fields_from_plan(p, "price", "When a flat is listed, before it is sold.")
        solution = Solution(**fields)
        self.assertEqual(solution.spatial.lat_column, "lat")
        with self.assertRaises(ValueError):
            Solution(**{**fields, "group_column": "rooms"})  # one way to split
        with self.assertRaises(ValueError):
            Solution(**{**fields, "forbidden": [{"column": "lat", "reason": "x"}]})  # the blocks' coordinates cannot be forbidden


class EngineTest(unittest.TestCase):
    def test_the_engine_keeps_each_block_on_one_side_and_the_maps_pack_runs_as_places(self):
        try:
            from dclab_rnd import kinds
            from dclab_rnd.draft import pack
            from dclab_rnd.expansion import runner
            from dclab_rnd.kinds import acceptance
            from dclab_rnd.models.gateway import private_scope
            from dclab_rnd.studio import engine
        except ImportError as error:
            self.skipTest(str(error))
        kind = kinds.get("places")
        case = kind.case(0)
        with tempfile.TemporaryDirectory() as tmp:
            store, pid = acceptance.build(case, Path(tmp))
            with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
                engine.run_all(store, pid)
            data = store.read_stage(pid, "data")["evidence"]
            run = acceptance.Run(case, store.get(pid)["solution"], {s: store.read_stage(pid, s) for s in engine.STAGE_KEYS})
        self.assertEqual(data["problem_kind"], "places")
        self.assertIn("_spatial_block", data["holdout_split"])  # the derived blocks are the groups of the split
        self.assertEqual(data["group_stats"]["train_groups"], data["spatial"]["train_blocks"])
        self.assertEqual(data["spatial"]["blocks_on_both_sides"], 0)
        self.assertTrue(all(c["ok"] for c in kind.accept(run)), kind.accept(run))
        few = kind.case(0)
        with tempfile.TemporaryDirectory() as tmp:
            store, pid = acceptance.build(few, Path(tmp))
            project = store.get(pid)
            project["solution"]["spatial"]["block_km"] = 30.0  # four blocks: too few for the folds
            store.save(project)
            with self.assertRaises(ValueError) as refused, runner.fast_profile(), private_scope(True):
                engine.prepare(store, store.get(pid))
        self.assertIn("choose smaller blocks", str(refused.exception))
        self.assertEqual({c["key"]: c for c in pack.cards([{"kind": k.key, "status": "supported", "missing": [],
                                                              "checks": [{"decision": "d", "ok": True}]} for k in kinds.all_kinds()])}["maps"]["maturity"],
                         "Supported")


if __name__ == "__main__":
    unittest.main()
