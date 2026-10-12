"""Package 18.5: the feature language (dclab_core.features), its checks, its fold safety, the feature engineer role,
and the engineered recipe on the fraud replica. No model is called: the engineer here is scripted."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pydantic  # noqa: F401
except ImportError as error:
    PYDANTIC_MISSING = str(error)
else:
    PYDANTIC_MISSING = ""

REPEAT = {"name": "biker_dropoff_repeat", "op": "count", "by": ["biker_id", {"place": "destination"}]}
SHORT = {"name": "biker_short_share", "op": "mean", "of": {"below": [{"distance": ["source", "destination"]}, 50]}, "by": ["biker_id"]}


def rides(n=500, seed=0):
    rng = np.random.default_rng(seed)
    lat, lon = 35.7 + rng.normal(0, 0.02, n), 51.4 + rng.normal(0, 0.02, n)
    return pd.DataFrame({"biker_id": rng.integers(0, 30, n), "created": pd.Timestamp("2026-03-01") + pd.to_timedelta(np.sort(rng.integers(0, 20 * 86400, n)), unit="s"),
                         "source_lat": lat, "source_lon": lon, "destination_lat": lat + rng.normal(0, 0.01, n),
                         "destination_lon": lon + rng.normal(0, 0.01, n), "fare": rng.gamma(3, 30, n), "is_case": rng.integers(0, 2, n)})


class LanguageTest(unittest.TestCase):
    def language(self):
        from dclab_core.features import Language

        return Language(["source_lat", "source_lon", "destination_lat", "destination_lon", "fare"], target="is_case",
                        time_column="created", entities=["biker_id"])

    def test_programs_that_read_the_target_or_a_forbidden_column_are_refused(self):
        kept, refused = self.language().check_all([
            REPEAT, SHORT,
            {"name": "leak", "op": "mean", "of": "is_case", "by": ["biker_id"]},                 # the target
            {"name": "secret", "op": "value", "of": "refund_paid_after"},                          # not a usable column
            {"name": "biker_as_value", "op": "value", "of": "biker_id"},                           # an entity groups, never a value
            {"name": "magic", "op": "predict", "of": "fare"},                                      # not an operation
            {"name": "biker_dropoff_repeat", "op": "count", "by": ["biker_id"]},                   # a name used twice
            {"name": "far", "op": "value", "of": {"distance": ["source", "moon"]}}])
        self.assertEqual([p["name"] for p in kept], ["biker_dropoff_repeat", "biker_short_share"])
        self.assertEqual(len(refused), 6)
        self.assertTrue(any("target" in r["why"] for r in refused))

    def test_no_test_row_changes_a_training_feature_and_events_see_no_later_row(self):
        from dclab_core.features import Programs

        frame = rides()
        programs = [REPEAT, SHORT, {"name": "gap", "op": "since_previous", "by": ["biker_id"]},
                    {"name": "fare_roll", "op": "rolling_mean", "of": "fare", "by": ["biker_id"], "rows": 3},
                    {"name": "fare_vs_biker", "op": "deviation", "of": "fare", "by": ["biker_id"]},
                    {"name": "biker_share", "op": "share", "by": ["biker_id"]}]
        for mode in ("audit", "event"):
            with self.subTest(mode=mode):
                model = Programs(self.language(), programs, mode).fit(frame.iloc[:300])
                alone = model.transform(frame.iloc[:300])
                pd.testing.assert_frame_equal(model.transform(frame.iloc[:300]), alone)
                model.transform(frame.iloc[300:].assign(fare=1e6))  # other rows never touch these
                pd.testing.assert_frame_equal(model.transform(frame.iloc[:300]), alone)
        event = Programs(self.language(), programs, "event")
        early = event.transform(frame.iloc[:200])
        mixed = event.transform(pd.concat([frame.iloc[:200], frame.iloc[200:].assign(fare=1e6, biker_id=frame["biker_id"].iloc[0])], ignore_index=True))
        pd.testing.assert_frame_equal(mixed.iloc[:200].reset_index(drop=True), early)  # no later row reaches back
        self.assertTrue(np.isnan(early["prog__gap"].iloc[0]))  # the first row of a biker has no previous one

    def test_rows_at_the_same_time_never_see_each_other_and_a_missing_time_has_no_past(self):
        from dclab_core.features import Programs

        n = 600
        rng = np.random.default_rng(5)
        frame = pd.DataFrame({"biker_id": rng.integers(0, 10, n), "fare": rng.gamma(3, 30, n),
                              "created": pd.to_datetime("2026-03-01") + pd.to_timedelta(rng.integers(0, 10, n), unit="D")})
        y = rng.integers(0, 2, n)
        frame = frame.assign(y=y).sort_values(["created", "y"]).drop(columns="y").reset_index(drop=True)  # labels sorted within each day
        programs = [{"name": "n", "op": "count", "by": ["biker_id"]}, {"name": "gap", "op": "since_previous", "by": ["biker_id"]},
                    {"name": "roll", "op": "rolling_mean", "of": "fare", "by": ["biker_id"], "rows": 3}]
        out = Programs(self.language(), programs, "event").transform(frame)
        reordered = Programs(self.language(), programs, "event").transform(frame.sample(frac=1, random_state=0).sort_values("created", kind="stable"))
        for column in out:  # the file's order inside a timestamp changes nothing: same-time rows are not each other's past
            np.testing.assert_allclose(np.sort(out[column].fillna(-1).to_numpy()), np.sort(reordered[column].fillna(-1).to_numpy()))
        self.assertFalse((out["prog__gap"] == 0).any())
        missing = frame.copy()
        missing.loc[0, "created"] = pd.NaT
        self.assertTrue(np.isnan(Programs(self.language(), programs, "event").transform(missing)["prog__n"].iloc[0]))

    def test_the_behaviour_recipe_written_in_the_language(self):
        from dclab_core.features import Programs, behaviour_programs

        language = self.language()
        programs = behaviour_programs(["biker_id"], language, ["fare"])
        names = {p["name"] for p in programs}
        self.assertTrue({"trip_metres", "biker_id_destination_repeat", "biker_id_short_share", "fare_per_metre"} <= names)
        out = Programs(language, programs, "audit").transform(rides())
        self.assertEqual(list(out.columns), [f"prog__{p['name']}" for p in programs])
        self.assertTrue(np.isfinite(out["prog__fare_per_metre"]).all())


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class EngineerTest(unittest.TestCase):
    def test_the_role_keeps_checked_programs_and_says_what_it_refused(self):
        from dclab_core.agents import ScriptedAdvisor
        from dclab_core.features import Language, engineer

        frame = rides()
        language = Language(["source_lat", "source_lon", "destination_lat", "destination_lon", "fare"], target="is_case",
                            time_column="created", entities=["biker_id"])
        advisor = ScriptedAdvisor({"feature_engineer": {"programs": [REPEAT, {"name": "x", "op": "mean", "of": "is_case", "by": ["biker_id"]}]}})
        out = engineer(advisor, frame, language, moment="In a daily audit", task="binary", entities=["biker_id"])
        self.assertEqual([p["name"] for p in out["programs"]], ["biker_dropoff_repeat"])
        self.assertEqual(len(out["refused"]), 1)
        [(purpose, evidence)] = advisor.requests
        self.assertEqual(evidence["coordinate_pairs"], ["destination", "source"])
        self.assertNotIn("35.7", str(evidence))  # names and shares, never a value
        self.assertEqual(engineer(None, frame, language)["programs"], [])  # no model: no engineered recipe

    def test_on_the_fraud_replica_the_ladder_keeps_the_engineered_programs_for_a_measured_gain(self):
        from dclab_core.replicas import delivery_fraud as dfr
        from dclab_rnd.expansion import runner
        from dclab_rnd.studio import ProjectStore, data as sd, engine
        from dclab_rnd.studio.solution import Solution

        replica = dfr.generate(seed=0)
        store = ProjectStore(Path(tempfile.mkdtemp()))
        project = store.create("Rides", "delivery", "Find fraudulent rides")
        paths = dfr.save(replica, Path(tempfile.mkdtemp()))
        (store.data_dir(project["id"]) / dfr.RIDE_FILE).write_bytes(paths["rides"].read_bytes())
        sd.attach_data(store, project["id"], dfr.RIDE_FILE)
        (store.data_dir(project["id"]) / dfr.SAMPLE_FILE).write_bytes(paths["sample"].read_bytes())
        project = sd.attach_labels(store, project["id"], dfr.SAMPLE_FILE, target="is_fraud")
        project["solution"] = Solution(**{**project["suggestion"], "prediction_moment": "After delivery, in a daily audit of the day's rides.",
                                          "group_column": "biker_id"}).model_dump()
        project["feature_programs"] = [REPEAT, SHORT]
        project["settings"]["quick"] = True
        store.save(project)
        with runner.fast_profile():
            engine.execute(store, project["id"], "data")
            engine.execute(store, project["id"], "leakage")
            features = engine.execute(store, project["id"], "features")
        rows = {r["recipe"]: r for r in features["evidence"]["stage_results"]}
        self.assertIn("engineered", rows)
        gain = runner._mean(rows["engineered"], "average_precision") - runner._mean(rows["raw"], "average_precision")
        self.assertGreater(gain, 0.1)  # the two programs alone lift the raw recipe on folds grouped by biker


if __name__ == "__main__":
    unittest.main()
