"""Package 15.1: the delivery-fraud replica, and the decisions DCLab must reach on it (Part 4 of the playbook).

``ReplicaTest`` pins the replica itself: deterministic, every trap planted, and the signal recoverable by the method a
careful data scientist used on the real challenge (behaviour features, folds grouped by biker), not by the raw columns.

``AcceptanceTest`` is what "DCLab can solve this" means. Each test is marked ``expectedFailure`` until the package named
in it is built; unittest reports an unexpected success as a failure, so the package that makes one pass must remove
its marker. The function names in these tests are a proposal: a later package may rename them, never weaken the
decisions they check.

Baseline recorded on 2026-10-12 (seed 0, labels joined correctly by hand, the engine as it was): holdout average
precision 0.277 (95 % cluster bootstrap over 48 bikers 0.086-0.501) against a chance level of 0.006, with the row-wise
recipe "selected_mi" and extra trees. The hand method below (behaviour features, folds grouped by biker) reaches
0.47-0.54 over seeds 0-2, and 0.41-0.67 on five simulated holdouts of seed 0.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core.replicas import delivery_fraud as dfr  # noqa: E402

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
except ImportError as error:  # the agent environment
    SKLEARN_MISSING = str(error)
else:
    SKLEARN_MISSING = ""

try:
    from dclab_rnd.studio import ProjectStore, data as sd, engine
    from dclab_rnd.studio.solution import Solution
except ImportError as error:  # pydantic is not part of requirements/ci.txt
    STUDIO_MISSING = str(error)
else:
    STUDIO_MISSING = ""

MOMENT = "After delivery, in a daily audit of the day's rides; no label is known when a ride is scored."
BUDGET = 0.005  # the share of rides the quality team can review
_REPLICA = {}


def replica(seed: int = 0) -> "dfr.Replica":
    """Generate each seed once per process (a few seconds each)."""
    if seed not in _REPLICA:
        _REPLICA[seed] = dfr.generate(seed=seed)
    return _REPLICA[seed]


def _scorable(rides: pd.DataFrame) -> pd.DataFrame:
    return rides[(rides["status"] == "DELIVERED") & rides["biker_id"].notna()].copy()


def _behaviour(d: pd.DataFrame) -> pd.DataFrame:
    """The hand solution's core features: distance, repeat drop-off, the biker's short-trip rate, fare per metre."""
    lat1, lon1, lat2, lon2 = (np.radians(d[c]) for c in ("source_lat", "source_lon", "destination_lat", "destination_lon"))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    dist = 2 * 6_371_000 * np.arcsin(np.sqrt(a))
    cell = d["destination_lat"].round(4).astype(str) + "_" + d["destination_lon"].round(4).astype(str)
    out = pd.DataFrame({"distance_m": dist}, index=d.index)
    out["dropoff_repeat"] = d.groupby([d["biker_id"], cell])["order_id"].transform("size")
    out["biker_short_rate"] = (dist < 50).groupby(d["biker_id"]).transform("mean")
    out["fare_per_metre"] = d["biker_fare"] / np.maximum(dist, 20)
    return out


def _oof_ap(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, grouped: bool) -> float:
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=0) if grouped else StratifiedKFold(3, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for train, test in cv.split(X, y, groups):
        model = RandomForestClassifier(150, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=1, random_state=0)
        oof[test] = model.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])[:, 1]
    return float(average_precision_score(y, oof))


class ReplicaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replica = replica(0)

    def test_same_seed_same_files_other_seed_other_files(self):
        again = dfr.generate(seed=0)
        pd.testing.assert_frame_equal(self.replica.rides, again.rides)
        pd.testing.assert_frame_equal(self.replica.sample, again.sample)
        other = replica(1)
        self.assertFalse(other.rides["order_id"].equals(self.replica.rides["order_id"]))

    def test_the_traps_are_planted(self):
        r, f = self.replica, self.replica.facts
        # the coarse key inflates the labels many times over; the order is the right key
        self.assertGreaterEqual(f["pair_join_rows"], 10 * f["known_fraud_rows"])
        self.assertEqual(f["accounts"], 3)
        # partial labels: the sample is a share of the planted frauds, and some of its rows are not in the table
        self.assertLess(f["known_fraud_rows"], f["planted_fraud_rows"])
        self.assertEqual(f["sample_rows_not_in_table"], 5)
        # one name, two meanings: sender_phone holds the receiver's phone in the sample
        self_dealing = r.sample[r.sample["reason"] == dfr.SELF_DEALING]
        self.assertTrue((self_dealing["sender_phone"] == self_dealing["biker_phone"]).all())
        self.assertEqual(int(r.sample["sender_phone"].isin(set(r.rides["sender_phone"])).sum()), 0)
        matched = r.sample.merge(r.rides, on="order_id", suffixes=("_sample", ""))
        self.assertLess(np.isclose(matched["biker_fare_sample"], matched["biker_fare"]).mean(), 0.3)
        # rare, and clustered on a few bikers
        self.assertLess(f["known_fraud_rows"] / f["scorable_rides"], 0.01)
        self.assertLessEqual(f["fraud_bikers"], 0.2 * self.replica.rides["biker_id"].nunique())
        # cancelled rides have no biker
        cancelled = r.rides[r.rides["status"] == "CANCELLED"]
        self.assertGreater(len(cancelled), 0)
        self.assertTrue(cancelled["biker_id"].isna().all())
        # some orders have several drop-offs, and order ids rise with time (as in real exports)
        self.assertGreater(int(r.rides["order_id"].duplicated().sum()), 0)
        self.assertTrue(r.rides["order_id"].is_monotonic_increasing)
        self.assertTrue(pd.to_datetime(r.rides["created_date"]).is_monotonic_increasing)

    @unittest.skipIf(SKLEARN_MISSING, SKLEARN_MISSING)
    def test_the_signal_is_in_behaviour_and_a_random_split_flatters_raw_columns(self):
        d = _scorable(self.replica.rides)
        y = d["order_id"].isin(set(self.replica.sample["order_id"])).astype(int).to_numpy()
        groups = d["biker_id"].to_numpy()
        raw = d[["biker_fare", "source_lat", "source_lon", "destination_lat", "destination_lon"]]
        behaviour = _oof_ap(_behaviour(d), y, groups, grouped=True)
        raw_grouped = _oof_ap(raw, y, groups, grouped=True)
        raw_random = _oof_ap(raw.assign(biker_id=d["biker_id"]), y, groups, grouped=False)
        chance = y.mean()
        self.assertGreater(behaviour, 25 * chance)        # recoverable, honestly (bikers unseen in test)
        self.assertGreater(behaviour, 3 * raw_grouped)    # by behaviour, not by the raw columns
        self.assertGreater(raw_random, 2 * raw_grouped)   # a random split rewards recognising the biker

    def test_save_writes_the_two_files_marked_synthetic_and_never_the_truth(self):
        directory = Path(tempfile.mkdtemp())
        paths = dfr.save(self.replica, directory)
        self.assertEqual(sorted(p.name for p in directory.iterdir()), sorted([dfr.RIDE_FILE, dfr.SAMPLE_FILE, dfr.INFO_FILE]))
        info = paths["info"].read_text()
        self.assertIn('"synthetic": true', info)
        rides = pd.read_csv(paths["rides"])
        self.assertTrue(rides.columns[0].startswith("Unnamed"))  # the index column arrives as files often do
        self.assertNotIn("kind", rides.columns)
        self.assertNotIn("receiver_phone", rides.columns)


@unittest.skipIf(STUDIO_MISSING, STUDIO_MISSING)
class AcceptanceTest(unittest.TestCase):
    """The decisions DCLab must reach on the replica. Each marker names the package that removes it."""

    @classmethod
    def setUpClass(cls):
        cls.replica = replica(0)

    def test_labels_are_joined_on_the_order_and_the_pair_join_is_refused(self):
        from dclab_rnd.studio import labels

        review = labels.review(self.replica.rides, self.replica.sample)
        self.assertEqual(review["key"], ["order_id"])
        self.assertEqual(review["rows_marked"], self.replica.facts["known_fraud_rows"])
        pair = next(c for c in review["candidates"] if set(c["key"]) == {"account_id", "biker_id"})
        self.assertEqual(pair["verdict"], "refused")
        self.assertEqual(pair["rows_marked"], self.replica.facts["pair_join_rows"])
        self.assertEqual(review["not_found"], self.replica.facts["sample_rows_not_in_table"])
        self.assertIn("sender_phone", review["disagreeing_columns"])
        self.assertIn("biker_fare", review["disagreeing_columns"])

    def test_labels_from_a_sample_are_declared_partial(self):
        solution = Solution(target="is_fraud", task="binary", positive_label="yes", prediction_moment=MOMENT,
                            group_column="biker_id", labels="partial")
        self.assertEqual(solution.labels, "partial")

    def test_metrics_at_an_alert_budget(self):
        from dclab_rnd.expansion import runner

        y = np.r_[np.ones(5), np.zeros(995)]
        scores = np.r_[np.linspace(0.9, 0.99, 5), np.linspace(0, 0.8, 995)]
        metrics = runner.binary_metrics(y, scores, alert_budget=0.005)
        self.assertEqual(metrics["recall_at_budget"], 1.0)
        self.assertEqual(metrics["precision_at_budget"], 1.0)
        Solution(target="is_fraud", task="binary", positive_label="yes", prediction_moment=MOMENT, alert_budget=0.005)

    def test_the_screen_holds_rules_anomaly_and_pu_families(self):
        from dclab_rnd.expansion import runner

        families = set(runner.model_families("binary_imbalanced")) | set(runner.RARE_FAMILIES)  # the engine adds these for a rare event
        self.assertTrue({"percentile_rules", "isolation_forest", "ecod", "pu_bagging"} <= families)

    def test_the_pack_is_found_from_a_sparse_label_file_without_the_word_fraud(self):
        from dclab_rnd.draft import pack

        found = pack.detect("Which rides should our quality team review first?",
                            analysis={"labels": {"rows_marked": 100, "rows": 16_437}})
        self.assertEqual(found["key"], "imbalanced")
        self.assertEqual(found["source"], "data")

    def test_a_finished_project_exports_a_scoring_service(self):
        from dclab_rnd.studio import service

        self.assertTrue(callable(service.export))  # tested end to end in tests/test_service_export.py

    def test_end_to_end_the_engine_reaches_the_hand_solutions_decisions(self):
        store = ProjectStore(Path(tempfile.mkdtemp()))
        project = store.create("Rides", "delivery", "Find fraudulent activity of bikers and customers")
        paths = dfr.save(self.replica, Path(tempfile.mkdtemp()))  # the person's two files
        directory = store.data_dir(project["id"])
        (directory / dfr.RIDE_FILE).write_bytes(paths["rides"].read_bytes())
        sd.attach_data(store, project["id"], dfr.RIDE_FILE)
        (directory / dfr.SAMPLE_FILE).write_bytes(paths["sample"].read_bytes())
        project = sd.attach_labels(store, project["id"], dfr.SAMPLE_FILE, target="is_fraud")
        suggestion = project["suggestion"]
        self.assertEqual(suggestion["labels"], "partial")  # 15.2: labels from a sample file are partial
        # the entity and the budget are stated here; asking for them is Home's job (15.6)
        project["solution"] = Solution(**{**suggestion, "prediction_moment": MOMENT, "group_column": "biker_id",
                                          "alert_budget": BUDGET}).model_dump()
        store.save(project)
        engine.run_all(store, project["id"])
        features = store.read_stage(project["id"], "features")
        self.assertEqual(features["decision"]["chosen"], "entity_behaviour")    # 15.4: behaviour, not raw columns
        self.assertIn("biker_id", features["evidence"]["cv_protocol"])          # folds grouped by the entity
        models = store.read_stage(project["id"], "models")
        screened = {row["model"] for row in models["evidence"]["model_results"]}
        self.assertTrue({"isolation_forest", "pu_bagging"} <= screened)          # 15.5: the families compared
        self.assertTrue(models["decision"]["rule"])                              # chosen by a stated rule
        final = store.read_stage(project["id"], "final")["evidence"]
        chance = final["baselines"]["majority_class"]["average_precision_random_ranking"]
        # a holdout grouped by biker holds a handful of fraud bikers (4 to 11 in a simulation of the hand method), so the
        # interval is wide and its lower end can fall below chance with a good model: the test asks for the interval to
        # be computed over bikers and for the point estimate to be far above chance (the hand method: about 0.5 vs 0.006)
        self.assertIn("cluster bootstrap", final["holdout_primary_metric_ci"]["method"])
        self.assertGreater(final["holdout_metrics"]["average_precision"], 20 * chance)
        self.assertGreater(final["holdout_metrics"]["average_precision"], 0.35)  # the engine as it was: 0.277
        self.assertGreater(final["holdout_metrics"]["recall_at_budget"], BUDGET)  # 15.3: better than reviewing at random
        self.assertIn("lower bound", final["precision_note"])                    # partial labels: precision is a floor
        limits = " ".join(final["limits"])
        self.assertIn("sender_phone", limits)  # 15.2: a column that means something else in the label file is a stated limit


if __name__ == "__main__":
    unittest.main()
