"""34.7 with 34.6 in the engine: a project whose settings ask for a search gets Optuna on the stage's choice and the
screen's next family, the winner confirmed once on a selection fold set aside before the search, beside the choice
without a search scored the same way, and the final stage fitting the chosen settings; a project that does not ask runs
exactly as before. The holdout is read once, by the final stage only.

The review of 34.7: on a time split with entities, the selection fold never holds an entity the search rows hold; a
classification keeps every class on both sides (another seed, or the search skipped with the reason); the review's kept
change is the base of every trial, never weighted twice; the ledger says the search chose the family; a run's lineage
counts the search among its settings.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import optuna  # noqa: F401

    from dclab_rnd import kinds, ledger
    from dclab_rnd.kinds import acceptance
except ImportError as error:  # pragma: no cover
    MISSING = str(error)
else:
    MISSING = ""


def run(search):
    from dclab_rnd.expansion import runner
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import engine

    store, pid = acceptance.build(kinds.get("binary").case(0), Path(tempfile.mkdtemp()))
    if search is not None:
        project = store.get(pid)
        project["settings"] = {**project.get("settings", {}), "search": search}
        store.save(project)
    with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
        engine.run_all(store, pid)
    return store, pid


@unittest.skipIf(MISSING, MISSING)
class SearchEngineTest(unittest.TestCase):
    def setUp(self):
        self.book = ledger.FileLedger(Path(tempfile.mkdtemp()) / "ledger", "wtest")
        ledger.install(self.book)

    def tearDown(self):
        ledger.install(None)

    def test_a_search_is_recorded_confirmed_on_its_selection_fold_and_fitted_by_the_final_stage(self):
        store, pid = run({"minutes": 1, "trials": 6, "families": 2})
        models, final = store.read_stage(pid, "models"), store.read_stage(pid, "final")
        found = models["evidence"]["search"]
        self.assertNotIn("skipped", found)
        self.assertGreaterEqual(found["attempts"], 3)  # two families' starting settings and at least one trial
        chosen = found["selection"]
        self.assertEqual(chosen["schema_version"], "dclab_core.selection_record.v1")
        self.assertEqual(chosen["top_k"], 1)
        self.assertIsNotNone(chosen["selection_score"])
        self.assertIn("selection fold", found["rule"])
        self.assertIn("pooled out-of-fold", found["rule"])
        against = found["without_search"]  # the stage's own choice, unsearched, on the same selection fold
        self.assertIn(against["family"], [r["model"] for r in models["evidence"]["ranking"]])
        self.assertEqual(len(against["gain_interval"]), 2)
        if against["name"] == f"{found['family']}:default" and not found["params"]:
            self.assertEqual(against["gain"], 0.0)
        self.assertEqual(models["evidence"]["selected_model"], found["family"])
        self.assertEqual(models["evidence"]["selected_metric_mean"], chosen["search_score"])  # the chosen one's own number
        s1 = next(c for c in models["claims"] if c["claim_id"].endswith("-S1"))
        self.assertIn("without a search", s1["statement"])
        self.assertEqual(final["evidence"]["selected_model_params"] or {}, found["params"] or {})  # fitted as searched, never re-tuned
        self.assertEqual(final["evidence"]["optimization_search_space"], [])
        self.assertIn("search chose these settings", final["evidence"]["selection_rule"])
        self.assertEqual(len(found["every_attempt"]), found["attempts"])  # every attempt, in the order made
        self.assertEqual([a["index"] for a in found["every_attempt"]], list(range(found["attempts"])))
        from dclab_rnd.studio import engine

        p = engine.prepare(store, store.get(pid))  # the selection fold left every stage's folds before any stage scored
        aside = set(p.search_cut["select_rows"].tolist())
        self.assertFalse(aside & {int(i) for a, b in p.bundle.cv_splits for i in list(a) + list(b)})
        self.assertIn("set aside for the search", models["evidence"]["cv_protocol"])
        self.assertEqual(store.get(pid).get("holdout_uses"), 1)  # read once, by the final stage
        family = [r for r in self.book.records(point="model.family")]
        self.assertTrue(family)
        self.assertEqual(family[-1].answers[0].answerer, "rule:search_select")

    def test_the_exported_code_reproduces_a_searched_holdout_number(self):
        import json
        import subprocess

        from dclab_rnd.studio import code

        store, pid = run({"minutes": 1, "trials": 4, "families": 2})
        folder = Path(tempfile.mkdtemp()) / "code"
        code.export(store, pid, folder)
        out = subprocess.run([sys.executable, "-I", str(folder / "pipeline.py")], capture_output=True, text=True, cwd=tempfile.mkdtemp(), timeout=900)
        self.assertEqual(out.returncode, 0, out.stderr[-1500:])
        result = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertTrue(result["same"], result)  # the searched settings are what the code fits

    def test_a_skipped_search_leaves_the_stage_as_it_was(self):
        from dclab_rnd.studio import search

        plain_store, plain = run(None)
        with mock.patch.object(search, "cut", return_value="the latest rows set aside for the selection fold miss a class"):
            store, pid = run({"minutes": 1, "trials": 4, "families": 2})
        models, final = store.read_stage(pid, "models"), store.read_stage(pid, "final")
        self.assertIn("miss a class", models["evidence"]["search"]["skipped"])
        s1 = next(c for c in models["claims"] if c["claim_id"].endswith("-S1"))
        self.assertEqual(s1["kind"], "fact")
        plain_models = plain_store.read_stage(plain, "models")
        for key in ("selected_model", "selected_metric_mean", "cv_protocol"):
            self.assertEqual(models["evidence"][key], plain_models["evidence"][key], key)
        self.assertTrue(final["evidence"]["optimization_search_space"])  # the final stage tunes as it always did
        self.assertFalse(final["evidence"]["tuning_decision"]["from_search"])
        self.assertEqual([r.answers[0].answerer for r in self.book.records(point="model.family")][-1], "rule:rank_rows")

    def test_the_models_stage_and_its_search_never_read_the_holdout(self):
        from dclab_rnd.expansion import datasets, runner
        from dclab_rnd.models.gateway import private_scope
        from dclab_rnd.studio import engine

        store, pid = acceptance.build(kinds.get("binary").case(0), Path(tempfile.mkdtemp()))
        project = store.get(pid)
        project["settings"] = {**project.get("settings", {}), "search": {"minutes": 1, "trials": 4, "families": 2}}
        store.save(project)

        def sealed(self):
            raise AssertionError("the holdout was read before the final stage")

        with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            for stage in engine.STAGE_KEYS[:-2]:
                engine.execute(store, pid, stage)
            with mock.patch.object(datasets.TaskBundle, "X_test", property(sealed)), mock.patch.object(datasets.TaskBundle, "y_test", property(sealed)):
                engine.execute(store, pid, "models")  # the screen, the review and the search
            self.assertIn("search", store.read_stage(pid, "models")["evidence"])
            engine.execute(store, pid, engine.STAGE_KEYS[-1])
        self.assertEqual(store.get(pid).get("holdout_uses"), 1)

    def test_on_a_time_split_the_leakage_stages_random_folds_skip_the_selection_fold(self):
        from dclab_rnd.expansion import runner
        from dclab_rnd.models.gateway import private_scope
        from dclab_rnd.studio import engine

        case = kinds.get("binary").case(0)
        case.solution = {**case.solution, "time_column": "signup_time"}
        store, pid = acceptance.build(case, Path(tempfile.mkdtemp()))
        project = store.get(pid)
        project["settings"] = {**project.get("settings", {}), "search": {"minutes": 1, "trials": 4, "families": 2}}
        store.save(project)
        seen = {}
        real = runner.cross_validate

        def recording(bundle, recipe, family, *args, **kwargs):
            if kwargs.get("config_id") == "safe_random_kfold":
                seen["rows"] = {int(i) for a, b in kwargs["splits"] for i in list(a) + list(b)}
            return real(bundle, recipe, family, *args, **kwargs)

        with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None), \
                mock.patch.object(runner, "cross_validate", recording):
            for stage in engine.STAGE_KEYS[:2]:
                engine.execute(store, pid, stage)
            p = engine.prepare(store, store.get(pid))
        self.assertEqual(p.bundle.spec.split_strategy, "time")
        self.assertTrue(seen["rows"])
        self.assertFalse(seen["rows"] & set(p.search_cut["select_rows"].tolist()))  # the latest training rows stay unread
        self.assertIn("random_vs_time_cv", store.read_stage(pid, "leakage")["evidence"])

    def test_a_persons_adopted_change_turns_the_search_flag_off(self):
        from dclab_rnd.expansion import runner
        from dclab_rnd.models.gateway import private_scope
        from dclab_rnd.studio import engine

        store, pid = acceptance.build(kinds.get("binary").case(0), Path(tempfile.mkdtemp()))
        project = store.get(pid)
        project["settings"] = {**project.get("settings", {}), "search": {"minutes": 1, "trials": 4, "families": 2}}
        store.save(project)
        with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            for stage in engine.STAGE_KEYS[:-1]:
                engine.execute(store, pid, stage)
            models, features = store.read_stage(pid, "models"), store.read_stage(pid, "features")
            project = store.get(pid)
            project["adopted_improvement"] = {"family": models["evidence"]["selected_model"], "recipe": features["evidence"]["selected_recipe"],
                                              "params": {}}
            store.save(project)
            engine.execute(store, pid, engine.STAGE_KEYS[-1])
        final = store.read_stage(pid, "final")
        self.assertFalse(final["evidence"]["tuning_decision"]["from_search"])  # a person's change on top: tuned on the folds as before
        self.assertTrue(final["evidence"]["optimization_search_space"])

    def test_without_the_setting_nothing_changes(self):
        store, pid = run(None)
        self.assertNotIn("search", store.read_stage(pid, "models")["evidence"])
        self.assertEqual([r.answers[0].answerer for r in self.book.records(point="model.family")][-1:], ["rule:rank_rows"])

    def test_the_budget_is_read_and_bounded(self):
        from dclab_rnd.studio import lineage, search

        self.assertIsNone(search.settings({"settings": {}}))
        self.assertEqual(search.settings({"settings": {"search": True}}), search.DEFAULT)
        self.assertEqual(search.settings({"settings": {"search": {"minutes": 9999, "trials": 0, "families": 9}}}),
                         {"minutes": 120.0, "trials": 1, "families": 5})
        self.assertIsNone(search.settings({"settings": {"search": {"minutes": "soon"}}}))
        self.assertIn("search", lineage.RUN_SETTINGS)  # a searched run is not the same run as an unsearched one


@unittest.skipIf(MISSING, MISSING)
class SearchCutTest(unittest.TestCase):
    """The selection fold, cut fairly (the review's F2 and F3), on plain arrays."""

    def test_a_time_split_with_entities_keeps_the_selection_fold_free_of_the_search_rows_entities(self):
        from dclab_rnd.studio import search

        rng = np.random.default_rng(0)
        entities = np.array([f"c{k}" for k in rng.integers(0, 120, 1000)])  # many entities recur over time
        entities[800:] = [f"new{k}" for k in rng.integers(0, 60, 200)]  # the latest rows: mostly new entities
        entities[800:820] = entities[:20]  # a few returning ones
        y = rng.integers(0, 2, 1000)
        found = search.cut(1000, y=y, entities=entities, split="time_group")
        self.assertNotIsInstance(found, str)
        rows, select, how = found
        self.assertEqual(how, "time")
        self.assertFalse(set(entities[select]) & set(entities[rows]))
        self.assertLess(len(select), 200)  # the returning entities' rows were dropped
        self.assertTrue((select >= 800).all())  # still the latest rows

    def test_a_cut_missing_a_class_is_drawn_again_or_the_search_is_skipped(self):
        from dclab_rnd.studio import search

        y = np.zeros(400, dtype=int)
        y[:6] = 1  # six positives, all at the start: the latest 20% hold none
        reason = search.cut(400, y=y, entities=None, split="time")
        self.assertIsInstance(reason, str)
        self.assertIn("miss a class", reason)
        entities = np.repeat(np.arange(40), 10).astype(str)
        y = np.zeros(400, dtype=int)
        y[np.isin(entities, ["3", "17"])] = 1  # two positive entities: some seeds put both on one side
        rows, select, _ = search.cut(400, y=y, entities=entities, split="group")
        self.assertEqual(set(y[rows]), {0, 1})
        self.assertEqual(set(y[select]), {0, 1})
        lone = np.zeros(400, dtype=int)
        lone[entities == "5"] = 1  # one positive entity can never be on both sides
        self.assertIn("every class", search.cut(400, y=lone, entities=entities, split="group"))

    def test_a_large_group_never_leaves_one_group_to_search_and_folds_keep_entities_apart(self):
        from dclab_rnd.studio import search

        sizes = [25, 900, 25, 25, 25]
        entities = np.concatenate([np.full(k, f"g{i}") for i, k in enumerate(sizes)])
        found = search.cut(len(entities), y=None, entities=entities, split="group")
        if not isinstance(found, str):  # a fair cut, or the reason none exists: never one group to search or most rows set aside
            rows, select, _ = found
            self.assertGreaterEqual(len(set(entities[rows])), 2)
            self.assertLessEqual(len(select), 0.5 * len(entities))

        class Bundle:
            task_type = "timeseries_regression"

            class y_train:
                @staticmethod
                def to_numpy():
                    return np.zeros(len(entities))

        self.assertEqual(search._folds(Bundle, np.arange(25), "group", 3, entities), [])  # one group: no folds, so the search skips
        rng = np.random.default_rng(1)
        timed = np.array([f"c{k}" for k in rng.integers(0, 80, 1000)])
        Bundle.y_train.to_numpy = staticmethod(lambda: np.zeros(1000))
        for fit, valid in search._folds(Bundle, np.arange(1000), "time_group", 3, timed):
            self.assertFalse(set(timed[fit]) & set(timed[valid]))
            self.assertGreaterEqual(len(valid), 10)

    def test_the_reviews_class_weight_is_never_drawn_twice(self):
        from dclab_rnd.studio import search

        class Trial:
            def suggest_float(self, name, low, high, log=False):
                return low

            def suggest_categorical(self, name, choices):
                return choices[-1]  # "balanced" whenever offered

        self.assertEqual(search.space(Trial(), "logistic_regression", "binary")["model__class_weight"], "balanced")
        self.assertNotIn("model__class_weight", search.space(Trial(), "logistic_regression", "binary", weighted=True))

    def test_the_search_starts_from_the_reviews_change_and_says_when_it_is_skipped(self):
        from dclab_rnd.studio import search

        seen = []

        def fake_fit(bundle, recipe, family, params, X_fit, y_fit, X_valid, seed=0):
            seen.append((family, dict(params or {})))
            return {"predictions": np.linspace(0.1, 0.9, len(X_valid))}

        from dclab_rnd.expansion import runner
        from dclab_rnd.models.gateway import private_scope
        from dclab_rnd.studio import engine

        store, pid = acceptance.build(kinds.get("binary").case(0), Path(tempfile.mkdtemp()))
        with private_scope(True):
            p = engine.prepare(store, store.get(pid))
        screen = [{"model": "logistic_regression"}, {"model": "hist_gradient_boosting"}]
        base = {"family": "logistic_regression", "params": {"review__class_weight": "balanced"}}
        with mock.patch.object(runner, "fit_and_predict", fake_fit):
            found = search.run(p, "base", screen, {"minutes": 1, "trials": 4, "families": 2}, base=base)
        self.assertEqual(found["without_search"]["name"], "logistic_regression:reviewed")
        logistic = [params for family, params in seen if family == "logistic_regression"]
        self.assertTrue(logistic)
        self.assertTrue(all(params.get("review__class_weight") == "balanced" for params in logistic))  # the base of every trial
        self.assertFalse(any("model__class_weight" in params for params in logistic))  # never weighted twice
        self.assertFalse(any("review__class_weight" in params for family, params in seen if family != "logistic_regression"))
        with mock.patch.object(search, "cut", return_value="the latest rows set aside for the selection fold miss a class"):
            skipped = search.run(p, "base", screen, {"minutes": 1, "trials": 4, "families": 2}, base=base)
        self.assertIn("miss a class", skipped["skipped"])
        self.assertNotIn("family", skipped)


if __name__ == "__main__":
    unittest.main()
