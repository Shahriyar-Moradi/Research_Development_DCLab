"""Synthetic tables from a description (dclab_rnd/draft/synthetic.py).

The model only proposes a schema; deterministic code validates it and draws seeded rows that are always
labelled synthetic. These tests pin the validators, determinism, the target calibration and direction,
missing values, time order, the templates and the model round trip (with a fake client).
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from pydantic import ValidationError

    from dclab_rnd.draft import synthetic as syn
except ImportError as error:  # pydantic is not part of requirements/ci.txt
    MISSING = str(error)
else:
    MISSING = ""


def spec_dict(**overrides):
    base = {
        "name": "test table",
        "rows": 2000,
        "seed": 11,
        "columns": [
            {"name": "row_id", "type": "id"},
            {"name": "x", "type": "numeric", "dist": "normal", "params": {"mean": 0, "sd": 1}},
            {"name": "z", "type": "numeric", "dist": "normal", "params": {"mean": 5, "sd": 2}},
            {"name": "count", "type": "integer", "dist": "poisson", "params": {"lam": 3}},
            {"name": "plan", "type": "categorical", "categories": ["basic", "plus", "pro"], "weights": [2, 1, 1]},
            {"name": "flag", "type": "boolean", "p_true": 0.3},
            {"name": "when", "type": "datetime", "start": "2024-01-01", "end": "2024-12-31"},
            {"name": "note", "type": "text"},
        ],
        "target": {"name": "label", "type": "binary", "positive_rate": 0.3, "effects": {"x": 1.5, "plan=pro": 0.5}},
    }
    base.update(overrides)
    return base


def with_column(index, **changes):
    data = spec_dict()
    data["columns"][index] = {**data["columns"][index], **changes}
    return data


def with_target(**changes):
    data = spec_dict()
    data["target"] = {**data["target"], **changes}
    return data


class FakeClient:
    """Replies from a queue: a string is the content, an exception is raised."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, tools=None, max_tokens=1800):
        self.calls.append([dict(m) for m in messages])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"content": reply, "tool_calls": []}


MODEL_SPEC = {
    "name": "Hospital readmission (synthetic)",
    "rows": 999999,
    "columns": [
        {"name": "patient_ref", "type": "id"},
        {"name": "age", "type": "integer", "dist": "normal", "params": {"mean": 62, "sd": 14, "clip_min": 18, "clip_max": 100}},
        {"name": "ward", "type": "categorical", "categories": ["cardiology", "surgery", "general"]},
        {"name": "length_of_stay_days", "type": "integer", "dist": "poisson", "params": {"lam": 4}},
    ],
    "target": {"name": "readmitted_30d", "type": "binary", "positive_rate": 0.12, "effects": {"age": 0.6, "ward=cardiology": 0.4}},
}


@unittest.skipIf(MISSING, f"pydantic not installed: {MISSING}")
class ValidationTests(unittest.TestCase):
    def assert_rejected(self, data, *fragments):
        with self.assertRaises(ValidationError) as caught:
            syn.SyntheticSpec.model_validate(data)
        text = " ".join(syn.problems(caught.exception))
        self.assertNotIn("Value error,", text)
        for fragment in fragments:
            self.assertIn(fragment, text)

    def test_reference_spec_is_valid(self):
        spec = syn.SyntheticSpec.model_validate(spec_dict())
        self.assertEqual(spec.columns[4].weights, [0.5, 0.25, 0.25])  # normalised
        self.assertEqual(spec.target.classes, ["no", "yes"])  # binary default

    def test_unknown_effect_column(self):
        self.assert_rejected(with_target(effects={"income": 1.0}), "names no column")

    def test_effects_on_id_text_datetime_columns(self):
        for key in ("row_id", "note", "when"):
            with self.subTest(key=key):
                self.assert_rejected(with_target(effects={key: 1.0}), "effects can use numeric")

    def test_categorical_effect_needs_a_known_value(self):
        self.assert_rejected(with_target(effects={"plan": 1.0}), "plan=basic")
        self.assert_rejected(with_target(effects={"plan=gold": 1.0}), "not one of the categories")

    def test_target_name_clashes_with_column(self):
        self.assert_rejected(with_target(name="X"), "same name as a column")

    def test_duplicate_column_names(self):
        data = spec_dict()
        data["columns"].append({"name": "x", "type": "boolean"})
        self.assert_rejected(data, "column names must be unique")

    def test_distribution_params(self):
        self.assert_rejected(with_column(1, params={"mean": 0, "sd": 0}), "sd must be greater than 0")
        self.assert_rejected(with_column(1, params={"mean": 0, "sd": -2}), "sd must be greater than 0")
        self.assert_rejected(with_column(1, dist="uniform", params={"low": 5, "high": 5}), "low must be below high")
        self.assert_rejected(with_column(1, params={"mean": 0}), "missing sd")
        self.assert_rejected(with_column(1, params={"mean": 0, "std": 1}), "missing sd")
        self.assert_rejected(with_column(1, params={"mean": 0, "sd": 1, "skew": 2}), "unknown params skew")
        self.assert_rejected(with_column(3, params={"lam": 0}), "lam must be above 0")
        self.assert_rejected(with_column(1, dist="exponential", params={"scale": -1}), "scale must be greater than 0")
        self.assert_rejected(with_column(1, dist=None, params={}), "set dist to normal")
        self.assert_rejected(with_column(1, params={"mean": 0, "sd": 1, "clip_min": 3, "clip_max": 1}), "clip_min must be below")

    def test_dist_inferred_from_unambiguous_params(self):
        spec = syn.SyntheticSpec.model_validate(with_column(1, dist=None, params={"low": 0, "high": 10}))
        self.assertEqual(spec.columns[1].dist, "uniform")

    def test_categories_and_weights(self):
        self.assert_rejected(with_column(4, weights=[1, 1]), "2 weights for 3 categories")
        self.assert_rejected(with_column(4, categories=["only"], weights=[]), "2 to 50 categories")
        self.assert_rejected(with_column(4, categories=["a", "a", "b"], weights=[]), "same category twice")
        self.assert_rejected(with_column(1, categories=["a", "b"]), "apply only to categorical")

    def test_datetime_range(self):
        self.assert_rejected(with_column(6, start="2024-12-31", end="2024-01-01"), "start must be before end")
        self.assert_rejected(with_column(6, start="2024-05-01", end="2024-05-01"), "start must be before end")
        self.assert_rejected(with_column(6, start="yesterday-ish"), "not an ISO date")

    def test_other_limits(self):
        self.assert_rejected(spec_dict(rows=50), "rows")
        self.assert_rejected(spec_dict(rows=500_000), "rows")
        self.assert_rejected(with_column(1, null_rate=0.7), "null_rate")
        self.assert_rejected(with_column(0, null_rate=0.1), "ids are never missing")
        self.assert_rejected(with_target(positive_rate=0.001), "positive_rate must be between")
        self.assert_rejected(with_target(type="multiclass", positive_rate=None, classes=["a", "b"]), "3 to 20 classes")
        self.assert_rejected(with_target(type="regression", positive_rate=0.2), "applies only to a binary target")
        self.assert_rejected(with_target(noise=0), "noise must be greater than 0")
        self.assert_rejected(spec_dict(time_order=True, columns=[*spec_dict()["columns"][:6],
                                                                 {"name": "when", "type": "datetime", "null_rate": 0.1}]),
                             "cannot have missing values")
        self.assert_rejected(spec_dict(surprise=True), "surprise")


@unittest.skipIf(MISSING, f"pydantic not installed: {MISSING}")
class GenerationTests(unittest.TestCase):
    def test_deterministic_for_a_seed(self):
        spec = syn.SyntheticSpec.model_validate(spec_dict())
        first, again = syn.generate(spec), syn.generate(spec)
        pd.testing.assert_frame_equal(first, again)
        other = syn.generate(spec.model_copy(update={"seed": 12}))
        self.assertFalse(first.drop(columns=["row_id"]).equals(other.drop(columns=["row_id"])))

    def test_columns_and_types(self):
        spec = syn.SyntheticSpec.model_validate(spec_dict())
        frame = syn.generate(spec)
        self.assertEqual(list(frame.columns), [c.name for c in spec.columns] + ["label"])
        self.assertEqual(len(frame), 2000)
        self.assertEqual(str(frame["count"].dtype), "Int64")
        self.assertEqual(str(frame["flag"].dtype), "boolean")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["when"]))
        self.assertTrue(frame["row_id"].str.fullmatch(r"R\d{6}").all())
        self.assertTrue(frame["row_id"].is_unique)
        self.assertTrue(set(frame["plan"]) <= {"basic", "plus", "pro"})
        self.assertTrue(frame["note"].str.endswith(".").all())
        self.assertTrue(frame["when"].between(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")).all())
        self.assertAlmostEqual((frame["plan"] == "basic").mean(), 0.5, delta=0.05)

    def test_positive_rate_calibration(self):
        _, fraud = syn.template_for("card fraud")
        frame = syn.generate(fraud)
        self.assertAlmostEqual((frame["is_fraud"] == "yes").mean(), 0.005, delta=0.01)
        self.assertGreater((frame["is_fraud"] == "yes").sum(), 0)
        spec = syn.SyntheticSpec.model_validate(spec_dict(rows=20_000))
        rate = (syn.generate(spec)["label"] == "yes").mean()
        self.assertAlmostEqual(rate, 0.3, delta=0.01)
        small = syn.SyntheticSpec.model_validate(spec_dict(rows=100))
        self.assertAlmostEqual((syn.generate(small)["label"] == "yes").mean(), 0.3, delta=0.01)

    def test_effects_direction(self):
        frame = syn.generate(syn.SyntheticSpec.model_validate(spec_dict(rows=20_000)))
        high = frame["x"] > frame["x"].median()
        positive = frame["label"] == "yes"
        self.assertGreater(positive[high].mean(), positive[~high].mean() + 0.15)  # weight 1.5 on x
        self.assertGreater(positive[frame["plan"] == "pro"].mean(), positive[frame["plan"] == "basic"].mean())
        z_high = frame["z"] > frame["z"].median()  # z carries no effect
        self.assertAlmostEqual(positive[z_high].mean(), positive[~z_high].mean(), delta=0.03)

        regression = spec_dict(target={"name": "y", "type": "regression", "base": 100, "noise": 5,
                                       "effects": {"x": 10, "flag": -20, "plan=pro": 30}})
        frame = syn.generate(syn.SyntheticSpec.model_validate(regression))
        self.assertGreater(np.corrcoef(frame["x"], frame["y"])[0, 1], 0.5)
        flag = frame["flag"].astype(bool)
        self.assertAlmostEqual(frame.loc[flag, "y"].mean() - frame.loc[~flag, "y"].mean(), -20, delta=3)
        self.assertGreater(frame.loc[frame["plan"] == "pro", "y"].mean(), frame.loc[frame["plan"] == "basic", "y"].mean() + 20)

        multiclass = spec_dict(target={"name": "tier", "type": "multiclass", "classes": ["low", "mid", "high"], "effects": {"x": 2.0}})
        frame = syn.generate(syn.SyntheticSpec.model_validate(multiclass))
        index = frame["tier"].map({"low": 0, "mid": 1, "high": 2})
        high = frame["x"] > frame["x"].median()
        self.assertGreater(index[high].mean(), index[~high].mean() + 0.3)
        self.assertEqual(set(frame["tier"]), {"low", "mid", "high"})

    def test_null_rate_respected_and_target_uses_clean_values(self):
        data = spec_dict(rows=5000, target={"name": "y", "type": "regression", "noise": 0, "effects": {"x": 2.0}})
        for index, rate in ((1, 0.2), (3, 0.1), (4, 0.05), (5, 0.3), (6, 0.15), (7, 0.25)):
            data["columns"][index]["null_rate"] = rate
        spec = syn.SyntheticSpec.model_validate(data)
        frame = syn.generate(spec)
        for column in spec.columns:
            self.assertAlmostEqual(frame[column.name].isna().mean(), column.null_rate, delta=0.02, msg=column.name)
        self.assertEqual(frame["y"].isna().sum(), 0)
        # with noise 0 the target is exactly linear in the clean x, even where x is now missing
        present = frame["x"].notna()
        self.assertGreater(np.corrcoef(frame.loc[present, "x"], frame.loc[present, "y"])[0, 1], 0.9999)
        self.assertGreater(frame.loc[~present, "y"].std(), 0)

    def test_time_order(self):
        ordered = syn.generate(syn.SyntheticSpec.model_validate(spec_dict(time_order=True)))
        self.assertTrue(ordered["when"].is_monotonic_increasing)
        self.assertEqual(ordered["row_id"].tolist(), sorted(ordered["row_id"]))  # ids increase with time
        unordered = syn.generate(syn.SyntheticSpec.model_validate(spec_dict()))
        self.assertFalse(unordered["when"].is_monotonic_increasing)

    def test_save_writes_labelled_files(self):
        import pyarrow.parquet as pq

        spec = syn.SyntheticSpec.model_validate(spec_dict(rows=300))
        frame = syn.generate(spec)
        with tempfile.TemporaryDirectory() as tmp:
            out = syn.save(frame, spec, Path(tmp) / "draft")
            self.assertEqual((out["rows"], out["columns"]), (300, 9))
            back = pd.read_parquet(out["path"])
            self.assertEqual(back.shape, frame.shape)
            metadata = pq.read_schema(out["path"]).metadata
            self.assertTrue(json.loads(metadata[b"dclab.synthetic"])["synthetic"])
            stored = json.loads(Path(out["spec_path"]).read_text())
            self.assertTrue(stored["labels"]["synthetic"])
            self.assertEqual(syn.SyntheticSpec.model_validate(stored["spec"]), spec)


@unittest.skipIf(MISSING, f"pydantic not installed: {MISSING}")
class TemplateTests(unittest.TestCase):
    def test_every_template_generates_and_describes(self):
        self.assertTrue({"churn", "fraud", "demand", "credit", "maintenance"} <= set(syn.TEMPLATES))
        for key, spec in syn.TEMPLATES.items():
            with self.subTest(template=key):
                frame = syn.generate(spec)
                summary = syn.describe(spec, frame)
                json.dumps(summary)
                self.assertIs(summary["labels"]["synthetic"], True)
                self.assertIn("not to report real performance", summary["labels"]["warning"])
                self.assertEqual(summary["rows"], spec.rows)
                self.assertIsNotNone(summary["target"])
                if spec.target.type == "binary":
                    self.assertAlmostEqual(summary["target"]["positive_rate"], spec.target.positive_rate, delta=0.01)
        demand = syn.TEMPLATES["demand"]
        self.assertTrue(demand.time_order)
        self.assertEqual(demand.target.type, "regression")
        self.assertTrue(syn.generate(demand)["date"].is_monotonic_increasing)

    def test_template_for_routes_by_keyword(self):
        cases = {
            "Which subscribers will cancel next month?": "churn",
            "Detect fraudulent card transactions": "fraud",
            "Forecast daily sales for each store": "demand",
            "Predict loan default for new applicants": "credit",
            "Predict machine failure from vibration sensors": "maintenance",
            "Something completely different": "churn",
            "": "churn",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                key, spec = syn.template_for(prompt)
                self.assertEqual(key, expected)
                self.assertIsInstance(spec, syn.SyntheticSpec)

    def test_template_copies_are_independent(self):
        _, spec = syn.template_for("churn")
        spec.columns[1].name = "changed"
        self.assertNotEqual(syn.TEMPLATES["churn"].columns[1].name, "changed")


@unittest.skipIf(MISSING, f"pydantic not installed: {MISSING}")
class SpecFromModelTests(unittest.TestCase):
    def test_no_client_uses_template(self):
        spec, info = syn.spec_from_model(None, "predict loan default", rows=1234)
        self.assertEqual(info, {"source": "template", "template": "credit"})
        self.assertEqual(spec.rows, 1234)

    def test_valid_json_in_fence(self):
        client = FakeClient("Here is the spec:\n```json\n" + json.dumps(MODEL_SPEC) + "\n```\nHope it helps.")
        spec, info = syn.spec_from_model(client, "predict readmission", context={"problem": "readmission in 30 days"}, rows=3000)
        self.assertEqual(info["source"], "model")
        self.assertEqual(info["attempts"], 1)
        self.assertEqual(spec.rows, 3000)  # the argument wins over the model's 999999
        self.assertEqual(spec.target.name, "readmitted_30d")
        self.assertEqual(len(client.calls), 1)
        system, user = client.calls[0]
        self.assertEqual(system["role"], "system")
        self.assertIn("readmission in 30 days", user["content"])

    def test_invalid_then_valid_on_retry(self):
        bad = json.loads(json.dumps(MODEL_SPEC))
        bad["columns"][1]["params"]["sd"] = -3
        client = FakeClient(json.dumps(bad), json.dumps(MODEL_SPEC))
        _, info = syn.spec_from_model(client, "predict readmission")
        self.assertEqual(info, {"source": "model", "attempts": 2})
        retry = client.calls[1]
        self.assertEqual(retry[-2]["role"], "assistant")
        self.assertIn("sd must be greater than 0", retry[-1]["content"])

    def test_not_json_then_valid(self):
        client = FakeClient("I think a table with age and ward would work.", "{\"spec\": " + json.dumps(MODEL_SPEC) + "}")
        _, info = syn.spec_from_model(client, "predict readmission")
        self.assertEqual(info["source"], "model")
        self.assertIn("no JSON object", client.calls[1][-1]["content"])

    def test_twice_invalid_falls_back_to_template(self):
        client = FakeClient("not json", '{"name": "x", "columns": []}')
        spec, info = syn.spec_from_model(client, "detect card fraud", rows=50)
        self.assertEqual(info["source"], "template")
        self.assertEqual(info["template"], "fraud")
        self.assertIn("columns", info["fallback_reason"])
        self.assertEqual(spec.rows, 100)  # clamped
        self.assertEqual(len(client.calls), 2)

    def test_raising_client_falls_back(self):
        client = FakeClient(RuntimeError("APIConnectionError: the model request failed"), RuntimeError("again"))
        spec, info = syn.spec_from_model(client, "forecast demand", rows=10**7)
        self.assertEqual(info["source"], "template")
        self.assertEqual(info["template"], "demand")
        self.assertIn("RuntimeError", info["fallback_reason"])
        self.assertEqual(spec.rows, 200_000)  # clamped
        self.assertEqual(len(client.calls), 2)

    def test_too_many_columns_is_rejected(self):
        wide = json.loads(json.dumps(MODEL_SPEC))
        wide["columns"] += [{"name": f"c{i}", "type": "boolean"} for i in range(30)]
        client = FakeClient(json.dumps(wide), json.dumps(wide))
        _, info = syn.spec_from_model(client, "predict readmission")
        self.assertEqual(info["source"], "template")
        self.assertIn("at most 30", info["fallback_reason"])

    def test_no_data_rows_reach_the_model(self):
        frame = pd.DataFrame({"secret_value": ["ROW-SECRET-1", "ROW-SECRET-2"]})
        context = {"problem": "predict readmission", "answers": {"moment": "at discharge"}, "rows": [{"secret_value": "ROW-SECRET-3"}],
                   "preview": "ROW-SECRET-4", "frame": frame, "table_records": [{"secret_value": "ROW-SECRET-5"}]}
        client = FakeClient(json.dumps(MODEL_SPEC))
        syn.spec_from_model(client, "predict readmission", context=context)
        sent = json.dumps(client.calls)
        self.assertNotIn("ROW-SECRET", sent)
        self.assertIn("at discharge", sent)


if __name__ == "__main__":
    unittest.main()
