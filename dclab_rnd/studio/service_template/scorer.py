"""The project's scorer: the fitted feature builder and model, exactly as DCLab's final stage fitted them.

It needs no data: ``model.pkl`` holds the fitted encoder levels, the behaviour and program settings, and the model. The
modules the fitted objects come from are vendored in ``vendor/`` (DCLab's own code, as it was when exported).
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))


# pandas' default missing markers but "NA": in the columns DCLab trained with "NA" as a code (Namibia's), it stays a code
READ_NA = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NULL", "NaN",
           "None", "n/a", "nan", "null"]


def read_csv(data: bytes, meta: dict[str, Any], key: str | None = None) -> pd.DataFrame:
    """A CSV of rows to score, read as DCLab read its training table: "NA" a code in the columns ``model.json`` names
    (``na_codes``: decided on the training table, never on a batch, which may hold one row), missing everywhere else."""
    import io

    frame = pd.read_csv(io.BytesIO(data), dtype={key: str} if key else None, keep_default_na=False, na_values=READ_NA)
    codes = set(meta.get("na_codes") or [])
    for column in frame.columns:
        series = frame[column]
        if column in codes or column == key or not (series.dtype == object or isinstance(series.dtype, pd.StringDtype)):
            continue
        hits = series.eq("NA").fillna(False).astype(bool)
        if not hits.any():
            continue
        series = series.astype(object).mask(hits, np.nan)
        numbers = pd.to_numeric(series, errors="coerce")
        frame[column] = numbers if int(numbers.notna().sum()) == int(series.notna().sum()) else series
    return frame


class Scorer:
    def __init__(self, path: Path = HERE / "model.pkl", key: str | None = None):
        with open(path, "rb") as handle:
            self.artifact = pickle.load(handle)  # noqa: S301 - the project's own artifact, written by DCLab's export
        self.meta = json.loads((HERE / "model.json").read_text(encoding="utf-8"))
        self.key = key or os.environ.get("DCLAB_SERVICE_KEY")
        if not self.key:
            raise RuntimeError("Set DCLAB_SERVICE_KEY: the key DCLab gave with this export (it hashes category values)")
        entity = getattr(self.artifact["recipe"], "entity_model", None)
        if entity is not None:
            entity.salt = self.key  # its keys were stored hashed with this key

    def frame(self, rows: Any) -> pd.DataFrame:
        """Rows (a dict, a list of dicts or a frame) as the table the model was trained on: every input column, booleans
        as 0/1, the time column in the same text form; a missing column is missing in every row."""
        frame = pd.DataFrame([rows] if isinstance(rows, dict) else rows).copy()
        for column in self.meta["input_columns"]:
            if column not in frame:
                frame[column] = np.nan
        frame = frame[self.meta["input_columns"]]
        for column, kind in self.meta.get("input_types", {}).items():  # the training types, whatever the batch holds
            values = pd.to_numeric(frame[column], errors="coerce") if kind.startswith(("int", "Int", "float")) else None
            if values is not None and kind.startswith("Int"):
                frame[column] = values.round().astype("Int64")  # a nullable integer stays one
            elif values is not None and kind.startswith("int") and values.notna().all():
                frame[column] = values.round().astype("int64")  # the training type exactly (an object column built other features: 30.15's check)
            elif values is not None:
                frame[column] = values.astype(float)  # a missing value in an integer column: a float, as pandas reads one
        for column in frame.columns:
            if pd.api.types.is_bool_dtype(frame[column]) or str(frame[column].dtype) == "boolean":
                frame[column] = pd.to_numeric(frame[column].astype("object").map({True: 1.0, False: 0.0}), errors="coerce")
        time = self.meta.get("time_column")
        if time and self.meta.get("time_is_datetime"):
            frame[time] = pd.to_datetime(frame[time], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        # category levels are kept as salted hashes in the model: the input's values are hashed the same way
        import hashlib

        from dclab_rnd.expansion.runner import _as_text

        salt = self.key
        for column in self.artifact.get("hashed_columns", []):
            frame[column] = _as_text(frame[column]).map(lambda v: "h:" + hashlib.sha256(f"{salt}|{v}".encode()).hexdigest()[:16])
        return frame

    def predict(self, rows: Any) -> list[dict[str, Any]]:
        from dclab_rnd.expansion.runner import predict_scores

        frame = self.frame(rows)
        recipe, model = self.artifact["recipe"], self.artifact["model"]
        matrix, _ = recipe.transform(frame)
        scores = predict_scores(model, matrix, self.meta["task_type"], self.meta.get("n_classes"))
        change = self.meta.get("review_change") or {}
        if change.get("review__target") == "log1p":  # a review round kept a log target: back to the target's scale
            scores = np.expm1(scores)
        if change.get("review__clip") == "fit_range" and self.meta.get("target_range"):
            scores = np.clip(scores, *self.meta["target_range"])
        task, out = self.meta["task"], []
        for i in range(len(frame)):
            if task == "binary":
                score = float(scores[i])
                threshold = self.meta.get("threshold")
                out.append({"score": score, **({"alert": bool(score >= threshold)} if threshold is not None else {}),
                            **({"calibrated_probability": float(self._calibrated([score])[0])} if self.meta.get("calibrator") else {})})
            elif task == "multiclass":
                labels = self.meta["class_labels"]
                best = int(np.argmax(scores[i]))
                out.append({"class": labels[best], "probabilities": {labels[j]: float(scores[i][j]) for j in range(len(labels))}})
            else:
                out.append({"prediction": float(scores[i])})
        if self.meta.get("behaviour_within_batch") and len(frame) == 1:
            for row in out:
                row["note"] = "behaviour features need the rows scored with this one: a single row has no history"
        return out

    def _calibrated(self, scores: Any) -> Any:
        """The score as a probability, through the calibrator fitted on the training rows' out-of-fold scores."""
        from dclab_core.calibration import Calibrator, apply

        return apply(Calibrator.read(self.meta["calibrator"]), scores)

    def monitor(self, rows: Any, scores: Any = None) -> dict[str, Any]:
        """A batch read against the training rows' bins (``monitoring.json``): each column's and the score's PSI, the share
        the rule flags, and what crossed an alert line. ``scores``: the batch's scores when they were already computed."""
        from dclab_core.monitoring import MonitoringPlan, monitor

        if not self.meta.get("monitoring"):
            raise RuntimeError("This export carries no monitoring plan")
        raw = json.loads((HERE / self.meta["monitoring"]).read_text(encoding="utf-8"))
        hashed = bool(raw.pop("hashed_levels", False))
        plan = MonitoringPlan.read(raw)
        frame = pd.DataFrame([rows] if isinstance(rows, dict) else rows)
        if hashed:  # the plan keeps its levels hashed with the key: the batch's values are hashed the same way
            import hashlib

            for column in plan.columns:
                if column.kind == "category" and column.column in frame:
                    frame[column.column] = frame[column.column].map(
                        lambda v: v if pd.isna(v) else "h:" + hashlib.sha256(f"{self.key}|{v}".encode()).hexdigest()[:16])
        if scores is None and self.meta["task"] == "binary":
            scores = [r["score"] for r in self.predict(rows)]
        return monitor(plan, frame, scores).model_dump(mode="json")
