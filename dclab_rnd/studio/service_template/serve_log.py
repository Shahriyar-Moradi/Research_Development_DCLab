"""What the service keeps (34.9): every row it scores in ``predictions.jsonl`` and every outcome posted in ``outcomes.jsonl``,
beside the model, or in ``DCLAB_SERVICE_LOG_DIR`` when it is set (mount a volume there: a container's own disk is lost
with it). Keys are hashed with ``DCLAB_LOG_KEY`` (or, without it, the export's ``DCLAB_SERVICE_KEY``): set it once and
keep it across exports, so an outcome posted after a retrain still joins the prediction the model before it made. Each line is a ``dclab_core.serving`` record: the model and its version, the row's key as a salted hash, the
score and the decision, the time. No value of a scored row is ever written, nor the caller's key itself.

A row's key: name a column of the rows that identifies them (``key``: the order id), and post the outcome later with the
same value; with no key column, a key is made for each row and returned with its prediction. The two logs join by key
(``dclab_core.serving.join``).
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dclab_core import serving

HERE = Path(__file__).resolve().parent
PREDICTIONS, OUTCOMES = "predictions.jsonl", "outcomes.jsonl"
PROFILES = "profiles.jsonl"  # 34.10: scored rows' inputs as the plan's bins see them (shares of 20 rows or more, never a value)
ALERTS = "alerts.jsonl"  # 34.10: each alert /monitor raised, once, with its window and when it was first seen
MAX_OUTCOMES = 100_000  # outcomes in one request


class Refused(ValueError):
    """A request the service answers with 400 and the reason; nothing of it is logged."""


def log_dir() -> Path:
    return Path(os.environ.get("DCLAB_SERVICE_LOG_DIR") or HERE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _append(name: str, records: list[Any]) -> None:
    """The records appended as one write under an exclusive lock: two workers never interleave their lines."""
    if not records:
        return
    folder = log_dir()
    folder.mkdir(parents=True, exist_ok=True)
    text = "".join(record.model_dump_json() + "\n" for record in records)
    with open(folder / name, "a+b") as handle:
        try:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        except (ImportError, OSError):  # no lock on this system: one write call still keeps a line whole
            pass
        handle.seek(0, os.SEEK_END)
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":  # a writer killed mid-line: its torn line ends here, ours starts on the next
                text = "\n" + text
        handle.write(text.encode("utf-8"))
        handle.flush()


class Log:
    def __init__(self, scorer: Any):
        self.scorer, self.meta = scorer, scorer.meta
        self.model_id = str(self.meta.get("model_id") or "model")
        self.model_version = self.meta.get("model_version") or hashlib.sha256((HERE / "model.pkl").read_bytes()).hexdigest()[:16]
        self.secret = os.environ.get("DCLAB_LOG_KEY") or scorer.key  # outlives an export, so keys join across retrains
        self.profiler = None  # 34.10: bin counts gathered in this process until a profile covers enough rows

    def _rows(self, rows: Any) -> pd.DataFrame:
        if isinstance(rows, pd.DataFrame):
            return rows
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise Refused("send one row as an object or a list of objects")
        return pd.DataFrame(rows)

    def _keys(self, frame: pd.DataFrame, column: str | None) -> list[str]:
        if not column:
            return [serving.new_key() for _ in range(len(frame))]
        if not isinstance(column, str):
            raise Refused("key names one column of the rows")
        if column not in frame:
            raise Refused(f"the rows have no column {column!r} to key them by")
        values = frame[column]
        if values.isna().any() or (values.astype(str).str.strip() == "").any():
            raise Refused(f"every row needs its key: {column!r} is missing in some rows")
        return [serving._text(v) for v in values]

    def predict(self, rows: Any, key_column: str | None = None) -> dict[str, Any]:
        """Score the rows, log one record per row, and answer the predictions with each row's key and the batch."""
        frame = self._rows(rows)
        if frame.empty:
            raise Refused("send at least one row")
        keys = self._keys(frame, key_column)
        predictions = self.scorer.predict(frame)
        batch, at, task = serving.new_key(), _now(), self.meta["task"]
        labels = [serving._text(label) for label in self.meta.get("class_labels") or []]
        threshold = self.meta.get("threshold") if task == "binary" else None
        records = []
        for position, (key, out) in enumerate(zip(keys, predictions)):
            if task == "binary":
                score, chosen = out["score"], None
                decision = None if "alert" not in out else ("alert" if out["alert"] else "no_alert")
            elif task == "multiclass":
                chosen = labels.index(serving._text(out["class"]))
                score, decision = out["probabilities"][out["class"]], None
            else:
                score, chosen, decision = out["prediction"], None, None
            records.append(serving.PredictionRecord(
                model_id=self.model_id, model_version=self.model_version, batch=batch, position=position,
                key=serving.key_hash(key, self.secret), task="regression" if task not in ("binary", "multiclass") else task,
                score=score, predicted_class=chosen, decision=decision, threshold=threshold if decision else None, at=at))
        _append(PREDICTIONS, records)
        self._profile(frame, batch, at)
        return {"batch": batch, "predictions": [out | {"key": key} for key, out in zip(keys, predictions)]}

    def _profile(self, frame: pd.DataFrame, batch: str, at: str) -> None:
        """34.10: the scored rows' bin counts gathered in this process; a profile is written once it covers enough rows, with
        no link to a request (one row's profile would be its bins). Monitoring never fails a scored request."""
        from dclab_core import live

        try:
            if not self.meta.get("monitoring"):
                return
            plan, seen = self.scorer.monitoring_frame(frame)
            if plan is None or not plan.columns:
                return
            if self.profiler is None:
                self.profiler = live.Profiler(plan, self.model_id, self.model_version)
            made = self.profiler.add(seen, at)
            if made is not None:
                _append(PROFILES, [made])
        except Exception:  # noqa: BLE001 - the prediction is logged and answered; its profile is what is lost
            self.profiler = None

    def monitor(self, window: str = "day") -> dict[str, Any]:
        """34.10: the model in use read from its logs window by window: the score's and the inputs' drift, the share the
        rule flags, and live performance once outcomes are known, against the holdout (``dclab_core.live.monitor_log``)."""
        from dclab_core import live

        if window not in live.WINDOWS:
            raise Refused(f"window is one of {', '.join(live.WINDOWS)}")
        plan = self.scorer.monitoring_plan()
        if plan is None:
            raise Refused("this export carries no monitoring plan")
        folder = log_dir()

        def read(name: str, contract: Any) -> list[Any]:
            return serving.read_log(folder / name, contract) if (folder / name).exists() else []

        def mine(record: Any) -> bool:  # a shared log folder: only this model, this fit, against this fit's plan
            return record.model_id == self.model_id and record.model_version == self.model_version

        try:
            report = live.monitor_log(plan, [r for r in read(PREDICTIONS, serving.PredictionRecord) if mine(r)],
                                      read(OUTCOMES, serving.OutcomeRecord), profiles=[p for p in read(PROFILES, live.BatchProfile) if mine(p)],
                                      window=window, holdout=self.meta.get("live_reference"))
        except ValueError as error:  # a log line this reading cannot use: said, never a 500
            raise Refused(f"the logs cannot be read: {str(error)[:300]}") from None
        self._alerts(report)
        return report.model_dump(mode="json")

    def _alerts(self, report: Any) -> None:
        """Each alert kept once in ``alerts.jsonl``, the service's record of what monitoring found (34.10)."""
        import json

        path = log_dir() / ALERTS
        seen = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    seen.add((item.get("window"), item.get("alert")))
                except ValueError:
                    continue
        new = [{"model_id": self.model_id, "model_version": self.model_version, "window": w.window, "alert": a, "first_seen": _now()}
               for w in report.windows for a in w.alerts if (w.window, a) not in seen]
        if new:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("".join(json.dumps(item) + "\n" for item in new))

    def outcomes(self, body: Any) -> dict[str, Any]:
        """Record true outcomes: one ``{"key", "outcome", "known_at"?}`` or ``{"outcomes": [...]}``. All or nothing: one
        outcome that cannot be read refuses the request and nothing is recorded."""
        items = body.get("outcomes") if isinstance(body, dict) and "outcomes" in body else [body] if isinstance(body, dict) else body
        if not isinstance(items, list) or not items or not all(isinstance(i, dict) for i in items):
            raise Refused('send {"key": ..., "outcome": ..., "known_at": optional ISO time} or {"outcomes": [those]}')
        if len(items) > MAX_OUTCOMES:
            raise Refused(f"at most {MAX_OUTCOMES} outcomes in one request")
        task, now, records = self.meta["task"], _now(), []
        for number, item in enumerate(items):
            try:
                if "key" not in item or "outcome" not in item:
                    raise ValueError("each outcome needs its key and its outcome")
                coded = serving.code_outcome(item["outcome"], task if task in ("binary", "multiclass") else "regression",
                                             positive_label=self.meta.get("positive_label"),
                                             negative_label=self.meta.get("negative_label"), class_labels=self.meta.get("class_labels"))
                known = item.get("known_at") or now
                records.append(serving.OutcomeRecord(key=serving.key_hash(item["key"], self.secret), outcome=coded,
                                                     known_at=known, recorded_at=now))
            except (ValueError, TypeError) as error:  # the reason, never the value sent
                reason = str(error).splitlines()[0][:200] if not hasattr(error, "errors") else "known_at is not an ISO time with its zone"
                raise Refused(f"outcome {number}: {reason}") from None
        _append(OUTCOMES, records)
        return {"recorded": len(records)}

