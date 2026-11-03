"""DCLab Watch's ML side in MLflow (B4 of the 2026-11-03 plan): each model DCLab fits and each bench run, as an MLflow run.

Traces (``dclab_rnd.tracing``) show what the agents and the model calls did; this shows the classic ML: a student's or a decision
classifier's training (its point, family, strength, features, rows, tables, dev score beside the rule's, the card's version) and
each ``make bench`` run (each layer's state). It speaks MLflow's REST interface (``/api/2.0/mlflow/...``), so DCLab needs no
MLflow package; the server is the one ``scripts/watch.py mlflow-up`` starts.

On only when ``DCLAB_TRACE_EXPORT`` names ``mlflow``; a server that is down is counted, never raised. Only names and numbers
leave: parameters and metrics, never a row, a value of a table or a person's words.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

TIMEOUT = 3.0
_counts = {"runs": 0, "failed": 0}


def on() -> bool:
    raw = (os.environ.get("DCLAB_TRACE_EXPORT") or "").lower()
    return "mlflow" in [x.strip() for x in raw.split(",")]


def _base() -> str:
    url = os.environ.get("DCLAB_MLFLOW_URL") or "http://127.0.0.1:5000"
    return url.rstrip("/")


def _call(path: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(f"{_base()}/api/2.0/mlflow/{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310 - a configured local endpoint
        return json.loads(response.read() or b"{}")


def _experiment(name: str) -> str:
    try:
        return _call("experiments/create", {"name": name})["experiment_id"]
    except urllib.error.HTTPError:  # it exists already
        request = urllib.request.Request(f"{_base()}/api/2.0/mlflow/experiments/get-by-name?experiment_name={urllib.parse.quote(name)}")
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310
            return json.loads(response.read())["experiment"]["experiment_id"]


def log_run(experiment: str, run_name: str, params: dict[str, Any], metrics: dict[str, float], tags: Optional[dict[str, str]] = None) -> bool:
    """One finished run: parameters (short words), metrics (numbers) and tags; False when off or when MLflow is away."""
    if not on():
        return False
    try:
        exp = _experiment(experiment)
        now = int(time.time() * 1000)
        run = _call("runs/create", {"experiment_id": exp, "run_name": run_name[:250], "start_time": now,
                                    "tags": [{"key": k, "value": str(v)[:500]} for k, v in (tags or {}).items()]})["run"]["info"]["run_id"]
        _call("runs/log-batch", {"run_id": run,
                                 "params": [{"key": k[:250], "value": str(v)[:500]} for k, v in params.items() if v is not None],
                                 "metrics": [{"key": k[:250], "value": float(v), "timestamp": now, "step": 0}
                                             for k, v in metrics.items() if isinstance(v, (int, float)) and v == v]})
        _call("runs/update", {"run_id": run, "status": "FINISHED", "end_time": int(time.time() * 1000)})
        _counts["runs"] += 1
        return True
    except Exception:  # noqa: BLE001 - MLflow away: counted, never raised into DCLab
        _counts["failed"] += 1
        return False


def log_card(card: Any) -> bool:
    """A decision model's card (``dclab_core.decision_models.ModelCard``) as a run of the experiment ``dclab-decision-models``."""
    dev = getattr(card, "dev", None)
    rule = getattr(card, "rule_dev", None)
    metrics = {"dev_accuracy": getattr(dev, "accuracy", None), "rule_dev_accuracy": getattr(rule, "accuracy", None),
               "trained_on": card.trained_on, "tables": card.groups, "features": card.features,
               "dev_calibration_error": card.dev_calibration_error, **{f"oof.{k}": v for k, v in (card.oof_accuracy or {}).items()}}
    params = {"point": card.point, "family": card.family, "strength": card.strength, "version": card.version,
              "fields": ",".join(card.fields)[:500], "trained_set": card.trained_set[:16], "chosen_by": card.chosen_by[:250]}
    return log_run("dclab-decision-models", f"{card.point}.{card.family}", params, {k: v for k, v in metrics.items() if v is not None},
                   tags={"dclab.kind": "decision_model", "dclab.point": card.point})


def log_bench(record: dict[str, Any]) -> bool:
    """A recorded ``make bench`` run as a run of ``dclab-bench``: each layer's state (red 0, amber 1, thin 2, green 3) and times."""
    order = {"red": 0, "amber": 1, "thin": 2, "green": 3, "empty": -1}
    metrics = {f"layer.{k}": order.get(v, -1) for k, v in (record.get("states") or {}).items()}
    metrics.update({f"seconds.{c['suite']}": c.get("seconds") or 0 for c in record.get("commands", [])})
    return log_run("dclab-bench", f"bench {record.get('commit')}", {"commit": record.get("commit"), "dirty": record.get("dirty"),
                                                                     "layers": ",".join(record.get("layers") or [])}, metrics,
                   tags={"dclab.kind": "bench"})


def stats() -> dict[str, Any]:
    return dict(_counts)


__all__ = ["on", "log_run", "log_card", "log_bench", "stats"]
