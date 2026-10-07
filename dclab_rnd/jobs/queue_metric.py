"""The job queue, published where a cloud's autoscaler can read it, and the scaling rule (package 12.6).

Each worker process (``python -m dclab_rnd.worker``) counts, once a minute, the jobs waiting to be claimed (queued) and
the jobs waiting or running (active), in every workspace it runs, and publishes both:

    DCLAB_QUEUE_METRIC=cloudwatch   AWS CloudWatch, namespace DCLab, metrics QueuedJobs and ActiveJobs, dimension
                                     Deployment; the ECS workers scale on them (deploy/aws); credentials from the task's role
    DCLAB_QUEUE_METRIC=gcp          Google Cloud Monitoring, custom.googleapis.com/dclab/queued_jobs and active_jobs,
                                     labels deployment and instance (an alert reads them); token from the metadata server
    (unset)                         not published; the Prometheus gauges dclab_jobs_queued and dclab_jobs_active are set either way

The rule both clouds follow (``Rule``): jobs waiting two minutes in a row add a worker (two when more than five wait);
no job waiting or running for fifteen minutes removes one; never below the minimum or above the maximum. A worker is
removed only when none is busy. On AWS, CloudWatch alarms apply it, and a worker running a job also holds ECS scale-in
protection (``TaskProtection``), so the autoscaler never stops it. On Google Cloud (DCLAB_WORKER_POOL names the worker
pool), one worker at a time, the one holding an advisory lock in PostgreSQL, applies it by setting the pool's instance
count through the Cloud Run Admin API; a job claimed in the moment between its count and the change can still be
interrupted (it is then retried from the Compute page).

DCLAB_DEPLOYMENT names the deployment (staging, production). A failed publish or scaling call is logged and tried
again a minute later: it never stops the worker. Only counts leave the process, never a job.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

METADATA = "http://metadata.google.internal/computeMetadata/v1"
RUN_API = "https://run.googleapis.com/v2"
SCALER_LOCK = 0x44434C4153  # "DCLAS": the one worker that applies the rule on Google Cloud


def counts(services: Iterable[Any]) -> tuple[int, int]:
    """(queued, active): jobs waiting to be claimed, and jobs waiting or running, in every workspace this process runs."""
    queued = active = 0
    for s in services:
        rows = s.job_store.list(active=True, limit=10_000)
        active += len(rows)
        queued += sum(1 for j in rows if j.get("status") == "queued")
    return queued, active


def target() -> str | None:
    value = (os.environ.get("DCLAB_QUEUE_METRIC") or "").strip().lower()
    return value or None


def publish(queued: int, active: int, where: str | None = None, sender: Any = None) -> None:
    """Send the counts to ``where`` (cloudwatch or gcp); ``sender`` replaces the cloud client in tests."""
    from .. import observe

    observe.METRICS.set("dclab_jobs_queued", queued, "Jobs waiting to be claimed (this process's workspaces)")
    observe.METRICS.set("dclab_jobs_active", active, "Jobs waiting or running (this process's workspaces)")
    where = where or target()
    deployment = (os.environ.get("DCLAB_DEPLOYMENT") or "default").strip()
    if where == "cloudwatch":
        client = sender or _cloudwatch()
        dimensions = [{"Name": "Deployment", "Value": deployment}]
        client.put_metric_data(Namespace="DCLab", MetricData=[
            {"MetricName": "QueuedJobs", "Value": float(queued), "Unit": "Count", "Dimensions": dimensions},
            {"MetricName": "ActiveJobs", "Value": float(active), "Unit": "Count", "Dimensions": dimensions}])
    elif where == "gcp":
        instance = _metadata("/instance/id") if sender is None else "test"
        write = sender or _gcp_write
        write([_gcp_series("queued_jobs", queued, deployment, instance), _gcp_series("active_jobs", active, deployment, instance)])
    elif where is not None:
        raise ValueError(f"DCLAB_QUEUE_METRIC is cloudwatch or gcp, not {where!r}")


class Rule:
    """The scaling rule: +1 (or +2 above five waiting) after jobs waited two checks in a row, -1 after fifteen checks with
    nothing waiting or running; between ``low`` and ``high``. One check a minute."""

    def __init__(self, low: int, high: int):
        self.low, self.high = max(0, low), max(low, high)
        self.waiting = self.idle = 0

    def step(self, queued: int, active: int, current: int) -> int:
        self.waiting = self.waiting + 1 if queued > 0 else 0
        self.idle = self.idle + 1 if active == 0 else 0
        if self.waiting >= 2:
            self.waiting = 0
            return min(self.high, current + (2 if queued > 5 else 1))
        if self.idle >= 15:
            self.idle = 0
            return max(self.low, current - 1)
        return max(self.low, min(self.high, current))


class PoolScaler:
    """Applies the rule to a Cloud Run worker pool (DCLAB_WORKER_POOL=projects/<p>/locations/<r>/workerPools/<name>,
    DCLAB_WORKERS_MIN, DCLAB_WORKERS_MAX). Only the worker holding the advisory lock acts; ``call`` replaces the API in tests."""

    def __init__(self, pool: str, low: int, high: int, call: Callable[[str, str, dict | None], dict] | None = None, leader: Callable[[], bool] | None = None):
        self.pool, self.rule = pool, Rule(low, high)
        self.call = call or _run_api
        self.leader = leader or _leader()

    @classmethod
    def from_environment(cls) -> "PoolScaler | None":
        pool = (os.environ.get("DCLAB_WORKER_POOL") or "").strip()
        if not pool:
            return None
        return cls(pool, int(os.environ.get("DCLAB_WORKERS_MIN") or 1), int(os.environ.get("DCLAB_WORKERS_MAX") or 1))

    def check(self, queued: int, active: int) -> int | None:
        """One minute's check; returns the count it set, or None when it set nothing."""
        if not self.leader():
            return None
        current = int(((self.call("GET", self.pool, None).get("scaling") or {}).get("manualInstanceCount")) or self.rule.low)
        wanted = self.rule.step(queued, active, current)
        if wanted == current:
            return None
        self.call("PATCH", f"{self.pool}?updateMask=scaling.manualInstanceCount", {"scaling": {"manualInstanceCount": wanted}})
        return wanted


class TaskProtection:
    """ECS task scale-in protection while this worker runs a job (AWS; ECS_AGENT_URI is set inside a task): the service's
    autoscaler never stops a busy task. Set when the first job starts, cleared when the last ends, refreshed every two
    hours (it expires by itself after three, should the worker die without clearing it)."""

    REFRESH = 7200.0

    def __init__(self, uri: str | None = None, call: Callable[[str, dict], Any] | None = None, clock: Callable[[], float] | None = None):
        import time

        self.uri = (uri if uri is not None else os.environ.get("ECS_AGENT_URI") or "").rstrip("/")
        self.call, self.clock = call or _agent_put, clock or time.monotonic
        self.on: bool | None = None
        self.since = 0.0

    def update(self, busy: bool) -> bool:
        """Tell ECS when the state changes (or the protection is due a refresh); True when it called."""
        if not self.uri or (busy == self.on and not (busy and self.clock() - self.since >= self.REFRESH)):
            return False
        body = {"ProtectionEnabled": True, "ExpiresInMinutes": 180} if busy else {"ProtectionEnabled": False}
        self.call(f"{self.uri}/task-protection/v1/state", body)
        self.on, self.since = busy, self.clock()
        return True


def superseded(service: str | None = None, own: str | None = None, describe: Callable[[str, str], str] | None = None) -> bool:
    """True when this ECS task runs an older task definition than its service now has (a deployment is replacing it).
    The worker then drains: it claims nothing new, its running jobs finish, its protection clears, and ECS stops it,
    so a deployment neither interrupts a job nor waits on a worker that keeps taking new ones. DCLAB_ECS_SERVICE is
    cluster/service; outside ECS it is always False."""
    service = service if service is not None else (os.environ.get("DCLAB_ECS_SERVICE") or "")
    meta = os.environ.get("ECS_CONTAINER_METADATA_URI_V4") or ""
    if not service or "/" not in service or (own is None and not meta):
        return False
    if own is None:
        with urllib.request.urlopen(meta.rstrip("/") + "/task", timeout=5) as answer:
            task = json.loads(answer.read().decode())
        own = f"{task['Family']}:{task['Revision']}"
    cluster, name = service.split("/", 1)
    current = (describe or _describe_service)(cluster, name)
    return bool(current) and not current.endswith("/" + own)


def _describe_service(cluster: str, name: str) -> str:
    import boto3

    found = boto3.client("ecs").describe_services(cluster=cluster, services=[name]).get("services") or []
    return found[0].get("taskDefinition", "") if found else ""


def _agent_put(url: str, body: dict) -> None:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="PUT", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5):
        pass


def _leader() -> Callable[[], bool]:
    """True in the one worker holding the scaler's advisory lock (held on its own connection for the process's life)."""
    held: dict[str, Any] = {}

    def check() -> bool:
        import sqlalchemy as sa

        from ..storage import db

        if held.get("connection") is not None:
            try:  # a dropped connection (maintenance, failover) took the lock with it: try to win it again
                held["connection"].execute(sa.text("select 1"))
                return True
            except Exception:  # noqa: BLE001
                try:
                    held.pop("connection").close()
                except Exception:  # noqa: BLE001
                    pass

        connection = db.engine().connect().execution_options(isolation_level="AUTOCOMMIT")
        if connection.execute(sa.text("select pg_try_advisory_lock(:k)"), {"k": SCALER_LOCK}).scalar():
            held["connection"] = connection
            return True
        connection.close()
        return False
    return check


def _cloudwatch():
    import boto3

    return boto3.client("cloudwatch")


def _gcp_series(name: str, value: int, deployment: str, instance: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {"metric": {"type": f"custom.googleapis.com/dclab/{name}", "labels": {"deployment": deployment, "instance": instance}},  # one series per instance
            "resource": {"type": "global", "labels": {}}, "metricKind": "GAUGE", "valueType": "INT64",
            "points": [{"interval": {"endTime": now}, "value": {"int64Value": str(int(value))}}]}


def _metadata(path: str) -> str:
    request = urllib.request.Request(METADATA + path, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=5) as answer:
        return answer.read().decode()


def _token() -> str:
    return json.loads(_metadata("/instance/service-accounts/default/token"))["access_token"]


def _gcp_write(series: list[dict[str, Any]]) -> None:
    """Time series points through Cloud Monitoring's REST API, with the instance's own token (no key file)."""
    project = _metadata("/project/project-id")
    for s in series:
        s["resource"]["labels"] = {"project_id": project}
    body = json.dumps({"timeSeries": series}).encode()
    request = urllib.request.Request(f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10):
        pass


def _run_api(method: str, path: str, body: dict | None) -> dict:
    request = urllib.request.Request(f"{RUN_API}/{path}", data=json.dumps(body).encode() if body is not None else None, method=method,
                                     headers={"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=15) as answer:
        return json.loads(answer.read().decode() or "{}")
