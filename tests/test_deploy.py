"""Cloud deployment (package 12.6): the infrastructure code for AWS and Google Cloud, and what the app does for it
(the queue's length for the autoscaler, one migration at a time, health probes past the host check)."""
import os
import re
import shutil
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
DEPLOY = ROOT / "deploy"


def tf(path: str) -> str:
    return (DEPLOY / path).read_text(encoding="utf-8")


def every_tf() -> dict[str, str]:
    return {str(p.relative_to(DEPLOY)): p.read_text(encoding="utf-8") for p in sorted(DEPLOY.rglob("*.tf")) if ".terraform" not in p.parts}


class InfrastructureCodeTests(unittest.TestCase):
    def test_both_providers_have_a_staging_and_a_production_built_from_one_module(self):
        for cloud in ("aws", "gcp"):
            for env in ("staging", "production"):
                text = tf(f"{cloud}/{env}/main.tf")
                self.assertIn('source              = "../modules/dclab"', text)
                self.assertIn(f'deployment          = "{env}"', text)
                region = re.search(r'variable "region" \{(.*?)\n\}', text, flags=re.S).group(1)
                self.assertNotIn("default", region, f"{cloud}/{env}: the region is the owner's choice, not a default")

    def test_production_keeps_data_safe_and_staging_can_be_deleted(self):
        settings = {
            ("aws", "production"): ['db_multi_az         = true', 'db_backup_days      = 35', 'deletion_protection = true', 'nat_per_zone        = true'],
            ("aws", "staging"): ['deletion_protection = false', 'db_backup_days      = 7'],
            ("gcp", "production"): ['db_regional         = true', 'deletion_protection = true', 'db_log_days         = 7'],
            ("gcp", "staging"): ['deletion_protection = false'],
        }
        for (cloud, env), wanted in settings.items():
            for line in wanted:
                self.assertIn(line, tf(f"{cloud}/{env}/main.tf"), f"{cloud}/{env}")
        aws, gcp = "".join(every_tf()[k] for k in every_tf() if k.startswith("aws/modules")), "".join(every_tf()[k] for k in every_tf() if k.startswith("gcp/modules"))
        for text in (aws, gcp):  # backups with point-in-time recovery, encrypted storage, a versioned bucket, TLS
            self.assertRegex(text, r"point_in_time_recovery_enabled\s*=\s*true|backup_retention_period\s*=\s*var\.db_backup_days")
            self.assertRegex(text, r"versioning")
        self.assertIn("storage_encrypted            = true", aws)
        self.assertIn('ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"', aws)
        self.assertIn('ssl_mode        = "ENCRYPTED_ONLY"', gcp)
        self.assertIn('min_tls_version = "TLS_1_2"', gcp)

    def test_every_setting_the_code_gives_the_app_is_a_documented_one(self):
        documented = set(re.findall(r"\b(DCLAB_[A-Z0-9_]+|OPENAI_[A-Z_]+|AWS_[A-Z_]+)\b", (ROOT / ".env.example").read_text(encoding="utf-8")))
        used = set()
        for text in every_tf().values():
            used |= set(re.findall(r"\b(DCLAB_[A-Z0-9_]+)\s*=", text))
            used |= set(re.findall(r'"(DCLAB_[A-Z0-9_]+|OPENAI_[A-Z_]+)"', text))
        self.assertGreater(len(used), 8)
        self.assertEqual(sorted(used - documented), [], "a setting the cloud sets that .env.example does not explain")

    def test_no_secret_value_is_written_in_the_code(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import scan_secrets

        for name, text in every_tf().items():
            self.assertEqual([m for line in text.splitlines() for m in scan_secrets.matches(line)], [], name)
            # secrets reach the containers from the secret store, never as a plain environment value
            for key in ("OPENAI_API_KEY", "DCLAB_DATABASE_URL", "DCLAB_SECRET_KEY", "DCLAB_OIDC_CLIENT_SECRET"):
                self.assertNotRegex(text, rf"\b{key}\s*=\s*\"", f"{name} sets {key} as plain text")
        self.assertIn("*.tfstate", (ROOT / ".gitignore").read_text(encoding="utf-8"))  # state holds the generated passwords
        self.assertIn("*.tfvars", (ROOT / ".gitignore").read_text(encoding="utf-8"))

    def test_the_code_is_formatted(self):
        tool = shutil.which("tofu") or shutil.which("terraform")
        if tool is None:
            self.skipTest("neither tofu nor terraform is installed (make deploy-check validates with them)")
        done = subprocess.run([tool, "fmt", "-check", "-recursive", str(DEPLOY)], capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


class QueueMetricTests(unittest.TestCase):
    def test_the_queue_is_counted_in_every_workspace_and_published_where_asked(self):
        from dclab_rnd import observe
        from dclab_rnd.jobs import queue_metric

        class Store:
            def __init__(self, statuses):
                self.rows = [{"status": s} for s in statuses]

            def list(self, active=None, limit=200):
                return [r for r in self.rows if r["status"] in ("queued", "running")]

        services = [type("S", (), {"job_store": Store(["queued", "running", "done"])})(), type("S", (), {"job_store": Store(["queued", "queued"])})()]
        self.assertEqual(queue_metric.counts(services), (3, 4))

        sent = []
        cloudwatch = type("CW", (), {"put_metric_data": lambda self, **kw: sent.append(kw)})()
        with mock.patch.dict(os.environ, {"DCLAB_DEPLOYMENT": "staging"}):
            queue_metric.publish(3, 4, "cloudwatch", sender=cloudwatch)
            queue_metric.publish(5, 6, "gcp", sender=sent.append)
        data = {m["MetricName"]: m for m in sent[0]["MetricData"]}
        self.assertEqual((sent[0]["Namespace"], data["QueuedJobs"]["Value"], data["ActiveJobs"]["Value"]), ("DCLab", 3.0, 4.0))
        self.assertEqual(data["ActiveJobs"]["Dimensions"], [{"Name": "Deployment", "Value": "staging"}])
        series = {s["metric"]["type"].rsplit("/", 1)[1]: s for s in sent[1]}
        self.assertEqual(series["queued_jobs"]["points"][0]["value"], {"int64Value": "5"})
        self.assertEqual(series["active_jobs"]["metric"]["labels"]["deployment"], "staging")
        self.assertEqual((observe.METRICS.value("dclab_jobs_queued"), observe.METRICS.value("dclab_jobs_active")), (5, 6))
        with self.assertRaises(ValueError):
            queue_metric.publish(1, 1, "azure")
        queue_metric.publish(2, 2, None)  # off: only the gauges
        self.assertIn("# TYPE dclab_jobs_queued gauge", observe.METRICS.render())

    def test_the_rule_adds_workers_for_waiting_jobs_and_removes_one_only_when_none_is_busy(self):
        from dclab_rnd.jobs.queue_metric import Rule

        rule, n = Rule(1, 4), 1
        n = rule.step(3, 3, n)
        self.assertEqual(n, 1)  # one minute of waiting: not yet
        n = rule.step(3, 3, n)
        self.assertEqual(n, 2)  # two minutes: one more
        n = rule.step(8, 9, rule.step(8, 9, n))
        self.assertEqual(n, 4)  # more than five waiting: two more
        n = rule.step(9, 12, rule.step(9, 12, n))
        self.assertEqual(n, 4)  # never above the maximum
        for _ in range(30):  # jobs still running, none waiting: nobody is removed, however long
            n = rule.step(0, 2, n)
        self.assertEqual(n, 4)
        for minute in range(15):
            n = rule.step(0, 0, n)
        self.assertEqual(n, 3)  # fifteen idle minutes: one fewer
        for _ in range(100):
            n = rule.step(0, 0, n)
        self.assertEqual(n, 1)  # never below the minimum

    def test_one_worker_sizes_the_google_worker_pool(self):
        from dclab_rnd.jobs.queue_metric import PoolScaler

        calls, state = [], {"scaling": {"manualInstanceCount": 1}}

        def api(method, path, body):
            calls.append((method, path, body))
            if method == "PATCH":
                state["scaling"]["manualInstanceCount"] = body["scaling"]["manualInstanceCount"]
            return state

        pool = "projects/p/locations/europe-west3/workerPools/dclab-staging-worker"
        follower = PoolScaler(pool, 1, 3, call=api, leader=lambda: False)
        self.assertIsNone(follower.check(5, 5))
        self.assertEqual(calls, [])  # a worker without the lock does nothing
        leader = PoolScaler(pool, 1, 3, call=api, leader=lambda: True)
        self.assertIsNone(leader.check(2, 2))
        self.assertEqual(leader.check(2, 2), 2)
        self.assertEqual(calls[-1], ("PATCH", f"{pool}?updateMask=scaling.manualInstanceCount", {"scaling": {"manualInstanceCount": 2}}))
        with mock.patch.dict(os.environ, {"DCLAB_WORKER_POOL": pool, "DCLAB_WORKERS_MIN": "1", "DCLAB_WORKERS_MAX": "6"}):
            made = PoolScaler.from_environment()
        self.assertEqual((made.pool, made.rule.low, made.rule.high), (pool, 1, 6))
        with mock.patch.dict(os.environ, {"DCLAB_WORKER_POOL": ""}):
            self.assertIsNone(PoolScaler.from_environment())


class LeaderAndProtectionTests(unittest.TestCase):
    def test_one_leader_at_a_time_and_a_dropped_connection_gives_the_lock_up(self):
        import pgtest
        import sqlalchemy as sa

        from dclab_rnd.jobs import queue_metric
        from dclab_rnd.storage import db

        url = pgtest.require()
        with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": url}):
            first, second = queue_metric._leader(), queue_metric._leader()
            self.assertTrue(first())
            self.assertFalse(second())  # the lock is held: one worker applies the rule
            self.assertTrue(first())
            with db.engine(url).connect() as c:  # a failover drops the leader's connection, and its lock with it
                c.execute(sa.text("select pg_terminate_backend(pid) from pg_locks where locktype = 'advisory' and granted and objid = :k "
                                  "and database = (select oid from pg_database where datname = current_database())"),
                          {"k": queue_metric.SCALER_LOCK & 0xFFFFFFFF})
            import time

            deadline = time.monotonic() + 10  # the terminated backend ends a moment later; a worker checks again each minute
            while not second() and time.monotonic() < deadline:
                time.sleep(0.2)
            self.assertTrue(second())  # the other worker wins it now
            self.assertFalse(first())  # and the old leader knows it lost it
            db.engine(url).dispose()

    def test_a_busy_worker_holds_scale_in_protection(self):
        from dclab_rnd.jobs.queue_metric import TaskProtection

        calls, now = [], [0.0]
        p = TaskProtection("http://169.254.170.2/v4/abc", call=lambda url, body: calls.append((url, body)), clock=lambda: now[0])
        self.assertTrue(p.update(True))
        self.assertFalse(p.update(True))  # no call while nothing changes
        now[0] = TaskProtection.REFRESH + 1
        self.assertTrue(p.update(True))  # refreshed before it expires
        self.assertTrue(p.update(False))
        self.assertEqual(calls[0], ("http://169.254.170.2/v4/abc/task-protection/v1/state", {"ProtectionEnabled": True, "ExpiresInMinutes": 180}))
        self.assertEqual(calls[-1][1], {"ProtectionEnabled": False})
        self.assertFalse(TaskProtection("", call=calls.append).update(True))  # not in ECS: nothing to do


class DeploymentDrainTests(unittest.TestCase):
    def test_an_old_worker_drains_when_a_deployment_replaces_it(self):
        from dclab_rnd.jobs import queue_metric

        describe = lambda cluster, name: f"arn:aws:ecs:eu-central-1:123:task-definition/dclab-staging-worker:{7 if name == 'worker' else 0}"
        self.assertTrue(queue_metric.superseded("dclab-staging/worker", own="dclab-staging-worker:6", describe=describe))
        self.assertFalse(queue_metric.superseded("dclab-staging/worker", own="dclab-staging-worker:7", describe=describe))
        self.assertTrue(queue_metric.superseded("dclab-staging/worker", own="dclab-staging-worker:17", describe=describe))  # 17 is not 7
        self.assertFalse(queue_metric.superseded("", own="x:1", describe=describe))  # outside ECS: never
        with mock.patch.dict(os.environ, {"DCLAB_ECS_SERVICE": "dclab-staging/worker", "ECS_CONTAINER_METADATA_URI_V4": ""}):
            self.assertFalse(queue_metric.superseded(describe=describe))

    def test_a_draining_worker_finishes_what_it_runs_and_claims_nothing_new(self):
        import tempfile
        import time

        from dclab_rnd.jobs import FileJobs, Worker
        from dclab_rnd.jobs import worker as jw

        gate, started = threading.Event(), threading.Event()

        def run(env, payload):
            started.set()
            gate.wait(10)
        jw.handler("t-drain", run, lambda *a: None)
        self.addCleanup(jw.HANDLERS.pop, "t-drain", None)
        store = FileJobs(Path(tempfile.mkdtemp()) / "jobs.json")
        worker = Worker(store, None, threads=2, poll=0.05).start()
        self.addCleanup(worker.stop, 2.0)
        first = store.enqueue("t-drain", "a", {})
        self.assertTrue(started.wait(10))
        worker.drain()
        second = store.enqueue("t-drain", "b", {})
        worker.notify()
        time.sleep(0.5)
        self.assertEqual(store.get(second["id"])["status"], "queued")  # left for a newer worker
        gate.set()
        deadline = time.monotonic() + 10
        while store.get(first["id"])["status"] != "done" and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(store.get(first["id"])["status"], "done")  # the running one finished, not interrupted


class WatchStepTests(unittest.TestCase):
    def test_a_superseded_worker_drains_finishes_and_then_ends_so_ecs_starts_the_current_version(self):
        from dclab_rnd.worker import watch_step

        class W:
            def __init__(self):
                self.running, self.claimed, self.draining = {"j1": object()}, set(), False

            def drain(self):
                self.draining = True

            def has_work(self):
                return bool(self.running or self.claimed)

        workers = [W()]
        pool = type("Pool", (), {"all": lambda self: [type("S", (), {"worker": w})() for w in workers]})()
        calls = []
        protection = type("P", (), {"update": lambda self, busy: calls.append(busy)})()
        state, answers = {"ticks": 0, "drained": False}, iter([False, True])
        for _ in range(12):  # the first minute: not replaced
            self.assertFalse(watch_step(pool, protection, state, check=lambda: next(answers)))
        self.assertFalse(workers[0].draining)
        for _ in range(12):  # the second: a deployment replaced it, but a job still runs
            self.assertFalse(watch_step(pool, protection, state, check=lambda: next(answers)))
        self.assertTrue(workers[0].draining)
        self.assertTrue(calls[-1])  # protected while the job runs
        workers.append(W())  # a workspace opened meanwhile gets a draining worker too
        workers[1].running = {}
        self.assertFalse(watch_step(pool, protection, state, check=lambda: True))
        self.assertTrue(workers[1].draining)
        workers[0].running = {}  # the job finished, but another was being claimed as it drained
        workers[0].claimed = {"j2"}
        self.assertFalse(watch_step(pool, protection, state, check=lambda: True))  # claimed, not yet running: still busy
        self.assertTrue(calls[-1])
        workers[0].claimed = set()  # it ran and finished
        self.assertFalse(watch_step(pool, protection, state, check=lambda: True))  # idle once: not yet
        self.assertTrue(watch_step(pool, protection, state, check=lambda: True))  # idle twice in a row: end the process
        self.assertFalse(calls[-1])


class MigrationLockTests(unittest.TestCase):
    def test_instances_starting_together_migrate_one_at_a_time(self):
        import pgtest
        import sqlalchemy as sa

        from dclab_rnd.storage import db

        from sqlalchemy.engine import make_url

        url = pgtest.require()
        # an empty database, as on a first deployment: three instances start at once and all must end at the head
        target = make_url(url).set(database=f"{make_url(url).database}_lock").render_as_string(hide_password=False)
        admin = sa.create_engine(url, isolation_level="AUTOCOMMIT")
        name = make_url(target).database
        try:
            with admin.connect() as c:
                c.execute(sa.text(f'drop database if exists "{name}"'))
                c.execute(sa.text(f'create database "{name}"'))
        except Exception as exc:  # noqa: BLE001 — a role that cannot create databases
            self.skipTest(f"cannot create a scratch database: {type(exc).__name__}")

        def drop():
            db.engine(target).dispose()
            with admin.connect() as c:
                c.execute(sa.text(f'drop database if exists "{name}" with (force)'))
            admin.dispose()
        self.addCleanup(drop)
        seen, errors = [], []
        real = __import__("alembic.command", fromlist=["upgrade"]).upgrade

        def upgrade(config, revision):  # while one instance migrates, its lock is held
            with db.engine(target).connect() as c:  # this database's holders only (other test processes migrate theirs)
                seen.append(c.execute(sa.text("select count(*) from pg_locks where locktype = 'advisory' and granted and objid = :k "
                                              "and database = (select oid from pg_database where datname = current_database())"),
                                      {"k": db.MIGRATION_LOCK & 0xFFFFFFFF}).scalar())
            real(config, revision)

        with mock.patch("alembic.command.upgrade", upgrade):
            def start():
                try:
                    db.upgrade(target)
                except Exception as exc:  # noqa: BLE001
                    errors.append(repr(exc))
            threads = [threading.Thread(target=start) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(60)
        self.assertEqual(seen, [1, 1, 1])  # each ran holding the lock, and never two held it at once (the others waited)
        self.assertEqual(errors, [])
        self.assertEqual(db.current(target), db.head())


if __name__ == "__main__":
    unittest.main()
