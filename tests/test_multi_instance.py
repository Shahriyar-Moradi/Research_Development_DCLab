"""Several app instances on one database (package 10.5): two server processes and one worker process. A page
connected to one instance shows events produced through the other (LISTEN/NOTIFY), the worker's pipeline events
reach it too, two messages to one draft sent to different instances never interleave, and the model pause is one
row every process reads."""
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pgtest  # noqa: E402


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Server:
    def __init__(self, env: dict, args: list[str], port: int | None = None):
        self.port = port
        handle, name = tempfile.mkstemp(prefix="dclab-instance-", suffix=".log")  # read when it fails to start; in the test run's TMPDIR
        os.close(handle)
        self.log = Path(name)
        with self.log.open("w") as out:
            self.process = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}/api{path}"

    def ready(self, seconds: float = 180) -> "Server":
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise AssertionError("the server process exited:\n" + self.log.read_text()[-3000:])
            try:
                self.token = json.load(urllib.request.urlopen(self.url("/config"), timeout=2))["csrf"]
                return self
            except OSError:
                time.sleep(0.3)
        raise AssertionError("the server did not start:\n" + self.log.read_text()[-3000:])

    def call(self, method: str, path: str, body=None, raw: bytes | None = None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        request = urllib.request.Request(self.url(path), method=method, data=data,
                                         headers={"X-DCLab-Token": self.token, "Content-Type": "application/octet-stream" if raw is not None else "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def say(self, draft_id: str, text: str, seconds: float = 60) -> None:
        """Send a message as the page does: while the agent is answering (409, on whichever instance), wait and send again."""
        deadline = time.time() + seconds
        while True:
            try:
                self.call("POST", f"/drafts/{draft_id}/messages", {"text": text})
                return
            except urllib.error.HTTPError as error:
                if error.code != 409 or time.time() > deadline:
                    raise
                time.sleep(0.1)

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()


class Stream(threading.Thread):
    """Reads one draft's event stream from an instance and keeps every event with the time it arrived."""

    def __init__(self, server: Server, draft_id: str, after: int, seconds: float):
        super().__init__(daemon=True)
        self.address = server.url(f"/drafts/{draft_id}/events?after={after}&wait={seconds}")
        self.events: list[tuple[float, str, dict]] = []
        self.open = threading.Event()

    def run(self) -> None:
        with urllib.request.urlopen(self.address, timeout=120) as response:
            self.open.set()
            kind = None
            for raw in response:
                line = raw.decode().rstrip("\n")
                if line.startswith("event: "):
                    kind = line[7:]
                elif line.startswith("data: ") and kind:
                    self.events.append((time.time(), kind, json.loads(line[6:])))

    def has(self, predicate, seconds: float = 30.0):
        deadline = time.time() + seconds
        while time.time() < deadline:
            found = next((e for e in self.events if predicate(e[1], e[2])), None)
            if found:
                return found
            time.sleep(0.05)
        raise AssertionError(f"not seen on the other instance: {[(k, d.get('step') or d.get('text', '')[:40]) for _, k, d in self.events][-12:]}")


class TwoInstancesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("DCLAB_DATABASE_URL"):  # three server processes: run once, in the PostgreSQL suite (make test-pg)
            raise unittest.SkipTest("runs in the PostgreSQL suite")
        url = pgtest.require()
        from dclab_rnd.storage import db

        db.upgrade(url)
        home = Path(tempfile.mkdtemp())
        env = {**os.environ, "DCLAB_DATABASE_URL": url, "DCLAB_AGENT_HOME": str(home), "DCLAB_WORKER": "external",
               "DCLAB_NO_LIVE_MODELS": "1", "OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"}
        cls.servers = []
        try:
            for _ in range(2):
                port = free_port()
                cls.servers.append(Server(env, ["-m", "uvicorn", "dclab_rnd.agentic.server:app", "--port", str(port)], port))
            cls.worker = Server(env, ["-m", "dclab_rnd.worker", "--threads", "2"])
            cls.servers.append(cls.worker)
            cls.a, cls.b = (s.ready() for s in cls.servers[:2])
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        for server in getattr(cls, "servers", []):
            server.stop()

    def test_a_page_on_one_instance_sees_what_the_other_and_the_worker_produce(self):
        draft = self.a.call("POST", "/drafts", {"problem": "Predict which customers churn next month"})
        stream = Stream(self.b, draft["id"], after=0, seconds=60)  # the page is connected to B
        stream.start()
        stream.open.wait(30)
        stream.has(lambda kind, data: kind == "chat" and data.get("role") == "agent")  # the agent's first turn, run through A
        sent = time.time()
        self.a.say(draft["id"], "We decide at the monthly review")
        seen, _, _ = stream.has(lambda kind, data: kind == "chat" and data.get("text") == "We decide at the monthly review")
        self.assertLess(seen - sent, 10)
        csv = b"tenure,monthly,churned\n" + b"".join(f"{i % 70},{20 + i % 90},{i % 3 == 0}\n".encode() for i in range(200))
        self.a.call("PUT", f"/drafts/{draft['id']}/data?filename=c.csv", raw=csv)  # queued by A, run by the worker process
        stream.has(lambda kind, data: kind == "pipeline" and data.get("step") == "ready", seconds=90)

    def test_two_messages_to_one_draft_through_different_instances_never_interleave(self):
        draft = self.a.call("POST", "/drafts", {"problem": "Predict which invoices are paid late"})
        deadline = time.time() + 30
        while time.time() < deadline and not any(m["role"] == "agent" for m in self.b.call("GET", f"/drafts/{draft['id']}")["messages"]):
            time.sleep(0.2)
        texts = ["The first message, sent to A", "The second message, sent to B"]
        threads = [threading.Thread(target=s.say, args=(draft["id"], t)) for s, t in zip((self.a, self.b), texts)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        deadline = time.time() + 30
        while time.time() < deadline:
            messages = self.a.call("GET", f"/drafts/{draft['id']}")["messages"]
            if all(any(m.get("text") == t for m in messages) for t in texts) and messages[-1]["role"] == "agent":
                break
            time.sleep(0.2)
        users = [i for i, m in enumerate(messages) if m["role"] == "user"]
        self.assertEqual(len(users), 2)
        self.assertEqual(messages[users[0] + 1]["role"], "agent")  # the first turn answered before the second message was stored
        self.assertLess(users[0] + 1, users[1])


class BusyTests(unittest.TestCase):
    """409 "still answering" asks the turn lock every instance shares, and two messages at once to one instance: one wins."""

    def setUp(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        from unittest import mock
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "DCLAB_LLM_BASE_URL": "https://api.openai.com/v1"})
        env.start()
        self.addCleanup(env.stop)
        self.app = create_app(Path(tempfile.mkdtemp()))
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}
        self.draft = self.client.post("/api/drafts", json={"problem": "Predict which customers churn next month"}, headers=self.h).json()
        deadline = time.time() + 30
        while time.time() < deadline and self.client.post(f"/api/drafts/{self.draft['id']}/messages", json={"text": "hello"}, headers=self.h).status_code == 409:
            time.sleep(0.1)  # the first turn has finished once a message is accepted
        self.drafts = self.app.state.services.drafts

    def idle(self):
        deadline = time.time() + 30
        while time.time() < deadline and any(not t.done() for t in self.app.state.services.draft_tasks.values()):
            time.sleep(0.05)

    def test_a_turn_held_elsewhere_refuses_a_message(self):
        self.idle()
        held, release = threading.Event(), threading.Event()

        def hold():  # another instance's turn, as far as this one can tell: only the shared lock says so
            with self.drafts.turn(self.draft["id"]):
                held.set()
                release.wait(10)
        threading.Thread(target=hold, daemon=True).start()
        held.wait(5)
        self.assertEqual(self.client.post(f"/api/drafts/{self.draft['id']}/messages", json={"text": "now?"}, headers=self.h).status_code, 409)
        release.set()

    def test_two_messages_at_once_to_one_instance_one_is_refused(self):
        import httpx

        self.idle()

        async def both():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
                return await asyncio.gather(*(c.post(f"/api/drafts/{self.draft['id']}/messages", json={"text": t}, headers=self.h)
                                              for t in ("one", "two")))
        statuses = sorted(r.status_code for r in asyncio.run(both()))
        self.assertEqual(statuses, [202, 409])


class SharedStateTests(unittest.TestCase):
    def test_a_listener_wakes_when_another_connection_emits(self):
        from dclab_rnd.storage import db
        from dclab_rnd.storage.notify import Listener
        from dclab_rnd.storage.postgres import PgDrafts

        url = pgtest.require()
        db.upgrade(url)
        drafts = PgDrafts(db.workspace_for(Path(tempfile.mkdtemp()), url), Path(tempfile.mkdtemp()), url)
        draft = drafts.create("Predict churn")
        listening = Listener(url).start()
        self.addCleanup(listening.stop)
        self.assertTrue(listening.connected.wait(10))

        async def wait():
            waiting = asyncio.ensure_future(listening.wait(draft["id"], 10))
            await asyncio.sleep(0.2)
            clock = time.time()
            await asyncio.to_thread(drafts.emit, draft["id"], "status", {"note": "from another instance"})
            return await waiting, time.time() - clock
        woke, seconds = asyncio.run(wait())
        self.assertTrue(woke)
        self.assertLess(seconds, 5)

    def test_the_model_pause_is_one_row_every_process_reads(self):
        from unittest import mock

        from dclab_rnd.models import pause

        url = pgtest.require()
        with mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": url}):
            from dclab_rnd.storage import db

            db.upgrade(url)
            pause.clear()
            name = pause.key("https://api.example/v1", "m", "sk-1")
            self.assertNotIn("sk-1", name)
            pause.put(name, time.time() + 60, "the provider refused the key")
            pause._LOCAL.clear()  # another process: nothing in its memory
            self.assertEqual(pause.get(name)[1], "the provider refused the key")
            self.assertEqual(pause.get(pause.key("https://api.example/v1", "m", "sk-2")), (0.0, ""))  # a new key is not paused
            pause.clear()
            self.assertEqual(pause.get(name), (0.0, ""))


if __name__ == "__main__":
    unittest.main()
