"""The frontend's API client (package 13.1): generated from the server's routes, used by every page instead of path
strings, checking its arguments at run time; and the same loading, empty and failed states on every page."""
import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "dclab_rnd" / "agentic" / "web" / "src"


def sources() -> dict[str, str]:
    files = {p.name: p.read_text(encoding="utf-8") for p in sorted((SRC / "views").glob("*.html"))}
    files["core.js"] = (SRC / "core.js").read_text(encoding="utf-8")
    return files


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from dclab_rnd.agentic.web import client
        except ImportError as error:
            raise unittest.SkipTest(f"Studio dependencies not installed: {error}")
        cls.client = client
        cls.generated = client.generate()

    def test_the_committed_client_is_what_the_routes_give(self):
        self.assertEqual((SRC / "api.js").read_text(encoding="utf-8"), self.generated, "run: make web")

    def test_every_page_calls_the_client_never_a_path_string(self):
        names = set(re.findall(r"^    (\w+): \[", self.generated, flags=re.M))
        used = set()
        for name, text in sources().items():
            # a request with a path written by hand: DC.api('/…'), api(`/…`), or a link to /api/…
            self.assertNotRegex(text, r"(?<![\w.])(?:DC\.)?api\(\s*[`'\"]", f"{name} calls the API with a path string: use DC.client")
            for line in text.splitlines():
                if re.search(r"[`'\"]/api/", line) and "fetch('/api/config'" not in line and "fetch('/api/auth/password'" not in line:
                    self.fail(f"{name} writes an /api/ path: use DC.client.href or DC.client.path\n{line.strip()[:160]}")
            used |= set(re.findall(r"DC\.client\.(?:path\.|href\.)?(\w+)\(", text))
        self.assertGreater(len(used), 60)
        self.assertEqual(sorted(used - names), [], "a page calls a route the server does not have")

    def test_generating_the_client_opens_no_workspace_and_no_database(self):
        import os
        import tempfile

        never = Path(tempfile.mkdtemp()) / "never"
        script = ("import dclab_rnd.agentic.server as server\n"
                  "assert 'app' not in vars(server), 'importing the server built the app'\n"
                  "from dclab_rnd.agentic.web import client\n"
                  "assert len(client.schema()['paths']) > 50\n")
        env = {**os.environ, "DCLAB_DATABASE_URL": "postgresql+psycopg://nobody@127.0.0.1:1/nothing", "DCLAB_ALLOWED_HOSTS": "example.com",
               "DCLAB_AGENT_HOME": str(never), "DCLAB_AUTH": "password"}
        done = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertFalse(never.exists())  # the configured workspace folder was not even made

    def test_an_unexpected_operation_id_is_refused(self):
        self.assertEqual(self.client.route_name("get_project_api_projects__project_id__get", "get", "/api/projects/{project_id}"), "get_project")
        with self.assertRaises(ValueError):
            self.client.route_name("custom_id", "get", "/api/projects")

    def test_the_client_builds_paths_and_refuses_wrong_arguments(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        script = """
const vm = require('vm');
const calls = [];
const window = { DC: { api: (path, opts) => { calls.push([path, opts.method, opts.body || null, Object.keys(opts).sort()]); return Promise.resolve(null); } } };
vm.runInNewContext(process.argv[1], { window, encodeURIComponent });
const c = window.DC.client, out = { calls };
c.getProject('a b/c');
c.runAll('p1', { query: { start: 'leakage', wait: undefined } });
c.saveSolution('p1', { body: { target: 'y' }, quiet: true });
c.approveStage('p1', 'models', { body: { choice: 'm' } });
out.href = c.href.exportReport('p1');
out.path = c.path.events('d1', { query: { after: 3 } });
for (const [label, fn] of [['missing', () => c.getProject()], ['unknown', () => c.opsJobs({ query: { lmit: 2 } })]]) {
  try { fn(); out[label] = null; } catch (e) { out[label] = e.message; }
}
console.log(JSON.stringify(out));
"""
        done = subprocess.run([node, "-e", script, self.generated], capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        out = json.loads(done.stdout)
        self.assertEqual(out["calls"], [
            ["/projects/a%20b%2Fc", "GET", None, ["method"]],
            ["/projects/p1/run?start=leakage", "POST", None, ["method"]],
            ["/projects/p1/solution", "PUT", {"target": "y"}, ["body", "method", "quiet"]],
            ["/projects/p1/stages/models/approve", "POST", {"choice": "m"}, ["body", "method"]],
        ])
        self.assertEqual((out["href"], out["path"]), ("/api/projects/p1/export/report", "/drafts/d1/events?after=3"))
        self.assertEqual(out["missing"], "DC.client.getProject needs project_id")
        self.assertEqual(out["unknown"], "DC.client.opsJobs has no query parameter lmit")


class StatesTests(unittest.TestCase):
    def test_pages_share_one_way_to_say_loading_empty_and_failed(self):
        core = (SRC / "core.js").read_text(encoding="utf-8")
        for part in ("loading(box", "empty(box", "failed(box", "data-state-retry", "Try again", "loading.track(watch, el, out"):
            self.assertIn(part, core)
        self.assertIn("err.status !== 401", core)  # a signed-out answer goes to sign-in, not to a failed state
        for name, text in sources().items():
            # a failure shown by hand, without the shared state: <div class="empty">…${esc(e.message)}
            self.assertNotRegex(text, r'class="empty"[^<]*\$\{esc\(e(rr)?\.message\)\}', f"{name}: use DC.states.failed")


if __name__ == "__main__":
    unittest.main()
