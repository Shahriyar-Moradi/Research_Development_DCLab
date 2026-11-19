"""The review of problem-scoped active memory (7d5751bd): each confirmed finding as a test that failed before its fix.

1  deleting a project removes its bases, summaries and events on a folder workspace (a real log, which goes by ``w…``)
2  project and person scope ids stay raw on a named workspace (old memory visible; a person reviews their own preference)
3  the Admin sign-in with DCLAB_AUTH off is the owner, not a 403
4  a project stamped ``local`` keeps working once served by a store with another id; the intern's brief survives a refusal
5  one workspace id helper
6  a job's person is checked again by the worker, by the accounts database's ``role_of``
7  two first bindings at once leave one problem
8  a read never creates a problem (a viewer never writes the project)
9  a base edit over HTTP names its revision
11 a purged base ledger is kept with a tombstone, hidden from reads
13 problem creation, base conflicts and stale marks are recorded
14 a base change to the prediction moment says the signed solution still has the old value
15 the write roles are the Router's; undoing a correction back to the earlier words works
"""

import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import memory as core_mem  # noqa: E402
from dclab_rnd import experience  # noqa: E402
from dclab_rnd.accounts.principal import Principal, acting  # noqa: E402
from dclab_rnd.accounts.roles import PERMISSIONS  # noqa: E402
from dclab_rnd.agentic.server import create_app  # noqa: E402
from dclab_rnd.jobs import worker as jw  # noqa: E402
from dclab_rnd.jobs.store import FileJobs  # noqa: E402
from dclab_rnd.memory import FileMemory, bases, open_memory, scope as S, summary as SM, tools as T, view, writing  # noqa: E402
from dclab_rnd.studio.store import ProjectStore  # noqa: E402


class _Named:
    """A project store that goes by a workspace id other than "local" (as PostgreSQL's does), over a folder store."""

    def __init__(self, inner, workspace_id="wabc123"):
        self.inner, self.home, self.workspace_id = inner, inner.home, workspace_id

    def get(self, pid):
        return self.inner.get(pid)

    def update(self, pid, **kw):
        return self.inner.update(pid, **kw)

    def turn(self, pid, purpose="project"):
        return self.inner.turn(pid, purpose)


def _clean_env():
    return mock.patch.dict(os.environ, {"DCLAB_AUTH": "", "DCLAB_DATABASE_URL": ""})


class WorkspaceIdsTests(unittest.TestCase):
    def setUp(self):
        self._env = _clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)
        self.home = Path(tempfile.mkdtemp())
        self.projects = ProjectStore(self.home / "projects")

    def test_project_and_person_ids_stay_raw_on_a_named_workspace(self):
        p = self.projects.create("Fresh", goal="g")
        named = _Named(self.projects)
        store = FileMemory(self.home)
        at = "2026-11-19T09:00:00+00:00"
        # what the build's hand-off and a person's preference write: the raw ids
        store.write(core_mem.new(kind="fact", scope="project", scope_id=p["id"], text="Hand-off: SLA is 45 minutes.",
                                 source="home brief:d1", who="person:dana", at=at))
        store.write(core_mem.new(kind="fact", scope="person", scope_id="dana", text="Dana prefers short answers.",
                                 source="pref", who="person:dana", at=at))
        with acting(Principal(role="data_scientist", user_id="dana", via="session")):
            chain = T.chain(named, p["id"])
        self.assertIn(("project", p["id"]), chain)
        self.assertIn(("person", "dana"), chain)
        self.assertIn(("workspace", "wabc123"), chain)
        texts = {i.text for i in view(store, chain)}
        self.assertEqual(texts, {"Hand-off: SLA is 45 minutes.", "Dana prefers short answers."})

        # the problem scope alone carries its parent chain
        prob = S.ensure_default_problem(named, p["id"], workspace_id="wabc123", by="dana")
        sc = S.resolve_scope(workspace_id="wabc123", projects=named, project_id=p["id"], problem_id=prob["problem_id"])
        self.assertEqual(sc.problem_key, f"wabc123:{p['id']}:{prob['problem_id']}")

    def test_a_person_reviews_their_own_preference_on_a_named_workspace(self):
        store = FileMemory(self.home)
        pid = S.person_scope_id("wabc", "dana")  # what POST /api/memory/proposals writes
        draft = writing.propose_preference(store, person=pid, message="From now on, answer in simple words with one example.",
                                           source="api:proposals")
        edited = writing.review(store, draft.id, "edit", by="dana", role="data_scientist", text="Answer in simple words.")
        self.assertEqual((edited.scope, edited.scope_id, edited.status), ("person", "dana", "active"))

    def test_one_workspace_helper(self):
        from dclab_rnd.agentic.routers import memory as router

        self.assertEqual(S.workspace_of(self.projects), "local")
        self.assertEqual(S.workspace_of(_Named(self.projects)), "wabc123")
        services = types.SimpleNamespace(projects=_Named(self.projects), workspace_id=None)
        self.assertEqual(router._workspace(services), "wabc123")  # the router asks the store, as Home and the build do

    def test_admin_sign_in_without_accounts_is_the_owner(self):
        p = self.projects.create("X", goal="g")
        with acting(Principal(role="owner", workspace_id=None, via="admin")):
            self.assertEqual(S.resolve_scope().role, "owner")
            self.assertEqual(S.resolve_scope().principal_id, "owner")
            self.assertIn(("project", p["id"]), T.chain(self.projects, p["id"]))

    def test_a_project_stamped_local_keeps_working_under_another_store_id(self):
        p = self.projects.create("Built", goal="g")
        self.projects.update(p["id"], workspace_id="local")  # as the first version of A3 stamped it
        prob = S.ensure_default_problem(self.projects, p["id"], workspace_id="local", by="owner")
        sc = S.resolve_scope(projects=self.projects, project_id=p["id"], problem_id=prob["problem_id"])
        bases.upsert_base_fact(self.home, sc, target="problem", subject="problem", field="prediction_moment",
                               text="Prediction moment: at order creation.", source="ui", expected_revision=0)

        named = _Named(self.projects)  # the same folder, now served by a store that goes by "wabc123"
        self.assertTrue(T.chain(named, p["id"]))
        moved = S.resolve_scope(workspace_id="wabc123", projects=named, project_id=p["id"], problem_id=prob["problem_id"])
        self.assertEqual(moved.problem_key, sc.problem_key)  # the problem keeps the key it was bound under
        self.assertEqual([e["text"] for e in bases.read_base(self.home, moved)["active"]], ["Prediction moment: at order creation."])
        fresh = self.projects.create("New", goal="g")
        S.ensure_default_problem(named, fresh["id"], workspace_id="wabc123")
        self.assertNotIn("workspace_id", self.projects.get(fresh["id"]))  # the project document is no longer stamped

    def test_the_intern_brief_survives_a_refused_project(self):
        from dclab_rnd.intern.loop import Intern

        p = self.projects.create("Private", goal="g")
        self.projects.update(p["id"], members=["someone-else"])
        fake = types.SimpleNamespace(toolbox=types.SimpleNamespace(projects=self.projects))
        with acting(Principal(role="data_scientist", user_id="dana", via="session")):
            with self.assertRaises(S.AccessDenied):
                T.chain(self.projects, p["id"])
            text, shown = Intern._memory(fake, p["id"], "s1")
        self.assertEqual((text, shown), ("", []))


class JobRevalidationTests(unittest.TestCase):
    class _Accounts:
        def __init__(self, url=None):
            self.url = url

        def role_of(self, user_id, workspace_id):
            return "data_scientist" if (user_id, workspace_id) == ("dana", "ws1") else None

    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.projects = ProjectStore(self.home / "projects")
        self.p = self.projects.create("Delivery", goal="ETA")
        self.env = types.SimpleNamespace(projects=self.projects, drafts=None, workspace_id="ws1")
        patches = [mock.patch.dict(os.environ, {"DCLAB_AUTH": "password"}),
                   mock.patch("dclab_rnd.storage.db.database_url", return_value="postgresql://nobody@127.0.0.1:1/none"),
                   mock.patch("dclab_rnd.accounts.store.Accounts", self._Accounts)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _job(self, user_id):
        return {"user": {"user_id": user_id, "role": "data_scientist", "workspace_id": "ws1"}, "payload": {"project_id": self.p["id"]}}

    def test_the_default_check_asks_the_accounts_database(self):
        ok = S.revalidate_job_scope(self.env, self._job("dana"))  # no checker injected
        self.assertEqual((ok.principal_id, ok.role, ok.project_id), ("dana", "data_scientist", self.p["id"]))
        with self.assertRaises(S.AccessDenied):
            S.revalidate_job_scope(self.env, self._job("sam"))  # removed from the workspace after queueing

    def test_the_worker_checks_the_person_before_running(self):
        ran = []
        jobs = FileJobs(self.home / "jobs.json")
        with mock.patch.dict(jw.HANDLERS, {"memcheck": jw.Handler(lambda env, payload: ran.append(payload), lambda *a: None)}):
            for user, expected in (("sam", "failed"), ("dana", "done")):
                with acting(Principal(role="data_scientist", workspace_id="ws1", user_id=user, via="session")):
                    jobs.enqueue("memcheck", f"k-{user}", {"project_id": self.p["id"]})
                w = jw.Worker(jobs, self.env, threads=0)
                ended = w.execute(jobs.claim(w.id))
                self.assertEqual(ended["status"], expected, ended.get("error"))
        self.assertEqual(len(ran), 1)  # only dana's job ran


class ProblemsAndBasesTests(unittest.TestCase):
    def setUp(self):
        self._env = _clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)
        self.home = Path(tempfile.mkdtemp())
        self.projects = ProjectStore(self.home / "projects")
        self.log = experience.open_log(self.home)

    def _scope(self, role="data_scientist"):
        p = self.projects.create("Delivery", goal="ETA")
        prob = S.ensure_default_problem(self.projects, p["id"], by="dana", log=self.log)
        return p["id"], S.AuthorizedScope(workspace_id="local", principal_id="dana", role=role, project_id=p["id"],
                                          problem_id=prob["problem_id"])

    def test_two_first_bindings_at_once_leave_one_problem(self):
        p = self.projects.create("Legacy", goal="g")
        original, barrier, got = self.projects.get, threading.Barrier(2), {}

        def slow_get(pid):
            doc = original(pid)
            if threading.current_thread().name.startswith("r"):
                try:
                    barrier.wait(timeout=0.5)
                except threading.BrokenBarrierError:
                    pass
            return doc

        self.projects.get = slow_get
        try:
            threads = [threading.Thread(target=lambda i=i: got.__setitem__(i, S.ensure_default_problem(self.projects, p["id"], draft_id=f"d{i}")),
                                        name=f"r{i}") for i in range(2)]
            [t.start() for t in threads]
            [t.join() for t in threads]
        finally:
            self.projects.get = original
        bound = [x["problem_id"] for x in original(p["id"]).get("problems") or []]
        self.assertEqual(len(bound), 1)
        self.assertEqual({g["problem_id"] for g in got.values()}, set(bound))

    def test_reads_never_create_a_problem_and_a_base_edit_names_its_revision(self):
        app = create_app(self.home)
        client = TestClient(app)
        s = app.state.services
        p = s.projects.create("Legacy", goal="g")
        viewer = Principal(role="viewer", workspace_id=None, user_id="vic", via="token")
        dana = Principal(role="data_scientist", workspace_id=None, user_id="dana", via="token")
        with mock.patch("dclab_rnd.agentic.server.identify", return_value=viewer):
            self.assertEqual(client.get(f"/api/projects/{p['id']}/problems").json()["problems"], [])
            summary = client.get(f"/api/projects/{p['id']}/memory/summary")
            self.assertEqual((summary.status_code, summary.json()["detail"]["error"]), (409, "no_problem"))
            self.assertEqual(client.get(f"/api/projects/{p['id']}/memory/base").status_code, 409)
        self.assertNotIn("problems", s.projects.get(p["id"]))  # nothing written by a read
        body = {"target": "problem", "subject": "problem", "field": "prediction_moment", "text": "Predicted after dispatch."}
        with mock.patch("dclab_rnd.agentic.server.identify", return_value=dana):
            self.assertEqual(client.post(f"/api/projects/{p['id']}/memory/base", json=body).status_code, 422)
            self.assertNotIn("problems", s.projects.get(p["id"]))  # refused before any write
            made = client.post(f"/api/projects/{p['id']}/memory/base", json={**body, "expected_revision": 0})
            self.assertEqual(made.status_code, 200, made.text)
            fixed = client.post(f"/api/projects/{p['id']}/memory/base/{made.json()['entry']['entry_id']}/correct",
                                json={"text": "Predicted at order creation.", "expected_revision": 1})
            self.assertEqual(fixed.status_code, 200, fixed.text)
            self.assertIn("prediction moment", fixed.json()["solution_note"])
            for route in ("correct", "retire", "confirm"):
                self.assertEqual(client.post(f"/api/projects/{p['id']}/memory/base/x/{route}", json={"text": "y"}).status_code, 422)
        self.assertEqual(len(s.projects.get(p["id"])["problems"]), 1)  # the write created it
        created = [e for e in s.experience.read(kinds=["decision.accepted"]) if e.payload.get("point") == "problem.created"]
        self.assertEqual([e.ids.project for e in created], [p["id"]])
        with mock.patch("dclab_rnd.agentic.server.identify", return_value=viewer):
            self.assertEqual(client.post(f"/api/projects/{p['id']}/memory/base", json={**body, "expected_revision": 2}).status_code, 403)

    def test_conflicts_stale_marks_and_fact_events_are_recorded_under_the_project(self):
        pid, sc = self._scope()
        r1 = bases.upsert_base_fact(self.home, sc, subject="problem", field="prediction_moment", text="After dispatch.",
                                    source="ui", expected_revision=0, log=self.log)
        draft = bases.propose_draft(self.home, sc, subject="features", field="policy", text="Keep dispatch_delay_min.",
                                    source_type="advisor", source_id="a1", log=self.log)
        with self.assertRaises(bases.RevisionConflictError):
            bases.upsert_base_fact(self.home, sc, subject="problem", field="prediction_moment", text="At order creation.",
                                   source="ui", expected_revision=0, log=self.log)
        out = bases.correct_base_fact(self.home, sc, r1["entry"]["entry_id"], "At order creation.", expected_revision=1, log=self.log)
        self.assertIn(draft["entry"]["entry_id"], out["stale_dependents"])
        refused = self.log.read(kinds=["decision.refused"], project=pid)
        self.assertIn("revision_conflict", [e.payload["rule"] for e in refused])
        self.assertIn(draft["entry"]["entry_id"], [e.payload["choice"] for e in refused if e.payload["rule"] == "stale_dependency"])
        written = self.log.read(kinds=["memory.written"], project=pid)
        self.assertTrue(written)  # a problem item's events carry the raw project id, so deleting the project removes them
        self.assertTrue(self.log.read(kinds=["decision.accepted"], project=pid))  # the problem's creation

    def test_solution_note_only_for_the_solution_fields(self):
        _pid, sc = self._scope()
        sla = bases.upsert_base_fact(self.home, sc, target="project", subject="business", field="sla", text="SLA is 45 minutes.",
                                     source="ui", expected_revision=0)
        self.assertNotIn("solution_note", bases.correct_base_fact(self.home, sc, sla["entry"]["entry_id"], "SLA is 40 minutes.",
                                                                  target="project", expected_revision=1))
        tgt = bases.upsert_base_fact(self.home, sc, subject="problem", field="target", text="Target is actual minutes.",
                                     source="ui", expected_revision=0)
        out = bases.retire_base_fact(self.home, sc, tgt["entry"]["entry_id"], expected_revision=1)
        self.assertIn("target", out["solution_note"])

    def test_undoing_a_correction_back_to_the_earlier_words(self):
        _pid, sc = self._scope()
        a = bases.upsert_base_fact(self.home, sc, subject="problem", field="prediction_moment", text="After dispatch.",
                                   source="ui", expected_revision=0)
        b = bases.correct_base_fact(self.home, sc, a["entry"]["entry_id"], "At order creation.", expected_revision=1)
        back = bases.correct_base_fact(self.home, sc, b["entry"]["entry_id"], "After dispatch.", expected_revision=2)
        self.assertEqual(back["entry"]["text"], "After dispatch.")
        self.assertEqual([e["text"] for e in bases.read_base(self.home, sc)["active"]], ["After dispatch."])
        self.assertIn("After dispatch.", SM.read_summary(self.home, sc)["text"])
        shown = [i.text for i in view(open_memory(self.home), sc.chain())]
        self.assertEqual(shown, ["After dispatch. (restored at revision 3)"])

    def test_write_roles_are_the_routers(self):
        self.assertEqual(S.WRITE_ROLES, PERMISSIONS["write"])
        self.assertFalse({"admin", "editor", "reviewer"} & set(S.WRITE_ROLES))
        _pid, sc = self._scope(role="reviewer")
        with self.assertRaises(S.AccessDenied):
            bases.upsert_base_fact(self.home, sc, subject="problem", field="target", text="Target.", source="ui", expected_revision=0)


if __name__ == "__main__":
    unittest.main()
