"""Package 19.4: the record of every session is found and read. A search by kind, project, draft, person, words, code
and dates answers with the matching events and their visits; "every session where a build was refused" and "every
visit that touched project P" are one search each; the page's routes show a timeline without a tool's values and
without prompts unless asked; only the owner reads them. The log tests run on PostgreSQL under ``make test-pg``."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    from dclab_rnd import experience
    from dclab_rnd.accounts import guard
    from dclab_rnd.accounts.principal import Principal, acting
    from dclab_rnd.accounts.roles import ROLES, allows
    from dclab_rnd.agentic.api_models import api_routes
    from dclab_rnd.agentic.server import create_app
    from dclab_rnd.experience import finder
except ImportError as error:  # the agent environment has no pandas
    MISSING = str(error)
else:
    MISSING = ""


def _log(home: Path):
    """The workspace's log as the server would open it: PostgreSQL under make test-pg, the folder otherwise."""
    if os.environ.get("DCLAB_DATABASE_URL"):
        from dclab_rnd.storage import db

        db.upgrade()
        return experience.PgLog(db.workspace_for(home))
    return experience.open_log(home)


def seed(log):
    """Three visits over two days: one refused build, one run on project p1, one with an error and a denied tool."""
    r = lambda kind, actor, payload, at, **ids: experience.record(kind, actor, payload, log=log, at=at, **ids)  # noqa: E731
    r("draft.created", "person", {"problem": "Find fraudulent delivery orders"}, "2026-10-12T09:00:00+00:00", visit="vA", conversation="dA", user="u1")
    r("chat.person", "person", {"text": "Nothing is labelled, the orders are not marked"}, "2026-10-12T09:01:00+00:00", visit="vA", conversation="dA")
    r("draft.refused", "system", {"reason": "A list of cases alone cannot be built", "rule": "cases_alone"}, "2026-10-12T09:02:00+00:00",
      visit="vA", conversation="dA")
    r("draft.created", "person", {"problem": "Predict late orders"}, "2026-10-13T10:00:00+00:00", visit="vB", conversation="dB")
    r("draft.built", "person", {"project": "p1"}, "2026-10-13T10:05:00+00:00", visit="vB", conversation="dB", project="p1")
    r("graph.move", "system", {"move": "run_stage", "allowed": True}, "2026-10-13T10:06:00+00:00", visit="vB", project="p1")
    r("stage.finished", "system", {"stage": "final", "metric": "roc_auc", "value": 0.81}, "2026-10-13T10:09:00+00:00", visit="vB", project="p1")
    r("model.request", "agent", {"purpose": "draft_chat", "cost": 0.0002, "currency": "EUR", "prompt": "the whole prompt", "reply": "ok"},
      "2026-10-13T10:02:00+00:00", visit="vB", conversation="dB")
    r("tool.call", "agent", {"tool": "run_stage", "verdict": "denied", "code": "DCLAB-R17", "arguments": {"customer": "Jane Roe"}},
      "2026-10-13T11:00:00+00:00", visit="vC", project="p1")
    r("error.raised", "system", {"where": "/api/drafts/{draft_id}/build", "code": "E_PARSE", "message": "could not read the file"},
      "2026-10-13T11:01:00+00:00", visit="vC")


@unittest.skipIf(MISSING, f"needs the studio's dependencies ({MISSING})")
class FinderTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.log = _log(self.home)
        seed(self.log)

    def visits(self, **query):
        return [c.visit for c in finder.search(self.log, **query)["sessions"]]

    def test_a_refused_build_is_one_search(self):
        self.assertEqual(self.visits(kind="draft.refused"), ["vA"])
        found = finder.search(self.log, kind="refused")
        self.assertEqual({e.kind for e in found["events"]}, {"draft.refused", "tool.call"})  # an allowed move is not refused
        self.assertEqual(sorted(c.visit for c in found["sessions"]), ["vA", "vC"])

    def test_a_project_is_one_search(self):
        self.assertEqual(sorted(self.visits(project="p1")), ["vB", "vC"])
        self.assertEqual(self.visits(draft="dA"), ["vA"])

    def test_words_codes_families_and_dates(self):
        self.assertEqual(self.visits(text="NOTHING IS LABELLED"), ["vA"])           # what a person wrote, in any case
        self.assertEqual(self.visits(text="the whole prompt"), [])            # model text is not searched
        self.assertEqual(self.visits(text="jane roe"), [])                     # a tool's arguments (cell values) are not searched
        self.assertEqual(self.visits(code="dclab-r17"), ["vC"])
        self.assertEqual(self.visits(code="DCLAB-R1"), [])                    # a code is a whole code: R1 is not R17
        self.assertEqual(self.visits(code="cases_alone"), ["vA"])
        self.assertEqual(sorted(self.visits(kind="draft")), ["vA", "vB"])    # a family: every draft.*
        self.assertEqual(self.visits(kind="errors"), ["vC"])
        self.assertEqual(self.visits(until="2026-10-13"), ["vA"])             # until is excluded: the 12th only
        self.assertEqual(sorted(self.visits(since="2026-10-13T10:30:00+00:00")), ["vC"])
        self.assertEqual(self.visits(kind="stage.finished", since="2026-10-13", until="2026-10-14"), ["vB"])

    def test_newest_first_and_cards(self):
        found = finder.search(self.log, project="p1")
        times = [e.at for e in found["events"]]
        self.assertEqual(times, sorted(times, reverse=True))
        card = {c.visit: c for c in found["sessions"]}["vB"]
        self.assertEqual((card.outcome, card.cost), ("finished", {"EUR": 0.0002}))

    def test_a_person_and_a_local_day(self):
        self.assertEqual(self.visits(person="u1"), ["vA"])
        self.assertEqual(self.visits(person="u2"), [])
        # the 12th in Tehran runs from 20:30 UTC on the 11th to 20:30 UTC on the 12th: vA at 09:00 UTC
        self.assertEqual(self.visits(since="2026-10-12T00:00:00+03:30", until="2026-10-13T00:00:00+03:30"), ["vA"])
        # the 13th in UTC-5 runs from 05:00 UTC on the 13th to 05:00 UTC on the 14th: vB at 10:00 and vC at 11:00 UTC
        self.assertEqual(sorted(self.visits(since="2026-10-13T00:00:00-05:00", until="2026-10-14T00:00:00-05:00")), ["vB", "vC"])
        self.assertEqual(self.visits(since="2026-10-13T11:00:30Z"), ["vC"])  # a trailing Z is UTC

    def test_refusals_are_not_crowded_out_by_allowed_moves(self):
        for i in range(30):  # newer than every refusal, all allowed
            experience.record("graph.move", "system", {"move": "run_stage", "allowed": True}, log=self.log,
                              at=f"2026-10-14T10:{i:02d}:00+00:00", visit="vD", project="p1")
        found = finder.search(self.log, kind="refused", scan=20)
        self.assertEqual(sorted(c.visit for c in found["sessions"]), ["vA", "vC"])
        self.assertFalse(found["capped"])

    def test_dates_that_are_not_times(self):
        for bad in ("yesterday", "9999-12-31T23:00:00-05:00", "0001-01-01T00:00:00+05:00"):
            with self.assertRaises(ValueError, msg=bad):
                finder.search(self.log, since=bad)

    def test_the_command_line_shows_no_values_and_no_prompts(self):
        import contextlib
        import io
        from unittest import mock

        from dclab_rnd.experience import __main__ as cli

        for argv in (["search", "--kind", "tool"], ["search", "--kind", "model"], ["show", "vB"], ["show", "vC"], ["show", "vB", "--json"]):
            out = io.StringIO()
            with mock.patch.object(cli, "_log", return_value=self.log), mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": ""}), \
                    contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(argv), 0, argv)
            text = out.getvalue()
            self.assertNotIn("Jane Roe", text, argv)
            self.assertNotIn("the whole prompt", text, argv)
        out = io.StringIO()
        with mock.patch.object(cli, "_log", return_value=self.log), mock.patch.dict(os.environ, {"DCLAB_DATABASE_URL": ""}), \
                contextlib.redirect_stdout(out):
            cli.main(["show", "vB", "--model-text"])
        self.assertIn("the whole prompt", out.getvalue())

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            finder.search(self.log, kind="draft.exploded")

    def test_a_page_never_shows_cell_values_and_model_text_only_on_request(self):
        tool = finder.search(self.log, kind="tool.call")["events"][0]
        shown = finder.view(tool)
        self.assertNotIn("arguments", shown["payload"])
        self.assertIn("arguments:cell_values", shown["redacted"])
        self.assertNotIn("arguments", finder.view(tool, model_text=True)["payload"])
        request = finder.search(self.log, kind="model.request")["events"][0]
        self.assertNotIn("prompt", finder.view(request)["payload"])
        self.assertEqual(finder.view(request, model_text=True)["payload"]["prompt"], "the whole prompt")


@unittest.skipIf(MISSING, f"needs the studio's dependencies ({MISSING})")
class RoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = Path(tempfile.mkdtemp())
        cls.client = TestClient(create_app(cls.home))
        seed(cls.client.app.state.services.experience)

    def test_the_latest_sessions(self):
        r = self.client.get("/api/experience/sessions")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        visits = [c["visit"] for c in body["items"]]
        self.assertTrue({"vA", "vB", "vC"} <= set(visits), visits)
        self.assertGreaterEqual(body["totals"]["refused"], 2)
        self.assertIn("draft.refused", body["kinds"])
        a = next(c for c in body["items"] if c["visit"] == "vA")
        self.assertEqual(a["problem"], "Find fraudulent delivery orders")
        for card in body["items"]:  # what the page prints: words and numbers, never an object
            self.assertTrue(card["family"] is None or isinstance(card["family"], str), card["family"])

    def test_a_draft_names_its_family(self):
        drafts = self.client.app.state.services.drafts
        d = drafts.create("Detect fraudulent delivery orders; nothing is labelled")
        experience.record("draft.created", "person", {"problem": d["problem"]}, log=self.client.app.state.services.experience,
                          visit="vF", conversation=d["id"])
        card = next(c for c in self.client.get("/api/experience/sessions").json()["items"] if c["visit"] == "vF")
        self.assertIsInstance(card["family"], str)
        self.assertIn(card["family"], ("anomaly", "rare_event"))

    def test_a_search_and_a_timeline(self):
        r = self.client.get("/api/experience/search", params={"kind": "draft.refused"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual([c["visit"] for c in r.json()["sessions"]], ["vA"])
        t = self.client.get("/api/experience/sessions/vB").json()
        self.assertEqual([e["kind"] for e in t["events"]][:2], ["draft.created", "model.request"])
        request = next(e for e in t["events"] if e["kind"] == "model.request")
        self.assertNotIn("prompt", request["payload"])
        t = self.client.get("/api/experience/sessions/vB", params={"model_text": "true"}).json()
        self.assertEqual(next(e for e in t["events"] if e["kind"] == "model.request")["payload"]["prompt"], "the whole prompt")

    def test_bad_queries(self):
        self.assertEqual(self.client.get("/api/experience/search", params={"kind": "nope"}).status_code, 422)
        self.assertEqual(self.client.get("/api/experience/search", params={"since": "yesterday"}).status_code, 422)
        self.assertEqual(self.client.get("/api/experience/search", params={"until": "9999-12-31T23:00:00-05:00"}).status_code, 422)
        self.assertEqual(self.client.get("/api/experience/search", params={"project": "../etc"}).status_code, 422)
        self.assertEqual(self.client.get("/api/experience/sessions/nobody").status_code, 404)
        self.assertEqual(self.client.get("/api/experience/sessions", params={"limit": 0}).status_code, 422)

    def test_only_the_owner_reads_timelines(self):
        routes = [r for r in api_routes(self.client.app) if getattr(r, "path", "").startswith("/api/experience")]
        self.assertEqual(sorted((m, r.path) for r in routes for m in r.methods),  # reading them, and deleting a session (19.10)
                         [("DELETE", "/api/experience/sessions/{visit}"), ("GET", "/api/experience/errors"), ("GET", "/api/experience/search"),
                          ("GET", "/api/experience/sessions"), ("GET", "/api/experience/sessions/{visit}"), ("POST", "/api/experience/ui")])
        for route in routes:
            for method in route.methods:
                if (method, route.path) == ("POST", "/api/experience/ui"):  # 23.2: the browser writes its own events, as any signed-in role
                    self.assertEqual(guard.permission(method, route.path), "read")
                    continue
                self.assertEqual(guard.permission(method, route.path), "timelines", route.path)
        self.assertEqual([role for role in ROLES if allows(role, "timelines")], ["owner"])
        with acting(Principal(role="reviewer", user_id="u1")), self.assertRaises(HTTPException) as refused:
            guard.demand("timelines", "read the sessions' record")
        self.assertEqual(refused.exception.status_code, 403)
        from unittest import mock

        reviewer = Principal(role="reviewer", user_id="u1", via="session")
        with mock.patch.object(guard, "current", return_value=reviewer):  # through the app: the route itself refuses
            for path in ("/api/experience/sessions", "/api/experience/sessions/vA", "/api/experience/search?kind=refused"):
                self.assertEqual(self.client.get(path).status_code, 403, path)
            self.assertEqual(self.client.get("/api/config").status_code, 200)  # the patch reaches the guard, not the whole app

    def test_a_long_visit_says_it_was_cut(self):
        from unittest import mock

        from dclab_rnd.agentic.pages import sessions

        with mock.patch.object(sessions, "TIMELINE", 2):
            t = self.client.get("/api/experience/sessions/vB").json()
        self.assertTrue(t["cut"])
        self.assertEqual(len(t["events"]), 2)
        self.assertFalse(self.client.get("/api/experience/sessions/vB").json()["cut"])


if __name__ == "__main__":
    unittest.main()
