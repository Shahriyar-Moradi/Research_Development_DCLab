"""23.2: the browser's own events (page views, actions, uploads, errors shown) reach the record under the browser's
visit, as the person; a value that is not a route, an element's name or a code is refused, so nothing a person typed
can enter; every signed-in role may send them."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from dclab_core import experience as contract
    from dclab_rnd.accounts.guard import permission
except ImportError as error:  # the agent environment may lack pydantic
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class ValidatorTest(unittest.TestCase):
    def test_only_routes_names_codes_and_counts_pass(self):
        self.assertEqual(contract.ui_payload("ui.page_viewed", {"view": "home", "pane": "", "route": "#home", "query": "secret"}),
                         {"view": "home", "route": "#home"})  # an empty pane is left out; a field the kind has not is dropped
        self.assertEqual(contract.ui_payload("ui.upload_finished", {"view": "home", "status": "ok", "seconds": 1.5}), {"view": "home", "status": "ok", "seconds": 1.5})
        for bad in ({"view": "home", "name": "my password is hunter2"}, {"view": "home", "name": "x=1&y=2"}, {"view": "a b"}):
            with self.assertRaises(ValueError):
                contract.ui_payload("ui.action", bad)
        for bad in ({"view": "home", "bytes": -1}, {"view": "home", "bytes": True}, {"view": "home", "bytes": "12"}):
            with self.assertRaises(ValueError):
                contract.ui_payload("ui.upload_started", bad)
        with self.assertRaises(ValueError):
            contract.ui_payload("chat.person", {"text": "hello"})
        self.assertEqual(set(contract.UI_KINDS), {"ui.page_viewed", "ui.action", "ui.upload_started", "ui.upload_finished", "ui.error_shown"})


@unittest.skipIf(MISSING, MISSING)
class RouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_rnd.agentic.server import create_app

        cls.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        cls.services = cls.client.app.state.services
        cls.h = {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}  # the first request starts the browser's visit

    def test_the_events_land_under_the_visit_and_nothing_typed_enters(self):
        batch = [{"kind": "ui.page_viewed", "view": "home", "pane": "", "route": "#home"},
                 {"kind": "ui.action", "view": "home", "name": "home-start", "target": "button"},
                 {"kind": "ui.upload_started", "view": "home", "file_kind": "csv", "bytes": 25693},
                 {"kind": "ui.upload_finished", "view": "home", "status": "ok", "seconds": 1.5},
                 {"kind": "ui.error_shown", "view": "home", "code": "failed", "status": "500"},
                 {"kind": "ui.page_viewed", "view": "learn", "route": "#learn", "query": "my secret plans"},
                 {"kind": "ui.action", "view": "home", "name": "my password is hunter2", "target": "input"},
                 {"kind": "chat.person", "text": "hello"}]
        out = self.client.post("/api/experience/ui", json={"events": batch}, headers=self.h).json()
        self.assertEqual(out, {"recorded": 6, "refused": 2})
        mine = [e for e in self.services.experience.read() if e.kind.startswith("ui.")]
        self.assertEqual(len(mine), 6)
        self.assertEqual({e.ids.visit for e in mine} - {None}, {mine[0].ids.visit})  # one visit: the browser's cookie
        self.assertEqual({e.actor for e in mine}, {"person"})
        text = " ".join(str(e.payload) for e in mine)
        self.assertNotIn("secret", text)
        self.assertNotIn("hunter2", text)
        self.assertEqual(next(e for e in mine if e.payload.get("view") == "learn").payload, {"view": "learn", "route": "#learn"})
        self.assertEqual(self.client.post("/api/experience/ui", json={"events": [{"kind": "ui.action"}] * 51}, headers=self.h).status_code, 422)
        self.assertEqual(self.client.post("/api/experience/ui", json=[], headers=self.h).status_code, 422)

    def test_every_signed_in_role_may_send_them(self):
        self.assertEqual(permission("POST", "/api/experience/ui"), "read")


if __name__ == "__main__":
    unittest.main()
