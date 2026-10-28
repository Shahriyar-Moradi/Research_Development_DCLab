"""Phase A7: DCLab's agents behind an external chat, over /mcp (docs/guides/AGENTIC_FOUNDATION_PLAN.md).

A7.2, the Home agent: a chat starts a draft, answers the agent's questions, gives it a table, builds the project on the
person's words and then talks to the built project's orchestrator, all through MCP tool calls that go through the page's
own routes, with no model (the agents' rules). A7.3, every operation of the product through one bridge. A7.4, the
workspace chat's tables, which no route serves.
"""

import asyncio
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


class McpChatCase(unittest.TestCase):
    """The standalone /mcp on a fresh workspace, and a tool call through it."""

    def setUp(self):
        try:
            import mcp  # noqa: F401
            from starlette.testclient import TestClient
            from dclab_rnd import mcp_server
        except ImportError as error:
            self.skipTest(f"dependencies not installed: {error}")
        self.workspace = Path(tempfile.mkdtemp())
        self.app = mcp_server.standalone_app(self.workspace)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.services = self.client.app.state.services
        self.n = 0

    def call(self, name, arguments):
        self.n += 1
        response = self.client.post("/mcp", json={"jsonrpc": "2.0", "id": self.n, "method": "tools/call",
                                                   "params": {"name": name, "arguments": arguments}}, headers=HEADERS)
        self.assertEqual(response.status_code, 200, response.text[:300])
        return json.loads(response.json()["result"]["content"][0]["text"])

    def need_sample(self, key="bank_marketing"):
        from dclab_rnd.studio import data as studio_data

        if key not in {s["key"] for s in studio_data.sample_catalog()}:
            self.skipTest(f"the {key} sample table is not part of this edition")


class HomeAgentOverMcpTests(McpChatCase):
    def test_the_chat_scopes_are_listed_beside_the_project_tools(self):
        response = self.client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}, headers=HEADERS)
        tools = {t["name"]: t for t in response.json()["result"]["tools"]}
        self.assertTrue({"home_start", "home_message", "home_add_data", "home_status", "home_build", "home_list"} <= set(tools))
        self.assertTrue({"dclab_capabilities", "dclab_call", "list_tables", "describe_table", "compute"} <= set(tools))
        self.assertTrue({"create_project", "run_stage", "search_evidence"} <= set(tools))  # the project tools stay
        self.assertEqual(tools["home_build"]["inputSchema"]["required"], ["draft_id", "person_words"])

    def test_a_problem_to_a_built_project_through_the_chat(self):
        self.need_sample()
        started = self.call("home_start", {"problem": "Rank clients for the term-deposit call campaign"})
        draft = started["draft_id"]
        self.assertFalse(started["answering"], started)
        self.assertTrue(any(m["role"] == "agent" for m in started["messages"]), started)
        self.assertTrue(started["open_questions"], "the agent asks its questions")
        self.assertTrue(all("options" in q for q in started["open_questions"]))
        self.assertTrue(started["workflow"], "the agent draws the solution workflow")
        self.assertIn(draft, [d["draft_id"] for d in self.call("home_list", {})["drafts"]])
        # The agent takes an answer: the person's words reach it and it replies with its next step.
        asked = started["open_questions"][0]
        answered = self.call("home_message", {"draft_id": draft, "text": "We call each client once; a call costs about 5 euros."})
        self.assertTrue(any(m["role"] == "agent" for m in answered["messages"]), answered)
        self.assertEqual(answered["messages"][0]["role"], "user")
        stored = self.services.drafts.get(draft)
        self.assertTrue(any(q.get("answered") for q in stored["questions"]) or asked not in answered["open_questions"], stored["questions"])
        # Words that do not ask to build never build; asking to build without a table is answered with what is missing.
        self.assertIn("do not ask to build", self.call("home_build", {"draft_id": draft, "person_words": "what is the outcome?"})["error"])
        refused = self.call("home_build", {"draft_id": draft, "person_words": "build it"})
        self.assertFalse(refused["built"], refused)
        self.assertIn("table", " ".join(m["text"] for m in refused["messages"] if m["role"] == "agent").lower(), refused)
        # The table, as the page attaches a sample; the tool returns once the agent has read it (its message on that table).
        added = self.call("home_add_data", {"draft_id": draft, "sample": "bank_marketing"})
        self.assertEqual(added["table_status"], "ready", added)
        self.assertTrue(any(m["role"] == "agent" and m["kind"] == "analysis" for m in added["messages"]), added["messages"])
        # A chat builds only through home_build (its host asks the person first): home_message refuses words that ask to build.
        self.assertIn("home_build", self.call("home_message", {"draft_id": draft, "text": "build it"})["error"])
        built = self.call("home_build", {"draft_id": draft, "person_words": "build it"})
        self.assertTrue(built["built"] and built["project_id"], built)
        project = self.services.projects.get(built["project_id"])
        self.assertIn("duration", {f["column"] for f in project["solution"]["forbidden"]})  # the audit's leak, kept by the build's checks
        self.assertEqual(self.call("home_status", {"draft_id": draft})["project_id"], built["project_id"])
        self.assertIn("new draft", self.call("home_build", {"draft_id": draft, "person_words": "build it"})["note"])  # a built draft: the orchestrator's version
        # On a built draft the Home orchestrator answers.
        answer = self.call("home_message", {"draft_id": draft, "text": "What is the problem this project solves?"})
        self.assertTrue(any(m["role"] == "agent" for m in answer["messages"]), answer)
        stages = self.call("home_message", {"draft_id": draft, "text": "Which stages has the project run?"})
        cards = [m["card"] for m in stages["messages"] if m.get("card")]
        self.assertTrue(cards and cards[0]["title"] and (cards[0]["facts"] or cards[0].get("rows")), stages["messages"])  # the cards, not only text
        # Both sides recorded: the MCP calls kept whole; the draft's events marked as an external chat's; its decisions in the ledger.
        calls = [json.loads(x)["tool"] for x in (self.workspace / "mcp" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(calls[:3], ["home_start", "home_list", "home_message"])
        events = self.services.experience.read(conversation=draft)
        created = [e for e in events if e.kind == "draft.created"]
        self.assertEqual([(e.actor, e.payload.get("surface")) for e in created], [("person", "mcp")])
        said = [e.payload.get("surface") for e in events if e.kind == "chat.person"]
        self.assertTrue(said and set(said) == {"mcp"}, said)  # the chat's words, never a person's own on Home
        agents = [e.payload.get("surface") for e in events if e.kind == "chat.agent"]
        self.assertTrue(agents and set(agents) == {"mcp"}, agents)  # the data-ready turn too, though a job's worker ran it
        self.assertTrue(self.services.ledger.records(draft=draft), "the draft's decisions in the workspace's ledger")

    def test_a_turn_under_way_is_said_never_waited_on_forever(self):
        from dclab_rnd import chat_agents

        started = self.call("home_start", {"problem": "Predict churn for the telecom customers next month"})
        draft = started["draft_id"]
        held, release = threading.Event(), threading.Event()

        def hold():  # another turn (the page's, or another instance's on PostgreSQL) holds the draft
            with self.services.drafts.turn(draft):
                held.set()
                release.wait(20)

        thread = threading.Thread(target=hold)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(release.set)
        held.wait(10)
        self.assertTrue(self.call("home_status", {"draft_id": draft})["answering"])
        self.assertIn("still answering", self.call("home_message", {"draft_id": draft, "text": "hello"})["error"])
        out = asyncio.run(chat_agents._settle(self.services, draft, 0, wait_s=0.5))
        self.assertTrue(out["answering"] and "home_status" in out["note"], out)
        release.set()
        thread.join(10)
        self.assertFalse(self.call("home_status", {"draft_id": draft})["answering"])

    def test_paths_the_chat_may_never_send(self):
        self.need_sample()
        started = self.call("home_start", {"problem": "Predict churn for the telecom customers next month"})
        draft = started["draft_id"]
        folder = Path(tempfile.mkdtemp())

        def file(*parts, text="a,b\n1,2\n"):
            path = folder.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return path

        cases = {
            "sealed": [file("holdout_customers.csv"), file("hold-out", "rows.csv"), file("sealed", "rows.csv"), file("test_sealed.parquet")],
            "hidden": [file(".kaggle", "rows.csv")],
            "key or a credential": [file("kaggle.json", text='{"key": "x"}'), file("credentials.json")],
            "workspace": [self.workspace / "drafts" / "x.csv"],
        }
        cases["workspace"][0].parent.mkdir(parents=True, exist_ok=True)
        cases["workspace"][0].write_text("a,b\n1,2\n", encoding="utf-8")
        (folder / "public").mkdir()
        (folder / "public" / "link").symlink_to(folder / "sealed")  # a link that hides a sealed folder's name
        cases["sealed"].append(folder / "public" / "link" / "rows.csv")
        for why, paths in cases.items():
            for path in paths:
                with self.subTest(path=str(path)):
                    self.assertIn(why, self.call("home_add_data", {"draft_id": draft, "path": str(path)}).get("error", ""))
        self.assertIn("error", self.call("home_add_data", {"draft_id": draft}))  # neither sample nor path
        self.assertIn("error", self.call("home_message", {"draft_id": "nope", "text": "hi"}))
        from types import SimpleNamespace

        from dclab_rnd import chat_agents

        signed_in = SimpleNamespace(settings=SimpleNamespace(auth="password", agent_home=self.workspace), store=self.services.store)
        self.assertIn("sign-in", chat_agents._path_refusal(signed_in, file("unsealed", "ok.csv")))  # read on the server's machine: never with accounts on
        self.assertIsNone(chat_agents._path_refusal(self.services, file("unsealed", "ok.csv")))
        # A table in a folder whose name only contains a sealed word is taken, through the page's own upload route.
        from dclab_rnd.studio import data as studio_data

        frame, _ = studio_data.load_sample("bank_marketing")
        table = folder / "unsealed" / "clients.csv"
        frame.head(400).to_csv(table, index=False)
        added = self.call("home_add_data", {"draft_id": draft, "path": str(table)})
        self.assertEqual(added["table_status"], "ready", added)
        self.assertEqual(added["table"]["name"], "clients.csv")


class PersonSurfaceTests(McpChatCase):
    def test_a_click_relayed_by_dclab_code_starts_a_visit_on_its_surface(self):
        for surface in ("dclab-code", "dclab-code:relay", "made-up"):
            from starlette.testclient import TestClient

            fresh = TestClient(self.app)  # a new visit each time (no visit cookie carried over)
            r = fresh.post("/api/drafts", json={"problem": "Predict which shipments arrive late"},
                           headers={"x-dclab-token": self.services.csrf, "x-dclab-surface": surface})
            self.assertEqual(r.status_code, 201, r.text[:200])
        surfaces = [e.payload.get("surface") for e in self.services.experience.read() if e.kind == "visit.started"]
        self.assertTrue({"dclab-code", "dclab-code:relay"} <= set(surfaces), surfaces)
        self.assertNotIn("made-up", surfaces)  # only the surfaces DCLab Code names
        self.call("home_list", {})  # through /mcp the header changes nothing: an MCP call stays the chat's
        self.assertNotIn("dclab-code", [e.payload.get("surface") for e in self.services.experience.read() if e.kind == "visit.started"][len(surfaces):])


class LongWriteTests(unittest.TestCase):
    def test_a_write_longer_than_the_wait_goes_on_and_the_chat_is_told_how_to_follow_it(self):
        from dclab_rnd import chat_agents

        finished = []

        class Done:
            status_code = 200
            headers = {"content-type": "application/json"}
            content = b'{"status": "done"}'

            def json(self):
                return {"status": "done"}

        async def slow(*args, **kwargs):
            await asyncio.sleep(0.6)
            finished.append(args[2])
            return Done()

        async def scenario():
            with mock.patch.object(chat_agents, "_request", slow), mock.patch.object(chat_agents, "_template", lambda s, m, p: "/api/projects/{project_id}/team"), \
                    mock.patch.object(chat_agents, "CALL_WAIT_S", 0.1):
                out = await chat_agents.dclab_call(object(), "POST", "/api/projects/p1/team", body={})
                self.assertTrue(out["running"], out)
                self.assertEqual(out["status"], 202)
                self.assertIn("GET /api/projects/p1/team", out["says"])
                self.assertIn("never send it again", out["says"])
                self.assertEqual(finished, [], "still working when the chat was answered")
                await asyncio.sleep(1.0)
                self.assertEqual(finished, ["/api/projects/p1/team"], "the walk went on after the answer")
                quick = await chat_agents.dclab_call(object(), "GET", "/api/projects/p1/team")
                self.assertEqual(quick["result"], {"status": "done"}, "a read is waited for")

        asyncio.run(scenario())


class ScrubTests(unittest.TestCase):
    def test_secrets_are_hidden_at_any_depth(self):
        from dclab_rnd import chat_agents

        got = chat_agents._scrub({"csrf": "x", "a": {"api_key": "y", "list": [{"Token": "z", "n": 1}]}, "name": "ok"})
        self.assertEqual(got, {"csrf": "(hidden)", "a": {"api_key": "(hidden)", "list": [{"Token": "(hidden)", "n": 1}]}, "name": "ok"})


class ChatBuildRuleTests(McpChatCase):
    """Every build during an external chat's call goes through home_build: the Home agent's and the orchestrator's alike."""

    def test_build_from_chat_refuses_inside_a_chats_call_unless_home_build_granted_it(self):
        from dclab_rnd import experience

        draft = self.services.drafts.create("Predict which shipments arrive late next week", None)["id"]
        with experience.via("mcp"):
            refused = self.services.build_from_chat(draft)
            self.assertEqual(refused.get("rule"), "chat_build", refused)
            with experience.chat_build():
                granted = self.services.build_from_chat(draft)
        self.assertNotEqual(granted.get("rule"), "chat_build", granted)  # past the chat rule: the build's own checks speak (no table yet)
        self.assertNotEqual(self.services.build_from_chat(draft).get("rule"), "chat_build")  # the page's own build is untouched


    def test_a_build_the_person_approved_on_a_card_that_went_stale_is_refused(self):
        """A host shows the person the draft before a build (DCLab Code's card) and sends what it showed: a draft changed
        since then is never built on that yes."""
        draft = self.services.drafts.create("Predict which shipments arrive late next week", None)["id"]
        d = self.services.drafts.get(draft)
        d["understanding"] = {"target": "late", "prediction_moment": "at dispatch"}
        self.services.drafts.save(d)
        stale = self.call("home_build", {"draft_id": draft, "person_words": "build it", "shown": {"target": "churn", "prediction_moment": "at dispatch"}})
        self.assertIn("changed since", stale.get("error", ""), stale)
        self.assertIn("target", stale["error"])
        fresh = self.call("home_build", {"draft_id": draft, "person_words": "build it", "shown": {"target": "late", "prediction_moment": "at dispatch", "costs": None}})
        self.assertNotIn("changed since", fresh.get("error", "") or "", fresh)  # past the check: the build's own checks speak


class PathRuleTests(McpChatCase):
    def test_folder_identity_names_and_held_out(self):
        from dclab_rnd import chat_agents

        folder = Path(tempfile.mkdtemp())
        ok = {}
        for name in ("kaggle_house_prices.csv", "tokens.parquet", "secretary_survey.csv"):
            (folder / name).write_text("a\n1\n", encoding="utf-8")
            ok[name] = chat_agents._path_refusal(self.services, folder / name)
        self.assertEqual(ok, {n: None for n in ok}, "ordinary tables are not keys")
        for name in ("kaggle.json", "credentials.json", "id_rsa", "server.pem"):
            (folder / name).write_text("x", encoding="utf-8")
            self.assertIn("key or a credential", chat_agents._path_refusal(self.services, folder / name) or "", name)
        for name in ("held_out.csv", "heldout_rows.csv"):
            (folder / name).write_text("a\n1\n", encoding="utf-8")
            self.assertIn("sealed", chat_agents._path_refusal(self.services, folder / name) or "", name)
        inside = self.workspace / "drafts" / "x.csv"
        inside.parent.mkdir(parents=True, exist_ok=True)
        inside.write_text("a\n1\n", encoding="utf-8")
        cased = Path(str(inside).replace("/drafts/", "/DRAFTS/"))
        if cased.exists():  # a file system that ignores letter case (macOS): the same folder by another spelling
            self.assertIn("workspace", chat_agents._path_refusal(self.services, cased) or "")


class CapabilitiesOverMcpTests(McpChatCase):
    """A7.3: every operation of the product, listed from its own API description and run inside the same app."""

    def test_the_chat_lists_and_searches_every_operation(self):
        everything = self.call("dclab_capabilities", {"limit": 100})
        self.assertGreater(everything["matching"], 150)  # the whole product API, not a hand-picked list
        found = self.call("dclab_capabilities", {"query": "lineage"})["operations"]
        self.assertTrue(found and "lineage" in found[0]["path"], found[:2])
        consent = [o for o in self.call("dclab_capabilities", {"query": "consent", "method": "PUT"})["operations"] if o["path"] == "/api/consent"]
        self.assertEqual([o["chat"] for o in consent], ["the person's, on its page"])
        rows = [o for o in self.call("dclab_capabilities", {"query": "rows", "method": "GET"})["operations"] if o["path"].endswith("/rows")]
        self.assertTrue(rows and all(o["chat"].startswith("never") for o in rows), rows)
        asks = [o for o in self.call("dclab_capabilities", {"query": "evidence ask", "method": "POST"})["operations"] if o["path"] == "/api/evidence/ask"]
        self.assertEqual([o["chat"] for o in asks], ["runs"])

    def test_an_operation_runs_with_the_pages_checks_and_its_background_work_goes_on(self):
        samples = self.call("dclab_call", {"method": "GET", "path": "/api/samples"})
        self.assertTrue(samples["ok"], samples)
        self.assertIsInstance(samples["result"], list)
        made = self.call("dclab_call", {"method": "POST", "path": "/api/drafts", "body": {"problem": "Predict which shipments arrive late"}})
        self.assertEqual(made["status"], 201, made)
        draft = made["result"]["id"]
        deadline = time.monotonic() + 30  # the route's background turn runs on the server's loop after the call returned
        while time.monotonic() < deadline:
            got = self.call("dclab_call", {"method": "GET", "path": f"/api/drafts/{draft}"})["result"]
            if any(m["role"] == "agent" for m in got.get("messages") or []):
                break
            time.sleep(0.3)
        self.assertTrue(any(m["role"] == "agent" for m in got.get("messages") or []), "the Home agent answered the draft made through the bridge")
        bad = self.call("dclab_call", {"method": "POST", "path": "/api/drafts", "body": {"problem": "x"}})
        self.assertEqual(bad["status"], 422)  # the route's own check

    def test_the_persons_operations_and_odd_paths_are_refused(self):
        project = self.call("create_project", {"name": "Gate", "goal": "Predict heart disease from the visit's measurements"})["project_id"]
        draft = self.call("dclab_call", {"method": "POST", "path": "/api/drafts", "body": {"problem": "Predict which shipments arrive late"}})["result"]["id"]
        persons = (("PUT", "/api/consent", {"use": "train", "on": True}), ("POST", "/api/models/routing", {}), ("POST", "/api/auth/tokens", {}),
                   ("PUT", "/api/twin/settings", {"on": True}), ("POST", f"/api/projects/{project}/approvals", {"gate": "solution"}),
                   ("PATCH", f"/api/projects/{project}", {"policy": {"require_holdout_approval": False}}),
                   ("POST", f"/api/projects/{project}/stages/data/approve", {}), ("POST", f"/api/projects/{project}/run", {}),
                   ("POST", "/api/decide/choose", {}), ("POST", "/api/feedback", {}), ("DELETE", f"/api/projects/{project}", None),
                   ("POST", f"/api/drafts/{draft}/messages", {"text": "build it"}), ("POST", f"/api/drafts/{draft}/build/run", {}),
                   ("PUT", f"/api/drafts/{draft}/solution", {"target": "x"}), ("POST", f"/api/projects/{project}/team", {"approve": True}),
                   ("POST", f"/api/projects/{project}/team", {"answer": "drop it, no data"}),  # the team's drop question is the person's
                   ("POST", f"/api/drafts/{draft}/team", {"approve": True}), ("POST", f"/api/drafts/{draft}/team", {"answer": "late orders"}),
                   ("POST", f"/api/drafts/{draft}/map", {"redraw": True}))  # a redraw drops the person's edits and confirmation
        for method, path, body in persons:
            with self.subTest(f"{method} {path}"):
                self.assertIn("person's", self.call("dclab_call", {"method": method, "path": path, "body": body}).get("error", ""))
        for path in ("/api/%63onsent", "/api/consent#x", "/api/twin/settings%73", "/api/consent?x=1", "/api//consent",
                     f"/api/drafts/./{draft}", "/api/./consent"):  # never matched by the raw string; a dot segment is dropped by the client
            with self.subTest(path):
                self.assertIn("as written", self.call("dclab_call", {"method": "PUT", "path": path, "body": {"use": "train", "on": True}}).get("error", ""))
        self.assertFalse(self.services.projects.get(project).get("approvals"), "no gate approved through the chat")
        for path in (f"/api/drafts/{draft}/events", f"/api/projects/{project}/export/code", f"/api/drafts/{draft}/assets/a1/rows"):
            with self.subTest(path):
                self.assertIn("never reads", self.call("dclab_call", {"method": "GET", "path": path}).get("error", ""))
        self.assertIn("error", self.call("dclab_call", {"method": "GET", "path": "/mcp"}))
        self.assertIn("fill every", self.call("dclab_call", {"method": "GET", "path": "/api/projects/{project_id}"}).get("error", ""))
        self.assertIn("error", self.call("dclab_call", {"method": "GET", "path": "/api/../etc/passwd"}))
        self.assertIn("No GET", self.call("dclab_call", {"method": "GET", "path": "/api/nothing-here"}).get("error", ""))
        self.assertTrue(self.call("dclab_call", {"method": "GET", "path": "/api/consent"})["ok"])  # reading a switch is fine
        # The page's request token never reaches a chat (with it a chat could act as the person through another door)
        self.assertIn("never reads", self.call("dclab_call", {"method": "GET", "path": "/api/config"}).get("error", ""))
        self.assertNotIn(self.services.csrf, json.dumps([self.call("dclab_call", {"method": "GET", "path": p}) for p in ("/api/samples", "/api/consent", "/api/workspace")]))
        # Whatever the routes record during the chat's calls is the agent's, in the chat's visit: never a person's page visit.
        rows = self.services.audit.query(500, 0)[1]
        self.assertTrue(rows, "the routes' audit rows")
        self.assertNotIn("human", {r.get("actor") for r in rows})
        visits = {e.payload.get("surface") for e in self.services.experience.read() if e.kind == "visit.started"}
        self.assertEqual(visits, {"mcp"}, visits)
        from dclab_rnd import experience

        persons = [e.kind for e in self.services.experience.read() if e.actor == "person" and e.kind not in experience.RELAYED]
        self.assertEqual(persons, [], "no route a chat ran recorded a person's choice")

    def test_a_chat_starts_a_team_and_only_the_person_answers_or_approves_it(self):
        from dclab_rnd.chat_agents import _refusal_of

        for template in ("/api/drafts/{draft_id}/team", "/api/projects/{project_id}/team"):
            with self.subTest(template):
                self.assertIsNone(_refusal_of("POST", template, {}), "starting the walk is agent work: it pauses before any build or holdout")
                for body in ({"approve": True}, {"approve": False}, {"answer": "x"}):
                    self.assertIn("person's", _refusal_of("POST", template, body) or "")

    def test_a_long_result_is_kept_in_the_workspace(self):
        from dclab_rnd import chat_agents

        with mock.patch.object(chat_agents, "RESULT_MAX", 200):
            got = self.call("dclab_call", {"method": "GET", "path": "/api/samples"})
        self.assertTrue(got["truncated"], got)
        self.assertTrue(Path(got["saved"]).is_file())
        self.assertEqual(Path(got["saved"]).parent, self.workspace / "mcp" / "results")


class WorkspaceTablesOverMcpTests(McpChatCase):
    """A7.4: the workspace chat's read tools (30.21), masked as for its own agent; no route serves them."""

    def test_tables_described_and_computed_never_a_private_projects(self):
        self.need_sample("heart_disease")
        open_one = self.call("create_project", {"name": "Heart", "goal": "Predict heart disease from the visit's measurements"})["project_id"]
        self.call("use_sample", {"project_id": open_one, "key": "heart_disease"})
        private = self.call("create_project", {"name": "Private", "goal": "Predict heart disease, a private copy"})["project_id"]
        self.call("use_sample", {"project_id": private, "key": "heart_disease"})
        settings = {**(self.services.projects.get(private).get("settings") or {}), "private": True}
        self.services.projects.update(private, settings=settings)
        tables = self.call("list_tables", {})["tables"]
        names = [t["table"] for t in tables]
        self.assertTrue(any(n.startswith(open_one + ":") for n in names), names)
        self.assertFalse(any(n.startswith(private + ":") for n in names), "a private project's table is never offered")
        table = next(t for t in tables if t["table"].startswith(open_one + ":"))
        described = self.call("describe_table", {"table": table["table"]})
        self.assertEqual(described["rows"], table["rows"])
        self.assertNotIn("values", json.dumps(described))  # kinds, shares and counts: never a value
        counted = self.call("compute", {"table": table["table"], "op": "count"})
        self.assertEqual(json.dumps(counted).count(str(table["rows"])) >= 1, True, counted)
        self.assertIn("error", self.call("compute", {"table": "nope", "op": "count"}))


class LongAgentsOverMcpTests(McpChatCase):
    """A7.4: an intern job started over MCP finishes; a team walk started over MCP pauses for the person, resumes on the
    person's own answer (never the chat's), and is followed over MCP."""

    def project(self):
        self.need_sample("bank_marketing")
        pid = self.call("create_project", {"name": "Long agents", "goal": "Predict which clients subscribe to the term deposit"})["project_id"]
        self.call("use_sample", {"project_id": pid, "key": "bank_marketing"})
        return pid

    def person(self):
        return {"x-dclab-token": self.services.csrf}  # the page's own request: the person, never the chat

    def saved(self, pid):
        """The person saves the proposed solution on the page (the team runs from a saved solution, the person's)."""
        proposal = self.client.post(f"/api/projects/{pid}/solution/proposal", json={"target": "target"}, headers=self.person())
        self.assertEqual(proposal.status_code, 200, proposal.text[:300])
        p = proposal.json()
        solution = {"target": "target", "task": p.get("task") or "binary", "positive_label": str(p.get("positive_label") or "1"),
                    "prediction_moment": "Before the campaign call, from what the bank knows about the client",
                    "forbidden": [{"column": f["column"], "reason": f.get("reason") or "known only after the call"} for f in p.get("forbidden") or []],
                    "identifiers": p.get("identifiers") or [], "time_column": None, "group_column": None, "text_columns": [],
                    "metric": p.get("metric"), "notes": ""}
        saved = self.client.put(f"/api/projects/{pid}/solution", json=solution, headers=self.person())
        self.assertEqual(saved.status_code, 200, saved.text[:300])

    def test_an_intern_job_started_over_mcp_finishes(self):
        pid = self.project()
        got = self.call("dclab_call", {"method": "POST", "path": "/api/intern/sessions", "query": {"wait": True},
                                       "body": {"task": "Propose a leakage-safe solution for this project", "project_id": pid}})
        session = got.get("body") or got.get("result") or got
        if got.get("running"):  # still working after the call's wait (a loaded machine): follow it as the reply says
            self.assertIn("GET /api/intern/sessions", got["says"])
            for _ in range(180):
                listed = self.call("dclab_call", {"method": "GET", "path": "/api/intern/sessions"})
                mine = [x for x in (listed.get("body") or listed.get("result") or []) if x.get("project_id") == pid]
                if mine and mine[0].get("status") not in ("queued", "running"):
                    session = mine[0]
                    break
                time.sleep(5)
        self.assertIn(session.get("status"), ("finished", "completed", "done", "budget_exhausted"), str(got)[:300])
        again = self.call("dclab_call", {"method": "GET", "path": f"/api/intern/sessions/{session['id']}"})
        self.assertEqual((again.get("body") or again.get("result") or again).get("status"), session.get("status"))

    def test_a_team_walk_started_over_mcp_pauses_and_resumes_on_the_persons_answer(self):
        pid = self.project()
        self.saved(pid)
        started = self.call("dclab_call", {"method": "POST", "path": f"/api/projects/{pid}/team", "body": {}})
        first = started.get("body") or started.get("result") or started
        waiting = first.get("asks") if first.get("status") == "paused" else None
        if started.get("running"):  # still working after the call's wait (a loaded machine): follow the board as the reply says
            self.assertIn(f"GET /api/projects/{pid}/team", started["says"])
            for _ in range(180):
                seen = self.call("dclab_call", {"method": "GET", "path": f"/api/projects/{pid}/team"})
                view = seen.get("body") or seen.get("result") or seen
                if isinstance(view.get("waiting"), dict) and view["waiting"].get("key"):
                    waiting = view["waiting"]
                    break
                time.sleep(5)
        self.assertTrue(waiting, f"the walk did not pause for the person: {str(first)[:300]}")
        refused = self.call("dclab_call", {"method": "POST", "path": f"/api/projects/{pid}/team", "body": {"approve": True}})
        self.assertIn("person", json.dumps(refused))  # the chat never answers or approves for the person
        answer = {"approve": True} if waiting.get("kind") == "approval" else {"answer": "yes"}
        resumed = self.client.post(f"/api/projects/{pid}/team", json=answer, headers=self.person())
        self.assertEqual(resumed.status_code, 200, resumed.text[:300])
        after = resumed.json()
        self.assertNotEqual((after.get("asks") or {}).get("key"), waiting.get("key"))  # past that pause: done, or the next one
        seen = self.call("dclab_call", {"method": "GET", "path": f"/api/projects/{pid}/team"})
        view = seen.get("body") or seen.get("result") or seen
        self.assertNotEqual(((view.get("waiting") or {}) if isinstance(view.get("waiting"), dict) else {}).get("key"), waiting.get("key"))


class ChatConversationsKeptApartTests(unittest.TestCase):
    """A conversation an external chat started is not a Home conversation: its "person" turns are that chat model's."""

    def test_the_conversations_set_leaves_a_chat_started_draft_out(self):
        from dclab_core import datasets

        def events(conversation, surface):
            created = {"problem": "Predict churn next month", "pack": None, **({"surface": surface} if surface else {})}
            return [{"id": f"{conversation}1", "kind": "draft.created", "actor": "person", "payload": created, "ids": {"conversation": conversation}},
                    {"id": f"{conversation}2", "kind": "chat.agent", "actor": "agent", "payload": {"text": "Which column is the outcome?", "surface": surface or "home"},
                     "ids": {"conversation": conversation}},
                    {"id": f"{conversation}3", "kind": "draft.built", "actor": "system", "payload": {}, "ids": {"conversation": conversation}}]

        rows, _ = datasets.conversations(events("cHome", None) + events("cMcp", "mcp") + events("cPage", "home"))
        self.assertEqual(sorted(r["conversation"] for r in rows), ["cHome", "cPage"])
        turns, _ = datasets.episodes([dict(e, at="2026-10-25T10:00:0%d+00:00" % i) for i, e in enumerate(events("cMcp", "mcp") + events("cHome", None))], text=True)
        said = {t["conversation"]: t["message"] for t in turns}
        self.assertEqual(said, {"cMcp": None, "cHome": "Predict churn next month"})  # the chat model's words are never a person's message

    def test_the_intern_and_project_chats_mark_a_chats_words(self):
        from dclab_rnd import experience

        self.assertEqual(experience.chat_surface("intern"), "intern")
        with experience.via("mcp"):
            self.assertEqual((experience.chat_surface("intern"), experience.chat_surface("project")), ("mcp", "mcp"))


if __name__ == "__main__":
    unittest.main()
