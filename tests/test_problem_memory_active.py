"""Packages A3, A4, A5, and A6 (Agentic_Plan_and_Context_Plan_2026-11-20):
- A3: Problem identity, durable bindings, host-resolved authorized scope (two people, two workspaces, two projects,
      two sibling problems, two sessions, reused display names, guessed IDs, unbound draft, missing/revoked worker
      identity, legacy v1 reads).
- A4: Editable Project & Problem bases with optimistic revision conflict (409), idempotent material events,
      provisional drafts that retries never activate, corrections & retirements invalidating dependents.
- A5: Active versioned resumable summary with CAS cursor publish rejecting stale worker builds, viewer-only
      personal preference overlay, and source purge removing summaries and cached search vectors.
- A6: Full 4-step Dana delivery-ETA acceptance story across Home, Lead, Advisor, API routes, delayed Advisor
      response after correction, advisory-only gated roles toggle, and legacy behavior with switch off.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import case as C, memory as core_mem  # noqa: E402
from dclab_rnd.accounts.principal import ANONYMOUS, Principal, acting  # noqa: E402
from dclab_rnd.agentic.server import create_app  # noqa: E402
from dclab_rnd.agents import case_agent, decision_loop, default_registry  # noqa: E402
from dclab_rnd.core_advisor import GatewayAdvisor, check_recommendation_freshness  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402
from dclab_rnd.experience import retention  # noqa: E402
from dclab_rnd.memory import FileMemory, bases, open_memory, scope as mem_scope, summary as mem_summary, view  # noqa: E402
from dclab_rnd.settings import Settings  # noqa: E402
from dclab_rnd.studio.store import ProjectStore  # noqa: E402


class _FakeClient:
    def __init__(self, reply, *, on_call=None):
        self.reply = reply
        self.on_call = on_call
        self.calls = 0
        self.model = "fake-advisor"

    def complete(self, messages, tools=None, max_tokens=None):
        self.calls += 1
        if self.on_call is not None:
            self.on_call()
        if isinstance(self.reply, list):
            return self.reply.pop(0)
        return self.reply


class _FakeGateway:
    def __init__(self, clients):
        self.clients = dict(clients)

    def client(self, purpose, **kw):
        return self.clients.get(purpose)

    def available(self, purpose):
        return purpose in self.clients


class A3IdentityAndScopeTests(unittest.TestCase):
    def test_legacy_v1_memory_item_reads_and_preserves_id_and_correction_link(self):
        legacy_id = core_mem.item_id("fact", "project", "p1", "Target is churn within 30 days.")
        v1_doc = {
            "schema_version": "dclab_core.memory_item.v1",
            "id": legacy_id,
            "kind": "fact",
            "scope": "project",
            "scope_id": "p1",
            "text": "Target is churn within 30 days.",
            "source": "note:p1:n1",
            "who": "person:owner",
            "status": "active",
            "correction": False,
            "created_at": "2026-10-28T09:00:00+00:00",
        }
        loaded = core_mem.MemoryItem.read(v1_doc)
        self.assertEqual(loaded.id, legacy_id)
        self.assertEqual(loaded.schema_version, "dclab_core.memory_item.v2")
        fixed = core_mem.new(kind="fact", scope="project", scope_id="p1", text="Target is churn within 14 days.",
                             source=f"correction:{legacy_id}", who="person:owner", at="2026-10-29T09:00:00+00:00",
                             correction=True, supersedes=legacy_id)
        shown = core_mem.resolve([loaded, fixed], [("project", "p1")])
        self.assertEqual([i.id for i in shown], [fixed.id])

    def test_isolation_across_people_workspaces_projects_siblings_sessions_and_reused_names(self):
        home = Path(tempfile.mkdtemp())
        projects = ProjectStore(home / "projects")
        drafts = DraftStore(home / "drafts")
        store = FileMemory(home)

        # Two workspaces, two projects with the SAME display name, two sibling problems with the SAME display name.
        p_ws1 = projects.create("Delivery ETA", goal="Predict delivery time")
        projects.update(p_ws1["id"], workspace_id="ws1", members=["dana", "colleague"])
        p_ws2 = projects.create("Delivery ETA", goal="Predict delivery time")
        projects.update(p_ws2["id"], workspace_id="ws2", members=["outsider"])

        prob_a = mem_scope.ensure_default_problem(projects, p_ws1["id"], workspace_id="ws1", by="dana", name="ETA")
        prob_b = mem_scope.create_problem(projects, p_ws1["id"], "ETA", workspace_id="ws1", by="dana")  # reused display name
        prob_ws2 = mem_scope.ensure_default_problem(projects, p_ws2["id"], workspace_id="ws2", by="outsider", name="ETA")
        self.assertNotEqual(prob_a["problem_id"], prob_b["problem_id"])

        dana = Principal(role="data_scientist", workspace_id="ws1", user_id="dana", via="session", workspaces=(("ws1", "WS1", "data_scientist"),))
        colleague = Principal(role="data_scientist", workspace_id="ws1", user_id="colleague", via="session", workspaces=(("ws1", "WS1", "data_scientist"),))
        outsider = Principal(role="data_scientist", workspace_id="ws2", user_id="outsider", via="session", workspaces=(("ws2", "WS2", "data_scientist"),))
        ws1_non_member = Principal(role="data_scientist", workspace_id="ws1", user_id="stranger", via="session", workspaces=(("ws1", "WS1", "data_scientist"),))

        # Ambiguous project with two active problems requires explicit problem selection.
        with self.assertRaises(mem_scope.AmbiguousProblemError) as amb:
            mem_scope.resolve_scope(workspace_id="ws1", principal=dana, projects=projects, project_id=p_ws1["id"], require_problem=True)
        self.assertEqual(len(amb.exception.candidates), 2)

        # Guessed cross-project problem_id fails closed.
        with self.assertRaises(mem_scope.AccessDenied):
            mem_scope.resolve_scope(workspace_id="ws1", principal=dana, projects=projects, project_id=p_ws1["id"],
                                    problem_id=prob_ws2["problem_id"], require_problem=True)

        # Outsider workspace and non-member in same workspace fail closed.
        with self.assertRaises(mem_scope.AccessDenied):
            mem_scope.resolve_scope(workspace_id="ws1", principal=outsider, projects=projects, project_id=p_ws1["id"],
                                    problem_id=prob_a["problem_id"], require_problem=True)
        with self.assertRaises(mem_scope.AccessDenied):
            mem_scope.resolve_scope(workspace_id="ws1", principal=ws1_non_member, projects=projects, project_id=p_ws1["id"],
                                    problem_id=prob_a["problem_id"], require_problem=True)

        # Unbound draft cannot access project/problem memory.
        unbound = drafts.create("Draft without project")
        with self.assertRaises(mem_scope.AccessDenied):
            mem_scope.resolve_scope(workspace_id="ws1", principal=dana, projects=projects, drafts=drafts,
                                    draft_id=unbound["id"], project_id=p_ws1["id"])

        # Resolve scopes for Dana (prob A, session 1), Dana (prob B, session 2), Colleague (prob A, session 2).
        s_dana_a1 = mem_scope.resolve_scope(workspace_id="ws1", principal=dana, projects=projects,
                                            project_id=p_ws1["id"], problem_id=prob_a["problem_id"], session_id="sess1", require_problem=True)
        s_dana_b2 = mem_scope.resolve_scope(workspace_id="ws1", principal=dana, projects=projects,
                                            project_id=p_ws1["id"], problem_id=prob_b["problem_id"], session_id="sess2", require_problem=True)
        s_coll_a2 = mem_scope.resolve_scope(workspace_id="ws1", principal=colleague, projects=projects,
                                            project_id=p_ws1["id"], problem_id=prob_a["problem_id"], session_id="sess2", require_problem=True)

        # Store personal, project, sibling problem, and session items.
        at = "2026-11-19T09:00:00+00:00"
        store.write(core_mem.new(kind="fact", scope="person", scope_id=s_dana_a1.person_key, text="Dana private note: show brief tables.", source="pref:1", who="person:dana", at=at))
        store.write(core_mem.new(kind="fact", scope="project", scope_id=s_dana_a1.project_key, text="Shared business glossary: SLA is 45m.", source="proj:1", who="person:dana", at=at))
        store.write(core_mem.new(kind="fact", scope="problem", scope_id=s_dana_a1.problem_key, text="Problem A prediction moment: after dispatch.", source="prob:a", who="person:dana", at=at))
        store.write(core_mem.new(kind="fact", scope="problem", scope_id=s_dana_b2.problem_key, text="Problem B prediction moment: at order creation.", source="prob:b", who="person:dana", at=at))
        store.write(core_mem.new(kind="fact", scope="session", scope_id=s_dana_a1.session_key, text="Session 1 scratch note.", source="sess:1", who="person:dana", at=at))

        # Positive & negative read checks.
        dana_a1_texts = {i.text for i in view(store, s_dana_a1.chain(include_personal=True, include_session=True))}
        self.assertIn("Dana private note: show brief tables.", dana_a1_texts)
        self.assertIn("Problem A prediction moment: after dispatch.", dana_a1_texts)
        self.assertIn("Session 1 scratch note.", dana_a1_texts)
        self.assertNotIn("Problem B prediction moment: at order creation.", dana_a1_texts)

        coll_a2_texts = {i.text for i in view(store, s_coll_a2.chain(include_personal=True, include_session=True))}
        self.assertIn("Shared business glossary: SLA is 45m.", coll_a2_texts)
        self.assertIn("Problem A prediction moment: after dispatch.", coll_a2_texts)
        self.assertNotIn("Dana private note: show brief tables.", coll_a2_texts)
        self.assertNotIn("Session 1 scratch note.", coll_a2_texts)

        dana_b2_texts = {i.text for i in view(store, s_dana_b2.chain(include_personal=True, include_session=True))}
        self.assertIn("Problem B prediction moment: at order creation.", dana_b2_texts)
        self.assertNotIn("Problem A prediction moment: after dispatch.", dana_b2_texts)

    def test_worker_job_revalidation_fails_closed_when_missing_or_revoked(self):
        home = Path(tempfile.mkdtemp())
        projects = ProjectStore(home / "projects")
        p = projects.create("Delivery", goal="ETA")
        projects.update(p["id"], workspace_id="ws1")
        prob = mem_scope.ensure_default_problem(projects, p["id"], workspace_id="ws1", by="dana")
        env = type("Env", (), {"projects": projects, "drafts": None, "workspace_id": "ws1"})()

        # Missing worker principal under auth mode fails closed.
        with mock.patch.dict(os.environ, {"DCLAB_AUTH": "password"}):
            with self.assertRaises(mem_scope.AccessDenied):
                mem_scope.revalidate_job_scope(env, {"payload": {"project_id": p["id"], "problem_id": prob["problem_id"]}})

            # Active member succeeds; revoked member fails closed immediately.
            job = {"user": {"user_id": "dana", "role": "data_scientist", "workspace_id": "ws1"},
                   "payload": {"project_id": p["id"], "problem_id": prob["problem_id"]}}
            ok_scope = mem_scope.revalidate_job_scope(env, job, require_problem=True,
                                                      membership_checker=lambda u, w: "data_scientist" if u == "dana" else None)
            self.assertEqual((ok_scope.principal_id, ok_scope.problem_id), ("dana", prob["problem_id"]))

            with self.assertRaises(mem_scope.AccessDenied):
                mem_scope.revalidate_job_scope(env, job, require_problem=True, membership_checker=lambda u, w: None)


class A4A5BasesAndSummaryTests(unittest.TestCase):
    def _setup(self):
        home = Path(tempfile.mkdtemp())
        projects = ProjectStore(home / "projects")
        p = projects.create("Delivery", goal="ETA")
        prob = mem_scope.ensure_default_problem(projects, p["id"], workspace_id="local", by="dana")
        sc = mem_scope.AuthorizedScope(workspace_id="local", principal_id="dana", role="data_scientist",
                                       project_id=p["id"], problem_id=prob["problem_id"])
        return home, projects, p["id"], prob["problem_id"], sc

    def test_optimistic_conflict_idempotent_events_and_provisional_drafts(self):
        home, _projects, _pid, _prob, sc = self._setup()

        r1 = bases.upsert_base_fact(home, sc, target="problem", subject="problem", field="prediction_moment",
                                    text="Prediction moment: after dispatch.", source="home:turn1", expected_revision=0)
        self.assertEqual(r1["revision"], 1)

        # Racing edit with stale expected_revision=0 raises RevisionConflictError exposing both edits.
        with self.assertRaises(bases.RevisionConflictError) as conflict:
            bases.upsert_base_fact(home, sc, target="problem", subject="problem", field="prediction_moment",
                                   text="Prediction moment: at order creation.", source="ui:edit2", expected_revision=0)
        detail = conflict.exception.to_detail()
        self.assertEqual(detail["current_revision"], 1)
        self.assertEqual(detail["attempted"]["text"], "Prediction moment: at order creation.")

        # Project base edit stays independent from problem base revision.
        rp = bases.upsert_base_fact(home, sc, target="project", subject="glossary", field="sla",
                                    text="Delivery SLA target is 45 minutes.", source="ui:proj", expected_revision=0)
        self.assertEqual(rp["revision"], 1)
        self.assertEqual(bases.base_revision(home, sc, "problem"), 1)

        # Material event delivered twice yields one entry; greetings are skipped.
        ev1 = bases.record_material_event(home, sc, source_type="decision", source_id="d1", source_revision="1",
                                          text="Excluded courier_rating because it is recorded after delivery.", outcome_status="succeeded")
        ev2 = bases.record_material_event(home, sc, source_type="decision", source_id="d1", source_revision="1",
                                          text="Excluded courier_rating because it is recorded after delivery.", outcome_status="succeeded")
        self.assertFalse(ev1["replayed"])
        self.assertTrue(ev2["replayed"])
        self.assertEqual(ev1["entry"]["entry_id"], ev2["entry"]["entry_id"])
        greet = bases.record_material_event(home, sc, source_type="chat", source_id="c1", source_revision="1", text="thanks!")
        self.assertTrue(greet["skipped"])

        # Advisor proposal stays provisional draft even on retry; only explicit confirmation activates it.
        d1 = bases.propose_draft(home, sc, subject="features", field="policy",
                                 text="Keep dispatch_delay_min because prediction is after dispatch.",
                                 source_type="advisor", source_id="adv1", source_revision="1", uncertainty="medium")
        d2 = bases.propose_draft(home, sc, subject="features", field="policy",
                                 text="Keep dispatch_delay_min because prediction is after dispatch.",
                                 source_type="advisor", source_id="adv1", source_revision="1", uncertainty="medium")
        self.assertEqual(d1["entry"]["status"], "draft")
        self.assertEqual(d2["entry"]["status"], "draft")
        self.assertTrue(d2["replayed"])

        # Correction marks the draft stale and blocks un-rechecked confirmation!
        bases.correct_base_fact(home, sc, r1["entry"]["entry_id"], "Prediction moment: at order creation.",
                                expected_revision=1, reason="corrected by Dana")
        with self.assertRaises(bases.StaleDependencyError) as stale_err:
            bases.confirm_draft(home, sc, d1["entry"]["entry_id"], action="accept")
        self.assertTrue(any(c["field"] == "prediction_moment" for c in stale_err.exception.changed_sources))

    def test_summary_cas_rejects_stale_worker_and_purge_clears_vectors_and_summary(self):
        home, _projects, pid, prob_id, sc = self._setup()
        r1 = bases.upsert_base_fact(home, sc, target="problem", subject="problem", field="prediction_moment",
                                    text="Prediction moment: after dispatch.", source="home:1", expected_revision=0)

        # Slow worker captures snapshot BEFORE correction arrives.
        slow_snap = mem_summary.capture_snapshot(home, sc)
        slow_built = mem_summary.build_from_snapshot(slow_snap)

        # Concurrent correction lands while worker is building.
        bases.correct_base_fact(home, sc, r1["entry"]["entry_id"], "Prediction moment: at order creation.",
                                expected_revision=1, reason="Dana corrected moment")

        # Stale worker tries to publish -> REJECTED by CAS cursor check!
        pub = mem_summary.publish_summary(home, sc, slow_built)
        self.assertFalse(pub["published"])
        self.assertEqual(pub["reason"], "stale_source_cursor")

        # Later reader immediately sees the correction.
        fresh = mem_summary.read_summary(home, sc, auto_rebuild=True)
        self.assertTrue(any("at order creation" in f["text"] for f in fresh["active_facts"]))
        self.assertFalse(any("after dispatch" in f["text"] for f in fresh["active_facts"]))

        # Seed a fake cached search vector for the fact text and purge project -> both summary & vector vanish.
        import hashlib
        vec_file = home / "memory" / "vectors-test.json"
        key = hashlib.sha256("title: none | text: \0Prediction moment: at order creation.".encode()).hexdigest()[:24]
        vec_file.write_text(json.dumps({key: [0.1, 0.2], "keep_other": [0.9, 0.8]}), encoding="utf-8")

        retention.forget(project=pid, log=type("L", (), {"home": home, "workspace": "local"})(), ledger=None)
        rebuilt = mem_summary.read_summary(home, sc, auto_rebuild=True)
        self.assertEqual(rebuilt["active_facts"], [])
        remaining_vecs = json.loads(vec_file.read_text(encoding="utf-8"))
        self.assertNotIn(key, remaining_vecs)
        self.assertIn("keep_other", remaining_vecs)


class A6AcceptanceStoryEndToEndTests(unittest.TestCase):
    def test_dana_four_step_acceptance_story_delayed_advisor_and_switch_toggle(self):
        home = Path(tempfile.mkdtemp())
        app = create_app(home)
        client = TestClient(app)
        s = app.state.services
        decision_loop.set_settings(home, enabled=True, advisory_gated_roles=True)

        # Create project with Dana & Colleague as members (outsider excluded).
        p = s.projects.create("Delivery ETA", goal="Predict actual_minutes for food orders")
        s.projects.update(p["id"], workspace_id="local", members=["dana", "colleague"])

        dana = Principal(role="data_scientist", workspace_id=None, user_id="dana", via="token", workspaces=(("local", "Local", "data_scientist"),))
        colleague = Principal(role="data_scientist", workspace_id=None, user_id="colleague", via="token", workspaces=(("local", "Local", "data_scientist"),))
        outsider = Principal(role="data_scientist", workspace_id=None, user_id="outsider", via="token", workspaces=(("local", "Local", "data_scientist"),))

        def _as_user(who):
            return mock.patch("dclab_rnd.agentic.server.identify", return_value=who)

        # Step 1: Dana records prediction moment "after dispatch" on active problem via API route.
        with _as_user(dana):
            probs = client.get(f"/api/projects/{p['id']}/problems").json()
            prob_eta = probs["problems"][0]["problem_id"]
            up = client.post(f"/api/projects/{p['id']}/memory/base", json={
                "target": "problem", "problem_id": prob_eta, "subject": "problem",
                "field": "prediction_moment", "text": "Delivery ETA is predicted after dispatch.",
                "source": "home:dana_turn_1", "expected_revision": 0,
            })
            self.assertEqual(up.status_code, 200, up.text)
            entry_v1 = up.json()["entry"]
            # Shared project fact + Dana's private preference.
            client.post(f"/api/projects/{p['id']}/memory/base", json={
                "target": "project", "subject": "business", "field": "sla",
                "text": "On-time SLA threshold is 45 minutes.", "source": "ui:proj", "expected_revision": 0,
            })
            open_memory(home).write(core_mem.new(
                kind="fact", scope="person", scope_id="dana",
                text="Dana private preference: bullet points only.", source="pref:dana", who="person:dana",
                at="2026-11-19T08:00:00+00:00",
            ))

        # Step 2: Fresh next-day session resumes with active summary; Lead sees recorded moment without asking again.
        dana_scope = mem_scope.resolve_scope(workspace_id="local", principal=dana, projects=s.projects,
                                             project_id=p["id"], problem_id=prob_eta, require_problem=True)
        with _as_user(dana):
            sum_resp = client.get(f"/api/projects/{p['id']}/memory/summary", params={"problem_id": prob_eta})
            self.assertEqual(sum_resp.status_code, 200)
            sum_doc = sum_resp.json()
            self.assertEqual(sum_doc["resume_url"], f"/projects/{p['id']}?problem={prob_eta}")
            self.assertIn("after dispatch", sum_doc["text"])
            self.assertTrue(any("bullet points only" in x["text"] for x in sum_doc["personal_preferences"]))

        from dclab_core.replicas import delivery_time
        frame = delivery_time.generate(seed=11, n=60).orders
        table_view = {"table": "delivery_time.csv", "rows": len(frame),
                      "columns": [{"name": c, "kind": "numeric", "noticed": (["outcome-like name"] if c == "actual_minutes" else [])}
                                  for c in frame.columns]}
        reg = default_registry()
        next_day_state = case_agent.CaseState(
            case=C.new("case-resume"), table=table_view, frame=frame,
            workspace_home=home, project_id=p["id"], problem_id=prob_eta, authorized_scope=dana_scope,
        )
        menu_day2 = decision_loop.build_menu(next_day_state, reg)
        prompt_ctx = menu_day2.as_prompt_dict()["problem_context"]
        self.assertIn("after dispatch", prompt_ctx["text"])
        self.assertEqual(menu_day2.problem_revision, 1)

        # Advisor explains motivation pinned to revision 1 AND handles delayed response after Dana's correction!
        def _correct_while_advisor_runs():
            bases.correct_base_fact(
                home, dana_scope, entry_v1["entry_id"],
                "Delivery ETA is predicted at order creation (before dispatch).",
                expected_revision=1, reason="Dana corrected prediction time to order creation",
            )

        adv_reply = {"content": json.dumps({
            "recommendation": "Keep dispatch_time because prediction happens after dispatch.",
            "reason": "Problem revision 1 states prediction is after dispatch.",
            "uncertainty": "low",
            "suggested_check": "Verify dispatch_time availability.",
        }), "finish_reason": "stop"}
        advisor = GatewayAdvisor(gateway=_FakeGateway({"problem_analyst": _FakeClient(adv_reply, on_call=_correct_while_advisor_runs)}),
                                 project_id=p["id"])
        delayed_rec = advisor.advise_gated_role("problem_analyst", {"columns": list(frame.columns)},
                                                home=home, summary=sum_doc, scope=dana_scope)
        # Step 3: Delayed Advisor response immediately detected as STALE naming changed source!
        self.assertTrue(delayed_rec["provisional"])
        self.assertFalse(delayed_rec["carries_approval"])
        self.assertTrue(delayed_rec["stale"])
        self.assertFalse(delayed_rec["may_execute"])
        self.assertTrue(any(c["field"] == "prediction_moment" for c in delayed_rec["changed_sources"]))

        # Pending action built against old revision 1 CANNOT execute unchecked now that revision is 2!
        stale_attempt = decision_loop.dispatch_choice(
            next_day_state, menu_day2,
            {"choice": "inspect", "reason": "Action prepared before correction", "menu_revision": menu_day2.revision,
             "depends_on_problem_revision": 1, "arguments": {"outcome": "actual_minutes"}},
            reg,
        )
        self.assertEqual((stale_attempt.status, stale_attempt.blocked_reason, stale_attempt.committed_effects),
                         ("blocked", "stale_problem_fact", 0))
        self.assertTrue(any(c["field"] == "prediction_moment" for c in stale_attempt.result["changed_sources"]))

        # Fresh menu sees revision 2 ("at order creation") and executes recheck cleanly.
        fresh_menu = decision_loop.build_menu(next_day_state, reg)
        self.assertEqual(fresh_menu.problem_revision, 2)
        self.assertIn("at order creation", fresh_menu.as_prompt_dict()["problem_context"]["text"])
        recheck = decision_loop.dispatch_choice(
            next_day_state, fresh_menu,
            {"choice": "inspect", "reason": "Recheck dispatch leakage at order creation.", "menu_revision": fresh_menu.revision,
             "depends_on_problem_revision": 2, "arguments": {"outcome": "actual_minutes"}},
            reg,
        )
        self.assertEqual(recheck.status, "succeeded")

        # Step 4: Sibling problem in the same project has its own separate prediction moment;
        # colleague reads shared facts but NOT Dana's private note; outsider receives NO facts (403).
        with _as_user(dana):
            sib_resp = client.post(f"/api/projects/{p['id']}/problems", json={"name": "Post-delivery late fee risk"})
            self.assertEqual(sib_resp.status_code, 200)
            prob_sib = sib_resp.json()["problem_id"]
            client.post(f"/api/projects/{p['id']}/memory/base", json={
                "target": "problem", "problem_id": prob_sib, "subject": "problem",
                "field": "prediction_moment", "text": "Late fee risk is predicted 1 hour after delivery.",
                "source": "ui:sib", "expected_revision": 0,
            })
            sib_sum = client.get(f"/api/projects/{p['id']}/memory/summary", params={"problem_id": prob_sib}).json()
            self.assertIn("1 hour after delivery", sib_sum["text"])
            self.assertNotIn("at order creation", sib_sum["text"])

        with _as_user(colleague):
            coll_sum = client.get(f"/api/projects/{p['id']}/memory/summary", params={"problem_id": prob_eta}).json()
            self.assertIn("On-time SLA threshold is 45 minutes.", coll_sum["text"])
            self.assertIn("at order creation", coll_sum["text"])
            self.assertEqual(coll_sum["personal_preferences"], [])  # Dana's private preference never leaks

        with _as_user(outsider):
            self.assertEqual(client.get(f"/api/projects/{p['id']}/memory/summary", params={"problem_id": prob_eta}).status_code, 403)
            self.assertEqual(client.get(f"/api/projects/{p['id']}/memory/base", params={"problem_id": prob_eta}).status_code, 403)
            self.assertEqual(client.get("/api/memory", params={"project_id": p["id"], "problem_id": prob_eta}).status_code, 403)

        # Toggle switch OFF -> advisory gated role reports disabled and older project still opens cleanly.
        decision_loop.set_settings(home, enabled=False, advisory_gated_roles=False)
        off_adv = advisor.advise_gated_role("problem_analyst", {"columns": []}, home=home)
        self.assertFalse(off_adv["advisory_enabled"])
        self.assertFalse(decision_loop.is_enabled(home))
        self.assertEqual(s.projects.get(p["id"])["name"], "Delivery ETA")


if __name__ == "__main__":
    unittest.main()
