"""W5 of the collaborative workspace (2026-11-20): every team agent knows its job, and the deterministic agents carry the
owner's two walkthroughs end to end through the outbox.

The job description of every directory row; the team guard refusing an action outside the directory, an agent's approval,
a human answer with no person, and contacting customers; the renewal round trip (steps 1 to 4) with one answer and one
report in the Business thread and a terminal record behind every claimed action; the blocker that names the missing field
and who supplies it; the food delivery branches finished in opposite orders and joined exactly once, the parent saying
which is still out; a denied specialist action reported with its next owner and no invented score; the kitchen's +8
minutes superseding the ETA report and leaving acceptance's snapshot alone; and a delivery handed over twice running no
agent step again. On files and on PostgreSQL (``tests/run_parallel.py --pg``; never run directly: that empties
``dclab_test``).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pgtest  # noqa: E402

from dclab_core import team_work as tw  # noqa: E402
from dclab_rnd import context, experience  # noqa: E402
from dclab_rnd.agents import default_registry  # noqa: E402
from dclab_rnd.jobs import open_jobs  # noqa: E402
from dclab_rnd.memory.scope import create_problem  # noqa: E402
from dclab_rnd.storage import open_stores, open_team_work  # noqa: E402
from dclab_rnd.teamspace import agents, delivery, fixtures, history_lines  # noqa: E402
from dclab_rnd.teamspace.threads import threads_for  # noqa: E402
from dclab_rnd.teamspace.tools import TeamTurn  # noqa: E402

BRIEF = ("Customers are leaving. Who should Sales call before renewal? Eligible: accounts renewing in the next 60 days. "
         "Decision date: seven days before each renewal. Requested: can the data rank accounts by churn risk?")
ANSWER = "Failure to renew by the contractual renewal date."
DISPATCH_Q = ("For order 1042 at its first dispatch: when will it arrive, and will a courier accept the first offer? "
              "Estimate the arrival time and the probability of acceptance.")
FEASIBLE = "The sample has renewal dates and mature outcomes. We can score seven days before renewal."
LABEL = "Simulated (fixture sample:renewal_v1):"
FOOD = "At 18:00 the estimated arrival was 18:32. The first-offer acceptance estimate was 72%."


class _Contract:
    env: dict = {}

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"DCLAB_AUTH": "", "DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS": "0", **self.env})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.home = Path(tempfile.mkdtemp()) / "workspace"
        self.home.mkdir()
        from dclab_rnd.storage import db

        projects, drafts, _ = open_stores(self.home)
        wid = db.workspace_for(self.home) if db.database_url(required=False) else None
        self.services = SimpleNamespace(store=SimpleNamespace(home=self.home), projects=projects, drafts=drafts, workspace_id=wid,
                                        experience=experience.open_log(self.home, wid), job_store=open_jobs(self.home),
                                        teamwork=open_team_work(self.home, projects=projects))
        self.work = self.services.teamwork
        self.ws = self.work.workspace
        self.pid = projects.create("Renewal retention", goal="Who should Sales call")["id"]
        self.prob = create_problem(projects, self.pid, "Renewal", workspace_id=self.ws)["problem_id"]
        self.sara = {"user_id": "sara", "role": "data_scientist", "workspace_id": self.ws}
        target = delivery.set_target(None)  # the real default: W3's thread post, then the agent's turn
        self.addCleanup(delivery.set_target, target)
        checker = delivery.set_membership_checker(None)
        self.addCleanup(delivery.set_membership_checker, checker)
        token = context.set_services(self.services)
        self.addCleanup(context.reset_services, token)
        self.threads = threads_for(self.services)
        self.biz_thread = self.threads.open_thread(self.pid, self.prob, "business", label="Business")[0]["id"]  # the person's thread

    # ------------------------------------------------------------------ helpers
    def pump(self, only=None, limit=80):
        """Deliver what is pending, task by task, as the worker's jobs would (``only``: these tasks' rows, nothing else)."""
        for _ in range(limit):
            heads = [h for h in self.work.pending_heads() if only is None or h["task_id"] in only]
            if not heads:
                return
            head = heads[0]
            lane = delivery.lane_of(self.work, head)  # a child's reply travels in its parent's lane
            for row in delivery.run(self.services, {"project_id": head["project_id"], "task_id": lane}):
                self.assertEqual(row["status"], "delivered", row.get("reason"))
        raise AssertionError("the deliveries never settled")

    def view(self, task_id):
        return self.work.task_view(self.pid, task_id)

    def steps(self, agent=None):
        return [e for e in self.services.experience.read(kinds=["agent.step"])
                if agent is None or e.payload.get("agent") == agent]

    def open_renewal(self, task_id="T-renewal", sample="sample:renewal_v1"):
        out = agents.open_case(self.services, self.pid, self.prob, task_id=task_id, question=BRIEF, thread_id=self.biz_thread, member="sara",
                               initiator=self.sara, inputs=(sample,))
        self.assertEqual(out["refused"], [])
        return out

    def renewal_to_the_end(self, task_id="T-renewal", sample="sample:renewal_v1"):
        self.open_renewal(task_id, sample)
        self.pump()
        out = agents.person_replies(self.services, self.pid, task_id, person="sara", text=ANSWER, initiator=self.sara)
        self.assertEqual(out["refused"], [])
        self.pump()

    def open_dispatch(self, task_id):
        out = agents.open_case(self.services, self.pid, self.prob, task_id=task_id, question=DISPATCH_Q, thread_id=self.biz_thread,
                               member="sara", initiator=self.sara, inputs=(fixtures.DISPATCH,),
                               expected_output="a dispatch brief: the arrival and the first-offer acceptance estimates")
        self.assertEqual(out["refused"], [])
        self.pump(only={task_id})  # Data Science accepts and opens its two child tasks; nothing else moves

    def messages(self, team):
        thread = self.threads.find_thread(self.pid, self.prob, team)
        return self.threads.messages(thread["id"]) if thread else []

    def said(self, team, kind, task_id=None):
        return [m for m in self.messages(team) if (m.get("handoff") or {}).get("kind") == kind
                and (task_id is None or m["handoff"]["task_id"] == task_id)]

    def all_messages(self):
        return sum(len(self.threads.messages(t["id"])) for t in self.threads.threads(project_id=self.pid))

    def turn(self, agent, task_id, **kw):
        return TeamTurn(self.services, self.work, self.pid, agent, task_id=task_id, **kw)

    # ------------------------------------------------------------------ job descriptions
    def test_every_directory_row_has_a_job_description_built_from_the_directory(self):
        self.assertEqual(set(agents.CONFIGS), set(tw.DIRECTORY))
        registry = default_registry()
        for name, config in tw.DIRECTORY.items():
            agent = agents.CONFIGS[name]
            self.assertIs(agent.config, config)  # one source of truth: the directory's own row
            job = agents.job_description(name)
            self.assertEqual([a["action"] for a in job["actions"]], [a.action for a in config.action_directory])
            self.assertEqual(job["required_report"], list(config.required_report))
            self.assertEqual(job["reads"], list(config.memory_scope))
            self.assertIn("approve", job["forbidden"])
            self.assertTrue(job["text"].startswith(config.identity) and "may never" in job["text"], name)
            for tool in agent.policy_tools:
                self.assertEqual(registry.get(tool).scope, "team", tool)
        ds = agents.job_description("data_science")
        self.assertIn("contact customers", " ".join(ds["may_never"]))
        self.assertEqual(next(a for a in ds["actions"] if a["action"] == "confirm_target")["tool"], "team_record_human_response")
        self.assertFalse(next(a for a in agents.job_description("ml_engineering")["actions"] if a["action"] == "run_training")["available"])
        self.assertEqual(agents.config_of("model:eta").name, "model")

    # ------------------------------------------------------------------ the guard
    def test_the_guard_refuses_outside_the_directory_an_approval_a_human_answer_without_a_person_and_contacting_customers(self):
        self.open_renewal()
        registry = default_registry()
        before = len(self.view("T-renewal")["actions"])
        ds, business, sales = (self.turn(a, "T-renewal") for a in ("data_science", "business", "sales"))
        cases = [
            ("team_act", {"task_id": "T-renewal", "action": "run_training"}, ds, "not in data_science's action directory"),
            ("team_act", {"task_id": "T-renewal", "action": "approve"}, business, "never approves"),
            ("team_record_human_response", {"task_id": "T-renewal", "action": "confirm_target", "response_id": "r1"}, ds,
             "only with the person who responded"),
            ("team_record_human_response", {"task_id": "T-renewal", "action": "confirm_target", "response_id": "r1",
                                            "responded_by": "omar"}, ds, "no answer r1 by omar was recorded by business"),
            ("team_act", {"task_id": "T-renewal", "action": "contact_customers"}, business, "contact customers"),
            ("team_act", {"task_id": "T-renewal", "action": "contact_customers"}, sales, "contact customers"),
            ("team_act", {"task_id": "T-renewal", "action": "inspect_schema"}, business, "not in business's action directory"),
            ("team_model_call", {"task_id": "T-renewal", "action": "inspect_schema"}, ds, "bound to tool:inspect_schema"),
            ("team_accept", {"task_id": "T-renewal"}, business, "only its owner"),
            ("team_answer", {"task_id": "T-renewal", "response_id": "r1", "text": "made up"}, business, "never made up"),
        ]
        for tool, args, ctx, says in cases:
            with self.subTest(tool=tool, args=args, agent=ctx.agent):
                out = registry.call(tool, args, ctx)
                self.assertIn("error", out)
                self.assertIn(says, out["error"])
        self.assertIn("error", registry.call("team_accept", {"task_id": "T-renewal"}, None))  # outside a team turn
        self.assertEqual(len(self.view("T-renewal")["actions"]), before)  # nothing refused was recorded
        self.assertEqual(self.view("T-renewal")["task"].status, "open")

    # ------------------------------------------------------------------ the renewal round trip
    def test_a_turn_is_a_traced_run_with_a_step_per_tool_call(self):
        self.open_renewal()
        self.pump()
        steps = self.steps("teamspace.data_science")
        self.assertEqual([s.payload.get("tool") for s in steps], ["team_accept", "team_clarify", None])  # None: its closing words
        self.assertEqual(len({s.ids.run for s in steps}), 1)
        calls = [e for e in self.services.experience.read(kinds=["tool.call"]) if e.ids.run == steps[0].ids.run]
        self.assertEqual([c.payload["tool"] for c in calls], ["team_accept", "team_clarify"])
        runs = [e for e in self.services.experience.read(kinds=["run.recorded"]) if e.payload.get("name") == "teamspace.data_science"]
        self.assertEqual(len(runs), 1)
        self.assertEqual([s.payload.get("tool") for s in self.steps("teamspace.business")], ["team_delegate", None, "team_clarify", None])

    def test_the_ledger_records_the_agents_choices_as_rules_never_as_a_model(self):
        from dclab_rnd.ledger import capture

        seen = []
        real = capture._put

        def keep(point, subject, answers, value, used, **kw):
            seen.append((point, [(a.by, a.answerer) for a in answers if a is not None], used, kw.get("level")))
            return real(point, subject, answers, value, used, **kw)

        with mock.patch.object(capture, "_put", keep):
            self.open_renewal()
            self.pump()
        mine = [s for s in seen if s[0] in ("agent.tool_choice", "workflow.stop")]
        self.assertTrue(mine)
        for point, answers, used, level in mine:
            self.assertEqual(used, "rule", point)
            self.assertEqual(level, 0)
            self.assertTrue(all(by == "rule" and who.startswith("rule:teamspace.") for by, who in answers), answers)

    def test_the_renewal_round_trip_returns_one_answer_and_one_report_to_the_business_thread(self):
        self.open_renewal()
        self.pump()
        view = self.view("T-renewal")
        self.assertEqual(view["task"].status, "waiting_for_person")  # the clarification keeps the task open
        self.assertEqual(view["next_owner"], "sara")
        self.assertEqual([m["text"] for m in self.said("business", "clarification")], [agents.CLARIFY_TARGET])  # to the person
        asked = self.said("data_science", "clarification")  # and kept in Data Science's own thread
        self.assertEqual([(m["text"], m["author"]["id"]) for m in asked], [(agents.CLARIFY_TARGET, "data_science")])

        out = agents.person_replies(self.services, self.pid, "T-renewal", person="sara", text=ANSWER, initiator=self.sara)
        self.assertEqual([s["tool"] for s in out["steps"]], ["team_record_human_response", "team_answer"])
        self.pump()

        view = self.view("T-renewal")
        self.assertEqual(view["task"].status, "complete")
        [report] = view["reports"]
        # the checks ran on an invented sample: they are simulated, and so is the report (the reviewer's finding 5)
        self.assertEqual((report.id, report.status, report.answer, report.simulated), ("T-renewal@r1", "complete", FEASIBLE, True))
        self.assertEqual({l.status for l in report.actions.completed if l.actor == "data_science" and l.action != "confirm_target"},
                         {"simulated"})
        self.assertTrue(report.next_task.closed)
        self.assertIn(ANSWER, report.evidence[0].assumptions[0])
        self.assertTrue(any("renewed_by_date" in x for x in report.limits))
        # every claimed action has a terminal record, an actor, and a person behind every human answer
        listed = report.actions.completed + report.actions.failed + report.actions.skipped
        stored = {a.action_id: a for a in view["actions"]}
        for line in listed:
            self.assertTrue(tw.can_claim(stored[line.action_id]), line)
            self.assertTrue(line.actor)
        self.assertTrue(all(tw.can_claim(a) for a in view["actions"]))
        humans = [a for a in view["actions"] if a.source.startswith("human:")]
        self.assertEqual(sorted(a.action for a in humans), ["confirm_target", "confirm_terms"])
        self.assertTrue(all(a.responded_by.id == "sara" for a in humans))
        self.assertEqual({l.action for l in report.actions.completed},
                         {"delegate_analysis", "confirm_terms", "confirm_target", "inspect_schema", "check_leakage", "define_baseline",
                          "explain_uncertainty"})
        # the original Business thread: one answer, one linked report; Business presented it once
        replies = [d for d in self.work.deliveries(project_id=self.pid) if d["kind"] == "reply"]
        self.assertEqual([(d["thread_id"], d["item_id"], d["status"]) for d in replies], [(self.biz_thread, "T-renewal@r1", "delivered")])
        answers = self.said("business", "reply")  # one answer message, linking its one report
        self.assertEqual([(m["thread_id"], m["text"].split("\n")[0], m["handoff"]["item_id"]) for m in answers],
                         [(self.biz_thread, f"{LABEL} {FEASIBLE}", "T-renewal@r1")])
        self.assertEqual(self.four_questions(answers[0]), {
            "found": f"{LABEL} {FEASIBLE}", "why": answers[0]["text"].split("\n")[1][len("Why: "):],
            "next": "closed, nothing is owed.", "changed": "first report (T-renewal@r1)."})
        self.assertIn("schema:sample:renewal_v1", self.four_questions(answers[0])["why"])
        self.assertEqual([m["handoff"]["kind"] for m in self.messages("business")], ["clarification", "reply"])
        self.assertEqual([m["handoff"]["kind"] for m in self.messages("data_science")], ["task", "clarification", "answer"])
        presented = [a for a in view["actions"] if a.action == "present_options"]
        self.assertEqual([(a.actor.id, a.output_ref) for a in presented], [("business", "presented:T-renewal@r1")])
        self.assertEqual(history_lines(self.work.handoff_history(self.pid, self.prob)), [
            f"1. Business → Data Science · {BRIEF}",
            f"2. Data Science → Business · {agents.CLARIFY_TARGET}",
            f"3. Business → Data Science · {ANSWER}",
            f"4. Data Science → Business · {LABEL} {FEASIBLE}"])

    def four_questions(self, message):
        """What a Business member reads from one answer message alone: what was found, why, who acts next, what changed."""
        found, why, nxt, changed = message["text"].split("\n")
        self.assertTrue(why.startswith("Why: ") and nxt.startswith("Next: ") and changed.startswith("Changed: "), message["text"])
        return {"found": found, "why": why[len("Why: "):], "next": nxt[len("Next: "):], "changed": changed[len("Changed: "):]}

    def test_a_sample_without_renewal_dates_returns_a_blocker_naming_the_field_and_who_supplies_it(self):
        self.renewal_to_the_end(sample="sample:renewal_no_dates")
        view = self.view("T-renewal")
        self.assertEqual(view["task"].status, "blocked")
        [report] = view["reports"]
        self.assertEqual(report.status, "blocked")
        self.assertIn("renewal_date", report.answer)
        self.assertIn("Business team can supply it", report.answer)
        self.assertEqual((report.next_task.owner, report.next_task.closed), ("business", False))
        self.assertIn("renewal_date", report.next_task.unblocks)
        self.assertEqual([l.action for l in report.actions.completed if l.actor == "data_science"], ["confirm_target", "inspect_schema"])
        self.assertNotIn("score seven days", report.answer)

    def test_a_duplicate_delivery_does_not_repeat_an_agent_step(self):
        self.renewal_to_the_end()
        rows = self.work.deliveries(project_id=self.pid)
        self.assertEqual([r["kind"] for r in rows], ["task", "clarification", "answer", "reply"])
        steps, actions, messages = len(self.steps()), len(self.view("T-renewal")["actions"]), self.all_messages()
        for row in rows:  # each row handed over again, as after a crash between the agent's turn and "delivered"
            receipt = delivery.default_target(self.services, row)
            self.assertFalse(receipt["posted"], row["kind"])
            self.assertTrue(receipt["agent"]["skipped"], row["kind"])
            self.assertEqual(receipt["agent"]["steps"], [])
        self.assertEqual(self.all_messages(), messages)
        self.assertEqual(len(self.steps()), steps)
        self.assertEqual(len(self.view("T-renewal")["actions"]), actions)
        self.assertEqual(len(self.view("T-renewal")["reports"]), 1)
        self.assertEqual(messages, 5)  # the four rows posted once each, and Data Science's own note of its question
        delivery.run(self.services, {"project_id": self.pid, "task_id": "T-renewal"})  # a stray job adds nothing either
        self.assertEqual(len(self.steps()), steps)

    def test_an_answer_by_another_person_is_refused_and_nothing_travels(self):
        self.open_renewal()
        self.pump()
        omar = {**self.sara, "user_id": "omar"}
        out = agents.person_replies(self.services, self.pid, "T-renewal", person="omar", text="Cancellation.", initiator=omar)
        self.assertIn("not this case's decision owner", out["refused"][0])
        # the reviewer's trigger: omar's turn writes down an answer as sara's
        out = agents.person_replies(self.services, self.pid, "T-renewal", person="sara", text="Cancellation.", initiator=omar)
        self.assertIn("this turn acts for omar; it cannot record an answer by sara", out["refused"][0])
        registry = default_registry()
        made_up = registry.call("team_record_human_response", {"task_id": "T-renewal", "action": "record_decision", "response_id": "made-up",
                                                                "responded_by": "sara"}, self.turn("business", "T-renewal", initiator=omar))
        self.assertIn("this turn acts for omar", made_up["error"])
        relayed = registry.call("team_record_human_response", {"task_id": "T-renewal", "action": "confirm_target", "response_id": "made-up",
                                                               "responded_by": "sara"}, self.turn("data_science", "T-renewal", initiator=self.sara))
        self.assertIn("no answer made-up by sara was recorded by business", relayed["error"])  # the owner never invents the relay
        self.assertFalse([a for a in self.view("T-renewal")["actions"] if a.action == "record_decision"])
        confirm = next(a for a in self.view("T-renewal")["actions"] if a.action == "confirm_terms")
        self.assertEqual((confirm.status, confirm.responded_by), ("running", None))
        self.assertFalse([d for d in self.work.deliveries(project_id=self.pid) if d["kind"] == "answer"])
        self.assertEqual(self.view("T-renewal")["task"].status, "waiting_for_person")

    def test_a_member_who_lost_access_gets_no_tool_run_in_their_name(self):
        self.open_renewal()
        self.pump()
        delivery.set_membership_checker(lambda uid, wid: None if uid == "sara" else "data_scientist")  # sara was removed
        out = agents.person_replies(self.services, self.pid, "T-renewal", person="sara", text=ANSWER, initiator=self.sara)
        self.assertIn("can no longer change work in this project", out["refused"][0])
        self.assertFalse([d for d in self.work.deliveries(project_id=self.pid) if d["kind"] == "answer"])
        confirm = next(a for a in self.view("T-renewal")["actions"] if a.action == "confirm_terms")
        self.assertEqual(confirm.status, "running")  # still waiting for a person who may answer

    # ------------------------------------------------------------------ food delivery
    def food(self, task_id, first, second):
        self.open_dispatch(task_id)
        children = [c.id for c in self.view(task_id)["children"]]
        self.assertEqual(children, [f"{task_id}.eta", f"{task_id}.acceptance"])
        self.assertEqual(self.view(task_id)["task"].status, "waiting_for_team")
        self.pump(only={f"{task_id}.{first}"})  # one branch answers; its reply reaches Data Science
        return children

    def test_the_two_branches_finish_in_opposite_orders_and_are_joined_exactly_once(self):
        for n, (first, second) in enumerate((("eta", "acceptance"), ("acceptance", "eta"))):
            task_id = f"T-dispatch-{n}"
            with self.subTest(first=first):
                self.food(task_id, first, second)
                view = self.view(task_id)
                self.assertEqual(view["join"].outstanding, [f"{task_id}.{second}"])
                self.assertEqual(view["next_owner"], f"model:{second}")
                self.assertEqual(view["reports"], [])  # nothing returned to Business while a branch is out
                shown = agents.preview_report(self.services, self.pid, task_id)
                self.assertEqual((shown.status, shown.outstanding), ("partial", [f"{task_id}.{second}"]))
                self.assertIn(f"Still waiting on: {task_id}.{second}", shown.answer)
                self.assertIn(f"Still waiting on: {task_id}.{second}.", shown.limits)

                self.pump()
                view = self.view(task_id)
                [report] = view["reports"]
                self.assertEqual(report.status, "complete")
                self.assertEqual(report.answer, "At 18:00 the estimated arrival was 18:32. The first-offer acceptance estimate was 72%.")
                self.assertEqual(sorted(report.child_reports), sorted([f"{task_id}.eta@r1", f"{task_id}.acceptance@r1"]))
                self.assertTrue(report.simulated)
                self.assertEqual(sorted(e.ref for e in report.evidence), ["fixture:A1", "fixture:E1"])
                reviews = [a.input_revision for a in view["actions"] if a.action == "review_evidence"]
                self.assertEqual(sorted(reviews), sorted(report.child_reports))  # each branch read once
                for child in view["children"]:
                    [mine] = self.view(child.id)["reports"]
                    self.assertTrue(mine.simulated)
                    self.assertEqual([l.status for l in mine.actions.completed], ["simulated"])
                replies = [d for d in self.work.deliveries(project_id=self.pid, task_id=task_id) if d["kind"] == "reply"]
                self.assertEqual(len(replies), 1)
                [answer] = self.said("business", "reply", task_id)
                self.assertEqual(answer["handoff"]["item_id"], report.id)
                # a fixture result says so in the thread and in the history (the reviewer's finding 4)
                self.assertEqual(answer["text"].split("\n")[0], f"Simulated (fixture E1/A1): {FOOD}")
                lines = history_lines(self.work.handoff_history(self.pid, self.prob))
                self.assertIn(f"Data Science → Business · Simulated (fixture E1/A1): {FOOD}", "\n".join(lines))
                self.assertIn("Simulated (fixture E1): At 18:00 the estimated arrival was 18:32.", "\n".join(lines))
                for kind in ("eta", "acceptance"):  # each child's reply reached Data Science's thread once
                    self.assertEqual(len(self.said("data_science", "reply", f"{task_id}.{kind}")), 1)

    def test_a_denied_specialist_action_is_reported_with_its_next_owner_and_no_invented_score(self):
        with mock.patch.dict(fixtures.MODELS, clear=False):
            del fixtures.MODELS[("acceptance", fixtures.DISPATCH)]
            self.food("T-dispatch", "acceptance", "eta")
            self.pump()
        child = self.view("T-dispatch.acceptance")
        [denied] = child["reports"]
        self.assertEqual(denied.status, "blocked")
        self.assertEqual([(l.action, l.status) for l in denied.actions.failed], [("score", "denied")])
        self.assertEqual(denied.next_task.owner, "ml_engineering")
        self.assertFalse(denied.simulated)
        self.assertEqual(denied.evidence, [])
        [report] = self.view("T-dispatch")["reports"]
        self.assertEqual(report.status, "partial")
        self.assertEqual(report.next_task.owner, "ml_engineering")
        self.assertIn("the score was denied", report.answer)
        self.assertIn("18:32", report.answer)
        for text in (report.answer, denied.answer, *report.limits):
            self.assertNotIn("72", text)
            self.assertNotIn("%", text)
        self.assertEqual(sorted(report.child_reports), ["T-dispatch.acceptance@r1", "T-dispatch.eta@r1"])
        # the same reply handed over again adds nothing
        reply = next(d for d in self.work.deliveries(project_id=self.pid, task_id="T-dispatch.acceptance") if d["kind"] == "reply")
        actions = len(self.view("T-dispatch")["actions"])
        before = self.all_messages()
        self.assertTrue(delivery.default_target(self.services, reply)["agent"]["skipped"])
        self.assertEqual(self.all_messages(), before)
        self.assertEqual(len(self.view("T-dispatch")["actions"]), actions)
        self.assertEqual(len(self.view("T-dispatch")["reports"]), 1)

    def test_the_kitchens_eight_minutes_supersede_the_eta_report_and_leave_acceptance_alone(self):
        self.food("T-dispatch", "eta", "acceptance")
        self.pump()
        acceptance = self.view("T-dispatch.acceptance")
        out = agents.correct(self.services, self.pid, "T-dispatch.eta", kitchen_minutes=8)
        self.assertEqual(out["refused"], [])
        eta = self.view("T-dispatch.eta")
        old, new = sorted(eta["reports"], key=lambda r: r.revision)
        self.assertEqual((old.status, old.superseded_by, new.supersedes), ("superseded", new.id, old.id))
        self.assertIn("18:40", new.answer)
        self.assertIn("was 18:32", new.answer)
        self.assertTrue(new.simulated)
        self.assertEqual(new.evidence[0].input_revision, f"{fixtures.DISPATCH};kitchen+8m")
        self.pump()
        parent = self.view("T-dispatch")
        first, second = sorted(parent["reports"], key=lambda r: r.revision)
        self.assertEqual((first.status, second.supersedes), ("superseded", first.id))
        self.assertEqual(sorted(second.child_reports), ["T-dispatch.acceptance@r1", "T-dispatch.eta@r2"])
        self.assertIn("18:40", second.answer)
        self.assertIn("72%", second.answer)
        after = self.view("T-dispatch.acceptance")  # untouched: the same report, actions and snapshot
        self.assertEqual(after["reports"], acceptance["reports"])
        self.assertEqual(after["actions"], acceptance["actions"])
        self.assertEqual(after["reports"][0].evidence[0].input_revision, fixtures.DISPATCH)
        first_msg, second_msg = self.said("business", "reply", "T-dispatch")
        changed = self.four_questions(second_msg)  # what changed since the last message, from the thread alone
        self.assertEqual(changed["changed"], "replaces T-dispatch@r1; the current report is T-dispatch@r2.")
        self.assertTrue(changed["found"].startswith("Simulated (fixture E1/A1): Revised: the estimated arrival is 18:40"))
        self.assertEqual(self.four_questions(first_msg)["changed"], "first report (T-dispatch@r1).")
        self.assertIn("fixture:E1", changed["why"])
        presented = [a.output_ref for a in parent["actions"] if a.action == "present_options"]
        self.assertEqual(presented, ["presented:T-dispatch@r1", "presented:T-dispatch@r2"])
        self.assertEqual(agents.correct(self.services, self.pid, "T-dispatch.eta", kitchen_minutes=8)["skipped"], True)  # once


class _Crashes:
    """Crash injection and races (the reviewer's findings 2 and 3), on files and PostgreSQL."""

    def test_a_crash_while_reporting_leaves_neither_the_move_nor_the_report_and_the_turn_resumes(self):
        self.open_renewal()
        self.pump()
        agents.person_replies(self.services, self.pid, "T-renewal", person="sara", text=ANSWER, initiator=self.sara)
        real, fired = type(self.work)._keep_report, {"n": 0}

        def crash(store, *a, **k):
            fired["n"] += 1
            if fired["n"] == 1:
                raise RuntimeError("worker killed")
            return real(store, *a, **k)
        with mock.patch.object(type(self.work), "_keep_report", crash):
            head = next(h for h in self.work.pending_heads())
            ended = delivery.run(self.services, {"project_id": self.pid, "task_id": delivery.lane_of(self.work, head)})
        self.assertEqual(ended[-1]["status"], "pending")  # a temporary failure: counted, retried
        view = self.view("T-renewal")
        self.assertEqual((view["task"].status, view["reports"]), ("accepted", []))  # neither the move nor the report
        self.pump()
        view = self.view("T-renewal")
        self.assertEqual((view["task"].status, [r.id for r in view["reports"]]), ("complete", ["T-renewal@r1"]))
        self.assertEqual(len(self.said("business", "reply")), 1)
        self.assertFalse([d for d in self.work.deliveries(project_id=self.pid) if d["status"] != "delivered"])

    def test_a_crash_after_a_delegation_resumes_and_the_parent_waits_with_its_child(self):
        real, fired = type(self.work).record_action, {"n": 0}

        def crash(store, project_id, action):
            if action.action == "delegate_specialist" and not fired["n"]:
                fired["n"] += 1
                raise RuntimeError("worker killed")
            return real(store, project_id, action)
        agents.open_case(self.services, self.pid, self.prob, task_id="T-dispatch", question=DISPATCH_Q, thread_id=self.biz_thread,
                         member="sara", initiator=self.sara, inputs=(fixtures.DISPATCH,), expected_output="a dispatch brief")
        with mock.patch.object(type(self.work), "record_action", crash):
            delivery.run(self.services, {"project_id": self.pid, "task_id": "T-dispatch"})
        view = self.view("T-dispatch")
        self.assertEqual(view["task"].status, "waiting_for_team")  # saved with its first child, in one transaction
        self.assertEqual([c.id for c in view["children"]], ["T-dispatch.eta"])
        self.pump()
        view = self.view("T-dispatch")
        self.assertEqual(sorted(a.input_revision for a in view["actions"] if a.action == "delegate_specialist"),
                         ["T-dispatch.acceptance", "T-dispatch.eta"])
        [report] = view["reports"]
        self.assertEqual((report.status, report.answer), ("complete", FOOD))

    def test_two_branch_replies_share_one_lane_and_a_forced_race_is_a_duplicate_not_a_failure(self):
        agents.open_case(self.services, self.pid, self.prob, task_id="D", question=DISPATCH_Q, thread_id=self.biz_thread,
                         member="sara", initiator=self.sara, inputs=(fixtures.DISPATCH,), expected_output="a dispatch brief")
        self.pump(only={"D"})
        for child in ("D.eta", "D.acceptance"):  # each specialist answers; its reply waits in the parent's lane
            delivery.run(self.services, {"project_id": self.pid, "task_id": child})
        a, b = [d for d in self.work.deliveries(status="pending", project_id=self.pid) if d["kind"] == "reply"]
        self.assertEqual({delivery.lane_of(self.work, a), delivery.lane_of(self.work, b)}, {"D"})
        self.assertEqual(delivery.enqueue(self.services, a)["id"], delivery.enqueue(self.services, b)["id"])  # one job, in order
        real, turns = type(self.work).report_task, {}

        def race(store, *args, **kw):  # turn B runs to its end inside turn A, just before A stores its report
            if "B" not in turns:
                turns["B"] = None  # (turn B's own report_task call goes straight through)
                turns["B"] = delivery.default_target(self.services, b)
            return real(store, *args, **kw)
        with mock.patch.object(type(self.work), "report_task", race):
            turns["A"] = delivery.default_target(self.services, a)  # no PermanentFailure: the loser is a duplicate
        self.assertEqual(turns["A"]["agent"]["steps"][-1], "team_return_report")
        [report] = self.view("D")["reports"]
        self.assertEqual(report.answer, FOOD)
        self.assertEqual(len([d for d in self.work.deliveries(project_id=self.pid, task_id="D") if d["kind"] == "reply"]), 1)


class FileTeamAgentsTests(_Crashes, _Contract, unittest.TestCase):
    env = {"DCLAB_DATABASE_URL": ""}

    def test_with_teams_set_up_the_person_answers_only_for_a_team_they_are_in(self):
        from dclab_rnd.storage import open_teams

        teams = open_teams(self.home)
        business = teams.create_team("Business", "business")
        teams.add_member(business["id"], "lee")
        self.open_renewal()
        self.pump()
        out = agents.person_replies(self.services, self.pid, "T-renewal", person="sara", text=ANSWER, initiator=self.sara)
        self.assertIn("not in the workspace's business team", out["refused"][0])
        teams.add_member(business["id"], "sara")
        self.assertEqual(agents.person_replies(self.services, self.pid, "T-renewal", person="sara", text=ANSWER,
                                               initiator=self.sara)["refused"], [])
        self.pump()
        self.assertEqual(self.view("T-renewal")["task"].status, "complete")


class PgTeamAgentsTests(_Crashes, _Contract, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = pgtest.require()
        cls.env = {"DCLAB_DATABASE_URL": cls.url}


if __name__ == "__main__":
    unittest.main()
