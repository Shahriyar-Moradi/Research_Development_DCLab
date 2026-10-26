"""Package 28.5: the solution team on a programme project: the parts routed in parallel by the router the programme calls,
the plan drawn from those routes, the auditor's screen before the holdout (a planted leak sent back to the person and kept
out), the holdout read once after the person approves it, a part that cannot run put to the person, every node a step and
a hand-off, the walk's record kept."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import langgraph  # noqa: F401

    from dclab_core.replicas import fraud_programme as fp
    from dclab_rnd import experience
    from dclab_rnd.studio import ProjectStore, programme as sp
    from tests.test_studio_programme import SMALL, make_project
except ImportError as error:  # pandas or langgraph are not in every environment
    MISSING = str(error)
else:
    MISSING = ""


class Person:
    """Approves every gate it is shown unless told otherwise; answers a part that cannot run with ``drop``."""

    def __init__(self, keep_out=True, holdout=True, drop=""):
        self.keep_out, self.holdout, self.drop = keep_out, holdout, drop
        self.gates, self.asked = [], []

    def approve(self, gate, summary):
        self.gates.append((gate, summary))
        return self.keep_out if gate == "audit" else self.holdout if gate == "holdout" else True

    def ask(self, field, question):
        self.asked.append((field, question))
        return self.drop


def leak(seed=0):
    """A column a careless export would carry: the settled amount, written after a case is confirmed (a neutral name)."""
    cases = set(fp.generate(seed=seed, **SMALL).cases["order_id"])
    return {"basket_total": lambda rides: rides["fare"] + rides["order_id"].isin(cases).astype(float) * 40.0}


@unittest.skipIf(MISSING, MISSING)
class TeamTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.store = ProjectStore(self.home / "projects")
        self.log = experience.FileLog(self.home / "experience", "wtest")
        experience.install(self.log)
        self.addCleanup(experience.install, None)
        from dclab_rnd import ledger

        self.book = ledger.FileLedger(self.home / "ledger", "wtest")
        ledger.install(self.book)
        self.addCleanup(ledger.install, None)

    def team(self):
        from dclab_rnd.orchestra import checkpoints
        from dclab_rnd.orchestra.solution import SolutionTeam

        cm = checkpoints.open_checkpointer(self.home, database_url="")
        saver = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return SolutionTeam(self.store, saver, home=self.home)

    def test_the_team_is_a_session_of_the_orchestra_solution_agent(self):  # the harness's probe
        from dclab_core import solutions
        from dclab_rnd.orchestra.solution import drive

        pid, _ = make_project(self.store)
        m, _ = sp.maps(self.store.get(pid))
        rules = solutions.routes(m, sp.plan_card(self.store, self.store.get(pid)))
        person = Person()
        out = drive(self.team(), pid, person)
        self.assertEqual(out["next"], "done", out["says"])
        project = self.store.get(pid)
        view = sp.view(project)
        self.assertEqual((view["state"], view["holdout_reads"], project["holdout_uses"]), ("composed", 1, 1))  # read once
        self.assertEqual({k: v["chosen"] for k, v in sp.plan_of(project)["routes"].items()}, {k: r.chosen for k, r in rules.items()})  # the router's own routes
        self.assertEqual([g for g, _ in person.gates], ["holdout"])  # nothing to keep out: the approval to read the holdout alone
        self.assertEqual(project["auditor"]["last"]["verdict"]["verdict"], "pass")  # a clean replica raises no suspect
        events = self.log.read()
        steps = [e.payload.get("tool") for e in events if e.kind == "agent.step" and e.payload["agent"] == "orchestra.solution"]
        self.assertEqual(steps.count("route"), len(m.sub_problems))  # one route a part, in parallel
        for node in ("design", "audit", "compose", "report"):
            self.assertIn(node, steps)
        self.assertEqual([e.payload["team"] for e in events if e.kind == "team.recorded"], ["solution"])
        handoffs = [e.payload["from_node"] for e in events if e.kind == "team.handoff"]
        self.assertEqual(handoffs[:1], ["design"])
        decisions = self._records()
        self.assertTrue(decisions and all(r["used"] == "rule" and r["value_used"] is False for r in decisions))  # every screened column a decision
        self.assertEqual({r["subject"]["key"] for r in decisions}, set(project["auditor"]["last"]["screened"]))

    def _records(self, point="feature.leakage_suspect"):
        return [json.loads(r.model_dump_json()) for r in self.book.records(point=point)]

    def test_a_planted_leak_is_sent_back_and_kept_out_before_the_holdout(self):
        from dclab_rnd.orchestra.solution import drive

        pid, _ = make_project(self.store, rides_extra=leak())
        person = Person(keep_out=True)
        out = drive(self.team(), pid, person)
        self.assertEqual(out["next"], "done", out["says"])
        gates = [g for g, _ in person.gates]
        self.assertEqual(gates, ["audit", "holdout"])  # the suspect first, then the holdout
        self.assertIn("basket_total", " ".join(person.gates[0][1]["suspects"]))
        project = self.store.get(pid)
        self.assertIn("basket_total", sp.excluded(project))
        for leaf in sp.stored(project).leaves:
            self.assertNotIn("basket_total", leaf.columns)  # no leaf read it
        history = [h["verdict"]["verdict"] for h in project["auditor"]["history"]]
        self.assertEqual(history, ["send_back", "pass"])  # sent back once, screened again clean
        self.assertEqual(project["holdout_uses"], 1)
        kept = [r for r in self._records() if r["subject"]["key"] == "basket_total" and r["used"] == "person"]
        self.assertEqual([r["value_used"] for r in kept], [True])  # the person's decision on the suspect, beside the rule's

    def test_a_suspect_the_person_keeps_is_their_decision_and_the_holdout_still_read_once(self):
        from dclab_rnd.orchestra.solution import drive

        pid, _ = make_project(self.store, rides_extra=leak())
        out = drive(self.team(), pid, Person(keep_out=False))
        self.assertEqual(out["next"], "done", out["says"])
        project = self.store.get(pid)
        self.assertNotIn("basket_total", sp.excluded(project))
        self.assertEqual(project["holdout_uses"], 1)
        kept = [r for r in self._records() if r["subject"]["key"] == "basket_total" and r["used"] == "person"]
        self.assertEqual([r["value_used"] for r in kept], [False])
        from dclab_rnd import ledger

        labels = [x for x in self.book.labels() if x.record == kept[0]["id"]] if hasattr(self.book, "labels") else []
        self.assertEqual([x.source for x in labels], ["user_decision_exposed"])  # shown the rule's yes, the person said no: a label

    def test_only_a_person_keeps_a_column_out(self):
        from dclab_rnd.studio import leak_audit

        pid, _ = make_project(self.store)
        with self.assertRaises(leak_audit.AuditError):
            leak_audit.forbid(self.store, pid, ["fare"], "an agent's idea", actor="agent")
        with self.assertRaises(leak_audit.AuditError):
            leak_audit.forbid(self.store, pid, ["not_a_column"], "a person's idea", actor="human")
        leak_audit.forbid(self.store, pid, ["fare"], "a person's idea", actor="human")
        self.assertIn("fare", sp.excluded(self.store.get(pid)))

    def test_the_team_resumes_from_a_new_team_and_reads_the_holdout_once(self):
        pid, _ = make_project(self.store)
        out = self.team().run(pid)
        self.assertEqual((out["status"], out["asks"].get("gate")), ("paused", "holdout"))
        self.assertEqual(int(self.store.get(pid).get("holdout_uses") or 0), 0)  # nothing read while it waits
        out = self.team().run(pid, resume=True)  # another process, the same checkpoints
        self.assertEqual(out["next"], "done", out["says"])
        self.assertEqual(self.store.get(pid)["holdout_uses"], 1)

    def test_a_part_that_cannot_run_is_put_to_the_person(self):
        from dclab_rnd.orchestra.solution import drive

        pid, _ = make_project(self.store, tables=False, plan_with_accounts=False)  # no table of accounts: the rings cannot run
        stopped = drive(self.team(), pid, Person(drop=""))
        self.assertEqual(stopped["next"], "stop")
        self.assertEqual(int(self.store.get(pid).get("holdout_uses") or 0), 0)  # the person stops: the holdout stays closed
        pid2, _ = make_project(self.store, tables=False, plan_with_accounts=False)
        person = Person(drop="We have no table of accounts yet; the rings wait for it.")
        out = drive(self.team(), pid2, person)
        self.assertEqual(out["next"], "done", out["says"])
        self.assertTrue(person.asked and person.asked[0][0].startswith("drop:"))
        self.assertEqual(self.store.get(pid2)["holdout_uses"], 1)

    def test_the_code_export_of_a_team_built_programme_reproduces_its_number(self):
        import subprocess

        from dclab_rnd.orchestra.solution import drive
        from dclab_rnd.studio import code

        pid, _ = make_project(self.store)
        drive(self.team(), pid, Person())
        folder = Path(tempfile.mkdtemp()) / "code"
        code.export(self.store, pid, folder)
        run = subprocess.run([sys.executable, "-I", str(folder / "programme.py")], capture_output=True, text=True, cwd=tempfile.mkdtemp(), timeout=900)
        self.assertEqual(run.returncode, 0, run.stdout[-800:] + run.stderr[-800:])
        self.assertTrue(json.loads(run.stdout.strip().splitlines()[-1])["same"])  # the holdout number, recomputed


    def _reattach(self, pid, rides_extra=None):
        """The person uploads the export again (the table, its cases and the accounts), now as ``rides_extra`` says."""
        from dclab_rnd.studio import data as sd

        r = fp.generate(seed=0, **SMALL)
        for name, values in (rides_extra or {}).items():
            r.rides[name] = values(r.rides)
        d = self.store.data_dir(pid)
        r.rides.to_csv(d / "rides.csv", index=False)
        sd.attach_data(self.store, pid, "rides.csv")
        r.cases.to_csv(d / "cases.csv", index=False)
        sd.attach_labels(self.store, pid, "cases.csv", target="is_case", key=["order_id"])
        r.accounts.to_csv(d / "accounts.csv", index=False)
        sd.attach_table(self.store, pid, "accounts.csv", "accounts")
        project = self.store.get(pid)
        project["suggestion"] = {**(project.get("suggestion") or {}), "alert_budget": 0.02}
        self.store.save(project)

    def test_a_table_changed_after_a_run_is_screened_again_before_its_holdout(self):  # the review of 28.5, finding 1
        from dclab_rnd.orchestra.solution import drive

        pid, _ = make_project(self.store)
        stopped = drive(self.team(), pid, Person(holdout=False))  # screened clean; the person declines the holdout
        self.assertEqual((stopped["next"], int(self.store.get(pid).get("holdout_uses") or 0)), ("stop", 0))
        self._reattach(pid, leak())  # the export again, now carrying the leak
        person = Person(keep_out=True)
        out = drive(self.team(), pid, person)
        self.assertEqual(out["next"], "done", out["says"])
        self.assertEqual([g for g, _ in person.gates], ["audit", "holdout"])  # screened again, and the holdout approved again
        project = self.store.get(pid)
        for leaf in sp.stored(project).leaves:
            self.assertNotIn("basket_total", leaf.columns)
        self.assertEqual(project["holdout_uses"], 1)

    def test_a_persons_ban_outlives_a_new_label_file(self):  # the review of 28.5, finding 2
        from dclab_rnd.studio import data as sd, leak_audit

        pid, r = make_project(self.store, rides_extra=leak())
        leak_audit.forbid(self.store, pid, ["basket_total"], "written after the case is confirmed", actor="human")
        r.cases.to_csv(self.store.data_dir(pid) / "cases.csv", index=False)
        sd.attach_labels(self.store, pid, "cases.csv", target="is_case", key=["order_id"])
        self.assertIn("basket_total", sp.excluded(self.store.get(pid)))

    def test_the_persons_approval_is_the_projects_approval_of_the_holdout(self):  # the review of 28.5, finding 3
        from dclab_rnd.orchestra.solution import drive

        pid, _ = make_project(self.store)
        project = self.store.get(pid)
        project["policy"] = {"require_holdout_approval": True}
        self.store.save(project)
        out = drive(self.team(), pid, Person())
        self.assertEqual(out["next"], "done", out["says"])
        project = self.store.get(pid)
        self.assertEqual([(a["gate"], a["by"]) for a in project.get("approvals") or []], [("holdout", "person")])
        self.assertEqual(project["holdout_uses"], 1)

    def test_a_declined_holdout_reads_nothing(self):
        from dclab_rnd.orchestra.solution import drive

        pid, _ = make_project(self.store)
        out = drive(self.team(), pid, Person(holdout=False))
        self.assertEqual(out["next"], "stop")
        self.assertIn("approval to read the holdout", out["says"])
        self.assertEqual(int(self.store.get(pid).get("holdout_uses") or 0), 0)
        declined = [e.payload for e in self.log.read() if e.kind == "team.interrupt" and e.payload["status"] == "declined"]
        self.assertEqual([(d["interrupt"], d["gate"]) for d in declined], [("approval", "holdout")])

    def test_a_start_while_a_pause_waits_returns_the_pause_and_redoes_nothing(self):  # the review of 28.5, finding 5
        pid, _ = make_project(self.store, rides_extra=leak())
        team = self.team()
        first = team.run(pid)
        self.assertEqual(first["asks"].get("gate"), "audit")
        screens = len(self.store.get(pid)["auditor"]["history"])
        again = team.run(pid)
        self.assertEqual(again["asks"], first["asks"])
        self.assertEqual(len(self.store.get(pid)["auditor"]["history"]), screens)  # not screened again, no round spent

    def test_the_router_fans_out_one_send_a_part(self):
        from langgraph.types import Send

        pid, _ = make_project(self.store)
        team = self.team()
        m, _ = sp.maps(self.store.get(pid))
        sends = team._fan({"project_id": pid, "next": "route", "card": {}})
        self.assertTrue(all(isinstance(x, Send) and x.node == "route_part" for x in sends))
        self.assertEqual(sorted(x.arg["sub"] for x in sends), sorted(s.id for s in m.sub_problems))  # all in one step: run in parallel

    def test_the_team_resumes_in_another_process(self):
        import subprocess

        pid, _ = make_project(self.store)
        out = self.team().run(pid)
        self.assertEqual(out["asks"].get("gate"), "holdout")
        script = ("import sys; sys.path.insert(0, %r)\n"
                  "from pathlib import Path\n"
                  "from dclab_rnd.studio import ProjectStore\n"
                  "from dclab_rnd.orchestra import checkpoints\n"
                  "from dclab_rnd.orchestra.solution import SolutionTeam\n"
                  "home = Path(%r)\n"
                  "with checkpoints.open_checkpointer(home, database_url='') as saver:\n"
                  "    print(SolutionTeam(ProjectStore(home / 'projects'), saver, home=home).run(%r, resume=True)['next'])\n") % (str(ROOT), str(self.home), pid)
        run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600, env={**__import__("os").environ, "DCLAB_NO_LIVE_MODELS": "1"})
        self.assertEqual(run.stdout.strip().splitlines()[-1:], ["done"], run.stderr[-1200:])
        self.assertEqual(self.store.get(pid)["holdout_uses"], 1)

    def test_the_board_after_the_audit_keeps_the_plans_blockers(self):  # the review of 28.5, finding 12
        pid, _ = make_project(self.store, tables=False, plan_with_accounts=False)
        self.team().run(pid)
        boards = [json.loads(x)["board"] for x in (self.home / "orchestra" / "boards.jsonl").read_text().splitlines()]
        designed = next(b for b in boards if b["handoffs"][-1]["from_node"] == "design")
        audited = next(b for b in boards if b["handoffs"][-1]["from_node"] == "audit")
        self.assertTrue(designed["blockers"])
        self.assertEqual(audited["blockers"][:len(designed["blockers"])], designed["blockers"])


@unittest.skipIf(MISSING, MISSING)
class TrainingDataTests(unittest.TestCase):
    """28.9: a team's walks as training data, under the train consent: one example per hand-off, the board before and after,
    paired with the walk's own record, never guessed."""

    def setUp(self):
        from dclab_rnd import consent

        self.home = Path(tempfile.mkdtemp())
        self.store = ProjectStore(self.home / "projects")
        experience.install(experience.FileLog(self.home / "experience", "w"))
        self.addCleanup(experience.install, None)
        consent.set_workspace_consent(self.home, "train", True)  # switched on before the walks: they are written at that level
        self.pid, self.replica = make_project(self.store)
        project = self.store.get(self.pid)
        project["settings"]["consent"] = {"train": True}
        self.store.save(project)
        consent._PROJECTS.clear()  # as a change through the API does: the switch read while the project was built is old
        consent._STORES[str(self.home)] = self.store  # the stamp reads this workspace's projects, on files or PostgreSQL alike
        self.addCleanup(consent._STORES.pop, str(self.home), None)

    def walk(self, person=None, finish=True):
        from dclab_rnd.orchestra import checkpoints
        from dclab_rnd.orchestra.solution import SolutionTeam, drive

        with checkpoints.open_checkpointer(self.home, database_url="") as saver:
            team = SolutionTeam(self.store, saver, home=self.home)
            return drive(team, self.pid, person or Person()) if finish else team.run(self.pid)

    def items(self, switches=None, drafts=None):
        from dclab_rnd import datasets as studio_sets

        return studio_sets._team_items(self.home, switches or {"train": True}, {self.pid: self.store.get(self.pid)}, drafts or {})

    def test_a_walk_becomes_one_example_a_handoff_with_no_value_of_the_table(self):
        from dclab_core import datasets as builders
        from dclab_rnd.lake import Run, session_examples

        self.walk()
        items = self.items()
        self.assertEqual([i["status"] for i in items], ["completed"])
        rows, manifest = builders.team_walks(items)
        self.assertEqual(len(rows), len((self.home / "orchestra" / "boards.jsonl").read_text().splitlines()))  # every hand-off
        self.assertIsNone(rows[0]["before"])
        self.assertEqual(rows[1]["before"]["version"], rows[0]["after"]["version"])
        self.assertTrue({"solution_design", "leakage_judgment", "read_results", "report"} <= set(manifest.counts["family"]))
        examples = session_examples({"team_walks/boards": [{"row": r, "source": "w", "from_sealed": False} for r in rows]}, Run())
        self.assertEqual(len(examples), len(rows))
        self.assertEqual({e.provenance.origin for e in examples}, {"rule_prod", "human_exposed"})  # DCLab's rules; the person's approval
        blob = json.dumps([e.messages for e in examples])
        for value in self.replica.rides["order_id"].astype(str).head(50):
            self.assertNotIn(value, blob)  # no cell of the table
        self.assertEqual(self.items({"train": False}), [])  # the workspace's switch off

    def test_an_unfinished_walk_is_left_out(self):
        out = self.walk(finish=False)  # paused at the holdout: no record of the walk yet
        self.assertEqual(out["status"], "paused")
        self.assertEqual(self.items(), [])

    def test_a_walk_a_model_answered_in_is_a_teachers_and_one_by_another_model_is_left_out(self):
        from dclab_core import datasets as builders
        from dclab_rnd.lake import Run, session_examples

        self.walk()
        path = self.home / "orchestra" / "walks.jsonl"
        rec = json.loads(path.read_text().splitlines()[0])
        for models, expected in ((["claude-sonnet-5-5"], {"teacher_claude"}), (["qwen2.5-coder:1.5b"], set())):
            path.write_text(json.dumps({**rec, "models": models}) + "\n")
            rows, _ = builders.team_walks(self.items())
            examples = session_examples({"team_walks/boards": [{"row": r, "source": "w", "from_sealed": False} for r in rows]}, Run())
            self.assertEqual({e.provenance.origin for e in examples}, expected, models)
            self.assertTrue(all(e.provenance.split == "eval_only" for e in examples))

    def test_a_walk_written_before_training_was_allowed_never_leaves(self):
        from dclab_rnd import consent

        consent.set_workspace_consent(self.home, "train", False)
        consent._PROJECTS.clear()
        self.walk()  # written at operate
        self.assertEqual(self.items(), [])  # switched on later: still not

    def test_the_person_s_words_are_redacted_on_the_board(self):
        from dclab_core import datasets as builders

        board = {"version": 3, "problem": {"labels": "Maria Lopez keeps them, call +44 20 7946 0958", "family": "rare_event"},
                 "parts": [{"part": "sp1", "title": "Ask Maria Lopez at +44 20 7946 0958", "status": "open"}],
                 "blockers": ["sp1 (Ask Maria Lopez at +44 20 7946 0958): no table"]}
        rows, _ = builders.team_walks([{"thread": "draft:d", "team": "orchestra", "steps": [{"node": "analyse", "what": "x", "before": None, "after": board}]}])
        self.assertNotIn("7946", json.dumps(rows))

    def test_a_walk_of_a_deleted_projects_draft_never_leaves(self):
        from dclab_rnd import datasets as studio_sets
        from dclab_rnd.orchestra import board as B

        b = B.advance(B.new("draft:d1"), "analyse", "analysed", "orchestra", home=self.home, walk="w1")
        B.record("draft:d1", "orchestra", [], b, "completed", B.now(), home=self.home, walk="w1", models=[])
        self.assertEqual(studio_sets._team_items(self.home, {"train": True}, {}, {"d1": {"id": "d1"}})[0]["walk"], "w1")
        self.assertEqual(studio_sets._team_items(self.home, {"train": True}, {}, {"d1": {"id": "d1", "project_id": "gone"}}), [])


@unittest.skipIf(MISSING, MISSING)
class SingleProblemTests(unittest.TestCase):
    """30.1: a project of one problem through the team: the engine's stages through the validator, the auditor after the
    leakage stage, a planted leak sent back and kept out through the solution, the holdout read once after approval."""

    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        experience.install(experience.FileLog(self.home / "experience", "w"))
        self.addCleanup(experience.install, None)

    def project(self, leak: bool):
        import dataclasses

        from dclab_rnd import kinds
        from dclab_rnd.kinds import acceptance

        case = kinds.get("binary").case(0)
        if leak:  # a neutral name, built from the outcome: the leakage stage's own screen finds it on the training rows
            table = case.table.copy()
            table["account_score"] = (table["churned"].astype(str).str.lower() == "yes").astype(float) * 10 + (table["monthly_fee"] % 3) / 100
            case = dataclasses.replace(case, table=table)
        store, pid = acceptance.build(case, self.home / "projects")
        if leak:  # the proposal's own audit forbade it; a person unticked it on the Solution page: the team's auditor is what is left
            project = store.get(pid)
            before = [f["column"] for f in project["solution"]["forbidden"]]
            self.assertIn("account_score", before)  # the product caught it first
            project["solution"]["forbidden"] = [f for f in project["solution"]["forbidden"] if f["column"] != "account_score"]
            store.save(project)
        return store, pid

    def run_team(self, store, pid, person):
        from unittest import mock

        from dclab_rnd.expansion import runner
        from dclab_rnd.orchestra import checkpoints
        from dclab_rnd.orchestra.solution import SolutionTeam, drive

        with runner.fast_profile(), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None), \
                checkpoints.open_checkpointer(self.home, database_url="") as saver:
            return drive(SolutionTeam(store, saver, home=self.home), pid, person)

    def test_a_planted_leak_in_a_project_of_one_problem_is_kept_out_before_the_holdout(self):
        store, pid = self.project(leak=True)
        person = Person(keep_out=True)
        out = self.run_team(store, pid, person)
        self.assertEqual(out["next"], "done", out["says"])
        gates = [g for g, _ in person.gates]
        self.assertEqual(gates, ["audit", "holdout"])
        self.assertIn("account_score", " ".join(person.gates[0][1]["suspects"]))
        project = store.get(pid)
        self.assertIn("account_score", [f["column"] for f in project["solution"]["forbidden"]])  # the person's ban, in the solution
        self.assertEqual(int(project.get("holdout_uses") or 0), 1)
        self.assertEqual([h["verdict"]["verdict"] for h in project["auditor"]["history"]], ["send_back", "pass"])
        self.assertTrue(all(((project.get("stages") or {}).get(k) or {}).get("status") in ("completed", "approved") for k in ("data", "leakage", "features", "models", "final")))
        # walked again on the finished project, the team reports: the solution report from the records, never an empty answer
        again = self.run_team(store, pid, Person(keep_out=True))
        self.assertEqual(again["next"], "done", again["says"])
        self.assertIn("roc_auc", again["says"].lower().replace("-", "_"))
        report = next(s for s in again["steps"] if s["action"] == "report")["result"]
        self.assertGreater(report["characters"], 200)
        self.assertEqual(int(store.get(pid).get("holdout_uses") or 0), 1, "reporting reads no holdout")


@unittest.skipIf(MISSING, MISSING)
class JourneyTests(unittest.TestCase):
    def test_a_programme_journey_reaches_the_same_result_through_the_team(self):
        from dclab_rnd.agent_eval.journeys import programme as jp

        person = jp.load("dev")[0]
        direct = jp.play(person)
        team = jp.play(person, team=True)
        self.assertEqual((team["right"], team["parts"]), (direct["right"], direct["parts"]), team.get("why"))
        self.assertEqual(team["got"]["approaches"], direct["got"]["approaches"])  # the same routes, the same plan


@unittest.skipIf(MISSING, MISSING)
class ScreenTests(unittest.TestCase):
    def frame(self, n=400, positives=40, seed=0):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(seed)
        y = pd.Series([1] * positives + [0] * (n - positives))
        return pd.DataFrame({"fare": rng.normal(10, 2, n), "after": y * 50 + rng.normal(0, 1, n),
                             "filled_later": [1.0 if v else None for v in y], "code": ["a" if v else "b" for v in y],
                             "many": [f"c{i}" for i in range(n)]}), y

    def test_a_column_that_alone_separates_the_cases_is_a_suspect_and_every_column_is_scored(self):
        from dclab_core import auditor

        frame, y = self.frame()
        suspects, counts = auditor.screen(frame, y, frame.index, list(frame.columns))
        self.assertEqual({s.column for s in suspects}, {"after", "filled_later", "code"})  # a value, a gap, a code that gives it away
        self.assertNotIn("fare", {s.column for s in suspects})
        self.assertEqual(set(counts["scores"]), {"fare", "after", "filled_later", "code"})  # "many" codes: not screened
        self.assertEqual(counts["columns"], 4)

    def test_too_few_cases_say_nothing_and_the_rounds_end_in_a_stop(self):
        from dclab_core import auditor

        frame, y = self.frame(positives=auditor.MIN_POSITIVES - 1)
        suspects, counts = auditor.screen(frame, y, frame.index, list(frame.columns))
        self.assertEqual((suspects, auditor.verdict("p", [], counts).verdict), ([], "not_screened"))
        frame, y = self.frame()
        suspects, counts = auditor.screen(frame, y, frame.index, ["after"])
        self.assertEqual([auditor.verdict("p", suspects, counts, round_=r).verdict for r in (1, 2, 3)], ["send_back", "send_back", "stop"])
        self.assertEqual(auditor.verdict("p", [], counts).verdict, "pass")

    def test_the_screen_reads_only_the_rows_it_is_given(self):
        from dclab_core import auditor

        frame, y = self.frame()
        fit = frame.index[100:]  # the cases are the first 40 rows: none on these rows
        suspects, counts = auditor.screen(frame, y, fit, ["after"])
        self.assertEqual((counts["rows"], counts["positives"], suspects), (300, 0, []))


if __name__ == "__main__":
    unittest.main()
