"""The Home orchestrator on a built project's chat (the owner, 2026-10-20): "ok do it" and "ok" got nonsense, "build improvements"
got the advice again, and a failed improvement job was never said. Now a turn is planned (by the model, or by the rules when none
serves), handed to specialists whose tables appear in the chat, a yes carries out the pending offer, and a job is followed on a
task board that says how it ended."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.draft import orchestrator as orch  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402

RECORDS = {
    "data": {"primary_metric": "roc_auc", "evidence": {"train_rows": 6400, "holdout_rows": 1600, "holdout_split": "latest 20% by time", "cv_protocol": "3 time-ordered folds",
                                                       "target_summary": {"positive_rate_train": 0.21}, "risk_features": {"customer_id": ["identifier_like"]}}},
    "leakage": {"primary_metric": "roc_auc", "evidence": {"declared_leakage_features": ["refund_amount"], "apparent_lift": 0.08, "heuristic_review_candidates": []}},
    "features": {"primary_metric": "roc_auc", "evidence": {"selected_recipe": "raw", "selected_vs_reference_gain": 0.0,
                                                           "stage_results": [{"recipe": "raw", "description": "the inputs as they are", "feature_count_mean": 67,
                                                                              "metrics": {"roc_auc": {"mean": 0.7624, "std": 0.0356}}}]}},
    "models": {"primary_metric": "roc_auc", "evidence": {"feature_recipe": "raw", "cv_protocol": "3 time-ordered folds", "selected_model": "logistic_regression",
                                                         "ranking": [{"model": "logistic_regression", "mean": 0.7624, "std": 0.0356, "adjusted_score": 0.7535, "elapsed_seconds": 1.8},
                                                                     {"model": "extra_trees", "mean": 0.7274, "std": 0.0131, "adjusted_score": 0.7241, "elapsed_seconds": 10.0},
                                                                     {"model": "lightgbm", "mean": 0.7016, "std": 0.0286, "adjusted_score": 0.6944, "elapsed_seconds": 345.0}]}},
    "final": {"primary_metric": "roc_auc", "evidence": {"model": "logistic_regression", "holdout_metrics": {"roc_auc": 0.7235},
                                                        "holdout_primary_metric_ci": {"low": 0.6731, "high": 0.7716}, "cv_selected_metric_mean": 0.764,
                                                        "cv_to_holdout_gap": -0.0405, "holdout_rows": 1600, "train_rows": 6400, "baselines": {"majority_class": {"roc_auc": 0.5}},
                                                        "tuning_decision": {"gain": 0.0017, "accepted": True, "required_gain": 0.001}}},
}


def data_card():
    """The card of a small churn table (column names and bands only), as a built project keeps it."""
    import numpy as np
    import pandas as pd

    from dclab_core.datacard import card_of

    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({"customer_id": [f"c{i}" for i in range(n)], "signup_date": pd.date_range("2025-01-01", periods=n, freq="D"),
                          "plan": rng.choice(["basic", "pro"], n), "spend": rng.gamma(2.0, 30.0, n).round(2),
                          "churned": (rng.random(n) < 0.2).astype(int), "churned_60d": (rng.random(n) < 0.3).astype(int)})
    return card_of({"churn.parquet": frame}, target="churned").model_dump(mode="json")


CARD = None


class Projects:
    """The project store's surface the orchestrator reads, on one built project."""

    def __init__(self, home):
        global CARD
        CARD = CARD or data_card()
        self.home = home
        self.doc = {"id": "p1", "name": "Churn", "solution": {"target": "churned", "task": "binary", "prediction_moment": "At the monthly snapshot of each customer."},
                    "running": None, "stages": {k: {"status": "completed"} for k in RECORDS}, "data": {"filename": "churn.parquet", "card": CARD}}

    def get(self, pid):
        return json.loads(json.dumps(self.doc))

    def save(self, doc):
        self.doc = json.loads(json.dumps(doc))

    def update(self, pid, **changes):
        self.doc.update(changes)

    def records(self, pid):
        return RECORDS

    def read_stage(self, pid, name):
        return RECORDS.get(name)

    def log(self, *a, **k):
        pass


class Jobs:
    def __init__(self):
        self.jobs = {}

    def get(self, job_id):
        return self.jobs[job_id]

    def latest(self, kind, key):
        mine = [j for j in self.jobs.values() if j["kind"] == kind and j.get("key") == key]
        return mine[-1] if mine else None


class Services:
    def __init__(self, gateway=None):
        self.home = Path(tempfile.mkdtemp())
        self.projects, self.job_store, self.gateway, self.submitted = Projects(self.home), Jobs(), gateway, []
        self.store = type("S", (), {"home": self.home})()

    def submit(self, kind, key, payload, here=False, **_):
        job = {"id": f"j{len(self.submitted) + 1}", "kind": kind, "key": key, "status": "queued", "payload": payload}
        self.submitted.append(job)
        self.job_store.jobs[job["id"]] = dict(job)
        return job


def call(name, arguments, n):
    return {"content": "", "tool_calls": [{"id": f"c{n}", "name": name, "arguments": arguments}], "usage": {"input_tokens": 10, "output_tokens": 5},
            "assistant_message": {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{n}", "type": "function",
                                                                                       "function": {"name": name, "arguments": json.dumps(arguments)}}]}}


def answer(text):
    return {"content": text, "tool_calls": [], "usage": {"input_tokens": 10, "output_tokens": 5}, "assistant_message": {"role": "assistant", "content": text}}


class Seen:
    prompts = []

    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, messages, tools=None, max_tokens=1800, **options):
        Seen.prompts.append(json.dumps(messages, default=str))
        if not self.replies:
            raise RuntimeError("the script ran out")
        return self.replies.pop(0)


def gateway(replies):
    from dclab_rnd.models import FileUsage, Gateway

    script = Seen(replies)  # one script across the turns: each turn's client reads on from where the last one stopped
    return Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=lambda tier, purpose: script)


CLEAN = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "DCLAB_NO_LIVE_MODELS")} | \
    {"OPENAI_API_KEY": "k", "DCLAB_BUILT_CHAT_SERVE": "model"}  # the model serves here, whatever a stored sealed result says (the review of step 3)


class Chat:
    def __init__(self, test, gw=None):
        self.store = DraftStore(Path(tempfile.mkdtemp()))
        self.draft = self.store.create("Which customers churn next month?")["id"]
        self.store.update(self.draft, lambda d: d.update(status="built", project_id="p1"))
        self.services = Services(gw)
        self.agent = HomeAgent(self.store, None)

    def say(self, text):
        self.agent.reply_built(self.draft, text, orch.Orchestrator(self.services, self.agent), "p1")
        return self.store.get(self.draft)["messages"]

    def since(self, text):
        before = len(self.store.get(self.draft)["messages"])
        return self.say(text)[before:]


class RulesTests(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"})
        env.start()
        self.addCleanup(env.stop)

    def test_asked_which_models_were_trained_it_shows_the_models_table(self):
        chat = Chat(self)
        new = chat.since("what are models you have traimed")
        card = next(m for m in new if m["kind"] == "project_card")["card"]
        self.assertEqual([r["family"] for r in card["rows"]], ["logistic_regression", "extra_trees", "lightgbm"])
        self.assertTrue(card["rows"][0]["chosen"])
        self.assertEqual(card["agent"], "Records reader")
        reply = new[-1]
        self.assertEqual(reply["kind"], "project_answer")
        self.assertIn("Records reader", reply["team"])
        self.assertTrue(reply["chips"])

    def test_how_to_improve_offers_the_ways_and_ok_do_it_starts_them_once(self):
        chat = Chat(self)
        new = chat.since("how can we improve the model?")
        self.assertEqual(next(m for m in new if m["kind"] == "project_card")["card"]["title"], "Ways this model could improve")
        self.assertEqual(chat.services.submitted, [])  # a question gets the ways, not a run
        self.assertIn("Yes, try them", new[-1]["chips"])
        new = chat.since("ok do it")
        self.assertEqual(len(chat.services.submitted), 1)  # the yes carried out the offer
        board = next(m for m in new if m["kind"] == "task_board")
        self.assertEqual(board["tasks"][0]["status"], "running")
        self.assertEqual(board["tasks"][0]["agent"], "Experimenter")
        chat.since("ok")
        self.assertEqual(len(chat.services.submitted), 1)  # the offer was used: a second "ok" starts nothing
        new = chat.since("build improvements")  # live, 2026-10-20: the model started a second run while the first was queued
        self.assertEqual(len(chat.services.submitted), 1)
        self.assertIn("already queued", new[-1]["text"])

    def test_build_improvements_and_improve_it_for_me_start_the_experimenter(self):
        for text in ("build improvements", "ok improve the model for me", "please tune it now"):
            chat = Chat(self)
            chat.say(text)
            self.assertEqual([j["kind"] for j in chat.services.submitted], ["improve"], text)

    def test_several_things_asked_at_once_are_several_tables(self):
        chat = Chat(self)
        new = chat.since("show the evaluation and the models and the leakage")
        titles = [m["card"]["title"] for m in new if m["kind"] == "project_card"]
        self.assertIn("Models trained and compared", titles)
        self.assertIn("Leakage audit", titles)
        self.assertTrue(any(t.startswith("Evaluation") or "evaluation" in t.lower() for t in titles) or "Evaluator" in new[-1]["text"])

    def test_a_failed_job_is_said_on_the_board_and_in_the_chat(self):
        chat = Chat(self)
        chat.say("improve the model for me")
        board = next(m for m in chat.store.get(chat.draft)["messages"] if m["kind"] == "task_board")
        job = chat.services.submitted[0]["id"]
        chat.services.job_store.jobs[job].update(status="failed", error="RuntimeError: no handler for jobs of kind 'improve'")
        o = orch.Orchestrator(chat.services, chat.agent)
        self.assertTrue(o.settle(chat.draft, board["id"], board["tasks"][0]["id"], chat.services.job_store.get(job)))
        msgs = chat.store.get(chat.draft)["messages"]
        self.assertEqual(next(m for m in msgs if m["id"] == board["id"])["tasks"][0]["status"], "failed")
        self.assertIn("restart the app", msgs[-1]["text"])
        o.settle(chat.draft, board["id"], board["tasks"][0]["id"], chat.services.job_store.get(job))
        self.assertEqual(sum("did not finish" in (m.get("text") or "") for m in chat.store.get(chat.draft)["messages"]), 1)  # said once

    def test_the_other_parts_of_the_app_are_reached_too(self):
        chat = Chat(self)
        chat.services.projects.list = lambda limit=None: [chat.services.projects.get("p1")]
        chat.services.projects.activity = lambda pid, limit=60: [{"at": "2026-10-20T10:00:00", "kind": "stage_completed", "payload": {"stage": "final"}}]
        titles = [m["card"]["title"] for m in chat.since("list my projects") if m["kind"] == "project_card"]
        self.assertEqual(titles, ["Your projects"])
        titles = [m["card"]["title"] for m in chat.since("what happened in this project") if m["kind"] == "project_card"]
        self.assertEqual(titles, ["What happened in this project"])
        new = chat.since("what cells were added to the notebook so far")
        self.assertIn("No cell has been added", new[-1]["text"])  # a list, not a new cell

    def test_an_offer_answers_the_next_message_only(self):  # the review: a thank-you two turns later started a run
        chat = Chat(self)
        chat.say("how can it improve?")
        chat.say("show the evaluation")
        chat.say("great, thanks")
        self.assertEqual(chat.services.submitted, [])

    def test_what_was_tried_is_the_last_run_not_the_ideas(self):
        self.assertEqual(orch.plan("what did you try to improve?", None), [("show_improvements", {})])

    def test_no_write_while_the_project_runs_something(self):
        chat = Chat(self)
        chat.services.projects.doc["running"] = "improve"
        chat.services.projects.doc["improvements"] = {"candidates": [{"id": "w", "clears": True, "adoptable": True}], "best": "w"}
        new = chat.since("adopt the best")
        self.assertIn("Something is running", new[-1]["text"])
        self.assertNotIn("adopted_improvement", chat.services.projects.doc)

    def test_a_word_to_build_with_nothing_offered_says_it_is_built(self):
        chat = Chat(self)
        self.assertIn("already built", chat.say("build it")[-1]["text"])

    def test_every_turn_is_kept_whole(self):
        chat = Chat(self)
        chat.say("what are the models")
        rows = [json.loads(l) for f in (chat.services.home / "home_orchestrator").glob("*.jsonl") for l in f.read_text().splitlines()]
        self.assertEqual(rows[-1]["text"], "what are the models")
        self.assertEqual(rows[-1]["path"], "rules")
        self.assertEqual(rows[-1]["cards"], ["Models trained and compared"])

    def test_the_rules_turn_is_a_session_of_its_agent(self):
        from dclab_rnd import experience

        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "wtest")
        experience.install(log)
        self.addCleanup(experience.install, None)
        Chat(self).say("which model is best")
        steps = [e for e in log.read() if e.kind == "agent.step"]
        self.assertIn("show_models", {e.payload.get("tool") for e in steps})


class ProblemAnalystTests(unittest.TestCase):  # step 1 of connecting understanding to the built chat (the owner, 2026-10-21)
    def setUp(self):
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"})
        env.start()
        self.addCleanup(env.stop)

    def test_a_changed_outcome_is_shown_before_and_after_and_a_yes_confirms_it_without_touching_the_project(self):
        chat = Chat(self)
        new = chat.since("actually the outcome should be churned_60d, not churned")
        card = next(m for m in new if m["kind"] == "project_card")["card"]
        self.assertEqual(card["title"], "What would change")
        self.assertEqual(card["rows"], [{"field": "the outcome column", "before": "churned", "after": "churned_60d"}])
        self.assertTrue(any("new problem" in line for line in card["lines"]))  # what it means: a fresh split, everything redone
        self.assertEqual(card["agent"], "Problem analyst")
        self.assertIn("Yes, confirm it", new[-1]["chips"])
        new = chat.since("yes, confirm it")
        draft = chat.store.get(chat.draft)
        self.assertEqual(draft["problem_change"]["status"], "confirmed")
        self.assertEqual(len(draft["problem_versions"]), 1)
        self.assertIn("Confirmed as version", new[-1]["text"])
        self.assertEqual(chat.services.projects.doc["solution"]["target"], "churned")  # the built project is not changed
        self.assertEqual(chat.services.submitted, [])

    def test_confirmed_changes_add_up_and_a_replaced_proposal_is_said(self):  # live, 2026-10-21
        chat = Chat(self)
        chat.say("actually the outcome should be churned_60d, not churned")
        t = orch.Turn(chat.services, "p1", chat.draft, chat.agent, "x")
        self.assertIn("error", orch.confirm_problem_change(t))  # never confirmed without the person having been asked (the review)
        t.offered = {"tool": "confirm_problem_change", "args": {"change": chat.store.get(chat.draft)["problem_change"]["id"]}}
        self.assertIn("confirmed", orch.confirm_problem_change(t, change="outcome: churned -> churned_60d"))  # named in words: still the pending one
        new = chat.since("we should predict churn within 60 days instead")
        self.assertEqual(chat.store.get(chat.draft)["problem_change"]["problem"]["outcome"], "churned_60d")  # on top of the confirmed version
        new = chat.since("no, the prediction should be made at signup")
        card = next(m for m in new if m["kind"] == "project_card")["card"]
        self.assertTrue(any("replaces the earlier proposal" in n for n in card["notes"]))
        facts = dict(next(m for m in chat.since("what are we predicting?") if m["kind"] == "project_card")["card"]["facts"])
        self.assertEqual(facts["The outcome column"], "churned_60d")
        self.assertIn("still answers", " ".join(str(v) for v in facts.values()))

    def test_the_reviews_messages_are_read_as_they_mean(self):  # the review, 2026-10-21: refusals never act; ordinary requests stay ordinary
        cols = ["customer_id", "plan", "churned", "churned_60d"]
        for said in ("don't improve it", "no, don't improve it, show the models", "stop the improvement run", "cancel, don't adopt it",
                     "no don't adopt the best way", "no, don't write a cell"):
            self.assertFalse({n for n, _ in orch.plan(said, {"tool": "improve_model"}, cols)} & {"improve_model", "adopt_improvement", "write_notebook_cell"}, said)
        self.assertEqual([n for n, _ in orch.plan("plot the predictions in a different color", None, cols)], ["write_notebook_cell"])
        self.assertEqual([n for n, _ in orch.plan("change the 30-day features", None, cols)], ["show_features"])
        self.assertNotIn("propose_problem_change", [n for n, _ in orch.plan("why does it predict wrong on the plan column", None, cols)])
        self.assertNotEqual([n for n, _ in orch.plan("improve the predictions instead", None, cols)], ["show_problem"])

    def test_an_unanswered_proposal_expires_and_cannot_be_confirmed_later(self):
        chat = Chat(self)
        chat.say("we should predict churn within 60 days instead")
        chat.say("show the models")  # the offer was not answered
        self.assertEqual(chat.store.get(chat.draft)["problem_change"]["status"], "expired")
        chat.say("yes")
        self.assertNotIn("problem_versions", chat.store.get(chat.draft))

    def test_no_declines_the_proposal(self):
        chat = Chat(self)
        chat.say("we should predict churn within 60 days instead")
        self.assertIn("nothing changes", chat.since("No, keep the problem as built")[-1]["text"])
        self.assertEqual(chat.store.get(chat.draft)["problem_change"]["status"], "declined")

    def test_a_horizon_and_a_moment_are_read_from_the_words(self):
        chat = Chat(self)
        card = next(m for m in chat.since("we should predict churn within 60 days instead") if m["kind"] == "project_card")["card"]
        self.assertEqual(card["rows"][0]["after"], "60 days")
        card = next(m for m in chat.since("no, the prediction should be made at signup") if m["kind"] == "project_card")["card"]
        self.assertEqual(card["rows"][0]["field"], "the prediction moment")

    def test_an_outcome_that_cannot_be_one_is_never_confirmed(self):
        chat = Chat(self)
        new = chat.since("actually the target should be customer_id")
        self.assertIn("does not fit the table", new[-1]["text"])
        chat.since("yes")
        self.assertNotEqual(chat.store.get(chat.draft)["problem_change"]["status"], "confirmed")

    def test_words_that_change_the_problem_without_saying_how_ask_which_part(self):
        chat = Chat(self)
        new = chat.since("this is the wrong problem")
        self.assertEqual(next(m for m in new if m["kind"] == "project_card")["card"]["title"], "The problem this project answers")
        self.assertIn("Which part changes", new[-1]["text"])
        self.assertEqual([n for n, _ in orch.plan("try lightgbm instead", None, ["churned"])], ["show_models"])  # a model is not a problem
        self.assertEqual(orch.plan("actually the outcome should be churned_60d, not churned", None, ["churned", "churned_60d", "converted_this_month"])[0][1]["outcome"],
                         "churned_60d")  # the column after "not" is the old one, whatever its length
        self.assertEqual([n for n, _ in orch.plan("no, show the models", {"tool": "improve_model"}, [])], ["show_models"])
        self.assertEqual(orch.plan("no thanks", {"tool": "improve_model"}, []), [])

    def test_a_column_that_is_not_in_the_table_is_said(self):
        chat = Chat(self)
        t = orch.Turn(chat.services, "p1", chat.draft, chat.agent, "x")
        out = orch.propose_problem_change(t, outcome="revenue_next_year")
        self.assertIn("not a column", out["error"])


class SolutionPlannerTests(unittest.TestCase):  # step 2 (the owner, 2026-10-21): a confirmed problem version becomes a new project version
    def setUp(self):
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"})
        env.start()
        self.addCleanup(env.stop)
        import numpy as np
        import pandas as pd

        from dclab_rnd.studio import ProjectStore, data

        home = Path(tempfile.mkdtemp())
        self.projects = ProjectStore(home / "projects")
        self.store = DraftStore(home / "drafts")
        pid = self.projects.create("Churn", "general", "Which customers churn")["id"]
        rng = np.random.default_rng(1)
        n = 600
        pd.DataFrame({"customer_id": [f"c{i}" for i in range(n)], "signup_date": pd.date_range("2024-01-01", periods=n, freq="D"),
                      "plan": rng.choice(["basic", "pro"], n), "spend": rng.gamma(2.0, 30.0, n).round(2),
                      "churned": (rng.random(n) < 0.2).astype(int), "churned_60d": (rng.random(n) < 0.3).astype(int)}).to_parquet(
            self.projects.data_dir(pid) / "churn.parquet", index=False)
        data.attach_data(self.projects, pid, "churn.parquet")
        project = self.projects.get(pid)
        project["solution"] = {"target": "churned", "task": "binary", "prediction_moment": "At the monthly snapshot of each customer.", "forbidden": [],
                               "identifiers": ["customer_id"], "time_column": "signup_date", "metric": "roc_auc"}
        self.projects.save(project)
        self.pid = pid
        self.draft = self.store.create("Which customers churn next month?")["id"]
        self.store.update(self.draft, lambda d: d.update(status="built", project_id=pid, solution=project["solution"],
                                                          assets=[{"id": "a1", "name": "churn", "status": "ready", "clean_file": "clean.parquet", "kind": "upload"}],
                                                          active_asset="a1"))
        self.built = []
        projects = self.projects

        def build_from_chat(draft_id):
            self.built.append(draft_id)
            return {"project_id": projects.create("Churn v2", "general", "x")["id"], "name": "Churn v2", "started": True, "session_id": "s1"}
        self.services = type("S", (), {"projects": projects, "gateway": None, "job_store": None, "build_from_chat": staticmethod(build_from_chat),
                                       "store": type("H", (), {"home": home})()})()
        self.agent = HomeAgent(self.store, None)

    def say(self, text):
        before = len(self.store.get(self.draft)["messages"])
        self.agent.reply_built(self.draft, text, orch.Orchestrator(self.services, self.agent), self.pid)
        return self.store.get(self.draft)["messages"][before:]

    def test_a_confirmed_outcome_is_planned_then_built_as_a_new_version_on_two_yeses(self):
        self.say("actually the outcome should be churned_60d, not churned")
        self.assertIn("Build the new version", self.say("yes")[-1]["chips"])
        new = self.say("build it")
        card = next(m for m in new if m["kind"] == "project_card")["card"]
        self.assertEqual(card["title"], "Plan for version 2")
        rows = {r["part"]: r for r in card["rows"]}
        self.assertEqual((rows["Outcome"]["v1"], rows["Outcome"]["new"]), ("churned", "churned_60d"))
        self.assertIn("churned", rows["Never used (unknown at the moment)"]["new"])  # the old outcome is forbidden by default
        self.assertEqual(self.built, [])  # a plan, not a build
        new = self.say("yes, build it")
        self.assertEqual(len(self.built), 1)
        nd = self.store.get(self.built[0])
        asset = nd["assets"][0]
        self.assertEqual(asset["suggestion"]["target"], "churned_60d")
        self.assertIn("churned", {f["column"] for f in asset["suggestion"]["forbidden"]})
        self.assertTrue((self.store.data_dir(nd["id"]) / "clean.parquet").exists())
        self.assertEqual(nd["version_of"]["project"], self.pid)
        self.assertIn("churned_60d", [c["name"] for c in nd["analysis"]["columns"]])  # the build reads the outcome among these (the e2e check)
        build = next(m for m in new if m["kind"] == "build")
        self.assertEqual(self.projects.get(build["project_id"])["version_of"]["project"], self.pid)
        self.assertEqual(self.projects.get(self.pid)["solution"]["target"], "churned")  # version 1 untouched
        self.assertEqual(self.store.get(self.draft)["problem_versions"][-1]["built"], build["project_id"])

    def test_nothing_is_built_without_the_yes_to_the_plan(self):
        self.say("actually the outcome should be churned_60d, not churned")
        self.say("yes")
        t = orch.Turn(self.services, self.pid, self.draft, self.agent, "x")
        self.assertIn("error", orch.build_new_version(t))  # no plan offered the turn before
        self.say("build it")
        self.say("show the problem")  # the plan's offer was not answered
        t = orch.Turn(self.services, self.pid, self.draft, self.agent, "x")
        self.assertIn("error", orch.build_new_version(t))
        self.assertEqual(self.built, [])

    def test_the_new_version_never_forbids_less_than_before(self):  # the review, 2026-10-21 (findings 1 and 2)
        project = self.projects.get(self.pid)
        project["solution"]["forbidden"] = [{"column": "spend", "reason": "known only after the month ends"}]
        self.projects.save(project)
        self.say("actually the outcome should be churned_60d, not churned")
        self.say("yes")
        rows = {r["part"]: r for r in next(m for m in self.say("build it") if m["kind"] == "project_card")["card"]["rows"]}
        never = rows["Never used (unknown at the moment)"]["new"]
        self.assertIn("spend", never)  # version 1's forbidden column is carried
        self.assertIn("churned", never)
        later = orch._forbidden_for({"outcome": "plan"}, {"target": "churned", "forbidden": []}, {"forbidden": []},
                                    [{"problem": {"outcome": "churned_60d"}}], "signup_date")
        self.assertEqual([f["column"] for f in later], ["churned", "churned_60d"])  # every earlier version's outcome, never the time column

    def test_build_words_are_strict_and_need_the_persons_yes(self):  # findings 3, 4 and 5
        for said in ("make a plot of it", "create a cell for that", "run that cell", "make it a bar chart", "start it again", "run the notebook cell that plots"):
            self.assertNotIn("plan_new_version", [n for n, _ in orch.plan(said, None, ["churned"], True)], said)
        self.assertEqual(orch.plan("build the new version", {"tool": "build_new_version", "args": {}}, [], True)[0][0], "build_new_version")
        self.say("actually the outcome should be churned_60d, not churned")
        self.say("yes")
        self.say("build it")
        t = orch.Turn(self.services, self.pid, self.draft, self.agent, "show the problem")
        t.offered = {"tool": "build_new_version"}
        self.assertIn("did not say yes", orch.build_new_version(t)["error"])
        self.assertEqual(self.built, [])

    def test_a_refused_build_reuses_its_chat_on_the_retry(self):  # finding 5
        refusals = ["the table is not ready"]
        projects = self.projects

        def build_from_chat(draft_id):
            self.built.append(draft_id)
            if refusals:
                return {"refused": refusals.pop()}
            return {"project_id": projects.create("Churn v2", "general", "x")["id"], "name": "Churn v2", "started": True}
        self.services.build_from_chat = staticmethod(build_from_chat)
        self.say("actually the outcome should be churned_60d, not churned")
        self.say("yes")
        self.say("build it")
        self.assertIn("could not be built", self.say("yes, build it")[-1]["text"])
        self.say("build it")
        self.say("yes, build it")
        self.assertEqual(len(self.built), 2)
        self.assertEqual(self.built[0], self.built[1])  # the same chat, not an orphan per try
        self.assertEqual(sum(1 for m in self.store.get(self.built[1])["messages"] if str(m.get("id", "")).startswith("mv")), 1)

    def test_a_horizon_alone_cannot_be_built_from_this_table_and_says_why(self):
        self.say("we should predict churn within 60 days instead")
        new = self.say("yes")
        self.assertIn("cannot be built from this table", new[-1]["text"])
        new = self.say("build it")
        self.assertIn("horizon", new[-1]["text"])
        self.assertEqual(self.built, [])


class ModelTests(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, CLEAN)
        env.start()
        self.addCleanup(env.stop)
        Seen.prompts = []

    def test_the_model_plans_the_turn_and_a_number_it_made_up_is_dropped(self):
        from dclab_rnd import experience

        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "wtest")
        experience.install(log)
        self.addCleanup(experience.install, None)
        gw = gateway([call("show_models", {}, 1), answer("Logistic regression leads at 0.7624 on the folds. It would gain 0.0912 with more trees.")])
        chat = Chat(self, gw)
        new = chat.since("which model is the best one?")
        self.assertTrue(any(m["kind"] == "project_card" for m in new))
        reply = new[-1]
        self.assertEqual(reply["path"], "model")
        self.assertIn("0.7624", reply["text"])
        self.assertNotIn("0.0912", reply["text"])  # no tool returned it
        steps = [e for e in log.read() if e.kind == "agent.step"]
        self.assertIn("show_models", {e.payload.get("tool") for e in steps})
        for c in (e for e in log.read() if e.kind == "tool.call"):  # every tool call has its step, in the same run
            self.assertIn((c.ids.run, c.payload["tool"]), {(e.ids.run, e.payload.get("tool")) for e in steps})

    def test_the_model_reads_the_conversation_so_a_yes_follows_its_offer(self):
        gw = gateway([call("improvement_ideas", {}, 1), answer("Here are the ways. Shall I try them?"),
                      call("improve_model", {}, 2), answer("Started: the Experimenter is trying them on the training folds.")])
        chat = Chat(self, gw)
        chat.say("how could it be better")
        chat.say("ok")
        self.assertEqual([j["kind"] for j in chat.services.submitted], ["improve"])
        last_prompt = Seen.prompts[-2]  # the request that chose improve_model
        self.assertIn("how could it be better", last_prompt)
        self.assertIn("Shall I try them?", last_prompt)
        self.assertIn("pending_offer", last_prompt)

    def test_a_model_failing_after_a_write_does_not_run_it_again(self):  # the review: the rules replanned and called the write twice
        chat = Chat(self, gateway([call("improve_model", {}, 1)]))  # the second request fails: the script ran out
        new = chat.since("improve the model for me")
        self.assertEqual(len(chat.services.submitted), 1)
        board = next(m for m in new if m["kind"] == "task_board")
        self.assertEqual([t["status"] for t in board["tasks"]], ["running"])
        self.assertIn("Experimenter", new[-2]["text"] if new[-2]["kind"] == "project_answer" else new[-1]["text"] + json.dumps(new))

    def test_the_model_proposes_a_problem_change_and_confirms_it_on_a_yes(self):
        gw = gateway([call("propose_problem_change", {"horizon": "60 days", "said": "churn in 60 days"}, 1), answer("Here is what would change. Confirm?"),
                      call("confirm_problem_change", {}, 2), answer("Confirmed.")])
        chat = Chat(self, gw)
        chat.say("we care about churn in 60 days, not 30")
        chat.say("yes")
        self.assertEqual(chat.store.get(chat.draft)["problem_change"]["status"], "confirmed")
        self.assertIn("table_columns", Seen.prompts[0])  # the model sees the table's column names to name an outcome exactly

    def test_the_rules_reading_is_recorded_beside_the_model_and_the_switch_can_hand_the_chat_to_the_rules(self):  # step 3
        chat = Chat(self, gateway([call("show_models", {}, 1), answer("Logistic regression leads.")]))
        chat.say("which model is the best?")
        rows = [json.loads(l) for f in (chat.services.home / "home_orchestrator").glob("*.jsonl") for l in f.read_text().splitlines()]
        self.assertEqual(rows[-1]["path"], "model")
        self.assertEqual(rows[-1]["rules_shadow"], ["show_models"])  # the rules' own reading, never used
        with mock.patch.object(orch, "_model_serves", return_value=False):
            new = chat.since("show the models")
        self.assertEqual(new[-1]["path"], "rules")

    def test_a_model_whose_every_tool_call_is_refused_leaves_the_turn_to_the_rules(self):  # found on a small local model (2026-10-21)
        chat = Chat(self, gateway([call("show_problem", {"answer": 5}, n) for n in range(1, 7)] + [answer("")]))
        new = chat.since("show me the models")
        self.assertTrue(any(m["kind"] == "project_card" for m in new))  # the rules' table, not the model's refusals
        self.assertEqual(new[-1]["path"], "rules")
        self.assertNotIn("Unknown argument", json.dumps(new))
        rows = [json.loads(l) for f in (chat.services.home / "home_orchestrator").glob("*.jsonl") for l in f.read_text().splitlines()]
        self.assertTrue(rows[-1]["model_steps"] and "model_error" in rows[-1])  # what the model tried is kept

    def test_a_failing_model_leaves_the_turn_to_the_rules(self):
        chat = Chat(self, gateway([]))  # the script is empty: the first request fails
        new = chat.since("show me the models")
        self.assertTrue(any(m["kind"] == "project_card" for m in new))
        self.assertEqual(new[-1]["path"], "rules")


class GroundingTests(unittest.TestCase):
    def test_a_list_repeating_the_table_is_taken_out(self):  # the owner: "bring the table instead of the texts"
        kept, out = orch.without_lists("Five families were compared.\n1. logistic_regression 0.7624\n2. extra_trees 0.7274\nLogistic regression leads.")
        self.assertEqual(kept, "Five families were compared.\nLogistic regression leads.")
        self.assertEqual(len(out), 2)

    def test_rounded_and_percent_numbers_count_as_returned(self):
        kept, dropped = orch.grounded("The score is 0.76. That is 76% of the cases. It has 3 folds. The gain is 0.21.", [{"mean": 0.7624}])
        self.assertIn("0.76", kept)
        self.assertIn("76%", kept)
        self.assertIn("3 folds", kept)
        self.assertEqual(dropped, ["The gain is 0.21."])


if __name__ == "__main__":
    unittest.main()
