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
        from dclab_rnd.storage.notes import merge

        if "memory" in changes:  # as the real stores do: notes are added to what is stored
            changes = {**changes, "memory": merge(self.doc.get("memory"), changes["memory"])}
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
        self.assertIn("did not say yes", orch.confirm_problem_change(t)["error"])  # asked is not agreed (Claude, live, 2026-10-21)
        t.text = "build a new version with 5 folds"
        self.assertIn("did not say yes", orch.confirm_problem_change(t)["error"])
        t.text = "yes, use churned_60d"
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


class ProjectMemoryTests(unittest.TestCase):  # the Home team remembers decisions and preferences across chats (the owner, 2026-10-21)
    def setUp(self):
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"})
        env.start()
        self.addCleanup(env.stop)

    def other_chat(self, chat):
        """A second chat on the same project."""
        draft = chat.store.create("Same project, another day")["id"]
        chat.store.update(draft, lambda d: d.update(status="built", project_id="p1"))
        return draft

    def test_a_preference_is_kept_and_another_chat_on_the_project_reads_it(self):
        chat = Chat(self)
        new = chat.since("remember that we prefer fewer false alarms over catching every case")
        notes = chat.services.projects.doc["memory"]
        self.assertEqual([(n["kind"], n["move"], n["who"]) for n in notes], [("preference", "remember", "person")])
        self.assertEqual(notes[0]["decision"], "we prefer fewer false alarms over catching every case")
        self.assertIn("Kept in the project's memory: \"we prefer fewer false alarms", new[-1]["text"])
        other = self.other_chat(chat)
        before = len(chat.store.get(other).get("messages") or [])
        chat.agent.reply_built(other, "what do you remember?", orch.Orchestrator(chat.services, chat.agent), "p1")
        card = next(m for m in chat.store.get(other)["messages"][before:] if m["kind"] == "project_card")["card"]
        self.assertEqual(card["title"], "What the team remembers")
        self.assertEqual(card["rows"][0]["what"], "we prefer fewer false alarms over catching every case")

    def test_questions_complaints_and_problem_changes_are_never_kept_as_preferences(self):  # the review, 2026-10-21
        for said in ("why do we always lose recall?", "I always get confused by AUC, what is it?", "we never had churn labels before 2020, is that a problem?",
                     "I never asked for that", "note how many rows are missing", "we always predict at signup, not at renewal", "from now on, predict at signup",
                     "remember: the outcome is churned_60d", "Never-mind", "never mind", "I don't want to improve it", "no thanks"):
            self.assertNotIn("remember", [n for n, _ in orch.plan(said, None, ["churned", "churned_60d"])], said)
        self.assertNotEqual(orch.plan("the model uses too much memory", None)[:1], [("show_memory", {})])
        for said in ("remember that we prefer fewer false alarms", "I prefer fewer false alarms", "from now on explain results in plain words",
                     "always explain the numbers simply", "please keep in mind that the team reviews 50 cases a day"):
            self.assertEqual(orch.plan(said, None)[0][0], "remember", said)

    def test_the_model_cannot_keep_a_note_the_person_did_not_ask_for(self):  # the review: it would be labelled the person's
        chat = Chat(self)
        t = orch.Turn(chat.services, "p1", chat.draft, chat.agent, "show the models, I don't care about recall today")
        self.assertIn("error", orch.remember(t, note="we don't care about recall"))
        self.assertNotIn("memory", chat.services.projects.doc)

    def test_a_job_saving_the_whole_project_keeps_a_note_written_meanwhile(self):  # the real file store
        from dclab_rnd.studio.store import ProjectStore

        store = ProjectStore(Path(tempfile.mkdtemp()))
        p = store.create("Churn")
        job_copy = store.get(p["id"])  # a job reads the project ...
        store.update(p["id"], memory=[orch._new_note("preference", "we prefer recall", "asked", "person", "remember")])
        store.update(p["id"], memory=[orch._new_note("preference", "explain simply", "asked", "person", "remember")])  # a second chat
        job_copy["stages"] = {"final": {"status": "completed"}}
        store.save(job_copy)  # ... and saves all of it minutes later
        self.assertEqual([n["decision"] for n in store.get(p["id"])["memory"]], ["we prefer recall", "explain simply"])
        from dclab_rnd.studio import memory as studio_memory

        note = store.get(p["id"])["memory"][0]["id"]
        stale = store.get(p["id"])
        studio_memory.remove(store, p["id"], note, "person", "not true any more")
        store.save(stale)  # a copy read before the removal does not bring it back
        self.assertTrue(store.get(p["id"])["memory"][0]["removed"])

    def test_a_confirmed_problem_change_is_remembered(self):
        chat = Chat(self)
        chat.say("actually the outcome should be churned_60d, not churned")
        self.assertNotIn("memory", chat.services.projects.doc)  # a proposal is not a decision
        chat.say("yes, confirm it")
        note = chat.services.projects.doc["memory"][-1]
        self.assertEqual((note["kind"], note["move"], note["who"]), ("decision", "confirm_problem_change", "person"))
        self.assertIn("churned_60d", note["decision"])
        self.assertIn("still answers the problem it was built for", note["reason"])

    def test_the_words_kept_are_redacted_and_declines_are_not_preferences(self):
        chat = Chat(self)
        chat.say("remember to send the results to dana@example.com")
        self.assertNotIn("dana@example.com", json.dumps(chat.services.projects.doc["memory"]))
        for said in ("I don't want to improve it", "never mind", "no thanks"):
            self.assertNotIn("remember", [n for n, _ in orch.plan(said, None)], said)
        self.assertEqual(orch.plan("what did we decide?", None)[0][0], "show_memory")

    def test_the_model_reads_the_memory(self):
        with mock.patch.dict(os.environ, CLEAN):
            Seen.prompts = []
            chat = Chat(self, gateway([answer("Noted.")]))
            chat.services.projects.doc["memory"] = [orch._new_note("preference", "we prefer recall over precision", "the person asked", "person", "remember")]
            chat.say("hello")
        self.assertTrue(any("we prefer recall over precision" in p for p in Seen.prompts))


class SeveralTasksTests(unittest.TestCase):  # "improve and plot in parallel" (the owner, 2026-10-21)
    def setUp(self):
        env = mock.patch.dict(os.environ, {**CLEAN, "DCLAB_BUILT_CHAT_SERVE": "rules"})  # the rules plan; a model client writes the cells
        env.start()
        self.addCleanup(env.stop)
        cell = {"id": "c1", "request": "plot", "code": "plt.plot(fold_scores)", "explanation": "", "problems": [], "status": "proposed"}
        write = mock.patch("dclab_rnd.studio.notebook_agent.write", return_value=cell)
        self.write = write.start()
        self.addCleanup(write.stop)

    def test_one_message_with_two_asks_starts_the_improvement_and_writes_the_plot(self):
        self.assertEqual([n for n, _ in orch.plan("improve the model for me and plot each fold's score", None)], ["improve_model", "write_notebook_cell"])
        chat = Chat(self, gateway([]))
        new = chat.since("improve the model for me and plot each fold's score")
        self.assertEqual([j["kind"] for j in chat.services.submitted], ["improve"])
        self.assertTrue(any(m["kind"] == "notebook_cell" for m in new))
        self.assertEqual(self.write.call_args.args[2], "plot each fold's score")  # the cell is asked for its own part
        board = [x for m in new if m["kind"] == "task_board" for x in m["tasks"]]
        self.assertTrue({x["agent"] for x in board} >= {"Experimenter", "Notebook coder"}, board)  # both on one board

    def test_a_plot_is_written_while_the_improvement_runs_but_not_while_a_stage_runs(self):
        chat = Chat(self, gateway([]))
        chat.services.projects.doc["running"] = "improve"
        self.assertTrue(any(m["kind"] == "notebook_cell" for m in chat.since("plot the confusion matrix")))
        chat.services.projects.doc["running"] = "final"  # a stage holds the whole project for minutes: a cell would be lost
        new = chat.since("plot the confusion matrix")
        self.assertFalse(any(m["kind"] == "notebook_cell" for m in new))
        self.assertIn("Something is running", new[-1]["text"])

    def test_yes_but_is_not_a_yes_and_a_plot_made_better_is_not_a_model_run(self):  # the review, 2026-10-21
        for said in ("right, but show me the leakage first", "yes but plot the roc curve instead", "yes, but not now, show the models"):
            for offered in ("confirm_problem_change", "improve_model"):
                self.assertNotIn(offered, [n for n, _ in orch.plan(said, {"tool": offered})], said)
        for said in ("plot spend by month, and make it look better", "plot the roc curve and try a better threshold", "plot the roc curve, then increase the font size"):
            self.assertEqual([n for n, _ in orch.plan(said, None)], ["write_notebook_cell"], said)
        self.assertEqual([n for n, _ in orch.plan("yes, and add column spend2 = spend * 2", {"tool": "confirm_problem_change"})],
                         ["confirm_problem_change", "propose_data_change"])
        chat = Chat(self, gateway([]))
        chat.store.update(chat.draft, lambda d: d.setdefault("agent", {}).update(offer={"tool": "improve_model", "args": {}}))
        chat.since("yes but plot the roc curve instead")
        self.assertEqual(chat.services.submitted, [])  # no job started

    def test_the_model_cannot_start_the_run_on_a_yes_held_back(self):  # live with Claude, 2026-10-21
        for said, starts in (("yes, but show me the leakage first", False), ("yes", True), ("not now, maybe later", False), ("yes, and plot each fold", True)):
            chat = Chat(self, gateway([call("improve_model", {}, 1), answer("Done.")]))
            chat.store.update(chat.draft, lambda d: d.setdefault("agent", {}).update(offer={"tool": "improve_model", "args": {}}))
            with mock.patch.dict(os.environ, {"DCLAB_BUILT_CHAT_SERVE": "model"}):
                chat.since(said)
            self.assertEqual(bool(chat.services.submitted), starts, said)
        chat = Chat(self, gateway([call("improve_model", {}, 1), answer("Done.")]))
        with mock.patch.dict(os.environ, {"DCLAB_BUILT_CHAT_SERVE": "model"}):
            chat.since("how can the model improve?")  # a question: ideas, not a run
        self.assertEqual(chat.services.submitted, [])
        chat = Chat(self, gateway([call("improve_model", {}, 1), answer("Here is the audit."), call("improve_model", {}, 2), answer("Started.")]))
        chat.store.update(chat.draft, lambda d: d.setdefault("agent", {}).update(offer={"tool": "improve_model", "args": {}}))
        with mock.patch.dict(os.environ, {"DCLAB_BUILT_CHAT_SERVE": "model"}):
            chat.since("yes, but show me the leakage first")
            self.assertEqual(chat.services.submitted, [])
            chat.since("yes")  # the offer is still open: this yes starts it
        self.assertEqual([j["kind"] for j in chat.services.submitted], ["improve"])

    def test_a_worker_in_another_process_waits_for_the_project_lock(self):  # the file store; PostgreSQL locks the row
        import subprocess
        import time

        from dclab_rnd.studio.store import ProjectStore

        store = ProjectStore(Path(tempfile.mkdtemp()))
        pid = store.create("Churn")["id"]
        holder = subprocess.Popen([sys.executable, "-c", (
            "import sys, time; sys.path.insert(0, %r)\n"
            "from dclab_rnd.studio.store import ProjectStore, _project_lock\n"
            "s = ProjectStore(%r)\n"
            "with _project_lock(%r, s.directory(%r)):\n"
            "    print('held', flush=True); time.sleep(1.5)\n") % (str(ROOT), str(store.home), pid, pid)], stdout=subprocess.PIPE, text=True)
        self.assertEqual(holder.stdout.readline().strip(), "held")
        started = time.monotonic()
        store.update(pid, notebook_agent_cells=[{"id": "c1"}])
        waited = time.monotonic() - started
        holder.wait(10)
        self.assertGreater(waited, 0.8)  # it waited for the other process, rather than writing under it
        self.assertEqual(store.get(pid)["notebook_agent_cells"], [{"id": "c1"}])

    def test_a_yes_and_a_single_ask_are_never_split(self):
        self.assertEqual([n for n, _ in orch.plan("yes, and plot each fold's score", {"tool": "improve_model"})], ["improve_model", "write_notebook_cell"])
        self.assertEqual([n for n, _ in orch.plan("yes, and plot it", {"tool": "build_new_version"})], ["write_notebook_cell"])  # building: a plain yes only
        self.assertEqual([n for n, _ in orch.plan("plot the spend and the tenure", None)], ["write_notebook_cell"])
        self.assertEqual([n for n, _ in orch.plan("how can it improve and add column b = spend / 2", None)], ["improvement_ideas"])  # one offer a turn


class ReplyLengthTests(unittest.TestCase):  # live with Claude, 2026-10-21: four sentences for "two", and "Do you want me to?" alone
    def test_a_short_follow_up_keeps_the_sentence_it_follows(self):
        said = ("The last run found two ways. One is a recipe (about +0.005). The other is a blend (about +0.003). Neither can be adopted here. "
                "You already confirmed a version 2, so I can plan that version next. Do you want me to?")
        short = orch.shorten(said, 4)
        self.assertTrue(short.endswith("so I can plan that version next. Do you want me to?"), short)
        self.assertEqual(len(re_sentences(short)), 4)
        said = ("Logistic regression is the chosen model. It scored about 0.84 on the folds. The last run tried seven ways. Two cleared the margin. "
                "The next step is to plan the confirmed version 2. Say \"plan it\" and I'll run it.")  # live: the request kept its step
        self.assertTrue(orch.shorten(said, 4).endswith("The next step is to plan the confirmed version 2. Say \"plan it\" and I'll run it."))

    def test_the_persons_length_is_kept(self):
        self.assertEqual([orch.asked_length(x) for x in ("Summarize the project for my boss in two sentences", "in one line please",
                                                          "give me a 3-sentence summary", "explain it in a single sentence", "tell me everything")],
                         [2, 1, 3, 1, None])
        with mock.patch.dict(os.environ, CLEAN):
            Seen.prompts = []
            chat = Chat(self, gateway([call("show_result", {}, 1), answer("It ranks customers by their risk of leaving. It scored 0.7235 on the held-back cases. "
                                                                            "It needs a later period to be trusted. The report can be downloaded.")]))
            reply = chat.since("Explain the held-back result in two sentences")[-1]["text"]
        self.assertEqual(len(re_sentences(reply)), 2, reply)
        self.assertTrue(any("at most 2 sentence" in p for p in Seen.prompts))  # the model is told too


def re_sentences(text):
    import re

    return [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]


class ReporterTests(unittest.TestCase):  # "write me the report for my manager", cited and from the records (the owner, 2026-10-21)
    def setUp(self):
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"})
        env.start()
        self.addCleanup(env.stop)
        from dclab_core.contracts import ReportSection, SolutionReport

        report = SolutionReport(title="Solution report: Churn", sections=[
            ReportSection(title="The problem as understood", body="- Predict churned at the monthly snapshot.\n- Binary outcome.", cites=["brief.target"]),
            ReportSection(title="What the data shows", body="- 6400 training rows.", cites=["data.evidence.train_rows"]),
            ReportSection(title="Results, with their intervals", body="- roc_auc 0.7235 on the held-back cases (95% 0.6731 to 0.7716).", cites=["final.evidence.holdout_metrics"]),
            ReportSection(title="Limits, and what the data cannot show", body="- One time period only.", cites=[]),
            ReportSection(title="Next steps", body="- Monitor the score monthly.", cites=[])], numbers={"final.evidence.holdout_metrics.roc_auc": 0.7235})
        patch = mock.patch("dclab_rnd.studio.solution_report.latest", return_value={"report": report, "markdown": "# r", "cached": False})
        patch.start()
        self.addCleanup(patch.stop)

    def test_the_manager_gets_the_problem_result_limits_and_next_steps_with_the_full_report_to_download(self):
        chat = Chat(self)
        chat.services.projects.doc["memory"] = [orch._new_note("preference", "we prefer fewer false alarms", "asked", "person", "remember")]
        new = chat.since("write me the report for my manager")
        card = next(m for m in new if m["kind"] == "project_card")["card"]
        self.assertEqual((card["agent"], card["title"]), ("Reporter", "Report for your manager"))
        self.assertEqual([line.split(":")[0] for line in card["lines"][:-1]],
                         ["The problem as understood", "The problem as understood", "Results, with their intervals", "Limits, and what the data cannot show", "Next steps"])
        self.assertIn("0.7235", " ".join(card["lines"]))
        self.assertNotIn("6400", " ".join(card["lines"]))  # the data section is for the team
        self.assertEqual(card["lines"][-1], "What you decided: we prefer fewer false alarms")
        self.assertEqual(card["download"], {"route": "exportSolutionReport", "label": "Download the full report"})
        self.assertIn("final.evidence.holdout_metrics", card["proof"])

    def test_report_requests_reach_the_reporter_and_other_summaries_do_not(self):  # the review, 2026-10-21
        for said in ("write me the report for my manager", "Summarize the project for my manager", "Can you write up the results for my boss?",
                     "Download the report", "I need the final report", "give me a summary of the project", "prepare a report for the board"):
            self.assertEqual([n for n, _ in orch.plan(said, None)][:1], ["write_report"], said)
        for said, other in (("Remember that we need a monthly report for the board", "remember"), ("give me a summary of the data", "show_data"),
                            ("I want a summary of the leakage findings", "show_leakage"), ("Can you give me a summary of the models trained?", "show_models"),
                            ("Give me a summary of the stages", "show_stages"), ("write a cell with the classification report", "write_notebook_cell"),
                            ("Plot the confusion matrix and give me a summary", "write_notebook_cell"),
                            ("give me the report of the experiment you just ran", "show_improvements"),
                            ("Show me the results of the improvements and write a summary", None)):
            planned = [n for n, _ in orch.plan(said, None)]
            self.assertNotIn("write_report", planned, said)
            if other:
                self.assertIn(other, planned, said)

    def test_without_the_final_stage_the_reporter_says_what_is_missing(self):
        chat = Chat(self)
        with mock.patch.dict(RECORDS):
            del RECORDS["final"]
            result = orch.write_report(orch.Turn(chat.services, "p1", chat.draft, chat.agent, "write the report"))
        self.assertIn("error", result)

    def test_the_team_gets_the_data_and_decisions_too_and_the_improvement_report_is_not_this_one(self):
        self.assertEqual(orch.plan("write the report for the team", None), [("write_report", {"audience": "team"})])
        for said in ("what does the report say?", "show me the report of the last improvement run"):
            self.assertNotIn("write_report", [n for n, _ in orch.plan(said, None)], said)
        chat = Chat(self)
        card = next(m for m in chat.since("Write the report for the team") if m["kind"] == "project_card")["card"]
        self.assertEqual(card["title"], "Report for the team")
        self.assertIn("6400", " ".join(card["lines"]))


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

    def test_a_model_that_only_talks_where_the_rules_would_act_leaves_the_turn_to_the_rules(self):  # qwen2.5:1.5b on Ollama, 2026-10-21
        said = ("For the project's next version, 5-fold cross-validation is proposed. The stage runs have been completed. "
                "Now, a new version is awaiting a confirmation.")  # nothing was planned: a claim no tool backs
        chat = Chat(self, gateway([answer(said)]))
        new = chat.since("show me the models")
        self.assertEqual(new[-1]["path"], "rules")
        self.assertTrue(any(m["kind"] == "project_card" for m in new))
        self.assertNotIn("awaiting a confirmation", json.dumps([m.get("text") for m in new]))

    def test_a_tool_call_written_as_text_or_an_empty_reply_leaves_the_turn_to_the_rules(self):  # qwen2.5-coder:1.5b on Ollama, 2026-10-21
        for said in ('To do that, call the `solve` function:\n\n```json\n{\n  "name": "solve",\n}\n```',
                     '```json\n{"name": "improvement_ideas", "arguments": {"problem": {}}}\n```', "ok"):
            with self.subTest(said=said[:20]):
                chat = Chat(self, gateway([answer(said)]))
                new = chat.since("show me the models")
                self.assertEqual(new[-1]["path"], "rules")
                self.assertTrue(any(m["kind"] == "project_card" for m in new))  # the rules' table
                self.assertNotIn('"name"', json.dumps([m.get("text") for m in new]))

    def test_a_refused_cell_and_a_failing_model_never_write_the_cell_twice(self):  # the review of 2026-10-21: the rules replanned the write
        refused = {"id": "c1", "request": "plot", "code": "x = 1", "explanation": "", "problems": ["reads the holdout"], "status": "refused"}
        with mock.patch("dclab_rnd.studio.notebook_agent.write", return_value=refused) as write:
            chat = Chat(self, gateway([call("write_notebook_cell", {"request": "plot the confusion matrix"}, 1)]))  # then the script runs out
            new = chat.since("plot the confusion matrix")
        self.assertEqual(write.call_count, 1)
        self.assertEqual(sum(1 for m in new if m["kind"] == "notebook_cell"), 1)

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



class DataEngineerTests(unittest.TestCase):  # "add this column", "join this table", "what's missing in the data" (the owner, 2026-10-21)
    setUp = SolutionPlannerTests.setUp
    say = SolutionPlannerTests.say

    def card(self, new):
        return next(m for m in new if m["kind"] == "project_card")["card"]

    def other_chat_table(self, name, frame):
        import pandas as pd  # noqa: F401

        d = self.store.create(f"the {name} table")["id"]
        frame.to_parquet(self.store.data_dir(d) / "clean.parquet", index=False)
        self.store.update(d, lambda doc: doc.update(assets=[{"id": "b1", "name": name, "status": "ready", "clean_file": "clean.parquet"}]))
        return d

    def test_a_new_column_is_checked_confirmed_planned_and_built_in_a_new_version_only(self):
        from dclab_rnd.studio import data as studio_data

        new = self.say("add column spend_per_day = spend / 30")
        card = self.card(new)
        self.assertEqual((card["agent"], card["title"]), ("Data engineer", "What would change in the data"))
        self.assertIn("add spend_per_day = spend / 30", card["lines"][0])
        self.assertIn("Confirmed as version", self.say("yes")[-1]["text"])
        version = self.store.get(self.draft)["problem_versions"][-1]
        self.assertEqual([o["name"] for o in version["data_ops"]], ["spend_per_day"])
        self.assertEqual(self.projects.get(self.pid)["memory"][-1]["move"], "confirm_data_change")
        plan = self.card(self.say("yes"))
        self.assertTrue(any("Data changes in this version" in line for line in plan["lines"]))
        self.say("build it")
        self.assertEqual(len(self.built), 1)
        import pandas as pd

        built = pd.read_parquet(self.store.data_dir(self.built[0]) / "clean.parquet")
        self.assertIn("spend_per_day", built.columns)
        new_card = self.store.get(self.built[0])["data_card"]
        self.assertIn("spend_per_day", [c["name"] for tb in new_card["tables"] for c in tb["columns"]])
        v1 = studio_data.load_table(studio_data.data_path(self.projects, self.projects.get(self.pid)))
        self.assertNotIn("spend_per_day", v1.columns)  # the built project's table never changes

    def test_a_column_from_the_outcome_or_a_forbidden_one_or_code_is_refused(self):
        project = self.projects.get(self.pid)
        project["solution"]["forbidden"] = [{"column": "churned_60d", "reason": "after the moment"}]
        self.projects.save(project)
        for said, why in (("add column x = churned * 2", "outcome"), ("add column y = churned_60d + 1", "forbidden"),
                          ("add column z = __import__('os').getcwd()", "cannot make")):
            new = self.say(said)
            self.assertIn(why, new[-1]["text"], said)
            self.assertNotIn("data_change", self.store.get(self.draft))
        self.assertIn("formula", self.say("add a new column for the spend per day")[-1]["text"])

    def test_a_lookup_table_from_another_chat_joins_and_an_event_table_is_refused(self):
        import pandas as pd

        self.other_chat_table("plans", pd.DataFrame({"plan": ["basic", "pro"], "price": [5.0, 9.0]}))
        self.other_chat_table("orders", pd.DataFrame({"plan": ["basic", "basic", "pro"], "amount": [1.0, 2.0, 3.0]}))
        card = self.card(self.say("join plans on plan"))
        self.assertIn("join plans on plan: 1 column (price), found for 100% of rows", card["lines"][0])
        self.say("no")
        self.assertIn("table of events", self.say("join orders on plan")[-1]["text"])
        self.assertIn("no table named", self.say("join tickets on plan")[-1]["text"])

    def test_a_joined_copy_of_a_later_outcome_is_left_out_of_that_version(self):  # the review: renamed for the clash, still the outcome
        import pandas as pd

        self.other_chat_table("plans", pd.DataFrame({"plan": ["basic", "pro"], "churned_60d": [0, 1], "price": [5.0, 9.0]}))
        self.say("join plans on plan")
        self.say("yes")
        self.say("actually the outcome should be churned_60d, not churned")
        new = self.say("yes, confirm it")
        self.assertEqual(self.store.get(self.draft)["problem_versions"][-1]["data_ops"], [])
        self.assertIn("Left out of this version", new[-1]["text"])

    def test_a_column_from_a_later_versions_outcome_is_refused_in_either_order(self):  # the review: AUC 0.87 from a noisy copy
        self.say("actually the outcome should be churned_60d, not churned")
        self.say("yes")
        self.assertIn("outcome", self.say("add column leak = churned_60d + spend / 60")[-1]["text"])
        self.assertNotIn("data_change", self.store.get(self.draft))

    def test_a_data_change_that_reads_a_new_outcome_is_left_out_of_that_version(self):
        self.say("add column flag60 = churned_60d * 1")
        self.say("yes")
        self.say("actually the outcome should be churned_60d, not churned")
        new = self.say("yes, confirm it")
        version = self.store.get(self.draft)["problem_versions"][-1]
        self.assertEqual(version["data_ops"], [])
        self.assertIn("Left out of this version", new[-1]["text"])

    def test_a_join_on_the_outcome_a_changed_table_and_a_clashing_name_are_refused(self):
        import pandas as pd

        from dclab_rnd.studio import derive

        self.other_chat_table("labels", pd.DataFrame({"churned": [0, 1], "status": ["kept", "lost"]}))
        self.assertIn("outcome", self.say("join labels on churned")[-1]["text"])
        frame = pd.DataFrame({"plan": ["a", "b"], "churn": [1, 2], "plans_churn": [0, 1]})
        with self.assertRaises(derive.ChangeRefused):
            derive.check_join(frame, pd.DataFrame({"plan": ["a", "b"], "churn": [7, 8]}), "plans", "plan", "plans_churn", [])
        plans = pd.DataFrame({"plan": ["a", "b"], "price": [1.0, 2.0]})
        op = derive.check_join(frame, plans, "plans", "plan", "x", [])
        with self.assertRaises(derive.ChangeRefused):  # the table gained a column after it was checked
            derive.apply(frame, [op], lambda name: plans.assign(churned=[0, 1]))

    def test_the_chain_keeps_data_changes_and_the_problem_after_a_version_is_built(self):
        self.say("actually the outcome should be churned_60d, not churned")
        self.say("yes")
        self.say("yes")
        self.say("build it")
        self.say("add column spend2 = spend * 2")
        self.say("yes")
        version = self.store.get(self.draft)["problem_versions"][-1]
        self.assertEqual(version["problem"]["outcome"], "churned_60d")  # not back to version 1's outcome
        self.assertEqual([o["name"] for o in version["data_ops"]], ["spend2"])

    def test_text_columns_and_requests_that_are_not_data_changes(self):
        import pandas as pd

        from dclab_rnd.studio import derive

        frame = pd.DataFrame({"plan": ["basic", None, "pro"], "day": ["2025-01-01", "2025-01-05", "2025-02-01"]})
        self.assertEqual(derive.compute(frame, "is_missing(plan)").tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(derive.compute(frame, "month(day)").tolist(), [1.0, 1.0, 2.0])
        with self.assertRaises(derive.ChangeRefused):
            derive.compute(pd.DataFrame({"t": [1, 2]}), "t ** -1")
        for said in ("add a column chart of churn by plan", "create a feature importance chart", "compute the feature importance",
                     "derive the feature list", "look up the score on the holdout", "can I join the team?", "join us",
                     "add a field note: churn is monthly", "what's the null hypothesis?", "what's missing from the report?"):
            self.assertNotIn("propose_data_change", [n for n, _ in orch.plan(said, None)], said)
            self.assertNotIn("data_gaps", [n for n, _ in orch.plan(said, None)], said)

    def test_what_is_missing_reads_the_data_card(self):
        self.assertEqual(orch.plan("what's missing in the data?", None)[0][0], "data_gaps")
        self.assertEqual(orch.plan("are there missing values?", None)[0][0], "data_gaps")
        card = self.card(self.say("what is missing in the data?"))
        self.assertEqual((card["agent"], card["title"]), ("Data engineer", "What is missing in the data"))


if __name__ == "__main__":
    unittest.main()
