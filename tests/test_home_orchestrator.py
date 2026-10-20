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


class Projects:
    """The project store's surface the orchestrator reads, on one built project."""

    def __init__(self, home):
        self.home = home
        self.doc = {"id": "p1", "name": "Churn", "solution": {"target": "churned", "task": "binary"}, "running": None,
                    "stages": {k: {"status": "completed"} for k in RECORDS}}

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


CLEAN = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "DCLAB_NO_LIVE_MODELS")} | {"OPENAI_API_KEY": "k"}


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
