"""Written first requests to Home (P6 of the 2026-10-29 plan): code assigns every request's kind of help, language and
style before it is written; the writer never chooses a label; a request the plain reading disagrees with is kept apart
and never trained on; every request and reply is kept whole, the model's thinking with it; a run is capped at EUR 1."""

import json
import os
import sys
import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval.home import intent_requests as I  # noqa: E402

ORIG_BATCHES = I.batches


class FakeClient:
    model = "claude-sonnet-5-5"

    def __init__(self, reply_for):
        self.reply_for, self.spent_eur, self.sent, self.run_limit_eur = reply_for, 0.0, [], None

    def complete(self, messages, tools=None, max_tokens=None):
        self.sent.append(messages)
        self.spent_eur += 0.01
        return {"content": self.reply_for(messages), "finish_reason": "stop", "usage": {"output_tokens": 10}}


def writer_reply(messages):
    """A reply that writes every slot and also tries to name a kind of its own (which must be ignored)."""
    slots = json.loads(messages[1]["content"])["slots"]
    return json.dumps({"requests": [{"slot": s["slot"], "text": f"request for {s['slot']}", "kind": "advice"} for s in slots]
                       + [{"slot": "not-in-this-batch", "text": "stray"}]})


def plain_reply(messages):
    """The plain reading: right on every request but those of a slot ending in 0 (read as advice)."""
    text = json.loads(messages[1]["content"])["request"]
    track = "advice" if text.endswith("0") else I.slots_by_text[text]
    return json.dumps({"tracks": [track], "exit": "answer", "fields": {}})


class Base(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.home = Path(tempfile.mkdtemp())
        files = mock.patch.multiple(I, OUT=self.home / "out.json", FLAGGED=self.home / "flagged.json", MANIFEST=self.home / "manifest.json",
                                    BLIND=self.home / "blind.json", BLIND_ANSWERS=self.home / "answers.json")
        files.start()
        self.addCleanup(files.stop)
        self.clients = []

    def client_for(self, reply_for):
        def make(purpose, limit):
            c = FakeClient(reply_for)
            c.purpose = purpose
            self.clients.append(c)
            return c
        return make


class Slots(Base):
    def test_the_tracks_are_homes(self):
        from dclab_rnd.draft import intent

        self.assertEqual(I.TRACKS, intent.TRACKS)

    def test_code_draws_every_truth_in_the_planned_shares(self):
        s = I.slots()
        self.assertEqual(len(s), 600)
        self.assertEqual(s, I.slots())  # the same seed, the same truth
        self.assertEqual(Counter(r["kind"] for r in s), {t: 100 for t in I.TRACKS})
        self.assertEqual(Counter(r["language"] for r in s), {"en": 360, "fa": 150, "mixed": 90})
        self.assertEqual(Counter(r["need"] for r in s if r["kind"] == "analysis")["segments"], 30)
        for t in I.TRACKS:
            self.assertEqual(Counter(r["writer"] for r in s if r["kind"] == t), {"a": 50, "b": 50})
        for r in s:
            self.assertIn(r["need"], I.NEEDS[r["kind"]])
            self.assertEqual("second_kind" in r, r["style"] == "two_kinds")
            if r["style"] == "two_kinds":
                self.assertNotEqual(r["second_kind"], r["kind"])
        self.assertTrue(all(len(b) == I.PER_BATCH and len({r["kind"] for r in b}) == 1 for b in I.batches("a").values()))

    def test_writer_c_writes_english_only_with_its_own_slots(self):
        c = I.slots_c()
        self.assertEqual(len(c), 300)
        self.assertEqual({r["language"] for r in c}, {"en"})  # the owner, 2026-10-29: "for now you can skip persian"
        self.assertEqual(Counter(r["kind"] for r in c), {t: 50 for t in I.TRACKS})
        self.assertFalse({r["slot"] for r in c} & {r["slot"] for r in I.slots()})
        self.assertEqual(I.request("c", next(iter(I.batches("c").values())))[0]["content"],
                         __import__("dclab_rnd.prompts", fromlist=["text"]).text("home_request_writer_c"))

    def test_a_request_holds_the_slots_and_the_writers_prompt(self):
        batch = next(iter(I.batches("b").values()))
        messages = I.request("b", batch)
        self.assertEqual(messages[0]["content"], __import__("dclab_rnd.prompts", fromlist=["text"]).text("home_request_writer_b"))
        sent = json.loads(messages[1]["content"])["slots"]
        self.assertEqual([s["slot"] for s in sent], [r["slot"] for r in batch])
        self.assertTrue(all(s["kind_means"] and s["style_means"] and s["language_means"] for s in sent))
        self.assertNotIn("writer", sent[0])  # the writer is not told which writer it is, nor the batch


class Writing(Base):
    def test_the_truth_is_the_slots_never_the_reply_and_a_run_is_resumable(self):
        with mock.patch.object(I, "batches", lambda w, per_track=100: dict(list(ORIG_BATCHES(w).items())[:2])):
            out = I.write("a", 1.0, home=self.home, client_for=self.client_for(writer_reply))
            self.assertEqual(out["batches"], 2)
            sent = len(self.clients)
            I.write("a", 1.0, home=self.home, client_for=self.client_for(writer_reply))
            self.assertEqual(len(self.clients), sent)  # an answered batch is not asked again
        rows = I.written(self.home)
        self.assertEqual(len(rows), 20)
        truth = {r["slot"]: r for r in I.slots()}
        for r in rows:
            slot = truth[r["id"].removeprefix("IR1-")]
            self.assertEqual((r["track"], r["lang"], r["style"], r["writer"]), (slot["kind"], slot["language"], slot["style"], "a"))
            self.assertEqual(r["label_source"], "written")
        self.assertNotIn("stray", [r["text"] for r in rows])
        kept = json.loads(next((self.home / "writer_a").glob("*.json")).read_text(encoding="utf-8"))
        self.assertIn("request", kept)
        self.assertIn("content", kept["reply"])  # the whole request and reply

    def test_a_run_over_one_euro_is_refused_and_the_total_is_capped(self):
        with self.assertRaises(SystemExit):
            I.write("a", 1.5, home=self.home, client_for=self.client_for(writer_reply))
        (self.home / "checks").mkdir()
        (self.home / "checks" / "x.json").write_text(json.dumps({"spent_eur": 3.95}))
        self.assertAlmostEqual(I._run_cap(1.0, self.home), 0.05)

    def test_a_budget_refusal_stops_the_run_cleanly(self):
        def refusing(purpose, limit):
            c = FakeClient(writer_reply)
            c.complete = mock.Mock(side_effect=RuntimeError("BudgetExceeded: the next request could cost more"))
            return c
        out = I.write("b", 1.0, home=self.home, client_for=refusing)
        self.assertIn("stopped", out)
        self.assertEqual(list((self.home / "writer_b").glob("*.json")), [])


class Checking(Base):
    def test_a_disagreeing_reading_is_kept_apart_and_never_in_the_training_file(self):
        with mock.patch.object(I, "batches", lambda w, per_track=100: dict(list(ORIG_BATCHES(w).items())[:1])):
            I.write("a", 1.0, home=self.home, client_for=self.client_for(writer_reply))
        rows = I.written(self.home)
        I.slots_by_text = {r["text"]: r["track"] for r in rows}
        self.addCleanup(lambda: delattr(I, "slots_by_text"))
        out = I.check(1.0, home=self.home, client_for=self.client_for(plain_reply), workers=2)
        self.assertEqual(out["checked"], 10)
        self.assertTrue(all(c.purpose == "home_agent" for c in self.clients[-10:]))
        again = I.check(1.0, home=self.home, client_for=self.client_for(plain_reply))
        self.assertEqual(again["checked"], 0)  # a checked request is not read again
        result = I.assemble(self.home)
        kept, flagged = json.loads(I.OUT.read_text()), json.loads(I.FLAGGED.read_text())
        self.assertEqual(result["kept"] + result["flagged"], 10)
        expected_flagged = {r["id"] for r in rows if r["text"].endswith("0") and r["track"] != "advice"
                            and not (r["style"] == "two_kinds" and "advice" in r["tracks"])}
        self.assertEqual({r["id"] for r in flagged}, expected_flagged)
        self.assertFalse({r["id"] for r in kept} & {r["id"] for r in flagged})
        self.assertTrue(all(r["check"] for r in flagged))
        manifest = json.loads(I.MANIFEST.read_text())
        self.assertEqual((manifest["kept"], manifest["flagged"], manifest["label_source"]), (len(kept), len(flagged), "written"))
        self.assertAlmostEqual(manifest["spent_eur"], 0.11, places=6)  # one writer batch and ten readings, 0.01 each

    def test_a_two_kind_request_agrees_when_its_kind_is_among_the_readings(self):
        row = {"track": "prediction", "style": "two_kinds"}
        self.assertTrue(I.agrees(row, {"tracks": ["preparation", "prediction"]})[0])
        self.assertFalse(I.agrees({"track": "prediction", "style": "plain"}, {"tracks": ["preparation", "prediction"]})[0])
        self.assertFalse(I.agrees(row, {"tracks": []})[0])


class Blind(Base):
    def test_the_blind_sample_shows_words_only_and_never_overwrites_answers(self):
        rows = [{"id": f"IR1-{t[:4]}-{i:03d}", "text": f"{t} {i}", "track": t} for t in I.TRACKS for i in range(12)]
        I.OUT.write_text(json.dumps(rows[:60]))
        I.FLAGGED.write_text(json.dumps([{**r, "check": "x"} for r in rows[60:]]))
        I.blind_sample()
        sample = json.loads(I.BLIND.read_text())
        self.assertEqual(len(sample), 60)
        self.assertEqual(set(sample[0]), {"n", "id", "text"})
        answers = json.loads(I.BLIND_ANSWERS.read_text())
        answers[0]["track"] = rows[[r["id"] for r in rows].index(answers[0]["id"])]["track"]
        I.BLIND_ANSWERS.write_text(json.dumps(answers))
        I.blind_sample()
        self.assertEqual(json.loads(I.BLIND_ANSWERS.read_text())[0]["track"], answers[0]["track"])
        self.assertEqual(I.blind_score()["labelled"], 1)


class ThinkingKept(unittest.TestCase):
    def test_the_writers_call_keeps_the_thinking_summary(self):
        from dclab_rnd import experience
        from dclab_rnd.models import gateway as gw
        from dclab_rnd.models.usage import FileUsage

        clean = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY",
                                 "DCLAB_NO_LIVE_MODELS", "DCLAB_THINKING")} | {"OPENAI_API_KEY": "k", "DCLAB_LOG_PROMPTS": "1"}
        home = Path(tempfile.mkdtemp())

        class Scripted:
            thinking = None

            def complete(self, messages, tools=None, max_tokens=1800, **options):
                return {"content": writer_reply(messages), "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1},
                        "thinking": [{"text": "a Persian vague request about forecasting"}],
                        "assistant_message": {"role": "assistant", "content": "x"}}

        with mock.patch.dict(os.environ, clean), mock.patch.object(experience, "current_log", lambda: None):
            g = gw.Gateway(FileUsage(home / "model_requests.jsonl"), transport=lambda tier, purpose: Scripted())
            with mock.patch.object(I, "batches", lambda w, per_track=100: dict(list(ORIG_BATCHES(w).items())[:1])):
                I.write("a", 1.0, home=home / "run", client_for=lambda purpose, limit: g.client(purpose))
        docs = [json.loads(line) for p in home.rglob("model_exchanges/**/*.jsonl") for line in p.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["purpose"], "journey_writer")
        self.assertEqual(docs[0]["thinking_asked"], "adaptive")
        self.assertEqual(docs[0]["thinking"], [{"text": "a Persian vague request about forecasting"}])



if __name__ == "__main__":
    unittest.main()
