"""P8 (2026-10-29): the routing collector and the m7_routing sets, the 7th model's data.

One example per routing decision Home made: the person's words only with train consent, every answerer's reading (the
model's with its API thinking summary, never the block's signature; our own model's in shadow), the label by class (real,
written, silver) kept apart and weighted; the written requests (never the checker's flagged ones) and the eval runs (a
sealed run as its shape only, in the sealed split); one request's words in one split; a Home exam's request never in train;
a deleted draft's words kept; the GLiNER2 export with each class apart; coverage's counts per point and class; the ledger
hook that records every reading.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)

AT = "2026-10-28T10:00:00+00:00"
TRACKS = ["prediction", "analysis", "preparation", "reporting", "business", "advice"]
EXAM_TEXT = "Which of our stores will run out of milk next week?"


def decision(ident, draft, point="request.intent", answers=None, used="rule", value="analysis", at=AT, state=None):
    return {"schema_version": "dclab_core.decision_record.v2", "id": ident, "point": point, "subject": {"kind": "draft", "key": draft},
            "state": state if state is not None else {"has_data": False, "words": 6, "signals": 1}, "options": TRACKS,
            "answers": answers if answers is not None else [{"by": "rule", "answerer": "rule:intent_rules_v2", "value": value}],
            "value_used": value, "used": used, "draft": draft, "conversation": draft, "at": at, "consent": "train"}


def label(ident, record, value, by="person", source="user_decision_exposed", supersedes=None):
    return {"schema_version": "dclab_core.decision_label.v5", "id": ident, "record": record, "point": "request.intent", "value": value,
            "source": source, "by": by, "at": AT, "supersedes": supersedes}


def draft_doc(ident, problem, said=()):
    return {"id": ident, "created": AT, "problem": problem, "status": "draft", "settings": {},
            "messages": [{"id": f"m{i}", "role": "user", "kind": "text", "text": t, "at": when} for i, (when, t) in enumerate(said)]}


def exchange(draft, request, at, thinking=True, purpose="home_intent", reply='{"tracks": ["analysis"], "quote": "my customers"}'):
    return {"at": at, "model_request": request, "purpose": purpose, "model": "claude-sonnet-5-5", "draft": draft, "outcome": "ok", "whole": True,
            "messages": [{"role": "system", "content": "Say what kind of help."}, {"role": "user", "content": "find groups in my customers"}],
            "reply": reply, "thinking": ([{"text": "Groups of customers is analysis, not a prediction.", "signature": "SIGNATURE-BLOB"}] if thinking else [])}


@unittest.skipIf(MISSING, MISSING)
class RoutingContractTest(unittest.TestCase):
    def test_a_row_has_its_words_or_says_why_a_label_its_class_and_silver_weighs_less(self):
        from dclab_core.routing_example import RoutingExample, RoutingReading, label_class
        from dclab_core.store import Provenance

        prov = Provenance(entry="e", sha256="0" * 64, source="s", workspace="dev", at=AT)
        base = dict(id="m7_routing:x", split="train", provenance=prov, point="request.intent", origin="real", split_group="request:1")
        RoutingExample(**base, text="find groups in my customers")
        with self.assertRaises(ValueError):
            RoutingExample(**base)  # no words and no reason
        with self.assertRaises(ValueError):
            RoutingExample(**base, text="x", label="analysis")  # a label without its class
        with self.assertRaises(ValueError):
            RoutingExample(**base, text="x", label="analysis", label_source="silver", weight=1.0)  # silver weighs at most 0.5
        with self.assertRaises(ValueError):
            RoutingExample(**base, text="x", state={"sample_value": "Dubai refunds 41.5"})  # a value in the state
        with self.assertRaises(ValueError):
            RoutingReading(by="model", answerer="home_intent", value="analysis")  # a model's reading without its thinking or why
        with self.assertRaises(ValueError):
            RoutingReading(by="rule", answerer="rule", thinking="x", thinking_source="recorded")  # only a model thinks
        self.assertEqual(label_class("user_decision_exposed", scripted=False), "real")
        self.assertEqual(label_class("user_decision_exposed", scripted=True), "written")  # a script's click is no human's
        self.assertEqual(label_class("model", scripted=False), "silver")
        self.assertIsNone(label_class("rule", scripted=False))

    def test_m7_routing_is_a_model_type_of_its_own_and_a_manifest_may_name_it(self):
        from dclab_core.store import EXAMPLE_SHAPES, MODEL_TYPES, ModelType, SetManifest
        import typing

        self.assertIn("m7_routing", MODEL_TYPES)
        self.assertEqual(EXAMPLE_SHAPES["m7_routing"], ("routing_example",))
        self.assertNotIn("m7_routing", typing.get_args(ModelType))  # the other example contracts keep their shape
        m = SetManifest(model_type="m7_routing", name="request.intent", version="v1", shape="s", split_rule="r", built_at=AT)
        self.assertEqual(m.schema_version, "dclab_core.set_manifest.v2")
        with self.assertRaises(ValueError):
            SetManifest(model_type="m9_nothing", name="x", version="v1", shape="s", split_rule="r", built_at=AT)


@unittest.skipIf(MISSING, MISSING)
class RoutingCollectorTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout, Store

        self.tmp = Path(tempfile.mkdtemp())
        self.layout = Layout(self.tmp / "store")
        self.home = self.tmp / "home"  # the written requests and the exams' files
        self.results = self.tmp / "results"
        self.home.mkdir()
        self.results.mkdir()
        store = Store(self.layout)
        real = [decision("dra000000001", "d1", answers=[{"by": "rule", "answerer": "rule:intent_rules_v2", "value": "prediction"},
                                                       {"by": "model_role", "answerer": "home_intent", "value": "analysis", "confidence": 0.8},
                                                       {"by": "classifier", "answerer": "classifier:intent_tfidf_v1", "value": "analysis"}],
                         used="model_role"),
                decision("dra000000002", "d2", answers=[{"by": "rule", "answerer": "rule:intent_rules_v2", "value": "prediction"}], value="prediction",
                         at="2026-10-28T10:05:00+00:00"),
                decision("dra000000003", "d3"),
                decision("dra000000004", "d4", point="workflow.next_question", value="target", state={"open_questions": 0, "step": 1}),
                decision("dra000000005", "d5"),  # an exam's request typed live
                decision("dra000000006", "d6", point="column.is_identifier")]  # not a routing point
        store.keep_log("decisions", "dev", "ledger/decisions/2026/10/28.jsonl", "".join(json.dumps(d) + "\n" for d in real).encode(), consent="train")
        store.keep_log("decisions", "dev", "ledger/labels/2026/10/28.jsonl", "".join(json.dumps(x) + "\n" for x in [
            label("dl1", "dra000000001", "analysis"),  # the person's confirm on the card: real
            label("dl3", "dra000000003", "reporting"), label("dl3x", "dra000000003", None, supersedes="dl3")]).encode(), consent="train")  # retracted
        drafts = {"d1": draft_doc("d1", "find groups in my customers", [("2026-10-28T10:00:30+00:00", "and send them offers later")]),
                  "d2": draft_doc("d2", "I have data analysis problems", [("2026-10-28T10:01:00+00:00", "my sales file")]),
                  "d3": draft_doc("d3", "weekly sales dashboard"), "d4": draft_doc("d4", "predict churn"), "d5": draft_doc("d5", EXAM_TEXT)}
        for ident, doc in drafts.items():
            store.keep("conversations", "dev", f"drafts/{ident}/draft.json", json.dumps(doc).encode(), consent="train")
        store.keep_log("model_requests", "live_tests__x", "model_exchanges/2026/10/28.jsonl",  # kept in another workspace: joined by draft
                       (json.dumps(exchange("d1", "mr0000000001", "2026-10-28T09:59:58+00:00")) + "\n" +
                        json.dumps(exchange("d2", "mr0000000002", "2026-10-28T10:04:58+00:00", reply='{"tracks": []}')) + "\n").encode(), consent="train")
        # a scripted workspace (its person's click is a script's) and one whose consent is improve (never trained on)
        store.keep_log("decisions", "e2e__scripted", "ledger/decisions/2026/10/28.jsonl", (json.dumps(decision("drs000000001", "s1")) + "\n").encode(), consent="train")
        store.keep_log("decisions", "e2e__scripted", "ledger/labels/2026/10/28.jsonl", (json.dumps(label("dls1", "drs000000001", "analysis")) + "\n").encode(), consent="train")
        store.keep("conversations", "e2e__scripted", "drafts/s1/draft.json", json.dumps(draft_doc("s1", "count orders per city")).encode(), consent="train")
        store.keep_log("decisions", "improve_only", "ledger/decisions/2026/10/28.jsonl", (json.dumps(decision("dri000000001", "i1")) + "\n").encode(), consent="improve")
        # the written requests (one flagged by the checker), the dev set, a dev run and a sealed run, the exam's file
        written = [{"id": "IR1-anal-001", "text": "split my shoppers into groups", "track": "analysis", "tracks": ["analysis"], "lang": "en", "writer": "a",
                    "plain_reading": ["analysis"]},
                   {"id": "IR1-busi-002", "text": "چرا فروش ما افت کرد", "track": "business", "tracks": ["business"], "lang": "fa", "writer": "b",
                    "plain_reading": ["business"]}]
        flagged = [{"id": "IR1-advi-003", "text": "tell me about churn", "track": "advice", "lang": "en", "writer": "a", "check": "read as prediction"}]
        dev = [{"id": "d-a1", "lang": "en", "data": ["sales"], "text": "weekly sales dashboard", "gold": {"tracks": ["reporting"], "exit": "dashboard"}}]
        sealed = [{"id": "s-1", "lang": "en", "data": [], "text": EXAM_TEXT, "gold": {"tracks": ["prediction"]}}]
        (self.home / "intent_requests_v1.json").write_text(json.dumps(written, ensure_ascii=False))
        (self.home / "intent_requests_v1_flagged.json").write_text(json.dumps(flagged))
        (self.home / "requests_dev.json").write_text(json.dumps(dev))
        (self.home / "requests_sealed.json").write_text(json.dumps(sealed))
        run = {"id": "HOME-001", "split": "dev", "at": AT, "arms": ["rules", "model", "plain"],
               "rows": [{"case": "d-a1", "lang": "en", "observed": {"rules": {"track": "analysis", "fields": {"target": "sales"}},
                                                                     "model": {"track": "reporting"}, "plain": {"track": "reporting"}}}]}
        exam = {"id": "HOME-002", "split": "sealed", "at": AT, "arms": ["rules", "model", "plain"],
                "rows": [{"case": "s-1", "lang": "en", "observed": {"rules": {"track": "analysis", "fields": {"target": "milk out of stock"}},
                                                                     "model": {"track": "prediction"}}}]}
        (self.results / "HOME-001_dev.json").write_text(json.dumps(run))
        (self.results / "HOME-002_sealed.json").write_text(json.dumps(exam))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self):
        from dclab_rnd.store import Store, shapers
        from dclab_rnd.store.keepers import routing as routing_keeper
        from dclab_rnd.store.sets import routing

        store = Store(self.layout)  # the routing keeper's sources kept as the conductor keeps them
        for src in routing_keeper.sources({"home_requests": str(self.home), "home_results": str(self.results)}):
            store.keep("tests", src.workspace, src.source, src.data, consent=src.consent, transform=src.transform)
        shapers.shape(self.layout)
        return routing.build(self.layout, scripted=frozenset({"e2e__scripted"}), home=self.home, results=self.results)

    def rows(self, point="request.intent"):
        out = {}
        for split in ("train", "dev", "sealed"):
            path = self.layout.sets("m7_routing", point, "v1") / f"{split}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
                r = json.loads(line)
                out[r["id"]] = {**r, "_split": split}
        return out

    def test_each_routing_decision_with_its_words_readings_thinking_and_label_class(self):
        report = self.build()
        self.assertIn("m7_routing/request.intent", report["sets"])
        self.assertIn("m7_routing/workflow.next_question", report["sets"])
        self.assertNotIn("m7_routing/column.is_identifier", report["sets"])  # not a routing point
        self.assertEqual(report["out_of_sets"].get("no train consent (the decision)"), 1)  # the improve-only workspace
        rows = self.rows()
        r1 = rows["m7_routing:dev:dra000000001"]
        self.assertEqual(r1["text"], "find groups in my customers")  # what was said before the decision, nothing after
        self.assertEqual((r1["label"], r1["label_source"], r1["label_by"], r1["weight"]), ("analysis", "real", "person:card", 1.0))
        model = next(x for x in r1["readings"] if x["by"] == "model")
        self.assertEqual(model["thinking"], "Groups of customers is analysis, not a prediction.")  # the summary's text, joined across workspaces
        self.assertEqual(model["thinking_source"], "recorded")
        self.assertNotIn("SIGNATURE-BLOB", json.dumps(rows))  # the block's signature is never kept
        self.assertTrue(model["used"] and model["confidence"] == 0.8)
        own = next(x for x in r1["readings"] if x["by"] == "own_model")
        self.assertTrue(own["shadow"] and not own["used"])
        r2 = rows["m7_routing:dev:dra000000002"]  # the rules routed; the model read it beside them and said the words do not tell
        self.assertEqual(r2["text"], "I have data analysis problems\nmy sales file")  # read again after their answer: both said by then
        beside = next(x for x in r2["readings"] if x["by"] == "model")
        self.assertIsNone(beside["value"])
        self.assertFalse(beside["used"])
        self.assertIsNone(r2["label"])  # no person label, and a model that abstained gives no silver
        r3 = rows["m7_routing:dev:dra000000003"]  # the person's label was retracted; the words are the dev set's: its gold is written
        self.assertEqual((r3["label"], r3["label_source"], r3["label_by"], r3["origin"]), ("reporting", "written", "eval:requests_dev", "eval"))
        scripted = rows["m7_routing:e2e__scripted:drs000000001"]
        self.assertEqual((scripted["label_source"], scripted["label_by"], scripted["origin"]), ("written", "scripted:card", "scripted"))

    def test_written_requests_and_eval_runs_flagged_never_sealed_as_shape_only(self):
        self.build()
        rows = self.rows()
        written = rows["m7_routing:written:IR1-anal-001"]
        self.assertEqual((written["label"], written["label_source"], written["weight"], written["origin"]), ("analysis", "written", 0.5, "written"))
        self.assertEqual(rows["m7_routing:written:IR1-busi-002"]["language"], "fa")
        self.assertFalse(any("IR1-advi-003" in k for k in rows))  # the checker's flagged request never enters
        dev_run = rows["m7_routing:eval:HOME-001:d-a1"]
        self.assertEqual(dev_run["text"], "weekly sales dashboard")
        self.assertEqual({x["answerer"]: x["value"] for x in dev_run["readings"]},
                         {"rule:home_eval": "analysis", "home_agent": "reporting", "home_eval_plain": "reporting"})
        exam = rows["m7_routing:eval:HOME-002:s-1"]
        self.assertEqual(exam["_split"], "sealed")
        self.assertIsNone(exam["text"])
        self.assertNotIn("milk out of stock", json.dumps(rows))  # a sealed run's quoted fields are never kept
        self.assertNotIn(EXAM_TEXT.lower(), json.dumps(rows["m7_routing:eval:HOME-002:s-1"]).lower())
        live = rows["m7_routing:dev:dra000000005"]  # the exam's request typed live: sealed, as its exam row is (one split per request)
        self.assertEqual(live["_split"], "sealed")
        self.assertEqual(live["exam"], "home:sealed")
        groups = {}
        for r in rows.values():
            groups.setdefault(r["split_group"], set()).add(r["_split"])
        self.assertTrue(all(len(s) == 1 for s in groups.values()), groups)  # one request's words in one split
        dev3 = rows["m7_routing:dev:dra000000003"]
        self.assertEqual(dev3["split_group"], dev_run["split_group"])

    def test_an_unrun_exam_request_is_sealed_and_the_words_come_from_the_store_alone(self):
        os.remove(self.results / "HOME-002_sealed.json")  # the exam not run yet
        self.build()
        self.assertEqual(self.rows()["m7_routing:dev:dra000000005"]["_split"], "sealed")
        # the build reads only what the store kept (no workspace file): a draft deleted in the product keeps its row's words
        from dclab_rnd.store.sets import routing

        routing.build(self.layout, scripted=frozenset({"e2e__scripted"}), home=self.home, results=self.results)
        self.assertEqual(self.rows()["m7_routing:dev:dra000000001"]["text"], "find groups in my customers")

    def test_no_thinking_without_the_words_no_exchange_without_train_consent_and_an_exam_seen_only_by_the_model(self):
        from dclab_rnd.store import Store

        store = Store(self.layout)
        model = [{"by": "rule", "answerer": "rule:intent_rules_v2", "value": None}, {"by": "model_role", "answerer": "home_intent", "value": "prediction"}]
        store.keep_log("decisions", "dev", "ledger/decisions/2026/10/27.jsonl", "".join(json.dumps(d) + "\n" for d in [
            decision("dra000000070", "d70", answers=model, used="model_role", value="prediction"),  # the draft turned training off
            decision("dra000000079", "d79", answers=model, used="model_role", value="prediction")]).encode(), consent="train")  # no draft kept
        off = draft_doc("d70", "find groups in my customers")
        off["settings"] = {"consent": {"train": False}}
        store.keep("conversations", "dev", "drafts/d70/draft.json", json.dumps(off).encode(), consent="train")
        exam_exchange = exchange("d79", "mr0000000079", AT)
        exam_exchange["messages"][1]["content"] = EXAM_TEXT
        store.keep_log("model_requests", "dev", "model_exchanges/2026/10/27.jsonl", "".join(json.dumps(x) + "\n" for x in [
            exchange("d70", "mr0000000070", AT), exam_exchange]).encode(), consent="train")
        plain = exchange(None, "mr0000000099", AT, purpose="home_agent")  # an improve-only workspace's reading of a written request
        plain["messages"][1]["content"] = "split my shoppers into groups"
        store.keep_log("model_requests", "improve_only", "model_exchanges/2026/10/27.jsonl", (json.dumps(plain) + "\n").encode(), consent="improve")
        os.remove(self.results / "HOME-002_sealed.json")  # the exam not run yet
        self.build()
        rows = self.rows()
        off_row = rows["m7_routing:dev:dra000000070"]
        self.assertIsNone(off_row["text"])
        self.assertTrue(all(not x.get("thinking") for x in off_row["readings"]))  # the thinking reads the words: not kept either
        exam_row = rows["m7_routing:dev:dra000000079"]
        self.assertEqual(exam_row["_split"], "sealed")  # the model read an exam's words: sealed though the draft is gone
        self.assertNotIn("Groups of customers", json.dumps(exam_row))
        written = rows["m7_routing:written:IR1-anal-001"]
        self.assertTrue(all(not x.get("thinking") for x in written["readings"]))  # never an improve-only workspace's exchange

    def test_a_real_request_without_a_confirmed_label_gets_claudes_reading_as_silver_apart(self):
        from dclab_rnd.store import Store, shapers

        store = Store(self.layout)
        store.keep_log("decisions", "dev", "ledger/decisions/2026/10/29.jsonl", (json.dumps(decision(
            "dra000000009", "d9", answers=[{"by": "rule", "answerer": "rule:intent_rules_v2", "value": None},
                                           {"by": "model_role", "answerer": "home_intent", "value": "preparation"}], used="model_role",
            value="preparation", at="2026-10-29T10:00:00+00:00")) + "\n").encode(), consent="train")
        store.keep("conversations", "dev", "drafts/d9/draft.json", json.dumps(draft_doc("d9", "merge these two files please")).encode(), consent="train")
        self.build()
        r = self.rows()["m7_routing:dev:dra000000009"]
        self.assertEqual((r["label"], r["label_source"], r["label_by"], r["weight"]), ("preparation", "silver", "model:home_intent", 0.5))
        self.assertTrue(next(x for x in r["readings"] if x["by"] == "model")["no_thinking"])  # no exchange: says why

    def test_claudes_blind_routing_label_fills_an_unlabelled_decision_as_silver_and_never_beats_a_persons(self):
        """2026-10-29 (the owner: "you can use Claude labels"): dclab_rnd.store.routing_labels writes the store's own file."""
        labels = Path(self.layout.root) / "labels"
        labels.mkdir(parents=True, exist_ok=True)
        (labels / "m7_routing_claude.jsonl").write_text(
            json.dumps({"example": "m7_routing:dev:dra000000002", "value": "analysis", "model": "claude-sonnet-5-5"}) + "\n"
            + json.dumps({"example": "m7_routing:dev:dra000000001", "value": "prediction", "model": "claude-sonnet-5-5"}) + "\n"
            + json.dumps({"example": "m7_routing:dev:dra000000003", "value": "not an option", "model": "claude-sonnet-5-5"}) + "\n")
        self.build()
        rows = self.rows()
        r2 = rows["m7_routing:dev:dra000000002"]
        self.assertEqual((r2["label"], r2["label_source"], r2["label_by"], r2["weight"]),
                         ("analysis", "silver", "model:claude-sonnet-5-5:routing_labeler", 0.5))
        self.assertEqual((rows["m7_routing:dev:dra000000001"]["label"], rows["m7_routing:dev:dra000000001"]["label_source"]), ("analysis", "real"))
        self.assertEqual(rows["m7_routing:dev:dra000000003"]["label_source"], "written")  # its written gold stands; an off-list answer is nothing

    def test_claudes_label_carries_its_api_thinking_summary_as_the_answers_reasoning(self):
        """2026-10-30, the owner: "use Claude's thinking to add the reason ... exactly summarized thinking": the labeller's kept
        exchange (by its request id) gives the label its reason, never a field the reply wrote; a shadow reading, never used."""
        from dclab_rnd.store import Store
        from dclab_rnd.store.sets import routing as R

        Store(self.layout).keep_log("model_requests", "dev", "model_exchanges/2026/10/29.jsonl",
                                    (json.dumps(exchange("routing-labels", "mrlabel00001", "2026-10-29T17:10:49+00:00", purpose="routing_labeler",
                                                         reply='{"answer": "analysis", "confidence": "high"}')) + "\n").encode(), consent="train")
        labels = Path(self.layout.root) / "labels"
        labels.mkdir(parents=True, exist_ok=True)
        (labels / "m7_routing_claude.jsonl").write_text(json.dumps({"example": "m7_routing:dev:dra000000002", "value": "analysis",
                                                                     "model": "claude-sonnet-5-5", "request": "mrlabel00001",
                                                                     "reason": "a field the reply wrote"}) + "\n")
        self.build()
        r2 = self.rows()["m7_routing:dev:dra000000002"]
        labeller = [x for x in r2["readings"] if x["answerer"] == "routing_labeler"]
        self.assertEqual(len(labeller), 1)
        self.assertTrue(labeller[0]["shadow"] and not labeller[0]["used"])
        self.assertEqual(labeller[0]["thinking"], "Groups of customers is analysis, not a prediction.")
        home_reading = {"by": "model", "answerer": "home_intent", "value": "analysis", "thinking": "Home's own reading.", "thinking_source": "recorded"}
        first = {**r2, "readings": [home_reading, *r2["readings"]]}  # Home's reading with the same answer comes first
        self.assertEqual(R.chat_row(first)["messages"][-1]["reasoning"], "Groups of customers is analysis, not a prediction.")  # the giver's
        answer = R.chat_row(r2)["messages"][-1]
        self.assertEqual((answer["reasoning"], answer["reasoning_source"]), ("Groups of customers is analysis, not a prediction.", "recorded"))
        self.assertNotIn("a field the reply wrote", json.dumps(r2))
        self.assertEqual(R.counts(self.layout)["request.intent"]["label_thinking"], 1)

    def test_claudes_next_question_labels_are_readings_never_labels(self):
        """2026-10-30, the next-question study: Claude saw a view that did not say which questions were settled, and agreed
        with Home 47 times in 144; its answers stay as readings for evaluation, never as silver labels."""
        labels = Path(self.layout.root) / "labels"
        labels.mkdir(parents=True, exist_ok=True)
        (labels / "m7_routing_claude.jsonl").write_text(json.dumps({"example": "m7_routing:dev:dra000000004", "value": "prediction",  # one of the fixture's options
                                                                     "model": "claude-sonnet-5-5"}) + "\n")
        self.build()
        r4 = self.rows("workflow.next_question")["m7_routing:dev:dra000000004"]
        self.assertNotEqual(r4.get("label_by"), "model:claude-sonnet-5-5:routing_labeler")
        self.assertIn("routing_labeler", [x["answerer"] for x in r4["readings"]])  # kept, as a shadow reading

    def test_the_labeller_is_told_what_each_fact_means(self):
        from dclab_rnd.store import routing_labels as rl

        shown = rl.view({"point": "workflow.next_question", "text": "predict churn", "options": ["action", "labels"],
                         "state": {"labels": "unknown", "action_known": False}})
        self.assertIn("says nothing about whether past outcomes are recorded", shown["facts_mean"]["labels"])
        self.assertIn("action_known", shown["facts_mean"])

    def test_the_gliner2_export_keeps_each_label_class_apart_and_coverage_counts_them(self):
        from dclab_rnd.lake import answers
        from dclab_rnd.store import coverage
        from dclab_rnd.store.sets import routing

        self.build()
        tests = list(answers._routing(Path("."), root=self.layout.sets("m7_routing")))
        self.assertTrue(tests)
        self.assertEqual({t["apart"] for t in tests}, {t["label_origin"] for t in tests})
        self.assertTrue({"real", "written"} <= {t["apart"] for t in tests})
        intent = next(t for t in tests if t["sample_id"] == "m7_routing:dev:dra000000001")
        row = answers.gliner2_row(intent)
        self.assertEqual(row["input"], "find groups in my customers")
        self.assertEqual(row["output"]["classifications"][0]["true_label"], ["analysis"])
        self.assertIn("label_descriptions", row["output"]["classifications"][0])
        counts = routing.counts(self.layout)["request.intent"]  # train and dev read; the sealed split counted from its manifest only
        manifest = json.loads((self.layout.sets("m7_routing", "request.intent", "v1") / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(counts["rows"] + counts["sealed_not_read"], len(self.rows()))
        self.assertGreaterEqual(manifest["label_sources"].get("real", 0), 1)
        self.assertTrue({"real", "written", "silver"} & set(manifest["weights"]))
        text = "\n".join(coverage.render_routing(routing.counts(self.layout)))
        self.assertIn("request.intent", text)
        self.assertIn("silver", text)
        self.assertFalse(any(t.get("sample_id") == "m7_routing:dev:dra000000005" for t in tests))  # an exam's request never leaves

    def test_the_chat_and_table_exports_keep_each_class_apart_and_never_an_exam(self):
        import csv

        from dclab_rnd.store.sets import routing

        self.build()
        out = self.tmp / "datasets"
        manifest = routing.export(self.layout, out=out)
        root = out / "v1"
        classes = {p.parent.name for p in (root / "chat").rglob("*.jsonl")}
        self.assertTrue({"real", "written"} <= classes)
        chats = [json.loads(line) for p in (root / "chat").rglob("*.jsonl") for line in p.read_text(encoding="utf-8").splitlines()]
        r1 = next(c for c in chats if c["id"] == "m7_routing:dev:dra000000001")
        self.assertEqual(r1["messages"][1]["content"], "find groups in my customers")
        self.assertEqual((r1["messages"][2]["content"], r1["messages"][2]["reasoning_source"]), ("analysis", "recorded"))
        with (root / "table.csv").open(encoding="utf-8") as fh:
            table = list(csv.DictReader(fh))
        self.assertTrue(any(r["id"] == "m7_routing:written:IR1-busi-002" and r["language"] == "fa" for r in table))
        self.assertNotIn(EXAM_TEXT, (root / "table.csv").read_text(encoding="utf-8"))
        self.assertGreaterEqual(manifest["skipped"].get("an exam's request", 0), 1)

    def test_the_keepers_sources_are_not_placed_as_code_judged_by_a_test(self):
        from dclab_core.store import Provenance, StoreRecord
        from dclab_rnd.store import curators

        prov = Provenance(entry="e", sha256="0" * 64, source="home_eval/HOME-001_dev.json", workspace="rnd", at=AT, consent="train")
        record = StoreRecord(id="tests:rnd:x#0", kind="tests", type="test_run", at=AT, body={}, provenance=prov)
        self.assertIsNone(curators.place("m7_judges", record))


@unittest.skipIf(MISSING, MISSING)
class RoutingKeeperAndHookTest(unittest.TestCase):
    def test_a_sealed_run_is_kept_as_its_shape_and_a_sealed_request_file_never(self):
        from dclab_rnd.store.keepers import routing

        tmp = Path(tempfile.mkdtemp())
        (tmp / "requests_dev.json").write_text("[]")
        (tmp / "requests_sealed_v2.json").write_text("[]")
        (tmp / "intent_requests_v1.json").write_text("[]")
        (tmp / "HOME-009_sealed_v2.json").write_text(json.dumps({"id": "HOME-009", "split": "sealed_v2", "rows": [
            {"case": "s2-p1", "lang": "en", "observed": {"model": {"track": "prediction", "fields": {"target": "bookings cancelled"}}}}]}))
        found = {s.source: s for s in routing.sources({"home_requests": str(tmp), "home_results": str(tmp)})}
        self.assertIn("home_requests/requests_dev.json", found)
        self.assertNotIn("home_requests/requests_sealed_v2.json", found)  # an exam's words are never kept by this keeper
        shaped = json.loads(found["home_eval/HOME-009_sealed_v2.json"].transform((tmp / "HOME-009_sealed_v2.json").read_bytes()))
        self.assertEqual(shaped["rows"][0]["observed"]["model"]["track"], "prediction")
        self.assertNotIn("bookings cancelled", json.dumps(shaped))
        shutil.rmtree(tmp, ignore_errors=True)

    def test_the_routing_keeper_is_registered_and_claims_no_other_keepers_path(self):
        from dclab_rnd.store.keepers import standard

        reg = standard.install({})
        self.assertIn("routing", reg)
        self.assertEqual(reg["routing"].feeds, ("m7_routing",))

    def test_routing_read_records_every_reading_with_the_models_request_and_our_own_in_shadow(self):
        from dclab_rnd import ledger
        from dclab_rnd.ledger import capture

        home = Path(tempfile.mkdtemp())
        led = ledger.open_ledger(home, "wtest")
        ledger.install(led)
        try:
            rid = capture.routing_read("d1", "request.intent", "analysis", "model_role", options=TRACKS,
                                       state={"has_data": False, "words": 5, "signals": 0},
                                       rule={"answerer": "rule:intent_rules_v2", "value": "prediction"},
                                       model={"answerer": "home_intent", "value": "analysis", "confidence": 0.9, "request": "mr00000000aa"},
                                       own={"answerer": "classifier:intent_tfidf_v1", "value": "analysis", "confidence": 0.7})
            self.assertTrue(rid)
            record = next(r for r in led.records(draft="d1") if r.id == rid)
            by = {a.by: a for a in record.answers}
            self.assertEqual(by["model_role"].source, "mr00000000aa")
            self.assertEqual(by["classifier"].value, "analysis")
            self.assertEqual(record.used, "model_role")
            self.assertIsNone(capture.routing_read("d1", "request.intent", "analysis", "classifier"))  # our own model never routes yet
        finally:
            ledger.install(None)
            shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
