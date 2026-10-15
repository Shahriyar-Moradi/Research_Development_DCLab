"""23.10: the lake's shape. The episodes and model-call sets are built from exported events and decisions, every number traces
to event ids, free text appears only at train, the manifest names the scrubber and the commit, the sets are also written as
columns, and a deletion is followed by a rebuild that no longer holds the rows."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core import columnar, datasets as builders, redact
except ImportError as error:  # the agent environment may lack pydantic
    MISSING = str(error)
else:
    MISSING = ""


def ev(ident, kind, payload, at, conversation="d1", project="p1", consent="train", request=None, model_request=None):
    return {"id": ident, "kind": kind, "at": at, "seq": 0, "payload": payload, "consent": consent,
            "ids": {"conversation": conversation, "project": project, "model_request": model_request, "request": request}}


def flow():
    return [
        ev("e01", "draft.created", {"problem": "Which customers will leave?"}, "2026-10-16T10:00:00+00:00", project=None),
        ev("e02", "chat.agent", {"text": "Which column says who left?", "kind": "question", "message": "m1", "version": "core-0.46.0/prompts-aa"}, "2026-10-16T10:00:01+00:00", project=None, model_request="mr1"),
        ev("e03", "model.request", {"purpose": "home_agent", "tier": "t", "model": "m", "input_tokens": 100, "output_tokens": 20, "cost": 0.01, "outcome": "ok",
                                    "prompt": "user: Which customers will leave?", "reply": "Which column says who left?"}, "2026-10-16T10:00:01+00:00", project=None, model_request="mr1"),
        ev("e04", "model.checked", {"purpose": "home_agent", "passed": "yes"}, "2026-10-16T10:00:02+00:00", project=None, model_request="mr1"),
        ev("e05", "feedback.given", {"on": "answer:m1", "kind": "thumbs", "rating": "helpful"}, "2026-10-16T10:00:03+00:00", project=None),
        ev("e06", "chat.person", {"text": "The churned column", "surface": "home"}, "2026-10-16T10:01:00+00:00", project=None),
        ev("e07", "draft.built", {"project": "p1"}, "2026-10-16T10:01:30+00:00", project="p1"),
        ev("e08", "pipeline.step_finished", {"run": "r1", "step": "fold 1: metrics", "kind": "metric", "metric": "roc_auc", "value": 0.8, "seconds": 0.5, "fold": 1},
           "2026-10-16T10:02:00+00:00", conversation=None),
        ev("e09", "graph.move", {"move": "start_final", "allowed": False}, "2026-10-16T10:02:01+00:00", conversation=None),
        ev("e10", "outcome.observed", {"metric": "roc_auc", "value": 0.8, "verdict": "well", "observed": "holdout"}, "2026-10-16T10:03:00+00:00", conversation=None),
    ]


def records():
    return [{"id": "dr000001", "point": "split.strategy", "used": "rule", "project": "p1", "conversation": None, "at": "2026-10-16T10:02:00+00:00"},
            {"id": "dr000002", "point": "problem.family", "used": "rule", "project": None, "conversation": "d1", "at": "2026-10-16T10:00:30+00:00"}]


@unittest.skipIf(MISSING, MISSING)
class EpisodesTest(unittest.TestCase):
    def test_a_turn_joins_what_happened_in_its_window_by_the_ids(self):
        rows, manifest = builders.episodes(flow(), records(), text=True, workspace="w", drafts={"d1": "p1"})
        self.assertEqual([r["turn"] for r in rows], [0, 1])
        first, second = rows
        self.assertEqual((first["message"], first["agent"]["replies"], first["agent"]["kinds"], first["model_calls"]["n"], first["model_calls"]["tokens_in"]),
                         ("Which customers will leave?", 1, ["question"], 1, 100))
        self.assertEqual([d["id"] for d in first["decisions"]], ["dr000002"])  # the ledger's record of the conversation, inside the window
        self.assertEqual(first["feedback"], [{"kind": "thumbs", "rating": "helpful", "on": "answer:m1"}])
        self.assertEqual((second["message"], second["steps"]["finished"], second["steps"]["metrics"][0]["value"], second["moves"], second["outcomes"][0]["verdict"]),
                         ("The churned column", 1, 0.8, [{"move": "start_final", "allowed": False}], "well"))  # project events of the window join by the project
        self.assertEqual([d["id"] for d in second["decisions"]], ["dr000001"])
        self.assertEqual(manifest.rows, 2)
        self.assertEqual(manifest.builder, "episodes")

    def test_every_number_traces_to_event_ids_and_the_sources_name_them(self):
        rows, manifest = builders.episodes(flow(), records(), text=True, drafts={"d1": "p1"})
        by_id = {e["id"]: e for e in flow()}
        for row in rows:
            cited = [by_id[i] for i in row["event_ids"]]
            self.assertEqual(row["model_calls"]["n"], sum(1 for e in cited if e["kind"] == "model.request"))
            self.assertEqual(row["model_calls"]["tokens_in"], sum(e["payload"].get("input_tokens", 0) for e in cited if e["kind"] == "model.request"))
            self.assertEqual(row["steps"]["finished"], sum(1 for e in cited if e["kind"] == "pipeline.step_finished"))
            self.assertEqual(row["agent"]["chars"], sum(len(e["payload"]["text"]) for e in cited if e["kind"] == "chat.agent"))
        sources = {(s.kind, s.id) for s in manifest.sources}
        self.assertTrue({("event", "e03"), ("event", "e08"), ("decision", "dr000001"), ("conversation", "d1")} <= sources)

    def test_under_improve_a_row_has_no_free_text(self):
        rows, manifest = builders.episodes(flow(), records(), text=False, drafts={"d1": "p1"})
        blob = json.dumps(rows)
        self.assertEqual([r["message"] for r in rows], [None, None])
        for words in ("Which customers", "churned column", "Which column says"):
            self.assertNotIn(words, blob)
        self.assertEqual(rows[0]["message_chars"], len("Which customers will leave?"))  # a count survives
        self.assertEqual(manifest.use, "improve")
        self.assertIn("no free text", " ".join(manifest.limits))

    def test_a_conversation_with_no_message_makes_no_row_and_whole_groups_split_together(self):
        rows, _ = builders.episodes([ev("x1", "stage.finished", {"stage": "data"}, "2026-10-16T10:00:00+00:00")], [], text=True)
        self.assertEqual(rows, [])
        rows, _ = builders.episodes(flow(), records(), text=True, drafts={"d1": "p1"})
        self.assertEqual(len({r["split"] for r in rows}), 1)  # the turns of one project are one group

    def test_a_project_event_belongs_to_one_turn_and_later_work_to_none(self):
        events = flow() + [ev(f"s{i:03d}", "pipeline.step_finished", {"run": "r2", "step": "fold", "kind": "metric", "metric": "auc", "value": 0.7, "seconds": 0.1},
                              f"2026-11-20T10:{i // 60:02d}:{i % 60:02d}+00:00", conversation=None) for i in range(700)]  # a month later
        events += [ev("o1", "chat.person", {"text": "A second session on the same project"}, "2026-10-16T10:02:30+00:00", conversation="i1")]
        events += [ev("o2", "pipeline.step_finished", {"run": "r3", "step": "fold", "kind": "metric", "metric": "auc", "value": 0.9, "seconds": 0.1},
                     "2026-10-16T10:02:40+00:00", conversation=None)]
        rows, manifest = builders.episodes(events, records(), text=True, drafts={"d1": "p1"})
        by = {(r["conversation"], r["turn"]): r for r in rows}
        self.assertEqual(by["d1", 1]["steps"]["finished"], 1)  # the month-later steps are in no turn
        self.assertEqual(by["i1", 0]["steps"]["finished"], 1)  # the step after the second session started belongs to it alone
        total = sum(r["steps"]["finished"] for r in rows)
        self.assertEqual(total, 2)
        for r in rows:  # every number traces to the events named, with none cut
            self.assertEqual(r["steps"]["finished"], sum(1 for i in r["event_ids"] if i.startswith(("e08", "o2", "s"))))

    def test_a_long_turn_names_every_event_and_says_how_many_of_a_cut_list_there_were(self):
        events = [ev("p0", "chat.person", {"text": "run everything"}, "2026-10-16T10:00:00+00:00", project=None),
                  *[ev(f"m{i:03d}", "graph.move", {"move": f"move{i}", "allowed": True}, f"2026-10-16T10:{i // 60:02d}:{i % 60:02d}+00:00", project=None)
                    for i in range(1, 600)]]
        rows, _ = builders.episodes(events, [], text=True)
        self.assertEqual((len(rows[0]["event_ids"]), len(rows[0]["moves"]), rows[0]["moves_total"]), (600, 40, 599))

    def test_a_time_written_with_an_offset_is_the_instant_it_names(self):
        events = [ev("p0", "chat.person", {"text": "hello"}, "2026-10-16T10:00:00+00:00", project=None),
                  ev("p1", "chat.person", {"text": "again"}, "2026-10-16T12:30:00+02:00", project=None),
                  ev("a1", "chat.agent", {"text": "yes", "kind": "text"}, "2026-10-16T10:00:20Z", project=None)]
        rows, _ = builders.episodes(events, [], text=True)
        self.assertEqual([(r["turn"], r["agent"]["replies"]) for r in rows], [(0, 1), (1, 0)])  # 12:30+02:00 is 10:30 UTC: the reply at 10:00:20Z is turn 0's


@unittest.skipIf(MISSING, MISSING)
class ModelCallsTest(unittest.TestCase):
    def test_a_call_carries_its_check_and_the_feedback_on_the_message_it_produced(self):
        rows, manifest = builders.model_calls(flow(), workspace="w", drafts={"d1": "p1"})
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["purpose"], row["prompt"], row["reply"], row["check"], row["feedback"], row["label"]),
                         ("home_agent", "user: Which customers will leave?", "Which column says who left?", {"passed": "yes", "reason": None},
                          [{"kind": "thumbs", "rating": "helpful"}], "approved"))
        self.assertEqual({(s.kind, s.id) for s in manifest.sources}, {("event", "e03"), ("event", "e04")})
        self.assertEqual(manifest.use, "train")

    def test_an_edit_is_the_label_and_a_call_with_no_kept_text_is_not_here(self):
        events = flow() + [ev("e11", "feedback.given", {"on": "answer:m1", "kind": "edited_output", "edited": "Name the column that says who left."}, "2026-10-16T10:00:04+00:00", project=None),
                           ev("e12", "model.request", {"purpose": "intern", "outcome": "ok"}, "2026-10-16T10:00:05+00:00", model_request="mr2")]
        rows, _ = builders.model_calls(events)
        self.assertEqual((len(rows), rows[0]["label"], rows[0]["edited"]), (1, "edit", "Name the column that says who left."))  # the intern's call kept no text


@unittest.skipIf(MISSING or WORKSPACE_MISSING if "WORKSPACE_MISSING" in globals() else MISSING, "needs the studio")
class ExportTextTest(unittest.TestCase):
    """A model's prompts and replies leave only under train, redacted, and only from a purpose allowed to keep them."""

    def events(self):
        from dclab_core import experience as x

        def request(purpose, ident):
            return x.make("model.request", "model", {"workspace": "w", "project": "p1"}, {"purpose": purpose, "outcome": "ok", "prompt": "user: write to jane.roe@example.com",
                                                                                           "reply": "Dear Jane Roe, call +44 20 7946 0958"}, id=ident, at="2026-10-16T10:00:00+00:00", consent="train")

        return [request("home_agent", "ev000001"), request("evidence_answer", "ev000002"), request("problem_analyst", "ev000003"), request("embedding", "ev000004"),
                request("not_a_purpose", "ev000005")]

    def test_only_a_purpose_shown_the_persons_own_words_keeps_its_text_and_it_is_redacted(self):
        on = {"improve": True, "train": True}
        projects = {"p1": {"id": "p1", "settings": {"consent": on}}}
        out = consent.export_events(self.events(), "train", on, projects, {}, model_text=True)
        by = {d["id"]: d["payload"] for d in out}
        self.assertEqual(sorted(by["ev000001"]), ["outcome", "prompt", "purpose", "reply"])
        self.assertNotIn("jane.roe", by["ev000001"]["prompt"] + by["ev000001"]["reply"])  # redacted
        self.assertIn("[EMAIL]", by["ev000001"]["prompt"])
        for ident in ("ev000002", "ev000003", "ev000004", "ev000005"):  # lessons of other projects, file names, evidence records, an unknown purpose
            self.assertNotIn("prompt", by[ident], ident)
            self.assertNotIn("reply", by[ident], ident)
        plain = consent.export_events(self.events(), "train", on, projects, {})
        self.assertTrue(all("prompt" not in d["payload"] for d in plain))  # without the flag, no purpose keeps it

    def test_the_ids_that_join_an_event_to_a_person_a_request_or_a_trace_stay_home(self):
        from dclab_core import experience as x

        ids = {"workspace": "w", "project": "p1", "user": "u1", "request": "d" * 16, "trace": "a" * 32, "span": "b" * 16, "parent_span": "c" * 16}
        event = x.make("stage.finished", "system", ids, {"stage": "data"}, id="ev000001", at="2026-10-16T10:00:00+00:00", consent="train")
        on = {"improve": True, "train": True}
        out = consent.export_events([event], "train", on, {"p1": {"id": "p1", "settings": {"consent": on}}}, {})
        self.assertEqual(len(out), 1)
        self.assertEqual(set(out[0]["ids"]) & {"user", "request", "trace", "span", "parent_span"}, set())
        self.assertEqual(out[0]["ids"]["project"], "p1")  # the ids that say which project stay

    def test_it_is_refused_for_improve(self):
        on = {"improve": True, "train": False}
        with self.assertRaises(consent.ConsentRefused):
            consent.export_events(self.events(), "improve", on, {"p1": {"id": "p1", "settings": {"consent": on}}}, {}, model_text=True)


@unittest.skipIf(MISSING, MISSING)
class ManifestAndColumnsTest(unittest.TestCase):
    def test_the_manifest_names_the_scrubber_and_reads_the_older_versions(self):
        _, manifest = builders.episodes(flow(), records(), text=True)
        self.assertEqual((manifest.scrubber, manifest.schema_version), (redact.VERSION, "dclab_core.dataset_manifest.v4"))
        old = json.loads(manifest.model_dump_json())
        old["schema_version"] = "dclab_core.dataset_manifest.v3"
        for k in ("scrubber", "commit", "files"):
            old.pop(k)
        self.assertEqual(builders.DatasetManifest.read(old).schema_version, "dclab_core.dataset_manifest.v4")

    def test_a_column_that_mixes_kinds_is_written_as_text_and_a_file_that_cannot_be_written_leaves_no_old_one(self):
        import pandas as pd

        rows = [{"point": "split.strategy", "value_used": "time", "n": 1}, {"point": "holdout.reuse", "value_used": False, "n": 2.5}, {"point": "x", "value_used": 3, "n": 2**70}]
        path = Path(tempfile.mkdtemp()) / "rows.parquet"
        self.assertTrue(columnar.write_parquet(path, rows))
        back = pd.read_parquet(path)
        self.assertEqual([json.loads(v) for v in back["value_used"]], ["time", False, 3])  # every row of the mixed column is JSON text
        self.assertEqual(back["point"].tolist(), ["split.strategy", "holdout.reuse", "x"])  # a column of one kind is as it is
        self.assertEqual(json.loads(back["n"][2]), 2**70)
        with mock.patch.object(columnar, "frame_of", side_effect=ValueError("cannot")):
            self.assertFalse(columnar.write_parquet(path, rows))
        self.assertFalse(path.exists(), "an older file that holds rows the JSONL may no longer hold must not stay")

    def test_rows_are_written_as_columns_with_nested_values_as_json_and_read_back(self):
        import pandas as pd

        rows, _ = builders.episodes(flow(), records(), text=True, drafts={"d1": "p1"})
        path = Path(tempfile.mkdtemp()) / "rows.parquet"
        self.assertTrue(columnar.write_parquet(path, rows))
        back = pd.read_parquet(path)
        self.assertEqual(len(back), 2)
        self.assertEqual(json.loads(back["decisions"][0])[0]["id"], "dr000002")
        self.assertEqual(back["turn"].tolist(), [0, 1])
        self.assertFalse(columnar.write_parquet(path, []))  # no rows: no file
        self.assertFalse(path.exists())
        with mock.patch.object(columnar, "available", return_value=False):
            self.assertFalse(columnar.write_parquet(path, rows))  # no pyarrow: the JSONL stands alone


try:
    from fastapi.testclient import TestClient

    from dclab_rnd import consent, datasets, experience, ledger
    from dclab_rnd.experience import retention
    WORKSPACE_MISSING = ""
except ImportError as error:
    WORKSPACE_MISSING = str(error)


@unittest.skipIf(MISSING or WORKSPACE_MISSING, MISSING or WORKSPACE_MISSING)
class WorkspaceTest(unittest.TestCase):
    """A seeded workspace: sets by consent, the columns and the commit in each, a deletion followed by a rebuild."""

    def setUp(self):
        from dclab_rnd.agentic.server import create_app

        self.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        self.s = self.client.app.state.services
        self.home = Path(self.s.store.home)
        self.h = {"X-DCLab-Token": self.client.get("/api/config").json()["csrf"]}
        for use in ("improve", "train"):
            consent.set_workspace_consent(self.home, use, True)
        self.p1 = self.client.post("/api/projects", json={"name": "Mine", "goal": "g"}, headers=self.h).json()["id"]
        self.client.patch(f"/api/projects/{self.p1}", json={"settings": {"consent": {"train": True, "improve": True}}}, headers=self.h)
        consent.invalidate(self.p1)
        log, book = self.s.experience, self.s.ledger
        rec = lambda kind, actor, payload, at, **ids: experience.record(kind, actor, payload, log=log, at=at, **ids)  # noqa: E731
        rec("draft.created", "person", {"problem": "Which customers will leave us?"}, "2026-10-16T10:00:00+00:00", conversation="d1", project=self.p1)
        rec("chat.agent", "agent", {"text": "Which column says who left?", "kind": "question", "message": "m1"}, "2026-10-16T10:00:01+00:00",
            conversation="d1", project=self.p1, model_request="mr1")
        rec("model.request", "model", {"purpose": "home_agent", "tier": "t", "model": "m", "input_tokens": 10, "output_tokens": 5, "outcome": "ok",
                                       "prompt": "user: Which customers will leave us?", "reply": "Which column says who left?"},
            "2026-10-16T10:00:01+00:00", conversation="d1", project=self.p1, model_request="mr1")
        rec("model.checked", "system", {"purpose": "home_agent", "passed": "yes"}, "2026-10-16T10:00:02+00:00", conversation="d1", project=self.p1, model_request="mr1")
        rec("chat.person", "person", {"text": "The churned column"}, "2026-10-16T10:01:00+00:00", conversation="d1", project=self.p1)
        self.record = ledger.decide("problem.family", ("draft", "d1"), answers=[ledger.answer("rule", "r", "classification")], value_used="classification",
                                    used="rule", ledger=book, project=self.p1, conversation="d1")

    def built(self, out):
        return {(b["builder"], b["name"]): b for b in out["built"]}

    def rows(self, entry, name="rows.jsonl"):
        path = Path(entry["folder"]) / name
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def test_at_train_the_episodes_and_the_model_calls_are_written_with_columns_the_scrubber_and_the_commit(self):
        out = self.built(datasets.build(self.s))
        episodes, calls = out["episodes", "turns"], out["model_calls", "kept"]
        rows = self.rows(episodes) + self.rows(episodes, "sealed.jsonl")
        self.assertEqual([r["message"] for r in sorted(rows, key=lambda r: r["turn"])], ["Which customers will leave us?", "The churned column"])
        self.assertEqual(rows[0]["model_calls"]["n"] + rows[1]["model_calls"]["n"], 1)
        manifest = json.loads((Path(episodes["folder"]) / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["scrubber"], manifest["schema_version"]), (redact.VERSION, "dclab_core.dataset_manifest.v4"))
        self.assertRegex(manifest["commit"] or "", r"^[0-9a-f]{40}(\+dirty)?$")  # a build from uncommitted code says so
        self.assertTrue(set(manifest["files"]) <= {"rows.jsonl", "rows.parquet", "sealed.jsonl", "sealed.parquet"} and manifest["files"])
        if columnar.available():
            import pandas as pd

            name = "rows.parquet" if (Path(episodes["folder"]) / "rows.parquet").exists() else "sealed.parquet"
            self.assertGreaterEqual(len(pd.read_parquet(Path(episodes["folder"]) / name)), 1)
        taken = self.rows(calls) + self.rows(calls, "sealed.jsonl")
        self.assertEqual((taken[0]["prompt"], taken[0]["check"]["passed"], taken[0]["purpose"]), ("user: Which customers will leave us?", "yes", "home_agent"))
        self.assertTrue(all("@" not in r["prompt"] for r in taken))  # redacted on the way out

    def test_at_improve_alone_the_episodes_have_no_free_text_and_the_model_calls_are_not_built(self):
        consent.set_workspace_consent(self.home, "train", False)
        out = self.built(datasets.build(self.s))
        self.assertNotIn(("model_calls", "kept"), out)
        self.assertNotIn(("decisions", "problem.family"), out)
        episodes = out["episodes", "turns"]
        rows = self.rows(episodes) + self.rows(episodes, "sealed.jsonl")
        self.assertTrue(rows)
        blob = json.dumps(rows)
        for words in ("Which customers", "churned column", "Which column says"):
            self.assertNotIn(words, blob)
        self.assertEqual([r["decisions"] for r in rows], [[]] * len(rows))  # decisions leave for train only
        self.assertEqual(json.loads((Path(episodes["folder"]) / "manifest.json").read_text())["use"], "improve")

    def test_a_deletion_marks_the_sets_stale_and_the_rebuild_no_longer_holds_the_rows(self):
        datasets.build(self.s)
        root = self.home / datasets.FOLDER
        before = " ".join(p.read_text(encoding="utf-8", errors="ignore") for p in root.rglob("*.jsonl"))
        self.assertIn("Which customers", before)  # the rows are in the sets before the deletion
        self.assertFalse((root / datasets.STALE).exists())
        gone = retention.forget(project=self.p1, log=self.s.experience, ledger=self.s.ledger)
        self.assertGreater(gone["events"], 0)
        self.assertTrue((root / datasets.STALE).exists())
        result = datasets.sweep_due(self.s)
        self.assertIsNotNone(result)
        self.assertFalse((root / datasets.STALE).exists())
        self.assertTrue((root / datasets.BUILT).exists())
        text = " ".join(p.read_bytes().decode("utf-8", errors="ignore") for p in root.rglob("*") if p.is_file())  # rows, columns and manifests
        for words in ("Which customers", "churned column", self.record.id):
            self.assertNotIn(words, text)  # the rows are not in any set any more
        self.assertIsNone(datasets.sweep_due(self.s))  # nothing is due until tomorrow or the next deletion

    def test_the_sets_are_rebuilt_once_a_day_where_consent_allows_and_never_when_it_does_not(self):
        import os
        import time

        root = self.home / datasets.FOLDER
        self.assertIsNotNone(datasets.sweep_due(self.s))  # never built: due
        self.assertIsNone(datasets.sweep_due(self.s))
        self.assertIsNotNone(datasets.sweep_due(self.s, now=time.time() + datasets.NIGHTLY_SECONDS + 5))  # a day later
        with mock.patch.dict(os.environ, {"DCLAB_DATASETS_NIGHTLY": "0"}):
            self.assertIsNone(datasets.sweep_due(self.s, now=time.time() + 3 * datasets.NIGHTLY_SECONDS))  # the daily build is switched off
            datasets.mark_stale(self.home)
            self.assertIsNotNone(datasets.sweep_due(self.s))  # a deletion's rebuild is not
        self.assertTrue(any(root.glob("*/*/manifest.json")))
        for use in ("improve", "train"):
            consent.set_workspace_consent(self.home, use, False)  # a withdrawal marks the sets
        self.assertTrue((root / datasets.STALE).exists())
        cleaned = datasets.sweep_due(self.s)
        self.assertIsNotNone(cleaned)
        self.assertFalse(any(root.glob("*/*/manifest.json")), "the sets of a use now off are removed")
        self.assertIsNone(datasets.sweep_due(self.s))  # nothing on, nothing left: nothing to do

    def test_a_deletion_during_a_build_is_not_lost_and_a_build_that_fails_waits_an_hour(self):
        import time

        root = self.home / datasets.FOLDER
        real = datasets.build

        def deleting(services, root_=None):
            datasets.mark_stale(self.home)  # a deletion lands while the sets are being built
            return real(services, root_)

        datasets.mark_stale(self.home)
        with mock.patch.object(datasets, "build", deleting):
            datasets.sweep_due(self.s)
        self.assertTrue((root / datasets.STALE).exists(), "the sweep cleared a marker written after it began")
        with mock.patch.object(datasets, "build", side_effect=RuntimeError("down")):
            with self.assertRaises(RuntimeError):
                datasets.sweep_due(self.s)
            self.assertTrue((root / datasets.STALE).exists() and (root / datasets.FAILED).exists())
            self.assertIsNone(datasets.sweep_due(self.s))  # it does not try again every minute
        self.assertIsNotNone(datasets.sweep_due(self.s, now=time.time() + datasets.RETRY_SECONDS + 5))  # in an hour it does
        self.assertFalse((root / datasets.FAILED).exists())

    def test_one_set_that_cannot_be_written_never_costs_the_others(self):
        with mock.patch.object(builders, "outcome_labels", side_effect=ValueError("bad")):
            out = datasets.build(self.s)
        self.assertIn({"use": "outcomes", "why": "outcome_labels: ValueError"}, out["not_built"])
        self.assertIn(("episodes", "turns"), self.built(out))
        with mock.patch.object(columnar, "write_parquet", side_effect=RuntimeError("disk")):
            out = datasets.build(self.s)  # a columnar file that cannot be written is a set not built, not a crash
        self.assertTrue(out["not_built"])


if __name__ == "__main__":
    unittest.main()
