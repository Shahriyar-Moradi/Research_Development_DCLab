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


if __name__ == "__main__":
    unittest.main()
