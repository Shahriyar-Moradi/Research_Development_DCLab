"""The draft's data pipeline on Home: any file to a table, structural cleaning, descriptive analysis.

Fixtures live in tests/fixtures/draft/ (each under 20 KB). Excel and Parquet files are written
by the tests; the real Excel round trip runs only where openpyxl is installed.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.draft import analyze as an  # noqa: E402
from dclab_rnd.draft import clean as cl  # noqa: E402
from dclab_rnd.draft import structure as st  # noqa: E402

FIXTURES = ROOT / "tests/fixtures/draft"
HAS_OPENPYXL = importlib.util.find_spec("openpyxl") is not None
CUSTOM_REGEX = r"#(?P<req>\d+) (?P<user>\w+) -> (?P<method>[A-Z]+) (?P<path>\S+) took (?P<ms>\d+)ms"


class FakeClient:
    """The ``intern.llm.ChatClient`` shape: records every request, answers with fixed content or raises."""

    def __init__(self, content: str = "", error: Exception | None = None):
        self.content, self.error, self.calls = content, error, []

    def complete(self, messages, tools=None, max_tokens=1800):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return {"content": self.content, "tool_calls": []}


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name: str, content: str | bytes) -> Path:
        path = self.tmp / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
        return path


# ---------------------------------------------------------------------- structure


class StructureTabularTest(TempDirTest):
    def test_semicolon_csv_saved_as_txt(self):
        frame, info = st.to_table(FIXTURES / "messy_semicolon.txt")
        self.assertEqual(info["format"], "csv")
        self.assertEqual((info["rows"], info["columns"]), (14, 8))
        self.assertEqual(info["parse_rate"], 1.0)
        self.assertEqual(info["sample_lines"], [])
        self.assertEqual(list(frame.columns), ["Customer ID", "Name", "Signup Date", "Monthly Fee", "Discount", "Active", "Notes", "Fax"])
        self.assertIn("semicolon", info["notes"][0])
        self.assertEqual(frame.loc[0, "Monthly Fee"], "€431")

    def test_repeated_headers_and_bom(self):
        path = self.write("people.csv", "﻿a,a, b ,\n1,2,3,4\n5,6,7,8\n")
        frame, info = st.to_table(path)
        self.assertEqual(list(frame.columns), ["a", "a_2", "b", "column_4"])
        self.assertEqual(info["format"], "csv")

    def test_tab_separated_txt(self):
        frame, info = st.to_table(self.write("table.txt", "x\ty\n1\t2\n3\t4\n"))
        self.assertEqual(info["format"], "tsv")
        self.assertEqual(frame["y"].tolist(), [2, 4])

    def test_rows_over_the_cap_are_cut_with_a_note(self):
        path = self.write("big.csv", "n,v\n" + "".join(f"{i},{i * 2}\n" for i in range(30)))
        frame, info = st.to_table(path, max_rows=5)
        self.assertEqual(len(frame), 5)
        self.assertTrue(any("first 5" in n for n in info["notes"]))

    def test_rows_with_extra_fields_are_skipped_and_counted(self):
        frame, info = st.to_table(self.write("ragged.csv", "a,b\n1,2\n3,4,5\n6,7\n"))
        self.assertEqual(len(frame), 2)
        self.assertAlmostEqual(info["parse_rate"], 2 / 3, places=4)
        self.assertIn("Skipped 1", info["notes"][0])

    def test_parquet_with_list_cells(self):
        path = self.tmp / "items.parquet"
        pd.DataFrame({"id": [1, 2, 3], "tags": [["a", "b"], ["c"], []]}).to_parquet(path)
        frame, info = st.to_table(path)
        self.assertEqual(info["format"], "parquet")
        self.assertEqual(frame["tags"].tolist(), ["a|b", "c", None])
        self.assertTrue(any("Joined lists" in n for n in info["notes"]))

    def test_excel_reads_the_first_sheet_with_data(self):
        class FakeBook:
            sheet_names = ["Cover", "Data", "Other"]

            def __init__(self, path):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def parse(self, name, nrows=None):
                return pd.DataFrame() if name == "Cover" else pd.DataFrame({" amount ": [1, 2, 3], "city": ["a", "b", "c"]})

        path = self.write("book.xlsx", b"PK\x03\x04 not a real workbook")
        with mock.patch.object(st.pd, "ExcelFile", FakeBook):
            frame, info = st.to_table(path)
        self.assertEqual(info["format"], "excel")
        self.assertEqual(list(frame.columns), ["amount", "city"])
        self.assertIn("'Data'", info["notes"][0])

    def test_excel_without_its_engine_says_what_to_do(self):
        path = self.write("book.xlsx", b"PK\x03\x04")
        with mock.patch.object(st.pd, "ExcelFile", side_effect=ImportError("Missing optional dependency 'openpyxl'")):
            with self.assertRaises(st.StructureError) as caught:
                st.to_table(path)
        self.assertIn("CSV", str(caught.exception))

    @unittest.skipUnless(HAS_OPENPYXL, "openpyxl is not installed")
    def test_excel_round_trip(self):
        path = self.tmp / "real.xlsx"
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(path, index=False)
        frame, info = st.to_table(path)
        self.assertEqual((info["format"], info["rows"], list(frame.columns)), ("excel", 2, ["a", "b"]))


class StructureJsonTest(TempDirTest):
    def test_nested_json_uses_the_largest_list_of_objects(self):
        frame, info = st.to_table(FIXTURES / "customers_nested.json")
        self.assertEqual(info["format"], "json_nested")
        self.assertEqual(info["rows"], 6)
        for column in ("id", "name", "address.city", "address.geo.lat", "tags", "orders.count", "profile.a.b.c.d"):
            self.assertIn(column, frame.columns)
        self.assertNotIn("orders", frame.columns)
        self.assertEqual(frame["tags"].tolist()[:2], ["web", "new|mobile"])
        self.assertEqual(frame["orders.count"].tolist(), [0, 1, 2, 0, 1, 2])
        self.assertEqual(json.loads(frame.loc[3, "profile.a.b.c.d"]), {"e": 3})
        notes = " ".join(info["notes"])
        self.assertIn("`data.customers`", notes)
        self.assertIn("Flattened 4 levels of nesting into 8 columns", notes)
        self.assertIn("`orders.count`", notes)

    def test_object_of_objects_becomes_keyed_records(self):
        data = {"u1": {"age": 30, "plan": "pro"}, "u2": {"age": 41, "plan": "basic"}, "u3": {"age": 25}}
        frame, info = st.to_table(self.write("users.json", json.dumps(data)))
        self.assertEqual(frame["key"].tolist(), ["u1", "u2", "u3"])
        self.assertEqual(info["format"], "json_nested")

    def test_list_of_records(self):
        frame, info = st.to_table(self.write("rows.json", json.dumps([{"a": 1, "b": {"c": 2}}, {"a": 3, "b": {"c": 4}}, 7])))
        self.assertEqual(info["format"], "json_records")
        self.assertEqual(list(frame.columns), ["a", "b.c"])
        self.assertAlmostEqual(info["parse_rate"], 2 / 3, places=4)

    def test_json_lines_with_a_bad_line(self):
        frame, info = st.to_table(FIXTURES / "events.jsonl")
        self.assertEqual(info["format"], "jsonl")
        self.assertEqual(info["rows"], 20)
        self.assertAlmostEqual(info["parse_rate"], 20 / 21, places=4)
        self.assertIn("device.os", frame.columns)
        self.assertIn("1 of 21 lines were not valid JSON", info["notes"][0])

    def test_json_lines_hidden_in_json_and_ndjson_files(self):
        lines = "\n".join(json.dumps({"i": i, "v": i * 1.5}) for i in range(5))
        for name in ("events.json", "events.ndjson", "events.log"):
            with self.subTest(name=name):
                frame, info = st.to_table(self.write(name, lines))
                self.assertEqual((info["format"], len(frame)), ("jsonl", 5))

    def test_mostly_bad_json_lines_are_refused(self):
        path = self.write("bad.jsonl", '{"a": 1}\nnot json\nnot json either\n{"a": 2\n')
        with self.assertRaises(st.StructureError) as caught:
            st.to_table(path)
        self.assertIn("not valid JSON", str(caught.exception))


class StructureLogTest(TempDirTest):
    def test_combined_access_log(self):
        frame, info = st.to_table(FIXTURES / "access.log")
        self.assertEqual((info["format"], info["parser"]), ("log_combined", "builtin"))
        self.assertEqual(info["rows"], 50)
        self.assertEqual(info["parse_rate"], 0.96)
        self.assertEqual(len(info["sample_lines"]), 5)
        self.assertIn("48 of 50 lines matched the access-log pattern; 2 kept as unparsed", info["notes"][0])
        for column in ("ip", "time", "method", "path", "status", "bytes", "referrer", "user_agent", "unparsed_line"):
            self.assertIn(column, frame.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["time"]))
        self.assertEqual(frame.loc[0, "time"].isoformat(), "2024-10-10T13:55:36-07:00")
        parsed = frame[frame["unparsed_line"].isna()]
        self.assertEqual(set(parsed["status"].astype(int)), {200, 302, 304, 404, 500})
        self.assertTrue((parsed.loc[parsed["status"] == 304, "bytes"] == 0).all())  # "-" means no body
        self.assertTrue(parsed["referrer"].isna().any())  # "-" means no referrer
        self.assertEqual(frame.loc[2, "path"], "/api/items?page=2&size=20")
        self.assertEqual(frame["unparsed_line"].dropna().tolist(), ["nginx: worker process restarted", "## log rotated ##"])

    def test_key_value_log(self):
        frame, info = st.to_table(FIXTURES / "app.log")
        self.assertEqual(info["format"], "log_keyvalue")
        self.assertEqual(info["parse_rate"], 1.0)
        self.assertEqual(list(frame.columns), ["timestamp", "level", "request_id", "user", "duration_ms", "status", "msg"])
        self.assertEqual(str(frame.loc[1, "timestamp"]), "2024-05-02 08:00:13.007000")
        self.assertEqual(frame.loc[3, "msg"], "payment for order 1003 failed")
        self.assertEqual(frame["level"].unique().tolist(), ["INFO", "WARN", "ERROR", "DEBUG"])
        self.assertEqual(frame["duration_ms"].dtype, object)  # values of keys stay text; clean reads numbers

    def test_timestamp_level_message_lines(self):
        lines = "\n".join(f"2024-01-0{1 + i % 9}T10:0{i % 10}:00Z [{['INFO', 'ERROR'][i % 2]}] job {i} finished" for i in range(12))
        for name in ("worker.log", "worker.txt"):
            with self.subTest(name=name):
                frame, info = st.to_table(self.write(name, lines))
                self.assertEqual(info["format"], "log_lines")
                self.assertEqual(list(frame.columns), ["timestamp", "level", "message"])
                self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["timestamp"]))
                self.assertEqual(frame.loc[1, "message"], "job 1 finished")

    def test_times_without_a_year_stay_text(self):
        lines = "\n".join(f"Oct  {i % 9 + 1} 14:32:0{i % 10} host sshd[12{i}]: Accepted password for u{i}" for i in range(10))
        frame, info = st.to_table(self.write("auth.log", lines))
        self.assertEqual((info["format"], list(frame.columns)), ("log_lines", ["line", "timestamp"]))
        self.assertEqual(frame.loc[0, "timestamp"], "Oct  1 14:32:00")
        self.assertTrue(any("year" in n for n in info["notes"]))

    def test_unknown_log_lines_fall_back_to_one_row_per_line(self):
        frame, info = st.to_table(FIXTURES / "custom.log")
        self.assertEqual((info["format"], info["parser"], info["parse_rate"]), ("log_lines", "builtin", 0.0))
        self.assertEqual(list(frame.columns), ["line"])
        self.assertEqual(len(frame), 40)

    def test_plain_text_paragraphs(self):
        frame, info = st.to_table(FIXTURES / "notes.txt")
        self.assertEqual(info["format"], "text")
        self.assertEqual(list(frame.columns), ["text", "chars", "words"])
        self.assertEqual(len(frame), 4)
        self.assertTrue(frame.loc[1, "text"].startswith("Customers on the basic plan"))
        self.assertIn("pro plan. Some of them", frame.loc[1, "text"])  # lines of a paragraph are joined
        self.assertEqual(frame.loc[0, "words"], len(frame.loc[0, "text"].split()))
        self.assertTrue(info["sample_lines"])

    def test_plain_text_without_blank_lines_is_one_row_per_line(self):
        frame, info = st.to_table(self.write("todo.txt", "buy milk and bread\ncall the bank about the card\nwrite the report"))
        self.assertEqual((info["format"], len(frame)), ("text", 3))


class StructureModelTest(unittest.TestCase):
    def test_a_good_model_pattern_is_checked_and_used(self):
        reply = "```json\n" + json.dumps({"regex": CUSTOM_REGEX, "types": {"ms": "int", "req": "int", "nope": "int", "user": "color"}}) + "\n```"
        client = FakeClient(reply)
        frame, info = st.to_table(FIXTURES / "custom.log", client=client)
        self.assertEqual(len(client.calls), 1)
        shown = client.calls[0][-1]["content"].splitlines()[1:]
        self.assertEqual(len(shown), st.MODEL_LINES)  # the model sees 20 of the 40 lines, never more
        self.assertEqual((info["format"], info["parser"], info["parse_rate"]), ("log_lines", "model", 1.0))
        self.assertEqual(list(frame.columns), ["req", "user", "method", "path", "ms"])
        self.assertEqual(frame["ms"].dtype, "int64")
        self.assertEqual(frame.loc[1, "user"], "bob")
        self.assertTrue(any("checked before use" in n for n in info["notes"]))
        # raw lines left the machine: the record the user sees says so
        self.assertEqual(info["model_lines_sent"], st.MODEL_LINES)
        self.assertTrue(any("were sent to the configured model" in n for n in info["notes"]))

    def test_the_pipeline_can_be_told_never_to_send_lines(self):
        import os
        import shutil
        import tempfile
        from unittest import mock
        from dclab_rnd.draft import pipeline
        from dclab_rnd.draft.store import DraftStore

        reply = json.dumps({"regex": CUSTOM_REGEX, "types": {"ms": "int", "req": "int"}})
        for setting, calls, parser in (("1", 1, "model"), ("0", 0, "builtin")):
            store = DraftStore(Path(tempfile.mkdtemp()))
            draft = store.create("Find slow requests in our service log")
            shutil.copyfile(FIXTURES / "custom.log", store.data_dir(draft["id"]) / "custom.log")
            asset = pipeline.new_asset(store, draft["id"], "upload", "custom.log", "custom.log")
            client = FakeClient(reply)
            with mock.patch.dict(os.environ, {"DCLAB_MODEL_READS_SAMPLE_LINES": setting}):
                final = pipeline.run(store, draft["id"], asset["id"], None, client)
            self.assertEqual((final["status"], len(client.calls), final["structure"]["parser"]), ("ready", calls, parser), setting)
            self.assertEqual("model_lines_sent" in final["structure"], setting == "1")

    def test_bad_model_patterns_are_ignored(self):
        cases = {
            "does not compile": json.dumps({"regex": "(?P<a>[", "types": {}}),
            "one field": json.dumps({"regex": r"(?P<line>.*)", "types": {}}),
            "reads too few lines": json.dumps({"regex": r"#(?P<req>\d+) (?P<user>alice) .*", "types": {}}),
            "not json": "I think the lines are request logs.",
        }
        for label, reply in cases.items():
            with self.subTest(label):
                frame, info = st.to_table(FIXTURES / "custom.log", client=FakeClient(reply))
                self.assertEqual((info["format"], info["parser"]), ("log_lines", "builtin"))
                self.assertEqual(list(frame.columns), ["line"])
                self.assertTrue(any("model" in n for n in info["notes"]))

    def test_a_failing_client_falls_back_with_a_note(self):
        frame, info = st.to_table(FIXTURES / "custom.log", client=FakeClient(error=RuntimeError("timeout")))
        self.assertEqual((info["parser"], len(frame)), ("builtin", 40))
        self.assertTrue(any("RuntimeError" in n for n in info["notes"]))

    def test_the_model_is_not_asked_when_a_built_in_pattern_reads_the_file(self):
        client = FakeClient(json.dumps({"regex": CUSTOM_REGEX}))
        st.to_table(FIXTURES / "access.log", client=client)
        st.to_table(FIXTURES / "messy_semicolon.txt", client=client)
        self.assertEqual(client.calls, [])


class StructureErrorTest(TempDirTest):
    def test_unusable_files_raise_a_message_for_the_user(self):
        cases = {
            "empty.csv": b"",
            "blank.txt": b"   \n\n  \n",
            "header_only.csv": b"a,b,c\n",
            "broken.json": b'{"a": [1, 2,',
            "image.bin": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
        }
        for name, content in cases.items():
            with self.subTest(name):
                with self.assertRaises(st.StructureError) as caught:
                    st.to_table(self.write(name, content))
                self.assertGreater(len(str(caught.exception)), 20)
        with self.assertRaises(st.StructureError):
            st.to_table(self.tmp / "missing.csv")

    def test_oversized_files_are_refused_before_reading(self):
        path = self.write("data.csv", "a,b\n1,2\n")
        with mock.patch.object(st, "MAX_BYTES", 4), mock.patch.object(st.pd, "read_csv") as read:
            with self.assertRaises(st.StructureError) as caught:
                st.to_table(path)
        read.assert_not_called()
        self.assertIn("limit", str(caught.exception))

    def test_unique_names(self):
        self.assertEqual(st.unique_names([" a", "a", "a_2", "", 3]), ["a", "a_2", "a_2_2", "column_4", "3"])


# ---------------------------------------------------------------------- clean


def entry(log, step):
    found = [e for e in log if e["step"] == step]
    return found[0] if found else None


class CleanTest(unittest.TestCase):
    def test_names(self):
        frame = pd.DataFrame([[1, 2, 3]], columns=[" a  b ", "a b", "c"])
        out, log = cl.clean(frame)
        self.assertEqual(list(out.columns), ["a b", "a b_2", "c"])
        self.assertEqual(entry(log, "names")["changed"], 2)

    def test_missing_markers(self):
        frame = pd.DataFrame({"note": ["x", " NA ", "null", "-", "?", "ok", None, "nan", "Na", "None"], "n": range(10)})
        out, log = cl.clean(frame)
        self.assertEqual(entry(log, "missing_markers")["changed"], 6)
        self.assertEqual(int(out["note"].isna().sum()), 7)  # the six markers plus the cell that was already empty
        self.assertEqual(out["note"].dropna().tolist(), ["x", "ok", "Na"])

    def test_trim(self):
        frame = pd.DataFrame({"name": [" a", "b  c", "d", "  e  f  "], "n": [1, 2, 3, 4]})
        out, log = cl.clean(frame)
        self.assertEqual(out["name"].tolist(), ["a", "b c", "d", "e f"])
        self.assertEqual(entry(log, "trim")["changed"], 3)

    def test_numbers(self):
        good = ["$1,200", "€3", "1_000", "2 500", "-£7.5", "12", "0.5", "1e3", "+4", "£ 9"] * 2
        frame = pd.DataFrame({
            "price": good[:19] + ["call us"],                    # 19 of 20 read: converted, one becomes missing
            "share": ["5%", "10%", "12.5%", "0%"] * 5,            # percentages kept as written
            "zip": ["02134", "10001", "94105", "60601"] * 5,      # leading zero: codes, stays text
            "mixed": ["1", "2", "x", "y"] + ["3"] * 16,          # 90% read: stays text
            "decimal_comma": ["1,5", "2,25", "3,0", "4,75"] * 5,  # not a thousands separator: stays text
        })
        out, log = cl.clean(frame)
        step = entry(log, "numbers")
        self.assertEqual(step["columns"], ["price", "share"])
        self.assertEqual(step["changed"], 40)
        self.assertEqual(out["price"].tolist()[:10], [1200.0, 3.0, 1000.0, 2500.0, -7.5, 12.0, 0.5, 1000.0, 4.0, 9.0])
        self.assertTrue(np.isnan(out.loc[19, "price"]))
        self.assertIn("1 value that did not read as numbers became missing: 'call us'", step["detail"])
        self.assertEqual(out["share"].tolist()[:4], [5.0, 10.0, 12.5, 0.0])
        self.assertIn("percentages", step["detail"])
        for column in ("zip", "mixed", "decimal_comma"):
            self.assertEqual(out[column].dtype, object, column)

    def test_whole_numbers_become_integers(self):
        out, _ = cl.clean(pd.DataFrame({"n": ["1,000", "2", "3"], "m": ["1", "2", None]}))
        self.assertEqual(str(out["n"].dtype), "int64")
        self.assertEqual(str(out["m"].dtype), "float64")  # a gap needs a float column

    def test_dates(self):
        frame = pd.DataFrame({
            "signed": ["2024-01-05", "2024-01-06 10:30:00", "2024-02-10", "2024-03-01", "2024-03-02",
                       "2024-03-03", "2024-03-04", "2024-03-05", "2024-03-06", "unknown"],
            "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"] * 2,
            "phone": ["0612345678", "0698765432"] * 5,
        })
        out, log = cl.clean(frame)
        step = entry(log, "dates")
        self.assertEqual(step["columns"], ["signed"])
        self.assertEqual(step["changed"], 10)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out["signed"]))
        self.assertEqual(out.loc[1, "signed"], pd.Timestamp("2024-01-06 10:30:00"))
        self.assertTrue(pd.isna(out.loc[9, "signed"]))
        self.assertIn("'unknown'", step["detail"])
        self.assertEqual(out["weekday"].dtype, object)
        self.assertEqual(out["phone"].dtype, object)

    def test_booleans(self):
        frame = pd.DataFrame({"row": range(7), "active": ["yes", "No", "Y", "n", "TRUE", "f", None],
                              "maybe": ["yes", "no", "maybe", "yes", "no", "yes", "no"]})
        out, log = cl.clean(frame)
        self.assertEqual(str(out["active"].dtype), "boolean")
        self.assertEqual(out["active"].tolist()[:6], [True, False, True, False, True, False])
        self.assertTrue(pd.isna(out.loc[6, "active"]))
        self.assertEqual(entry(log, "booleans")["changed"], 6)
        self.assertEqual(out["maybe"].dtype, object)

    def test_empty_rows_and_columns(self):
        frame = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"], "c": [None, None, None]})
        out, log = cl.clean(frame)
        step = entry(log, "empty")
        self.assertEqual(list(out.columns), ["a", "b"])
        self.assertEqual(len(out), 2)
        self.assertEqual((step["changed"], step["columns"], step["rows_before"], step["rows_after"]), (2, ["c"], 3, 2))

    def test_duplicates(self):
        frame = pd.DataFrame({"a": [1, 2, 1, 1, 3], "b": ["x", "y", "x", "x", "y"]})
        out, log = cl.clean(frame)
        step = entry(log, "duplicates")
        self.assertEqual((step["changed"], step["rows_before"], step["rows_after"]), (2, 5, 3))
        self.assertEqual(out["a"].tolist(), [1, 2, 3])
        self.assertEqual(list(out.index), [0, 1, 2])

    def test_steps_run_in_order_and_only_changes_are_logged(self):
        frame = pd.DataFrame({" Fee ": ["$10", "$20", "$20", None], "when": ["2024-01-01", "2024-01-02", "2024-01-02", None],
                              "ok": [" yes", "no", "no", None], "blank": ["", " ", "", None]})
        out, log = cl.clean(frame)
        self.assertEqual([e["step"] for e in log], ["names", "missing_markers", "trim", "numbers", "dates", "booleans", "empty", "duplicates"])
        self.assertEqual([e["changed"] for e in log], [1, 3, 1, 3, 3, 3, 2, 1])
        self.assertEqual(list(out.columns), ["Fee", "when", "ok"])
        self.assertEqual(len(out), 2)
        for item in log:
            self.assertEqual(set(item), {"step", "title", "detail", "columns", "rows_before", "rows_after", "changed"})
            self.assertGreater(item["changed"], 0)
        tidy = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        self.assertEqual(cl.clean(tidy)[1], [])

    def test_the_input_frame_is_not_changed(self):
        frame = pd.DataFrame({" a ": [" 1,000 ", "2", "2", None], "b": ["yes", "no", "no", None], "c": ["NA", "x", "x", None]})
        before = frame.copy(deep=True)
        cl.clean(frame)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(list(frame.columns), [" a ", "b", "c"])

    def test_category_code_columns(self):
        frame = pd.DataFrame({"region": [1, 2, 3, 1, 2] * 4, "fee": np.linspace(1, 50, 20), "flag": [True, False] * 10, "city": ["a"] * 20})
        self.assertEqual(cl.category_code_columns(frame), ["region"])


# ---------------------------------------------------------------------- analyze


def customers(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    days = [d for d in pd.date_range("2023-01-01", "2023-12-31") if d.month != 6]  # June is missing
    frame = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "signup": [days[i % len(days)] for i in range(n)],
        "x1": x1, "x2": x1 + rng.normal(scale=0.3, size=n), "x3": -x1 + rng.normal(scale=0.5, size=n),
        "revenue": 10 * x1 + rng.normal(size=n),         # named like a target: a target candidate
        **{f"c{k}": rng.integers(0, 5, n) for k in range(1, 6)},
        "flag": (x1 > 0).astype(int),                    # few levels: a target candidate, correlated with x1
        "country": ["DE"] * (n - 1) + ["FR"],
        "notes": np.where(np.arange(n) % 2 == 0, None, "called"),
        "final_refund_days": rng.uniform(0, 50, n),     # named like a value known only after the outcome
        "churned": rng.choice(["yes", "no"], n),
    })
    return pd.concat([frame, frame.iloc[:2]], ignore_index=True)  # two exact duplicate rows


class AnalyzeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = customers()
        cls.result = an.analyze(cls.frame)

    def test_output_is_plain_json(self):
        text = json.dumps(self.result, allow_nan=False)
        self.assertEqual(json.loads(text), self.result)
        odd = pd.DataFrame({"f": [1.5, np.nan, np.inf, 2.0], "i": pd.array([1, None, 3, 4], dtype="Int64"),
                            "b": pd.array([True, None, False, True], dtype="boolean"),
                            "t": pd.to_datetime(["2024-01-01T00:00:00+02:00"] * 3 + [None]), "s": ["a", None, "b", "a"]})
        json.dumps(an.analyze(odd), allow_nan=False)

    def test_summary_and_columns(self):
        summary = self.result["summary"]
        self.assertEqual((summary["rows"], summary["columns"], summary["duplicate_rows"]), (402, 16, 2))
        self.assertNotIn("sampled_rows", summary)
        by_name = {c["name"]: c for c in self.result["columns"]}
        self.assertLessEqual(len(by_name["x1"]["hist"]["counts"]), 20)
        self.assertEqual(sum(by_name["x1"]["hist"]["counts"]), 402)
        self.assertEqual(set(by_name["x1"]["quantiles"]), {"p01", "p25", "p50", "p75", "p99"})
        self.assertEqual(by_name["churned"]["kind"], "categorical")
        self.assertEqual(sum(n for _, n in by_name["churned"]["top"]) + by_name["churned"]["other_count"], 402)
        self.assertLessEqual(len(by_name["customer_id"]["top"]), 10)
        time = by_name["signup"]["time"]
        self.assertEqual((time["period"], len(time["by"])), ("month", 12))
        self.assertEqual(time["by"][5], ["2023-06", 0])

    def test_floats_are_rounded(self):
        x1 = next(c for c in self.result["columns"] if c["name"] == "x1")
        for value in [x1["mean"], x1["std"], *x1["quantiles"].values()]:
            digits = repr(value).split(".")[-1]
            self.assertLessEqual(len(digits.lstrip("0")) if abs(value) < 1 else len(digits), 4, value)

    def test_correlations_leave_out_the_target_and_outcome_like_columns(self):
        # Only the stated target and columns whose names read like an outcome; profile_table's wider
        # target_candidates list would leave nothing to describe on a narrow table.
        likely = {c["name"] for c in self.result["profile"]["columns"] if c.get("target_name_like")}
        self.assertTrue(likely)
        pairs = self.result["correlations"]
        named = {name for pair in pairs for name in (pair["a"], pair["b"])}
        self.assertFalse(named & likely)
        self.assertIn(("x1", "x2"), {(p["a"], p["b"]) for p in pairs})
        self.assertTrue(all(abs(p["rho"]) >= 0.3 for p in pairs))
        self.assertEqual(self.result["summary"]["correlation_excluded"], sorted(likely))
        with_target = an.analyze(self.frame, target="x3")
        named = {name for pair in with_target["correlations"] for name in (pair["a"], pair["b"])}
        self.assertNotIn("x3", named)
        self.assertIn("x3", {name for pair in pairs for name in (pair["a"], pair["b"])})

    def test_highlights(self):
        titles = {h["title"]: h for h in self.result["highlights"]}
        after = titles["Check when these values are known"]
        self.assertEqual((after["severity"], after["columns"]), ("warning", ["final_refund_days"]))
        self.assertIn("known when you predict", after["text"])
        self.assertEqual(titles["Columns with many missing values"]["columns"], ["notes"])
        self.assertIn("customer_id", titles["Likely identifiers"]["columns"])
        self.assertEqual(titles["Nearly constant columns"]["columns"], ["country"])
        self.assertIn("2 rows", titles["Duplicate rows"]["text"])
        self.assertIn("c1", titles["Numbers that may be category codes"]["columns"])
        self.assertIn("no rows in 1 of 12 months", titles["Gaps in time"]["text"])
        for item in self.result["highlights"]:
            self.assertIn(item["severity"], ("info", "warning"))

    def test_large_tables_are_described_from_a_seeded_sample(self):
        with mock.patch.object(an, "SAMPLE_ROWS", 300):
            first = an.analyze(self.frame)
            second = an.analyze(self.frame)
        self.assertEqual(first["summary"]["sampled_rows"], 300)
        self.assertEqual(first["summary"]["rows"], 402)
        x1 = next(c for c in first["columns"] if c["name"] == "x1")
        self.assertEqual(sum(x1["hist"]["counts"]), 300)
        self.assertEqual(first, second)

    def test_max_columns(self):
        result = an.analyze(self.frame, max_columns=3)
        self.assertEqual(len(result["columns"]), 3)
        self.assertEqual(result["summary"]["columns_described"], 3)


class PipelineTest(unittest.TestCase):
    def test_every_fixture_goes_through_structure_clean_and_analyze(self):
        for path in sorted(FIXTURES.iterdir()):
            with self.subTest(path.name):
                self.assertLess(path.stat().st_size, 20_000)
                frame, info = st.to_table(path)
                self.assertIn(info["format"], st.FORMATS)
                cleaned, _ = cl.clean(frame)
                json.dumps(an.analyze(cleaned), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
