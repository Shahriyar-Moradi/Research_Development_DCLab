"""Any file to a table: tabular files, JSON, JSON lines, logs and plain text.

A user on Home brings data in whatever shape they have. This module turns one file into a
pandas table and says how it read it (``info``), so the chat can explain the result and the
user can see when a file was read the wrong way. Parsing is deterministic: built-in readers
and log patterns come first. A configured model may *propose* one regular expression for log
lines the built-ins cannot read; code checks the proposal on the first 200 lines and uses it
only when it reads at least 80% of them and beats every built-in pattern (the model is
advisory, code decides). The model never sees more than 20 lines.

Choices worth knowing:

- Nested JSON: the largest list of objects in the tree becomes the rows (or, failing that, an
  object whose values are all objects, keyed by ``key``). Nested objects are flattened with
  ``.`` up to four levels; anything deeper stays as JSON text in one cell.
- Lists inside records: lists of plain values are joined with ``|`` into one cell, so the
  values stay readable and countable; lists of objects become their item count in
  ``<name>.count``, since one row cannot hold a sub-table. A note names those columns.
- Logs: lines no pattern reads are kept, raw, in ``unparsed_line``; nothing is dropped silently.
- Types: only what a format defines (log times, status codes, JSON numbers). Everything else
  is left to ``clean``, which logs each change for the user.
"""

from __future__ import annotations

import csv
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from dclab_rnd.studio.data import load_table

try:  # pandas >= 2.2
    from pandas.tseries.api import guess_datetime_format
except ImportError:  # pragma: no cover - older pandas
    guess_datetime_format = None

MAX_BYTES = 200 * 1024 * 1024
HEAD_BYTES = 64 * 1024
SAMPLE_LINES = 200        # a parser is judged on the first 200 non-empty lines
MODEL_LINES = 20          # the most the model ever sees
MODEL_LINE_CHARS = 400
MODEL_MAX_LINE = 2000     # longer lines are not fed to a model-written pattern (backtracking guard)
GOOD_PARSE = 0.8          # below this a model may propose a pattern
LOG_MIN_PARSE = 0.5       # a structured log pattern must read this share of lines to be chosen
MAX_NESTING = 4
MAX_SAMPLE_LINE_CHARS = 500
DELIMITERS = ",;\t|"
DELIMITER_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}
EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")
FORMATS = ("csv", "tsv", "excel", "parquet", "json_records", "json_nested", "jsonl",
           "log_combined", "log_keyvalue", "log_lines", "text")


class StructureError(ValueError):
    """The file cannot become a table; the message is plain English, safe to show, and says what to do."""


# ---------------------------------------------------------------------- shared helpers


def unique_names(names: Iterable[Any]) -> list[str]:
    """Column names as stripped strings, made unique with ``_2``, ``_3``; blanks become ``column_<n>``."""
    out: list[str] = []
    seen: set[str] = set()
    for position, name in enumerate(names):
        base = str(name).strip() or f"column_{position + 1}"
        candidate, n = base, 2
        while candidate in seen:
            candidate, n = f"{base}_{n}", n + 1
        seen.add(candidate)
        out.append(candidate)
    return out


_COMMA_FRACTION = re.compile(r"(\d{2}:\d{2}:\d{2}),(\d+)")


def parse_times(values: pd.Series, fmt: str | None = None) -> pd.Series:
    """Datetimes from strings, NaT where a value does not parse; mixed UTC offsets become UTC.

    Tries the given format, else a format guessed from the first value and ISO 8601 (both
    fast), and only then pandas' per-value ``mixed`` parsing (slow on large logs).
    """
    series = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    text = series.where(series.isna(), series.astype(str).str.replace(_COMMA_FRACTION, r"\1.\2", regex=True))
    present = int(text.notna().sum())
    candidates: list[str] = [fmt] if fmt else []
    if not fmt:
        first = text.dropna().head(1)
        if guess_datetime_format is not None and len(first):
            guessed = guess_datetime_format(str(first.iloc[0]))
            if guessed:
                candidates.append(guessed)
        candidates.append("ISO8601")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for candidate in candidates:
            parsed = _to_datetime(text, candidate)
            if parsed is not None and (fmt or present == 0 or parsed.notna().sum() >= 0.95 * present):
                return parsed
        parsed = _to_datetime(text, "mixed")
    return parsed if parsed is not None else pd.Series(pd.NaT, index=series.index)


def _to_datetime(text: pd.Series, fmt: str) -> pd.Series | None:
    try:
        out = pd.to_datetime(text, format=fmt, errors="coerce")
        if not pd.api.types.is_datetime64_any_dtype(out):  # mixed offsets come back as objects
            out = pd.to_datetime(text, format=fmt, errors="coerce", utc=True)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if pd.api.types.is_datetime64_any_dtype(out) else None


def _as_number(series: pd.Series) -> pd.Series:
    """Numbers from strings; whole numbers without gaps stay integers."""
    out = pd.to_numeric(series, errors="coerce")
    if out.notna().all() and len(out) and (out % 1 == 0).all():
        return out.astype("int64")
    return out.astype("float64")


def _decode(raw: bytes) -> tuple[str, str | None]:
    try:
        return raw.decode("utf-8-sig"), None
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "The file is not UTF-8; it was read as Latin-1, so check accented characters."


def _read_text(path: Path, info: dict[str, Any]) -> str:
    text, note = _decode(path.read_bytes())
    if note:
        info["notes"].append(note)
    return text


def _nonempty_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def _head_lines(head: bytes) -> list[str]:
    text = head.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    if len(head) >= HEAD_BYTES and lines:
        lines = lines[:-1]  # the last line may be cut in half
    return [line.rstrip() for line in lines if line.strip()]


def _samples(lines: list[str]) -> list[str]:
    return [line[:MAX_SAMPLE_LINE_CHARS] for line in lines[:5]]


# ---------------------------------------------------------------------- entry point


def to_table(path: Path, *, client: Any = None, max_rows: int = 500_000) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read ``path`` into a table and say how (see the module docstring).

    ``info`` holds ``format``, ``parser`` ("builtin" or "model"), ``parse_rate`` (share of
    non-empty lines or records that parsed; 1.0 for tabular files), ``rows``, ``columns``,
    ``notes`` (short sentences for the user) and ``sample_lines`` (logs and text only).
    ``client`` is optional (``intern.llm.ChatClient`` shape); it is asked at most once.
    """
    path = Path(path)
    if not path.is_file():
        raise StructureError(f"The file {path.name} was not found. Upload it again.")
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise StructureError(f"The file is {size / 1e6:,.0f} MB and the limit is {MAX_BYTES / 1e6:,.0f} MB. "
                             "Upload a sample (for example the first 500,000 rows) or split the file.")
    with path.open("rb") as handle:
        head = handle.read(HEAD_BYTES)
    if not head.strip():
        raise StructureError("The file is empty. Check that the export finished and upload it again.")
    kind = _detect(path, head)
    info: dict[str, Any] = {"format": kind, "parser": "builtin", "parse_rate": 1.0, "rows": 0, "columns": 0,
                            "notes": [], "sample_lines": []}
    reader = _READERS[kind]
    frame = reader(path, head, info, client, max_rows)
    frame.columns = unique_names(frame.columns)
    if len(frame) > max_rows:
        frame = frame.head(max_rows)
        info["notes"].append(f"The file has more than {max_rows:,} rows; kept the first {max_rows:,}.")
    if frame.shape[1] == 0 or len(frame) == 0:
        raise StructureError("No rows were found in the file. If it is a spreadsheet, check that the first sheet "
                             "holds the data with one header row; otherwise export it as CSV and try again.")
    frame = frame.reset_index(drop=True)
    info["rows"], info["columns"] = int(len(frame)), int(frame.shape[1])
    info["parse_rate"] = round(float(info["parse_rate"]), 4)
    return frame, info


def _detect(path: Path, head: bytes) -> str:
    """The reader for this file, from its extension and its first 64 KB."""
    suffix = path.suffix.lower()
    if suffix == ".parquet" or head.startswith(b"PAR1"):
        return "parquet"
    if suffix in EXCEL_SUFFIXES or head.startswith((b"PK\x03\x04", b"\xd0\xcf\x11\xe0")):  # xlsx (zip) or xls (OLE)
        return "excel"
    if b"\x00" in head:
        raise StructureError("This looks like a binary file, not text, CSV, Excel, Parquet or JSON. "
                             "Export the data as CSV or Excel and upload that.")
    lines = _head_lines(head)
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    if head.decode("utf-8-sig", errors="replace").lstrip()[:1] in ("{", "["):
        if _looks_like_json_lines(lines):
            return "jsonl"
        if suffix == ".json" or _whole_file_is_json(path):
            return "json"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix in (".tsv", ".tab"):
        return "tsv"
    if suffix == ".log":
        return "log"
    if _best_structured(lines[:SAMPLE_LINES])[1] >= GOOD_PARSE:
        return "log"
    delimiter = sniff_delimiter(lines)
    if delimiter:
        return "tsv" if delimiter == "\t" else "csv"
    return "log_or_text"


def _looks_like_json_lines(lines: list[str]) -> bool:
    sample = lines[:20]
    if len(sample) < 2:
        return False
    good = 0
    for line in sample:
        try:
            good += isinstance(json.loads(line), dict)
        except ValueError:
            pass
    return good >= max(2, 0.8 * len(sample))


def _whole_file_is_json(path: Path) -> bool:
    try:
        json.loads(_decode(path.read_bytes())[0])
    except ValueError:
        return False
    return True


def sniff_delimiter(lines: list[str]) -> str | None:
    """The delimiter that splits the first lines into the same number (>= 2) of fields, if one does."""
    sample = lines[:50]
    if len(sample) < 2:
        return None
    candidates: list[str] = []
    try:
        candidates.append(csv.Sniffer().sniff("\n".join(sample), delimiters=DELIMITERS).delimiter)
    except csv.Error:
        pass
    candidates += [d for d in DELIMITERS if d not in candidates]
    for delimiter in candidates:
        counts = [len(row) for row in csv.reader(sample, delimiter=delimiter)]
        modal, hits = Counter(counts).most_common(1)[0]
        if modal >= 2 and counts[0] == modal and hits >= 0.9 * len(counts):
            return delimiter
    return None


# ---------------------------------------------------------------------- tabular


def _read_delimited(path: Path, head: bytes, info: dict[str, Any], client: Any, max_rows: int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    default = "\t" if info["format"] == "tsv" else ","
    sep = sniff_delimiter(_head_lines(head)) or default
    info["format"] = "tsv" if sep == "\t" else "csv"
    if suffix not in (".csv", ".tsv", ".tab"):
        info["notes"].append(f"The {suffix + ' ' if suffix else ''}file holds {DELIMITER_NAMES[sep]}-separated values; read it as a table.")
    elif sep != default:
        info["notes"].append(f"The values are separated by a {DELIMITER_NAMES[sep]}, not a {DELIMITER_NAMES[default]}.")
    frame = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            frame = _read_csv(path, sep, encoding, max_rows, info)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise StructureError("The file has no columns. Check that it is a table with a header row.") from None
        except (pd.errors.ParserError, ValueError):
            raise StructureError("The file could not be read as a table (for example a quote that is never closed). "
                                 "Open it in a spreadsheet, save it as CSV again and upload that.") from None
    if frame is None:  # latin-1 decodes any byte, so this is only a safety net
        raise StructureError("The file's text encoding could not be read. Save it as UTF-8 CSV and try again.")
    if encoding == "latin-1":
        info["notes"].append("The file is not UTF-8; it was read as Latin-1, so check accented characters.")
    header = next(csv.reader(_head_lines(head)[:1], delimiter=sep), [])
    if len(header) == frame.shape[1] and len({h.strip() for h in header}) < len(header):
        frame.columns = unique_names(header)  # pandas would call a repeated header "a.1"; we say "a_2"
    return frame


def _read_csv(path: Path, sep: str, encoding: str, max_rows: int, info: dict[str, Any]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False, nrows=max_rows + 1)
    except pd.errors.ParserError:
        bad: list[list[str]] = []

        def skip(fields: list[str]) -> None:  # returning None tells pandas to drop the line
            bad.append(fields)

        frame = pd.read_csv(path, sep=sep, encoding=encoding, engine="python", on_bad_lines=skip, nrows=max_rows + 1)
        total = len(frame) + len(bad)
        info["parse_rate"] = len(frame) / total if total else 0.0
        info["notes"].append(f"Skipped {len(bad)} of {total:,} rows that had more fields than the header.")
        return frame


def _read_excel(path: Path, head: bytes, info: dict[str, Any], client: Any, max_rows: int) -> pd.DataFrame:
    try:
        book = pd.ExcelFile(path)
    except ImportError:
        raise StructureError("Reading Excel files needs a package that is not installed here (openpyxl for .xlsx, "
                             "xlrd for .xls). Save the sheet as CSV and upload that.") from None
    except Exception:  # noqa: BLE001 - the Excel engines raise many types; the file is simply not a workbook
        raise StructureError("The file could not be opened as an Excel workbook. Save it as .xlsx or CSV and try again.") from None
    with book:
        sheets = list(book.sheet_names)
        frame, used = pd.DataFrame(), None
        for name in sheets:
            try:
                frame = book.parse(name, nrows=max_rows + 1)
            except Exception:  # noqa: BLE001 - a damaged sheet; the message says what to do
                raise StructureError(f"The sheet '{name}' could not be read. Save the workbook as CSV and upload that.") from None
            if not frame.dropna(how="all").empty:
                used = name
                break
    if used is not None and len(sheets) > 1:
        info["notes"].append(f"Read the sheet '{used}', the first of {len(sheets)} sheets that holds data; "
                             "upload other sheets as separate files.")
    return frame


def _read_parquet(path: Path, head: bytes, info: dict[str, Any], client: Any, max_rows: int) -> pd.DataFrame:
    try:
        frame = load_table(path) if path.suffix.lower() == ".parquet" else pd.read_parquet(path)
    except ImportError:
        raise StructureError("Reading Parquet files needs pyarrow, which is not installed here. Upload a CSV instead.") from None
    except Exception:  # noqa: BLE001 - pyarrow raises many types; the file is simply unreadable
        raise StructureError("The file could not be read as Parquet. Export it again or upload a CSV instead.") from None
    return _settle_cells(frame.head(max_rows + 1), info)


# ---------------------------------------------------------------------- JSON


def _read_json(path: Path, head: bytes, info: dict[str, Any], client: Any, max_rows: int) -> pd.DataFrame:
    text = _read_text(path, info)
    try:
        data = json.loads(text)
    except ValueError as error:
        if _looks_like_json_lines(_nonempty_lines(text)):
            info["format"] = "jsonl"
            return _jsonl_frame(_nonempty_lines(text), info, max_rows)
        where = f" (line {error.lineno})" if isinstance(error, json.JSONDecodeError) else ""
        raise StructureError(f"The file is not valid JSON{where}. Check the export, or save the data as CSV or JSON lines.") from None
    if isinstance(data, list):
        info["format"] = "json_records"
        if not data:
            raise StructureError("The JSON file holds an empty list, so there are no rows.")
        dicts = [item for item in data if isinstance(item, dict)]
        if not dicts:
            if all(isinstance(item, list) for item in data):
                return pd.DataFrame(data[: max_rows + 1])
            return pd.DataFrame({"value": data[: max_rows + 1]})
        info["parse_rate"] = len(dicts) / len(data)
        if len(dicts) < len(data):
            info["notes"].append(f"Skipped {len(data) - len(dicts)} of {len(data):,} list items that were not objects.")
        return _records_frame(dicts[: max_rows + 1], info)
    if isinstance(data, dict):
        info["format"] = "json_nested"
        records, where = _find_records(data)
        if records is not None:
            others = [] if where == "(top level)" else [k for k in data if str(k) != where.split(".")[0]]
            info["notes"].append(f"Used the {len(records):,} records under `{where}` as rows"
                                 + (f"; the other top-level fields ({', '.join(map(str, others[:5]))}) were left out." if others else "."))
            return _records_frame(records[: max_rows + 1], info)
        columns = list(data.values())
        if columns and all(isinstance(v, list) for v in columns) and len({len(v) for v in columns}) == 1:
            info["notes"].append("The JSON object holds one list per column; read it as columns.")
            return _settle_cells(pd.DataFrame(data).head(max_rows + 1), info)
        info["notes"].append("The JSON holds a single object with no list of records; it became one row.")
        return _records_frame([data], info)
    raise StructureError("The JSON file holds a single value, not records. Upload a list of objects or a table.")


def _find_records(root: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str]:
    """The largest list of objects in the tree (lists win ties), and where it was found.

    Lists are not entered: one record's inner list (one customer's orders) is not the table.
    An object whose values are all objects with shared keys counts as records keyed by ``key``.
    """
    best: list[Any] = [None, "", -1, False]  # records, where, size, is_list

    def consider(records: list[dict[str, Any]], where: str, is_list: bool) -> None:
        if (len(records), is_list) > (best[2], best[3]):
            best[:] = [records, where or "(top level)", len(records), is_list]

    def walk(node: Any, where: str, depth: int) -> None:
        if depth > 20:
            return
        if isinstance(node, list):
            dicts = [item for item in node if isinstance(item, dict)]
            if dicts and len(dicts) >= 0.9 * len(node):
                consider(dicts, where, True)
            return
        if isinstance(node, dict):
            values = list(node.values())
            if len(values) >= 3 and all(isinstance(v, dict) for v in values):
                first = set(values[0])
                if all(first & set(v) for v in values):
                    consider([{"key": k, **v} for k, v in node.items()], where, False)
            for key, value in node.items():
                walk(value, f"{where}.{key}" if where else str(key), depth + 1)

    walk(root, "", 0)
    return best[0], best[1]


def _records_frame(records: list[dict[str, Any]], info: dict[str, Any]) -> pd.DataFrame:
    frame = pd.json_normalize(records, sep=".", max_level=MAX_NESTING)
    depth = max((str(c).count(".") for c in frame.columns), default=0)
    if depth:
        info["notes"].append(f"Flattened {depth} level{'s' if depth > 1 else ''} of nesting into {frame.shape[1]} columns.")
    return _settle_cells(frame, info)


def _settle_cells(frame: pd.DataFrame, info: dict[str, Any]) -> pd.DataFrame:
    """Lists and objects inside cells, made flat (see the module docstring)."""
    joined, counted, as_json = [], [], []
    names: list[str] = []
    values: list[pd.Series] = []
    for position in range(frame.shape[1]):
        name, series = str(frame.columns[position]), frame.iloc[:, position]
        if series.dtype == object:
            kinds = series.map(_cell_kind)
            if (kinds == "nested_list").any():
                name, series = f"{name}.count", series.map(_item_count)
                counted.append(name)
            else:
                if (kinds == "list").any():
                    series = series.map(_join_list)
                    joined.append(name)
                if (kinds == "dict").any():
                    series = series.map(lambda v: json.dumps(v, default=str) if isinstance(v, dict) else v)
                    as_json.append(name)
        names.append(name)
        values.append(series)
    if joined:
        info["notes"].append(f"Joined lists of values with '|' in {_names(joined)}.")
    if counted:
        info["notes"].append(f"Kept the number of items in {_names(counted)} (lists of objects); "
                             "upload those items as their own file to analyze them.")
    if as_json:
        info["notes"].append(f"Kept nested objects as JSON text in {_names(as_json)} (flattening stops at {MAX_NESTING} levels).")
    out = pd.DataFrame({i: series for i, series in enumerate(values)}, index=frame.index)
    out.columns = unique_names(names)
    return out


def _cell_kind(value: Any) -> str | None:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return "nested_list" if any(isinstance(v, (dict, list, tuple, np.ndarray)) for v in value) else "list"
    if isinstance(value, dict):
        return "dict"
    return None


def _item_count(value: Any) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(len(value))
    if isinstance(value, dict):
        return 1.0
    return np.nan


def _join_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return "|".join("" if v is None else str(v) for v in value) if value else None
    return value


def _names(columns: list[str], limit: int = 4) -> str:
    shown = ", ".join(f"`{c}`" for c in columns[:limit])
    return shown + (f" and {len(columns) - limit} more" if len(columns) > limit else "")


def _read_jsonl(path: Path, head: bytes, info: dict[str, Any], client: Any, max_rows: int) -> pd.DataFrame:
    return _jsonl_frame(_nonempty_lines(_read_text(path, info)), info, max_rows)


def _jsonl_frame(lines: list[str], info: dict[str, Any], max_rows: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    bad = total = 0
    for line in lines:
        if len(records) > max_rows:
            break
        total += 1
        try:
            value = json.loads(line)
        except ValueError:
            bad += 1
            continue
        records.append(value if isinstance(value, dict) else {"value": value})
    if not records or bad > 0.5 * total:
        raise StructureError(f"{bad} of {total:,} lines are not valid JSON. Check that each line holds one JSON object, "
                             "or upload the data as CSV.")
    info["parse_rate"] = len(records) / total
    if bad:
        info["notes"].append(f"{bad} of {total:,} lines were not valid JSON and were skipped.")
    return _records_frame(records, info)


# ---------------------------------------------------------------------- logs

TS = (r"(?:\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:\s?(?:Z\b|[+-]\d{2}:?\d{2}\b))?)?"
      r"|\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
      r"|\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}(?: [+-]\d{4})?"
      r"|[A-Za-z]{3} [ \d]\d \d{2}:\d{2}:\d{2}"
      r"|\d{1,2}/\d{1,2}/\d{4},? \d{1,2}:\d{2}(?::\d{2})?(?: ?[AaPp][Mm])?)")
LEVEL = r"(?:TRACE|DEBUG|INFO|NOTICE|WARNING|WARN|ERROR|ERR|CRITICAL|CRIT|FATAL|SEVERE|ALERT|EMERGENCY|EMERG)"
TS_START = re.compile(rf"^\[?(?P<timestamp>{TS})\]?")
LEAD_LEVEL = re.compile(rf"^[\s,|:-]*\[?(?P<level>{LEVEL})\b\]?", re.I)
LEVEL_BRACKET = re.compile(rf"\[(?P<level>{LEVEL})\]", re.I)
LEVEL_BARE = re.compile(rf"\b(?P<level>{LEVEL})\b")  # upper case only: "error" in prose is not a level
LEVEL_KEY = re.compile(r"\blevel=\"?(?P<level>[A-Za-z]+)", re.I)
GENERIC = re.compile(rf"^\[?(?P<timestamp>{TS})\]?[\s,|:-]*\[?(?P<level>{LEVEL})\b\]?[\s,|:-]*(?P<message>.*)$", re.I)
COMBINED = re.compile(
    r'^(?P<ip>\S+) (?P<ident>\S+) (?P<user>\S+) \[(?P<time>[^\]]+)\] "(?P<request>(?:[^"\\]|\\.)*)" '
    r'(?P<status>\d{3}|-) (?P<bytes>\d+|-)(?: "(?P<referrer>(?:[^"\\]|\\.)*)" "(?P<user_agent>(?:[^"\\]|\\.)*)")?(?P<extra>.*)$')
REQUEST = re.compile(r"^(?P<method>[A-Z]+) (?P<path>\S+)(?: (?P<protocol>[A-Z]+/[\d.]+))?$")
ACCESS_TIME = "%d/%b/%Y:%H:%M:%S %z"
KV_PAIR = re.compile(r"""(?:(?<=[\s,;\[])|^)(?P<key>[A-Za-z_][\w.\-]*)=(?P<value>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^\s,;]*)""")

Row = dict[str, Any] | None


def _parse_combined(line: str) -> Row:
    match = COMBINED.match(line)
    if not match:
        return None
    row = match.groupdict()
    request = REQUEST.match(row["request"] or "")
    row.update(request.groupdict() if request else {"method": None, "path": None, "protocol": None})
    return row


def _parse_keyvalue(line: str) -> Row:
    pairs = list(KV_PAIR.finditer(line))
    if len(pairs) < 2:
        return None
    values: dict[str, Any] = {}
    for pair in pairs:
        values.setdefault(pair["key"], _unquote(pair["value"]))
    pieces, position = [], 0
    for pair in pairs:
        pieces.append(line[position:pair.start()])
        position = pair.end()
    pieces.append(line[position:])
    lead, timestamp, level = pieces[0].strip(), None, None
    stamp = TS_START.match(lead)
    if stamp:
        timestamp, lead = stamp["timestamp"], lead[stamp.end():]
    word = LEAD_LEVEL.match(lead)
    if word:
        level, lead = word["level"].upper(), lead[word.end():]
    message = " ".join(" ".join(p.split()) for p in [lead, *pieces[1:]] if p.strip()).strip(" -|:")
    return {"__timestamp": timestamp, "__level": level, "__message": message or None, "__pairs": values}


def _unquote(value: str) -> str | None:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return re.sub(r"\\(.)", r"\1", value[1:-1])
    return value if value != "" else None


def _parse_generic(line: str) -> Row:
    match = GENERIC.match(line)
    if not match:
        return None
    return {"timestamp": match["timestamp"], "level": match["level"].upper(), "message": match["message"].strip() or None}


def _parse_fallback(line: str) -> tuple[dict[str, Any], bool]:
    row: dict[str, Any] = {"line": line}
    stamp = TS_START.match(line)
    if stamp:
        row["timestamp"] = stamp["timestamp"]
    level = LEVEL_BRACKET.search(line) or LEVEL_BARE.search(line) or LEVEL_KEY.search(line)
    if level:
        row["level"] = level["level"].upper()
    return row, bool(stamp or level)


STRUCTURED: list[tuple[str, Callable[[str], Row], str]] = [
    ("log_combined", _parse_combined, "the access-log pattern"),
    ("log_keyvalue", _parse_keyvalue, "the key=value pattern"),
    ("log_lines", _parse_generic, "the 'timestamp level message' pattern"),
]


def _share(parse: Callable[[str], Row], lines: list[str]) -> float:
    return sum(parse(line) is not None for line in lines) / len(lines) if lines else 0.0


def _best_structured(lines: list[str]) -> tuple[str, float]:
    """The structured parser that reads most lines; earlier (more specific) parsers win ties."""
    best = ("", 0.0)
    for name, parse, _ in STRUCTURED:
        rate = _share(parse, lines)
        if rate > best[1]:
            best = (name, rate)
    return best


def _read_lines(path: Path, head: bytes, info: dict[str, Any], client: Any, max_rows: int) -> pd.DataFrame:
    """Logs and plain text. Picks the parser that reads most of the first 200 lines.

    The structured patterns (access log, key=value, "timestamp level message") compete; one is
    used when it reads at least half of the sample. Otherwise a ``.log`` file (or a file where
    most lines carry a timestamp or level) becomes one row per line, and anything else is read
    as prose, one row per paragraph. The per-line fallback is not a pattern: it never stops a
    model from proposing a better one.
    """
    is_log = info["format"] == "log"
    lines = _nonempty_lines(_read_text(path, info))
    if not lines:
        raise StructureError("The file holds no text. Check that the export finished and upload it again.")
    info["sample_lines"] = _samples(lines)
    if len(lines) > max_rows:
        info["notes"].append(f"The file has {len(lines):,} lines; kept the first {max_rows:,}.")
        lines = lines[:max_rows]
    sample = lines[:SAMPLE_LINES]
    best_name, best_rate = _best_structured(sample)
    fallback_rate = sum(_parse_fallback(line)[1] for line in sample) / len(sample)
    if client is not None and best_rate < GOOD_PARSE:
        proposal = _ask_model(client, lines, best_rate, info)
        if proposal is not None:
            return _model_frame(lines, *proposal, best_rate, info)
    if best_rate >= LOG_MIN_PARSE:
        return _structured_frame(best_name, lines, info)
    if is_log or fallback_rate >= LOG_MIN_PARSE:
        return _fallback_frame(lines, info)
    return _text_frame(path, info)


def _assemble(lines: list[str], rows: list[Row], info: dict[str, Any], description: str) -> pd.DataFrame:
    parsed = sum(row is not None for row in rows)
    info["parse_rate"] = parsed / len(lines)
    if parsed < len(lines):
        info["notes"].append(f"{parsed:,} of {len(lines):,} lines matched {description}; "
                             f"{len(lines) - parsed:,} kept as unparsed in `unparsed_line`.")
    else:
        info["notes"].append(f"All {len(lines):,} lines matched {description}.")
    frame = pd.DataFrame([row if row is not None else {"unparsed_line": line} for row, line in zip(rows, lines)])
    if "unparsed_line" in frame:
        frame = frame[[c for c in frame.columns if c != "unparsed_line"] + ["unparsed_line"]]
    return frame


def _structured_frame(name: str, lines: list[str], info: dict[str, Any]) -> pd.DataFrame:
    parse, description = next((p, d) for n, p, d in STRUCTURED if n == name)
    info["format"] = name
    rows = [parse(line) for line in lines]
    stamp = "timestamp"
    if name == "log_keyvalue":
        rows, stamp = _keyvalue_rows(rows)
    frame = _assemble(lines, rows, info, description)
    if name == "log_combined":
        return _type_access_log(frame, info)
    if stamp in frame:  # the line's own leading timestamp; values of keys stay text for ``clean``
        frame[stamp] = _times_or_text(frame[stamp], stamp, info)
    return frame


def _times_or_text(values: pd.Series, name: str, info: dict[str, Any]) -> pd.Series:
    """Parsed times when nearly all read; otherwise the text as written (syslog times carry no year)."""
    parsed = parse_times(values)
    if parsed.notna().sum() >= 0.95 * values.notna().sum():
        return parsed
    info["notes"].append(f"Kept `{name}` as text: its values do not read as full dates (a year may be missing).")
    return values


def _keyvalue_rows(rows: list[Row]) -> tuple[list[Row], str]:
    """Key=value rows with the leading timestamp, level and leftover text under names no key uses."""
    keys = {k for row in rows if row for k in row["__pairs"]}
    names = {part: (part if part not in keys else f"line_{part}") for part in ("timestamp", "level", "message")}
    out: list[Row] = []
    for row in rows:
        if row is None:
            out.append(None)
            continue
        lead = {names[p]: row[f"__{p}"] for p in ("timestamp", "level") if row[f"__{p}"] is not None}
        tail = {names["message"]: row["__message"]} if row["__message"] is not None else {}
        out.append({**lead, **row["__pairs"], **tail})
    return out, names["timestamp"]


def _type_access_log(frame: pd.DataFrame, info: dict[str, Any]) -> pd.DataFrame:
    for column in ("ident", "user", "referrer", "user_agent"):
        if column in frame:
            frame[column] = frame[column].where(frame[column] != "-")
    frame["time"] = parse_times(frame["time"], ACCESS_TIME)
    frame["status"] = _as_number(frame["status"].where(frame["status"] != "-"))
    sent = frame["bytes"].where(frame["bytes"] != "-", "0").where(frame["bytes"].notna())  # "-" means no body was sent
    frame["bytes"] = _as_number(sent)
    if "extra" in frame:
        extra = frame["extra"].str.strip()
        frame["extra"] = extra.where(extra.fillna("") != "")
    keep_request = frame["method"].isna() & frame["request"].notna()
    drop = ["ident"] + ([] if keep_request.any() else ["request"])
    drop += [c for c in ("user", "referrer", "user_agent", "protocol", "extra") if c in frame and frame[c].isna().all()]
    order = ["ip", "user", "time", "method", "path", "protocol", "status", "bytes", "referrer", "user_agent", "request", "extra", "unparsed_line"]
    return frame[[c for c in order if c in frame and c not in drop]]


def _fallback_frame(lines: list[str], info: dict[str, Any]) -> pd.DataFrame:
    info["format"] = "log_lines"
    results = [_parse_fallback(line) for line in lines]
    frame = pd.DataFrame([row for row, _ in results])
    found = sum(ok for _, ok in results)
    info["parse_rate"] = found / len(lines)
    stamps = int(frame["timestamp"].notna().sum()) if "timestamp" in frame else 0
    levels = int(frame["level"].notna().sum()) if "level" in frame else 0
    info["notes"].append(f"No known log pattern fits most lines, so each line became one row in `line`; "
                         f"a timestamp was found in {stamps:,} and a level in {levels:,} of {len(lines):,} lines.")
    if "timestamp" in frame:
        frame["timestamp"] = _times_or_text(frame["timestamp"], "timestamp", info)
    return frame


def _text_frame(path: Path, info: dict[str, Any]) -> pd.DataFrame:
    text = _decode(path.read_bytes())[0].strip()
    if re.search(r"\n[ \t]*\n", text):
        chunks, unit = re.split(r"\n[ \t]*\n+", text), "paragraph"
    else:
        chunks, unit = text.splitlines(), "line"
    texts = [" ".join(chunk.split()) for chunk in chunks if chunk.strip()]
    info["format"], info["parse_rate"] = "text", 1.0
    info["notes"].append(f"Read the file as plain text: each {unit} became a row ({len(texts):,} rows), "
                         "with its length in characters and words.")
    return pd.DataFrame({"text": texts, "chars": [len(t) for t in texts], "words": [len(t.split()) for t in texts]})


# ---------------------------------------------------------------------- model-proposed pattern

MODEL_PROMPT = (
    "You write one Python regular expression that splits log lines into fields. Reply with JSON only, no prose: "
    '{"regex": "...", "types": {"<group>": "int|float|datetime|str"}}. Use named groups (?P<name>...) with short '
    "snake_case names, at least two of them. The pattern is applied with re.fullmatch, so it must cover the whole line."
)
MODEL_TYPES = ("int", "float", "datetime", "str")


def _ask_model(client: Any, lines: list[str], baseline: float, info: dict[str, Any]) -> tuple[re.Pattern[str], dict[str, str], float] | None:
    """One request for a pattern; code validates it and returns it only when it is better than the built-ins."""
    shown = [line[:MODEL_LINE_CHARS] for line in lines[:MODEL_LINES]]
    messages = [{"role": "system", "content": MODEL_PROMPT},
                {"role": "user", "content": "Lines:\n" + "\n".join(shown)}]
    # The one place raw content can reach a model: say so in the record the user sees (the pipeline lists these notes).
    info["model_lines_sent"] = len(shown)
    info["notes"].append(f"No built-in reader fits this file, so {len(shown)} of its lines (up to {MODEL_LINE_CHARS} characters each) "
                         "were sent to the configured model to work out the format.")
    try:
        reply = client.complete(messages, tools=None, max_tokens=600)
        content = reply.get("content", "") if isinstance(reply, dict) else ""
    except Exception as error:  # noqa: BLE001 - any provider failure means: use the built-in parser
        info["notes"].append(f"The model could not be asked for a pattern ({type(error).__name__}); used the built-in parser.")
        return None
    spec = _json_object(content)
    if not spec or not isinstance(spec.get("regex"), str):
        info["notes"].append("The model's answer held no usable pattern; used the built-in parser.")
        return None
    try:
        pattern = re.compile(spec["regex"])
    except Exception:  # noqa: BLE001 - re.error, RecursionError, OverflowError
        info["notes"].append("The model's pattern is not a valid regular expression; used the built-in parser.")
        return None
    if len(pattern.groupindex) < 2:
        info["notes"].append("The model's pattern names fewer than two fields; used the built-in parser.")
        return None
    sample = lines[:SAMPLE_LINES]
    rate = _share(lambda line: _model_row(pattern, line), sample)
    if rate < GOOD_PARSE or rate <= baseline:
        info["notes"].append(f"The model's pattern read only {rate:.0%} of the first {len(sample)} lines; used the built-in parser.")
        return None
    raw_types = spec.get("types") if isinstance(spec.get("types"), dict) else {}
    types = {str(k): str(v) for k, v in raw_types.items() if k in pattern.groupindex and v in MODEL_TYPES}
    return pattern, types, rate


def _json_object(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    for candidate in (text, *re.findall(r"\{.*\}", text, flags=re.S)):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _model_row(pattern: re.Pattern[str], line: str) -> Row:
    if len(line) > MODEL_MAX_LINE:
        return None
    match = pattern.fullmatch(line)
    if not match:
        return None
    row = {k: (v if v != "" else None) for k, v in match.groupdict().items()}
    return row if sum(v is not None for v in row.values()) >= 2 else None


def _model_frame(lines: list[str], pattern: re.Pattern[str], types: dict[str, str], rate: float, baseline: float,
                 info: dict[str, Any]) -> pd.DataFrame:
    info["format"], info["parser"] = "log_lines", "model"
    fields = list(pattern.groupindex)
    info["notes"].append(f"The built-in patterns read {baseline:.0%} of the first lines, so a model proposed one with the "
                         f"fields {', '.join(fields)}; it read {rate:.0%} of the first {min(len(lines), SAMPLE_LINES)} lines "
                         "and was checked before use.")
    rows = [_model_row(pattern, line) for line in lines]
    frame = _assemble(lines, rows, info, "the proposed pattern")
    for name, kind in types.items():
        if name not in frame or kind == "str":
            continue
        present = frame[name].notna()
        converted = parse_times(frame[name]) if kind == "datetime" else pd.to_numeric(frame[name], errors="coerce")
        if present.any() and converted[present].notna().mean() >= 0.95:
            frame[name] = _as_number(converted) if kind == "int" else converted
        else:
            info["notes"].append(f"Kept `{name}` as text: fewer than 95% of its values read as {kind}.")
    return frame


_READERS: dict[str, Callable[[Path, bytes, dict[str, Any], Any, int], pd.DataFrame]] = {
    "csv": _read_delimited, "tsv": _read_delimited, "excel": _read_excel, "parquet": _read_parquet,
    "json": _read_json, "jsonl": _read_jsonl, "log": _read_lines, "log_or_text": _read_lines,
}
