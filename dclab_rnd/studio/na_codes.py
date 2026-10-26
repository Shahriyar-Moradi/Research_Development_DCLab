""""NA" as a code, or as a missing value (the owner, 2026-10-25).

pandas (and R) read the text ``NA`` as a missing value. In a column of two-letter codes it is a code: Namibia's country
code, among ZA, KE and BW. Every table DCLab reads from text (an upload, a project's file, the code and notebook it
exports) is read with ``NA`` kept as text (``READ_NA``: pandas' own missing markers but that one), then
:func:`na_as_read` decides per column: a code in a column of two-letter codes, a missing value everywhere else, with a
column whose other values are all numbers read as numbers, as pandas would have read it. The cleaner keeps the same
rule (``draft/clean.py``).

The two functions are self-contained (only pandas and numpy, imported inside), so the exports embed their source as it
is (``SOURCE``): an exported pipeline reads its data exactly as the studio did, and reproduces its numbers.
"""

from __future__ import annotations

import inspect
from typing import Any

# pandas' default missing markers (pandas._libs.parsers.STR_NA_VALUES), "NA" left out: na_as_read decides it per column
READ_NA = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NULL", "NaN",
           "None", "n/a", "nan", "null"]


def code_column(values, share=0.9, least=3):  # no annotations: exported code embeds it as it is
    """Whether a column holds two-letter codes (countries, languages, regions): of its present text values other than
    "NA" itself, at least ``share`` are two capital letters, with at least ``least`` different ones. There "NA" is a code
    (Namibia's), not a missing value. Counted over the distinct values (a column's ``value_counts``), not cell by cell."""
    import re
    from collections import Counter

    try:
        pairs = [(k, int(n)) for k, n in values.value_counts(dropna=True).items()]
    except AttributeError:  # a plain list
        pairs = list(Counter(v for v in values if v is not None).items())
    markers = ("", "NA", "N/A", "n/a", "null", "NULL", "None", "none", "-", "?", "nan", "NaN", "<NA>", "#N/A")
    text = [(k.strip(), n) for k, n in pairs if isinstance(k, str) and k.strip() not in markers]
    total = sum(n for _, n in text)
    if not total:
        return False
    codes = [(k, n) for k, n in text if re.fullmatch(r"[A-Z]{2}", k)]
    return sum(n for _, n in codes) >= share * total and len({k for k, _ in codes}) >= least


def na_as_read(frame):
    """A table read with "NA" kept as text (``keep_default_na=False, na_values=READ_NA``), with "NA" decided per column:
    kept in a column of two-letter codes (``code_column``), a missing value everywhere else, the column then read as
    pandas would have read it (numbers, true/false, dates from a spreadsheet; numbers too large for 64 bits stay text).
    Returns the table and the columns where "NA" was kept as a code."""
    import numpy as np
    import pandas as pd

    kept = []
    for column in list(frame.columns):
        series = frame[column]
        if not (series.dtype == object or isinstance(series.dtype, pd.StringDtype)):
            continue
        hits = series.eq("NA").fillna(False).astype(bool)
        if not hits.any():
            continue
        if code_column(series):
            kept.append(str(column))
            continue
        series = series.astype(object).mask(hits, np.nan)
        inferred = series.infer_objects()  # text stays object on pandas 2 and becomes pandas 3's string type: still undecided
        if inferred.dtype != object and not isinstance(inferred.dtype, pd.StringDtype):  # a spreadsheet's dates or numbers
            frame[column] = inferred
            continue
        distinct = pd.Series(pd.unique(series.dropna()), dtype=object)  # judged on the distinct values, never cell by cell
        numbers = pd.to_numeric(distinct, errors="coerce")
        if bool(numbers.notna().all()) and not bool((numbers.abs() >= 2.0 ** 64).any()):  # every value a number (or none left)
            frame[column] = pd.to_numeric(series, errors="coerce")
            continue
        words = {str(v).strip().lower() for v in distinct}
        if words and words <= {"true", "false"}:
            frame[column] = series.map(lambda v: v if not isinstance(v, str) else v.strip().lower() == "true")
        else:
            frame[column] = inferred  # text, typed as pandas types text: object on pandas 2, its string type on pandas 3
    return frame, kept


def codes_kept(frame: Any) -> list[str]:
    """The columns of a table already read where "NA" is a code: what a service or a note needs, from the table itself."""
    import pandas as pd

    return [str(c) for c in frame.columns if (frame[c].dtype == object or isinstance(frame[c].dtype, pd.StringDtype))
            and bool(frame[c].eq("NA").fillna(False).any()) and code_column(frame[c])]


def kept_note(kept: list[str]) -> str:
    names = ", ".join(f"`{c}`" for c in kept[:5]) + (f" and {len(kept) - 5} more" if len(kept) > 5 else "")
    return (f"Read 'NA' as a code, not as a missing value, in {names}: a column of two-letter codes (as Namibia's country code "
            "is NA). If 'NA' means missing there, say so. Everywhere else 'NA' is read as missing.")


# what an exported pipeline or notebook defines before it reads a table (the same code, embedded)
SOURCE = f"READ_NA = {READ_NA!r}\n\n\n" + inspect.getsource(code_column) + "\n\n" + inspect.getsource(na_as_read)
