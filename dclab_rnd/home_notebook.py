"""A conversation's saved answers as a notebook (package H7, part c).

Each saved answer becomes two cells: what it computed (the plan's own description), then its plan run again by the same
engine (``dclab_core.query``) on the same cleaned table, with the numbers checked against the ones the chat showed. The
notebook holds plans and the shown numbers only, never a row; it reads ``clean.parquet`` beside it (the export puts the
person's own cleaned table there) and needs nothing but pandas and dclab-core.

``check`` runs the notebook's code the way a person would, so a test proves every saved answer reproduces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HEADER = '''\
import pandas as pd
from dclab_core import query as Q

table = pd.read_parquet("clean.parquet")  # the cleaned table the answers were computed from


def shown(v):
    """A number as the chat wrote it: two decimals at most, thousands separated; a withheld value says so."""
    if v is None:
        return "withheld (too few rows)"
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{v:,}" if isinstance(v, int) else str(v)


def numbers(result):
    return [shown(v) for row in result.rows for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)]
'''

ANSWER = '''\
plan = Q.QueryPlan.read({plan!r})
result = Q.execute(plan, table, floor={floor})
assert numbers(result) == {shown!r}, "this answer differs from the one the chat gave"
pd.DataFrame(result.rows, columns=result.columns)
'''


def saved(draft: dict[str, Any]) -> list[dict[str, Any]]:
    """The answers the person saved from the table now in use, in the order they were given (an answer from an earlier
    table cannot be computed again from this one)."""
    asset = draft.get("active_asset")
    return [a for a in draft.get("answers") or [] if a.get("saved") and (asset is None or a.get("asset") in (None, asset))]


def build(draft: dict[str, Any]) -> dict[str, Any]:
    """The notebook of a draft's saved answers (nbformat 4)."""
    from dclab_core import query as Q

    from .home_query import describe

    def cell(kind: str, source: str) -> dict[str, Any]:
        out = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        return {**out, "execution_count": None, "outputs": []} if kind == "code" else out

    cells = [cell("markdown", "# Answers from your table\n\nEach answer below is computed again from `clean.parquet` by the same "
                              "engine, and checked against the numbers the chat showed. Numbers per group only: a group with too "
                              "few rows is never shown."),
             cell("code", HEADER)]
    others = [a for a in draft.get("answers") or [] if a.get("saved") and a not in saved(draft)]
    if others:
        cells.append(cell("markdown", f"Saved answers from an earlier table: {len(others)}. They are not here, because that table "
                                      "is not in this export."))
    for answer in saved(draft):
        plan = Q.QueryPlan.model_validate(answer["plan"])
        cells += [cell("markdown", f"## {describe(plan)}"),
                  cell("code", ANSWER.format(plan=json.dumps(answer["plan"], sort_keys=True), floor=int(answer.get("floor") or Q.FLOOR),
                                             shown=list(answer.get("shown") or [])))]
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}}


def archive(store: Any, draft: dict[str, Any]) -> bytes:
    """The export: the notebook and the cleaned table it reads (the person's own table, checked against its recorded hash),
    as a zip. Refused (ValueError) when nothing is saved."""
    import io
    import zipfile

    from .home_query import table

    if not saved(draft):
        raise ValueError("No saved answer from this table yet: save an answer first")
    frame = table(store, draft)
    data = io.BytesIO()
    frame.to_parquet(data, index=False)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("answers.ipynb", json.dumps(build(draft), indent=1, ensure_ascii=False))
        z.writestr("clean.parquet", data.getvalue())
        z.writestr("README.md", "Open answers.ipynb next to clean.parquet and run every cell; each answer is computed again and "
                                "checked against the numbers the chat showed. Needs pandas, pyarrow and dclab-core.\n")
    return out.getvalue()


def check(notebook: dict[str, Any], folder: Path) -> int:
    """Run the notebook's code cells in ``folder`` (where clean.parquet is); the number of answers reproduced. A differing
    answer raises AssertionError."""
    import os

    here = os.getcwd()
    scope: dict[str, Any] = {}
    reproduced = 0
    try:
        os.chdir(folder)
        for c in notebook["cells"]:
            if c["cell_type"] == "code":
                exec(compile("".join(c["source"]), "<notebook>", "exec"), scope)  # noqa: S102 - our own generated cells
                reproduced += "assert numbers(result)" in "".join(c["source"])
    finally:
        os.chdir(here)
    return reproduced


__all__ = ["archive", "build", "check", "saved"]
