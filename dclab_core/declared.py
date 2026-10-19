"""Truths from a public table's own declarations (package 29.15): what a dataset's uploader said of its columns and its
outcome, read as the answer to a decision, and where a declaration is no truth.

A public catalog (OpenML first) declares, for each dataset, a row identifier, an outcome and each column's type. Those
are the uploader's words, not checked: they are a real table's labels where the invented sets have planted ones.

- ``identifier``: in a dataset that declares a row identifier, that column is one and every other declared column is
  not; a dataset that declares none says nothing of its columns (``NO_IDENTIFIER``: on OpenML most keys are left as
  features undeclared). An entity's key, which DCLab counts as an identifier, is labelled not one: a known limit.
- ``role``: the row identifier, else the declared type (numeric; nominal as a category code; string as free text; date).
- ``task``: the declared outcome's type with the number of values the data holds: numeric is regression, nominal binary
  with two values and multiclass with more. A numeric outcome with two values or fewer, or a nominal one with fewer than
  two, gives no truth: the declaration and the data disagree (``DISAGREE``); another type names no task (``NO_TASK``).
  A numeric outcome with a few whole values (a rating, a count) is regression, as declared.
"""

from __future__ import annotations

from typing import Optional

ROLE = {"numeric": "numeric", "nominal": "categorical_code", "string": "free_text", "date": "datetime"}
DISAGREE = "an outcome whose declaration and data disagree"
NO_TASK = "an outcome whose declared type names no task"
NO_IDENTIFIER = "a column of a dataset that declares no row identifier (its identifier truth is not said)"
IGNORED = "an ignored column (its role is not said)"
NO_ROLE = "a declared type with no role"


def identifier(is_row_identifier: bool) -> bool:
    """Whether a declared column is an identifier: only the one the dataset declares as its row identifier."""
    return bool(is_row_identifier)


def role(declared_type: Optional[str], is_row_identifier: bool = False) -> Optional[str]:
    """A declared column's role, or None when its type names none (``NO_ROLE``)."""
    return "identifier" if is_row_identifier else ROLE.get(str(declared_type or ""))


def task(declared_type: Optional[str], values: int) -> Optional[str]:
    """The outcome's task from its declared type and the number of distinct values in the data, or None (``DISAGREE``)."""
    values = int(values or 0)
    if declared_type == "numeric":
        return "regression" if values > 2 else None
    if declared_type == "nominal":
        return "binary" if values == 2 else "multiclass" if values > 2 else None
    return None


def why_no_task(declared_type: Optional[str]) -> str:
    """Why an outcome gives no task truth: its type names none, or its declaration and data disagree."""
    return DISAGREE if declared_type in ("numeric", "nominal") else NO_TASK


__all__ = ["ROLE", "DISAGREE", "NO_TASK", "NO_IDENTIFIER", "IGNORED", "NO_ROLE", "identifier", "role", "task", "why_no_task"]
