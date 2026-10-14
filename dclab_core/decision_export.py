"""Decision datasets in each model family's format (package 22.5), from one builder.

The same decision rows (``datasets.decisions`` or the generated sets of 22.1: a state of bands and name tokens, the
options, the label and its source, the group and the split) are written three ways, row for row:

- :func:`tabular`: one column per state field (name tokens joined), the label, its source, the group and the split,
  for classical models and AutoML (a host writes it as parquet).
- :func:`chat`: exactly what a model is asked when it answers (``request``: the system prompt
  ``prompts/decision_answer.md`` and the question, the options and the state as JSON) as the user's turn, and the answer
  as the assistant's (``{"messages": [...], "metadata": {...}}``), the shape Qwen, Llama, Mistral, Phi and Laya
  fine-tuning tools take; :func:`card` writes its Hugging Face dataset card. A model tuned on it is asked as it learned.
- :func:`typed`: the question, the state, the primitive (yes/no/unknown, a choice, a score) and the answer, the shape of
  a hosted judgment model's typed questions (the main product's Jev purposes).

A row of the sealed partition is never written here: a fine-tuned model must not have seen it, and the comparison reads
it once from where it was generated. Nothing in a state is a table's value (bands and the column's name tokens).

Pure: no dependency beyond the standard library and this package.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from dclab_core.ledger import POINTS, STATE_FIELDS

PURPOSE = "decision_answer"  # the prompt a model answering a decision is given (dclab_core/prompts), in training and when asked
NAMES = ("column", "target")  # whole names: a table carries their tokens only



def _usable(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Rows with a label that are not sealed: what a model may be trained or tuned on."""
    return [r for r in rows if r.get("split") != "sealed" and (r.get("label") or {}).get("value") is not None]


def _text(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def tabular(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One flat row per decision: every state field the ledger knows as a column (name tokens joined), the label."""
    out = []
    for r in _usable(rows):
        state = r.get("state") or {}
        flat = {f"state_{k}": (" ".join(map(str, state[k])) if isinstance(state.get(k), (list, tuple)) else state.get(k))
                for k in STATE_FIELDS if k not in NAMES}  # a whole name never: its tokens are there
        label = r["label"]
        out.append({"record": r.get("record"), "point": r.get("point"), **flat, "options": "|".join(map(str, r.get("options") or [])),
                    "label": _text(label["value"]), "label_source": label.get("source"), "group": r.get("group"), "split": r.get("split")})
    return out


def request(point: str, state: Mapping[str, Any], options: Iterable[Any]) -> dict[str, Any]:
    """What a model answering a decision is sent (beside its system prompt): the point, its question, the state without
    empty fields, and the options as text. The training conversations hold the same request."""
    return {"point": point, "question": POINTS[point].about, "state": {k: v for k, v in state.items() if v is not None},
            "options": [str(o) for o in options]}


def chat(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One conversation per decision: the request a model is sent, and the label as its answer (no reason: a reason
    learned from invented tables would be said about real ones)."""
    from dclab_core import prompts

    system = prompts.text(PURPOSE)
    out = []
    for r in _usable(rows):
        label = r["label"]
        answer = json.dumps({"answer": _text(label["value"])}, ensure_ascii=False)
        out.append({"messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": json.dumps(request(r["point"], r.get("state") or {}, r.get("options") or []), ensure_ascii=False)},
                                 {"role": "assistant", "content": answer}],
                    "metadata": {"record": r.get("record"), "point": r.get("point"), "split": r.get("split"), "group": r.get("group"),
                                 "label_source": label.get("source")}})
    return out


def typed(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One typed question per decision: the question, the state, the primitive and the answer."""
    out = []
    for r in _usable(rows):
        p = POINTS[r["point"]]
        out.append({"record": r.get("record"), "purpose": p.jev or p.key, "question": p.about, "primitive": p.answer,
                    "state": dict(r.get("state") or {}), "options": list(r.get("options") or []), "answer": _text(r["label"]["value"]),
                    "label_source": r["label"].get("source"), "split": r.get("split"), "group": r.get("group")})
    return out


def summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    usable = _usable(rows)
    return {"rows": len(usable), "by_point": dict(Counter(r["point"] for r in usable)), "by_split": dict(Counter(r.get("split") for r in usable)),
            "by_label_source": dict(Counter((r.get("label") or {}).get("source") for r in usable)),
            "groups": len({r.get("group") for r in usable})}


def card(name: str, rows: Iterable[Mapping[str, Any]], source: str, limits: Iterable[str] = ()) -> str:
    """The Hugging Face dataset card (README.md with its YAML header) for the chat export."""
    s = summary(rows)
    header = "\n".join(["---", f"pretty_name: {name}", "license: other", "language:", "- en", "task_categories:", "- text-generation",
                        "tags:", "- decision-making", "- data-science", "- synthetic", "configs:", "- config_name: default", "  data_files:",
                        "  - split: train", "    path: train.chat.jsonl", "  - split: validation", "    path: val.chat.jsonl", "---"])
    lines = [f"# {name}", "", f"Decisions DCLab makes while building a model, each with its label: {source}.", "",
             f"- {s['rows']} decisions from {s['groups']} groups (a group, one table or project, never spans two splits)",
             "- by point: " + ", ".join(f"{k} {v}" for k, v in sorted(s["by_point"].items())),
             "- by split: " + ", ".join(f"{k} {v}" for k, v in sorted(s["by_split"].items())),
             "- labels by source: " + ", ".join(f"{k} {v}" for k, v in sorted(s["by_label_source"].items())), "",
             "Each row's user turn holds the question, what is known about the column as bands (never a value of the table) and "
             "the options; the assistant's turn the answer as JSON. The sealed partition is not here: it is read once, to compare "
             "a tuned model with the rules and the classical classifiers on the same gate.", "", "## Limits", ""]
    lines += [f"- {x}" for x in limits] or ["- None stated."]
    return header + "\n\n" + "\n".join(lines) + "\n"
