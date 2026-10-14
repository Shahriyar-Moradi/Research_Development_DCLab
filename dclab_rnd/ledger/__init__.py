"""This studio's decision ledger (package 19.7): every decision it makes, where it makes it, in ``dclab_core.ledger``'s
shape, so decision models of any kind can be trained on them (Part 11) and the main product reads the same records.

Writing. ``decide(point, subject, state, answers, value_used, used)`` where a rule, a model role or a person decides;
the ids come from the context (the workspace and its ledger, the project, draft, run and conversation the session
record binds), as for ``dclab_rnd.experience``. Best effort: a record the contract refuses (a state carrying a value, an
answer outside its point) is reported on stderr and dropped; the decision itself stands. Inside ``experience.quiet()``
(a replay's scratch copy) nothing is written. ``label(record, value, source, by)`` adds what was right.

Where. PostgreSQL ``decision_record`` and ``decision_label`` (append-only) when ``DCLAB_DATABASE_URL`` is set;
otherwise ``ledger/decisions/YYYY/MM/DD.jsonl`` and ``ledger/labels/YYYY/MM/DD.jsonl`` in the workspace folder.

The state of a column is built by :func:`column_state` from the profile the studio already keeps, in Jev's bands: a
model trained on it answers the main product's questions on the same inputs. :data:`CAPTURED` names where each point
is decided; :data:`NOT_DECIDED` the points this studio does not decide, and why (a test keeps the two complete).
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import re
import secrets
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from dclab_core import ledger as contract
except ImportError:  # an environment without the solving core's dependencies: nothing is recorded
    contract = None  # type: ignore[assignment]

DROPPED = {"count": 0}
_LOCK = threading.Lock()
_BATCH: "contextvars.ContextVar[Optional[list]]" = None  # set below: the records of an open batch, written together

# where each point is decided in this studio (19.7); the test runs a project from Home to its report and checks them
CAPTURED = {
    "column.is_identifier": "studio/solution.py and draft/api.py: a saved solution, the proposal's identifiers against the person's",
    "feature.leakage_suspect": "the same save: the audit's flags against the columns the person forbade",
    "target.column": "the same save: the proposed outcome against the saved one",
    "task.framing": "the same save: the detected task against the saved one",
    "column.time": "the same save: the proposed time column against the saved one",
    "column.group": "the same save: the proposed entity column against the saved one",
    "metric.primary": "studio/engine.py prepare: the person's metric or the kind's default",
    "column.semantic_role": "studio/engine.py prepare: each column's role",
    "column.category_code": "studio/engine.py prepare: a numeric column read as codes",
    "split.strategy": "studio/engine.py prepare: the kind's split for the solution",
    "split.folds": "studio/engine.py prepare: the folds from the settings",
    "data.sampling": "studio/engine.py prepare: which rows the runs read",
    "features.recipe": "studio/engine.py stage_features: the recipe the rule kept",
    "model.family": "studio/engine.py stage_models: the family the rule kept",
    "tuning.accept": "studio/engine.py stage_final: whether the tuned configuration beat the default by its margin",
    "threshold.operating_point": "studio/engine.py stage_final: the operating point from the budget or costs",
    "holdout.reuse": "studio/graph.py: a final stage run again, with or without a person's reason",
    "workflow.stage_approval": "studio/engine.py approve: the rule's choice and the person's",
    "workflow.move": "studio/graph.py check: every move, allowed or refused",
    "labels.join_key": "studio/labels.py review: the key that joins the label file",
    "data.cases_table": "draft/pipeline.py: a table read as a list of known cases",
    "data.cleaning_step": "draft/clean.py: each cleaning step applied",
    "problem.family": "draft/api.py build: the card's family",
    "problem.ready": "draft/api.py build: whether the card is ready",
    "problem.pack": "draft/chat.py: the pack detected or chosen",
    "workflow.next_question": "draft/chat.py next_question: the question asked next, or none",
    "workflow.template": "draft/workflow.py: the solution workflow drawn",
    "agent.tool_choice": "agents/runtime.py: each tool a model chose among those it was offered",
    "workflow.stop": "agents/runtime.py: why an agent's run ended",
    "report.claim": "studio/solution_report.py: the report's lines kept and dropped",
    "lesson.accept": "lessons.py review: a reviewer's verdict on a lesson",
    "model.review": "studio/engine.py stage_models: the critic's rounds kept or dropped",
    "experiment.accept": "studio/engine.py stage_models: each review round's experiment kept or dropped",
    "experiment.next": "studio/engine.py stage_models: each review round's experiment, the rules' first and the critic's",
    "workflow.revisit": "studio/graph.py: a completed stage run again",
}
NOT_DECIDED = {
    "column.missing_value_action": "no choice is made: each model family imputes its own fixed way (expansion/runner.py)",
    "labels.situation": "decided with the card's family (problem.family) from the same answer; not a separate decision here",
    "improve.next_action": "no step proposes what to improve next yet (Part 9's workflow, 20.6)",
    "model.promote": "nothing is promoted: a research result is never production (DCLAB-R22)",
    "command.intent_route": "two word patterns, no decision with options (draft/chat.py)",
    "proposal.completeness": "the main product's point: proposals are not scored here",
    "ops.diagnose": "no model is in use here", "ops.retrain": "no model is in use here",
    "ops.release": "no model is in use here", "ops.rollback": "no model is in use here",
}


_BATCH = contextvars.ContextVar("dclab_ledger_batch", default=None)


@contextlib.contextmanager
def batch():
    """Records decided inside the block are written together when it ends (one transaction on PostgreSQL): a saved
    solution's hundreds of column decisions cost one write, not one each. A nested block joins the outer one."""
    if _BATCH.get() is not None:
        yield
        return
    pending: list = []
    token = _BATCH.set(pending)
    try:
        yield
    finally:
        _BATCH.reset(token)
        by_store: dict[int, tuple[Any, list]] = {}
        for store, kind, doc in pending:
            by_store.setdefault(id(store), (store, []))[1].append((kind, doc))
        for store, items in by_store.values():
            try:
                store.add_many([d for k, d in items if k == "record"], [d for k, d in items if k == "label"])
            except Exception as exc:  # noqa: BLE001 - never stop a person's work because the ledger is briefly away
                DROPPED["count"] += len(items)
                print(f"ledger: could not record {len(items)} decisions: {type(exc).__name__}", file=sys.stderr, flush=True)


def _write(store: Any, kind: str, doc: Any) -> None:
    pending = _BATCH.get()
    if pending is not None:
        pending.append((store, kind, doc))
    elif kind == "record":
        store.add(doc)
    else:
        store.add_label(doc)


# ---------------------------------------------------------------------- the stores

def _day(at: str) -> str:
    return datetime.fromisoformat(at).astimezone(timezone.utc).strftime("%Y/%m/%d")


from ..filelock import held  # noqa: E402  (19.10: appends and a deletion's rewrite never cross)


class FileLedger:
    """``ledger/decisions/YYYY/MM/DD.jsonl`` and ``ledger/labels/…``: one line per record, only ever appended."""

    def __init__(self, root: Path, workspace: str):
        self.root, self.workspace = Path(root), workspace

    def _append(self, kind: str, doc: Any) -> None:
        path = self.root / kind / f"{_day(doc.at)}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, held(self.root), path.open("a", encoding="utf-8") as out:
            out.write(doc.model_dump_json() + "\n")

    def add(self, record: Any) -> None:
        self._append("decisions", record)

    def add_label(self, label: Any) -> None:
        self._append("labels", label)

    def add_many(self, records: list, labels: list) -> None:
        for kind, docs in (("decisions", records), ("labels", labels)):
            by_day: dict[str, list] = {}
            for doc in docs:
                by_day.setdefault(_day(doc.at), []).append(doc)
            for day, items in by_day.items():
                path = self.root / kind / f"{day}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with _LOCK, held(self.root), path.open("a", encoding="utf-8") as out:
                    out.write("".join(d.model_dump_json() + "\n" for d in items))

    def _read(self, kind: str, model: Any) -> list[Any]:
        out = []
        for path in sorted((self.root / kind).glob("*/*/*.jsonl")) if (self.root / kind).exists() else []:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    out.append(model.read(line))
                except Exception:  # noqa: BLE001 - a line cut by a crash mid-write: the rest stands
                    continue
        return out

    def records(self, *, project: str | None = None, draft: str | None = None, point: str | None = None,
                run: str | None = None) -> list[Any]:
        return [r for r in self._read("decisions", contract.DecisionRecord)
                if (project is None or r.project == project) and (draft is None or r.draft == draft)
                and (point is None or r.point == point) and (run is None or r.run == run)]

    def labels(self, *, record: str | None = None, point: str | None = None) -> list[Any]:
        return [x for x in self._read("labels", contract.DecisionLabel)
                if (record is None or x.record == record) and (point is None or x.point == point)]


class PgLedger:
    def __init__(self, workspace: str, engine: Any = None, url: str | None = None):
        from ..storage import db

        self.workspace, self.engine = workspace, engine if engine is not None else db.engine(url)

    def _record_row(self, record: Any) -> dict[str, Any]:
        return {"id": record.id, "workspace_id": self.workspace, "at": record.at, "point": record.point, "kind": record.kind,
                "used": record.used, "project_id": record.project, "draft_id": record.draft, "run_id": record.run,
                "conversation": record.conversation, "doc": json.loads(record.model_dump_json())}

    def _label_row(self, label: Any) -> dict[str, Any]:
        return {"id": label.id, "workspace_id": self.workspace, "at": label.at, "record_id": label.record, "point": label.point,
                "source": label.source, "supersedes": label.supersedes, "doc": json.loads(label.model_dump_json())}

    def add(self, record: Any) -> None:
        self.add_many([record], [])

    def add_label(self, label: Any) -> None:
        self.add_many([], [label])

    def add_many(self, records: list, labels: list) -> None:
        import sqlalchemy as sa

        from ..storage.models import decision_label, decision_record

        with self.engine.begin() as c:  # one transaction: the records first, then the labels that name them
            for start in range(0, len(records), 1000):
                c.execute(sa.insert(decision_record), [self._record_row(r) for r in records[start:start + 1000]])
            for start in range(0, len(labels), 1000):
                c.execute(sa.insert(decision_label), [self._label_row(x) for x in labels[start:start + 1000]])

    def records(self, *, project: str | None = None, draft: str | None = None, point: str | None = None,
                run: str | None = None) -> list[Any]:
        import sqlalchemy as sa

        from ..storage.models import decision_record as t

        q = sa.select(t.c.doc).where(t.c.workspace_id == self.workspace)
        for column, value in ((t.c.project_id, project), (t.c.draft_id, draft), (t.c.point, point), (t.c.run_id, run)):
            if value is not None:
                q = q.where(column == value)
        with self.engine.connect() as c:
            return [contract.DecisionRecord.read(r[0]) for r in c.execute(q.order_by(t.c.at, t.c.id))]

    def labels(self, *, record: str | None = None, point: str | None = None) -> list[Any]:
        import sqlalchemy as sa

        from ..storage.models import decision_label as t

        q = sa.select(t.c.doc).where(t.c.workspace_id == self.workspace)
        for column, value in ((t.c.record_id, record), (t.c.point, point)):
            if value is not None:
                q = q.where(column == value)
        with self.engine.connect() as c:
            return [contract.DecisionLabel.read(r[0]) for r in c.execute(q.order_by(t.c.at, t.c.id))]


def open_ledger(home: Path, workspace_id: str | None = None) -> Any:
    from ..storage import db

    if workspace_id and db.database_url(required=False):
        return PgLedger(workspace_id)
    from ..experience import file_workspace_id

    return FileLedger(Path(home) / "ledger", workspace_id or file_workspace_id(home))


_INSTALLED: dict[str, Any] = {}


def install(ledger: Any) -> None:
    _INSTALLED["ledger"] = ledger


def current_ledger() -> Any:
    from .. import context

    services = context.services()
    return getattr(services, "ledger", None) if services is not None else _INSTALLED.get("ledger")


# ---------------------------------------------------------------------- writing

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _ids(ids: dict[str, Any]) -> dict[str, Any]:
    from ..experience import _IDS

    bound = _IDS.get()
    out = {"project": bound.get("project"), "run": bound.get("run"), "conversation": bound.get("conversation")}
    out.update({k: (str(v) if v is not None else None) for k, v in ids.items()})
    return out


def answer(by: str, answerer: str, value: Any, **extra: Any) -> Any:
    """One answer, for ``decide``; None when the contract cannot load (the agent environment)."""
    if contract is None:
        return None
    return contract.Answer(by=by, answerer=str(answerer)[:120], value=value, **{k: v for k, v in extra.items() if v is not None})


def decide(point: str, subject: tuple[str, str], *, answers: Iterable[Any], value_used: Any = None, used: str = "rule",
           state: dict[str, Any] | None = None, options: Iterable[Any] = (), level: int = 0, ledger: Any = None, **ids: Any) -> Optional[Any]:
    """Record one decision in the current workspace's ledger; the record, or None when nothing is kept or it was refused."""
    from ..experience import _QUIET

    target = ledger if ledger is not None else current_ledger()
    if target is None or contract is None or _QUIET.get():
        return None
    try:
        record = contract.DecisionRecord(
            id="dr" + secrets.token_hex(10), point=point, subject=contract.Subject(kind=subject[0], key=str(subject[1])[:200]),
            state={k: v for k, v in (state or {}).items() if v is not None}, options=[str(o)[:200] for o in options][:64],
            answers=[a for a in answers if a is not None], value_used=value_used, used=used, level=level, workspace=target.workspace,
            at=_now(), **{k: v for k, v in _ids(ids).items() if k in ("project", "draft", "run", "conversation", "dataset", "experiment") and v is not None})
    except Exception as exc:  # noqa: BLE001 - the contract refused it: the decision stands, its record is reported
        DROPPED["count"] += 1
        print(f"ledger: {point} refused: {str(exc).splitlines()[1] if len(str(exc).splitlines()) > 1 else exc}"[:300], file=sys.stderr, flush=True)
        return None
    try:
        _write(target, "record", record)
    except Exception as exc:  # noqa: BLE001 - never stop a person's work because the ledger is briefly away
        DROPPED["count"] += 1
        print(f"ledger: could not record {point}: {type(exc).__name__}", file=sys.stderr, flush=True)
        return None
    return record


def label(record: Any, value: Any, source: str, by: str, *, evidence: dict[str, Any] | None = None, supersedes: str | None = None,
          weight: float | None = None, ledger: Any = None) -> Optional[Any]:
    """What was right for ``record``, from one source (19.8 writes them from every source)."""
    from ..experience import _QUIET

    target = ledger if ledger is not None else current_ledger()
    if target is None or contract is None or record is None or _QUIET.get():
        return None
    try:
        out = contract.DecisionLabel(id="dl" + secrets.token_hex(10), record=record.id, point=record.point, value=value, source=source,
                                     by=str(by)[:120], at=_now(), supersedes=supersedes, weight=weight,
                                     evidence=contract.Evidence(**evidence) if evidence else None)
        _write(target, "label", out)
        return out
    except Exception as exc:  # noqa: BLE001
        DROPPED["count"] += 1
        print(f"ledger: label for {record.point} not kept: {type(exc).__name__}", file=sys.stderr, flush=True)
        return None


# ---------------------------------------------------------------------- states, in Jev's bands

DTYPES = {"numeric": "float", "integer": "integer", "float": "float", "categorical": "categorical", "text": "string", "datetime": "datetime",
          "boolean": "boolean", "id": "string", "string": "string"}


from dclab_core.bands import cardinality_band, name_tokens, ratio_band, rows_band  # noqa: E402,F401  (19.11: one banding)


def column_state(column: dict[str, Any], rows: int | None = None) -> dict[str, Any]:
    """A profiled column (``studio.data.profile_table``'s entry) as a state: its name, tokens, dtype and bands. No value."""
    name = str(column.get("name") or "")
    unique = column.get("unique")
    missing = column.get("missing_rate")
    ratio = column.get("unique_ratio")
    return {"column": name[:200] or None, "name_tokens": name_tokens(name) or None, "dtype": DTYPES.get(str(column.get("kind") or ""), "other"),
            "uniqueness": ratio_band(ratio) if isinstance(ratio, (int, float)) else None,
            "nulls": ratio_band(missing) if isinstance(missing, (int, float)) else None,
            "cardinality": cardinality_band(int(unique)) if isinstance(unique, (int, float)) else None}


