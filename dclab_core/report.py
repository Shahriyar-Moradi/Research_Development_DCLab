"""The solution report, from the records (package 17.7).

:func:`build` writes a :class:`~dclab_core.contracts.SolutionReport` from what a host recorded: the problem brief, the
data findings, the solution plan and the stage records of a run (plain dicts, as the R&D studio stores them: each with
``evidence``, ``claims``, ``decision``, ``setup_summary``; the final stage's evidence holds ``holdout_metrics``,
``holdout_primary_metric_ci``, ``baselines``, ``task_assessment``, ``precision_note`` and ``limits``). Its sections are
the problem as understood, what the data shows, the decisions and their reasons, the results with their intervals, the
limits (what the data cannot show among them), the next steps and, when a rubric is given, the deliverables map.

**Every number comes from a record.** The prose is written by code, never by a model, and a number enters it only
through the record field it is read from: ``SolutionReport.numbers`` maps each record path (``final.evidence.
holdout_metrics.average_precision``) to the value written, and each section cites the paths its numbers come from.
Text quoted from a record (a finding's statement, a decision's reason, the holdout's description) carries its own
numbers; each is traced to the n-th number of that text (``findings.F-cap.statement@0``), and a number in a mapping's
key (an interval named ``80%``) to the key (``final.evidence.task_assessment.intervals.80%#0``). A line whose numbers
cannot all be traced, or that calls a model production-ready, is left out (the check of ``dclab_rnd/cited.py``,
written here so the core stays pure). :func:`sources` maps each number written to its path; :func:`check` re-reads
every path against the records.

Renderers: :func:`to_markdown` and :func:`to_html` (self-contained, everything escaped) are pure; :func:`to_pdf` needs
reportlab, imported only when called (the ``pdf`` extra), and raises :class:`PdfUnavailable` without it.
"""

from __future__ import annotations

import html
import math
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union

from dclab_core.contracts import DataFindings, ProblemBrief, ReportSection, SolutionPlan, SolutionReport

STAGES = ("data", "leakage", "features", "models", "final")
STAGE_NAMES = {"data": "understand the data", "leakage": "audit leakage", "features": "feature recipes",
               "models": "model screen", "final": "tuning and the holdout"}
# a number as written: a sign only when it stands alone ("-2.4", not the "-05" of a date or the "08" of WF-08 glued to a
# word), digits with one decimal part, and a percent sign. Digits glued to a letter, an underscore or a dot (x1, top_2,
# v1.2) are part of a name, not a number.
NUMBER = re.compile(r"(?<![A-Za-z_\d.])-?\d+(?:\.\d+)?%?")
# what no sentence may say about a model, whatever surrounds it (dclab_rnd/cited.py's rule, kept strict here)
OVERCLAIM = re.compile(r"production[\s-]*read(?:y|iness)|read(?:y|iness)\s+(?:for|to\s+go\s+(?:to|into)|to\s+enter)\s+production"
                       r"|approved\s+for\s+production|fit\s+for\s+production", re.I)
_DASHES = re.compile(r"[‐-―−]")
_SUFFIX = re.compile(r"^(.*?)([@#])(\d+)$")
_STOPS = re.compile(r"([.!?])\.(?=\s|$)")  # "the order.." when a quoted sentence ends a line
MAX_CITES = 50          # ReportSection.cites
MAX_BODY = 18_000       # ReportSection.body is at most 20 000 characters
MAX_SECTIONS = 40
NOTE = "Every number in this report is copied from the run's records; none is recomputed or estimated here."

METRIC_LABEL = {
    "average_precision": "average precision", "roc_auc": "ROC AUC", "recall_at_budget": "recall at the review budget",
    "precision_at_budget": "precision at the review budget", "macro_f1": "macro F1", "weighted_f1": "weighted F1",
    "accuracy": "accuracy", "balanced_accuracy": "balanced accuracy", "mae": "MAE", "rmse": "RMSE", "r2": "R squared",
    "mape": "MAPE", "bias": "mean bias", "f1": "F1", "precision": "precision", "recall": "recall", "brier": "Brier score",
    "log_loss": "log loss", "average_precision_random_ranking": "average precision of a random ranking",
}
OTHER_METRICS = {
    "binary": ("roc_auc", "average_precision", "brier"),
    "multiclass": ("macro_f1", "accuracy", "balanced_accuracy", "weighted_f1"),
    "regression": ("mae", "rmse", "r2", "bias"),
}
FACTS = (("goal", "Goal"), ("action", "What a prediction feeds"), ("unit", "One scored row"), ("task", "Task"),
         ("target_column", "Target column"), ("target_definition", "Target definition"),
         ("label_source", "Where the labels come from"), ("label_coverage", "Label coverage"),
         ("prediction_moment", "Prediction moment"), ("time_column", "Time column"), ("horizon", "Horizon"),
         ("entities", "Entities rows repeat for"), ("primary_metric", "Metric named in the description"))
SEVERITY = {"critical": 0, "warning": 1, "info": 2}


class PdfUnavailable(RuntimeError):
    """The PDF renderer needs reportlab, which is an optional extra of dclab-core."""


# ---------------------------------------------------------------------------------------------------- numbers


def numbers_in(text: str) -> list[tuple[str, float]]:
    """The numbers a text writes, in order, each with its value (a percent is a share: ``6.0%`` is 0.06)."""
    return [(token, _value(token)[0]) for token in NUMBER.findall(text or "")]


def _value(token: str) -> tuple[float, int]:
    percent = token.endswith("%")
    body = token[:-1] if percent else token
    decimals = len(body.split(".", 1)[1]) if "." in body else 0
    value = float(body)
    return (value / 100, decimals + 2) if percent else (value, decimals)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def matches(token: str, value: Any) -> bool:
    """True when ``token`` is ``value`` as written: rounded to the decimals the token shows (a percent to its share)."""
    if not _finite(value):
        return False
    written, decimals = _value(token)
    return abs(written - float(value)) <= 0.5 * 10 ** -decimals + 1e-9 * max(1.0, abs(float(value)))


def fmt(value: float, style: str = "num") -> str:
    """How a record's number is written: ``int``, ``pct`` (a share as a percent), ``signed``, ``num`` (an integer as
    stored, a fraction with three decimals) or ``auto`` (a whole fraction such as a cap of 60.0 as an integer)."""
    whole = isinstance(value, int) or (style == "auto" and float(value).is_integer())
    value = float(value)
    if style == "int":
        return str(int(round(value)))
    if style == "pct":
        share = value * 100
        return (_sig(share, 2) if 0 < abs(share) < 0.1 else f"{share:.1f}") + "%"
    if style == "signed":
        return ("+" if value >= 0 else "") + fmt(value)
    if whole and abs(value) < 1e15:
        return str(int(value))
    if abs(value) >= 100:
        return f"{value:.1f}"
    if 0 < abs(value) < 0.001:
        return _sig(value, 2)
    return f"{value:.3f}"


def _sig(value: float, digits: int) -> str:
    places = max(0, digits - 1 - int(math.floor(math.log10(abs(value))))) if value else 0
    return f"{value:.{places}f}"


def _plain(value: float) -> str:
    """A number inside a rendered value: as stored, never in scientific notation."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    if abs(value) < 1e-4:
        return _sig(value, 3)
    return f"{value:.6f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------------------------------- text


_MD_SPECIAL = re.compile(r"([\\`*|<>])")
_MD_SPECIAL_BUT_CODE = re.compile(r"([\\*|<>])")


def render(value: Any) -> str:
    """A record's value as report text: one line, Markdown's special characters escaped, numbers never in exponent
    form. :func:`lookup` reads the numbers of quoted text through this same function, so the two always agree."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return _plain(float(value)) if math.isfinite(float(value)) else ""
    if isinstance(value, str):
        text = " ".join(unicodedata.normalize("NFKC", value).split())
        # balanced backticks are the record's own code spans (`biker_id`); a lone one is escaped
        return (_MD_SPECIAL_BUT_CODE if text.count("`") % 2 == 0 else _MD_SPECIAL).sub(r"\\\1", text)
    if isinstance(value, Mapping):
        if "column" in value:
            rest = "; ".join(f"{render(k)}: {render(v)}" for k, v in value.items() if k != "column")
            return render(value["column"]) + (f" ({rest})" if rest else "")
        return "; ".join(f"{render(k)}: {render(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(render(v) for v in value)
    return render(str(value))


def _clip(text: str, limit: int) -> str:
    """Cut at a space (never inside a number, so the numbers kept are the text's first ones)."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip("\\") + " …"


def overclaims(text: str) -> bool:
    return bool(OVERCLAIM.search(" ".join(_DASHES.sub("-", unicodedata.normalize("NFKC", text)).split())))


# ---------------------------------------------------------------------------------------------------- records


def _walk(node: Any, parts: list[str]) -> tuple[Any, Optional[str]]:
    """Longest key first, so a key holding a dot (``metric.primary``, a column name) is found whole."""
    if not parts:
        return node, None
    for i in range(len(parts), 0, -1):
        key = ".".join(parts[:i])
        if isinstance(node, Mapping) and key in node:
            child = node[key]
        elif isinstance(node, (list, tuple)) and i == 1 and key.isdigit() and int(key) < len(node):
            child = node[int(key)]
        else:
            continue
        try:
            value, last = _walk(child, parts[i:])
        except KeyError:
            continue
        return value, (last if last is not None else key)
    raise KeyError(".".join(parts))


def lookup(ctx: Mapping[str, Any], path: str) -> Any:
    """The record value at ``path`` in a :func:`context`. ``path@n``: the n-th number written in that value's text;
    ``path#n``: the n-th number in the name of the key the path ends at. Raises KeyError when there is none."""
    match = _SUFFIX.match(path)
    if not match:
        return _walk(ctx, path.split("."))[0]
    base, kind, index = match.group(1), match.group(2), int(match.group(3))
    value, key = _walk(ctx, base.split("."))
    found = numbers_in(render(value) if kind == "@" else render(key))
    if index >= len(found):
        raise KeyError(path)
    return found[index][1]


def _brief(brief: Any) -> tuple[Optional[ProblemBrief], Optional[str]]:
    if brief is None:
        return None, None
    if isinstance(brief, ProblemBrief):
        return brief, None
    try:
        return ProblemBrief.model_validate(dict(brief)), None
    except Exception as exc:  # noqa: BLE001 - a brief that does not read is a stated limit, not a crash
        return None, f"The problem brief given could not be read ({type(exc).__name__}), so it is left out."


def _findings(findings: Any) -> tuple[Optional[DataFindings], Optional[str]]:
    if findings is None:
        return None, None
    if isinstance(findings, DataFindings):
        return findings, None
    try:
        return DataFindings.model_validate(dict(findings)), None
    except Exception as exc:  # noqa: BLE001
        return None, f"The data findings given could not be read ({type(exc).__name__}), so they are left out."


def _plan(plan: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Decisions, risks and what would change the plan, from a SolutionPlan, its JSON, or the compact list a host keeps
    (``[{"decision", "choice", "why"}]``)."""
    if plan is None:
        return [], [], []
    if isinstance(plan, SolutionPlan):
        dumped = plan.model_dump(mode="json")
        return dumped["decisions"], dumped["risks"], dumped["would_change"]
    if isinstance(plan, Mapping):
        try:
            dumped = SolutionPlan.model_validate(dict(plan)).model_dump(mode="json")
            return dumped["decisions"], dumped["risks"], dumped["would_change"]
        except Exception:  # noqa: BLE001 - read leniently below
            items = plan.get("decisions") or []
            risks = [dict(r) for r in plan.get("risks") or [] if isinstance(r, Mapping)]
            would = [str(w) for w in plan.get("would_change") or []]
    elif isinstance(plan, (list, tuple)):
        items, risks, would = plan, [], []
    else:
        return [], [], []
    decisions, seen = [], set()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        key = str(item.get("key") or item.get("decision") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        decisions.append({"key": key, "choice": item.get("choice"), "reason": str(item.get("reason") or item.get("why") or ""),
                          "cites": list(item.get("cites") or []), "alternatives": list(item.get("alternatives") or []),
                          "source": item.get("source"), "status": item.get("status")})
    return decisions, risks, would


def context(*, brief: Any = None, findings: Any = None, plan: Any = None, records: Optional[Mapping[str, Any]] = None,
            rubric: Optional[list[Mapping[str, Any]]] = None, title: Optional[str] = None) -> dict[str, Any]:
    """Everything a report may cite, by the root of its paths: ``brief``, ``findings`` (by id), ``findings_meta``,
    ``plan`` (by decision key), ``plan_risks``, ``plan_would_change``, ``rubric``, ``title`` and each stage record
    (``data``, ``leakage``, ``features``, ``models``, ``final``)."""
    b, _ = _brief(brief)
    f, _ = _findings(findings)
    decisions, risks, would = _plan(plan)
    ctx: dict[str, Any] = {
        "title": title or "", "brief": b.model_dump(mode="json") if b else {},
        "findings": {x.id: x.model_dump(mode="json") for x in f.findings} if f else {},
        "findings_meta": {"training_rows": f.training_rows, "task": f.task, "target": f.target} if f else {},
        "plan": {d["key"]: d for d in decisions}, "plan_risks": risks, "plan_would_change": would,
        "rubric": [dict(r) for r in rubric or [] if isinstance(r, Mapping)],
    }
    for stage, record in (records or {}).items():
        if stage in STAGES and isinstance(record, Mapping):
            ctx[stage] = record
    return ctx


# ---------------------------------------------------------------------------------------------------- the writer


class _Writer:
    """Collects lines into sections. A number enters a line only through :meth:`n` (a record's number), :meth:`q`
    (a record's text, its numbers traced) or :meth:`key` (a key's name); :meth:`line` keeps a line only when every
    number it writes is one of those, and it calls no model production-ready."""

    def __init__(self, ctx: Mapping[str, Any]):
        self.ctx = ctx
        self.numbers: dict[str, float] = {}
        self.sections: list[ReportSection] = []
        self.limits: list[str] = []
        self.dropped = 0
        self._title: Optional[str] = None
        self._lines: list[tuple[str, list[str]]] = []
        self._refs: list[str] = []
        self._pending: list[str] = []

    # reading
    def get(self, path: str, default: Any = None) -> Any:
        try:
            return lookup(self.ctx, path)
        except (KeyError, TypeError, ValueError):
            return default

    def has(self, path: str) -> bool:
        return self.get(path) not in (None, "", [], {})

    def n(self, path: str, style: str = "num") -> Optional[str]:
        value = self.get(path)
        if not _finite(value):
            return None
        self.numbers[path] = float(value)
        self._pending.append(path)
        return fmt(value, style)

    def _trace(self, path: str, text: str, kind: str) -> None:
        for index, (_, value) in enumerate(numbers_in(text)):
            self.numbers[f"{path}{kind}{index}"] = value
            self._pending.append(f"{path}{kind}{index}")

    def q(self, path: str, limit: int = 600, code: bool = False) -> str:
        text = _clip(render(self.get(path)), limit)
        self._trace(path, text, "@")
        return f"`{text}`" if code and text and "`" not in text else text

    def key(self, path: str, code: bool = True) -> str:
        try:
            _, name = _walk(self.ctx, path.split("."))
        except KeyError:
            return ""
        text = render(name)
        self._trace(path, text, "#")
        return f"`{text}`" if code and text and "`" not in text else text

    def ref(self, *names: str) -> None:
        self._refs.extend(n for n in names if n not in self._refs)

    # writing
    def line(self, text: Optional[str], limit: bool = False) -> bool:
        paths, self._pending = self._pending, []
        if not text:
            return False
        text = _STOPS.sub(r"\1", text)
        traced = all(any(matches(token, self.numbers.get(p)) for p in paths) for token in NUMBER.findall(text))
        if not traced or overclaims(text) or len(set(paths)) > MAX_CITES:  # a section cites at most MAX_CITES paths
            self.dropped += 1
            return False
        self._lines.append((text, list(dict.fromkeys(paths))))
        if limit:
            self.limits.append(text[2:] if text.startswith("- ") else text)
        return True

    def section(self, title: str) -> None:
        self._flush()
        self._title, self._lines, self._refs, self._pending = title, [], [], []

    def _flush(self) -> None:
        if self._title is None:
            return
        lines = self._lines or [("Nothing in the records covers this yet.", [])]
        chunks: list[tuple[list[str], list[str]]] = [([], [])]
        table_head: list[str] = []
        for text, paths in lines:
            body, cites = chunks[-1]
            numeric = [p for p in paths if p in self.numbers]
            new = [p for p in numeric if p not in cites]
            size = sum(len(t) + 2 for t in body) + len(text)
            if body and (len(cites) + len(new) > MAX_CITES or size > MAX_BODY):
                chunks.append(([*table_head] if text.startswith("|") else [], []))
                body, cites = chunks[-1]
                new = [p for p in numeric if p not in cites]
            body.append(text)
            cites.extend(new)
            if text.startswith("| ") and not table_head:
                table_head = [text]
            elif text.startswith("|-") and len(table_head) == 1:
                table_head.append(text)
            elif not text.startswith("|"):
                table_head = []
        for index, (body, cites) in enumerate(chunks):
            if len(self.sections) >= MAX_SECTIONS:
                self.dropped += len(body)
                break
            room = MAX_CITES - len(cites)
            refs = [r for r in self._refs if r not in cites][:room] if index == 0 else []
            self.sections.append(ReportSection(title=self._title if index == 0 else f"{self._title} (continued)",
                                               body=_join(body), cites=cites + refs))

    def finish(self, title: str) -> SolutionReport:
        self._flush()
        self._title = None
        cited = {p for s in self.sections for p in s.cites}
        self._pending = []
        self._trace("title", render(title), "@")
        cited |= {p for p in self._pending if p.startswith("title@")}
        numbers = {p: v for p, v in self.numbers.items() if p in cited}
        return SolutionReport(title=title[:300], sections=self.sections, limits=self.limits[:30], numbers=numbers)


def _kind(text: str) -> str:
    return "list" if text.startswith("- ") else ("table" if text.startswith("|") else "para")


def _join(lines: list[str]) -> str:
    out = []
    for i, text in enumerate(lines):
        if i:
            out.append("\n" if _kind(text) == _kind(lines[i - 1]) != "para" else "\n\n")
        out.append(text)
    return "".join(out)


# ---------------------------------------------------------------------------------------------------- sections


def _metric(record_metric: Optional[str]) -> str:
    return METRIC_LABEL.get(record_metric or "", record_metric or "the primary metric")


def _task_of(ctx: Mapping[str, Any]) -> str:
    for stage in reversed(STAGES):
        task = (ctx.get(stage) or {}).get("task")
        if task:
            return str(task)
    return str((ctx.get("findings_meta") or {}).get("task") or (ctx.get("brief", {}).get("task") or {}).get("value") or "unknown")


def _problem(w: _Writer, has_brief: bool, notes: list[str]) -> None:
    w.section("The problem as understood")
    brief = w.ctx.get("brief") or {}
    unknown = []
    if not has_brief:
        w.line("No problem brief was given: the problem is described here only by what the run recorded.")
    else:
        for field, label in FACTS:
            fact = brief.get(field) or {}
            status = fact.get("status", "unknown")
            if status == "unknown":
                unknown.append(label.lower())
                continue
            value = w.q(f"brief.{field}.value", 500)
            if status == "stated":
                how = "stated in the person's own words" if fact.get("source") == "person" else "stated in the description"
            else:
                how = f"inferred: {w.q(f'brief.{field}.reason', 300)}"
            w.line(f"- **{label}**: {value} ({how})")
        for i, constraint in enumerate(brief.get("constraints") or []):
            w.line(f"- **Constraint ({render(constraint.get('kind', 'other')).replace('_', ' ')})**: {w.q(f'brief.constraints.{i}.text', 400)}")
        for i, _ in enumerate(brief.get("success_criteria") or []):
            w.line(f"- **Success criterion**: {w.q(f'brief.success_criteria.{i}.value', 400)}")
        if unknown:
            w.line("Not known yet: " + ", ".join(unknown) + ".")
    stage = next((s for s in reversed(STAGES) if w.ctx.get(s)), None)
    if stage:
        w.ref(stage)
        metric = _metric(w.get(f"{stage}.primary_metric"))
        moment = w.q(f"{stage}.decision_time_rule", 500).rstrip(".")
        w.line(f"The run treated it as a {w.q(f'{stage}.task')} problem scored by {metric}"
               + (f", predicted at this moment: {moment}" if moment else "") + ".")
    for note in notes:
        w.line(note)


def _data(w: _Writer, has_findings: bool) -> None:
    w.section("What the data shows")
    ctx = w.ctx
    if ctx.get("data"):
        w.ref("data")
        e = "data.evidence"
        rows, train, holdout = w.n(f"{e}.source_rows", "int"), w.n(f"{e}.train_rows", "int"), w.n(f"{e}.holdout_rows", "int")
        split = w.q(f"{e}.holdout_split", 400)
        if rows and train and holdout:
            w.line(f"The table had {rows} rows; the run trained on {train} and locked {holdout} as the holdout ({split}).")
        else:
            w.line(None)
        sampled = (ctx["data"].get("evidence") or {}).get("sampling") or {}
        if isinstance(sampled, Mapping) and _finite(sampled.get("rows_used")) and _finite(sampled.get("source_rows")) \
                and sampled["rows_used"] < sampled["source_rows"]:
            used = w.n(f"{e}.sampling.rows_used", "int")
            rule = w.q(f"{e}.sampling.rule", 300)
            w.line(f"The run used {used} of them" + (f" ({rule})" if rule else "") + ", so its numbers describe that sample.")
        if w.has(f"{e}.cv_protocol"):
            w.line(f"Choices were made on the training rows only, with {w.q(f'{e}.cv_protocol', 300)}.")
        task = str(ctx["data"].get("task") or "")
        t = f"{e}.target_summary"
        if task == "binary" and w.has(t):
            rate, positives, majority = w.n(f"{t}.positive_rate_train", "pct"), w.n(f"{t}.positives_train", "int"), w.n(f"{t}.majority_baseline_accuracy_train", "pct")
            w.line(f"Positives are {rate} of the training rows ({positives} rows); predicting the majority class for every row "
                   f"already scores {majority} accuracy, so accuracy alone cannot judge a model." if rate and positives and majority else None)
        elif task == "multiclass" and w.has(t):
            classes, small, large = w.n(f"{t}.classes", "int"), w.n(f"{t}.smallest_class_count", "int"), w.n(f"{t}.largest_class_count", "int")
            w.line(f"The target has {classes} classes; the smallest has {small} training rows and the largest {large}, so the "
                   f"rare classes weigh on a macro metric." if classes and small and large else None)
        elif task == "regression" and w.has(t):
            mean, median, low, high = (w.n(f"{t}.{k}", "auto") for k in ("mean_train", "median_train", "min_train", "max_train"))
            w.line(f"The target has a mean of {mean} and a median of {median} on the training rows, from {low} to {high}."
                   if None not in (mean, median, low, high) else None)
        g = f"{e}.group_stats"
        if w.has(g):
            groups, held, median = w.n(f"{g}.train_groups", "int"), w.n(f"{g}.holdout_groups", "int"), w.n(f"{g}.median_rows_per_group", "auto")
            w.line(f"Rows repeat for the group column: {groups} groups in training and {held} others in the holdout, a median of "
                   f"{median} rows each, so folds keep each group on one side." if groups and held and median else None)
        risky = ctx["data"].get("evidence", {}).get("risk_features") or {}
        if isinstance(risky, Mapping) and risky:
            parts = [f"{w.key(f'{e}.risk_features.{name}')} ({w.q(f'{e}.risk_features.{name}', 120)})" for name in list(risky)[:15]]
            w.line("Columns the profile flagged for review: " + "; ".join(parts) + ".")
    for stage in ("final", "data"):
        if (ctx.get(stage) or {}).get("labels", {}).get("coverage") == "partial":
            source = w.q(f"{stage}.labels.file", 200, code=True) or "a label file"
            provenance = w.q(f"{stage}.labels.provenance", 400)
            w.line(f"The labels come from {source} and are partial" + (f" ({provenance})" if provenance else "")
                   + ": a row without a label may be a case nobody found.")
            break
    if ctx.get("leakage"):
        w.ref("leakage")
        e = "leakage.evidence"
        evidence = ctx["leakage"].get("evidence") or {}
        metric = _metric(evidence.get("apparent_lift_metric"))
        if evidence.get("declared_leakage_features"):
            lift = w.n(f"{e}.apparent_lift", "signed")
            columns = w.q(f"{e}.declared_leakage_features", 400)
            w.line(f"Kept out as written after the prediction moment: {columns}."
                   + (f" Including them would move training-CV {metric} by {lift}, a gain that would not exist when predicting." if lift else ""))
        else:
            w.line("No column was declared as written after the prediction moment.")
        candidates = evidence.get("heuristic_review_candidates") or []
        if candidates:
            parts = [f"{w.q(f'{e}.heuristic_review_candidates.{i}.feature', 100, code=True)} ({w.q(f'{e}.heuristic_review_candidates.{i}.reasons', 160)})"
                     for i, _ in enumerate(candidates[:10])]
            w.line("Flagged by the leakage heuristics for a person's review: " + "; ".join(parts) + ".")
        gap = w.n(f"{e}.random_vs_time_cv.optimism_gap", "signed")
        w.line(f"A random split would score {metric} {gap} against the time-ordered one: it sees the future." if gap else None)
    findings = ctx.get("findings") or {}
    if findings:
        rows = w.n("findings_meta.training_rows", "int")
        w.line(f"The analysis behind the plan read {rows} rows (a seeded share of the table, before the run locked its holdout); "
               "its findings, most severe first:" if rows
               else "The analysis for the plan found, most severe first:")
        ordered = sorted(findings, key=lambda fid: SEVERITY.get(findings[fid].get("severity"), 3))
        for fid in ordered[:20]:
            p = f"findings.{fid}"
            w.ref(p)
            w.line(f"- **{w.q(f'{p}.topic', 80)}** ({findings[fid].get('severity', 'info')}, {w.q(f'{p}.id', 80, code=True)}): "
                   f"{w.q(f'{p}.statement', 500)} Implication: {w.q(f'{p}.implication', 500)}")
    elif not has_findings and not ctx.get("data"):
        w.line("Neither the data findings nor the data stage's record were given: nothing about the data can be reported.")


def _decisions(w: _Writer) -> None:
    w.section("Decisions and reasons")
    ctx = w.ctx
    plan = ctx.get("plan") or {}
    if plan:
        w.line("The solution plan, one decision per point, each with its reason and what was set aside:")
    for key, decision in list(plan.items())[:40]:
        p = f"plan.{key}"
        w.ref(p)
        status = "; ".join(f"{label}: {render(decision[k])}" for k, label in (("status", "status"), ("source", "source")) if decision.get(k))
        cites = ", ".join(w.q(f"{p}.cites.{i}", 80, code=True) for i, _ in enumerate((decision.get("cites") or [])[:6]))
        alternatives = "; ".join(f"{w.q(f'{p}.alternatives.{i}.choice', 160)} ({w.q(f'{p}.alternatives.{i}.why_not', 200)})"
                                 for i, a in enumerate((decision.get("alternatives") or [])[:4]) if isinstance(a, Mapping))
        reason = w.q(f"{p}.reason", 500)
        w.line(f"- **{w.q(f'{p}.key', 80)}**: {w.q(f'{p}.choice', 400) or 'none'}"
               + (f". {reason}" if reason else "") + (f" ({status})" if status else "")
               + (f" Rests on {cites}." if cites else "") + (f" Set aside: {alternatives}." if alternatives else ""))
    for stage, what in (("features", "Feature recipe"), ("models", "Model family")):
        record = ctx.get(stage) or {}
        decision = record.get("decision") or {}
        if not decision.get("chosen"):
            continue
        w.ref(stage)
        chosen = w.q(f"{stage}.decision.chosen", 120, code=True)
        rule = w.q(f"{stage}.decision.rule", 900)
        person = (f" A person chose it over the rule's {w.q(f'{stage}.decision.selected', 120, code=True)}."
                  if decision.get("overridden") else "")
        w.line(f"- **{what}**: {chosen}, by this rule: {rule}{person}")
        if stage == "features":
            e = record.get("evidence") or {}
            if e.get("reference_recipe") and e.get("reference_recipe") != decision.get("chosen"):
                gain = w.n("features.evidence.selected_vs_reference_gain", "signed")
                reference = w.q("features.evidence.reference_recipe", 120, code=True)
                w.line(f"- Against the first recipe, {reference}, it changes training-CV {_metric(record.get('primary_metric'))} by {gain} on the same folds." if gain else None)
        if stage == "models" and w.has("models.evidence.rare_event_choice.why"):
            w.line(f"- Why this family: {w.q('models.evidence.rare_event_choice.why', 600)}.")
    final = ctx.get("final") or {}
    tuning = (final.get("evidence") or {}).get("tuning_decision") or {}
    if tuning:
        w.ref("final")
        t = "final.evidence.tuning_decision"
        gain, required = w.n(f"{t}.gain", "signed"), w.n(f"{t}.required_gain")
        w.line(f"- **Tuning**: {'accepted' if tuning.get('accepted') else 'rejected'}: the best candidate, "
               f"{w.q(f'{t}.best_candidate', 120, code=True)}, changed training-CV {_metric(final.get('primary_metric'))} by {gain} "
               f"against a required {required}; the final model used {w.q('final.evidence.selected_optimization', 120, code=True)}."
               if gain and required else None)
    if not plan and not any((ctx.get(s) or {}).get("decision") for s in ("features", "models")) and not tuning:
        w.line("No decision was recorded: neither a solution plan nor the stages that choose a recipe, a model and its tuning.")


def _results(w: _Writer, notes: list[str]) -> None:
    w.section("Results, with their intervals")
    ctx = w.ctx
    final = ctx.get("final") or {}
    if not final:
        w.line("The holdout has not been scored: no result can be reported yet.")
        notes.append("- No final stage record: the model has not been scored on the holdout, so nothing here measures how well it predicts.")
    else:
        w.ref("final")
        e = "final.evidence"
        evidence = final.get("evidence") or {}
        primary = final.get("primary_metric") or ""
        label = _metric(primary)
        value = w.n(f"{e}.holdout_metrics.{primary}")
        model, recipe = w.q(f"{e}.model", 120, code=True), w.q(f"{e}.feature_recipe", 120, code=True)
        split = w.q(f"{e}.holdout_split", 400)
        w.line(f"The final model, {model} on the {recipe} recipe, scored {label} {value} on the holdout, used once ({split})."
               if value else None)
        low, high = w.n(f"{e}.holdout_primary_metric_ci.low"), w.n(f"{e}.holdout_primary_metric_ci.high")
        method = w.q(f"{e}.holdout_primary_metric_ci.method", 200)
        w.line(f"Its bootstrap interval runs from {low} to {high} ({method}): the range the holdout score could take on "
               f"another draw of the same rows." if low and high else None)
        cv, gap = w.n(f"{e}.cv_selected_metric_mean"), w.n(f"{e}.cv_to_holdout_gap", "signed")
        w.line(f"On the training folds it averaged {cv}; the holdout differs by {gap} (positive means the holdout scored better)."
               if cv and gap else None)
        baselines = evidence.get("baselines") or {}
        for name, values in baselines.items():
            if not isinstance(values, Mapping):
                continue
            parts = []
            for metric_name, v in values.items():
                written = w.n(f"{e}.baselines.{name}.{metric_name}")
                if written:
                    parts.append(f"{METRIC_LABEL.get(metric_name, render(metric_name))} {written}")
            if parts:
                w.line(f"- Baseline {w.key(f'{e}.baselines.{name}')} on the same holdout: " + ", ".join(parts) + ".")
        task = _task_of(ctx)
        kind = "binary" if "binary" in task else ("multiclass" if task == "multiclass" else "regression")
        metrics = evidence.get("holdout_metrics") or {}
        others = [f"{_metric(m)} {w.n(f'{e}.holdout_metrics.{m}')}" for m in OTHER_METRICS[kind] if m != primary and _finite(metrics.get(m))]
        w.line("Other holdout metrics: " + ", ".join(others) + "." if others else None)
        if kind == "binary" and _finite(metrics.get("alert_budget")):
            h = f"{e}.holdout_metrics"
            budget, alerts = w.n(f"{h}.alert_budget", "pct"), w.n(f"{h}.alerts_at_budget", "int")
            recall, precision = w.n(f"{h}.recall_at_budget"), w.n(f"{h}.precision_at_budget")
            r_low, r_high = w.n(f"{h}.recall_at_budget_ci.low"), w.n(f"{h}.recall_at_budget_ci.high")
            w.line(f"At the review budget of {budget} of rows ({alerts} alerts on the holdout), recall is {recall}"
                   + (f" (interval {r_low} to {r_high})" if r_low and r_high else "") + f" and precision {precision}."
                   if budget and alerts and recall and precision else None)
        _assessment(w, kind, evidence.get("task_assessment") or {})
    models = ctx.get("models") or {}
    ranking = (models.get("evidence") or {}).get("ranking") or []
    if ranking:
        w.ref("models")
        w.line(f"The model screen on the training folds ({w.q('models.evidence.cv_protocol', 300)}), mean and fold spread of "
               f"{_metric(models.get('primary_metric'))}:")
        for i, _ in enumerate(ranking[:10]):
            r = f"models.evidence.ranking.{i}"
            mean, std = w.n(f"{r}.mean"), w.n(f"{r}.std")
            w.line(f"- {w.q(f'{r}.model', 80, code=True)}: {mean} ± {std}" if mean and std else None)
    rounds = ((models.get("evidence") or {}).get("review") or {}).get("rounds") or []
    for i, r in enumerate(rounds):
        p = f"models.evidence.review.rounds.{i}"
        base, variant = w.n(f"{p}.base_mean"), w.n(f"{p}.variant_mean")
        w.line(f"- Review round {w.n(f'{p}.round', 'int') or ''}: {w.q(f'{p}.experiment', 80, code=True)} on {w.q(f'{p}.family', 80, code=True)}, "
               + (f"{base} → {variant} on the same folds, " if base and variant else "")
               + f"{'kept' if r.get('adopted') else 'not kept'} ({w.q(f'{p}.why', 400)}).")


def _assessment(w: _Writer, kind: str, assessment: Mapping[str, Any]) -> None:
    if not assessment:
        return
    a = "final.evidence.task_assessment"
    if kind == "binary":
        t = f"{a}.threshold"
        if w.has(t):
            value = w.n(f"{t}.value")
            w.line(f"The alert threshold {value} was chosen on the {w.q(f'{t}.chosen_on', 120)}: {w.q(f'{t}.chosen_by', 300)}." if value else None)
            alerts, precision, recall = w.n(f"{t}.holdout.alerts", "int"), w.n(f"{t}.holdout.precision"), w.n(f"{t}.holdout.recall")
            w.line(f"Rows it flagged on the holdout: {alerts}" + (f", with precision {precision}" if precision else "")
                   + (f" and recall {recall}" if recall else "") + "." if alerts else None)
        c = f"{a}.calibration"
        if w.has(c):
            brier, ece = w.n(f"{c}.brier"), w.n(f"{c}.expected_calibration_error")
            verdict = w.q(f"{c}.verdict", 120)
            w.line(f"Calibration: Brier score {brier}, expected calibration error {ece}" + (f" ({verdict})" if verdict else "") + "."
                   if brier and ece else None)
    elif kind == "multiclass":
        per_class = assessment.get("per_class") or []
        if per_class:
            w.line("Per class on the holdout (precision, recall):")
        for i, row in enumerate(per_class[:15]):
            p = f"{a}.per_class.{i}"
            precision, recall = w.n(f"{p}.precision"), w.n(f"{p}.recall")
            w.line(f"- {w.q(f'{p}.class', 80, code=True)}: {precision}, {recall}" if precision and recall else None)
        for i, _ in enumerate((assessment.get("most_confused") or [])[:5]):
            p = f"{a}.most_confused.{i}"
            rows, share = w.n(f"{p}.rows", "int"), w.n(f"{p}.share_of_true", "pct")
            w.line(f"- Confused: {w.q(f'{p}.true', 80, code=True)} taken for {w.q(f'{p}.predicted', 80, code=True)} on {rows} rows, "
                   f"{share} of its rows." if rows and share else None)
        for i, _ in enumerate((assessment.get("tiny_classes") or [])[:8]):
            p = f"{a}.tiny_classes.{i}"
            if isinstance(w.get(p), Mapping):
                recall = w.n(f"{p}.recall")
                w.line(f"- A tiny class, {w.q(f'{p}.class', 80, code=True)}" + (f", has recall {recall}." if recall else "."))
        top = assessment.get("top_k_accuracy") or {}
        parts = [f"{w.key(f'{a}.top_k_accuracy.{k}')} {w.n(f'{a}.top_k_accuracy.{k}')}" for k, v in top.items() if _finite(v)]
        w.line("Top-k accuracy: " + ", ".join(parts) + "." if parts else None)
        recall = w.n(f"{a}.macro_recall")
        w.line(f"Macro recall over every class: {recall}." if recall else None)
    else:
        for level in list(assessment.get("intervals") or {})[:4]:
            p = f"{a}.intervals.{level}"
            low, high, cover = w.n(f"{p}.from_prediction.0", "signed"), w.n(f"{p}.from_prediction.1", "signed"), w.n(f"{p}.holdout_coverage", "pct")
            w.line(f"- The {w.key(p, code=False)} interval adds from {low} to {high} to a prediction; it covered {cover} of the holdout rows."
                   if low and high and cover else None)
        c = f"{a}.at_cap"
        if w.has(c):
            cap, rows, at, below = w.n(f"{c}.cap", "auto"), w.n(f"{c}.holdout_rows_at_cap", "int"), w.n(f"{c}.mae_at_cap"), w.n(f"{c}.mae_below_cap")
            w.line(f"The target is capped at {cap}: on the {rows} holdout rows at the cap the MAE is {at}, against {below} below it."
                   if cap and rows and at and below else None)
        for column, segments in list((assessment.get("error_by_segment") or {}).items())[:3]:
            if not isinstance(segments, Mapping):
                continue
            p = f"{a}.error_by_segment.{column}"
            parts = []
            for name in list(segments)[:8]:
                mae, rows = w.n(f"{p}.{name}.mae"), w.n(f"{p}.{name}.rows", "int")
                if mae and rows:
                    parts.append(f"{w.key(f'{p}.{name}')} {mae} on {rows} rows")
            w.line(f"- MAE by {w.key(p)}: " + ", ".join(parts) + "." if parts else None)
        grow = w.n(f"{a}.errors_grow_with_prediction.spearman")
        w.line(f"Errors against the size of the prediction: Spearman correlation {grow}." if grow else None)


def _limits(w: _Writer, notes: list[str], partial: bool, unknown: list[str], missing: list[str]) -> None:
    w.section("Limits, and what the data cannot show")
    ctx = w.ctx
    final = ctx.get("final") or {}
    evidence = final.get("evidence") or {}
    for note in notes:
        w.line(note, limit=True)
    if missing:
        w.line(f"- No record for these stages: {', '.join(STAGE_NAMES[s] for s in missing)}; what they would have shown is not here.", limit=True)
    if evidence.get("precision_note"):
        w.line(f"- {w.q('final.evidence.precision_note', 600)}", limit=True)
    elif (evidence.get("task_assessment") or {}).get("precision_note"):
        w.line(f"- {w.q('final.evidence.task_assessment.precision_note', 600)}", limit=True)
    for i, _ in enumerate((evidence.get("limits") or [])[:8]):
        w.line(f"- {w.q(f'final.evidence.limits.{i}', 600)}", limit=True)
    if w.has("final.evidence.task_assessment.at_cap.note"):
        w.line(f"- At the cap: {w.q('final.evidence.task_assessment.at_cap.note', 300)}.", limit=True)
    seen = set()
    for i, claim in enumerate(final.get("claims") or []):
        for j, text in enumerate((claim.get("limitations") or []) if isinstance(claim, Mapping) else []):
            if text in seen or len(seen) >= 4:
                continue
            seen.add(text)
            w.line(f"- {w.q(f'final.claims.{i}.limitations.{j}', 500)}", limit=True)
    uses = w.n("final.evidence.holdout_uses_in_this_project", "int")
    if uses and _finite(evidence.get("holdout_uses_in_this_project")) and evidence["holdout_uses_in_this_project"] > 1:
        w.line(f"- The holdout was scored {uses} times in this project, so it is no longer untouched"
               + (f": {w.q('final.evidence.holdout_reuse_reason', 300)}." if w.has("final.evidence.holdout_reuse_reason") else "."), limit=True)
    else:
        w._pending = []
    if w.has("leakage.evidence.warning"):
        w.line(f"- {w.q('leakage.evidence.warning', 400)}", limit=True)
    for i, _ in enumerate((ctx.get("plan_risks") or [])[:6]):
        w.line(f"- A risk the plan names: {w.q(f'plan_risks.{i}.statement', 400)} To check: {w.q(f'plan_risks.{i}.check', 400)}", limit=True)
    if unknown:
        w.line("- The brief leaves these open, and the run assumed them: " + ", ".join(unknown) + ".", limit=True)
    if partial:
        w.line("- Rows without a label were scored as negatives although some may be cases nobody found: recall and precision "
               "are measured against the known cases only.", limit=True)
    w.line("- The holdout comes from the same data as the training rows: it cannot show how the model does on a later period, "
           "on another source, or after the process that writes the data changes.", limit=True)
    w.line("- Whether a column is known at the prediction moment is a fact about the process, not the data: the run relies on "
           "the moment as it was stated.", limit=True)
    w.line("- A holdout score is evidence about this data, not a reason to rely on the model in use: that needs the people who "
           "own the decision to review it, and monitoring once it runs.", limit=True)


def _next(w: _Writer, partial: bool, missing: list[str]) -> None:
    w.section("Next steps")
    ctx = w.ctx
    if missing:
        w.line(f"- Run the stages that have no record yet ({', '.join(STAGE_NAMES[s] for s in missing)}) before reading any result.")
    review = (((ctx.get("models") or {}).get("evidence") or {}).get("review") or {})
    tried = {r.get("experiment") for r in review.get("rounds") or []}
    for i, experiment in enumerate(((review.get("review") or {}).get("next_experiments") or [])[:6]):
        if not isinstance(experiment, Mapping) or experiment.get("name") in tried:
            continue
        p = f"models.evidence.review.review.next_experiments.{i}"
        w.line(f"- Try {w.q(f'{p}.name', 80, code=True)}: {w.q(f'{p}.expected_gain', 300)} (cost {w.q(f'{p}.cost', 20)}).")
    questions = sorted(enumerate((ctx.get("brief") or {}).get("open_questions") or []), key=lambda iq: iq[1].get("impact", 5))
    for i, _ in questions[:6]:
        w.line(f"- Ask: {w.q(f'brief.open_questions.{i}.text', 400)}")
    for i, _ in enumerate((ctx.get("plan_would_change") or [])[:5]):
        w.line(f"- Revisit the plan if this changes: {w.q(f'plan_would_change.{i}', 400)}")
    if partial:
        w.line("- Have a person review a sample of the highest-scored unlabelled rows, to learn how many are real cases the labels missed.")
    w.line("- Score rows from a later period than the holdout, and watch the inputs for drift, before the model drives decisions.")


# what a report never delivers by itself: a requirement naming one of these is mapped outside it (whole words only)
OUTSIDE = {"api": "an API", "apis": "an API", "fastapi": "an API", "flask": "an API", "endpoint": "an API", "endpoints": "an API",
           "service": "a service", "services": "a service", "docker": "a container", "dockerfile": "a container",
           "container": "a container", "containers": "a container", "deploy": "a deployment", "deployed": "a deployment",
           "deployment": "a deployment", "git": "source control", "github": "source control", "repository": "source control",
           "repo": "source control", "tests": "tests", "pytest": "tests", "unittest": "tests", "notebook": "a notebook",
           "notebooks": "a notebook", "code": "code", "dashboard": "a dashboard"}
RUBRIC_RULES = (
    (("leak",), "Decisions and reasons", "leakage"),
    (("threshold", "budget", "alert", "cost"), "Results, with their intervals", "final"),
    (("model", "classif", "regress", "predict", "algorithm", "train"), "Results, with their intervals", "final"),
    (("metric", "evaluat", "validat", "score", "accuracy", "precision", "recall", "auc", "error", "baseline"), "Results, with their intervals", "final"),
    (("feature",), "Decisions and reasons", "features"),
    (("data", "eda", "explor", "clean", "quality", "missing", "profil", "distribution"), "What the data shows", "data"),
    (("limit", "risk", "assumption", "uncertain"), "Limits, and what the data cannot show", None),
    (("next", "improv", "recommend", "future"), "Next steps", None),
    (("report", "summary", "write", "document", "present", "explain", "problem"), "this report", None),
)


def _where(text: str, ctx: Mapping[str, Any]) -> tuple[str, str]:
    lowered = text.lower()
    words = re.findall(r"[a-z]+", lowered)

    def has(stem: str) -> bool:
        return any(word.startswith(stem) for word in words)

    for word in words:
        if word in OUTSIDE:
            return "outside this report", f"not covered here: it asks for {OUTSIDE[word]}, which a report does not produce"
    for stems, section, stage in RUBRIC_RULES:
        if any(has(s) for s in stems):
            if stage and not ctx.get(stage):
                return section, f"open: the {STAGE_NAMES[stage]} stage has no record"
            return section, "covered"
    return "not found", "open: nothing in the records matches it"


def _deliverables(w: _Writer) -> None:
    rubric = w.ctx.get("rubric") or []
    root = "rubric"
    items = []
    for i, item in enumerate(rubric[:40]):
        name_key = next((k for k in ("requirement", "name", "text", "title") if item.get(k)), None)
        if name_key:
            items.append((f"{root}.{i}", name_key, item))
    if not items:
        deliverables = (w.ctx.get("brief") or {}).get("deliverables") or []
        root = "brief.deliverables"
        items = [(f"{root}.{i}", "name", d) for i, d in enumerate(deliverables[:40]) if d.get("name")]
    if not items:
        return
    w.section("Deliverables map")
    w.line("Each requirement, its points when the rubric gives them, and where this report or the run meets it:")
    w.line("| Requirement | Points | Where | Status |")
    w.line("|---|---|---|---|")
    for path, name_key, item in items:
        requirement = w.q(f"{path}.{name_key}", 300)
        points = w.n(f"{path}.points") or "–"
        if item.get("met_by"):
            where, status = w.q(f"{path}.met_by", 200), "named by the rubric"
        else:
            where, status = _where(str(item.get(name_key)), w.ctx)
        w.line(f"| {requirement} | {points} | {where} | {status} |")


# ---------------------------------------------------------------------------------------------------- build


def build(*, brief: Any = None, findings: Any = None, plan: Any = None, records: Optional[Mapping[str, Any]] = None,
          rubric: Optional[list[Mapping[str, Any]]] = None, title: Optional[str] = None) -> SolutionReport:
    """The solution report of a run. Every input may be missing: the report then says what is missing, and why that
    limits it, rather than failing. ``rubric``: a list of requirements (``{"requirement" | "name": text, "points":
    number, "met_by": optional text}``); without one, the brief's deliverables are mapped when it has any."""
    notes: list[str] = []
    b, brief_note = _brief(brief)
    f, findings_note = _findings(findings)
    notes += [f"- {n}" for n in (brief_note, findings_note) if n]
    if records is not None and not isinstance(records, Mapping):
        notes.append("- The stage records given are not a mapping of stage to record, so they are left out.")
        records = None
    ctx = context(brief=b, findings=f, plan=plan, records=records, rubric=rubric, title=title)
    w = _Writer(ctx)
    missing = [s for s in STAGES if not ctx.get(s)]
    partial = any((ctx.get(s) or {}).get("labels", {}).get("coverage") == "partial" for s in STAGES) or \
        (ctx["plan"].get("labels.coverage") or {}).get("choice") == "partial" or \
        (ctx["brief"].get("label_coverage") or {}).get("value") == "partial"
    unknown = [label.lower() for field, label in FACTS if b is not None and (ctx["brief"].get(field) or {}).get("status", "unknown") == "unknown"
               and field in ("prediction_moment", "label_coverage", "target_column", "unit", "entities")]
    _problem(w, b is not None, [n[2:] for n in notes])
    _data(w, f is not None)
    _decisions(w)
    result_notes: list[str] = []
    _results(w, result_notes)
    _limits(w, result_notes, partial, unknown, missing)
    _next(w, partial, missing)
    _deliverables(w)
    if w.dropped:
        w.section("About this report")
        w.line("Some lines were left out because a number in them could not be traced to a record, or because they "
               "called a model ready for use; the records hold the full detail.")
    name = (title or "").strip() or "Solution report"
    return w.finish(name)


# ---------------------------------------------------------------------------------------------------- tracing


def _tokens_with_paths(report: SolutionReport) -> list[tuple[str, str, list[str]]]:
    """(where, token, candidate paths) for every number the report writes."""
    out = [("title", token, [p for p in report.numbers if p.startswith("title@")]) for token in NUMBER.findall(report.title)]
    for section in report.sections:
        for text in (section.title, section.body):
            out += [(section.title, token, section.cites) for token in NUMBER.findall(text)]
    return out


def sources(report: SolutionReport) -> dict[str, str]:
    """Each number written in the report (as written) -> the record path it was read from."""
    out: dict[str, str] = {}
    for _, token, paths in _tokens_with_paths(report):
        if token in out:
            continue
        path = next((p for p in paths if p in report.numbers and matches(token, report.numbers[p])), None)
        if path is not None:
            out[token] = path
    return out


def check(report: SolutionReport, ctx: Mapping[str, Any]) -> list[str]:
    """Problems, as sentences: a number not traced to a cited path, a path whose record value differs from what the
    report says it read, a section that calls a model production-ready. Empty when the report is sound."""
    problems = []
    for where, token, paths in _tokens_with_paths(report):
        candidates = [p for p in paths if p in report.numbers and matches(token, report.numbers[p])]
        if not candidates:
            problems.append(f"{where}: {token} is not traced to a record")
            continue
        if not any(_finite(_safe(ctx, p)) and abs(float(_safe(ctx, p)) - report.numbers[p]) <= 1e-9 * max(1.0, abs(report.numbers[p]))
                   for p in candidates):
            problems.append(f"{where}: {token} does not match the record at {candidates[0]}")
    for text in [report.title, *(s.body for s in report.sections), *report.limits]:
        if overclaims(text):
            problems.append("a sentence calls a model production-ready")
    return problems


def _safe(ctx: Mapping[str, Any], path: str) -> Any:
    try:
        return lookup(ctx, path)
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------- renderers


def to_markdown(report: SolutionReport) -> str:
    parts = [f"# {report.title}", f"_{NOTE}_"]
    for section in report.sections:
        parts += [f"## {section.title}", section.body]
    return "\n\n".join(parts) + "\n"


def _blocks(body: str) -> list[tuple[str, list[str]]]:
    """A section body as blocks: ("para", [text]), ("list", items), ("table", rows of cells)."""
    blocks: list[tuple[str, list[Any]]] = []
    for raw in body.split("\n"):
        line = raw.rstrip()
        if not line:
            continue
        kind = _kind(line)
        if kind == "list":
            item = line[2:]
            if blocks and blocks[-1][0] == "list":
                blocks[-1][1].append(item)
            else:
                blocks.append(("list", [item]))
        elif kind == "table":
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
            if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
                continue
            if blocks and blocks[-1][0] == "table":
                blocks[-1][1].append(cells)
            else:
                blocks.append(("table", [cells]))
        else:
            blocks.append(("para", [line]))
    return blocks


_CODE = re.compile(r"(?<!\\)`([^`]+?)(?<!\\)`")
_BOLD = re.compile(r"(?<!\\)\*\*(.+?)(?<!\\)\*\*")
_UNESCAPE = re.compile(r"\\([\\`*|<>])")


def _inline_html(text: str) -> str:
    out = html.escape(text, quote=True)
    out = _CODE.sub(lambda m: "<code>" + m.group(1) + "</code>", out)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = re.sub(r"\\(&lt;|&gt;)", r"\1", out)
    return _UNESCAPE.sub(r"\1", out)


_CSS = """
:root{--ink:#1d2433;--muted:#5b6475;--line:#d9dee7;--bg:#ffffff;--soft:#f4f6fa;--accent:#2f5bd3}
@media (prefers-color-scheme: dark){:root{--ink:#e6e9ef;--muted:#a3abba;--line:#343b4a;--bg:#14181f;--soft:#1c212b;--accent:#7f9cf5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:860px;margin:0 auto;padding:32px 16px 64px}h1{font-size:1.7rem;margin:0 0 8px}h2{font-size:1.2rem;margin:32px 0 8px;padding-top:12px;border-top:1px solid var(--line)}
.note{color:var(--muted);font-style:italic}code{background:var(--soft);padding:1px 5px;border-radius:4px;font-size:.92em;word-break:break-word}
ul{padding-left:22px}li{margin:4px 0}table{border-collapse:collapse;width:100%;margin:8px 0;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{background:var(--soft)}
""".strip()


def to_html(report: SolutionReport) -> str:
    """One self-contained page: inline style, no script, every record text escaped."""
    out = ["<!doctype html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           f"<title>{html.escape(report.title)}</title>", f"<style>{_CSS}</style>", "</head>", "<body>", "<main>",
           f"<h1>{html.escape(report.title)}</h1>", f'<p class="note">{html.escape(NOTE)}</p>']
    for section in report.sections:
        out.append(f"<section><h2>{html.escape(section.title)}</h2>")
        for kind, items in _blocks(section.body):
            if kind == "para":
                out.append(f"<p>{_inline_html(items[0])}</p>")
            elif kind == "list":
                out.append("<ul>" + "".join(f"<li>{_inline_html(i)}</li>" for i in items) + "</ul>")
            else:
                head, *rows = items
                out.append("<table><thead><tr>" + "".join(f"<th>{_inline_html(c)}</th>" for c in head) + "</tr></thead><tbody>"
                           + "".join("<tr>" + "".join(f"<td>{_inline_html(c)}</td>" for c in row) + "</tr>" for row in rows)
                           + "</tbody></table>")
        out.append("</section>")
    out += ["</main>", "</body>", "</html>"]
    return "\n".join(out) + "\n"


_PDF_REPLACE = {"→": "->", "≥": ">=", "≤": "<=", "−": "-", "…": "...", "×": "x"}


def _inline_pdf(text: str) -> str:
    out = html.escape(text, quote=False)
    out = _CODE.sub(lambda m: '<font face="Courier">' + m.group(1) + "</font>", out)
    out = _BOLD.sub(r"<b>\1</b>", out)
    out = re.sub(r"\\(&lt;|&gt;)", r"\1", out)
    out = _UNESCAPE.sub(r"\1", out)
    for char, plain in _PDF_REPLACE.items():
        out = out.replace(char, plain)
    return out.encode("cp1252", errors="replace").decode("cp1252")  # what the standard PDF fonts can draw


def to_pdf(report: SolutionReport, path: Union[str, Path]) -> Path:
    """The report as a PDF at ``path``. Needs reportlab (``pip install "dclab-core[pdf]"``), imported only here."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise PdfUnavailable('The PDF report needs reportlab: pip install "dclab-core[pdf]" (or reportlab). '
                             "Markdown and HTML need nothing more.") from exc
    path = Path(path)
    styles = getSampleStyleSheet()
    body, bullet = styles["BodyText"], styles["BodyText"].clone("bullet", leftIndent=12, bulletIndent=2)
    cell = styles["BodyText"].clone("cell", fontSize=8, leading=10)
    story: list[Any] = [Paragraph(_inline_pdf(report.title), styles["Title"]), Paragraph(f"<i>{_inline_pdf(NOTE)}</i>", body), Spacer(1, 4 * mm)]
    width = A4[0] - 36 * mm
    for section in report.sections:
        story.append(Paragraph(_inline_pdf(section.title), styles["Heading2"]))
        for kind, items in _blocks(section.body):
            if kind == "para":
                story.append(Paragraph(_inline_pdf(items[0]), body))
            elif kind == "list":
                story += [Paragraph(_inline_pdf(i), bullet, bulletText="•") for i in items]
            else:
                columns = max(len(r) for r in items)
                rows = [[Paragraph(_inline_pdf(c), cell) for c in r] + [""] * (columns - len(r)) for r in items]
                table = Table(rows, colWidths=[width * (0.46 if i == 0 else 0.54 / max(columns - 1, 1)) for i in range(columns)], repeatRows=1)
                table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                           ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke)]))
                story.append(table)
        story.append(Spacer(1, 2 * mm))
    SimpleDocTemplate(str(path), pagesize=A4, title=report.title, author="DCLab", leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=16 * mm).build(story)
    return path
