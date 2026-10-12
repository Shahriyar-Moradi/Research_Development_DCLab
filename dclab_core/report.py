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
cannot all be traced, or that calls a model production-ready or deployable (``dclab_rnd/cited.py``'s readiness
rules, written here so the core stays pure), is left out, and :func:`build` runs that check once more on the finished
sections. :func:`sources` maps each number written to its path; :func:`check` re-reads every path against the records.
Record shapes no rule foresaw (a list where a mapping was expected, a missing part) cost a line, never the report.
The host says how it chose the rows the findings were read on (``rows_note``); the report writes it beside their count.

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
# a number as written: a sign only when it stands alone ("-2.4", not the "-05" of a date), digits with one decimal part
# or a leading-dot decimal (".5"), and a percent sign. Digits glued to a letter, an underscore or a dot (x1, top_2, v1.2)
# are part of a name, not a number.
NUMBER = re.compile(r"(?<![A-Za-z_\d.])-?(?:\d+(?:\.\d+)?|\.\d+)%?")
# what no sentence may say about a model, whatever surrounds it: dclab_rnd/cited.py's readiness and deployment rules,
# kept strict here (a denial does not excuse it), on text where "_" and every dash read as "-"
OVERCLAIM = re.compile(
    r"\b(?:production|deployment)[\s-]*(?:ready|readiness|grade|quality|approved|safe)\b|approved\s+for\s+production|"
    r"fit\s+for\s+production|read(?:y|iness)\s+(?:for|to\s+go\s+(?:to|into)|to\s+enter|for\s+use\s+in)\s+(?:production|deploy\w*)|"
    r"ready\s+to\s+(?:ship|launch|deploy|be\s+deployed|go\s+live)|safe\s+(?:to\s+deploy|for\s+production)|"
    r"\bgo(?:es|ing)?\s+(?:live|to\s+production|into\s+production)\b", re.I)
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
        text = " ".join(unicodedata.normalize("NFC", value).split())  # NFKC would read "10²" as "102"
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
    """Cut at a space, and never inside a number: the numbers kept are the text's first ones, each whole."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if text[limit] != " ":
        cut = cut.rsplit(" ", 1)[0] if " " in cut else re.sub(r"[-+]?[\d.,%]+$", "", cut)
    cut = cut.rstrip("\\").rstrip()
    return f"{cut} …" if cut else "…"


def overclaims(text: str) -> bool:
    plain = _DASHES.sub("-", unicodedata.normalize("NFKC", text)).replace("_", "-").replace("\\", "")
    return bool(OVERCLAIM.search(" ".join(plain.split())))


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
    ``path#n``: the n-th number in the name of the key the path ends at. Raises KeyError when there is none. A key that
    itself ends in ``@n`` (a metric named ``ndcg@10``) is read as that key first."""
    match = _SUFFIX.match(path)
    if not match:
        return _walk(ctx, path.split("."))[0]
    try:
        return _walk(ctx, path.split("."))[0]
    except KeyError:
        pass
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


def _m(value: Any) -> Mapping[str, Any]:
    """A record part read as a mapping: anything else (None, a list, a string) is empty."""
    return value if isinstance(value, Mapping) else {}


def _seq(value: Any) -> list[Any]:
    """A record part read as a list: anything else (None, a mapping, a string) is empty."""
    return list(value) if isinstance(value, (list, tuple)) else []


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


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
            items = _seq(plan.get("decisions"))
            risks = [dict(r) for r in _seq(plan.get("risks")) if isinstance(r, Mapping)]
            would = [w for w in _seq(plan.get("would_change")) if isinstance(w, str)]
    elif isinstance(plan, (list, tuple)):
        items, risks, would = list(plan), [], []
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
        decisions.append({"key": key, "choice": item.get("choice"), "reason": _str(item.get("reason")) or _str(item.get("why")),
                          "cites": _seq(item.get("cites")), "alternatives": _seq(item.get("alternatives")),
                          "source": item.get("source"), "status": item.get("status")})
    return decisions, risks, would


def context(*, brief: Any = None, findings: Any = None, plan: Any = None, records: Optional[Mapping[str, Any]] = None,
            rubric: Optional[list[Mapping[str, Any]]] = None, title: Optional[str] = None,
            rows_note: Optional[str] = None) -> dict[str, Any]:
    """Everything a report may cite, by the root of its paths: ``brief``, ``findings`` (by id), ``findings_meta``,
    ``plan`` (by decision key), ``plan_risks``, ``plan_would_change``, ``rubric``, ``title`` and each stage record
    (``data``, ``leakage``, ``features``, ``models``, ``final``)."""
    b, _ = _brief(brief)
    f, _ = _findings(findings)
    decisions, risks, would = _plan(plan)
    ctx: dict[str, Any] = {
        "title": title if isinstance(title, str) else "", "brief": b.model_dump(mode="json") if b else {},
        "findings": {x.id: x.model_dump(mode="json") for x in f.findings} if f else {},
        "findings_meta": {"training_rows": f.training_rows, "task": f.task, "target": f.target,
                          "rows_note": rows_note if isinstance(rows_note, str) else None} if f else {},
        "plan": {d["key"]: d for d in decisions}, "plan_risks": risks, "plan_would_change": would,
        "rubric": [dict(r) for r in _seq(rubric) if isinstance(r, Mapping)],
    }
    for stage, record in (records.items() if isinstance(records, Mapping) else ()):
        if stage in STAGES and isinstance(record, Mapping):
            ctx[stage] = record
    return ctx


# ---------------------------------------------------------------------------------------------------- the writer


def _same(a: Any, b: float) -> bool:
    return _finite(a) and abs(float(a) - b) <= 1e-9 * max(1.0, abs(b))


def _line_ok(text: str, paths: list[str], numbers: Mapping[str, float], ctx: Mapping[str, Any]) -> bool:
    """The core's citation check for one line: every number it writes is the value of one of ``paths`` as written, and
    that value is still what the record holds; and it calls no model production-ready."""
    if overclaims(text):
        return False
    for token in NUMBER.findall(text):
        if not any(p in numbers and matches(token, numbers[p]) and _same(_safe(ctx, p), numbers[p]) for p in paths):
            return False
    return True


class _Writer:
    """Collects lines into sections. A number enters a line only through :meth:`n` (a record's number), :meth:`q`
    (a record's text, its numbers traced) or :meth:`key` (a key's name); :meth:`line` keeps a line only when every
    number it writes is one of those, and it calls no model production-ready."""

    def __init__(self, ctx: Mapping[str, Any]):
        self.ctx = ctx
        self.numbers: dict[str, float] = {}
        self.sections: list[ReportSection] = []
        self.limits: list[str] = []
        self.used: set[str] = set()      # every record path a kept line read (numbers or not)
        self.dropped = 0
        self._title: Optional[str] = None
        self._lines: list[tuple[str, list[str]]] = []
        self._refs: list[str] = []
        self._pending: list[str] = []
        self._read: list[str] = []

    # reading
    def get(self, path: str, default: Any = None) -> Any:
        value = _safe(self.ctx, path)
        return default if value is None else value

    def has(self, path: str) -> bool:
        return self.get(path) not in (None, "", [], {})

    def n(self, path: str, style: str = "num") -> Optional[str]:
        value = self.get(path)
        if not _finite(value):
            return None
        self.numbers[path] = float(value)
        self._pending.append(path)
        self._read.append(path)
        return fmt(value, style)

    def _trace(self, path: str, text: str, kind: str) -> None:
        for index, (_, value) in enumerate(numbers_in(text)):
            self.numbers[f"{path}{kind}{index}"] = value
            self._pending.append(f"{path}{kind}{index}")

    def q(self, path: str, limit: int = 600, code: bool = False) -> str:
        value = self.get(path)
        text = _clip(render(value), limit)
        if text:
            self._read.append(path)
        self._trace(path, text, "@")
        return f"`{text}`" if code and text and "`" not in text else text

    def key(self, path: str, code: bool = True) -> str:
        try:
            _, name = _walk(self.ctx, path.split("."))
        except KeyError:
            return ""
        text = _clip(render(name), 200)
        self._read.append(path)
        self._trace(path, text, "#")
        return f"`{text}`" if code and text and "`" not in text else text

    def metric(self, path: str) -> str:
        """A metric's name from a record: its label when known, else the name itself, traced like any record text."""
        name = self.get(path)
        if isinstance(name, str) and name in METRIC_LABEL:
            self._read.append(path)
            return METRIC_LABEL[name]
        return self.q(path, 80) or "the primary metric"

    def ref(self, *names: str) -> None:
        self._refs.extend(n for n in names if n not in self._refs)

    # writing
    def line(self, text: Optional[str], limit: bool = False) -> bool:
        paths, self._pending = self._pending, []
        read, self._read = self._read, []
        if not text:
            return False
        text = _STOPS.sub(r"\1", text)
        paths = list(dict.fromkeys(paths))
        if not _line_ok(text, paths, self.numbers, self.ctx) or len(paths) > MAX_CITES:  # a section cites at most MAX_CITES paths
            self.dropped += 1
            return False
        self._lines.append((text, paths))
        self.used.update(read)
        if limit:
            self.limits.append(text[2:] if text.startswith("- ") else text)
        return True

    def section(self, title: str) -> None:
        self._flush()
        self._title, self._lines, self._refs, self._pending, self._read = title, [], [], [], []

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
        # the citation check once more, on what is written: each line against its section's cites and the records
        sections, kept_lines = [], []
        for section in self.sections:
            lines = [t for t in section.body.split("\n") if t]
            kept = [t for t in lines if _line_ok(t, section.cites, self.numbers, self.ctx)]
            self.dropped += len(lines) - len(kept)
            kept_lines += kept
            sections.append(section if len(kept) == len(lines) else
                            ReportSection(title=section.title, body=_join(kept) or "Nothing in the records covers this yet.",
                                          cites=section.cites))
        written = {t[2:] if t.startswith("- ") else t for t in kept_lines}
        limits = [text for text in self.limits if text in written]
        cited = {p for s in sections for p in s.cites}
        self._pending = []
        self._trace("title", title, "@")
        cited |= {p for p in self._pending if p.startswith("title@")}
        numbers = {p: v for p, v in self.numbers.items() if p in cited}
        return SolutionReport(title=title, sections=sections, limits=limits[:30], numbers=numbers)


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


def _task_of(ctx: Mapping[str, Any]) -> str:
    for stage in reversed(STAGES):
        task = _m(ctx.get(stage)).get("task")
        if isinstance(task, str) and task:
            return task
    task = _m(ctx.get("findings_meta")).get("task") or _m(_m(ctx.get("brief")).get("task")).get("value")
    return task if isinstance(task, str) else "unknown"


def _problem(w: _Writer, has_brief: bool, notes: list[str]) -> None:
    w.section("The problem as understood")
    brief = _m(w.ctx.get("brief"))
    unknown = []
    if not has_brief:
        w.line("No problem brief was given: the problem is described here only by what the run recorded.")
    else:
        for field, label in FACTS:
            fact = _m(brief.get(field))
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
        for i, _ in enumerate(_seq(brief.get("constraints"))):
            w.line(f"- **Constraint ({w.q(f'brief.constraints.{i}.kind', 40).replace('_', ' ')})**: {w.q(f'brief.constraints.{i}.text', 400)}")
        for i, _ in enumerate(_seq(brief.get("success_criteria"))):
            w.line(f"- **Success criterion**: {w.q(f'brief.success_criteria.{i}.value', 400)}")
        if unknown:
            w.line("Not known yet: " + ", ".join(unknown) + ".")
    stage = next((s for s in reversed(STAGES) if w.ctx.get(s)), None)
    if stage:
        w.ref(stage)
        metric = w.metric(f"{stage}.primary_metric")
        moment = w.q(f"{stage}.decision_time_rule", 500).rstrip(".")
        task = w.q(f"{stage}.task", 40)
        w.line(f"The run treated it as a{' ' + task if task else 'n unnamed'} problem scored by {metric}"
               + (f", predicted at this moment: {moment}" if moment else "") + ".")
    for note in notes:
        w.line(note)


def _data(w: _Writer, has_findings: bool, missing_note: Optional[str]) -> None:
    w.section("What the data shows")
    ctx = w.ctx
    data = _m(ctx.get("data"))
    if data:
        w.ref("data")
        e = "data.evidence"
        evidence = _m(data.get("evidence"))
        rows, train, holdout = w.n(f"{e}.source_rows", "int"), w.n(f"{e}.train_rows", "int"), w.n(f"{e}.holdout_rows", "int")
        split = w.q(f"{e}.holdout_split", 400)
        w.line(f"The table had {rows} rows; the run trained on {train} and locked {holdout} as the holdout"
               + (f" ({split})." if split else ".") if rows and train and holdout else None)
        sampled = _m(evidence.get("sampling"))
        if _finite(sampled.get("rows_used")) and _finite(sampled.get("source_rows")) and sampled["rows_used"] < sampled["source_rows"]:
            used = w.n(f"{e}.sampling.rows_used", "int")
            rule = w.q(f"{e}.sampling.rule", 300)
            w.line(f"The run used {used} of them" + (f" ({rule})" if rule else "") + ", so its numbers describe that sample.")
        if w.has(f"{e}.cv_protocol"):
            w.line(f"Choices were made on the training rows only, with {w.q(f'{e}.cv_protocol', 300)}.")
        task = _str(data.get("task"))
        t = f"{e}.target_summary"
        if task == "binary" and _m(evidence.get("target_summary")):
            rate, positives, majority = w.n(f"{t}.positive_rate_train", "pct"), w.n(f"{t}.positives_train", "int"), w.n(f"{t}.majority_baseline_accuracy_train", "pct")
            w.line(f"Positives are {rate} of the training rows ({positives} rows); predicting the majority class for every row "
                   f"already scores {majority} accuracy, so accuracy alone cannot judge a model." if rate and positives and majority else None)
        elif task == "multiclass" and _m(evidence.get("target_summary")):
            classes, small, large = w.n(f"{t}.classes", "int"), w.n(f"{t}.smallest_class_count", "int"), w.n(f"{t}.largest_class_count", "int")
            w.line(f"The target has {classes} classes; the smallest has {small} training rows and the largest {large}, so the "
                   f"rare classes weigh on a macro metric." if classes and small and large else None)
        elif task == "regression" and _m(evidence.get("target_summary")):
            mean, median, low, high = (w.n(f"{t}.{k}", "auto") for k in ("mean_train", "median_train", "min_train", "max_train"))
            w.line(f"The target has a mean of {mean} and a median of {median} on the training rows, from {low} to {high}."
                   if None not in (mean, median, low, high) else None)
        g = f"{e}.group_stats"
        if _m(evidence.get("group_stats")):
            groups, held, median = w.n(f"{g}.train_groups", "int"), w.n(f"{g}.holdout_groups", "int"), w.n(f"{g}.median_rows_per_group", "auto")
            w.line(f"Rows repeat for the group column: {groups} groups in training and {held} others in the holdout, a median of "
                   f"{median} rows each, so folds keep each group on one side." if groups and held and median else None)
        risky = _m(evidence.get("risk_features"))
        if risky:
            parts = [f"{w.key(f'{e}.risk_features.{name}')} ({w.q(f'{e}.risk_features.{name}', 120)})" for name in list(risky)[:15]
                     if isinstance(name, str)]
            w.line("Columns the profile flagged for review: " + "; ".join(parts) + "." if parts else None)
    for stage in ("final", "data"):
        if _m(_m(ctx.get(stage)).get("labels")).get("coverage") == "partial":
            source = w.q(f"{stage}.labels.file", 200, code=True) or "a label file"
            provenance = w.q(f"{stage}.labels.provenance", 400)
            w.line(f"The labels come from {source} and are partial" + (f" ({provenance})" if provenance else "")
                   + ": a row without a label may be a case nobody found.")
            break
    leakage = _m(ctx.get("leakage"))
    if leakage:
        w.ref("leakage")
        e = "leakage.evidence"
        evidence = _m(leakage.get("evidence"))
        if _seq(evidence.get("declared_leakage_features")):
            lift = w.n(f"{e}.apparent_lift", "signed")
            metric = w.metric(f"{e}.apparent_lift_metric")
            columns = w.q(f"{e}.declared_leakage_features", 400)
            w.line(f"Kept out as written after the prediction moment: {columns}."
                   + (f" Including them would move training-CV {metric} by {lift}, a gain that would not exist when predicting." if lift else ""))
        else:
            w.line("No column was declared as written after the prediction moment.")
        candidates = _seq(evidence.get("heuristic_review_candidates"))
        parts = [f"{w.q(f'{e}.heuristic_review_candidates.{i}.feature', 100, code=True)} ({w.q(f'{e}.heuristic_review_candidates.{i}.reasons', 160)})"
                 for i, c in enumerate(candidates[:10]) if isinstance(c, Mapping)]
        w.line("Flagged by the leakage heuristics for a person's review: " + "; ".join(parts) + "." if parts else None)
        gap = w.n(f"{e}.random_vs_time_cv.optimism_gap", "signed")
        metric = w.metric(f"{e}.apparent_lift_metric")
        w.line(f"A random split would score {metric} {gap} against the time-ordered one: it sees the future." if gap else None)
    findings = _m(ctx.get("findings"))
    if findings:
        rows = w.n("findings_meta.training_rows", "int")
        note = w.q("findings_meta.rows_note", 300)
        w.line((f"The analysis behind the plan read {rows} rows" if rows else "The analysis behind the plan")
               + (f" ({note})" if note else "") + "; its findings, most severe first:")
        ordered = sorted(findings, key=lambda fid: SEVERITY.get(_m(findings[fid]).get("severity"), 3))
        for fid in ordered[:20]:
            p = f"findings.{fid}"
            w.ref(p)
            w.line(f"- **{w.q(f'{p}.topic', 80)}** ({w.q(f'{p}.severity', 20)}, {w.q(f'{p}.id', 80, code=True)}): "
                   f"{w.q(f'{p}.statement', 500)} Implication: {w.q(f'{p}.implication', 500)}")
    elif not has_findings and not data:
        w.line("Neither the data findings nor the data stage's record were given: nothing about the data can be reported.")
    if missing_note:
        w.line(missing_note)


def _decisions(w: _Writer) -> None:
    w.section("Decisions and reasons")
    ctx = w.ctx
    plan = _m(ctx.get("plan"))
    if plan:
        w.line("The solution plan, one decision per point, each with its reason and what was set aside:")
    for key, decision in list(plan.items())[:40]:
        p = f"plan.{key}"
        w.ref(p)
        status = "; ".join(f"{label}: {w.q(f'{p}.{label}', 40)}" for label in ("status", "source") if w.has(f"{p}.{label}"))
        cites = ", ".join(w.q(f"{p}.cites.{i}", 80, code=True) for i, _ in enumerate(_seq(decision.get("cites"))[:6]))
        alternatives = "; ".join(f"{w.q(f'{p}.alternatives.{i}.choice', 160)} ({w.q(f'{p}.alternatives.{i}.why_not', 200)})"
                                 for i, a in enumerate(_seq(decision.get("alternatives"))[:4]) if isinstance(a, Mapping))
        reason = w.q(f"{p}.reason", 500)
        w.line(f"- **{w.q(f'{p}.key', 80)}**: {w.q(f'{p}.choice', 400) or 'none'}"
               + (f". {reason}" if reason else "") + (f" ({status})" if status else "")
               + (f" Rests on {cites}." if cites else "") + (f" Set aside: {alternatives}." if alternatives else ""))
    decided = False
    for stage, what in (("features", "Feature recipe"), ("models", "Model family")):
        record = _m(ctx.get(stage))
        decision = _m(record.get("decision"))
        if not decision.get("chosen"):
            continue
        decided = True
        w.ref(stage)
        chosen = w.q(f"{stage}.decision.chosen", 120, code=True)
        rule = w.q(f"{stage}.decision.rule", 900)
        person = (f" A person chose it over the rule's {w.q(f'{stage}.decision.selected', 120, code=True)}."
                  if decision.get("overridden") is True else "")
        w.line(f"- **{what}**: {chosen}" + (f", by this rule: {rule}" if rule else "") + person)
        evidence = _m(record.get("evidence"))
        if stage == "features" and evidence.get("reference_recipe") and evidence.get("reference_recipe") != decision.get("chosen"):
            gain = w.n("features.evidence.selected_vs_reference_gain", "signed")
            reference = w.q("features.evidence.reference_recipe", 120, code=True)
            metric = w.metric("features.primary_metric")
            w.line(f"- Against the first recipe, {reference}, it changes training-CV {metric} by {gain} on the same folds." if gain else None)
        if stage == "models" and w.has("models.evidence.rare_event_choice.why"):
            w.line(f"- Why this family: {w.q('models.evidence.rare_event_choice.why', 600)}.")
    tuning = _m(_m(_m(ctx.get("final")).get("evidence")).get("tuning_decision"))
    if tuning:
        w.ref("final")
        t = "final.evidence.tuning_decision"
        gain, required = w.n(f"{t}.gain", "signed"), w.n(f"{t}.required_gain")
        metric = w.metric("final.primary_metric")
        w.line(f"- **Tuning**: {'accepted' if tuning.get('accepted') is True else 'rejected'}: the best candidate, "
               f"{w.q(f'{t}.best_candidate', 120, code=True)}, changed training-CV {metric} by {gain} "
               f"against a required {required}; the final model used {w.q('final.evidence.selected_optimization', 120, code=True)}."
               if gain and required else None)
    if not plan and not decided and not tuning:
        w.line("No decision was recorded: neither a solution plan nor the stages that choose a recipe, a model and its tuning.")


def _interval(w: _Writer, path: str) -> tuple[Optional[str], Optional[str], bool]:
    """(low, high, drawn): an interval is written only when its bootstrap drew at least one resample."""
    ci = _m(w.get(path))
    if not ci:
        return None, None, False
    draws = ci.get("draws")
    if _finite(draws) and draws > 0:
        return w.n(f"{path}.low"), w.n(f"{path}.high"), True
    return None, None, False


def _results(w: _Writer, notes: list[str]) -> None:
    w.section("Results, with their intervals")
    ctx = w.ctx
    final = _m(ctx.get("final"))
    if not final:
        w.line("The holdout has not been scored: no result can be reported yet.")
        notes.append("- No final stage record: the model has not been scored on the holdout, so nothing here measures how well it predicts.")
    else:
        w.ref("final")
        e = "final.evidence"
        evidence = _m(final.get("evidence"))
        primary = _str(final.get("primary_metric"))
        label = w.metric("final.primary_metric")
        value = w.n(f"{e}.holdout_metrics.{primary}") if primary else None
        model, recipe = w.q(f"{e}.model", 120, code=True), w.q(f"{e}.feature_recipe", 120, code=True)
        split = w.q(f"{e}.holdout_split", 400)
        uses = evidence.get("holdout_uses_in_this_project")
        if _finite(uses) and uses > 1:
            once = f"on the holdout, which was scored {w.n(f'{e}.holdout_uses_in_this_project', 'int')} times in this project"
        else:
            once = "on the holdout, used once"
        w.line(f"The final model" + (f", {model}" + (f" on the {recipe} recipe" if recipe else "") + "," if model else "")
               + f" scored {label} {value} {once}" + (f" ({split})." if split else ".") if value else None)
        if _m(evidence.get("holdout_primary_metric_ci")):
            low, high, drawn = _interval(w, f"{e}.holdout_primary_metric_ci")
            method = w.q(f"{e}.holdout_primary_metric_ci.method", 200)
            if drawn and low and high:
                w.line(f"Its bootstrap interval runs from {low} to {high}" + (f" ({method})" if method else "")
                       + ": the range the holdout score could take on another draw of the same rows.")
            else:
                w.line("No bootstrap interval could be drawn" + (f" ({method})" if method else "")
                       + ": the holdout gave no resample on which the metric could be computed, so the score stands without a range.")
        cv, gap = w.n(f"{e}.cv_selected_metric_mean"), w.n(f"{e}.cv_to_holdout_gap", "signed")
        w.line(f"On the training folds it averaged {cv}; the holdout differs by {gap} (positive means the holdout scored better)."
               if cv and gap else None)
        for name, values in _m(evidence.get("baselines")).items():
            if not isinstance(values, Mapping) or not isinstance(name, str):
                continue
            parts = []
            for metric_name in values:
                if not isinstance(metric_name, str):
                    continue
                written = w.n(f"{e}.baselines.{name}.{metric_name}")
                if written:
                    known = METRIC_LABEL.get(metric_name) or w.key(f"{e}.baselines.{name}.{metric_name}")
                    parts.append(f"{known} {written}")
            w.line(f"- Baseline {w.key(f'{e}.baselines.{name}')} on the same holdout: " + ", ".join(parts) + "." if parts else None)
        task = _task_of(ctx)
        kind = "binary" if "binary" in task else ("multiclass" if task == "multiclass" else "regression")
        metrics = _m(evidence.get("holdout_metrics"))
        others = [f"{METRIC_LABEL[m]} {w.n(f'{e}.holdout_metrics.{m}')}" for m in OTHER_METRICS[kind] if m != primary and _finite(metrics.get(m))]
        w.line("Other holdout metrics: " + ", ".join(others) + "." if others else None)
        if kind == "binary" and _finite(metrics.get("alert_budget")):
            h = f"{e}.holdout_metrics"
            budget, alerts = w.n(f"{h}.alert_budget", "pct"), w.n(f"{h}.alerts_at_budget", "int")
            recall, precision = w.n(f"{h}.recall_at_budget"), w.n(f"{h}.precision_at_budget")
            r_low, r_high, drawn = _interval(w, f"{h}.recall_at_budget_ci")
            if drawn and r_low and r_high:
                interval = f" (interval {r_low} to {r_high})"
            else:
                interval = " (no interval could be drawn)" if _m(metrics.get("recall_at_budget_ci")) else ""
            w.line(f"At the review budget of {budget} of rows ({alerts} alerts on the holdout), recall is {recall}{interval}"
                   f" and precision {precision}." if budget and alerts and recall and precision else None)
        _assessment(w, kind, _m(evidence.get("task_assessment")))
    models = _m(ctx.get("models"))
    m_evidence = _m(models.get("evidence"))
    ranking = _seq(m_evidence.get("ranking"))
    if ranking:
        w.ref("models")
        protocol = w.q("models.evidence.cv_protocol", 300)
        w.line(f"The model screen on the training folds" + (f" ({protocol})" if protocol else "")
               + f", mean and fold spread of {w.metric('models.primary_metric')}:")
        for i, row in enumerate(ranking[:10]):
            r = f"models.evidence.ranking.{i}"
            mean, std = w.n(f"{r}.mean"), w.n(f"{r}.std")
            w.line(f"- {w.q(f'{r}.model', 80, code=True)}: {mean} ± {std}" if isinstance(row, Mapping) and mean and std else None)
    for i, r in enumerate(_seq(_m(m_evidence.get("review")).get("rounds"))):
        if not isinstance(r, Mapping):
            continue
        p = f"models.evidence.review.rounds.{i}"
        base, variant = w.n(f"{p}.base_mean"), w.n(f"{p}.variant_mean")
        why = w.q(f"{p}.why", 400)
        w.line(f"- Review round {w.n(f'{p}.round', 'int') or ''}: {w.q(f'{p}.experiment', 80, code=True)} on {w.q(f'{p}.family', 80, code=True)}, "
               + (f"{base} → {variant} on the same folds, " if base and variant else "")
               + f"{'kept' if r.get('adopted') is True else 'not kept'}" + (f" ({why})." if why else "."))


def _assessment(w: _Writer, kind: str, assessment: Mapping[str, Any]) -> None:
    if not assessment:
        return
    a = "final.evidence.task_assessment"
    if kind == "binary":
        t = f"{a}.threshold"
        threshold = _m(assessment.get("threshold"))
        if threshold:
            value = w.n(f"{t}.value")
            on, by = w.q(f"{t}.chosen_on", 120), w.q(f"{t}.chosen_by", 300)
            w.line(f"The alert threshold {value} was chosen" + (f" on the {on}" if on else "") + (f": {by}." if by else ".") if value else None)
            alerts, precision, recall = w.n(f"{t}.holdout.alerts", "int"), w.n(f"{t}.holdout.precision"), w.n(f"{t}.holdout.recall")
            w.line(f"Rows it flagged on the holdout: {alerts}" + (f", with precision {precision}" if precision else "")
                   + (f" and recall {recall}" if recall else "") + "." if alerts else None)
        if _m(assessment.get("calibration")):
            c = f"{a}.calibration"
            brier, ece = w.n(f"{c}.brier"), w.n(f"{c}.expected_calibration_error")
            verdict = w.q(f"{c}.verdict", 120)
            w.line(f"Calibration: Brier score {brier}, expected calibration error {ece}" + (f" ({verdict})" if verdict else "") + "."
                   if brier and ece else None)
    elif kind == "multiclass":
        per_class = _seq(assessment.get("per_class"))
        if per_class:
            w.line("Per class on the holdout (precision, recall):")
        for i, row in enumerate(per_class[:15]):
            p = f"{a}.per_class.{i}"
            precision, recall = w.n(f"{p}.precision"), w.n(f"{p}.recall")
            w.line(f"- {w.q(f'{p}.class', 80, code=True)}: {precision}, {recall}" if isinstance(row, Mapping) and precision and recall else None)
        for i, row in enumerate(_seq(assessment.get("most_confused"))[:5]):
            p = f"{a}.most_confused.{i}"
            rows, share = w.n(f"{p}.rows", "int"), w.n(f"{p}.share_of_true", "pct")
            w.line(f"- Confused: {w.q(f'{p}.true', 80, code=True)} taken for {w.q(f'{p}.predicted', 80, code=True)} on {rows} rows, "
                   f"{share} of its rows." if isinstance(row, Mapping) and rows and share else None)
        for i, row in enumerate(_seq(assessment.get("tiny_classes"))[:8]):
            if isinstance(row, Mapping):
                p = f"{a}.tiny_classes.{i}"
                recall = w.n(f"{p}.recall")
                w.line(f"- A tiny class, {w.q(f'{p}.class', 80, code=True)}" + (f", has recall {recall}." if recall else "."))
        top = _m(assessment.get("top_k_accuracy"))
        parts = [f"{w.key(f'{a}.top_k_accuracy.{k}')} {w.n(f'{a}.top_k_accuracy.{k}')}" for k, v in top.items() if isinstance(k, str) and _finite(v)]
        w.line("Top-k accuracy: " + ", ".join(parts) + "." if parts else None)
        recall = w.n(f"{a}.macro_recall")
        w.line(f"Macro recall over every class: {recall}." if recall else None)
    else:
        for level, interval in list(_m(assessment.get("intervals")).items())[:4]:
            if not isinstance(level, str) or not isinstance(interval, Mapping):
                continue
            p = f"{a}.intervals.{level}"
            low, high, cover = w.n(f"{p}.from_prediction.0", "signed"), w.n(f"{p}.from_prediction.1", "signed"), w.n(f"{p}.holdout_coverage", "pct")
            w.line(f"- The {w.key(p, code=False)} interval adds from {low} to {high} to a prediction; it covered {cover} of the holdout rows."
                   if low and high and cover else None)
        if _m(assessment.get("at_cap")):
            c = f"{a}.at_cap"
            cap, rows, at, below = w.n(f"{c}.cap", "auto"), w.n(f"{c}.holdout_rows_at_cap", "int"), w.n(f"{c}.mae_at_cap"), w.n(f"{c}.mae_below_cap")
            w.line(f"The target is capped at {cap}: on the {rows} holdout rows at the cap the MAE is {at}, against {below} below it."
                   if cap and rows and at and below else None)
        for column, segments in list(_m(assessment.get("error_by_segment")).items())[:3]:
            if not isinstance(column, str) or not isinstance(segments, Mapping):
                continue
            p = f"{a}.error_by_segment.{column}"
            parts = []
            for name in list(segments)[:8]:
                if not isinstance(name, str):
                    continue
                mae, rows = w.n(f"{p}.{name}.mae"), w.n(f"{p}.{name}.rows", "int")
                if mae and rows:
                    parts.append(f"{w.key(f'{p}.{name}')} {mae} on {rows} rows")
            w.line(f"- MAE by {w.key(p)}: " + ", ".join(parts) + "." if parts else None)
        grow = w.n(f"{a}.errors_grow_with_prediction.spearman")
        w.line(f"Errors against the size of the prediction: Spearman correlation {grow}." if grow else None)


def _limits(w: _Writer, notes: list[str], partial: bool, unknown: list[str], missing: list[str]) -> None:
    w.section("Limits, and what the data cannot show")
    ctx = w.ctx
    final = _m(ctx.get("final"))
    evidence = _m(final.get("evidence"))
    for note in notes:
        w.line(note, limit=True)
    if missing:
        w.line(f"- No record for these stages: {', '.join(STAGE_NAMES[s] for s in missing)}; what they would have shown is not here.", limit=True)
    if w.has("final.evidence.precision_note"):
        w.line(f"- {w.q('final.evidence.precision_note', 600)}", limit=True)
    elif w.has("final.evidence.task_assessment.precision_note"):
        w.line(f"- {w.q('final.evidence.task_assessment.precision_note', 600)}", limit=True)
    for i, _ in enumerate(_seq(evidence.get("limits"))[:8]):
        w.line(f"- {w.q(f'final.evidence.limits.{i}', 600)}" if w.has(f"final.evidence.limits.{i}") else None, limit=True)
    if w.has("final.evidence.task_assessment.at_cap.note"):
        w.line(f"- At the cap: {w.q('final.evidence.task_assessment.at_cap.note', 300)}.", limit=True)
    seen: set[str] = set()
    for i, claim in enumerate(_seq(final.get("claims"))):
        for j, text in enumerate(_seq(_m(claim).get("limitations"))):
            if not isinstance(text, str) or text in seen or len(seen) >= 4:
                continue
            seen.add(text)
            w.line(f"- {w.q(f'final.claims.{i}.limitations.{j}', 500)}", limit=True)
    uses = evidence.get("holdout_uses_in_this_project")
    if _finite(uses) and uses > 1:
        reason = w.q("final.evidence.holdout_reuse_reason", 300)
        w.line(f"- The holdout was scored {w.n('final.evidence.holdout_uses_in_this_project', 'int')} times in this project, so it is "
               "no longer untouched and the score is optimistic" + (f": {reason}." if reason else "."), limit=True)
    if w.has("leakage.evidence.warning"):
        w.line(f"- {w.q('leakage.evidence.warning', 400)}", limit=True)
    for i, risk in enumerate(_seq(ctx.get("plan_risks"))[:6]):
        if isinstance(risk, Mapping):
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
    review = _m(_m(_m(ctx.get("models")).get("evidence")).get("review"))
    tried = {r.get("experiment") for r in _seq(review.get("rounds")) if isinstance(r, Mapping)}
    for i, experiment in enumerate(_seq(_m(review.get("review")).get("next_experiments"))[:6]):
        if not isinstance(experiment, Mapping) or experiment.get("name") in tried:
            continue
        p = f"models.evidence.review.review.next_experiments.{i}"
        cost = w.q(f"{p}.cost", 20)
        w.line(f"- Try {w.q(f'{p}.name', 80, code=True)}: {w.q(f'{p}.expected_gain', 300)}" + (f" (cost {cost})." if cost else "."))
    questions = [(i, q) for i, q in enumerate(_seq(_m(ctx.get("brief")).get("open_questions"))) if isinstance(q, Mapping)]
    for i, _ in sorted(questions, key=lambda iq: iq[1].get("impact", 5) if _finite(iq[1].get("impact")) else 5)[:6]:
        w.line(f"- Ask: {w.q(f'brief.open_questions.{i}.text', 400)}")
    for i, _ in enumerate(_seq(ctx.get("plan_would_change"))[:5]):
        w.line(f"- Revisit the plan if this changes: {w.q(f'plan_would_change.{i}', 400)}")
    if partial:
        w.line("- Have a person review a sample of the highest-scored unlabelled rows, to learn how many are real cases the labels missed.")
    w.line("- Score rows from a later period than the holdout, and watch the inputs for drift, before the model drives decisions.")


# what a report never delivers by itself: a requirement asking for one of these is mapped outside it
OUTSIDE = (
    (r"\b(?:rest(?:ful)?\s+)?apis?\b|\bendpoints?\b|\bfastapi\b|\bflask\b", "an API"),
    (r"\b(?:web|scoring|prediction|inference|micro|rest)[\s-]?services?\b|\bserv(?:e|es|ed|ing)\b", "a service"),
    (r"\bdocker(?:file)?\b|\bcontaineri[sz](?:e|ed|ation)\b|\bcontainers?\b|\bkubernetes\b", "a container"),
    (r"\bdeploy(?:s|ed|ing|ment)?\b", "a deployment"),
    (r"\bsource\s+control\b|\bversion\s+control\b|\bgit(?:hub|lab)?\b|\brepositor(?:y|ies)\b|\brepo\b", "source control"),
    (r"\bunit\s+tests?\b|\btest\s+suites?\b|\bintegration\s+tests?\b|\bpytest\b|\bunittest\b|\bwrite\s+tests\b|\btests\s+for\b", "tests"),
    (r"\bnotebooks?\b|\bjupyter\b|\bipynb\b", "a notebook"),
    (r"\bdashboards?\b", "a dashboard"),
    (r"\bsource\s+code\b|\bclean\s+code\b|\bcodebase\b", "code"),
)
# a requirement's words -> the section that answers it, and the record paths a kept line must have read for it to count
RUBRIC_RULES = (
    (("leak",), "Decisions and reasons", ("leakage.",)),
    (("threshold", "budget", "alert", "cost"), "Results, with their intervals",
     ("final.evidence.holdout_metrics.alert_budget", "final.evidence.holdout_metrics.recall_at_budget", "final.evidence.task_assessment.threshold")),
    (("model", "classif", "regress", "predict", "algorithm", "train"), "Results, with their intervals", ("final.evidence.holdout_metrics",)),
    (("metric", "evaluat", "validat", "score", "accuracy", "precision", "recall", "auc", "error", "baseline"), "Results, with their intervals",
     ("final.evidence.holdout_metrics", "final.evidence.holdout_primary_metric_ci", "final.evidence.baselines")),
    (("feature",), "Decisions and reasons", ("features.",)),
    (("data", "eda", "explor", "clean", "quality", "missing", "profil", "distribution"), "What the data shows", ("data.", "findings.")),
    (("limit", "risk", "assumption", "uncertain"), "Limits, and what the data cannot show",
     ("final.evidence.limits", "final.evidence.precision_note", "final.claims", "plan_risks", "leakage.evidence.warning")),
    (("next", "improv", "recommend", "future"), "Next steps",
     ("models.evidence.review.review.next_experiments", "brief.open_questions", "plan_would_change")),
    (("report", "summary", "write", "document", "present", "explain", "problem"), "The problem as understood", ("brief.", "final.")),
)


def _where(text: str, used: set[str]) -> tuple[str, str]:
    lowered = " ".join(text.lower().split())
    for pattern, what in OUTSIDE:
        if re.search(pattern, lowered):
            return "outside this report", f"not covered here: it asks for {what}, which a report does not produce"
    words = re.findall(r"[a-z]+", lowered)
    for stems, section, prefixes in RUBRIC_RULES:
        if any(word.startswith(stem) for stem in stems for word in words):
            if any(path.startswith(prefix) for path in used for prefix in prefixes):
                return section, "covered"
            return section, f"see {section}; not covered by a record"
    return "not found", "open: nothing in the records matches it"


def _deliverables(w: _Writer) -> None:
    rubric = _seq(w.ctx.get("rubric"))
    items = []
    for i, item in enumerate(rubric[:40]):
        name_key = next((k for k in ("requirement", "name", "text", "title") if isinstance(_m(item).get(k), str) and item[k].strip()), None)
        if name_key:
            items.append((f"rubric.{i}", name_key, item))
    if not items:
        deliverables = _seq(_m(w.ctx.get("brief")).get("deliverables"))
        items = [(f"brief.deliverables.{i}", "name", d) for i, d in enumerate(deliverables[:40]) if isinstance(_m(d).get("name"), str)]
    if not items:
        return
    used = set(w.used)  # what the sections above kept: a requirement is covered only by a line written from its record
    w.section("Deliverables map")
    w.line("Each requirement, its points when the rubric gives them, and where this report or the run meets it:")
    w.line("| Requirement | Points | Where | Status |")
    w.line("|---|---|---|---|")
    for path, name_key, item in items:
        requirement = w.q(f"{path}.{name_key}", 300)
        points = w.n(f"{path}.points") or "–"
        if isinstance(item.get("met_by"), str) and item["met_by"].strip():
            where, status = w.q(f"{path}.met_by", 200), "named by the rubric, not checked against a record"
        else:
            where, status = _where(item[name_key], used)
        w.line(f"| {requirement} | {points} | {where} | {status} |")


# ---------------------------------------------------------------------------------------------------- build


SECTION_WRITERS = ("problem", "data", "decisions", "results", "limits", "next", "deliverables")


def build(*, brief: Any = None, findings: Any = None, plan: Any = None, records: Optional[Mapping[str, Any]] = None,
          rubric: Optional[list[Mapping[str, Any]]] = None, title: Optional[str] = None,
          rows_note: Optional[str] = None) -> SolutionReport:
    """The solution report of a run. Every input may be missing or malformed: the report then says what is missing, and
    why that limits it, rather than failing. ``rubric``: a list of requirements (``{"requirement" | "name": text,
    "points": number, "met_by": optional text}``); without one, the brief's deliverables are mapped when it has any.
    ``rows_note``: how the host chose the rows the findings were read on (written beside their count)."""
    notes: list[str] = []
    b, brief_note = _brief(brief)
    f, findings_note = _findings(findings)
    notes += [f"- {n}" for n in (brief_note, findings_note) if n]
    if records is not None and not isinstance(records, Mapping):
        notes.append("- The stage records given are not a mapping of stage to record, so they are left out.")
        records = None
    ctx = context(brief=b, findings=f, plan=plan, records=records, rubric=rubric, title=title, rows_note=rows_note)
    w = _Writer(ctx)
    missing = [s for s in STAGES if not ctx.get(s)]
    partial = any(_m(_m(ctx.get(s)).get("labels")).get("coverage") == "partial" for s in STAGES) or \
        _m(ctx["plan"].get("labels.coverage")).get("choice") == "partial" or \
        _m(ctx["brief"].get("label_coverage")).get("value") == "partial"
    unknown = [label.lower() for field, label in FACTS if b is not None and _m(ctx["brief"].get(field)).get("status", "unknown") == "unknown"
               and field in ("prediction_moment", "label_coverage", "target_column", "unit", "entities")]
    result_notes: list[str] = []
    failed: list[str] = []
    steps = {
        "problem": lambda: _problem(w, b is not None, [n[2:] for n in notes]),
        "data": lambda: _data(w, f is not None, None),
        "decisions": lambda: _decisions(w),
        "results": lambda: _results(w, result_notes),
        "limits": lambda: _limits(w, result_notes + [f"- The {name} section could not be written from these records, so it is "
                                                     "incomplete." for name in failed], partial, unknown, missing),
        "next": lambda: _next(w, partial, missing),
        "deliverables": lambda: _deliverables(w),
    }
    for name in SECTION_WRITERS:
        try:
            steps[name]()
        except Exception:  # noqa: BLE001 - a record shape no rule foresaw costs that section, never the report
            failed.append(name)
            w._pending, w._read = [], []
    if w.dropped:
        w.section("About this report")
        w.line("Some lines were left out because a number in them could not be traced to a record, or because they "
               "called a model ready for use; the records hold the full detail.")
    name = _clip(render(title), 300) if isinstance(title, str) else ""
    if not name or overclaims(name):
        name = "Solution report"
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


def _text(markdown: str) -> str:
    """Report text without its Markdown escapes (for a page title or a PDF's metadata)."""
    return _UNESCAPE.sub(r"\1", markdown)


def to_markdown(report: SolutionReport) -> str:
    title = " ".join(render(_text(report.title)).split()) or "Solution report"  # one heading line, escaped
    parts = [f"# {title}", f"_{NOTE}_"]
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
           f"<title>{html.escape(_text(report.title))}</title>", f"<style>{_CSS}</style>", "</head>", "<body>", "<main>",
           f"<h1>{html.escape(_text(report.title))}</h1>", f'<p class="note">{html.escape(NOTE)}</p>']
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
    SimpleDocTemplate(str(path), pagesize=A4, title=_text(report.title), author="DCLab", leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=16 * mm).build(story)
    return path
