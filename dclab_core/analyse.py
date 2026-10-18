"""Initial data analysis for decisions (package 17.3): tables in, :class:`~dclab_core.contracts.DataFindings` out.

Not a profile for its own sake: each finding ends in an *implication* for the plan and names the decision points it
bears on (``split.strategy``, ``metric.primary``, ``features.forbidden``, ...). Evidence is numbers and column names,
never row values.

What is computed on which rows. Anything that reads the target, or would steer a modelling choice (how the target
behaves, which columns predict it, drift, entities, conflicting labels), uses only the rows the host marks as training
(``train_index``; the main product's rule: the holdout is never looked at). Facts about the table's structure that need
no target (duplicates, keys, constant or impossible values) are computed on all rows, because a duplicate or an id
across the split *is* the problem, and their findings say so (``scope``).

Every signal that claims something about the target is measured *out of fold* (each half of the training rows is
scored with what the other half taught), against chance, so a column or an entity is not called predictive because it
memorises. Names alone never forbid a column: a name that sounds written after the outcome is a question to ask
unless the column also predicts the target.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

from dclab_core.advisor import Advisor
from dclab_core.agents import Role, run_role
from dclab_core.contracts import DataFindings, Finding, ProblemBrief, TableSummary, TaskType

POST_MOMENT_NAME = re.compile(r"(^|_)(actual|final|resolved|resolution|result|outcome|closed|refund(ed)?|chargeback|"
                              r"delivered|arrival|completed|settled|approved_by|rating|response|handled|paid_out)(_|$)", re.I)
HISTORY_NAME = re.compile(r"(^|_)(prev|previous|prior|last|avg|average|mean|median|count|sum|total|history|historical|"
                          r"expected|estimated|planned|scheduled|sla|target|lag|rolling|\d+d|\d+_days?)(_|$)", re.I)
ID_NAME = re.compile(r"(^id$|_id$|_ref$|^id_|_key$|_uuid$)", re.I)
TIME_NAME = re.compile(r"(date|time|_at$|timestamp|created|month)", re.I)
TINY_SHARE, TINY_ROWS = 0.01, 30
from dclab_core.modifiers import RARE_BELOW as RARE_SHARE  # noqa: E402  (30.11: one cut-off, was 1%)
LIST_CAP = 20

# single-column power, out of fold: AUC for binary (folded to >= 0.5), gain over the majority class normalised to [0, 1]
# for multiclass, |Spearman| for regression
POWER_ALARM = {"binary": 0.9, "multiclass": 0.8, "regression": 0.95}   # predicts the target almost perfectly
POWER_NAMED = {"binary": 0.6, "multiclass": 0.15, "regression": 0.3}   # with a post-outcome name: enough to forbid
ENTITY_MARGIN = {"binary": 0.6, "multiclass": 0.1, "regression": 0.3}  # the entity alone tells the target


def _slug(text: Any) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", str(text).lower()).strip("-")[:120] or "x"


# The decisions a finding may name (``affects``), each with the decision ledger's point it is recorded as (20.5); None:
# a plan decision the ledger does not record on its own yet (the target's transform, a risk to state, the baseline).
AFFECTS: dict[str, Optional[str]] = {
    "task.framing": "task.framing", "labels.coverage": "labels.situation", "labels.join": "labels.join_key",
    "split.strategy": "split.strategy", "metric.primary": "metric.primary", "metric.secondary": "metric.primary",
    "threshold.policy": "threshold.operating_point", "features.forbidden": "feature.leakage_suspect",
    "features.to_confirm": "feature.leakage_suspect", "features.identifiers": "column.is_identifier",
    "features.direction": "features.recipe", "features.by_segment": "features.recipe", "features.entity_history": "features.recipe",
    "data.cleaning": "data.cleaning_step", "model.families": "model.family",
    "target.transform": None, "target.log": None, "risk.check": None, "baseline": None,
}


def _f(fid: str, topic: str, statement: str, evidence: dict[str, Any], implication: str, affects: Sequence[str],
       severity: str = "info", scope: str = "training_rows") -> Finding:
    unknown = [a for a in affects if a not in AFFECTS]
    if unknown:  # a finding names decisions the plan and the ledger know, never a word of its own
        raise ValueError(f"finding {fid} names unknown decisions: {unknown}")
    return Finding(id=f"F-{_slug(fid)}", topic=topic, statement=statement[:3900], evidence=evidence,
                   implication=implication[:3900], affects=list(affects), severity=severity, scope=scope)


def _r(x: float, digits: int = 4) -> float:
    return float(round(float(x), digits))


def _names(values: Sequence[Any], n: int = 10) -> str:
    values = [str(v) for v in values]
    return ", ".join(values[:n]) + (f" and {len(values) - n} more" if len(values) > n else "")


def _texty(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _infer_task(y: pd.Series) -> TaskType:
    values = y.dropna()
    if values.nunique() == 2:
        return "binary"
    if _texty(values) or str(values.dtype) in ("category", "bool") or (
            pd.api.types.is_integer_dtype(values) and values.nunique() <= 20):
        return "multiclass"
    return "regression" if pd.api.types.is_numeric_dtype(values) else "multiclass"


POSITIVE_WORDS = {"1", "1.0", "true", "yes", "y", "fraud", "positive"}
NEGATIVE_WORDS = {"0", "0.0", "false", "no", "n", "negative", "legit", "clean"}


def _positive(y: pd.Series) -> Any:
    """The positive class: a yes/1/true paired with a no/0/false; otherwise the rarer class (labels coded 1/2 included)."""
    values = y.dropna().unique().tolist()
    words = {str(v).lower(): v for v in values}
    if len(values) == 2:
        pos = [k for k in words if k in POSITIVE_WORDS]
        neg = [k for k in words if k in NEGATIVE_WORDS]
        if len(pos) == 1 and len(neg) == 1:
            return words[pos[0]]
    return y.value_counts().index[-1]


def _time_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            out.append(c)
        elif _texty(df[c]) and TIME_NAME.search(str(c)):  # pandas 3 reads text as the string dtype, not object
            parsed = pd.to_datetime(df[c].dropna().head(200), errors="coerce", format="mixed")
            if len(parsed) and parsed.notna().mean() > 0.9:
                out.append(c)
    return out


def _as_time(s: pd.Series) -> pd.Series:
    return s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce", format="mixed")


def _decimal(s: pd.Series) -> bool:
    """A measured quantity (a fare, a coordinate) is not an identifier."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    values = pd.to_numeric(s.dropna(), errors="coerce")
    return bool(len(values) and values.notna().all() and (values % 1 != 0).any())


# ----------------------------------------------------------------------------------------- single-column power


def _halves(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(n) < 0.5


def _oof_encode(keys: pd.Series, y: pd.Series) -> pd.Series:
    """Each value's target mean learned on the other half of the rows (so an id cannot score itself)."""
    half = _halves(len(keys))
    out = pd.Series(np.nan, index=keys.index)
    for fit, apply in ((half, ~half), (~half, half)):
        means = y[fit].groupby(keys[fit]).mean()
        out[apply] = keys[apply].map(means).to_numpy()
    return out.fillna(float(y.mean()))


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _oof_modes(keys: pd.Series, y: pd.Series) -> float:
    """Accuracy of guessing each key's most common class, learned on the other half (vectorised)."""
    half = _halves(len(keys))
    hits = 0
    for fit, apply in ((half, ~half), (~half, half)):
        counts = pd.DataFrame({"k": keys[fit].to_numpy(), "y": y[fit].to_numpy()}).value_counts()
        modes = counts.reset_index().drop_duplicates("k").set_index("k")["y"]  # value_counts sorts by count, largest first
        fallback = y[fit].value_counts().index[0]
        guess = keys[apply].map(modes).fillna(fallback)
        hits += int((guess.to_numpy() == y[apply].to_numpy()).sum())
    return hits / len(y)


def _power(x: pd.Series, y: pd.Series, task: str, categorical: bool = False) -> Optional[float]:
    """How well one column alone predicts the target, out of fold, against chance: AUC (binary, folded to >= 0.5),
    accuracy gain over the majority class normalised to [0, 1] (multiclass), |Spearman| (regression)."""
    if pd.api.types.is_datetime64_any_dtype(x):
        return None  # a timestamp is judged by whether it recomputes the target (_leakage), not as a category
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if len(x) < 30 or x.nunique() < 2:
        return None
    numeric = (pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x) and not categorical
               and not ID_NAME.search(str(x.name or "")))
    if not numeric and x.nunique() > 0.9 * len(x):
        return None  # nearly unique text (free text, an id): no value repeats, so nothing can be learned out of fold
    keys = x.astype(str) if not numeric else pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop").astype(str)
    if task == "binary":
        target = (y == _positive(y)).astype(int)
        score = x.astype(float) if numeric else _oof_encode(keys, target)
        auc = _auc(score.to_numpy(dtype=float), target.to_numpy())
        return max(auc, 1 - auc)
    if task == "multiclass":
        majority = float(y.value_counts(normalize=True).iloc[0])
        if majority >= 1:
            return None
        return max(0.0, (_oof_modes(keys, y) - majority) / (1 - majority))
    score = x.astype(float) if numeric else _oof_encode(keys, y.astype(float))
    rho = pd.Series(score.to_numpy()).corr(pd.Series(y.to_numpy(dtype=float)), method="spearman")
    return None if pd.isna(rho) else abs(float(rho))


# ----------------------------------------------------------------------------------------- the checks


def _grain(df: pd.DataFrame, name: str, target: Optional[str]) -> list[Finding]:
    """Structure, no target: duplicates and keys, on all rows."""
    out = []
    features = [c for c in df.columns if c != target]
    dup = int(df.duplicated().sum())
    if dup:
        out.append(_f("duplicates", "grain", f"{dup} rows of {name} are exact copies of another row.",
                      {"duplicate_rows": dup, "rows": int(len(df)), "share": _r(dup / len(df))},
                      "Drop the copies before splitting, or keep each copy on the same side: a copy on both sides scores the "
                      "model on rows it trained on.", ["data.cleaning", "split.strategy"], "warning", "all_rows"))
    keys = []
    for c in features:
        s = df[c]
        if _decimal(s) or s.isna().mean() > 0.01:
            continue
        distinct = s.nunique() / max(len(s), 1)
        if s.is_unique or (ID_NAME.search(str(c)) and distinct >= 0.95):  # an id stays an id through a few copied rows
            keys.append(str(c))
    out.append(_f("keys", "grain", f"{name} has {len(df)} rows; {len(keys)} column(s) identify rows: {_names(keys) or 'none'}.",
                  {"rows": int(len(df)), "unique_columns": keys[:LIST_CAP]},
                  "A column that identifies rows can only memorise: keep it out of the features.",
                  ["features.identifiers"], "info", "all_rows"))
    return out


def _conflicts(train: pd.DataFrame, target: str) -> list[Finding]:
    features = [c for c in train.columns if c != target]
    if not features or len(train) > 500_000:
        return []
    g = train.groupby(features, dropna=False, sort=False)[target].nunique(dropna=True)
    conflict = int((g > 1).sum())
    if not conflict:
        return []
    return [_f("conflicting-labels", "grain", f"{conflict} groups of identical feature values carry different labels.",
               {"groups": conflict}, "Some noise is in the label itself; no model can be perfect on these rows.",
               ["metric.primary"], "info")]


def _target_findings(y: pd.Series, task: str, brief: Optional[ProblemBrief]) -> list[Finding]:
    out = []
    n = int(y.notna().sum())
    missing = int(y.isna().sum())
    if missing:
        out.append(_f("target-missing", "target", f"{missing} training rows have no label.", {"rows_without_label": missing},
                      "Rows without a label are not negatives: leave them out of training, or treat the labels as partial.",
                      ["labels.coverage"], "warning"))
    if n == 0:
        return out
    if task == "binary":
        pos = _positive(y)
        rate = float((y == pos).mean())
        rare = rate < RARE_SHARE
        out.append(_f("prevalence", "target", f"The positive class ({pos}) is {rate:.2%} of training rows.",
                      {"positive_class": str(pos), "positive_rate": _r(rate, 5), "positives": int((y == pos).sum()), "rows": n},
                      ("Rare events: accuracy is meaningless (always 'no' scores high); rank by average precision and plan a "
                       "review budget (recall in the top k)." if rare else
                       "Average precision or ROC AUC rank the cases; accuracy depends on the threshold."),
                      ["metric.primary", "threshold.policy", "split.strategy"], "warning" if rare else "info"))
        if brief is not None and brief.label_coverage.value == "partial":
            out.append(_f("partial-labels", "target", "The brief says the labels are a sample: unlabelled rows are not known negatives.",
                          {"positive_rate_lower_bound": _r(rate, 5)},
                          "Report precision as a lower bound; prefer methods that tolerate hidden positives (positive-unlabelled "
                          "learning, anomaly scores) and never call unlabelled rows clean.",
                          ["labels.coverage", "model.families", "metric.primary"], "warning"))
    elif task == "multiclass":
        counts = y.value_counts()
        share = counts / counts.sum()
        tiny = sorted(share[(share < TINY_SHARE) | (counts < TINY_ROWS)].index.astype(str).tolist())
        out.append(_f("classes", "target", f"{counts.size} classes; the largest is {share.iloc[0]:.1%} of rows, the smallest "
                      f"{share.iloc[-1]:.2%}.",
                      {"classes": int(counts.size), "largest_share": _r(share.iloc[0]), "smallest_share": _r(share.iloc[-1]),
                       "imbalance_ratio": _r(counts.iloc[0] / max(counts.iloc[-1], 1), 1)},
                      "Score with macro F1 (every class counts) beside accuracy, and stratify the folds by class.",
                      ["metric.primary", "split.strategy"]))
        if tiny:
            out.append(_f("tiny-classes", "target", f"{len(tiny)} classes have under {TINY_SHARE:.0%} of rows or {TINY_ROWS} rows: "
                          + _names(tiny) + ".",
                          {"tiny_classes": tiny[:LIST_CAP], "count": len(tiny),
                           "rows": {str(k): int(counts[k]) for k in counts.index if str(k) in tiny[:LIST_CAP]}},
                          "Too few rows to learn or to measure: consider merging them into an 'other' class or routing them by "
                          "rule, keep them in every fold (stratify), and report their recall separately.",
                          ["task.framing", "metric.primary", "split.strategy"], "warning"))
    else:
        out += _distribution(y)
    return out


def target_cap(y: Any) -> Optional[float]:
    """The value a numeric target is capped at: a spike at the maximum at least three times the next value's share,
    on a target with at least ten distinct values (a discrete scale or a constant is not capped). Read on training rows."""
    values = pd.Series(np.asarray(y, dtype=float)).dropna()
    if len(values) < 3 or values.nunique() < 10:
        return None
    top = float(values.max())
    at_top = float((values == top).mean())
    below = values[values < top]
    next_share = float((below == below.max()).mean()) * len(below) / len(values) if len(below) else 0.0
    return top if at_top > 0.02 and at_top >= 3 * max(next_share, 1e-9) else None


def _distribution(y: pd.Series) -> list[Finding]:
    values = y.dropna().astype(float)
    if len(values) < 3:
        return []
    top = float(values.max())
    at_top = float((values == top).mean())
    below = values[values < top]
    next_share = float((below == below.max()).mean()) * len(below) / len(values) if len(below) else 0.0
    capped = target_cap(values) is not None  # a spike at the maximum, not a discrete scale
    body = values[values < top] if capped else values
    skew = float(body.skew()) if len(body) > 2 else 0.0
    skew = 0.0 if not np.isfinite(skew) else skew
    q05, q50, q95 = (float(body.quantile(q)) for q in (0.05, 0.5, 0.95))
    tail_ratio = (q95 - q50) / (q50 - q05) if q50 > q05 else 1.0  # robust to a cap and to outliers, unlike skew
    long_tail = skew > 1 or tail_ratio > 1.5  # Bowley skewness above 0.2: a noticeably longer right tail
    out = [_f("distribution", "target", f"The target ranges {values.min():.4g} to {top:.4g} (median {values.median():.4g}, "
              f"skew {skew:.2f}" + (" below the cap" if capped else "") + ").",
              {"min": _r(values.min()), "median": _r(values.median()), "max": _r(top), "skew": _r(skew, 2),
               "upper_to_lower_tail": _r(tail_ratio, 2), "zero_share": _r((values == 0).mean()),
               "negative_share": _r((values < 0).mean())},
              ("A long right tail: compare a log target, and use MAE or a quantile loss rather than squared error, which the "
               "tail would dominate." if long_tail else
               "Close to symmetric: MAE reads in the target's own unit; RMSE if large errors cost far more."),
              ["metric.primary", "target.transform"], "warning" if long_tail else "info")]
    if capped:
        out.append(_f("cap", "target", f"{at_top:.1%} of rows sit exactly at the maximum ({top:.4g}): a cap, not a measurement.",
                      {"cap": _r(top), "share_at_cap": _r(at_top), "share_at_next_value": _r(next_share)},
                      "The true value above the cap is unknown (censored): squared error chases the cap; evaluate below and at "
                      "the cap separately, consider predicting 'at the cap' as its own question, and say so in the limits.",
                      ["metric.primary", "task.framing", "risk.check"], "warning"))
    return out


def _entities(df: pd.DataFrame, y: Optional[pd.Series], task: str, brief: Optional[ProblemBrief], target: Optional[str],
              proposed: Optional[Mapping[str, str]] = None) -> list[Finding]:
    named = [str(c) for c in (brief.entities.value or [])] if brief is not None and brief.entities.known else []
    proposed = dict(proposed or {})  # the data investigator's entities (18.3): checked like an id-like name, not like the person's word
    usable = [c for c in df.columns if c != target and not _decimal(df[c])]
    ruled = [c for c in usable if str(c) in named or (ID_NAME.search(str(c)) and not df[c].is_unique)]
    ruled = sorted(ruled, key=lambda c: str(c) not in named)[:6]  # the person's entities first, then id-like names
    extra = [c for c in usable if str(c) in proposed and c not in ruled and not df[c].is_unique][:6]  # own slots: never pushes a rule out
    candidates = ruled + extra
    out = []
    for column in candidates:
        sizes = df.groupby(column, dropna=True).size()
        if sizes.empty or sizes.max() < 2 or (str(column) not in named and sizes.median() < 2):
            continue  # an id that repeats only through duplicated rows is not an entity
        evidence: dict[str, Any] = {"column": str(column), "entities": int(sizes.size),
                                    "rows_per_entity_median": _r(sizes.median(), 1), "rows_per_entity_max": int(sizes.max())}
        times = _time_columns(df)
        if times:  # how many of the latest fifth of rows belong to an entity already seen: a time-and-entity split's cost
            order = np.argsort(_as_time(df[times[0]]).to_numpy(), kind="stable")
            keys = df[column].astype(str).to_numpy()[order]
            cut = int(round(0.8 * len(keys)))
            if 0 < cut < len(keys):
                evidence["later_rows_seen_before"] = _r(float(np.isin(keys[cut:], keys[:cut]).mean()))
        tells = False
        if y is not None and task in ("binary", "multiclass", "regression"):
            power = _power(df[column], y, task, categorical=True)  # the entity alone, learned on other rows
            if power is not None:
                evidence["target_from_entity_out_of_fold"] = _r(power)
                tells = power >= ENTITY_MARGIN[task]
            if task == "binary":
                hit = (y == _positive(y)).groupby(df[column]).sum()
                evidence["entities_with_positives"] = int((hit > 0).sum())
        out.append(_f(f"entity-{column}", "entities",
                      f"Rows repeat for {column}: {sizes.size} values, a median of {sizes.median():.0f} rows each.", evidence,
                      (f"Group the folds by {column}: with rows of one {column} on both sides a model can recognise it "
                       f"rather than learn the pattern" + (", and which one it is tells the target on rows it never saw." if tells else ".")),
                      ["split.strategy", "features.direction"], "warning" if tells or str(column) in named else "info"))
    return out


def _time(df: pd.DataFrame, y: Optional[pd.Series], task: str, brief: Optional[ProblemBrief], target: Optional[str]) -> list[Finding]:
    named = brief.time_column.value if brief is not None and brief.time_column.known else None
    columns = [named] if named in df.columns else _time_columns(df)
    if not columns:
        return []
    column = columns[0]
    t = _as_time(df[column])
    if t.notna().mean() < 0.8:
        return []
    evidence: dict[str, Any] = {"column": str(column), "start": str(t.min())[:19], "end": str(t.max())[:19],
                                "days": int((t.max() - t.min()).days)}
    drift = ""
    if y is not None and len(df) >= 60 and y.notna().any():
        order = t.rank(method="first")
        first, last = order <= len(order) / 3, order > 2 * len(order) / 3
        if task == "binary":
            pos = _positive(y)
            a, b = float((y[first] == pos).mean()), float((y[last] == pos).mean())
            evidence.update(positive_rate_first_third=_r(a, 5), positive_rate_last_third=_r(b, 5))
            moved = (a == 0 < b) or (a > 0 and (b / a > 1.5 or b / a < 0.67))
            drift = " The positive rate moves over time." if moved else ""
        elif task == "multiclass":
            pa, pb = y[first].value_counts(normalize=True), y[last].value_counts(normalize=True)
            tvd = 0.5 * float(pa.subtract(pb, fill_value=0).abs().sum())
            new = sorted(set(pb.index.astype(str)) - set(pa.index.astype(str)))
            evidence.update(class_shift_tvd=_r(tvd), classes_only_in_last_third=new[:LIST_CAP])
            drift = (" Classes appear late: " + _names(new) + ".") if new else (" The class mix moves." if tvd > 0.1 else "")
        else:
            ya, yb = y[first].astype(float), y[last].astype(float)
            sd = float(y.astype(float).std())
            shift = (yb.mean() - ya.mean()) / sd if sd > 0 and np.isfinite(sd) else 0.0
            shift = 0.0 if not np.isfinite(shift) else shift
            evidence.update(mean_shift_in_sd=_r(shift, 2))
            drift = " The target's level moves over time." if abs(shift) > 0.3 else ""
    return [_f("time", "time", f"Rows span {evidence['days']} days by {column}.{drift}", evidence,
               ("Rows have an order in time: validate on later rows than the model trained on (a forward split), and read "
                "drift as a reason to retrain." + (" A random split would hide what the model cannot know yet." if drift else "")),
               ["split.strategy", "risk.check"], "warning" if drift else "info")]


def _missing_power(x: pd.Series, y: pd.Series, task: str) -> Optional[float]:
    """How well *whether* a value is missing predicts the target (a value written only when the outcome did not happen)."""
    rate = float(x.isna().mean())
    if not 0.02 <= rate <= 0.98:
        return None
    return _power(x.isna().astype(int).rename(x.name), y, task, categorical=True)


def _leakage(df: pd.DataFrame, y: Optional[pd.Series], task: str, target: Optional[str],
             suspects: Optional[Mapping[str, str]] = None) -> list[Finding]:
    """Columns written after the outcome. ``suspects``: columns the data investigator reads as written after the
    prediction moment, with its reason (18.3); they are tested exactly as a telling name is, never trusted."""
    if y is None:
        return []
    suspects = dict(suspects or {})
    out = []
    times = _time_columns(df)
    easy, powers = _easy_task(df, y, task, target, suspects)
    for column in df.columns:
        if column == target:
            continue
        if str(column) in easy:  # one of several strong, unrelated, continuous columns: a question, not an accusation
            out.append(easy[str(column)])
            continue
        by_name = bool(POST_MOMENT_NAME.search(str(column))) and not HISTORY_NAME.search(str(column))
        suspected = str(column) in suspects
        named = by_name or suspected
        power = powers.get(str(column))  # computed once, in _easy_task
        absent = _missing_power(df[column], y, task)
        evidence: dict[str, Any] = {"column": str(column)}
        if power is not None:
            evidence["single_column_power"] = _r(power)
        if absent is not None:
            evidence["missing_indicator_power"] = _r(absent)
            power = absent if power is None else max(power, absent)
        if suspected:
            evidence["suspected_by"] = "the data investigator" + (" and the name" if by_name else "")
            evidence["investigator_reason"] = suspects[str(column)][:300]
        why = ("its name says it is written after the outcome" if by_name else
               f"the data investigator reads it as written after the moment ({suspects.get(str(column), '')[:120]})")
        if power is not None and power >= POWER_ALARM[task]:
            severity, says = "critical", "on its own it predicts the target almost perfectly" + (f"; {why}" if named else "")
        elif named and power is not None and power >= POWER_NAMED[task]:
            severity, says = "warning", f"{why}, and it predicts the target"
        elif named:
            out.append(_f(f"name-{column}", "leakage", f"{column}: {why}, but it predicts little.",
                          evidence, f"Ask when {column} is written; keep it only if it is known at the prediction moment.",
                          ["risk.check"], "info"))
            continue
        else:
            continue
        out.append(_f(f"leak-{column}", "leakage", f"{column}: {says}.", evidence,
                      f"Keep {column} out unless it is known at the prediction moment; ask when it is written.",
                      ["features.forbidden"], severity))
    if task == "regression":
        out += _sum_identity(df, y)
    if task == "regression" and len(times) >= 2:
        target_values = y.astype(float)
        spread = float(target_values.std()) if target_values.notna().sum() > 1 else 0.0
        for i, a in enumerate(times):
            for b in times[i + 1:]:
                gap = (_as_time(df[b]) - _as_time(df[a])).dt.total_seconds()
                for unit, scale in (("seconds", 1.0), ("minutes", 60.0), ("hours", 3600.0), ("days", 86400.0)):
                    diff = (gap / scale).abs()
                    ok = diff.notna() & target_values.notna()
                    if ok.sum() > 30 and spread > 0 and float(np.median(np.abs(diff[ok] - target_values[ok]))) < 0.05 * spread:
                        later = b if float(gap.median()) >= 0 else a
                        out.append(_f(f"leak-recompute-{a}-{b}", "leakage", f"The target equals the time between {a} and {b}, in {unit}.",
                                      {"columns": [str(a), str(b)], "later": str(later), "unit": unit},
                                      f"{later} is written when the outcome is known: keep it out.", ["features.forbidden"], "critical"))
                        break
    return out


COPY_LEVELS = 10      # a column with this few values that predicts the target almost perfectly is a coded copy
NEAR_PERFECT = {"binary": 0.99, "multiclass": 0.95, "regression": 0.99}  # beyond this a column stays a leak, whatever else
SHARED = 0.3          # |within-class Spearman| with another column: the column measures something real beyond the label
SAMPLE = 5000


def _beyond_label(x: pd.Series, others: pd.DataFrame, y: pd.Series, task: str) -> float:
    """How much of a column's variation, once the target is taken out, it shares with another column: the largest
    |Spearman| within each class (weighted), or between rank residuals after the target (regression). An honest
    measurement shares real variation with its neighbours (a tumour's radius and perimeter, even among benign ones); a
    leak is the label plus independent noise, so within a class it shares nothing."""
    frame = pd.concat([x.rename("__x"), others], axis=1)
    frame["__y"] = y.to_numpy()
    frame = frame.dropna(subset=["__x", "__y"])
    if len(frame) > SAMPLE:
        frame = frame.sample(SAMPLE, random_state=0)
    if len(frame) < 30 or others.shape[1] == 0:
        return 0.0
    if task == "regression":
        ry = frame["__y"].rank()
        resid = {}
        for c in ["__x", *others.columns]:
            r = frame[c].rank()
            ok = r.notna()
            slope = np.polyfit(ry[ok], r[ok], 1) if ok.sum() > 2 else (0.0, 0.0)
            resid[c] = r - (slope[0] * ry + slope[1])
        res = pd.DataFrame(resid)
        rho = res.drop(columns="__x").corrwith(res["__x"], method="spearman").abs()
        return float(rho.max()) if rho.notna().any() else 0.0
    best = 0.0
    for c in others.columns:
        total, weight = 0.0, 0
        for _, part in frame.groupby("__y"):
            if len(part) >= 20 and part["__x"].nunique() > 2 and part[c].nunique() > 2:
                r = part["__x"].corr(part[c], method="spearman")
                if pd.notna(r):
                    total, weight = total + abs(float(r)) * len(part), weight + len(part)
        if weight:
            best = max(best, total / weight)
    return best


def _easy_task(df: pd.DataFrame, y: pd.Series, task: str, target: Optional[str],
               suspects: Mapping[str, str]) -> tuple[dict[str, Finding], dict[str, Optional[float]]]:
    """Strong columns on a task that is simply easy. A leak is the label plus noise: it stands out alone, or beside
    other leaks, and within a class it shares nothing with any other column. When two or more continuous columns each
    predict the target almost (not perfectly) and each shares real variation with other columns within the classes,
    and nothing in their names or the investigator's reading says they are written late, each is a question for the
    person (kept out until confirmed: features.to_confirm), not an accusation. A near-perfect column, a coded copy (few
    values), a telling name, or a strong column with no such structure stays a leak. Returns the questions and every
    column's single-column power (for _leakage, which would compute them again)."""
    numeric = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    powers: dict[str, Optional[float]] = {}
    candidates = []
    for column in df.columns:
        if column == target:
            continue
        name = str(column)
        x = df[column]
        powers[name] = None if (ID_NAME.search(name) and x.is_unique) else _power(x, y, task)
        if str(column) in suspects or (POST_MOMENT_NAME.search(name) and not HISTORY_NAME.search(name)) or ID_NAME.search(name):
            continue
        if column not in numeric or x.nunique() <= COPY_LEVELS:
            continue
        power = powers[name]
        if power is not None and POWER_ALARM[task] <= power < NEAR_PERFECT[task]:
            candidates.append((column, power))
    if len(candidates) < 2:
        return {}, powers
    # the structure must be shared with a column that is not itself strong: a leak's noisy copy shares the leak's noise
    weak = [c for c in numeric if (powers.get(str(c)) or 0.0) < POWER_ALARM[task]]
    strong = []
    for column, power in candidates:
        shared = _beyond_label(df[column], df[[c for c in weak if c != column]], y, task)
        if shared >= SHARED:
            strong.append((column, power, shared))
    if len(strong) < 2:
        return {}, powers
    out = {}
    for column, power, shared in strong:
        out[str(column)] = _f(f"confirm-{column}", "leakage",
                              f"{column} predicts the target almost perfectly on its own (power {power:.3f}), and so do "
                              f"{len(strong) - 1} other column(s); each also moves with other columns within the classes "
                              f"(|Spearman| {shared:.2f}), as real measurements do and leaks do not: the task may simply be easy.",
                              {"column": str(column), "single_column_power": _r(power), "basis": "strong, with real structure beyond the label",
                               "shared_within_class": _r(shared), "strong_columns": [str(c) for c, _, _ in strong][:20]},
                              f"Confirm that {column} is known at the prediction moment; it stays out of the model until a person says so.",
                              ["features.to_confirm"], "info")
    return out, powers


MAX_SUM_COLUMNS = 40


SUM_SAMPLE = 50_000


def _sum_identity(df: pd.DataFrame, y: pd.Series) -> list[Finding]:
    """Two numeric columns that add up to the target: its parts, known only once the target is. A column that already
    equals the target, or one that is mostly zero, makes no pair (a final amount plus a mostly-zero tip is not a sum).
    Screened on a seeded sample of rows, confirmed on all."""
    target = y.astype(float)
    spread = float(target.std()) if target.notna().sum() > 1 else 0.0
    if spread <= 0:
        return []
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
               and df[c].nunique() > 2][:MAX_SUM_COLUMNS]
    if len(numeric) < 2:
        return []
    values = df[numeric].astype(float).to_numpy()
    t = target.to_numpy()
    rows = np.arange(len(t))
    if len(rows) > SUM_SAMPLE:
        rows = np.sort(np.random.default_rng(0).choice(rows, SUM_SAMPLE, replace=False))
    tol = 0.005 * spread

    def close(total: np.ndarray, ys: np.ndarray) -> bool:
        ok = ~np.isnan(total) & ~np.isnan(ys)
        return int(ok.sum()) > 30 and float(np.median(np.abs(total[ok] - ys[ok]))) < tol

    usable = [j for j in range(len(numeric))
              if not close(values[rows, j], t[rows]) and float(np.nanmedian(np.abs(values[rows, j]))) >= tol]
    out = []
    for x, i in enumerate(usable):
        for j in usable[x + 1:]:
            a, b = numeric[i], numeric[j]
            if close(values[rows, i] + values[rows, j], t[rows]) and close(values[:, i] + values[:, j], t):
                out.append(_f(f"leak-sum-{a}-{b}", "leakage", f"{a} and {b} add up to the target.",
                              {"columns": [str(a), str(b)]}, f"{a} and {b} are the target's parts: keep both out.",
                              ["features.forbidden"], "critical"))
    return out[:5]


def _missing(df: pd.DataFrame, y: Optional[pd.Series], task: str, target: Optional[str]) -> list[Finding]:
    out = []
    for column in df.columns:
        if column == target:
            continue
        miss = df[column].isna()
        rate = float(miss.mean())
        if rate == 0 or rate == 1:
            continue
        evidence: dict[str, Any] = {"column": str(column), "missing_rate": _r(rate)}
        speaks = False
        if y is not None and miss.sum() >= 10 and (~miss).sum() >= 10:
            if task == "regression":
                sd = float(y.astype(float).std())
                d = (y[miss].astype(float).mean() - y[~miss].astype(float).mean()) / sd if sd > 0 else 0.0
                d = 0.0 if not np.isfinite(d) else d
                evidence["target_shift_when_missing_in_sd"] = _r(d, 2)
                speaks = abs(d) > 0.2
            elif task == "binary":
                pos = _positive(y)
                a, b = float((y[miss] == pos).mean()), float((y[~miss] == pos).mean())
                evidence.update(positive_rate_when_missing=_r(a, 5), positive_rate_when_present=_r(b, 5))
                speaks = (b == 0 < a) or (b > 0 and (a / b > 1.5 or a / b < 0.67))
            elif task == "multiclass":
                pa, pb = y[miss].value_counts(normalize=True), y[~miss].value_counts(normalize=True)
                tvd = 0.5 * float(pa.subtract(pb, fill_value=0).abs().sum())
                evidence["class_mix_shift_tvd"] = _r(tvd)
                speaks = tvd > 0.1
        if speaks or rate > 0.3:
            out.append(_f(f"missing-{column}", "missingness",
                          f"{column} is missing in {rate:.1%} of rows" + (", and its absence predicts the target." if speaks else "."),
                          evidence,
                          ("Missingness speaks: keep a missing indicator, and check whether the value is missing because of the "
                           "outcome (then it is leakage) or before it (then it is signal)." if speaks else
                           "Impute inside the training folds and keep a missing indicator."),
                          ["features.direction", "features.forbidden"] if speaks else ["features.direction"], "warning" if speaks else "info"))
    return out


def _columns(df: pd.DataFrame, target: Optional[str]) -> list[Finding]:
    out = []
    constant = [str(c) for c in df.columns if c != target and df[c].nunique(dropna=False) <= 1]
    if constant:
        out.append(_f("constant", "columns", f"{len(constant)} column(s) hold a single value: {_names(constant)}.",
                      {"columns": constant[:LIST_CAP]}, "Drop them: they carry nothing.", ["features.direction"], "info", "all_rows"))
    for c in df.columns:
        if c == target or not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        if re.search(r"(distance|age|price|amount|count|quantity|duration|minutes|weight|items|value)", str(c), re.I):
            negative = int((df[c] < 0).sum())
            if negative:
                out.append(_f(f"impossible-{c}", "quality", f"{negative} rows have a negative {c}, which cannot be.",
                              {"column": str(c), "rows": negative},
                              f"Treat negative {c} as missing (or fix the sign if the source is known) inside the pipeline, and ask "
                              "where the rows come from.", ["data.cleaning"], "warning", "all_rows"))
    return out


# ----------------------------------------------------------------------------------------- a label file


KEYLIKE_NAME = re.compile(r"(^id$|_id$|_ref$|^id_|_key$|_uuid$|phone|email|_number$|_code$)", re.I)


ROW_NUMBER_NAME = re.compile(r"^(unnamed: ?\d+|index|level_\d+|row|row_?(id|no|num|number))$", re.I)


def _row_number(s: pd.Series) -> bool:
    """A column that only numbers the rows of its file: a pandas index saved with the CSV ("Unnamed: 0", "index").
    Two files' row numbers say nothing about each other. (By name only: an order id may well run 0, 1, 2 ...)"""
    return bool(ROW_NUMBER_NAME.match(str(s.name or "")))


def _key_candidates(main: pd.DataFrame, labels: pd.DataFrame) -> list[tuple[str, ...]]:
    """Columns that could identify a case: an id-like name, or nearly distinct within the label file; and pairs of
    id-like columns (a coarse pair such as (customer, courier) is the classic way to inflate labels)."""
    shared = [c for c in labels.columns if c in main.columns and labels[c].notna().any()
              and not _row_number(main[c]) and not _row_number(labels[c])]  # a saved index is a row number, never a key
    single = [c for c in shared if not _decimal(labels[c]) and (KEYLIKE_NAME.search(str(c)) or labels[c].nunique() >= 0.8 * labels[c].notna().sum())]
    idlike = [c for c in single if ID_NAME.search(str(c))]
    pairs = [(a, b) for i, a in enumerate(idlike) for b in idlike[i + 1:]]
    return [(c,) for c in single][:20] + pairs[:20]


def _key_text(s: pd.Series) -> pd.Series:
    """Keys as text, with 1305684.0 and 1305684 the same (an id read as a float in one file, an integer in the other).
    Done on the text, never through a float, so long ids keep every digit."""
    if pd.api.types.is_float_dtype(s):
        whole = s.notna() & (s % 1 == 0) & (s.abs() < 2 ** 53)
        out = s.astype(str)
        out[whole] = s[whole].astype("int64").astype(str)
        return out
    return s.astype(str).str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)


def label_join(main: pd.DataFrame, labels: pd.DataFrame) -> list[dict[str, Any]]:
    """For each key that could identify a case (one column or a pair): how many label rows it finds and how many rows
    of the main table one label row would mark. Package 15.2 builds the studio's label review on it."""
    out = []
    for key in _key_candidates(main, labels):
        cols = list(key)
        label_keys = labels[cols].dropna().apply(_key_text)
        if label_keys.empty:
            continue
        main_keys = main[cols].dropna().apply(_key_text)
        joined = pd.MultiIndex.from_frame(main_keys) if len(cols) > 1 else pd.Index(main_keys[cols[0]])
        wanted = pd.MultiIndex.from_frame(label_keys) if len(cols) > 1 else pd.Index(label_keys[cols[0]])
        marked = int(joined.isin(wanted.unique()).sum())
        found = int(wanted.isin(joined.unique()).sum())   # label rows whose key is in the table
        out.append({"key": "+".join(map(str, cols)), "label_rows": int(len(label_keys)), "label_rows_found": found,
                    "rows_marked": marked, "rows_per_label": _r(marked / max(found, 1), 2)})
    return sorted(out, key=lambda r: (r["rows_per_label"] > 1.5, -r["label_rows_found"], r["rows_per_label"]))


def _disagreeing(main: pd.DataFrame, labels: pd.DataFrame, key: str) -> dict[str, float]:
    """For a single-column key: the shared columns whose values differ between the files on the matched rows."""
    if "+" in key:
        return {}
    left = labels.dropna(subset=[key]).assign(**{key: lambda d: _key_text(d[key])})
    right = main.dropna(subset=[key]).assign(**{key: lambda d: _key_text(d[key])}).drop_duplicates(subset=[key])
    matched = left.merge(right, on=key, suffixes=("__labels", "__main"))
    out = {}
    for column in [c for c in labels.columns if c in main.columns and c != key]:
        a, b = matched.get(f"{column}__labels"), matched.get(f"{column}__main")
        if a is None or b is None or len(matched) < 5:
            continue
        both = a.notna() & b.notna()
        if both.sum() < 5:
            continue
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            same = np.isclose(a[both].astype(float), b[both].astype(float), rtol=1e-6, atol=1e-9)
        else:
            same = a[both].astype(str).to_numpy() == b[both].astype(str).to_numpy()
        agree = float(np.mean(same))
        if agree < 0.9:
            out[str(column)[:100]] = _r(agree, 3)
    return out


def _labels_unmatched(table: pd.DataFrame, name: str, labels: pd.DataFrame, label_name: str) -> list[Finding]:
    """Label rows whose key matches no row of the whole table: key membership, a fact of structure (no target is read)."""
    joins = label_join(table, labels)
    if not joins:
        return []
    best = joins[0]
    missing = int(best["label_rows"] - best["label_rows_found"])
    if missing <= 0:
        return []
    return [_f("labels-unmatched", "labels", f"{missing} of {best['label_rows']} rows of {label_name} match no row of {name} on "
               f"{best['key']}.", {"key": best["key"], "label_rows": int(best["label_rows"]), "unmatched": missing},
               "Keep them apart: they cannot be scored, but they can check the model from outside; ask why they are missing.",
               ["labels.join", "risk.check"], "warning" if missing > 0.1 * best["label_rows"] else "info", "all_rows")]


def _labels(main: pd.DataFrame, name: str, labels: pd.DataFrame, label_name: str) -> list[Finding]:
    joins = label_join(main, labels)
    if not joins:
        return [_f("labels-no-key", "labels", f"{label_name} shares no column with {name} that could identify a case.",
                   {"label_rows": int(len(labels))}, "The labels cannot be joined; ask which key identifies a case.",
                   ["labels.join"], "critical", "label_file")]
    best = joins[0]
    out = [_f("labels-join", "labels", f"{label_name} joins {name} best on {best['key']}: {best['label_rows_found']} of "
              f"{best['label_rows']} label rows found among the {len(main)} training rows, {best['rows_marked']} of them marked.",
              {"key": best["key"], "training_rows": int(len(main)),
               "joins": {j["key"][:100]: {k: v for k, v in j.items() if k != "key"} for j in joins[:12]}},
              f"Join on {best['key']}. The label rows not among the training rows are either held out (scored once, at the end) "
              f"or not in {name} at all (kept apart: they can check the model from outside).",
              ["labels.join", "labels.coverage"], "info", "label_file")]
    for j in joins[1:]:
        if j["rows_per_label"] > 1.5 and j["label_rows_found"] > 0:
            out.append(_f(f"labels-inflation-{j['key'].replace('+', '-')}", "labels",
                          f"Joining on {j['key']} would mark {j['rows_marked']} rows from {j['label_rows_found']} labels "
                          f"({j['rows_per_label']} rows per label).", dict(j),
                          f"Never join on {j['key']}: it labels every row that shares the value, most of which are not the cases.",
                          ["labels.join"], "critical", "label_file"))
    differ = _disagreeing(main, labels, best["key"])
    if differ:
        shown = dict(list(differ.items())[:LIST_CAP])
        out.append(_f("labels-disagree", "labels", f"After joining on {best['key']}, {len(differ)} column(s) with the same name hold "
                      "different values in the two files: " + _names(list(differ)) + ".", {"share_equal": shown, "count": len(differ)},
                      "A shared name does not mean a shared meaning: never take these columns from the label file, ask what "
                      "each one means there, and state it as a limit (the label file may know something the table does not).",
                      ["labels.join", "risk.check"], "warning", "label_file"))
    return out


# ----------------------------------------------------------------------------------------- entry point


# ------------------------------------------------------------------------------------------------ the data investigator

MAX_INVESTIGATED = 150  # three lists of 50 lines: the advisor's evidence limits


def _kind(x: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(x):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(x) or (TIME_NAME.search(str(x.name or "")) and not pd.api.types.is_numeric_dtype(x)
                                                   and _as_time(x).notna().mean() > 0.9):
        return "datetime"
    if pd.api.types.is_numeric_dtype(x):
        return "integer" if pd.api.types.is_integer_dtype(x) else "number"
    return "text" if x.astype(str).str.len().mean() > 40 else "category"


def _investigator_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    df = inputs["features"]
    lines = [f"{str(c)[:200]} | {_kind(df[c])} | missing {float(df[c].isna().mean()):.2f} | "
             f"distinct {float(df[c].nunique()) / max(len(df), 1):.2f}" for c in list(df.columns)[:MAX_INVESTIGATED]]
    return {"prediction_moment": str(inputs.get("moment") or "not stated")[:600], "target": str(inputs["target"])[:200],
            "task": inputs["task"], "rows": int(len(df)), "columns_total": int(df.shape[1]), "columns_shown": len(lines),
            # one line per column ("name | kind | missing 0.12 | distinct 0.98"), in lists of 50: a summary, never a table
            "columns": {f"{i + 1}-{i + len(chunk)}": chunk for i, chunk in
                        ((i, lines[i:i + 50]) for i in range(0, len(lines), 50))}}


def _investigator_parse(answer: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Keep proposals about columns that exist (never the target), each with a short reason."""
    columns = {str(c) for c in inputs["features"].columns} - {str(inputs["target"])}
    out: dict[str, dict[str, str]] = {"after": {}, "entities": {}}
    for key in ("after", "entities"):
        items = answer.get(key) or []
        if not isinstance(items, list):
            raise ValueError(f"{key} must be a list")
        for item in items[:50]:
            if not isinstance(item, Mapping):
                continue
            column, why = item.get("column"), item.get("why")
            if isinstance(column, str) and column in columns and isinstance(why, str) and why.strip():
                out[key][column] = " ".join(why.split())[:300]
    return out


DATA_INVESTIGATOR = Role(
    name="data_investigator",
    description="Reads the column names and kinds against the prediction moment; proposes columns written after it and "
                "entities. Every proposal is a check analyse runs, never a finding by itself.",
    evidence=_investigator_evidence, parse=_investigator_parse, fallback=lambda inputs: {"after": {}, "entities": {}})


def _investigate(advisor: Advisor, features: pd.DataFrame, train: pd.DataFrame, target: Optional[str], task: str,
                 moment: Optional[str], findings: list[Finding]) -> dict[str, dict[str, str]]:
    return run_role(DATA_INVESTIGATOR, advisor, {"features": features, "target": target, "task": task, "moment": moment}).output


def _places(df: pd.DataFrame, y: Optional[pd.Series], task: str, target: Optional[str]) -> list[Finding]:
    """Places that cluster (16.10): when the rows carry coordinates and the target is alike for nearby rows (Moran's I
    over each row's nearest neighbours, permutation p-value), a random split scores how well a model remembers an area.
    The block size follows the distance within which targets stay alike; how much a random split flatters a model is
    measured on the training rows with random folds against folds of whole blocks."""
    from dclab_core import spatial
    from dclab_core.behaviour import coordinate_pairs

    pairs = coordinate_pairs([c for c in df.columns if c != target])
    if not pairs or y is None or len(df) < 200 or task not in ("binary", "regression"):
        return []
    _, lat, lon = pairs[0]
    la, lo = pd.to_numeric(df[lat], errors="coerce"), pd.to_numeric(df[lon], errors="coerce")
    ok = (la.notna() & lo.notna() & y.notna()).to_numpy()
    if ok.mean() < 0.9:
        return []
    if ((la.abs() > 90) | (lo.abs() > 180)).mean() > 0.01:  # metres or another projection: blocks in km cannot be read from it
        return [_f("places", "places", f"{lat} and {lon} are not latitude and longitude in degrees (values beyond ±90 or ±180).",
                   {"lat_column": str(lat), "lon_column": str(lon), "degrees": False},
                   "Places cannot be split into blocks from these columns; give coordinates in degrees to test areas apart.",
                   ["split.strategy"], "info")]
    values = ((y == _positive(y)).astype(float) if task == "binary" else pd.to_numeric(y, errors="coerce").rank()).to_numpy()[ok]
    moran = spatial.morans_i(values, la.to_numpy()[ok], lo.to_numpy()[ok], permutations=99)
    if moran["i"] is None:
        return []
    reach = spatial.correlation_range(values, la.to_numpy()[ok], lo.to_numpy()[ok])
    span = float(reach["span_km"] or 0.0)
    block = float(np.clip(reach["range_km"] or span, span / 40 if span else 0.1, span / 4 if span else 1.0))  # at least four blocks a side
    block = max(round(block * 2) / 2, 0.5)
    strong = moran["i"] >= 0.1 and (moran["p_value"] or 1) <= 0.01
    evidence: dict[str, Any] = {"lat_column": str(lat), "lon_column": str(lon), "moran_i": moran["i"], "p_value": moran["p_value"],
                                "neighbours": moran["k"], "range_km": reach["range_km"], "span_km": reach["span_km"], "block_km": _r(block, 2)}
    if strong:
        numeric = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
        rows = np.flatnonzero(ok)[:5000]
        try:
            gap = spatial.split_optimism(df.iloc[rows][numeric], np.asarray(values)[: len(rows)], la.to_numpy()[rows], lo.to_numpy()[rows],
                                         block_km=block, task="binary" if task == "binary" else "regression")
            evidence.update(random_folds=gap["random"], block_folds=gap["blocked"], optimism=gap["optimism"], score=gap["metric"], blocks=gap["blocks"])
        except ValueError:
            pass
    statement = (f"Nearby rows have alike targets: Moran's I {moran['i']} over each row's {moran['k']} nearest neighbours (p {moran['p_value']}), "
                 f"alike within about {reach['range_km']} km." if strong else
                 f"The rows carry coordinates ({lat}, {lon}); nearby targets are not clearly alike (Moran's I {moran['i']}, p {moran['p_value']}).")
    if "optimism" in evidence:
        statement += f" Random folds score {evidence['random_folds']} {evidence['score']}, folds of whole {block:g} km blocks {evidence['block_folds']}."
    implication = (f"Split by blocks of about {block:g} km, so the test is on areas the model has not seen; a random split would score how "
                   "well it remembers an area." if strong else "A split by area is not needed for this target.")
    return [_f("places", "places", statement, evidence, implication, ["split.strategy"], "warning" if strong else "info")]


# ---------------------------------------------------------------------------------------------- 20.5: by family

NO_LABEL = ("anomaly", "streams")  # families read without an outcome; rare_event joins them when its labels are partial or absent
ROBUST_Z = 3.5
MAX_NORMAL = 12


def _numbers(train: pd.DataFrame, skip: Sequence[Optional[str]]) -> list[str]:
    """Numeric columns a detector would read: not the target or a named column, not an identifier, not constant."""
    out = []
    for c in train.columns:
        if c in skip or ID_NAME.search(str(c)) or not pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c]):
            continue
        x = pd.to_numeric(train[c], errors="coerce")
        if x.notna().sum() >= 20 and x.nunique() > 2:
            out.append(c)
    return out[:40]


def _robust(x: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    med = float(x.median()) if len(x) else 0.0
    mad = float((x - med).abs().median()) * 1.4826 if len(x) else 0.0
    return med, (mad if mad > 0 else float(x.std() or 0.0) or 1.0)


def _stamps(train: pd.DataFrame, time: Optional[str]) -> Optional[pd.Series]:
    """A time column read as dates, or None when it is not one: numbers are never read as nanoseconds since 1970, and a
    span under a day says nothing per day."""
    if not time or time not in train.columns:
        return None
    raw = train[time]
    if pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_datetime64_any_dtype(raw):
        return None
    stamps = _as_time(raw)
    if stamps.notna().mean() < 0.9 or (stamps.max() - stamps.min()) < pd.Timedelta(days=1):
        return None
    return stamps


def _normal(train: pd.DataFrame, cols: list[str]) -> list[Finding]:
    """The shape of normal: each column's robust centre and spread, its skew and how much of it sits at zero."""
    if not cols:
        return []
    shape = {}
    for c in cols[:MAX_NORMAL]:
        x = pd.to_numeric(train[c], errors="coerce")
        med, spread = _robust(x)
        shape[str(c)[:60]] = {"median": _r(med, 3), "spread": _r(spread, 3), "skew": _r(float(x.skew() or 0.0), 2),
                              "zero_share": _r(float((x == 0).mean()), 3), "missing_share": _r(float(x.isna().mean()), 3)}
    skewed = [c for c, v in shape.items() if abs(v["skew"]) > 2]
    return [_f("normal", "normal", f"What is normal on {len(shape)} numeric columns of the training rows: their robust centre and "
               f"spread{(' (' + ', '.join(skewed[:5]) + ' strongly skewed)') if skewed else ''}.", {"columns": shape},
               "A detector scores distance from this normal: scale each column by its robust spread (a skewed one on a log "
               "scale), never by values of rows it will score later.", ["features.direction", "target.transform"])]


def _rules(train: pd.DataFrame, cols: list[str], entity: Optional[str]) -> list[Finding]:
    """The rows plain rules would flag (a value beyond 3.5 robust deviations), and whether a few entities hold them."""
    if not cols:
        return []
    hits = pd.Series(False, index=train.index)
    per = {}
    for c in cols:
        x = pd.to_numeric(train[c], errors="coerce")
        med, spread = _robust(x)
        h = ((x - med).abs() / spread) > ROBUST_Z
        per[str(c)[:60]] = _r(float(h.mean()), 4)
        hits |= h.fillna(False)
    share = float(hits.mean())
    evidence: dict[str, Any] = {"rows": int(len(train)), "rows_flagged": int(hits.sum()), "share": _r(share, 4),
                                "by_column": dict(sorted(per.items(), key=lambda kv: -kv[1])[:15])}
    held = None
    if entity and entity in train.columns and hits.any():
        by = train.loc[hits, entity].dropna().value_counts()
        if by.sum() > 0:
            top = max(1, int(np.ceil(0.05 * train[entity].nunique())))
            held = float(by.iloc[:top].sum() / by.sum())
            evidence["held_by_top_5pct_entities"] = _r(held, 3)
    many = share > 0.10
    statement = (f"{share:.1%} of the training rows sit beyond {ROBUST_Z} robust deviations on at least one column"
                 + (f"; the top 5% of {entity} values hold {held:.0%} of them" if held is not None else "") + ".")
    implication = ("Rules alone would flag too many rows for a review budget: rank rows instead, and keep the rules as features."
                   if many else "Rules give a first ranking to beat: a detector earns its place only by finding what they miss.")
    return [_f("rules", "rules", statement, evidence, implication, ["threshold.policy", "model.families"], "warning" if many else "info")]


def _behaviour(train: pd.DataFrame, entity: Optional[str], time: Optional[str]) -> list[Finding]:
    """Each entity's activity: rows per entity, and how many entities sit far from the others."""
    if not entity or entity not in train.columns:
        return []
    counts = train[entity].value_counts()
    if len(counts) < 5:
        return []
    med, spread = _robust(counts.astype(float))
    far = float((((counts - med) / spread) > ROBUST_Z).mean())
    evidence: dict[str, Any] = {"column": str(entity)[:120], "entities": int(len(counts)), "rows_per_entity_median": _r(float(counts.median()), 2),
                                "rows_per_entity_p90": _r(float(counts.quantile(0.9)), 2), "share_of_entities_far_busier": _r(far, 4),
                                "rows_held_by_top_1pct": _r(float(counts.iloc[:max(1, len(counts) // 100)].sum() / counts.sum()), 4)}
    stamps = _stamps(train, time)
    if stamps is not None:
        per_day = train.assign(_d=stamps.dt.floor("D")).groupby([entity, "_d"]).size()
        if len(per_day):
            evidence["rows_per_entity_day_p99"] = _r(float(per_day.quantile(0.99)), 2)
    warn = far > 0.01
    return [_f(f"behaviour-{entity}", "behaviour", f"{len(counts)} {entity} values: a median of {counts.median():g} rows each"
               + (f"; {far:.1%} of them far busier than the rest" if warn else "") + ".", evidence,
               f"Score each row against its {entity}'s own history as well as against everyone (counts per day, the row against "
               f"its {entity}'s usual): an anomaly is often normal for the table and odd for its {entity}. Keep a {entity}'s rows "
               "together when testing.", ["features.entity_history", "split.strategy", "model.families"], "warning" if warn else "info")]


def _detectable(train: pd.DataFrame, cols: list[str], seed: int = 0) -> list[Finding]:
    """A planted-anomaly check: plant two generic shapes in a copy of the training rows (one value far from normal; a
    pair of columns that move together, broken) and measure what a detector on these columns finds at a review budget
    of 1% of the rows. It says what a detector can see in these columns, not where the real anomalies are."""
    if len(cols) < 2 or len(train) < 200:
        return []
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    X = train[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    for c in cols:  # a skewed count or amount is read on a log scale, as the shape of normal says to
        if X[c].min() >= 0 and float(X[c].skew() or 0) > 2:
            X[c] = np.log1p(X[c])
    scale = {c: _robust(X[c]) for c in cols}
    Z = pd.DataFrame({c: (X[c] - scale[c][0]) / scale[c][1] for c in cols}, index=X.index)
    k = max(10, int(0.01 * len(Z)))  # planted rows, and the review budget: recall at k can reach 1
    shapes: dict[str, Any] = {}
    signed = Z.corr(method="spearman").where(~np.eye(len(cols), dtype=bool)).stack()
    pair = signed.abs().idxmax() if len(signed) and signed.abs().max() >= 0.5 else None
    for shape in ("far_value", "broken_pair"):
        if shape == "broken_pair" and pair is None:
            continue
        planted = Z.copy()
        rows = rng.choice(len(Z), size=k, replace=False)
        if shape == "far_value":  # beyond every training value of the column, on the side the column can take
            for r in rows:
                c = cols[int(rng.integers(len(cols)))]
                upward = X[c].min() >= 0 or bool(rng.integers(2))
                far = 8.0 + float(rng.uniform(0, 8))  # each its own distance: identical outliers mask each other
                value = Z[c].max() + far if upward else Z[c].min() - far
                planted.iloc[r, planted.columns.get_loc(c)] = value
        else:  # each value typical, together unusual: against the pair's own direction
            a, b = pair
            positive = signed[pair] > 0
            planted.iloc[rows, planted.columns.get_loc(a)] = planted[a].quantile(0.1)
            planted.iloc[rows, planted.columns.get_loc(b)] = planted[b].quantile(0.9 if positive else 0.1)
        truth = np.zeros(len(Z), dtype=int)
        truth[rows] = 1
        forest = IsolationForest(n_estimators=150, max_samples=min(1024, len(planted)), random_state=seed)  # 256 rows hide a 1% shape
        score = -forest.fit(planted).score_samples(planted)
        top = np.argsort(-score)[:k]
        shapes[shape] = {"planted": k, "recall_at_budget": _r(float(truth[top].sum() / k), 3),
                         "average_precision": _r(float(average_precision_score(truth, score)), 3), "chance": _r(k / len(Z), 4)}
        if shape == "broken_pair":
            shapes[shape]["columns"] = f"{str(pair[0])[:60]} with {str(pair[1])[:60]}"
    missed = [s for s, v in shapes.items() if v["recall_at_budget"] < 0.5]
    statement = ("On these columns a detector " + "; ".join(f"finds {v['recall_at_budget']:.0%} of {v['planted']} planted {s.replace('_', ' ')} "
                                                          f"rows among its top {v['planted']}" for s, v in shapes.items()) + ".")
    implication = ("A shape it misses on these columns needs features that make it visible (per-entity history, the relation "
                   "between the columns); evaluate any detector on planted anomalies of the shapes found."
                   if missed else "A detector on these columns can find both shapes; the real anomalies may take other shapes, so "
                   "keep a review loop whose confirmed cases become labels.")
    return [_f("detectable", "detectability", statement, {"shapes": shapes, "detector": "isolation forest (150 trees of up to 1024 rows) on "
                                                           "robust-scaled columns, skewed ones on a log scale", "budget": "as many rows as were planted (1%, at least 10)"},
               implication,
               ["model.families", "features.direction", "risk.check"], "warning" if missed else "info")]


def _censoring(train: pd.DataFrame, duration: Optional[str], event: Optional[str]) -> list[Finding]:
    if not duration or not event or duration not in train.columns or event not in train.columns:
        return []
    from dclab_core.survival import censoring_check, duration_check

    out = []
    raw = train[event]
    ended = raw.astype(float) if (pd.api.types.is_numeric_dtype(raw) or pd.api.types.is_bool_dtype(raw)) else None
    if ended is None or ended.dropna().nunique() > 2:
        return [_f("censoring", "censoring", f"The event flag {event} cannot be read as ended or not (it is not a 0/1 flag).",
                   {"values": int(raw.nunique())}, "Ask which value means the event happened, and code it as 1, the rest 0.",
                   ["task.framing"], "warning")]
    frame = train.assign(**{event: ended}).dropna(subset=[event, duration])
    flag = frame[event] != 0
    ok = censoring_check(flag.astype(int))
    censored = float(1 - flag.mean()) if len(flag) else 0.0
    follow = pd.to_numeric(frame[duration], errors="coerce")
    readable = bool(ok.get("ok"))
    statement = (f"{censored:.0%} of the training rows had not ended when the data was taken: their durations are lower bounds."
                 if readable else f"The durations cannot be read honestly: {ok.get('why')}.")
    out.append(_f("censoring", "censoring", statement,
                  {"censored_share": _r(censored, 4), "rows": int(len(frame)), "readable": readable,
                   "follow_up_median": _r(float(follow.median()), 2), "follow_up_p90": _r(float(follow.quantile(0.9)), 2)},
                  ("Model the time to the event with censoring (Kaplan-Meier, Cox, a discrete hazard): a regression on the durations, or "
                   "dropping the rows still going, biases every horizon. Score only horizons the follow-up reaches." if readable else
                   "Ask whether rows still going were left out of the export or marked as ended: without them no horizon can be scored."),
                  ["task.framing", "metric.primary", "split.strategy"], "warning" if (censored > 0.05 or not readable) else "info"))
    if not readable:
        return out
    train = frame.assign(**{event: flag.astype(int)})
    ids = [c for c in train.columns if ID_NAME.search(str(c))]  # an identifier grows with the start date: no measure of follow-up
    past = duration_check(train, duration=duration, event=event, exclude=ids)
    for c in (past.get("past_the_start") or [])[:10]:
        out.append(_f(f"counted-{c}", "leakage", f"{c} moves with the duration both among rows still going and rows that ended: it was "
                      "counted during the follow-up, after the start.", {"column": str(c)[:120], "why": str((past.get("why") or {}).get(c, ""))[:300]},
                      f"Keep {c} out: at the start it was not known.", ["features.forbidden"], "warning"))
    return out


def _series(train: pd.DataFrame, target: Optional[str], series: Optional[str], time: Optional[str]) -> list[Finding]:
    if not time or time not in train.columns or not target or target not in train.columns:
        return []
    stamps = _stamps(train, time)
    if stamps is None:
        return []
    keys = train[series] if series and series in train.columns else pd.Series("all", index=train.index)
    frame = pd.DataFrame({"s": keys.values, "t": stamps.values, "y": pd.to_numeric(train[target], errors="coerce").values}).dropna()
    if frame.empty:
        return []
    lengths = frame.groupby("s")["t"].nunique()
    gap = frame.sort_values("t").groupby("s")["t"].diff().dt.total_seconds().median()
    step = "day" if gap and abs(gap - 86400) < 3600 else "week" if gap and abs(gap - 7 * 86400) < 86400 else "month" if gap and 27 * 86400 <= gap <= 32 * 86400 else "other"
    candidates = {"day": (7, 365), "week": (52, 4), "month": (12, 3), "other": (7, 12)}[step]
    acf: dict[str, float] = {}
    for lag in candidates:
        values = []
        for _, g in frame.sort_values("t").groupby("s"):
            y = g["y"].to_numpy()
            if len(y) > 2 * lag + 2 and np.std(y[:-lag]) > 0 and np.std(y[lag:]) > 0:
                values.append(float(np.corrcoef(y[:-lag], y[lag:])[0, 1]))
        values = [v for v in values if np.isfinite(v)]  # an intermittent series has no correlation to give
        if values:
            acf[str(lag)] = _r(float(np.mean(values)), 3)
    season = max(acf, key=acf.get) if acf and max(acf.values()) >= 0.3 else None
    short = float((lengths < 2 * (int(season) if season else 7)).mean())
    out = [_f("series", "series", f"{len(lengths)} series of a median {lengths.median():g} time steps ({step}s); {short:.0%} are short.",
              {"series": int(len(lengths)), "steps_median": _r(float(lengths.median()), 1), "step": step, "short_share": _r(short, 3)},
              "Backtest from rolling origins and keep a series' later steps after its earlier ones; short series lean on a model "
              "fitted across series, and every model must beat the seasonal naive forecast.", ["split.strategy", "model.families", "baseline"],
              "warning" if short > 0.2 else "info")]
    out.append(_f("season", "time", (f"The target repeats every {season} {step}s (autocorrelation {acf[season]})." if season
                                     else "No strong season among the usual periods."), {"autocorrelation_by_lag": acf, "step": step},
                  f"Use a seasonal naive baseline of {season} {step}s and season features." if season else
                  "A plain naive forecast is the baseline to beat.", ["baseline", "features.direction"]))
    return out


def _arms(train: pd.DataFrame, target: Optional[str], treatment: Optional[str]) -> list[Finding]:
    if not treatment or treatment not in train.columns:
        return []
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score

    t = train[treatment].astype(str)
    arms = t.value_counts(normalize=True)
    cols = [c for c in _numbers(train, [target, treatment])][:20]
    auc = None
    if len(arms) == 2 and cols:
        X = train[cols].apply(pd.to_numeric, errors="coerce")
        X = (X.fillna(X.median()) - X.median()) / X.std().replace(0, 1)
        y = (t == arms.index[0]).astype(int).to_numpy()
        p = cross_val_predict(LogisticRegression(max_iter=500), X, y, cv=3, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, p))
        auc = max(auc, 1 - auc)
    randomised = auc is not None and auc < 0.6
    shown = {str(k)[:40]: _r(v, 4) for k, v in arms.iloc[:10].items()}
    if len(arms) > 10:
        shown["(the other arms)"] = _r(float(arms.iloc[10:].sum()), 4)
    evidence: dict[str, Any] = {"arms": shown, "arm_count": int(len(arms)), "propensity_auc": _r(auc, 3) if auc is not None else None}
    if target and target in train.columns:
        outcome = pd.to_numeric(train[target], errors="coerce")
        if outcome.notna().mean() > 0.9:  # a numeric or 0/1 outcome; a text one is not averaged
            means = {str(k)[:40]: float(outcome[t == k].mean()) for k in arms.index[:5]}
            evidence["outcome_by_arm"] = {k: _r(v, 4) for k, v in means.items() if np.isfinite(v)}
    return [_f("arms", "arms", f"{len(arms)} arms; the covariates tell the arms apart with AUC {auc:.2f}." if auc is not None else
               f"{len(arms)} arms.", evidence,
               ("The arms look randomised: an uplift model can compare them directly, scored by the Qini curve on held-out rows."
                if randomised else "The arms differ before the treatment: weight by the propensity, or the uplift measured is who was chosen."),
               ["task.framing", "risk.check", "model.families"], "info" if randomised else "warning")]


def _popularity(train: pd.DataFrame, item: Optional[str]) -> list[Finding]:
    if not item or item not in train.columns:
        return []
    counts = train[item].value_counts()
    if len(counts) < 20:
        return []
    top = float(counts.iloc[:max(1, len(counts) // 100)].sum() / counts.sum())
    return [_f(f"popularity-{item}", "popularity", f"The top 1% of {item} values hold {top:.0%} of the events.",
               {"items": int(len(counts)), "top_1pct_share": _r(top, 4), "items_seen_once": _r(float((counts == 1).mean()), 4)},
               "Popularity is the baseline every ranking must beat, and a metric over all items rewards it: report coverage of "
               "the long tail beside it.", ["baseline", "metric.primary", "risk.check"], "warning" if top > 0.2 else "info")]


def _scales(train: pd.DataFrame, cols: list[str]) -> list[Finding]:
    if len(cols) < 2:
        return []
    spreads = {str(c)[:60]: _robust(train[c])[1] for c in cols}
    ratio = max(spreads.values()) / max(min(spreads.values()), 1e-12)
    return [_f("scales", "scales", f"The numeric columns' spreads differ by a factor of {ratio:.3g}.",
               {"columns": len(spreads), "spread_ratio": _r(ratio, 3)},
               "Distances are dominated by the widest column unless each is scaled first (robust scaling; a log for skewed counts).",
               ["features.direction"], "warning" if ratio > 100 else "info")]


def by_family(train: pd.DataFrame, family: Optional[str], target: Optional[str], y: Optional[pd.Series], entity: Optional[str] = None,
              time: Optional[str] = None, treatment: Optional[str] = None, event: Optional[str] = None,
              labelled: bool = False) -> list[Finding]:
    """The family's own findings, all on the training rows (20.5). Unknown or no family: none."""
    if not family:
        return []
    out: list[Finding] = []
    cols = _numbers(train, [target, entity, time, treatment, event])
    no_label = family in NO_LABEL or (family == "rare_event" and not labelled and (y is None or y.isna().mean() > 0.5))
    if no_label or (family == "rare_event"):
        out += _normal(train, cols) + _rules(train, cols, entity) + _behaviour(train, entity, time)
    if no_label:
        out += _detectable(train, cols)
    if family == "time_to_event":
        out += _censoring(train, target, event)
    if family == "forecasting":
        out += _series(train, target, entity, time)
    if family == "uplift":
        out += _arms(train, target, treatment)
    if family == "ranking":
        out += _popularity(train, entity)
    if family == "segmentation":
        out += _normal(train, cols) + _scales(train, cols)
    return out


def analyse(tables: Mapping[str, pd.DataFrame], target: Optional[str] = None, task: Optional[TaskType] = None,
            brief: Optional[ProblemBrief] = None, train_index: Optional[Sequence[Any]] = None,
            main: Optional[str] = None, labels: Optional[str] = None, advisor: Optional[Advisor] = None,
            moment: Optional[str] = None, family: Optional[str] = None, entity: Optional[str] = None,
            time: Optional[str] = None, treatment: Optional[str] = None, event: Optional[str] = None) -> DataFindings:
    """Analyse the tables for the decisions they imply. ``train_index``: the rows of the main table the host may look
    at for choices (its training partition); without it, every row is used and the findings say so. ``advisor``: the
    data investigator (18.3) may read the column names and kinds, never values, and propose columns written after the
    ``moment`` (default: the brief's) and entities; every proposal is tested here like a telling name. ``family`` (the
    problem card's, 20.5) adds that family's findings, each on the training rows only: without labels the shape of
    normal, the rows rules would flag, each ``entity``'s behaviour and which planted shapes a detector finds; the
    censoring and the columns counted during follow-up (``event``: the ended flag); the series and their season
    (``entity`` the series, ``time``); the arms (``treatment``); popularity (``entity`` the item); the columns' scales."""
    if not tables:
        return DataFindings()
    main = main or max((n for n in tables if n != labels), key=lambda n: len(tables[n]))
    df = tables[main]
    if target is None and brief is not None and brief.target_column.known and brief.target_column.value in df.columns:
        target = brief.target_column.value
    train = df.loc[list(train_index)] if train_index is not None else df
    y = train[target] if target and target in train.columns else None
    if task is None or task == "unknown":
        task = (brief.task.value if brief is not None and brief.task.known and brief.task.value != "unknown" else None) \
            or (_infer_task(y) if y is not None and y.notna().any() else "unknown")
    findings: list[Finding] = []
    if train_index is None and y is not None:
        findings.append(_f("no-split", "scope", "No training partition was given: target statistics use every row.",
                           {"rows": int(len(df))}, "Lock the holdout first and pass its training rows, so no choice looks at it.",
                           ["split.strategy"], "warning", "all_rows"))
    findings += _grain(df, main, target)
    if y is not None:
        findings += _conflicts(train, target)
        findings += _target_findings(y, task, brief)
    features = train.drop(columns=[target]) if target in train.columns else train
    if moment is None and brief is not None and brief.prediction_moment.known:
        moment = str(brief.prediction_moment.value)
    proposals = _investigate(advisor, features, train, target, task, moment, findings) if advisor is not None and y is not None else {}
    findings += _entities(train, y, task, brief, target, proposals.get("entities"))
    findings += _time(train, y, task, brief, target)
    findings += _places(train, y, task, target)
    findings += _leakage(features, y, task, target, proposals.get("after"))
    findings += _missing(train, y, task, target)
    findings += _columns(df, target)
    if labels and labels in tables:  # which rows the label file marks is read on the training rows only
        findings += _labels(train, main, tables[labels], labels)
        findings += _labels_unmatched(df, main, tables[labels], labels)
    findings += by_family(train, family, target, y, entity=entity, time=time, treatment=treatment, event=event,
                          labelled=bool(labels and labels in tables))
    seen, unique = set(), []
    for f in findings:
        if f.id not in seen:
            seen.add(f.id)
            unique.append(f)
    summaries = [TableSummary(name=str(n)[:300], role="labels" if n == labels else ("main" if n == main else "related"),
                              rows=int(len(t)), columns=int(t.shape[1])) for n, t in tables.items()]
    return DataFindings(task=task, target=target, tables=summaries[:20], training_rows=int(len(train)) if train_index is not None else None,
                        findings=unique[:200])
