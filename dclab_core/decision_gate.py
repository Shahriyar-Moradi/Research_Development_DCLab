"""The promotion gate of a decision model (package 22.3), written before any sealed run.

A decision model (a classifier of 22.2, or any answerer of 22.0) answers a point's sealed decisions once; the gate says
whether it may answer beside the rule. One AND gate, in the shape of the main product's R3: every sealed decision scored,
lower bounds on accuracy, precision and recall, a margin over the rule, no class it misses, and few decisions the rule
got right turned wrong. Every lower bound is drawn by a bootstrap clustered by table group (a table's decisions, and a
table and its hidden-name twin, are not independent); for a choice, recall is the mean over the classes in the truth
and precision the mean over the classes it answered. Its numbers are fixed here; a
threshold is never moved after a result is seen (a new version is a new gate, and a verdict names its version). The
demotion bars are declared with it and applied to a promoted model on any later labelled set.

A verdict is evidence about this model on these decisions; never a statement that anything is ready for production.
"""

from __future__ import annotations

from typing import Any, Optional

from dclab_core.answerers import _key
from dclab_core.decision_models import metrics

VERSION = "decision_gate_v1"
THRESHOLDS = {
    "scored_share_min": 1.0,          # every sealed decision answered or counted wrong; an error is a failure, never a skip
    "accuracy_lower_min": 0.90,       # the 95% lower bound of accuracy (an abstention is wrong), by table group
    "precision_lower_min": 0.80,      # of precision (yes/no: of the answer True; a choice: mean over the classes answered)
    "recall_lower_min": 0.80,         # and of recall (yes/no: of True; a choice: mean over the classes in the truth)
    "margin_lower_min": 0.0,          # the 95% lower bound of (model accuracy - rule accuracy), by table group: above it
    "class_min_rows": 30,             # a class counts once it has this many sealed decisions
    "class_recall_min": 0.70,         # and none may be below this: an aggregate must not hide a class the model misses
    "turned_max": 0.02,               # decisions the rule got right that the model gets wrong, over the rule's right ones
    "draws": 2000,                    # bootstrap draws (clustered by table group), seed 0
}
DEMOTION = {
    "accuracy_lower_below": 0.85,     # a promoted model is demoted when, on a later labelled set, any of these holds
    "margin_lower_below": -0.02,
    "turned_above": 0.05,
}
NOT_YET = ("Blind cases (the main product's R3 asks for them): the generated sets have benchmark labels only; "
           "decision_gate_v2 adds the blind labels of 22.4's disagreements.")


def _codes(truth: list[Any], model: list[Any], rule: list[Any], groups: list[Any]) -> tuple[Any, Any, Any, Any, list[str]]:
    import numpy as np

    keys = [_key(x) for x in truth]
    classes = sorted(set(keys) | {_key(x) for x in [*model, *rule] if x is not None})
    code = {c: i for i, c in enumerate(classes)}
    enc = lambda xs: np.array([-1 if x is None else code[_key(x)] for x in xs])  # noqa: E731
    ids = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    return np.array([code[k] for k in keys]), enc(model), enc(rule), np.array([ids[g] for g in groups]), classes


def _stats(t: Any, a: Any, w: Any, positive: Optional[int], true_classes: Any, said_classes: Any) -> tuple[float, float, float]:
    """Accuracy, precision and recall with rows weighted by ``w``: for yes/no, of the ``positive`` class; for a choice,
    precision the mean over ``said_classes`` and recall the mean over ``true_classes`` (a class without rows in a draw
    is left out of that draw's mean)."""
    import numpy as np

    total = w.sum()
    acc = float((w * (t == a)).sum() / total) if total else float("nan")
    if positive is not None:
        tp = (w * ((t == positive) & (a == positive))).sum()
        said, true = (w * (a == positive)).sum(), (w * (t == positive)).sum()
        return acc, float(tp / said) if said else float("nan"), float(tp / true) if true else float("nan")
    ps = [((w * ((t == c) & (a == c))).sum() / s) for c in said_classes if (s := (w * (a == c)).sum())]
    rs = [((w * ((t == c) & (a == c))).sum() / n) for c in true_classes if (n := (w * (t == c)).sum())]
    return acc, float(np.mean(ps)) if ps else float("nan"), float(np.mean(rs)) if rs else float("nan")


def bounds(truth: list[Any], answers: list[Any], groups: list[Any], draws: int = 2000, seed: int = 0) -> dict[str, list[Optional[float]]]:
    """Accuracy, precision and recall (as :func:`_stats`), each [value, 95% lower, 95% upper] from a bootstrap that
    draws whole table groups."""
    import numpy as np

    if not truth:
        return {"accuracy": [None] * 3, "precision": [None] * 3, "recall": [None] * 3}
    t, a, _, cl, classes = _codes(truth, answers, answers, groups)
    positive = classes.index("True") if set(classes) <= {"True", "False"} and "True" in classes else None
    true_classes, said_classes = np.unique(t), np.unique(a[a >= 0])
    point = _stats(t, a, np.ones(len(t)), positive, true_classes, said_classes)
    k = int(cl.max()) + 1
    rng = np.random.default_rng(seed)
    drawn = np.array([_stats(t, a, np.bincount(rng.integers(0, k, k), minlength=k)[cl].astype(float), positive, true_classes,
                             said_classes) for _ in range(draws)])

    def three(i: int) -> list[Optional[float]]:
        col = drawn[:, i][~np.isnan(drawn[:, i])]
        if np.isnan(point[i]) or not len(col):
            return [None, None, None]
        return [round(point[i], 4), round(float(np.percentile(col, 2.5)), 4), round(float(np.percentile(col, 97.5)), 4)]

    return {"accuracy": three(0), "precision": three(1), "recall": three(2)}


def margin(truth: list[Any], model: list[Any], rule: list[Any], groups: list[Any], draws: int = 2000,
           seed: int = 0) -> dict[str, Optional[float]]:
    """Model accuracy minus rule accuracy on the same decisions, with a 95% interval from a bootstrap that draws whole
    table groups (decisions of one table are not independent)."""
    import numpy as np

    if not truth:
        return {"difference": None, "lower": None, "upper": None}
    t, m, r, cl, _ = _codes(truth, model, rule, groups)
    d = (m == t).astype(float) - (r == t).astype(float)
    k = int(cl.max()) + 1
    rng = np.random.default_rng(seed)
    drawn = []
    for _ in range(draws):
        w = np.bincount(rng.integers(0, k, k), minlength=k)[cl].astype(float)
        drawn.append(float((d * w).sum() / w.sum()))
    return {"difference": round(float(d.mean()), 4), "lower": round(float(np.percentile(drawn, 2.5)), 4),
            "upper": round(float(np.percentile(drawn, 97.5)), 4)}


def verdict(truth: list[Any], model: list[Any], rule: list[Any], groups: list[Any], errors: int = 0,
            expected: Optional[int] = None) -> dict[str, Any]:
    """Apply the gate to one model's answers on a point's sealed decisions (None an abstention), beside the rule's
    answers on the same decisions and each decision's table group. ``errors``: decisions it failed to answer by an
    error; ``expected``: how many sealed decisions there are (all must be scored)."""
    t = THRESHOLDS
    expected = len(truth) if expected is None else expected
    scored = len(truth) - errors
    m = metrics(truth, model, groups=groups, draws=t["draws"])
    r = metrics(truth, rule, groups=groups, draws=t["draws"])
    gap = margin(truth, model, rule, groups, draws=t["draws"])
    keys = [_key(x) for x in truth]
    weak = []
    for c in sorted(set(keys)):
        idx = [i for i, k in enumerate(keys) if k == c]
        hit = sum(model[i] is not None and _key(model[i]) == c for i in idx)
        if len(idx) >= t["class_min_rows"] and hit / len(idx) < t["class_recall_min"]:
            weak.append(f"{c} ({hit}/{len(idx)})")
    rule_right = [i for i, k in enumerate(keys) if rule[i] is not None and _key(rule[i]) == k]
    turned = sum(not (model[i] is not None and _key(model[i]) == keys[i]) for i in rule_right)
    turned_share = turned / len(rule_right) if rule_right else 0.0

    def check(name: str, value: Any, need: str, ok: Any) -> dict[str, Any]:
        return {"criterion": name, "value": value, "need": need, "pass": bool(ok)}

    b, rb = bounds(truth, model, groups, t["draws"]), bounds(truth, rule, groups, t["draws"])
    lo = lambda name: b[name][1]  # noqa: E731
    criteria = [
        check("every sealed decision scored", f"{scored}/{expected}", f"{t['scored_share_min']:.0%}",
              expected and scored / expected >= t["scored_share_min"]),
        check("accuracy, 95% lower bound (by table group)", lo("accuracy"), f">= {t['accuracy_lower_min']}",
              lo("accuracy") is not None and lo("accuracy") >= t["accuracy_lower_min"]),
        check("precision, 95% lower bound (by table group)", lo("precision"), f">= {t['precision_lower_min']}",
              lo("precision") is not None and lo("precision") >= t["precision_lower_min"]),
        check("recall, 95% lower bound (by table group)", lo("recall"), f">= {t['recall_lower_min']}",
              lo("recall") is not None and lo("recall") >= t["recall_lower_min"]),
        check("margin over the rule, 95% lower bound (by table group)", gap["lower"], f"> {t['margin_lower_min']}",
              gap["lower"] is not None and gap["lower"] > t["margin_lower_min"]),
        check("no class it misses", weak or "none", f"no class with {t['class_min_rows']}+ decisions below recall {t['class_recall_min']}",
              not weak),
        check("the rule's right decisions kept", round(turned_share, 4), f"<= {t['turned_max']} turned wrong",
              turned_share <= t["turned_max"]),
    ]
    sealed_n = len(truth)
    return {"gate": VERSION, "pass": all(c["pass"] for c in criteria), "criteria": criteria,
            "bounds": {"model": b, "rule": rb, "note": "[value, 95% lower, 95% upper], bootstrap of whole table groups; the gate reads these"},
            "model": m.model_dump(), "rule": r.model_dump(), "margin": gap,
            "turned": [turned, len(rule_right)], "groups": len(set(groups)), "decisions": sealed_n,
            "not_yet": NOT_YET, "demotion": DEMOTION,
            "scope": "Evidence about this model on these generated decisions, not a claim that it is ready for production."}


__all__ = ["VERSION", "THRESHOLDS", "DEMOTION", "bounds", "margin", "verdict"]
