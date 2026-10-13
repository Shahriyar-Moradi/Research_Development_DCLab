"""
replay_tasks.py
===============

General tasks that keep the model's coding and quantitative reasoning in shape while it is fine-tuned on the
workflow (the "replay" share of the training mix). Every task is CHECKABLE, so the answers Qwen writes for them on
the Mac (mac/make_replay.py) are kept only when they are right:

* code tasks: write a function; its answer is run against tests whose expected values come from a reference
  implementation here (data helpers, leakage-safe preprocessing, metrics, algorithms, pandas, SQL);
* numeric tasks: ML arithmetic (precision and recall from a confusion matrix, expected value of an action,
  calibration error, population stability index, lift, prevalence); the final line must give the number.

The prompts carry no DCLab system prompt: they stand for the model's ordinary use, which must not drift.

    python research/llm-fine-tuning/experiments/sft_factory/replay_tasks.py   ->   out/mix/replay_tasks.jsonl
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "mix"

CODE_SUFFIX = "\n\nReturn only the Python code, in one ```python block. Use only the standard library{extra}."
NUM_SUFFIX = "\n\nShow your working briefly, then end with a last line of the form: ANSWER: <number> (rounded to 4 decimals)."


# ============================================================================ code tasks: reference + spec

def ref_moving_average(xs, k):
    return [round(sum(xs[i - k + 1:i + 1]) / k, 4) for i in range(k - 1, len(xs))]


def ref_merge_intervals(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def ref_rle(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{s[i]}{j - i}")
        i = j
    return "".join(out)


def ref_balanced(s):
    pairs, stack = {")": "(", "]": "[", "}": "{"}, []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


def ref_roc_auc(y, s):
    pos = [v for v, t in zip(s, y) if t == 1]
    neg = [v for v, t in zip(s, y) if t == 0]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def ref_prf(y, p):
    tp = sum(1 for a, b in zip(y, p) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, p) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, p) if a == 1 and b == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return (round(prec, 4), round(rec, 4), round(f1, 4))


def ref_ece(y, p, bins):
    total, n = 0.0, len(y)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, v in enumerate(p) if (lo <= v < hi) or (b == bins - 1 and v == 1.0)]
        if idx:
            total += len(idx) / n * abs(sum(y[i] for i in idx) / len(idx) - sum(p[i] for i in idx) / len(idx))
    return round(total, 4)


def ref_time_split(dates, share):
    order = sorted(range(len(dates)), key=lambda i: (dates[i], i))
    cut = int(len(dates) * share)
    return sorted(order[:cut]), sorted(order[cut:])


def ref_oof_target_encoding(cats, y, folds):
    out = [None] * len(cats)
    prior = sum(y) / len(y)
    for f in sorted(set(folds)):
        tr = [i for i in range(len(cats)) if folds[i] != f]
        sums, cnt = Counter(), Counter()
        for i in tr:
            sums[cats[i]] += y[i]
            cnt[cats[i]] += 1
        for i in range(len(cats)):
            if folds[i] == f:
                out[i] = round(sums[cats[i]] / cnt[cats[i]], 4) if cnt[cats[i]] else round(prior, 4)
    return out


def ref_fit_scaler(train_cols):
    out = []
    for col in train_cols:
        m = sum(col) / len(col)
        sd = math.sqrt(sum((v - m) ** 2 for v in col) / len(col))
        out.append((round(m, 4), round(sd, 4)))
    return out


def ref_dedupe(rows):
    seen, out = set(), []
    for r in rows:
        t = tuple(r)
        if t not in seen:
            seen.add(t)
            out.append(list(r))
    return out


def ref_kfold(n, k):
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    out, start = [], 0
    for s in sizes:
        out.append(list(range(start, start + s)))
        start += s
    return out


def ref_lis(xs):
    import bisect
    tails = []
    for x in xs:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def ref_window_max(xs, k):
    return [max(xs[i:i + k]) for i in range(len(xs) - k + 1)]


def ref_parse_kv(s):
    out = {}
    for part in filter(None, s.split(";")):
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        out[k] = int(v) if v.lstrip("-").isdigit() else v
    return out


def ref_topk(words, k):
    c = Counter(words)
    return [w for w, _ in sorted(c.items(), key=lambda t: (-t[1], t[0]))[:k]]


def ref_psi(a, b):
    total = 0.0
    for x, y in zip(a, b):
        x, y = max(x, 1e-6), max(y, 1e-6)
        total += (x - y) * math.log(x / y)
    return round(total, 4)


def ref_group_sum(records):
    out = {}
    for k, v in records:
        out[k] = round(out.get(k, 0) + v, 4)
    return dict(sorted(out.items()))


def ref_sql_count(rows):
    c = Counter(r[0] for r in rows if r[1] >= 0)
    return sorted(c.items(), key=lambda t: (-t[1], t[0]))


CODE = [
    ("moving_average", "Write `{fn}(xs, k)` returning the trailing moving averages of the list `xs` with window `k` "
     "(one value per position from index k-1 on), each rounded to 4 decimals.",
     lambda r: (r.sample(range(-50, 50), 12), r.randint(2, 5)), ref_moving_average),
    ("merge_intervals", "Write `{fn}(intervals)` that merges overlapping (or touching) closed intervals given as "
     "[start, end] pairs and returns them sorted as lists of [start, end].",
     lambda r: ([[a, a + r.randint(0, 6)] for a in r.sample(range(0, 40), 8)],), ref_merge_intervals),
    ("run_length", "Write `{fn}(s)` that run-length encodes the string `s`: each run becomes the character followed by "
     "its count, e.g. 'aaab' -> 'a3b1'.",
     lambda r: ("".join(r.choice("abc") * r.randint(1, 4) for _ in range(6)),), ref_rle),
    ("balanced", "Write `{fn}(s)` returning True when every bracket in `s` ((), [], {{}}) is closed in the right order, "
     "ignoring other characters.",
     lambda r: ("".join(r.choice("()[]{}ab") for _ in range(r.randint(4, 12))),), ref_balanced),
    ("roc_auc", "Write `{fn}(y_true, scores)` computing ROC-AUC without any library: the share of (positive, negative) "
     "pairs where the positive scores higher, ties counting one half. Round to 4 decimals.",
     lambda r: ([1, 0] + [r.randint(0, 1) for _ in range(10)], [round(r.random(), 2) for _ in range(12)]), ref_roc_auc),
    ("precision_recall_f1", "Write `{fn}(y_true, y_pred)` returning (precision, recall, f1) for the positive class 1, "
     "each rounded to 4 decimals; return 0.0 for a ratio whose denominator is zero.",
     lambda r: ([r.randint(0, 1) for _ in range(14)], [r.randint(0, 1) for _ in range(14)]), ref_prf),
    ("ece", "Write `{fn}(y_true, probs, bins)` computing expected calibration error with `bins` equal-width bins on "
     "[0, 1] (a probability of exactly 1.0 goes in the last bin): the sum over non-empty bins of (bin share of rows) × "
     "|mean label − mean probability|. Round to 4 decimals.",
     lambda r: ([r.randint(0, 1) for _ in range(16)], [round(r.random(), 2) for _ in range(16)], r.choice([4, 5, 10])), ref_ece),
    ("time_split", "Write `{fn}(dates, share)` that returns (train_indices, test_indices): the earliest `share` of rows "
     "by date (ties broken by original position; count = int(len * share)) train, the rest test; both lists sorted. "
     "Dates are ISO strings 'YYYY-MM-DD'.",
     lambda r: ([f"2024-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}" for _ in range(10)], r.choice([0.6, 0.75, 0.8])), ref_time_split),
    ("oof_target_encoding", "Write `{fn}(categories, y, folds)` returning out-of-fold target encoding: for each row, "
     "the mean of `y` for its category computed only on rows of OTHER folds; if the category never appears in the "
     "other folds, use the overall mean of `y`. Round each value to 4 decimals.",
     lambda r: ([r.choice("ABCD") for _ in range(12)], [r.randint(0, 1) for _ in range(12)], [i % 3 for i in range(12)]), ref_oof_target_encoding),
    ("fit_scaler", "Write `{fn}(train_columns)` that takes a list of numeric columns (each a list) from the TRAINING "
     "rows only and returns [(mean, population_std), ...], each rounded to 4 decimals, to be reused on test rows.",
     lambda r: ([[r.randint(0, 20) for _ in range(6)] for _ in range(3)],), ref_fit_scaler),
    ("dedupe_rows", "Write `{fn}(rows)` that removes exact duplicate rows (lists) and keeps the first occurrence, "
     "preserving order; return a list of lists.",
     lambda r: ([[r.randint(0, 2), r.randint(0, 2)] for _ in range(10)],), ref_dedupe),
    ("kfold_indices", "Write `{fn}(n, k)` that splits indices 0..n-1 into k contiguous folds whose sizes differ by at "
     "most one, larger folds first; return a list of k lists.",
     lambda r: (r.randint(7, 23), r.randint(2, 5)), ref_kfold),
    ("lis_length", "Write `{fn}(xs)` returning the length of the longest strictly increasing subsequence of `xs`.",
     lambda r: (r.sample(range(0, 60), 14),), ref_lis),
    ("window_max", "Write `{fn}(xs, k)` returning the maximum of every contiguous window of length `k`.",
     lambda r: ([r.randint(-9, 9) for _ in range(12)], r.randint(2, 4)), ref_window_max),
    ("parse_kv", "Write `{fn}(s)` that parses 'key=value;key=value' into a dict, stripping spaces, turning values made "
     "of digits (with an optional leading minus) into int, keeping others as strings, and ignoring empty parts.",
     lambda r: (";".join(f" {r.choice('abcxyz')}{i} = {r.choice([str(r.randint(-5, 50)), 'on', 'v2'])}" for i in range(4)) + ";",), ref_parse_kv),
    ("top_k_words", "Write `{fn}(words, k)` returning the k most frequent words, ties broken alphabetically.",
     lambda r: ([r.choice(["data", "model", "split", "leak", "fold"]) for _ in range(15)], r.randint(1, 4)), ref_topk),
    ("psi", "Write `{fn}(expected_shares, actual_shares)` computing the population stability index over matching bins: "
     "sum of (e − a) × ln(e / a), with each share floored at 1e-6. Round to 4 decimals.",
     lambda r: ([0.1, 0.2, 0.3, 0.4], [round(x, 2) for x in (0.1 + r.random() * 0.1, 0.2, 0.3, 0.4 - r.random() * 0.1)]), ref_psi),
    ("group_sum", "Write `{fn}(records)` that takes (key, value) pairs and returns a dict of key -> sum of values, "
     "rounded to 4 decimals, with keys in sorted order.",
     lambda r: ([(r.choice("xyz"), round(r.uniform(-3, 3), 2)) for _ in range(9)],), ref_group_sum),
    ("sqlite_counts", "Write `{fn}(rows)` that loads (name, amount) rows into an in-memory sqlite3 table and returns, "
     "with ONE SQL query, a list of (name, count) for rows with amount >= 0, ordered by count descending then name.",
     lambda r: ([(r.choice(["ana", "bo", "cy"]), r.randint(-3, 9)) for _ in range(10)],), ref_sql_count),
]

FN_NAMES = {"moving_average": ["moving_average", "rolling_mean"], "merge_intervals": ["merge_intervals", "merge_ranges"],
            "run_length": ["run_length_encode", "rle"], "balanced": ["is_balanced", "brackets_ok"],
            "roc_auc": ["roc_auc", "auc_score"], "precision_recall_f1": ["precision_recall_f1", "prf"],
            "ece": ["expected_calibration_error", "ece"], "time_split": ["time_split", "split_by_time"],
            "oof_target_encoding": ["oof_target_encode", "target_encode_oof"], "fit_scaler": ["fit_scaler", "train_stats"],
            "dedupe_rows": ["dedupe_rows", "drop_duplicates"], "kfold_indices": ["kfold_indices", "contiguous_folds"],
            "lis_length": ["lis_length", "longest_increasing"], "window_max": ["window_max", "sliding_max"],
            "parse_kv": ["parse_kv", "parse_settings"], "top_k_words": ["top_k_words", "most_common_words"],
            "psi": ["psi", "population_stability"], "group_sum": ["group_sum", "sum_by_key"],
            "sqlite_counts": ["sqlite_counts", "count_names"]}


def py(obj) -> str:
    """A Python literal for test arguments and expected values (JSON spelling of True/False/None fixed)."""
    return json.dumps(obj).replace("true", "True").replace("false", "False").replace("null", "None")


def code_tasks(rng, per_template):
    out = []
    for name, spec, gen, ref in CODE:
        for v in range(per_template):
            fn = FN_NAMES[name][v % len(FN_NAMES[name])]
            cases = [gen(rng) for _ in range(4)]
            tests = [f"assert _norm({fn}(*{py(list(c))})) == _norm({py(ref(*c))})" for c in cases]
            prompt = spec.format(fn=fn) + CODE_SUFFIX.format(extra=" (sqlite3 is in it)" if name == "sqlite_counts" else "")
            out.append({"id": f"code-{name}-{v}", "kind": "code", "prompt": prompt, "function": fn, "tests": tests,
                        "reference": name})
    return out


# ============================================================================ numeric tasks

def numeric_tasks(rng, n):
    out = []
    for i in range(n):
        kind = i % 6
        if kind == 0:
            tp, fp, fn = rng.randint(20, 90), rng.randint(5, 60), rng.randint(5, 60)
            ask = "recall" if i % 2 else "precision"
            prompt = (f"A churn model flags customers. On a test set it gets {tp} true positives, {fp} false positives "
                      f"and {fn} false negatives. What is its {ask}?")
            ans = tp / (tp + fn) if ask == "recall" else tp / (tp + fp)
        elif kind == 1:
            p, value, cost = round(rng.uniform(0.05, 0.6), 2), rng.randint(50, 900), rng.randint(2, 40)
            uplift = round(rng.uniform(0.02, 0.2), 2)
            prompt = (f"Calling a customer costs {cost}. Without a call the customer converts with probability {p}; a call "
                      f"raises that probability by {uplift} (absolute). A conversion is worth {value}. What is the expected "
                      f"net gain of calling, compared with not calling?")
            ans = uplift * value - cost
        elif kind == 2:
            bins = [(round(rng.uniform(0.05, 0.95), 2), rng.randint(0, 1)) for _ in range(6)]
            prompt = ("Six predictions fall into a single calibration bin: probabilities "
                      f"{[b[0] for b in bins]} with outcomes {[b[1] for b in bins]}. What is the absolute gap between the "
                      "mean predicted probability and the observed positive rate in this bin?")
            ans = abs(sum(b[0] for b in bins) / 6 - sum(b[1] for b in bins) / 6)
        elif kind == 3:
            e = [0.25, 0.25, 0.25, 0.25]
            a = [round(x, 2) for x in (0.25 + rng.uniform(-0.1, 0.1), 0.25, 0.25)]
            a.append(round(1 - sum(a), 2))
            prompt = (f"A feature's bin shares are {e} in training and {a} in production. Compute the population stability "
                      f"index, sum of (e − a) × ln(e / a).")
            ans = sum((x - y) * math.log(x / y) for x, y in zip(e, a))
        elif kind == 4:
            base, top = round(rng.uniform(0.02, 0.15), 3), round(rng.uniform(0.1, 0.4), 3)
            prompt = (f"The overall purchase rate is {base}. Among the 10% of visitors the model ranks highest, the purchase "
                      f"rate is {top}. What is the lift of that top decile?")
            ans = top / base
        else:
            rows, dup = rng.randint(800, 5000), round(rng.uniform(0.02, 0.2), 3)
            pos = rng.randint(50, 400)
            prompt = (f"A table has {rows} rows, {pos} of them positive. A share {dup} of all rows are exact duplicates of "
                      f"positive rows and are removed. What is the positive rate after removing them?")
            removed = round(rows * dup)
            ans = (pos - removed) / (rows - removed) if pos > removed else None
            if ans is None:
                continue
        out.append({"id": f"num-{i}", "kind": "numeric", "prompt": prompt + NUM_SUFFIX, "answer": round(ans, 4), "tolerance": 0.0006})
    return out


def main():
    rng = random.Random(7)
    tasks = code_tasks(rng, per_template=6) + numeric_tasks(rng, 120)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "replay_tasks.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks)
    print(json.dumps(dict(Counter(t["kind"] for t in tasks)), indent=1))


if __name__ == "__main__":
    main()
