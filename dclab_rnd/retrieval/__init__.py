"""Evidence search, measured (package A5.1): keyword search, a vector search, and a hybrid of the two.

- ``bm25``: the evidence index's own keyword search (``EvidenceIndex.search``), what the product used until now.
- ``lsa``: vectors for every record from the index's own words (TF-IDF reduced by SVD, latent semantic analysis):
  local, deterministic and needing no model, so a question can find a record that shares no word with it but shares
  the words that keep company with it. It is not a neural sentence embedding: an embeddings endpoint through the
  model gateway would be the next retriever to measure, with the same harness.
- ``hybrid``: the two ranked lists merged by reciprocal rank fusion.

``measure()`` runs a frozen question set (``questions_v1.json``: 60 questions written by hand, each with the record
ids that answer it, in groups) and reports recall at 5 for each method, per group and overall. The product's search
(``search()``) uses the hybrid only when the latest stored measurement says it is not worse than keyword search on
any group; otherwise it stays keyword search. The index itself stays generated (``make knowledge``): the vectors are
computed from it when it is loaded, never stored by hand.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = Path(__file__).resolve().parent / "questions_v1.json"
RESULTS = ROOT / "evidence" / "campaigns" / "retrieval_v1" / "results"
K = 5
RRF = 60  # the usual constant of reciprocal rank fusion
METHODS = ("bm25", "lsa", "hybrid")


def _index():
    from dclab_rnd import tools

    return tools._index()


def _text(record: dict[str, Any]) -> str:
    return f"{record.get('title', '')} {record.get('text', '')}"


_FITTED: dict[str, tuple[Any, Any, Any, list[str]]] = {}


def _lsa(index: Any, components: int = 100) -> tuple[Any, Any, Any, list[str]]:
    """The fitted vectorizer, reducer, record vectors and ids for this index (fitted once per index content)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    signature = _signature(index)
    if signature in _FITTED:
        return _FITTED[signature]
    records = index.records
    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform([_text(r) for r in records])
    svd = TruncatedSVD(n_components=min(components, matrix.shape[1] - 1, len(records) - 1), algorithm="arpack", random_state=0)
    vectors = normalize(svd.fit_transform(matrix))
    _FITTED.clear()  # one index at a time: a rebuilt index replaces the old vectors
    _FITTED[signature] = (vectorizer, svd, vectors, [r["record_id"] for r in records])
    return _FITTED[signature]


def _signature(index: Any = None) -> str:
    import hashlib

    return hashlib.sha256("".join(r["record_id"] + _text(r) for r in (index or _index()).records).encode()).hexdigest()[:16]


def ranked(question: str, method: str, k: int = K, index: Any = None) -> list[str]:
    """The record ids the method ranks first for the question, in ``index`` (the repository's index by default)."""
    index = index or _index()
    if method == "bm25":
        return [h["record_id"] for h in index.search(question, k=k)]
    if method == "lsa":
        from sklearn.preprocessing import normalize

        vectorizer, svd, vectors, ids = _lsa(index)
        query = normalize(svd.transform(vectorizer.transform([question])))
        scores = (vectors @ query.T).ravel()
        order = scores.argsort()[::-1][:k]
        return [ids[i] for i in order if scores[i] > 0]
    if method == "hybrid":
        fused: dict[str, float] = {}
        for part in ("bm25", "lsa"):
            for rank, rid in enumerate(ranked(question, part, k=4 * k, index=index)):
                fused[rid] = fused.get(rid, 0.0) + 1.0 / (RRF + rank + 1)
        return [rid for rid, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
    raise ValueError(f"unknown method {method!r}")


def questions(path: Path = QUESTIONS) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def measure(qs: list[dict[str, Any]] | None = None, methods: tuple[str, ...] = METHODS, k: int = K) -> dict[str, Any]:
    """Recall at k per method, per group and overall: the share of a question's answer ids found in the top k."""
    qs = qs if qs is not None else questions()
    rows = []
    for q in qs:
        row = {"id": q["id"], "group": q["group"]}
        for method in methods:
            top = ranked(q["question"], method, k)
            row[method] = round(len(set(q["answers"]) & set(top)) / len(q["answers"]), 3)
        rows.append(row)
    groups = sorted({r["group"] for r in rows})
    recall = {m: {"overall": round(sum(r[m] for r in rows) / len(rows), 3),
                  **{g: round(sum(r[m] for r in rows if r["group"] == g) / sum(1 for r in rows if r["group"] == g), 3) for g in groups}}
              for m in methods}
    return {"k": k, "questions": len(rows), "groups": {g: sum(1 for r in rows if r["group"] == g) for g in groups}, "recall": recall, "rows": rows,
            "chosen": choose(recall, groups)}


def choose(recall: dict[str, dict[str, float]], groups: list[str]) -> str:
    """The hybrid only if it is not worse than keyword search on any group (and not worse overall)."""
    if "hybrid" not in recall or "bm25" not in recall:
        return "bm25"
    worse = [g for g in [*groups, "overall"] if recall["hybrid"][g] < recall["bm25"][g]]
    return "bm25" if worse else "hybrid"


def chosen() -> str:
    """The method the latest stored measurement chose (keyword search when none is stored)."""
    runs = sorted(RESULTS.glob("RET-*.json")) if RESULTS.is_dir() else []
    for path in reversed(runs):
        try:
            return json.loads(path.read_text(encoding="utf-8"))["measurement"]["chosen"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return "bm25"


def search(question: str, k: int = K, index: Any = None) -> list[dict[str, Any]]:
    """The evidence records for a question in ``index`` (the caller's, so a rebuilt index is searched as it is now), by
    the measured method; each hit has the record's fields, its keyword score (0 when keyword search did not find it)
    and the method. Keyword search when the vectors cannot be built (an environment without scikit-learn)."""
    index = index or _index()
    method = chosen()
    if method != "bm25":
        try:
            import sklearn  # noqa: F401
        except ImportError:
            method = "bm25"
    if method == "bm25":
        return index.search(question, k=k)
    by_id = {r["record_id"]: r for r in index.records}
    keyword = {h["record_id"]: h.get("score", 0.0) for h in index.search(question, k=4 * k)}
    return [{**by_id[rid], "score": keyword.get(rid, 0.0), "method": method} for rid in ranked(question, method, k, index=index)]


def main(argv: list[str] | None = None) -> int:
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.retrieval", description="Measure evidence search on the question set (A5.1).")
    parser.add_argument("--output", type=Path, help="store the measurement in this new file (never overwritten)")
    args = parser.parse_args(argv)
    m = measure()
    groups = list(m["groups"])
    print(f"Recall at {m['k']} on {m['questions']} questions ({', '.join(f'{g} {n}' for g, n in m['groups'].items())}):")
    print(f"  {'method':8s} {'overall':>8s} " + " ".join(f"{g:>10s}" for g in groups))
    for method, r in m["recall"].items():
        print(f"  {method:8s} {r['overall']:8.3f} " + " ".join(f"{r[g]:10.3f}" for g in groups))
    print(f"Chosen for the product: {m['chosen']} (the hybrid only if it is not worse on any group)")
    if args.output:
        found = re.match(r"([A-Z]+-\d+)_", args.output.name)
        report = {"experiment_id": found.group(1) if found else args.output.stem, "campaign_id": "retrieval_v1", "kind": "retrieval_recall",
                  "status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "question": "Which evidence search finds the records that answer real questions, keyword, vectors or both?",
                  "question_set": QUESTIONS.name, "index_signature": _signature(), "measurement": m,
                  "limitations": ["Sixty hand-written questions about one index; a group has 6 to 15 questions, so one question moves a group's recall by about 0.1.",
                                  "The vector retriever is latent semantic analysis of the index's own words, not a neural sentence embedding."]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
            handle.write("\n")
        print(f"Stored {args.output}")
    return 0
