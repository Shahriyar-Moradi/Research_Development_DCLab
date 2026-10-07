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
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = Path(__file__).resolve().parent / "questions_v1.json"
RESULTS = ROOT / "evidence" / "campaigns" / "retrieval_v1" / "results"
K = 5
RRF = 60  # the usual constant of reciprocal rank fusion
METHODS = ("bm25", "lsa", "hybrid")
EMBED_METHODS = ("embed", "hybrid_embed")  # measured when an embeddings client is given (the gateway's "embedding" purpose)
CANDIDATES = ("hybrid", "hybrid_embed", "embed")  # what may replace keyword search, if it is not worse on any group
EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH, EMBED_CHARS = 64, 6000


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


def embedding_model() -> str:
    import os

    return os.environ.get("DCLAB_EMBEDDING_MODEL", "").strip() or EMBED_MODEL


def cache_dir() -> Path:
    import os

    return Path(os.environ.get("DCLAB_EMBEDDINGS_DIR") or Path(os.environ.get("DCLAB_AGENT_HOME") or ROOT / "agent_runs") / "embeddings")


class Embedder:
    """Neural embeddings of the index's records and of questions, through a gateway client for the ``embedding``
    purpose (so every request is routed, capped and counted). The record vectors are a file beside the workspace,
    named by the model and the index's content; nothing is stored in the generated evidence files."""

    def __init__(self, client: Any, folder: Path | None = None, model: str | None = None):
        self.client, self.folder, self.model = client, Path(folder) if folder else cache_dir(), model or embedding_model()
        self._queries: dict[str, Any] = {}
        self._records: dict[str, tuple[list[str], Any]] = {}

    def _path(self, index: Any) -> Path:
        return self.folder / f"{re.sub(r'[^A-Za-z0-9._-]+', '-', self.model)}-{_signature(index)}.npz"

    def _embed(self, texts: list[str]) -> Any:
        import numpy as np

        out = []
        for start in range(0, len(texts), EMBED_BATCH):
            out.extend(self.client.embed([t[:EMBED_CHARS] or " " for t in texts[start:start + EMBED_BATCH]])["vectors"])
        matrix = np.asarray(out, dtype="float32")
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    def records(self, index: Any, build: bool = False) -> tuple[list[str], Any] | None:
        """(ids, vectors) of the index: from memory, from the file, or (only with ``build``) from the provider."""
        import numpy as np

        path = self._path(index)
        key = str(path)
        if key in self._records:
            return self._records[key]
        if path.is_file():
            try:
                with path.open("rb") as handle, np.load(handle, allow_pickle=False) as stored:
                    ids, vectors = list(stored["ids"]), stored["vectors"]
                if len(ids) == len(index.records) and vectors.shape[0] == len(ids):
                    self._records[key] = (ids, vectors)
                    return self._records[key]
            except Exception:  # noqa: BLE001 — a truncated or damaged file is the same as no file: a search falls back, a build replaces it
                pass
        if not build or self.client is None:
            return None
        ids = [r["record_id"] for r in index.records]
        vectors = self._embed([_text(r) for r in index.records])
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        with part.open("wb") as handle:  # whole or not at all: an interrupted build never leaves a file a search would read
            np.savez(handle, ids=np.asarray(ids), vectors=vectors)
        part.replace(path)
        self._records[key] = (ids, vectors)
        return self._records[key]

    def prefetch(self, questions: list[str]) -> None:
        todo = [q for q in dict.fromkeys(questions) if q not in self._queries]
        if todo:
            for q, v in zip(todo, self._embed(todo)):
                self._queries[q] = v

    def query(self, question: str) -> Any:
        self.prefetch([question])
        return self._queries[question]


def _embed_rank(question: str, index: Any, embedder: Embedder, k: int) -> list[str]:
    stored = embedder.records(index)
    if stored is None:
        raise RuntimeError("the record embeddings are not built: run make embeddings")  # a deployment builds its own, in its workspace folder
    ids, vectors = stored
    scores = vectors @ embedder.query(question)
    return [ids[i] for i in scores.argsort()[::-1][:k]]


def ranked(question: str, method: str, k: int = K, index: Any = None, embedder: Embedder | None = None) -> list[str]:
    """The record ids the method ranks first for the question, in ``index`` (the repository's index by default)."""
    index = index or _index()
    if method == "embed":
        if embedder is None:
            raise ValueError("the embed method needs an embedder")
        return _embed_rank(question, index, embedder, k)
    if method == "hybrid_embed":
        if embedder is None:
            raise ValueError("the hybrid_embed method needs an embedder")
        fused: dict[str, float] = {}
        for part in (ranked(question, "bm25", 4 * k, index), _embed_rank(question, index, embedder, 4 * k)):
            for rank, rid in enumerate(part):
                fused[rid] = fused.get(rid, 0.0) + 1.0 / (RRF + rank + 1)
        return [rid for rid, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
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


def default_embedder(index: Any) -> Embedder | None:
    """An embedder on the process's gateway, when a model serves the ``embedding`` purpose and the record vectors are built."""
    try:
        from dclab_rnd.models import installed

        client = installed().client("embedding", model=embedding_model())
    except Exception:  # noqa: BLE001
        return None
    if client is None:
        return None
    try:
        embedder = Embedder(client)
        return embedder if embedder.records(index) is not None else None
    except Exception:  # noqa: BLE001 — never an error on the page: keyword search
        return None


def questions(path: Path = QUESTIONS) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def measure(qs: list[dict[str, Any]] | None = None, methods: tuple[str, ...] = METHODS, k: int = K, embedder: Embedder | None = None) -> dict[str, Any]:
    """Recall at k per method, per group and overall: the share of a question's answer ids found in the top k.
    With an ``embedder`` the two embedding methods are measured too (its record vectors are built first if missing)."""
    qs = qs if qs is not None else questions()
    if embedder is not None:
        methods = tuple(dict.fromkeys([*methods, *EMBED_METHODS]))
        embedder.records(_index(), build=True)
        embedder.prefetch([q["question"] for q in qs])
    rows = []
    for q in qs:
        row = {"id": q["id"], "group": q["group"]}
        for method in methods:
            top = ranked(q["question"], method, k, embedder=embedder)
            row[method] = round(len(set(q["answers"]) & set(top)) / len(q["answers"]), 3)
        rows.append(row)
    groups = sorted({r["group"] for r in rows})
    recall = {m: {"overall": round(sum(r[m] for r in rows) / len(rows), 3),
                  **{g: round(sum(r[m] for r in rows if r["group"] == g) / sum(1 for r in rows if r["group"] == g), 3) for g in groups}}
              for m in methods}
    return {"k": k, "questions": len(rows), "groups": {g: sum(1 for r in rows if r["group"] == g) for g in groups}, "recall": recall, "rows": rows,
            "chosen": choose(recall, groups)}


def choose(recall: dict[str, dict[str, float]], groups: list[str]) -> str:
    """A method replaces keyword search only if it is not worse than keyword search on any group (and not worse
    overall); of those that qualify, the best overall, and keyword search when none does or none is better."""
    if "bm25" not in recall:
        return "bm25"
    best, best_overall = "bm25", recall["bm25"]["overall"]
    for method in CANDIDATES:
        if method not in recall or any(recall[method][g] < recall["bm25"][g] for g in [*groups, "overall"]):
            continue
        if recall[method]["overall"] > best_overall:
            best, best_overall = method, recall[method]["overall"]
    return best


def chosen() -> str:
    """The method the latest stored measurement chose (keyword search when none is stored)."""
    runs = sorted(RESULTS.glob("RET-*.json")) if RESULTS.is_dir() else []
    latest = None
    for path in reversed(runs):
        try:
            measurement = json.loads(path.read_text(encoding="utf-8"))["measurement"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        latest = latest or measurement
        if "embed" in measurement["recall"]:  # a later plain run did not measure the neural methods: it does not undo what they showed
            return measurement["chosen"]
    return latest["chosen"] if latest else "bm25"


def search(question: str, k: int = K, index: Any = None) -> list[dict[str, Any]]:
    """The evidence records for a question in ``index`` (the caller's, so a rebuilt index is searched as it is now), by
    the measured method; each hit has the record's fields, its keyword score (0 when keyword search did not find it)
    and the method. Keyword search when the vectors cannot be built (an environment without scikit-learn)."""
    index = index or _index()
    method = chosen()
    embedder = None
    if method in EMBED_METHODS:  # only with the record vectors already built and a model that answers: a search never spends silently
        embedder = default_embedder(index)
        method = method if embedder is not None else "bm25"
    elif method != "bm25":
        try:
            import sklearn  # noqa: F401
        except ImportError:
            method = "bm25"
    if method == "bm25":
        return index.search(question, k=k)
    if embedder is not None:
        try:
            top = ranked(question, method, k, index=index, embedder=embedder)
        except Exception:  # noqa: BLE001 — a failed or refused embeddings request leaves keyword search, never an error on the page
            return index.search(question, k=k)
        by_id = {r["record_id"]: r for r in index.records}
        keyword = {h["record_id"]: h.get("score", 0.0) for h in index.search(question, k=4 * k)}
        return [{**by_id[rid], "score": keyword.get(rid, 0.0), "method": method} for rid in top]
    by_id = {r["record_id"]: r for r in index.records}
    keyword = {h["record_id"]: h.get("score", 0.0) for h in index.search(question, k=4 * k)}
    return [{**by_id[rid], "score": keyword.get(rid, 0.0), "method": method} for rid in ranked(question, method, k, index=index)]


def main(argv: list[str] | None = None) -> int:
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.retrieval", description="Measure evidence search on the question set (A5.1).")
    parser.add_argument("--output", type=Path, help="store the measurement in this new file (never overwritten)")
    parser.add_argument("--build-embeddings", action="store_true", help="only build the record vectors for the product (make embeddings): no measurement, no result file")
    parser.add_argument("--embeddings", action="store_true", help="also measure the neural retrievers: builds the record vectors and embeds the questions through "
                        "the gateway (a few thousand tokens, a fraction of a cent; needs a model for the embedding purpose)")
    args = parser.parse_args(argv)
    embedder = None
    if args.build_embeddings:
        args.embeddings = True
    if args.embeddings:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        from dclab_rnd.models.gateway import for_workspace

        gateway = for_workspace()
        gateway.shadows = None
        client = gateway.client("embedding", model=embedding_model())
        if client is None:
            print("No model serves the embedding purpose; nothing was sent.", file=sys.stderr)
            return 2
        embedder = Embedder(client)
    if args.build_embeddings:
        built = embedder.records(_index(), build=True)
        print(f"{len(built[0])} record vectors by {embedder.model} in {embedder._path(_index())} (€{round(client.spent_eur, 6)})")
        return 0
    m = measure(embedder=embedder)
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
                                  "The choice is made on the same questions it reports (no held-out set).",
                                  "lsa is latent semantic analysis of the index's own words, not a neural embedding; embed and hybrid_embed are neural "
                                  f"embeddings by {embedding_model()} when measured with --embeddings."],
                  **({"embedding_model": embedder.model, "embedding_spent_eur": round(embedder.client.spent_eur, 6)} if embedder else {})}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
            handle.write("\n")
        print(f"Stored {args.output}")
    return 0
