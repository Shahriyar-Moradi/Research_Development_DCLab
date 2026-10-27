"""Memory search measured before any vector store (package 35.0, docs/guides/MEMORY_PLAN.md).

The question: does a local embedding model find an agent's memory items better than search without a model, when the
question uses other words than the item or is asked in Persian? Keyword search (BM25) cannot: retrieval_v1 found 0 of
10 reworded questions, and its tokenizer reads no Persian letter at all.

- The corpus (``memory_v1/items.json``): 116 invented items shaped like DCLab's memory (facts about people, the
  workspace and ten projects, know-how, episodes); no customer's data.
- The questions, in three groups: ``direct`` (may share words with the answer), ``reworded`` and ``persian`` (share no
  word with it; the second written in Persian script). Dev (36, written by hand) chooses the model and the method;
  sealed (30, written by an independent agent that read only the corpus, hash pinned) is read once for the verdict.
- The baseline is ``hybrid`` (BM25 fused with LSA): the best search that needs no model, and the one that can find a
  reworded question at all; against BM25 alone (0 on reworded and Persian by construction) any hit would look like a gain.
- The candidates: three local models served by Ollama, each embedding through the gateway's ``embedding`` purpose on a
  local endpoint (nothing leaves the machine), with the prefixes their model cards ask for; each result names the
  model's weights (Ollama's digest).
- The gate (``GATE``) is frozen in the campaign's manifest before any run; a changed gate, question file or code is
  refused, the manifest cannot be written again once a result exists, and a new campaign version is the only way on.
- Capture: every result keeps each question's first ``K`` items per method (not only its recall), the times, the
  model's digest and memory, so the runs are training data for a retriever or a reranker; dev rows may train, sealed
  rows never (marked in each result).

Steps: ``freeze`` (once, before any run) → ``dev --ollama MODEL`` for every candidate → ``choose`` (prints the choice) →
``sealed --yes`` (once).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ROOT, Embedder, ranked

PACKAGE = "35.0"
CAMPAIGN_ID = "memory_retrieval_v1"
CAMPAIGN = ROOT / "evidence" / "campaigns" / CAMPAIGN_ID
FOLDER = Path(__file__).resolve().parent / "memory_v1"
ITEMS = FOLDER / "items.json"
DEV = FOLDER / "questions_dev.json"
SEALED = FOLDER / "questions_sealed.json"
SEALED_HASH = FOLDER / "sealed.sha256"
OLLAMA = "http://127.0.0.1:11434/v1"
OLLAMA_API = "http://127.0.0.1:11434/api"
K = 5
GROUPS = ("direct", "reworded", "persian")
UNSEEN = ("reworded", "persian")  # share no word with their answers (a test keeps it so); judged apart, never pooled
KEYWORD = ("bm25", "lsa", "hybrid")
NEURAL = ("embed", "hybrid_embed")
BASELINE = "hybrid"  # no model: BM25 fused with LSA
DRAWS, SEED = 2000, 35
LATENCY_MS = 1000.0  # median milliseconds to embed one question on this machine
MIN_GAIN = 0.30  # mean reworded gain: three more of ten questions found

# Download sizes read from ollama.com on 2026-10-28. Prefixes from each model's card: Qwen3-Embedding instructs the
# query only; EmbeddingGemma names the task for queries and the title for documents; BGE-M3 needs none.
CANDIDATES: dict[str, dict[str, Any]] = {
    "qwen3-embedding:0.6b": {"download": "639 MB", "mb": 639, "languages": "multilingual (Qwen3)",
                             "query_prefix": "Instruct: Given a question, retrieve the memory notes that answer it\nQuery:", "doc_prefix": ""},
    "bge-m3:567m": {"download": "1.2 GB", "mb": 1200, "languages": "multilingual (100+), 8K context",
                    "query_prefix": "", "doc_prefix": ""},
    "embeddinggemma:300m-qat-q8_0": {"download": "338 MB", "mb": 338, "languages": "100+ languages",
                                     "query_prefix": "task: search result | query: ", "doc_prefix": "title: none | text: "},
}

GATE = {
    "id": "memory_retrieval_gate_v2",
    "measure": "recall at 5 per question: the share of its answer items among the method's first five",
    "baseline": "hybrid (BM25 fused with LSA by reciprocal rank): the best search without a model",
    "choose_on_dev": ("only dev results made under this manifest, the first per candidate, every candidate measured; a (model, "
                      "method) with method embed or hybrid_embed qualifies when its dev recall on direct and on persian "
                      "questions is not below hybrid's and its median query time is at most 1000 ms; of those, the highest dev "
                      "recall on reworded questions, then on persian; a tie goes to the smaller download, then to embed (one "
                      "retriever). No qualifier: keywords stay, and the sealed set stays unread."),
    "sealed_once": ("the chosen (model, method) and hybrid on the 30 sealed questions, read once: a marker is written before the "
                    "file is opened, so a crashed run is not repeated"),
    "enter": ("embeddings enter only if all four hold on the sealed set, each a paired difference chosen minus hybrid per "
              "question: (1) reworded (10 questions): mean at least 0.30 and the 90% bootstrap interval's lower end above 0 "
              "(2,000 draws over questions, seed 35); (2) direct (10): mean not below 0; (3) persian (10): mean not below 0; "
              "(4) the median query time at most 1000 ms. Otherwise keywords stay, and the result is reported as it is. "
              "Persian recall is reported with its interval beside the verdict."),
    "limit": ("one invented corpus of 116 items and 66 questions; 10 sealed questions a group move a group's recall by 0.1 "
              "each; latency and memory are this machine's (an 8 GB MacBook Air) and change with load"),
}
CODE_FILES = ("dclab_rnd/retrieval/memory.py", "dclab_rnd/retrieval/__init__.py", "dclab_rnd/evidence_index.py")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_sha256() -> str:
    from ..campaign_capture import code_sha256 as whole

    return whole(CODE_FILES)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def items() -> list[dict[str, Any]]:
    return json.loads(ITEMS.read_text(encoding="utf-8"))["items"]


def index():
    """The corpus as an evidence index, so keyword, LSA and embedding search run exactly as they do on the evidence."""
    from ..evidence_index import EvidenceIndex

    return EvidenceIndex({"record_id": i["id"], "type": "memory", "title": "", "text": i["text"],
                          "metadata": {"kind": i["kind"], "scope": i["scope"]}} for i in items())


def questions(split: str, sealed_ok: bool = False) -> list[dict[str, Any]]:
    """The dev questions, or the sealed ones (only with ``sealed_ok``, and only while the file is the pinned one)."""
    if split == "dev":
        return json.loads(DEV.read_text(encoding="utf-8"))["questions"]
    if split != "sealed":
        raise ValueError(f"unknown split {split!r}")
    if not sealed_ok:
        raise PermissionError("the sealed questions are read once, by the sealed step")
    if sha256_of(SEALED) != SEALED_HASH.read_text(encoding="utf-8").split()[0]:
        raise SystemExit("the sealed questions changed since they were pinned: refused")
    return json.loads(SEALED.read_text(encoding="utf-8"))["questions"]


def manifest() -> dict[str, Any]:
    body = {"campaign": CAMPAIGN_ID, "package": PACKAGE, "k": K, "groups": list(GROUPS), "baseline": BASELINE,
            "methods": [*KEYWORD, *NEURAL], "candidates": CANDIDATES, "gate": GATE, "endpoint": OLLAMA,
            "latency_ms": LATENCY_MS, "min_gain": MIN_GAIN, "draws": DRAWS, "seed": SEED,
            "files": {p.name: sha256_of(p) for p in (ITEMS, DEV, SEALED)},
            "objective": ("Before any vector store: does a local embedding model find an agent's memory items better than search "
                          "without a model when the question uses other words or is asked in Persian, without losing the direct "
                          "questions?")}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def _results() -> list[Path]:
    folder = CAMPAIGN / "results"
    return sorted(folder.glob("MRT-*")) if folder.is_dir() else []


def freeze() -> dict[str, Any]:
    """Write the manifest once, before any result; afterwards refuse any change to the gate, the files or the code."""
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    path = CAMPAIGN / "manifest.json"
    want = manifest()
    if path.exists():
        held = json.loads(path.read_text(encoding="utf-8"))
        if held.get("sha256") != want["sha256"] or held.get("code_sha256") != code_sha256():
            raise SystemExit("The gate, the questions, the corpus or the code changed since the campaign was frozen: start a new campaign version")
        return held
    if _results():  # the review: a gate rewritten after dev numbers were seen is not a gate fixed before the run
        raise SystemExit("results exist without a manifest: the gate cannot be frozen after a run; start a new campaign version")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        commit = None
    held = {**want, "code_sha256": code_sha256(), "code_files": list(CODE_FILES), "git_commit": commit, "frozen_at": _now()}
    path.write_text(json.dumps(held, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return held


def _ollama(path: str, body: dict[str, Any] | None = None) -> Any:
    """Ollama's own API on this machine (the model's digest, what is loaded); None when it does not answer."""
    import urllib.request

    try:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(f"{OLLAMA_API}/{path}", data=data, headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 — a fixed local address
            return json.loads(response.read().decode())
    except Exception:  # noqa: BLE001
        return None


def model_digest(model: str) -> str | None:
    tags = _ollama("tags") or {}
    return next((m.get("digest") for m in tags.get("models", []) if m.get("name") == model or m.get("model") == model), None)


def loaded_mb(model: str) -> float | None:
    """The memory Ollama reports for the loaded model (after a run, while it is still loaded)."""
    running = _ollama("ps") or {}
    found = next((m for m in running.get("models", []) if m.get("name") == model or m.get("model") == model), None)
    return round(found["size"] / 1e6, 1) if found and found.get("size") else None


def local_embedder(model: str, folder: Path | None = None) -> Embedder:
    """An embedder whose requests go through the gateway's ``embedding`` purpose to Ollama on this machine; refused when
    the purpose would be served anywhere else (never a fallback to a paid tier). It sets the process's model settings
    to Ollama: a command-line run only, never inside the product."""
    from ..models import settings
    from ..models.gateway import Gateway

    if model not in CANDIDATES:
        raise SystemExit(f"{model} is not a candidate of {CAMPAIGN_ID}: {', '.join(CANDIDATES)}")
    os.environ.update({"DCLAB_LLM_PROVIDER": "", "OPENAI_API_KEY": "ollama", "OPENAI_BASE_URL": OLLAMA, "DCLAB_LLM_BASE_URL": OLLAMA, "OPENAI_MODEL": model})
    os.environ.update({f"DCLAB_TIER_{t.upper()}_{n}": "" for t in settings.ROUTABLE for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")})
    os.environ.pop("DCLAB_INTERN_MODEL", None)
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)  # a local model costs nothing and sends nothing away
    gateway = Gateway()
    gateway.shadows = None
    served = gateway.route("embedding")[0]
    if not settings.is_local(settings.tier(served).get("base_url") or ""):
        raise SystemExit(f"the embedding purpose would be served by the {served} tier, not by this machine: refused")
    client = gateway.client("embedding", model=model)
    if client is None:
        raise SystemExit("no model serves the embedding purpose: is Ollama running?")
    digest = model_digest(model)
    c = CANDIDATES[model]
    # the vector file is named by the weights, so a tag pulled again between dev and sealed never mixes old and new vectors
    label = f"{model}@{digest[:12]}" if digest else model
    embedder = Embedder(client, folder=folder, model=label, query_prefix=c["query_prefix"], doc_prefix=c["doc_prefix"])
    embedder.digest = digest
    return embedder


def _interval(values: list[float]) -> list[float] | None:
    import numpy as np

    if not values:
        return None
    a = np.asarray(values, dtype=float)
    draws = a[np.random.default_rng(SEED).integers(0, len(a), size=(DRAWS, len(a)))].mean(axis=1)
    return [round(float(x), 3) for x in np.quantile(draws, [0.05, 0.95])]


def measure(qs: list[dict[str, Any]], methods: tuple[str, ...], embedder: Embedder | None = None) -> dict[str, Any]:
    """Recall at K per question and method with each method's first K items; with an embedder, each question's
    embedding time (one at a time). Per group: the mean and its 90% bootstrap interval."""
    import numpy as np

    idx = index()
    build_ms = None
    if embedder is not None:
        start = time.perf_counter()
        embedder.records(idx, build=True)
        build_ms = round(1000 * (time.perf_counter() - start), 1)
    rows = []
    for q in qs:
        row: dict[str, Any] = {"id": q["id"], "group": q["group"], "answers": list(q["answers"]), "top": {}}
        if embedder is not None:
            start = time.perf_counter()
            embedder.query(q["question"])
            row["ms"] = round(1000 * (time.perf_counter() - start), 1)
        for method in methods:
            top = ranked(q["question"], method, K, index=idx, embedder=embedder)
            row["top"][method] = top
            row[method] = round(len(set(q["answers"]) & set(top)) / len(q["answers"]), 3)
        rows.append(row)

    def values(method: str, group: str | None) -> list[float]:
        return [r[method] for r in rows if group is None or r["group"] == group]

    recall = {m: {**{g: round(float(np.mean(values(m, g))), 3) if values(m, g) else 0.0 for g in GROUPS},
                  "overall": round(float(np.mean(values(m, None))), 3)} for m in methods}
    intervals = {m: {g: _interval(values(m, g)) for g in GROUPS} for m in methods}
    ms = [r["ms"] for r in rows if "ms" in r]
    return {"k": K, "questions": len(rows), "groups": {g: sum(r["group"] == g for r in rows) for g in GROUPS}, "recall": recall,
            "intervals_90": intervals, "median_query_ms": round(float(np.median(ms)), 1) if ms else None,
            "build_ms": build_ms, "rows": rows}


def _next_id() -> str:
    found = [int(m.group(1)) for p in _results() if (m := re.match(r"MRT-(\d+)_", p.name)) and int(m.group(1)) < 100]
    return f"MRT-{1 + max(found or [0]):03d}"


def _write(name: str, doc: dict[str, Any]) -> Path:
    path = CAMPAIGN / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:  # never overwritten
        json.dump(doc, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    return path


def _held() -> dict[str, Any]:
    if not (CAMPAIGN / "manifest.json").is_file():
        raise SystemExit("the campaign is not frozen: run the freeze step first")
    return freeze()  # refuses a changed gate, file or code


def _model_record(embedder: Embedder, model: str) -> dict[str, Any]:
    return {"model": model, "digest": getattr(embedder, "digest", None), "loaded_mb": loaded_mb(model), "candidate": CANDIDATES[model]}


def run_dev(model: str) -> dict[str, Any]:
    """One candidate on the dev questions, every method, stored as a new result."""
    if model not in CANDIDATES:
        raise SystemExit(f"{model} is not a candidate of {CAMPAIGN_ID}: {', '.join(CANDIDATES)}")
    held = _held()
    embedder = local_embedder(model)
    m = measure(questions("dev"), (*KEYWORD, *NEURAL), embedder)
    ident = _next_id()
    doc = {"experiment_id": ident, "id": ident, "campaign_id": CAMPAIGN_ID, "kind": "memory_retrieval_dev", "status": "completed",
           "completed_at": _now(), "split": "dev", **_model_record(embedder, model), "manifest_sha256": held["sha256"],
           "training_use": "allowed: invented corpus and hand-written dev questions",
           "question": "Recall at 5 of every method on the dev memory questions, with this candidate's embeddings.",
           "measurement": m, "limitations": [GATE["limit"], "Dev only: the choice is made here, the verdict on the sealed set."]}
    _write(f"{ident}_dev_{re.sub(r'[^A-Za-z0-9.-]+', '-', model)}.json", doc)
    return doc


def dev_results(manifest_sha256: str | None = None) -> list[dict[str, Any]]:
    """The first dev result of each candidate made under the frozen manifest (a rerun never changes the choice)."""
    if manifest_sha256 is None:
        path = CAMPAIGN / "manifest.json"
        manifest_sha256 = json.loads(path.read_text(encoding="utf-8"))["sha256"] if path.is_file() else None
    first: dict[str, dict[str, Any]] = {}
    for path in _results():
        if "_dev_" not in path.name:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("manifest_sha256") == manifest_sha256 and doc["model"] not in first:
            first[doc["model"]] = doc
    return list(first.values())


def choose(results: list[dict[str, Any]]) -> dict[str, Any]:
    """The gate's choice on dev (see ``GATE['choose_on_dev']``)."""
    measured = {r["model"] for r in results}
    missing = [c for c in CANDIDATES if c not in measured]
    if missing:
        return {"model": None, "method": None, "why": f"not measured on dev yet: {', '.join(missing)}"}
    options = []
    for r in results:
        m = r["measurement"]
        if m.get("median_query_ms") is None or m["median_query_ms"] > LATENCY_MS:
            continue
        base = m["recall"][BASELINE]
        for method in NEURAL:
            rec = m["recall"].get(method)
            if rec and rec["direct"] >= base["direct"] and rec["persian"] >= base["persian"]:
                options.append((-rec["reworded"], -rec["persian"], CANDIDATES[r["model"]]["mb"], NEURAL.index(method), r["model"], method, rec["reworded"]))
    if not options:
        return {"model": None, "method": None, "why": "no candidate kept the direct and persian questions and answered fast enough on dev: keywords stay"}
    *_, model, method, reworded = min(options)
    return {"model": model, "method": method, "why": f"highest dev recall on reworded questions ({reworded}) among the qualified"}


def verdict(rows: list[dict[str, Any]], method: str, model: str) -> dict[str, Any]:
    """The sealed verdict (see ``GATE['enter']``) on per-question rows holding the baseline's and the method's recall."""
    import numpy as np

    def diff(group: str) -> list[float]:
        return [r[method] - r[BASELINE] for r in rows if r["group"] == group]

    reworded, direct, persian = diff("reworded"), diff("direct"), diff("persian")
    low, high = _interval(reworded) or [0.0, 0.0]
    ms = [r["ms"] for r in rows if "ms" in r]
    median_ms = float(np.median(ms)) if ms else None
    mean = (lambda v: round(float(np.mean(v)), 3) if v else 0.0)  # noqa: E731
    checks = {"reworded_gain_clear": mean(reworded) >= MIN_GAIN and low > 0,
              "direct_not_worse": mean(direct) >= 0, "persian_not_worse": mean(persian) >= 0,
              "fast_enough": median_ms is not None and median_ms <= LATENCY_MS}
    failed = [k for k, ok in checks.items() if not ok]
    said = f"embeddings enter: {method} by {model}" if not failed else f"keywords stay: {', '.join(failed)} failed"
    return {"verdict": said, "checks": checks, "baseline": BASELINE,
            "reworded_mean_difference": mean(reworded), "reworded_interval_90": [round(low, 3), round(high, 3)],
            "direct_mean_difference": mean(direct), "persian_mean_difference": mean(persian),
            "persian_recall": mean([r[method] for r in rows if r["group"] == "persian"]),
            "persian_recall_interval_90": _interval([r[method] for r in rows if r["group"] == "persian"]),
            "median_query_ms": median_ms, "questions": {g: len(diff(g)) for g in GROUPS}}


def run_sealed(yes: bool = False) -> dict[str, Any]:
    """The verdict, once: the dev choice and the baseline on the sealed questions."""
    held = _held()
    marker = CAMPAIGN / "results" / "MRT-100.started"
    if marker.exists() or (CAMPAIGN / "results" / "MRT-100_sealed.json").exists():
        raise SystemExit("the sealed set was read already (MRT-100): a new campaign version is the only way to read another")
    choice = choose(dev_results(held["sha256"]))
    if choice["model"] is None and choice["why"].startswith("not measured"):
        raise SystemExit(choice["why"])
    if not yes:
        raise SystemExit("the sealed set is read once: give --yes")
    doc = {"experiment_id": "MRT-100", "id": "MRT-100", "campaign_id": CAMPAIGN_ID, "kind": "memory_retrieval_sealed", "status": "completed",
           "completed_at": _now(), "split": "sealed", "manifest_sha256": held["sha256"], "choice": choice,
           "training_use": "never: the sealed set is evaluation only",
           "question": GATE["enter"], "limitations": [GATE["limit"]]}
    if choice["model"] is None:
        doc["verdict"] = {"verdict": f"keywords stay: {choice['why']}", "checks": {}}
    else:
        embedder = local_embedder(choice["model"])
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("x", encoding="utf-8") as handle:  # the review: marked before the file is opened, so a crash is not a second chance
            handle.write(_now() + "\n")
        m = measure(questions("sealed", sealed_ok=True), (BASELINE, choice["method"]), embedder)
        doc.update(_model_record(embedder, choice["model"]))
        doc["measurement"] = m
        doc["verdict"] = verdict(m["rows"], choice["method"], choice["model"])
    _write("MRT-100_sealed.json", doc)
    return doc


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.retrieval.memory", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("freeze", help="write the gate into the campaign's manifest (once, before any run)")
    sub.add_parser("candidates", help="the models to pull into Ollama, with their sizes")
    dev = sub.add_parser("dev", help="one candidate on the dev questions")
    dev.add_argument("--ollama", required=True, metavar="MODEL", choices=list(CANDIDATES))
    sub.add_parser("choose", help="the dev choice, by the frozen gate")
    sealed = sub.add_parser("sealed", help="the verdict on the sealed questions (once)")
    sealed.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.step == "freeze":
        print(json.dumps({k: v for k, v in freeze().items() if k in ("campaign", "sha256", "code_sha256", "frozen_at")}, indent=1))
    elif args.step == "candidates":
        for name, c in CANDIDATES.items():
            print(f"ollama pull {name:32s} # {c['download']}, {c['languages']}")
    elif args.step == "dev":
        doc = run_dev(args.ollama)
        print(json.dumps({"id": doc["id"], "recall": doc["measurement"]["recall"], "median_query_ms": doc["measurement"]["median_query_ms"],
                          "loaded_mb": doc["loaded_mb"]}, indent=1))
    elif args.step == "choose":
        print(json.dumps(choose(dev_results()), indent=1))
    else:
        print(json.dumps(run_sealed(args.yes)["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
