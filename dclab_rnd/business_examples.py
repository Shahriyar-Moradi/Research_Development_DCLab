"""The business-examples knowledge base (package 30.5): ``make business-examples`` (``python -m dclab_rnd.business_examples``).

Built from what the repository already knows, each example with its source: every research track's prediction contract
and champion (``research/<track>/README.md`` and ``INDEX.md``), every replica's problem and the traps it plants
(``dclab_core/replicas``: the docstring a careful data scientist wrote for it), the owner's own fraud solution (its
README's results, when the folder is beside this one), and the examples written from public knowledge in
``dclab_rnd/knowledge/written_examples.json`` (marked ``written``: never a record of a run). Written to
``evidence/business_examples/examples.jsonl`` with its ``MANIFEST.json``; ``find(words)`` returns the closest for a
person's words. A role reads them as context only once an arm of the journeys measured it better (30.5's gate).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pkgutil
import re
import sys
from pathlib import Path
from typing import Any, Optional

from dclab_core.business_examples import BusinessExample, ExampleSource, Result, Understanding, brief, search

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "business_examples"
WRITTEN = ROOT / "dclab_rnd" / "knowledge" / "written_examples.json"
SNAP = ROOT.parent / "Snap_fruad" / "README.md"

FAMILY = {"binary": "classification", "multiclass": "classification", "regression": "regression", "stream": "streams",
          "forecasting": "forecasting", "uplift": "uplift", "ranking": "ranking", "segmentation": "segmentation"}


def _bullets(text: str) -> list[str]:
    """``* **Title.** words`` bullets as one line each: "Title: words"."""
    out = []
    for block in re.split(r"\n\s*\*\s+", "\n" + text)[1:]:
        line = " ".join(block.split())
        m = re.match(r"\*\*(.+?)\*\*\s*(.*)", line)
        out.append((f"{m.group(1).rstrip('.')}: {m.group(2)}" if m else line)[:600])
    return out


def from_replicas() -> list[BusinessExample]:
    import dclab_core.replicas as pkg

    out = []
    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: i.name):
        doc = (importlib.import_module(f"dclab_core.replicas.{info.name}").__doc__ or "").strip()
        if not doc:
            continue
        first, _, rest = doc.partition("\n\n")  # the first paragraph, however many lines it takes
        first = " ".join(first.split())
        m = re.match(r"(.+?), invented[,:]? (.+?)(?: \(an? ([a-z ]+?) replica\))?\.?$", first)
        title, problem, kind = (m.group(1), m.group(2), (m.group(3) or "").strip()) if m else (info.name.replace("_", " "), first, "")
        if "The traps a careful data scientist must see:" in doc:
            before, after = doc.split("The traps a careful data scientist must see:", 1)
            pitfalls = _bullets(after)
        elif re.search(r"\bThe traps?:", doc):  # written as a sentence: "The traps: one; two; three."
            before, after = re.split(r"\bThe traps?:", doc, maxsplit=1)
            pitfalls = [" ".join(x.split()).rstrip(".")[:600] for x in after.split(";") if x.strip()]
        else:
            before = doc  # no traps heading: the sentences that name one (a trap, a split, a leak, a copy, a must)
            pitfalls = [" ".join(x.split())[:600] for x in re.split(r"(?<=\.)\s+", rest)
                        if re.search(r"trap|split|leak|near cop|must|never", x, re.I)]
        traps = [before.split("\n\n", 1)[-1] if "\n\n" in before else "", ""]
        context = " ".join(traps[0].split())
        out.append(BusinessExample(id=f"replica:{info.name}", title=title[:200], problem=problem[:2000],
                                   understanding=Understanding(family=FAMILY.get(kind.split()[0] if kind else "", kind or None)),
                                   data=[context[:600]] if context else [], pitfalls=pitfalls[:20],
                                   source=ExampleSource(kind="replica", ref=f"dclab_core/replicas/{info.name}.py"), tags=["replica", kind] if kind else ["replica"]))
    return out


def _table(text: str, heading: str) -> dict[str, str]:
    section = text.split(heading, 1)[1] if heading in text else ""
    out = {}
    for line in section.split("\n## ")[0].splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[0] and not set(cells[0]) <= {"-", " "} and cells[0].lower() not in ("item", "dataset"):
            out[cells[0].lower()] = cells[1]
    return out


def from_tracks() -> list[BusinessExample]:
    out = []
    for readme in sorted((ROOT / "research").glob("*/README.md")):
        text = readme.read_text(encoding="utf-8")
        contract = _table(text, "## Prediction contract")
        if not contract:
            continue
        track = readme.parent.name
        title = (text.splitlines()[0].lstrip("# ").strip() or track)[:200]
        question = " ".join(text.split("## Research question", 1)[1].split("\n## ")[0].split()) if "## Research question" in text else title
        pitfalls = [f"Forbidden: {contract['forbidden']}"] if contract.get("forbidden") else []
        pitfalls += [f"Open issue: {contract['open issue']}"] if contract.get("open issue") else []
        results = []
        index = readme.parent / "INDEX.md"
        if index.exists():
            for line in index.read_text(encoding="utf-8").split("## Champion", 1)[-1].split("\n## ")[0].splitlines()[2:6]:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                try:
                    results.append(Result(metric=cells[2][:80], value=float(cells[3]), note=f"{cells[0]}: {cells[1]} {cells[4]}".strip()[:600],
                                          source=f"research/{track}/INDEX.md"))
                except (IndexError, ValueError):
                    continue
        out.append(BusinessExample(id=f"track:{track}", title=title, problem=question[:2000],
                                   understanding=Understanding(unit=contract.get("unit"), target=contract.get("target"),
                                                               moment=contract.get("prediction moment")),
                                   pitfalls=pitfalls, solution=[f"Primary metric: {contract['primary metric']}"] if contract.get("primary metric") else [],
                                   results=results[:12], source=ExampleSource(kind="research_track", ref=f"research/{track}/README.md"), tags=["track", track]))
    return out


def from_owner_fraud() -> list[BusinessExample]:
    """The owner's complete fraud solution, from its README: what it found and how it chose (no row of its data)."""
    if not SNAP.is_file():
        return []
    text = SNAP.read_text(encoding="utf-8")
    found = [" ".join(b.split())[:600] for b in re.findall(r"\* \*\*(?:Labels|What fraud looks like|Choice)\.\*\*(.+?)(?=\n\* |\n\n|\Z)", text, re.S)]
    return [BusinessExample(id="owner:delivery_fraud", title="Delivery fraud by bikers and customers (the owner's)", industry="delivery",
                            problem="Detect fraudulent rides by bikers and customers in one city's ride table; a small sample of known forged rides measures it.",
                            understanding=Understanding(unit="One delivered ride", target="A ride in the known fraud sample", labels="A sample of the cases found: positive-unlabelled",
                                                        family="rare_event"),
                            pitfalls=found[:3] + ["Joining the cases on the customer instead of the order turned 94 frauds into 3,416."],
                            solution=["Rules, anomaly detectors, supervised and positive-unlabelled models, compared on biker-grouped folds with a cluster bootstrap and paired tests",
                                      "PU bagging chosen: tied with the best on average precision, best ROC-AUC, the labels being a sample"],
                            results=[Result(metric="average_precision", value=0.095, note="random forest, biker-grouped CV", source="Snap_fruad/README.md"),
                                     Result(metric="average_precision", value=0.081, note="PU bagging (selected), biker-grouped CV", source="Snap_fruad/README.md")],
                            source=ExampleSource(kind="owner_solution", ref="Snap_fruad/README.md"), tags=["fraud", "positive-unlabelled", "owner"])]


def from_written() -> list[BusinessExample]:
    if not WRITTEN.is_file():
        return []
    return [BusinessExample.model_validate({**{k: v for k, v in d.items() if k != "ref"}, "source": {"kind": "written", "ref": d.get("ref") or "public knowledge"}})
            for d in json.loads(WRITTEN.read_text(encoding="utf-8"))]


def build(out: Path = OUT) -> dict[str, Any]:
    examples = [*from_tracks(), *from_replicas(), *from_owner_fraud(), *from_written()]
    out.mkdir(parents=True, exist_ok=True)
    lines = "".join(e.model_dump_json() + "\n" for e in examples)
    (out / "examples.jsonl").write_text(lines, encoding="utf-8")
    manifest = {"examples": len(examples), "by_source": {k: sum(1 for e in examples if e.source.kind == k) for k in sorted({e.source.kind for e in examples})},
                "sha256": hashlib.sha256(lines.encode()).hexdigest(), "rule": "An example states its source; a written one is composed from public "
                "knowledge and never a record of a run; a role reads them only after an arm measured it better."}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def load(out: Path = OUT) -> list[BusinessExample]:
    path = out / "examples.jsonl"
    return [BusinessExample.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []


def find(words: str, top: int = 3) -> list[dict[str, Any]]:
    """The examples closest to a person's words, as a role reads them."""
    return [{**brief(e), "score": s} for e, s in search(load(), words, top)]


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.business_examples", description=__doc__.split("\n")[0])
    parser.add_argument("--find", default=None, help="print the closest examples for these words")
    args = parser.parse_args(argv)
    if args.find:
        print(json.dumps(find(args.find), indent=1, ensure_ascii=False))
        return 0
    print(json.dumps(build(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
