#!/usr/bin/env python3
"""Build the DCLab R&D product demo into one self-contained HTML page.

    python docs/product-demo/build.py                 # writes docs/product-demo/index.html
    python docs/product-demo/build.py --refresh       # first re-extract the real evidence records,
                                                      # the copilot review and one SFT example
    python docs/product-demo/build.py --artifact OUT  # also write the page without the document
                                                      # skeleton (the shape the Artifact viewer expects)
    python docs/product-demo/build.py --check         # exit 1 if index.html is stale

Sources live in docs/product-demo/src/: shell.html (layout), styles.css (design system),
core.js (router, blueprint layer, tours, charts), features.js (what exists vs what is planned),
data.js (sample data), data/*.json (extracted from the repository) and views/*.html (one per page;
each view's <script> blocks are moved to the end of the page).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
ROOT = HERE.parents[1]
OUT = HERE / "index.html"
VIEWS = ["home", "new", "project", "contract", "notebook", "audit", "models", "reliability", "brief",
         "intern", "compute", "evidence", "lab", "benchmark", "policy", "packs", "integrations", "admin", "blueprint"]
HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
        '<meta name="color-scheme" content="light dark">\n')


def refresh() -> None:
    """Re-extract the parts of the demo that come straight from the repository."""
    records = []
    for line in (ROOT / "evidence/knowledge/rag/records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        meta = {k: v for k, v in (r.get("metadata") or {}).items() if isinstance(v, (str, int, float, bool, list))}
        records.append({"id": r["record_id"], "type": r["type"], "title": r["title"], "text": r["text"],
                        "citations": (r.get("citations") or [])[:6], "meta": meta})
    write_json("records.json", records)

    notebook = ROOT / "dclab_rnd/copilot/examples/leaky_bank_marketing.ipynb"
    run = subprocess.run([sys.executable, "-m", "dclab_rnd.copilot", "review", str(notebook), "--json"],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    review = json.loads(run.stdout)
    cells = [{"type": c["cell_type"], "source": "".join(c["source"])}
             for c in json.loads(notebook.read_text(encoding="utf-8"))["cells"]]
    findings = []
    for f in review["findings"]:
        item = {k: f.get(k) for k in ("detector", "severity", "cell", "line", "title", "message", "suggestion", "rules")}
        item["proof"] = [p["record_id"] for p in f.get("proof", [])]
        findings.append(item)
    write_json("copilot_review.json", {"summary": review["summary"], "cells": cells, "findings": findings})

    sft = ROOT / "research/llm-fine-tuning/experiments/sft/out_v3/train.chat.jsonl"
    example = None
    for line in sft.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (row.get("metadata") or {}).get("task") == "leakage_judgment":
            example = row
            break
    if example is None:
        example = json.loads(sft.read_text(encoding="utf-8").splitlines()[0])
    manifest = json.loads((sft.parent / "MANIFEST.json").read_text(encoding="utf-8"))
    write_json("sft.json", {"example": example, "manifest": {k: manifest.get(k) for k in
                ("total", "train", "val", "by_task", "validation_policy", "quality_gates", "seed")}})
    print(f"refreshed {len(records)} records, {len(findings)} copilot findings, 1 SFT example")


def write_json(name: str, data) -> None:
    (SRC / "data").mkdir(exist_ok=True)
    (SRC / "data" / name).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def js_literal(name: str) -> str:
    path = SRC / "data" / name
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def assemble() -> tuple[str, str]:
    """Return (head_part, body_part) of the page."""
    shell = (SRC / "shell.html").read_text(encoding="utf-8")
    css = (SRC / "styles.css").read_text(encoding="utf-8")
    markup, scripts = [], []
    for name in VIEWS:
        path = SRC / "views" / f"{name}.html"
        if not path.exists():
            markup.append(f'<section class="view" id="view-{name}" hidden><div class="empty">This page is not written yet.</div></section>')
            continue
        text = path.read_text(encoding="utf-8")
        scripts += [s.strip() for s in re.findall(r"<script>(.*?)</script>", text, flags=re.S)]
        markup.append(re.sub(r"<script>.*?</script>", "", text, flags=re.S).strip())
    data = ("window.DEMO_RECORDS = " + js_literal("records.json") + ";\n"
            "window.DEMO_REVIEW = " + js_literal("copilot_review.json") + ";\n"
            "window.DEMO_SFT = " + js_literal("sft.json") + ";\n")
    blocks = [data, (SRC / "data.js").read_text(encoding="utf-8"), (SRC / "features.js").read_text(encoding="utf-8"),
              (SRC / "core.js").read_text(encoding="utf-8")] + scripts
    script_html = "\n".join(f"<script>\n{b}\n</script>" for b in blocks)
    page = (shell.replace("/*STYLES*/", css)
                 .replace("<!--VIEWS-->", "\n".join(markup))
                 .replace("<!--SCRIPTS-->", script_html))
    cut = page.index('<div class="app">')
    return page[:cut], page[cut:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refresh", action="store_true", help="re-extract records, copilot review and SFT example")
    parser.add_argument("--artifact", type=Path, help="also write the skeleton-free page here")
    parser.add_argument("--check", action="store_true", help="exit 1 if index.html is out of date")
    args = parser.parse_args(argv)
    if args.refresh:
        refresh()
    head, body = assemble()
    full = f"{HEAD}{head}</head>\n<body>\n{body}</body>\n</html>\n"
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != full:
            print("docs/product-demo/index.html is stale: run python docs/product-demo/build.py", file=sys.stderr)
            return 1
        print("product demo is up to date")
        return 0
    OUT.write_text(full, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(full) / 1024:.0f} KB)")
    if args.artifact:
        args.artifact.write_text(head + body, encoding="utf-8")
        print(f"wrote {args.artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
