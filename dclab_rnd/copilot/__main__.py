"""CLI for the DCLab notebook copilot.

    python -m dclab_rnd.copilot review NOTEBOOK.ipynb                 # readable report in the terminal
    python -m dclab_rnd.copilot review NOTEBOOK.ipynb --json          # machine-readable findings
    python -m dclab_rnd.copilot review NOTEBOOK.ipynb --html out.html # "dclab notebook" review view
    python -m dclab_rnd.copilot review NOTEBOOK.ipynb --annotate out.ipynb  # copy with review cells inserted
    python -m dclab_rnd.copilot review NOTEBOOK.ipynb --fail-on high # exit 1 when such findings exist (CI gate)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyzer import SEVERITY_ORDER, review_notebook
from .render import annotate_notebook, to_html, to_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.copilot", description="Evidence-backed notebook review")
    sub = parser.add_subparsers(dest="command", required=True)
    review = sub.add_parser("review", help="review one notebook")
    review.add_argument("notebook", type=Path)
    review.add_argument("--json", action="store_true", help="print findings as JSON")
    review.add_argument("--markdown", type=Path, help="write a Markdown report")
    review.add_argument("--html", type=Path, help="write the HTML review view")
    review.add_argument("--annotate", type=Path, help="write a copy of the notebook with review cells")
    review.add_argument("--fail-on", choices=list(SEVERITY_ORDER), help="exit 1 if any finding is at least this severe")
    args = parser.parse_args(argv)

    report = review_notebook(args.notebook)
    if args.markdown:
        args.markdown.write_text(to_markdown(report), encoding="utf-8")
    if args.html:
        args.html.write_text(to_html(report), encoding="utf-8")
    if args.annotate:
        if args.annotate.resolve() == args.notebook.resolve():
            parser.error("--annotate must write to a new file; the original notebook is never modified")
        args.annotate.write_text(json.dumps(annotate_notebook(args.notebook, report), indent=1) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps({k: v for k, v in report.items() if k != "cells"}, indent=2, ensure_ascii=False))
    else:
        print(to_markdown(report))
    if args.fail_on:
        threshold = SEVERITY_ORDER[args.fail_on]
        if any(SEVERITY_ORDER[f["severity"]] <= threshold for f in report["findings"]):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
