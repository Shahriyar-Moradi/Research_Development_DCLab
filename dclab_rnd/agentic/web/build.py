#!/usr/bin/env python3
"""Build the DCLab product frontend into static files the server can send under a strict CSP.

    python -m dclab_rnd.agentic.web.build           # writes dclab_rnd/agentic/static/app/
    python -m dclab_rnd.agentic.web.build --check   # exit 1 if static/app/ is stale

Sources live in dclab_rnd/agentic/web/src/, copied from demo v1 (docs/product-demo/src, frozen):
shell.html (layout), styles.css (design system), core.js (router, page tabs, server API, charts),
data.js and features.js (sample data and the blueprint registry), data/*.json (records extracted from
the repository), fonts/ (copied as they are) and views/*.html (one per page). Unlike the demo build, nothing is inlined: the page
links app.css and loads every script from a file, because the server's CSP is script-src 'self' and
style-src 'self'. Per-element styles are written as data-style="" and applied by core.js.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
OUT = HERE.parent / "static" / "app"
VIEWS = ["home", "new", "project", "solution", "notebook", "audit", "models", "reliability", "brief",
         "intern", "compute", "evidence", "lab", "benchmark", "policy", "packs", "integrations", "admin", "blueprint"]
DATA = [("records.js", "DEMO_RECORDS", "records.json"), ("review.js", "DEMO_REVIEW", "copilot_review.json"),
        ("sft.js", "DEMO_SFT", "sft.json")]
HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
        '<meta name="color-scheme" content="light dark">\n')
INLINE_STYLE = re.compile(r'(?<![-\w])style\s*=\s*["\']', re.I)


def js_data(global_name: str, json_name: str) -> str:
    path = SRC / "data" / json_name
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return f"window.{global_name} = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/") + ";\n"


def assemble() -> dict[str, str]:
    """Return {relative output path: content} for the whole app."""
    files: dict[str, str] = {}
    markup, view_scripts = [], []
    for name in VIEWS:
        text = (SRC / "views" / f"{name}.html").read_text(encoding="utf-8")
        scripts = [s.strip() for s in re.findall(r"<script>(.*?)</script>", text, flags=re.S)]
        markup.append(re.sub(r"<script>.*?</script>", "", text, flags=re.S).strip())
        if scripts:
            files[f"js/views/{name}.js"] = "\n\n".join(scripts) + "\n"
            view_scripts.append(f"js/views/{name}.js")
    for out_name, global_name, json_name in DATA:
        files[f"js/{out_name}"] = js_data(global_name, json_name)
    files["js/data.js"] = (SRC / "data.js").read_text(encoding="utf-8")
    files["js/features.js"] = (SRC / "features.js").read_text(encoding="utf-8")
    files["js/core.js"] = (SRC / "core.js").read_text(encoding="utf-8")
    files["app.css"] = (SRC / "styles.css").read_text(encoding="utf-8")

    shell = (SRC / "shell.html").read_text(encoding="utf-8")
    shell = re.sub(r"<style>\s*/\*STYLES\*/\s*</style>", '<link rel="stylesheet" href="/static/app/app.css">', shell)
    order = [f"js/{d[0]}" for d in DATA] + ["js/data.js", "js/features.js", "js/core.js"] + view_scripts
    tags = "\n".join(f'<script src="/static/app/{path}"></script>' for path in order)
    page = shell.replace("<!--VIEWS-->", "\n".join(markup)).replace("<!--SCRIPTS-->", tags)
    cut = page.index('<div class="app">')
    files["index.html"] = f"{HEAD}{page[:cut]}</head>\n<body>\n{page[cut:]}</body>\n</html>\n"
    return files


def assets() -> dict[str, bytes]:
    """Binary files copied as they are: the fonts (SIL Open Font License; see fonts/README.md)."""
    folder = SRC / "fonts"
    return {f"fonts/{p.name}": p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file()} if folder.is_dir() else {}


def problems(files: dict[str, str]) -> list[str]:
    """Things the CSP would block: inline scripts, inline style blocks or attributes."""
    found = []
    page = files["index.html"]
    if re.search(r"<script(?![^>]*\bsrc=)[^>]*>", page):
        found.append("index.html has an inline <script>")
    if re.search(r"<style[\s>]", page):
        found.append("index.html has an inline <style>")
    if re.search(r"\son[a-z]+\s*=", re.sub(r"<script.*?</script>", "", page, flags=re.S)):
        found.append("index.html has an inline event handler")
    for path, text in files.items():
        if path.endswith((".html", ".js")) and INLINE_STYLE.search(text):
            found.append(f"{path} writes an inline style attribute (use data-style)")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if static/app/ is out of date")
    args = parser.parse_args(argv)
    files = assemble()
    binary = assets()
    issues = problems(files)
    if issues:
        print("\n".join(issues), file=sys.stderr)
        return 1
    if args.check:
        stale = [p for p, text in files.items() if not (OUT / p).exists() or (OUT / p).read_text(encoding="utf-8") != text]
        stale += [p for p, data in binary.items() if not (OUT / p).exists() or (OUT / p).read_bytes() != data]
        extra = [str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file() and str(p.relative_to(OUT)) not in files and str(p.relative_to(OUT)) not in binary] if OUT.exists() else []
        if stale or extra:
            print("dclab_rnd/agentic/static/app is stale: run python -m dclab_rnd.agentic.web.build"
                  + (f"\n  changed: {', '.join(stale)}" if stale else "") + (f"\n  extra: {', '.join(extra)}" if extra else ""), file=sys.stderr)
            return 1
        print("product frontend is up to date")
        return 0
    for path, text in files.items():
        target = OUT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    for path, data in binary.items():
        target = OUT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    if OUT.exists():
        for p in OUT.rglob("*"):
            if p.is_file() and str(p.relative_to(OUT)) not in files and str(p.relative_to(OUT)) not in binary:
                p.unlink()
    size = sum(len(t.encode("utf-8")) for t in files.values()) + sum(len(d) for d in binary.values())
    print(f"wrote dclab_rnd/agentic/static/app ({len(files) + len(binary)} files, {size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
