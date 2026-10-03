"""Render copilot reviews as Markdown, an annotated notebook, or the "dclab notebook" HTML view."""

from __future__ import annotations

import copy
import html
import json
import re
from pathlib import Path
from typing import Any

SEVERITY_LABEL = {"high": "High", "medium": "Medium", "low": "Low", "info": "Note"}


# --------------------------------------------------------------------------- markdown


def to_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    counts = ", ".join(f"{v} {k}" for k, v in sorted(s["by_severity"].items(), key=lambda kv: ["high", "medium", "low", "info"].index(kv[0])))
    lines = [f"# Copilot review: {report.get('notebook', 'notebook')}", "",
             f"{s['findings']} findings ({counts or 'none'}) across {s['code_cells']} code cells.", ""]
    for f in report["findings"]:
        lines += [f"## Cell {f['cell']}, line {f['line']} · {SEVERITY_LABEL[f['severity']]} · {f['title']}", "",
                  f["message"], "", f"**Fix.** {f['suggestion']}", "", "**Proof.**", ""]
        for p in f["proof"]:
            lines.append(f"- `{p['record_id']}` ({p['type']}): {p['title']} — cites {', '.join(c for c in p['citations'] if c)}")
        lines.append("")
    lines += ["---", "", *[f"- {lim}" for lim in report.get("limitations", [])], ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- annotated notebook


def annotate_notebook(source_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the notebook with a review markdown cell after each flagged cell (original untouched)."""
    nb = json.loads(Path(source_path).read_text(encoding="utf-8"))
    by_cell: dict[int, list[dict[str, Any]]] = {}
    for f in report["findings"]:
        by_cell.setdefault(f["cell"], []).append(f)
    cells = []
    for i, cell in enumerate(nb.get("cells", [])):
        cells.append(cell)
        if i in by_cell:
            body = ["> **DCLab copilot review**\n", ">\n"]
            for f in by_cell[i]:
                proof = ", ".join(f"`{p['record_id']}`" for p in f["proof"])
                body += [f"> **{SEVERITY_LABEL[f['severity']]} · line {f['line']} · {f['title']}.** {f['message']}\n", ">\n",
                         f"> *Fix:* {f['suggestion']}\n", ">\n", f"> *Proof:* {proof}\n", ">\n"]
            cells.append({"cell_type": "markdown", "metadata": {"dclab_copilot": True}, "source": body})
    out = copy.deepcopy(nb)
    out["cells"] = cells
    return out


# --------------------------------------------------------------------------- HTML


def _md_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def _markdown_cell(source: str) -> str:
    out, in_code, buf = [], False, []
    for line in source.splitlines():
        if line.startswith("```"):
            if in_code:
                out.append(f"<pre class='md-code'>{html.escape(chr(10).join(buf))}</pre>")
                buf = []
            in_code = not in_code
            continue
        if in_code:
            buf.append(line)
        elif line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 3)
            out.append(f"<h{level + 1} class='md-h'>{_md_inline(line.lstrip('#').strip())}</h{level + 1}>")
        elif line.strip():
            out.append(f"<p>{_md_inline(line)}</p>")
    return "\n".join(out)


def _code_cell(source: str, flagged: dict[int, str]) -> str:
    rows = []
    for n, line in enumerate(source.splitlines() or [""], 1):
        sev = flagged.get(n)
        cls = f" class='flag flag-{sev}'" if sev else ""
        rows.append(f"<tr{cls}><td class='ln'>{n}</td><td class='src'>{html.escape(line) or '&nbsp;'}</td></tr>")
    return "<div class='code-wrap'><table class='code'>" + "".join(rows) + "</table></div>"


def _finding_card(f: dict[str, Any], fid: str) -> str:
    proof = []
    for p in f["proof"]:
        cites = ", ".join(f"<code>{html.escape(c)}</code>" for c in p["citations"] if c)
        proof.append(
            f"<li><span class='pid'>{html.escape(p['record_id'])}</span> <span class='ptype'>{html.escape(p['type'].replace('_', ' '))}</span>"
            f"<p class='ptitle'>{html.escape(p['title'])}</p><p class='ptext'>{html.escape(p['text'][:900])}{'…' if len(p['text']) > 900 else ''}</p>"
            f"<p class='pcite'>Source: {cites}</p></li>"
        )
    return (
        f"<article class='note sev-{f['severity']}' id='{fid}' data-sev='{f['severity']}'>"
        f"<header><span class='chip chip-{f['severity']}'>{SEVERITY_LABEL[f['severity']]}</span>"
        f"<span class='where'>line {f['line']}</span></header>"
        f"<h3>{_md_inline(f['title'])}</h3><p>{_md_inline(f['message'])}</p>"
        f"<p class='fix'><span class='label'>Fix</span>{_md_inline(f['suggestion'])}</p>"
        f"<details><summary>Show proof · {len(f['proof'])} records</summary><ul class='proof'>{''.join(proof)}</ul></details>"
        "</article>"
    )


CSS = """
/* Layout: a lab notebook with a review margin. Cells on the left, evidence-backed notes pinned beside them. */
:root{
  --paper:#f2f5f6; --sheet:#ffffff; --ink:#15212b; --muted:#5b6b76; --rule:#d5dee3;
  --accent:#0d6b70; --accent-soft:#e1efef; --code-bg:#f7f9fa;
  --high:#b4232f; --high-soft:#fbe9ea; --medium:#a25a00; --medium-soft:#fdf0dc;
  --low:#4a5a67; --low-soft:#eaeff2; --info:#1f5f9e; --info-soft:#e4eef8;
  --display:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif;
  --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"JetBrains Mono","SFMono-Regular",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --paper:#0f1519; --sheet:#151e24; --ink:#e3eaee; --muted:#93a3ad; --rule:#26323a;
  --accent:#5cc2c4; --accent-soft:#163236; --code-bg:#111a1f;
  --high:#f0707a; --high-soft:#3a1a1e; --medium:#e9a74c; --medium-soft:#36270f;
  --low:#a6b4be; --low-soft:#1f2a31; --info:#7fb4ea; --info-soft:#16283a; color-scheme:dark}}
:root[data-theme="dark"]{
  --paper:#0f1519; --sheet:#151e24; --ink:#e3eaee; --muted:#93a3ad; --rule:#26323a;
  --accent:#5cc2c4; --accent-soft:#163236; --code-bg:#111a1f;
  --high:#f0707a; --high-soft:#3a1a1e; --medium:#e9a74c; --medium-soft:#36270f;
  --low:#a6b4be; --low-soft:#1f2a31; --info:#7fb4ea; --info-soft:#16283a; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font:15px/1.55 var(--body);padding:0 16px 48px}
.wrap{max-width:1240px;margin:0 auto}
.top{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--paper);border-bottom:1px solid var(--rule);
  padding-block:12px;display:flex;flex-wrap:wrap;gap:12px 20px;align-items:center;justify-content:space-between}
.brand{display:flex;gap:10px;align-items:baseline;min-width:0}
.brand b{font:600 13px/1 var(--display);letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
.brand span{font-family:var(--mono);font-size:13px;color:var(--muted);overflow-wrap:anywhere}
.agent{font-size:13px;color:var(--muted);display:flex;align-items:center;gap:8px}
.agent i{width:8px;height:8px;border-radius:50%;background:var(--accent);display:inline-block}
.filters{display:flex;flex-wrap:wrap;gap:6px}
.filters button{font:500 13px/1 var(--body);border:1px solid var(--rule);background:var(--sheet);color:var(--ink);
  padding:7px 10px;border-radius:999px;cursor:pointer;font-variant-numeric:tabular-nums}
.filters button[aria-pressed="true"]{border-color:var(--accent);background:var(--accent-soft)}
.filters button:focus-visible,summary:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.intro{padding-block:28px 8px;max-width:72ch}
.intro h1{font:600 clamp(26px,4vw,36px)/1.1 var(--display);margin:0 0 10px;text-wrap:balance}
.intro p{margin:0;color:var(--muted)}
.index{margin:18px 0 28px;border-top:1px solid var(--rule)}
.index a{display:grid;grid-template-columns:5.5em 4.5em 1fr;gap:12px;padding:9px 0;border-bottom:1px solid var(--rule);
  color:var(--ink);text-decoration:none;font-size:14px}
.index a:hover .t{color:var(--accent)}
.index .c{font-family:var(--mono);font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.row{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:20px;margin-bottom:18px;align-items:start}
.row.md .cell{background:transparent;border:none;padding:0 4px}
.cell{background:var(--sheet);border:1px solid var(--rule);border-radius:6px;min-width:0}
.cell-head{display:flex;justify-content:space-between;font:500 11px/1 var(--display);letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);padding:10px 12px 0}
.code-wrap{overflow-x:auto;padding:8px 0 10px}
table.code{border-collapse:collapse;font:13px/1.6 var(--mono);width:100%}
.code td{padding:0 12px;white-space:pre;vertical-align:top}
.code td.ln{width:1%;color:var(--muted);text-align:right;user-select:none;padding-right:10px;border-right:1px solid var(--rule)}
.code tr.flag-high td{background:var(--high-soft)} .code tr.flag-medium td{background:var(--medium-soft)}
.code tr.flag-low td{background:var(--low-soft)} .code tr.flag-info td{background:var(--info-soft)}
.md-h{font:600 20px/1.2 var(--display);margin:6px 0 8px} p{margin:0 0 8px}
.md-code,code{font-family:var(--mono);font-size:12.5px;background:var(--code-bg);border-radius:4px}
code{padding:1px 4px} .md-code{padding:10px 12px;overflow-x:auto;border:1px solid var(--rule)}
.margin{display:flex;flex-direction:column;gap:10px;min-width:0}
.margin.empty{color:var(--muted);font-size:13px;padding-top:12px}
.note{background:var(--sheet);border:1px solid var(--rule);border-left:4px solid var(--low);border-radius:6px;padding:12px 14px;min-width:0}
.note.sev-high{border-left-color:var(--high)} .note.sev-medium{border-left-color:var(--medium)} .note.sev-info{border-left-color:var(--info)}
.note header{display:flex;gap:8px;align-items:center;margin-bottom:6px}
.note h3{font:600 16px/1.3 var(--display);margin:0 0 6px}
.note p{font-size:14px}
.chip{font:600 11px/1 var(--display);letter-spacing:.06em;text-transform:uppercase;padding:4px 7px;border-radius:4px}
.chip-high{background:var(--high-soft);color:var(--high)} .chip-medium{background:var(--medium-soft);color:var(--medium)}
.chip-low{background:var(--low-soft);color:var(--low)} .chip-info{background:var(--info-soft);color:var(--info)}
.where{font-family:var(--mono);font-size:12px;color:var(--muted)}
.fix{background:var(--accent-soft);border-radius:4px;padding:8px 10px}
.label{font:600 11px/1 var(--display);letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-right:8px}
details{margin-top:6px} summary{cursor:pointer;color:var(--accent);font-size:13px;font-weight:500}
.proof{list-style:none;margin:10px 0 0;padding:0;display:flex;flex-direction:column;gap:10px}
.proof li{border-top:1px solid var(--rule);padding-top:8px}
.pid{font-family:var(--mono);font-size:12px;font-weight:600} .ptype{font-size:12px;color:var(--muted);margin-left:6px}
.ptitle{font-weight:600;font-size:13.5px;margin:4px 0} .ptext{font-size:13px;color:var(--muted)}
.pcite{font-size:12px;color:var(--muted);overflow-wrap:anywhere}
.foot{margin-top:32px;border-top:1px solid var(--rule);padding-top:14px;color:var(--muted);font-size:13px;max-width:80ch}
@media (max-width:820px){.row{grid-template-columns:minmax(0,1fr)}.index a{grid-template-columns:4.5em 4em 1fr}}
@media (prefers-reduced-motion:no-preference){.note{transition:opacity .15s}}
"""

JS = """
(function(){
  var buttons=[].slice.call(document.querySelectorAll('.filters button'));
  function apply(sev){
    buttons.forEach(function(b){b.setAttribute('aria-pressed', String(b.dataset.sev===sev));});
    document.querySelectorAll('.note').forEach(function(n){n.hidden = sev!=='all' && n.dataset.sev!==sev;});
    document.querySelectorAll('.index a').forEach(function(a){a.hidden = sev!=='all' && a.dataset.sev!==sev;});
    try{localStorage.setItem('dclab-copilot-filter',sev);}catch(e){}
  }
  buttons.forEach(function(b){b.addEventListener('click',function(){apply(b.dataset.sev);});});
  var saved='all'; try{saved=localStorage.getItem('dclab-copilot-filter')||'all';}catch(e){}
  if(!buttons.some(function(b){return b.dataset.sev===saved;})) saved='all';
  apply(saved);
})();
"""


def to_html(report: dict[str, Any], title: str = "DCLab Notebook Review") -> str:
    s = report["summary"]
    order = ["high", "medium", "low", "info"]
    counts = s["by_severity"]
    by_cell: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    index_rows = []
    for n, f in enumerate(report["findings"]):
        fid = f"f{n}"
        by_cell.setdefault(f["cell"], []).append((fid, f))
        index_rows.append(
            f"<a href='#{fid}' data-sev='{f['severity']}'><span class='c'>cell {f['cell']}</span>"
            f"<span class='chip chip-{f['severity']}'>{SEVERITY_LABEL[f['severity']]}</span><span class='t'>{_md_inline(f['title'])}</span></a>"
        )
    filters = [f"<button type='button' data-sev='all' aria-pressed='true'>All {s['findings']}</button>"]
    filters += [f"<button type='button' data-sev='{k}' aria-pressed='false'>{SEVERITY_LABEL[k]} {counts[k]}</button>" for k in order if counts.get(k)]
    rows = []
    for cell in report.get("cells", []):
        notes = by_cell.get(cell["index"], [])
        if cell["type"] == "markdown":
            rows.append(f"<section class='row md'><div class='cell'>{_markdown_cell(cell['source'])}</div><div class='margin'></div></section>")
            continue
        flagged: dict[int, str] = {}
        for _, f in notes:
            if f["line"] not in flagged or order.index(f["severity"]) < order.index(flagged[f["line"]]):
                flagged[f["line"]] = f["severity"]
        margin = "".join(_finding_card(f, fid) for fid, f in notes) or "No issues found in this cell."
        rows.append(
            f"<section class='row'><div class='cell'><div class='cell-head'><span>In [{cell['index']}]</span>"
            f"<span>{len(notes)} note{'s' if len(notes) != 1 else ''}</span></div>{_code_cell(cell['source'], flagged)}</div>"
            f"<div class='margin{' empty' if not notes else ''}'>{margin}</div></section>"
        )
    high = counts.get("high", 0)
    headline = (f"{high} problem{'s' if high != 1 else ''} would change the reported score" if high
                else "No score-changing problems found")
    limitations = " ".join(html.escape(x) for x in report.get("limitations", []))
    return f"""<title>{html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
  <div class="top">
    <div class="brand"><b>dclab notebook</b><span>{html.escape(report.get('notebook', 'notebook.ipynb'))}</span></div>
    <div class="filters" role="group" aria-label="Filter notes by severity">{''.join(filters)}</div>
    <div class="agent"><i aria-hidden="true"></i>Reviewed against {s['code_cells']} code cells · static, nothing executed</div>
  </div>
  <header class="intro">
    <h1>{headline}</h1>
    <p>Every note links to a DCLab rule and to a measured precedent from the R&amp;D campaigns. Open “Show proof” to read the evidence and where it came from.</p>
  </header>
  <nav class="index" aria-label="All findings">{''.join(index_rows)}</nav>
  {''.join(rows)}
  <p class="foot">{limitations}</p>
</div>
<script>{JS}</script>
"""
