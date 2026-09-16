"""Generate executive PDF report from MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.md."""

from __future__ import annotations

import base64
import os
import re
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.md"
PDF_PATH = ROOT / "MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.pdf"
HTML_TEMP = ROOT / "_report_preview.html"

# Base64 encode an image
def encode_image(img_rel_path: str) -> str:
    img_path = (ROOT / img_rel_path).resolve()
    if not img_path.exists():
        # fallback check under optimized_safe_model
        img_path = (ROOT / "optimized_safe_model" / img_rel_path).resolve()
    if img_path.exists():
        data = base64.b64encode(img_path.read_bytes()).decode("utf-8")
        return f"data:image/png;base64,{data}"
    print(f"Warning: image not found at {img_rel_path}", file=sys.stderr)
    return img_rel_path

def convert_md_to_html(md_text: str) -> str:
    # 1. Replace image paths with base64 data URIs
    def repl_img(match):
        alt = match.group(1)
        src = match.group(2)
        b64 = encode_image(src)
        return f'<div class="figure-container"><img alt="{alt}" src="{b64}"><div class="figure-caption">{alt}</div></div>'

    processed_md = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', repl_img, md_text)

    # 2. Render markdown using markdown-it
    md = MarkdownIt("gfm-like")
    content_html = md.render(processed_md)

    # Style and wrap
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HyperAck Master Classification & Optimization Report</title>
<!-- MathJax for rendering math equations -->
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  }},
  svg: {{
    fontCache: 'global'
  }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<style>
  @page {{
    size: A4 portrait;
    margin: 18mm 16mm 18mm 16mm;
    @bottom-right {{
      content: counter(page);
    }}
  }}

  *, *:before, *:after {{
    box-sizing: border-box;
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #111827;
    background: #FFFFFF;
    line-height: 1.55;
    font-size: 10.5pt;
    margin: 0;
    padding: 0;
    -webkit-font-smoothing: antialiased;
  }}

  /* Headings */
  h1, h2, h3, h4 {{
    color: #0F172A;
    font-weight: 700;
    line-height: 1.25;
    margin-top: 1.6em;
    margin-bottom: 0.6em;
    page-break-after: avoid;
    break-after: avoid;
  }}

  h1 {{
    font-size: 20pt;
    border-bottom: 2px solid #0F172A;
    padding-bottom: 8px;
    margin-top: 0;
  }}

  h2 {{
    font-size: 14pt;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 6px;
    margin-top: 1.8em;
  }}

  h3 {{
    font-size: 11.5pt;
    color: #1E293B;
  }}

  h4 {{
    font-size: 10.5pt;
  }}

  p, ul, ol {{
    margin-top: 0;
    margin-bottom: 0.85em;
  }}

  li {{
    margin-bottom: 0.35em;
  }}

  /* Tables */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1.2em 0 1.5em 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  thead {{
    display: table-header-group;
  }}

  th {{
    background: #0F172A;
    color: #FFFFFF;
    font-weight: 600;
    text-align: left;
    padding: 7px 9px;
    border: 1px solid #0F172A;
    font-size: 8.5pt;
    letter-spacing: 0.2px;
  }}

  th:not(:first-child), td:not(:first-child) {{
    text-align: left;
  }}

  td {{
    padding: 5.5px 8px;
    border: 1px solid #E2E8F0;
    color: #1E293B;
  }}

  tbody tr:nth-child(even) {{
    background: #F8FAFC;
  }}

  tbody tr:hover {{
    background: #F1F5F9;
  }}

  /* Code blocks & ASCII art */
  pre {{
    background: #0F172A;
    color: #E2E8F0;
    padding: 10px 14px;
    border-radius: 6px;
    font-family: "SF Mono", Menlo, Consolas, Monaco, monospace;
    font-size: 8.5pt;
    line-height: 1.45;
    overflow-x: auto;
    page-break-inside: avoid;
    break-inside: avoid;
    margin: 1em 0;
    border: 1px solid #1E293B;
  }}

  code {{
    font-family: "SF Mono", Menlo, Consolas, Monaco, monospace;
    font-size: 8.5pt;
    background: #F1F5F9;
    color: #0F172A;
    padding: 2px 4.5px;
    border-radius: 4px;
    border: 1px solid #E2E8F0;
  }}

  pre code {{
    background: transparent;
    color: inherit;
    padding: 0;
    border: none;
    font-size: inherit;
  }}

  /* Blockquotes & Callouts */
  blockquote {{
    margin: 1.2em 0;
    padding: 10px 16px;
    background: #F8FAFC;
    border-left: 4px solid #2563EB;
    color: #334155;
    font-style: italic;
  }}

  /* Figures */
  .figure-container {{
    margin: 1.5em 0;
    text-align: center;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  .figure-container img {{
    max-width: 96%;
    height: auto;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}

  .figure-caption {{
    font-size: 8.5pt;
    color: #64748B;
    margin-top: 6px;
    font-weight: 500;
  }}

  /* Highlights & Badges */
  strong {{
    font-weight: 700;
    color: #0F172A;
  }}

  hr {{
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 1.8em 0;
  }}

  /* Math styling */
  .MathJax {{
    font-size: 10pt !important;
  }}
</style>
</head>
<body>
{content_html}
</body>
</html>
"""
    return html

def main():
    if not MD_PATH.exists():
        raise SystemExit(f"Error: {MD_PATH} not found.")

    print(f"Reading {MD_PATH}...", flush=True)
    md_text = MD_PATH.read_text(encoding="utf-8")

    html = convert_md_to_html(md_text)
    HTML_TEMP.write_text(html, encoding="utf-8")
    print(f"Wrote temporary preview to {HTML_TEMP}", flush=True)

    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not Path(chrome_bin).exists():
        raise SystemExit(f"Chrome not found at {chrome_bin}")

    print("Running Chrome Headless PDF generator...", flush=True)
    cmd = [
        chrome_bin,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={PDF_PATH}",
        f"file://{HTML_TEMP.resolve()}",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if PDF_PATH.exists():
        size_kb = PDF_PATH.stat().st_size / 1024
        print(f"SUCCESS: Generated {PDF_PATH} ({size_kb:.1f} KB)", flush=True)
        # cleanup temporary html
        HTML_TEMP.unlink(missing_ok=True)
    else:
        print("Chrome output:\n", res.stdout, res.stderr, file=sys.stderr)
        raise SystemExit("PDF generation failed.")

if __name__ == "__main__":
    main()
