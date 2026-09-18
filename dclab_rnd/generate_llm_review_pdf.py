"""Generate a clean PDF report from completed campaign LLM reviews."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN = ROOT / "campaigns" / "model_building_50_v1"
MEMORY = CAMPAIGN / "llm_review_memory.jsonl"
RESULTS = CAMPAIGN / "results"

CONF_COLOR = {
    "high": "#16845B",
    "medium": "#9A6500",
    "low": "#B42318",
}
SEV_COLOR = {
    "high": "#B42318",
    "medium": "#9A6500",
    "low": "#5B6472",
}


def _safe_model_slug(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model.strip())


def _esc(text: object) -> str:
    return html.escape("" if text is None else str(text))


def _load_reviews(*, model_filter: str | None = None) -> list[dict]:
    memory_path = MEMORY
    if model_filter:
        tagged = CAMPAIGN / f"llm_review_memory__{_safe_model_slug(model_filter)}.jsonl"
        if tagged.exists():
            memory_path = tagged

    memory_rows = []
    if memory_path.exists():
        for line in memory_path.read_text().splitlines():
            if line.strip():
                memory_rows.append(json.loads(line))

    by_id = {r["experiment_id"]: r for r in memory_rows}
    reviews = []
    for path in sorted(RESULTS.glob("EXP-*.json")):
        data = json.loads(path.read_text())
        review_block = data.get("llm_review", {})
        if review_block.get("status") != "completed":
            continue
        full = review_block.get("review") or by_id.get(data.get("experiment_id"), {})
        if not full:
            continue
        full = dict(full)
        full.setdefault("experiment_id", data.get("experiment_id"))
        full.setdefault("dataset", data.get("dataset"))
        full.setdefault("evidence_path", str(path.relative_to(ROOT)))
        full.setdefault("model", review_block.get("model") or full.get("model"))
        full["kind"] = data.get("kind")
        full["question"] = data.get("question")
        if model_filter and (full.get("model") or "") != model_filter:
            continue
        reviews.append(full)
    reviews.sort(key=lambda r: r.get("experiment_id", ""))
    return reviews


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;'
        f'background:{color}18;color:{color};border:1px solid {color}55;'
        f'font-size:9pt;font-weight:700;letter-spacing:0.02em">{_esc(text)}</span>'
    )


def build_html(reviews: list[dict]) -> str:
    conf_counts = Counter((r.get("overall_confidence") or "unknown").lower() for r in reviews)
    datasets = sorted({r.get("dataset") for r in reviews})
    model = reviews[0].get("model", "gpt-4o-mini") if reviews else "n/a"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Summary table rows
    summary_rows = []
    for r in reviews:
        conf = (r.get("overall_confidence") or "unknown").lower()
        nxt = r.get("next_experiment") or {}
        summary_rows.append(
            "<tr>"
            f"<td>{_esc(r.get('experiment_id'))}</td>"
            f"<td>{_esc(r.get('dataset'))}</td>"
            f"<td>{_esc(r.get('kind', ''))}</td>"
            f"<td>{_badge(conf, CONF_COLOR.get(conf, '#5B6472'))}</td>"
            f"<td>{_esc((r.get('decision') or '')[:140])}</td>"
            f"<td>{_esc(nxt.get('title', ''))}</td>"
            "</tr>"
        )

    # Dataset rollup
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in reviews:
        by_dataset[r.get("dataset", "unknown")].append(r)

    dataset_cards = []
    for ds in datasets:
        items = by_dataset[ds]
        lows = sum(1 for x in items if (x.get("overall_confidence") or "").lower() == "low")
        highs = sum(1 for x in items if (x.get("overall_confidence") or "").lower() == "high")
        challenges = sum(len(x.get("challenges_to_claims") or []) for x in items)
        dataset_cards.append(
            f"""
            <div class="card">
              <div class="card-title">{_esc(ds)}</div>
              <div class="card-meta">{len(items)} reviews · {challenges} challenges · {lows} low / {highs} high confidence</div>
            </div>
            """
        )

    detail_sections = []
    for r in reviews:
        conf = (r.get("overall_confidence") or "unknown").lower()
        challenges = r.get("challenges_to_claims") or []
        risks = r.get("risks_and_missing_evidence") or []
        observed = r.get("observed_evidence") or []
        nxt = r.get("next_experiment") or {}

        challenge_html = ""
        if challenges:
            lis = []
            for c in challenges:
                sev = (c.get("severity") or "medium").lower()
                lis.append(
                    "<li>"
                    f"{_badge(sev, SEV_COLOR.get(sev, '#5B6472'))} "
                    f"<strong>{_esc(c.get('claim_id', ''))}</strong> — {_esc(c.get('issue', ''))}"
                    "</li>"
                )
            challenge_html = "<ul class='tight'>" + "".join(lis) + "</ul>"
        else:
            challenge_html = "<p class='muted'>No claim challenges recorded.</p>"

        risk_html = (
            "<ul class='tight'>" + "".join(f"<li>{_esc(x)}</li>" for x in risks) + "</ul>"
            if risks
            else "<p class='muted'>No explicit risks listed.</p>"
        )

        observed_html = ""
        if observed:
            lis = []
            for item in observed[:6]:
                if isinstance(item, dict):
                    lis.append(
                        f"<li>{_esc(item.get('statement', ''))}"
                        f"<div class='cite'>{_esc(item.get('citation', ''))}</div></li>"
                    )
                else:
                    lis.append(f"<li>{_esc(item)}</li>")
            observed_html = "<ul class='tight'>" + "".join(lis) + "</ul>"

        detail_sections.append(
            f"""
            <section class="exp">
              <h2>{_esc(r.get('experiment_id'))} · {_esc(r.get('dataset'))}</h2>
              <div class="meta-row">
                {_badge(conf + ' confidence', CONF_COLOR.get(conf, '#5B6472'))}
                <span class="pill">{_esc(r.get('kind', ''))}</span>
                <span class="pill">{_esc(r.get('model', model))}</span>
              </div>
              <p class="question"><strong>Question.</strong> {_esc(r.get('question', ''))}</p>
              <h3>Decision</h3>
              <p>{_esc(r.get('decision', ''))}</p>
              <h3>Interpretation</h3>
              <p>{_esc(r.get('interpretation', '—'))}</p>
              {"<h3>Observed evidence (cited)</h3>" + observed_html if observed_html else ""}
              <h3>Challenges to claims</h3>
              {challenge_html}
              <h3>Risks & missing evidence</h3>
              {risk_html}
              <h3>Smallest next experiment</h3>
              <div class="next">
                <div><strong>{_esc(nxt.get('title', '—'))}</strong></div>
                <div><em>Hypothesis:</em> {_esc(nxt.get('hypothesis', ''))}</div>
                <div><em>Success gate:</em> {_esc(nxt.get('success_gate', ''))}</div>
                <div><em>Why smallest:</em> {_esc(nxt.get('why_smallest', ''))}</div>
              </div>
              <p class="cite">Evidence: {_esc(r.get('evidence_path', ''))}</p>
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DCLab LLM Review Report</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 16mm 14mm 18mm 14mm;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Helvetica, Arial, sans-serif;
    color: #111827;
    font-size: 10pt;
    line-height: 1.45;
    margin: 0;
  }}
  h1 {{
    font-size: 22pt;
    margin: 0 0 6px 0;
    color: #0F172A;
    letter-spacing: -0.02em;
  }}
  h2 {{
    font-size: 13pt;
    margin: 0 0 8px 0;
    color: #0F172A;
    border-bottom: 1px solid #E5E7EB;
    padding-bottom: 4px;
  }}
  h3 {{
    font-size: 10.5pt;
    margin: 12px 0 4px 0;
    color: #1F2937;
  }}
  .subtitle {{ color: #5B6472; margin: 0 0 18px 0; font-size: 10pt; }}
  .hero {{
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 16px;
  }}
  .metrics {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 14px 0 18px 0;
  }}
  .metric {{
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 10px 12px;
    background: #FFFFFF;
  }}
  .metric .label {{ font-size: 8pt; color: #5B6472; font-weight: 700; text-transform: uppercase; }}
  .metric .value {{ font-size: 16pt; font-weight: 750; color: #111827; margin-top: 4px; }}
  .cards {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
    margin-bottom: 18px;
  }}
  .card {{
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 8px 10px;
    background: #FFFFFF;
  }}
  .card-title {{ font-weight: 700; color: #111827; }}
  .card-meta {{ font-size: 8.5pt; color: #5B6472; margin-top: 2px; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.5pt;
    margin: 8px 0 20px 0;
  }}
  th, td {{
    border: 1px solid #E5E7EB;
    padding: 6px 7px;
    vertical-align: top;
    text-align: left;
  }}
  th {{
    background: #F8FAFC;
    font-weight: 700;
    color: #111827;
  }}
  tr:nth-child(even) td {{ background: #FCFCFD; }}
  .exp {{
    page-break-inside: avoid;
    break-inside: avoid;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 12px 14px;
    margin: 0 0 12px 0;
    background: #FFFFFF;
  }}
  .meta-row {{ display: flex; gap: 8px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }}
  .pill {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    background: #F1F5F9;
    color: #334155;
    border: 1px solid #E2E8F0;
    font-size: 8.5pt;
    font-weight: 600;
  }}
  .question {{ color: #334155; }}
  .muted {{ color: #5B6472; }}
  .cite {{ color: #64748B; font-size: 8pt; margin-top: 2px; }}
  ul.tight {{ margin: 4px 0 0 18px; padding: 0; }}
  ul.tight li {{ margin: 0 0 4px 0; }}
  .next {{
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 9.5pt;
  }}
  .next div {{ margin: 2px 0; }}
  .footer {{
    margin-top: 18px;
    padding-top: 8px;
    border-top: 1px solid #E5E7EB;
    color: #64748B;
    font-size: 8.5pt;
  }}
  .note {{
    background: #FFFBEB;
    border: 1px solid #F1DE9D;
    color: #9A6500;
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 9pt;
    margin-bottom: 14px;
  }}
</style>
</head>
<body>
  <div class="hero">
    <h1>DCLab LLM Review Report</h1>
    <p class="subtitle">
      Campaign <strong>model_building_50_v1</strong> · Critic model <strong>{_esc(model)}</strong> ·
      Generated { _esc(generated) }
    </p>
    <div class="note">
      LLM role is critic / hypothesis generator only. Observed metrics remain authoritative.
      Reviews must not be treated as deployment approval.
    </div>
    <div class="metrics">
      <div class="metric"><div class="label">Reviews</div><div class="value">{len(reviews)}</div></div>
      <div class="metric"><div class="label">High conf.</div><div class="value">{conf_counts.get('high', 0)}</div></div>
      <div class="metric"><div class="label">Medium conf.</div><div class="value">{conf_counts.get('medium', 0)}</div></div>
      <div class="metric"><div class="label">Low conf.</div><div class="value">{conf_counts.get('low', 0)}</div></div>
    </div>
  </div>

  <h2>Dataset coverage</h2>
  <div class="cards">
    {''.join(dataset_cards)}
  </div>

  <h2>Executive summary</h2>
  <table>
    <thead>
      <tr>
        <th>ID</th><th>Dataset</th><th>Stage</th><th>Confidence</th><th>Decision</th><th>Next experiment</th>
      </tr>
    </thead>
    <tbody>
      {''.join(summary_rows)}
    </tbody>
  </table>

  <h2>Detailed reviews</h2>
  {''.join(detail_sections)}

  <div class="footer">
    Sources: llm_review_memory.jsonl and results/EXP-*.json · Repo path: campaigns/model_building_50_v1/
  </div>
</body>
</html>
"""


def build_markdown(reviews: list[dict]) -> str:
    lines = [
        "# DCLab LLM Review Report",
        "",
        f"Campaign: `model_building_50_v1`  ",
        f"Reviews: **{len(reviews)}**  ",
        f"Model: `{reviews[0].get('model', 'gpt-4o-mini') if reviews else 'n/a'}`",
        "",
        "> LLM is critic/proposer only. Metrics in result JSON remain authoritative.",
        "",
        "## Executive summary",
        "",
        "| ID | Dataset | Stage | Confidence | Decision | Next experiment |",
        "|---|---|---|---|---|---|",
    ]
    for r in reviews:
        nxt = (r.get("next_experiment") or {}).get("title", "")
        decision = (r.get("decision") or "").replace("|", "/")
        lines.append(
            f"| {r.get('experiment_id')} | {r.get('dataset')} | {r.get('kind', '')} | "
            f"{r.get('overall_confidence', '')} | {decision[:120]} | {nxt} |"
        )
    lines.extend(["", "## Detailed reviews", ""])
    for r in reviews:
        nxt = r.get("next_experiment") or {}
        lines.append(f"### {r.get('experiment_id')} · {r.get('dataset')}")
        lines.append("")
        lines.append(f"- **Confidence:** {r.get('overall_confidence')}")
        lines.append(f"- **Decision:** {r.get('decision')}")
        lines.append(f"- **Interpretation:** {r.get('interpretation', '—')}")
        lines.append(f"- **Next experiment:** {nxt.get('title')} — {nxt.get('success_gate')}")
        challenges = r.get("challenges_to_claims") or []
        if challenges:
            lines.append("- **Challenges:**")
            for c in challenges:
                lines.append(f"  - `{c.get('claim_id')}` ({c.get('severity')}): {c.get('issue')}")
        lines.append("")
    return "\n".join(lines) + "\n"


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not Path(chrome).exists():
        raise SystemExit(f"Chrome not found at {chrome}")
    cmd = [
        chrome,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        f"file://{html_path.resolve()}",
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=False)
    if not pdf_path.exists():
        raise SystemExit("PDF generation failed (Chrome did not write output).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM review PDF")
    parser.add_argument(
        "--model",
        default=None,
        help="Filter to reviews from this model and name outputs accordingly "
        "(e.g. gpt-5.4-mini)",
    )
    args = parser.parse_args()
    model_filter = args.model

    reviews = _load_reviews(model_filter=model_filter)
    if not reviews:
        raise SystemExit(
            "No completed LLM reviews found"
            + (f" for model={model_filter}." if model_filter else ".")
        )

    model_label = model_filter or reviews[0].get("model") or "llm"
    slug = _safe_model_slug(model_label)
    md_out = CAMPAIGN / f"LLM_REVIEW_REPORT__{slug}.md"
    html_out = CAMPAIGN / f"_llm_review_preview__{slug}.html"
    pdf_out = CAMPAIGN / f"LLM_REVIEW_REPORT__{slug}.pdf"
    # Also keep a latest convenience copy named for the model display string.
    latest_pdf = CAMPAIGN / "LLM_REVIEW_REPORT.pdf"
    latest_md = CAMPAIGN / "LLM_REVIEW_REPORT.md"

    md_out.write_text(build_markdown(reviews), encoding="utf-8")
    html_out.write_text(build_html(reviews), encoding="utf-8")
    print(f"Wrote {md_out.relative_to(ROOT)}")
    print(f"Wrote {html_out.relative_to(ROOT)}")

    html_to_pdf(html_out, pdf_out)
    size_kb = pdf_out.stat().st_size / 1024
    latest_md.write_text(md_out.read_text(encoding="utf-8"), encoding="utf-8")
    latest_pdf.write_bytes(pdf_out.read_bytes())
    print(f"SUCCESS: {pdf_out.relative_to(ROOT)} ({size_kb:.1f} KB)")
    print(f"Also updated: {latest_pdf.relative_to(ROOT)}")
    html_out.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
