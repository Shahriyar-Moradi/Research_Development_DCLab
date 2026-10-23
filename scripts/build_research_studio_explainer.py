"""Build the public-facing DCLab R&D and Research Studio explainer PDF.

The report is generated from checked-in campaign evidence and the local agent
ledger. It never reads raw dataset rows, OOF predictions, or LLM transcripts.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/DCLab_RandD_Research_Studio_Explained.pdf"
NAVY = colors.HexColor("#18364A")
TEAL = colors.HexColor("#147E70")
PALE = colors.HexColor("#EAF5F2")
INK = colors.HexColor("#253746")
MUTED = colors.HexColor("#586A78")
LINE = colors.HexColor("#DCE5E8")
LIGHT = colors.HexColor("#F5F8F9")
AMBER = colors.HexColor("#A45D16")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def txt(value) -> str:
    return escape(str(value if value is not None else "-")).replace("\n", "<br/>")


def number(value, places=4):
    return "-" if value in (None, "") else f"{float(value):.{places}f}"


def model_label(value):
    return {
        "logistic_regression": "logistic",
        "hist_gradient_boosting": "hist_boost",
        "random_forest": "random_forest",
    }.get(value, value or "-")


class GuideDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(
            str(filename), pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
            topMargin=22 * mm, bottomMargin=20 * mm,
            title="DCLab R&D Lab and Research Studio - Explained",
            author="DCLab R&D", subject="Evidence grounded system and experiment guide",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                      leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0)
        self.addPageTemplates(PageTemplate(id="normal", frames=[frame], onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(NAVY)
        canvas.rect(0, h - 11 * mm, w, 11 * mm, stroke=0, fill=1)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(colors.white)
        canvas.drawString(19 * mm, h - 7.4 * mm, "DCLab  /  R&D Lab and Research Studio")
        canvas.setStrokeColor(LINE)
        canvas.line(19 * mm, 15 * mm, w - 19 * mm, 15 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(19 * mm, 10.7 * mm, "Evidence guide  |  25 September 2026")
        canvas.drawRightString(w - 19 * mm, 10.7 * mm, str(doc.page))
        canvas.restoreState()


S = getSampleStyleSheet()
ST = {
    "title": ParagraphStyle("title", parent=S["Title"], fontName="Helvetica-Bold", fontSize=28, leading=32, textColor=NAVY, spaceAfter=12),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=14, leading=21, textColor=MUTED, spaceAfter=18),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=NAVY, spaceBefore=12, spaceAfter=10, keepWithNext=True),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=TEAL, spaceBefore=11, spaceAfter=6, keepWithNext=True),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.2, leading=13.5, textColor=INK, spaceAfter=7),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=7.8, leading=11.2, textColor=INK, spaceAfter=5),
    "tiny": ParagraphStyle("tiny", fontName="Helvetica", fontSize=6.8, leading=9.0, textColor=INK),
    "tablehead": ParagraphStyle("tablehead", fontName="Helvetica-Bold", fontSize=7.1, leading=9, textColor=colors.white),
    "callout": ParagraphStyle("callout", fontName="Helvetica-Bold", fontSize=10.5, leading=15, textColor=NAVY, spaceAfter=4),
    "caption": ParagraphStyle("caption", fontName="Helvetica", fontSize=7.5, leading=10.5, textColor=MUTED, spaceAfter=7),
}


def p(text, style="body"):
    return Paragraph(text, ST[style])


def prose(text, style="body"):
    return p(txt(text), style)


def bullet(text):
    return p("<font color='#147E70'><b>•</b></font>  " + text, "body")


def heading(title, level=1):
    return p(txt(title), "h1" if level == 1 else "h2")


def table(headers, rows, widths, font="small", row_background=True):
    data = [[p(txt(h), "tablehead") for h in headers]]
    for row in rows:
        data.append([p(txt(cell), font) for cell in row])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), .6, NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 1), (-1, -1), .25, LINE),
    ]
    if row_background:
        commands.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]))
    t.setStyle(TableStyle(commands))
    return t


def callout(title, body, width=172 * mm):
    t = Table([[[p(txt(title), "callout"), prose(body, "body")]]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#B9D9CF")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def ledger_runs():
    db = ROOT / "agent_runs/research.sqlite3"
    if not db.exists():
        return []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        runs = []
        for run_id, payload in conn.execute("SELECT id, payload FROM runs ORDER BY rowid DESC"):
            run = json.loads(payload)
            events = [(kind, json.loads(body)) for kind, body in conn.execute(
                "SELECT kind,payload FROM events WHERE run_id=? ORDER BY seq", (run_id,)
            )]
            run["trials"] = [body for kind, body in events if kind == "trial"]
            run["event_counts"] = dict(Counter(kind for kind, _ in events))
            run["synthesis"] = next((body for kind, body in reversed(events) if kind == "synthesis"), None)
            runs.append(run)
        return runs
    finally:
        conn.close()


def campaign_rows():
    paths = sorted((ROOT / "campaigns/model_building_50_v1/results").glob("EXP-*.json"))
    return [load_json(path) for path in paths]


def historical_rows():
    with (ROOT / "knowledge/experiment_registry.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_note(ids):
    return p("<b>Evidence:</b> " + ", ".join(ids), "caption")


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    campaign = campaign_rows()
    historical = historical_rows()
    churn = load_json(ROOT / "churn_exp/results/summary.json")
    runs = ledger_runs()
    rules = [json.loads(line) for line in (ROOT / "knowledge/model_building_rules.jsonl").read_text().splitlines() if line]
    workflows = load_json(ROOT / "knowledge/workflow_blocks.json")
    clean_manifest = load_json(ROOT / "agent_runs/clean_exports/MANIFEST.json")
    by_dataset = {}
    for row in campaign:
        by_dataset.setdefault(row["dataset"], []).append(row)

    story = []
    add = story.append
    extend = story.extend

    # Cover and reading map.
    extend([Spacer(1, 20 * mm), p("DCLab R&D Lab and<br/>Research Studio", "title"),
            p("A simple, evidence grounded explanation of the system, its agents, its experiments, and its results", "subtitle"),
            callout("The 30 second explanation", "DCLab's main product aims to help people build and operate machine learning work in one connected project. This R&D repository is its test workshop. It runs model building experiments on real datasets, records what worked and failed, and turns the evidence into reusable workflow guidance. The Research Studio adds AI agents that propose and explain bounded experiments while tested code calculates the scores."),
            Spacer(1, 8 * mm),
            prose("Prepared from the repository's reports, result JSON, normalized registry, and local Research Studio ledger. Results are as recorded on 25 September 2026. This is a reading guide to evidence, not a claim that a benchmark model is ready for real users."),
            source_note(["S1", "S2", "S3", "S4", "S5"]),
            PageBreak()])

    extend([heading("How to read this guide"),
            table(["If you want to know...", "Read"], [
                ("What is DCLab and why does the lab exist?", "Sections 1-2"),
                ("How do agents and experiments work?", "Sections 3-5"),
                ("What do the numbers mean?", "Sections 6-8"),
                ("What exactly happened in each campaign?", "Appendices A-F"),
                ("What can be reused or fine tuned?", "Sections 9-10"),
                ("Where are the underlying records?", "Sources and evidence map"),
            ], [76 * mm, 96 * mm]),
            Spacer(1, 6 * mm),
            callout("Read the label next to a number", "Historical registry scores, the controlled 50 experiment holdouts, churn development CV, and agentic development CV use different protocols. A higher number from one group does not prove its model is better than a number from another group."),
            PageBreak()])

    # The product and lab.
    extend([heading("1. What is DCLab?"),
            prose("DCLab is being built as a project centered environment for data scientists and ML engineers. Its intended path connects the question, data, experiments, model decisions, release, and monitoring. A developer should be able to see what was tried, why, what the tests found, and what can safely happen next. The main product and this R&D repository are separate codebases."),
            table(["Part", "Simple meaning", "Current relationship"], [
                ("DCLab main product", "The workspace where users would build and operate real ML projects.", "Its current truth and implementation live in the separate DCLab product repository."),
                ("R&D Lab", "The workshop for repeatable research on datasets, algorithms, features, leakage, and evaluation.", "This repository contains the experiments and evidence reviewed here."),
                ("Research Studio", "A local interface and agent loop for asking questions, running a small experiment series, and inspecting results.", "Implemented inside this R&D repository at 127.0.0.1:8765."),
                ("Knowledge memory", "Rules and examples learned from research, each linked to evidence.", "Useful input to future product decisions and possible SLM training after review."),
            ], [32 * mm, 69 * mm, 71 * mm]),
            heading("2. What is this lab trying to learn?"),
            prose("The lab asks practical questions: What do the columns mean? Which fields would exist when a prediction is made? Which features help and which create leakage? Does a simple linear model beat a tree? Does tuning produce a repeatable improvement? What happens if a field is missing? Which steps should be repeated on a new dataset?"),
            bullet("A good lab result is a narrow lesson with a source file, dataset, split, metric, limitation, and next test."),
            bullet("A failed or worse experiment is still valuable: it tells the next researcher which idea needs revision."),
            bullet("The longer term goal is a reusable workflow, not a single universal best model."),
            source_note(["S1", "S2", "S4", "S12"]),
            PageBreak()])

    # Inventory.
    extend([heading("3. What exists today?"),
            table(["Evidence collection", "Size", "What it answers", "Important limit"], [
                ("Historical normalized registry", "579 records / 11 datasets / 7 suites", "Which methods were tried across the older notebooks and runners?", "Protocols differ; only 10 records have full provenance. Two duplicate key warnings remain."),
                ("Controlled model building campaign", "50 tasks = 10 public datasets x 5 stages", "How does the complete data-to-model workflow behave under one campaign protocol?", "Ten final holdouts were used once. Cached category labels need source repair."),
                ("HyperAck core lifecycle", "83 records = 15 original + 15 safe + 53 optimized safe; 80 completed, 3 skipped", "What changes when post-outcome fare fields are excluded?", "Historical split was consulted repeatedly; fresh temporal confirmation is needed."),
                ("HyperAck four way benchmark", "44 additional registry rows", "How do safe/unsafe and baseline/optimized combinations compare?", "It overlaps the same problem and does not add 44 independent datasets."),
                ("Telco Churn fixed suite", "15 completed recipes", "Which model and charge/tenure features look useful?", "Development CV only; no independent confirmation."),
                ("Agentic Research Studio", f"{len(runs)} run records", "Can agents propose, execute, critique, and remember bounded tests?", "Adaptive development CV; failed and budget-limited runs are retained."),
            ], [38 * mm, 31 * mm, 53 * mm, 50 * mm]),
            Spacer(1, 5 * mm),
            prose("The counts overlap. For example, the 579-row registry includes the 127 HyperAck rows and ten final records from the 50-task campaign. Do not sum all rows and call them independent experiments."),
            source_note(["S2", "S3", "S4", "S5", "S6", "S7"]),
            PageBreak()])

    # Agent operations.
    extend([heading("4. How does the Research Studio work?"),
            prose("A person enters a goal, selects a project and dataset, and sets limits on experiments, rows, CV repeats, active minutes, and LLM calls. The backend stores a durable run ID. A graph coordinates specialist agents and a separate worker calculates the actual measurements."),
            table(["Order", "Actor", "What happens", "Output"], [
                ("1", "Deterministic profiler", "Reads an allowed data snapshot and summarizes rows, columns, missingness, duplicates, and target balance.", "Aggregate profile and data fingerprints."),
                ("2", "Research Planner", "Turns the user's goal into bounded questions, sequence, success criteria, and limitations.", "Typed agenda."),
                ("3", "Data Scientist", "Reviews the profile, feature availability assumptions, leakage risks, and sensible feature hypotheses.", "Typed data review."),
                ("4", "Experiment Designer", "Proposes one allowed model and feature configuration at a time, citing prior evidence when comparing.", "Validated experiment recipe."),
                ("5", "Deterministic worker", "Applies decision-time feature policy, runs fold-local preprocessing, fits models, and computes scores and diagnostics.", "Result JSON, OOF predictions, replay recipe, hashes."),
                ("6", "Results Critic", "Checks whether the measured change answers the question and states limitations and next test.", "Critique with evidence IDs."),
                ("7", "Knowledge Curator", "After the bounded loop, turns supported results and failures into narrow lessons and workflow blocks.", "Provisional synthesis."),
            ], [13 * mm, 32 * mm, 84 * mm, 43 * mm]),
            Spacer(1, 4 * mm),
            callout("Who owns the truth?", "Agents propose and interpret. The validated worker computes metrics. SQLite and on-disk files preserve the run. LangGraph coordinates and checkpoints the sequence; NOOA supplies typed agent classes. A human must still review business meaning, sensitive use, and any production decision."),
            source_note(["S8", "S9", "S10"]),
            PageBreak()])

    extend([heading("5. How is one experiment made?"),
            prose("The designer must submit a structured recipe: dataset, testable hypothesis, one of seven allowed model families, bounded model parameters, optional safe feature formulas, columns to drop, fields to simulate as missing, and cited earlier results. It cannot submit arbitrary Python to execute."),
            table(["Step", "Plain explanation", "Technical check"], [
                ("Define prediction moment", "When would the answer be needed?", "Block fields that appear only after the outcome, such as HyperAck final fares."),
                ("Choose a fair split", "Keep lookalike copies from appearing in both study and quiz.", "Exact duplicate allowed-input rows share a group in repeated 3-fold CV."),
                ("Prepare inside each fold", "Learn filling and category rules from study rows only.", "Imputer, scaler, encoder, derived features, and estimator are in one pipeline."),
                ("Train and predict", "Fit on two folds, predict on the third, repeat.", "Default is 2 repeats x 3 folds; collect out-of-fold probabilities."),
                ("Compare", "Ask whether the new choice helped the same data splits.", "Paired fold delta is descriptive, not a significance test."),
                ("Challenge", "Ask what would break at use time.", "Calibration, missing-input stress, sensitivity, and feature lineage diagnostics."),
                ("Save and critique", "Keep the result and the explanation together.", "Evidence ID, result, recipe, source/code hashes, critique, failure if any."),
            ], [30 * mm, 70 * mm, 72 * mm]),
            heading("A concrete example" ,2),
            prose("For Telco Churn, one agentic run first tested logistic regression on existing account features. The next recipe added a log transform of tenure while holding the model family fixed. Mean development CV ROC-AUC moved from 0.8447 to 0.8483; the paired mean gain was about 0.0036. A later histogram boosting test scored 0.8214. The sequence illustrates an interpretable ablation and a useful negative result. It does not show that logistic regression will win on future customers."),
            source_note(["S5", "S8", "S9"]),
            PageBreak()])

    # Metrics.
    extend([heading("6. What do the numbers mean?"),
            table(["Metric", "Simple meaning", "Better direction", "Common trap"], [
                ("ROC-AUC", "How often a random positive case ranks above a random negative case.", "Higher", "It does not choose a threshold or prove useful business decisions."),
                ("Average precision (AP)", "Precision across recall levels; useful when positives are rare.", "Higher", "Different class balance can change AP, so cross-dataset comparison is unfair."),
                ("Log loss", "Penalty for confident wrong probabilities.", "Lower", "A model may improve AUC while worsening probability quality."),
                ("Brier score", "Average squared error of predicted probabilities.", "Lower", "It mixes discrimination and calibration."),
                ("Recall", "Share of actual positives found at a chosen threshold.", "Higher, subject to cost", "Can rise by flagging almost everyone; inspect precision too."),
                ("F1", "Balance of precision and recall at one threshold.", "Higher, subject to cost", "Ignores true negatives and business costs."),
                ("Paired delta", "New score minus reference score on matching folds.", "Positive for AUC/AP", "Small deltas from reused CV can be noise or adaptive overfitting."),
            ], [28 * mm, 70 * mm, 30 * mm, 44 * mm]),
            Spacer(1, 5 * mm),
            callout("How to read 0.85 AUC", "It suggests useful ranking in that evaluation setup. It does not mean 85% accuracy, 85% precision, or an 85% chance that each individual prediction is correct."),
            PageBreak()])

    # Results summary.
    extend([heading("7. What have the results taught us?"),
            table(["Finding", "Measured support", "What to do with it"], [
                ("Leakage can dominate a leaderboard", "Historical safe/unsafe AUC gaps: bank +0.1328, online shoppers +0.1575, HyperAck +0.0347. The controlled campaign found bank +0.1842 and online shoppers +0.1968 on its own CV protocol.", "Define the prediction moment and audit timestamps/lineage before model comparison."),
                ("More features are not always better", "The controlled campaign retained raw features on 7/10 datasets; ratios on two and interactions on one.", "Start with raw inputs. Keep a feature recipe only when paired evidence and operational meaning justify it."),
                ("No model family wins everywhere", "LightGBM led mean historical rank; Extra Trees was selected on 7/10 controlled datasets; churn's best fixed suite used engineered logistic regression.", "Screen a linear baseline and a few nonlinear families on identical folds."),
                ("Input availability matters", "The worker measures the effect of masking selected input columns; history/price fields still need real availability contracts.", "Treat feature importance as a review candidate, not approval."),
                ("High benchmark scores need scrutiny", "Mushroom reached 1.0000 controlled holdout AUC; breast cancer reached 0.9960.", "Audit task ease, duplicates, proxies, split design, and intended use before claiming success."),
            ], [40 * mm, 70 * mm, 62 * mm]),
            Spacer(1, 4 * mm),
            prose("These are scoped findings, not universal laws. The 22 machine readable rules in this repository attach confidence, evidence, exceptions, and next tests to each principle."),
            source_note(["S2", "S3", "S4", "S11"]),
            PageBreak()])

    # Three projects.
    extend([heading("8. The three projects in the Studio"),
            table(["Project", "Question", "Results to remember", "Limit"], [
                ("Open tabular lab", "How well does the five stage model building procedure transfer across ten public classification datasets?", "50/50 controlled tasks complete. Raw features selected 7/10 times; Extra Trees 7/10 model screens.", "Legacy cached categories and public benchmark splits limit real-world claims."),
                ("HyperAck", "Can an order-time delivery acceptance model be accurate without final fare fields?", "Unsafe historical AUC roughly 0.9802; best optimized safe AUC roughly 0.9455. The 0.0347 gap illustrates leakage severity.", "Historical test use is adaptive; later-time, zone, and demand regime tests remain."),
                ("Telco Churn", "Can account and billing data help rank likely churn?", "Fixed 15 run leader CHURN-014: AUC 0.8499, AP 0.6708; agentic log-tenure test: development AUC 0.8483 on its separate 4,000-row protocol.", "No independent churn holdout or verified outcome window."),
            ], [30 * mm, 51 * mm, 53 * mm, 38 * mm]),
            heading("HyperAck: what safe and unsafe mean",2),
            prose("The unsafe tests used fields such as final_customer_fare and final_biker_fare, which are known only after the decision point in the order-time prediction story. Their high scores are useful for teaching leakage, not for selecting a deployable model. Safe tests exclude these fields. The small gain from extensive safe optimization also shows why a complicated model must justify its added cost."),
            heading("Churn: why a simple model can win",2),
            prose("A tenure/charge representation made logistic regression slightly stronger than the other tested families in the fixed suite. This is a dataset-specific development result; the agentic run used different row limits and split details, so its 0.8483 AUC is not a replication of the fixed suite's 0.8499 AUC."),
            source_note(["S2", "S3", "S4", "S5", "S7"]),
            PageBreak()])

    # Workflow.
    extend([heading("9. The reusable ten step model building workflow"),
            prose("These blocks are a guide for an engineer or a focused future agent. They are ordered so that each decision has the information it needs."),
            table(["ID", "Step", "What to check before moving on"], [
                (b["id"], b["name"], " -> ".join(b["flow"])) for b in workflows
            ], [17 * mm, 34 * mm, 121 * mm], font="small"),
            Spacer(1, 4 * mm),
            callout("Why this is a workflow rather than a checklist", "A result from a later step can send the team back: a suspiciously perfect AUC returns to leakage and split review; poor missing-input stress returns to feature design; unstable folds return to data and validation design."),
            source_note(["S4"]),
            PageBreak()])

    # Knowledge, SLM, UI.
    extend([heading("10. What gets saved, and how could an SLM learn from it?"),
            table(["Layer", "What is stored", "Why it matters"], [
                ("Raw trial evidence", "Result JSON, OOF predictions, recipe, data/code hashes, fold metrics, stress tests, calibration.", "Lets an expert reproduce and audit the actual measurement."),
                ("Human run summary", "Run card, leaderboard, numbered reports, failed attempts.", "Lets a person see the answer quickly and drill into details."),
                ("Research memory", "Cited lessons, counterevidence, limitations, and ten workflow blocks.", "Helps future agents retrieve a narrow precedent rather than guessing."),
                ("Clean SFT export", "Structured chat examples for agenda planning, data/leakage review, experiment proposals, critiques, and synthesis.", "A possible training source for a focused small language model after human curation."),
            ], [31 * mm, 72 * mm, 69 * mm]),
            Spacer(1, 5 * mm),
            prose(f"The current clean export manifest records {clean_manifest['trial_count']} completed compact trials and {clean_manifest['sft_record_count']} chat-formatted examples across {len(clean_manifest['sft_task_counts'])} task types. These examples are preparatory. The full trajectory is marked training_ready=false. Before training, verify every claim, remove secrets or personal data, check licenses, eliminate near duplicates, and split by source/dataset or task lineage so related experiments do not leak into evaluation."),
            prose("Fine tuning could teach a small model to follow a particular review format and recognize common research moves. It cannot guarantee factual memory, remove hallucination by itself, replace fresh data access, or make the whole system deterministic. A useful evaluation should compare the proposed SLM with the present agent plus tools on held-out datasets, citation accuracy, valid plans, error recovery, latency, and total cost."),
            heading("Where to click in the local Studio",2),
            table(["View", "Use it for"], [
                ("Research", "Set a goal and budget; watch run status, trial scores, critique, and input/output diagnostics."),
                ("Knowledge", "Read provisional lessons and open the full model building field guide."),
                ("Recipes & traces", "Inspect replay configuration and export the explicit run trajectory."),
            ], [43 * mm, 129 * mm]),
            source_note(["S8", "S9", "S10", "S13"]),
            PageBreak()])

    # Boundaries and next practical steps.
    extend([heading("11. What is proven, and what remains open?"),
            table(["Established in this repository", "Still requires new evidence"], [
                ("The studio can coordinate typed agent roles and calculate bounded tabular experiments with preserved local evidence.", "That a specialized SLM improves quality, cost, latency, or reliability over the current agent system."),
                ("Feature exclusion and fold-local preprocessing are implemented in the worker.", "That all source columns are truly available at the real prediction moment."),
                ("The controlled 50-task campaign completed and consumed ten final holdouts once.", "That its selections hold on new time periods, locations, source systems, and production populations."),
                ("The clean export and field guide exist.", "Human-reviewed training labels, grouped evaluation sets, and a validated fine-tuned model."),
                ("Historical model and feature results are collected.", "Reproduction of historical champions with complete data/code/environment provenance."),
            ], [86 * mm, 86 * mm]),
            Spacer(1, 6 * mm),
            heading("A sensible next research cycle",2),
            bullet("Choose one concrete DCLab workflow that currently takes too much human effort."),
            bullet("Write a measurable baseline: task completion, citation accuracy, invalid plans, human correction time, latency, and cost."),
            bullet("Use a new dataset or later time slice for evaluation. Do not train on or repeatedly tune against it."),
            bullet("Compare current agent plus deterministic worker, a narrower prompt, retrieval, and a small fine-tuned model."),
            bullet("Promote a capability only when the improvement survives independent testing and the failure modes are understood."),
            source_note(["S4", "S12", "S13"]),
            PageBreak()])

    # Appendix A: all 50 tasks by dataset.
    extend([heading("Appendix A. Every controlled campaign task"),
            prose("Each dataset has five experiment IDs. The sentences below are taken from the recorded claims and numerical evidence. The final AUC in stage five is a once-consumed holdout; stages one through four use training partition evidence and CV. These rows describe what was measured, not what to deploy."),
            source_note(["S3"]), PageBreak()])
    stage_names = {
        "data_understanding": "Data understanding",
        "leakage_audit": "Leakage audit",
        "feature_engineering": "Feature engineering",
        "model_selection": "Model screen",
        "optimization_reliability": "Optimize and evaluate",
    }
    stage_order = list(stage_names)
    for dataset, items in by_dataset.items():
        parts = {item["kind"]: item for item in items}
        if not all(stage in parts for stage in stage_order):
            continue
        add(heading(dataset.replace("_", " ").title()))
        add(prose("Five connected tests answer five different questions about the same dataset. Read the ID to find the full JSON evidence."))
        rows = []
        for stage in stage_order:
            item = parts[stage]
            evidence = item.get("evidence") or {}
            claims = item.get("claims") or []
            if stage == "leakage_audit" and len(claims) > 1:
                statement = claims[1]["statement"]
            else:
                statement = claims[0]["statement"] if claims else item.get("question", "")
            if stage == "optimization_reliability":
                statement += " Holdout ROC-AUC: " + number((evidence.get("holdout_metrics") or {}).get("roc_auc")) + "."
            rows.append((item["experiment_id"], stage_names[stage], statement))
        add(table(["ID", "Question stage", "Recorded result in plain words"], rows,
                  [24 * mm, 39 * mm, 109 * mm], font="small"))
        add(Spacer(1, 3 * mm))
        final = parts["optimization_reliability"]["evidence"]
        feat = parts["feature_engineering"]["evidence"]
        add(prose("Reading: selected feature recipe = " + str(feat.get("selected_stage")) +
                  "; final selected model = " + str(final.get("model")) +
                  "; holdout Brier = " + number((final.get("holdout_metrics") or {}).get("brier")) +
                  ". The result is specific to this split and candidate set.", "small"))
        add(PageBreak())

    # Appendix B: churn individual results.
    extend([heading("Appendix B. All 15 fixed churn experiments"),
            prose("These 15 recipes share one development CV study. The AUC change is relative to CHURN-002, the plain logistic baseline. Lower log loss is better; the highest AUC does not decide the best business threshold."),
            table(["ID", "Recipe", "Family", "AUC", "AP", "Log loss", "Delta AUC"], [
                (r["id"], r["title"], r["model"], number(r["metrics"]["roc_auc"]),
                 number(r["metrics"]["average_precision"]), number(r["metrics"]["log_loss"]),
                 number(r.get("paired_auc_delta_vs_logistic"), 4))
                for r in churn["leaderboard"]
            ], [20 * mm, 51 * mm, 35 * mm, 16 * mm, 16 * mm, 18 * mm, 16 * mm], font="tiny"),
            source_note(["S5"]), PageBreak()])

    # Appendix C: every agentic trial, including failed trial.
    extend([heading("Appendix C. Research Studio runs and individual trials"),
            prose("This is the current local ledger, which is more authoritative than an older generated report snapshot. Development CV uses repeated duplicate-group folds. A run can contain successful measured trials even when its overall status is budget_exhausted. Model labels are shortened here; the full recipe is stored under each trial ID."),
            table(["Run", "Dataset", "Status", "Trials", "LLM calls", "Tokens"], [
                (r["id"][:8], ", ".join(r["config"].get("datasets", [])), r["status"],
                 str(len(r["trials"])), str(r.get("llm_calls", 0)),
                 str((r.get("usage") or {}).get("total_tokens", "unreported")))
                for r in runs
            ], [24 * mm, 34 * mm, 36 * mm, 19 * mm, 24 * mm, 35 * mm]),
            Spacer(1, 5 * mm)])
    for run in runs:
        add(heading("Run " + run["id"][:8] + " - " + ", ".join(run["config"].get("datasets", [])), 2))
        if not run["trials"]:
            add(prose("No model trial was completed. The available data profile and failure event remain in the trajectory.", "small"))
            continue
        rows = []
        for trial in run["trials"]:
            metric = (trial.get("result") or {}).get("metrics") or {}
            delta = (trial.get("paired_comparison") or {}).get("mean_deltas", {}).get("roc_auc")
            rows.append((trial["id"].split(":")[-1], trial.get("status"),
                         (trial.get("plan") or {}).get("title", ""),
                         model_label((trial.get("plan") or {}).get("model", "")),
                         number(metric.get("roc_auc")), number(metric.get("average_precision")),
                         number(metric.get("log_loss")), number(delta)))
        add(table(["Trial", "Status", "Question / change", "Model", "AUC", "AP", "Loss", "Delta"], rows,
                  [17 * mm, 19 * mm, 55 * mm, 26 * mm, 13 * mm, 13 * mm, 13 * mm, 16 * mm], font="tiny"))
        add(Spacer(1, 3 * mm))
    extend([prose("For German Credit, trial-004 improved AUC relative to trial-003 but worsened log loss. For Bank Marketing, the forest's AUC dipped while AP and log loss improved. These disagreements are why one metric does not settle the model choice. One Bank Marketing trial failed, and the adult agentic run stopped before any model trial. The churn agentic run kept three measured trials despite the time budget ending."),
            source_note(["S6", "S8"]), PageBreak()])

    # Appendix D: 127 historical HyperAck rows grouped by suite.
    extend([heading("Appendix D. HyperAck historical experiment index"),
            prose("The 127 registry rows below include 83 core lifecycle records (80 completed, 3 skipped) and 44 four way benchmark rows. This is an index to historical measurements; the rows share data and often the same split, so they are not 127 independent confirmations. 'Skipped' means no measured score was produced for that record."),
            source_note(["S2", "S7"]), PageBreak()])
    hyper = [r for r in historical if r["dataset"] == "hyperack"]
    for suite in ["hyperack_initial", "hyperack_safe", "hyperack_optimized_safe", "hyperack_4way"]:
        items = [r for r in hyper if r["suite"] == suite]
        add(heading(suite.replace("_", " ").title().replace("Hyperack", "HyperAck") + f" ({len(items)} rows)", 2))
        add(table(["Experiment ID", "Model", "Mode", "AUC", "Recall", "Status"], [
            (r["experiment_id"], r["model_family"], r["mode"], number(r["roc_auc"]),
             number(r["recall"]), r["status"]) for r in items
        ], [67 * mm, 29 * mm, 17 * mm, 17 * mm, 19 * mm, 23 * mm], font="tiny"))
        add(Spacer(1, 4 * mm))
    add(PageBreak())

    # Registry summary and sources.
    extend([heading("Appendix E. The rest of the historical registry"),
            prose("The normalized registry contains 579 rows in seven suites. The table shows counts and the highest recorded AUC by dataset, including unsafe rows where present; it is not a safe-model leaderboard. Older suites used different experimental setups. The full 579-row lookup, including source path, model, status, and metrics, is knowledge/experiment_registry.csv."),
            table(["Dataset", "Rows", "Best registry AUC", "Best family"], [
                (dataset, str(len(items)), number(max((float(r["roc_auc"]), r["model_family"]) for r in items if r["roc_auc"])[0]),
                 max((float(r["roc_auc"]), r["model_family"]) for r in items if r["roc_auc"])[1])
                for dataset, items in sorted(((d, [r for r in historical if r["dataset"] == d]) for d in {r["dataset"] for r in historical}))
            ], [53 * mm, 26 * mm, 46 * mm, 47 * mm]),
            Spacer(1, 4 * mm),
            prose("The 50-task campaign, churn suite, and agentic trials receive detailed individual tables above. Appendix F lists the remaining 452 historical registry rows, giving this PDF an index of all 579 normalized records. Older results should be treated as candidate hypotheses until replicated with full provenance and a fresh evaluation design."),
            source_note(["S2", "S11"]), PageBreak()])

    # Appendix F: every remaining normalized registry row. Together with
    # Appendix D, this indexes all 579 rows without conflating protocols.
    remaining = [r for r in historical if r["dataset"] != "hyperack"]
    extend([heading("Appendix F. Remaining historical experiment index"),
            prose(f"The following {len(remaining)} records complete the 579-row normalized registry index alongside Appendix D's 127 HyperAck rows. This compact table shows a subset of fields. The registry CSV holds the full source path, other metrics, provenance flags, and notes for each row. Safe and unsafe records can appear together, so read Mode before interpreting AUC."),
            source_note(["S2"]), PageBreak()])
    for dataset in sorted({r["dataset"] for r in remaining}):
        items = sorted((r for r in remaining if r["dataset"] == dataset),
                       key=lambda r: (r["suite"], r["experiment_id"]))
        add(heading(dataset.replace("_", " ").title() + f" ({len(items)} registry rows)", 2))
        add(table(["Suite", "Experiment ID", "Model", "Mode", "AUC", "F1", "Status"], [
            (r["suite"], r["experiment_id"], model_label(r["model_family"]),
             r["mode"], number(r["roc_auc"]), number(r["f1"]), r["status"])
            for r in items
        ], [29 * mm, 51 * mm, 28 * mm, 16 * mm, 14 * mm, 14 * mm, 20 * mm], font="tiny"))
        add(Spacer(1, 4 * mm))
    add(PageBreak())

    extend([heading("Sources and evidence map"),
            prose("Repository paths below are relative to the R&D checkout. The DCLab product context lives in a separate local checkout and supplies product intent; run metrics come from this R&D repository."),
            table(["ID", "Source", "Used for"], [
                ("S1", "/Users/shahriar/Downloads/decision_ai/DCLAB_MASTER_CONTEXT.md", "Main DCLab concept and separation from the R&D repository."),
                ("S2", "knowledge/KNOWLEDGE_BASE.md; knowledge/evidence.json; knowledge/experiment_registry.csv", "579-row historical registry and cross-dataset/leakage summaries."),
                ("S3", "campaigns/model_building_50_v1/CAMPAIGN_REPORT.md; results/EXP-*.json", "All 50 controlled tasks and ten holdout outcomes."),
                ("S4", "knowledge/MODEL_BUILDING_FIELD_GUIDE.md; knowledge/model_building_rules.jsonl; knowledge/workflow_blocks.json", "Rules, limitations, and ten workflow blocks."),
                ("S5", "churn_exp/CHURN_BENCHMARK.md; churn_exp/results/summary.json; CHURN-*/result.json", "All 15 fixed churn results and protocol."),
                ("S6", "agent_runs/research.sqlite3; agent_runs/<run_id>/trajectory.json", "Current agentic run status, trials, critiques, and usage."),
                ("S7", "hyperack_exp/; safe_leakage_exp/; optimized_safe_model/; general_pipeline/results/", "HyperAck 83-test lifecycle and 44-row four way benchmark."),
                ("S8", "dclab_rnd/agentic/agents.py; engine.py; worker.py; schemas.py", "Agent roles, graph sequence, bounded recipe, evaluator."),
                ("S9", "dclab_rnd/agentic/store.py; archive.py; server.py", "Evidence store, archive, local API and UI."),
                ("S10", "dclab_rnd/agentic/static/index.html; app.js", "Research, Knowledge, and Recipes interface."),
                ("S11", "knowledge/DATA_QUALITY.md; knowledge/NEXT_EXPERIMENTS.md", "Registry quality caveats and evidence gaps."),
                ("S12", "DCLAB_intiatl_MASTER_CONTEXT.md", "Initial product vision, treated as older context."),
                ("S13", "agent_runs/clean_exports/MANIFEST.json; README.md", "Current compact trial and SFT export inventory."),
            ], [12 * mm, 105 * mm, 55 * mm], font="tiny"),
            Spacer(1, 4 * mm),
            prose("A practical reading order: start with this PDF, open the Field Guide for methodology, inspect a campaign EXP JSON or Studio trial result for a claim, then check its recipe, split, code/data hashes, and limitations before reuse.", "small")])

    GuideDoc(OUT).build(story)
    return OUT


if __name__ == "__main__":
    print(build())
