"""The problems report as a PDF, from problems.json (gather.py)."""

import json
import sys
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SRC, OUT = sys.argv[1], sys.argv[2]
D = json.load(open(SRC))
ss = getSampleStyleSheet()
INK, SOFT, LINE, BAND = colors.HexColor("#1f2937"), colors.HexColor("#4b5563"), colors.HexColor("#d1d5db"), colors.HexColor("#f3f4f6")
ACC = colors.HexColor("#1d4ed8")
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17, textColor=INK, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, textColor=ACC, spaceBefore=8, spaceAfter=4)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=10.5, textColor=INK, spaceBefore=6, spaceAfter=2)
P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=9, leading=12, textColor=INK, alignment=TA_LEFT)
SM = ParagraphStyle("sm", parent=P, fontSize=7.8, leading=10, textColor=SOFT)
CELL = ParagraphStyle("cell", parent=P, fontSize=7.6, leading=9.4)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")

WHY_NOT = {  # why a kind is not in self-play yet (from the self-play code: a case must state its outcome, and run in the engine loop)
    "metric_change": "runs only as a library (no engine run), so self-play cannot solve it yet",
    "entity_tables": "its outcome is built from event tables at each cutoff (inside its own fields), which self-play does not read yet",
    "matching": "its labels are pairs of rows (a copy of another bill), not one outcome column: not in self-play yet",
    "ranking": "its outcome is a list per customer, not one column: not in self-play yet",
    "segmentation": "no outcome to predict (groups are found): not in self-play yet",
    "streams": "its outcome is an alert window per machine over time, inside its own fields: not in self-play yet",
    "anomaly": "no labels at all (anomalies are found without them): not in self-play yet",
}
PLAIN = {
    "rare_event": "Find the few bad cases (fraud, abuse, defaults) among many good ones, with a fixed review budget.",
    "metric_change": "Explain why a business number (a conversion rate, a revenue) moved between two periods.",
    "entity_tables": "Predict something about each customer from their history of events (orders, visits), at a cutoff date.",
    "forecasting": "Predict the next days of many time series (sales per store and product).",
    "matching": "Find rows that are copies of the same thing (the same bill entered twice, the same person).",
    "ranking": "Order items for each customer so the ones they will buy come first.",
    "segmentation": "Group customers by how they behave, to treat each group differently.",
    "time_to_event": "Predict how long until something happens (a cancel, a failure), when some cases have not ended yet.",
    "uplift": "Find who changes their behaviour because of an action (an offer), not who would act anyway.",
    "streams": "Watch sensor readings over time and raise an alert before a machine fails.",
    "embedded": "Use long text (or images) by turning each item into numbers first, then a normal model.",
    "embedded_images": "The same, for pictures: each image becomes numbers through an encoder.",
    "places": "Predict a number for rows on a map (prices across a city), where neighbours are alike.",
    "text_tabular": "A yes or no outcome from short text plus normal fields (reviews, tickets, notes).",
    "anomaly": "Find strange rows when nobody has labelled any, with a review list of a fixed size.",
    "binary": "Yes or no for each row (will this customer leave?).",
    "multiclass": "One of several classes for each row (which department gets this ticket?).",
    "regression": "A number for each row (how many days will delivery take?).",
}


def para(text, style=P):
    return Paragraph(escape(str(text)), style)


def rich(text, style=P):
    return Paragraph(text, style)


def table(rows, widths, header=True, zebra=True):
    data = [[c if not isinstance(c, str) else Paragraph(escape(c), CELLB if (header and i == 0) else CELL) for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), BAND))
    if zebra:
        for i in range(2 if header else 1, len(rows), 2):
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fafafa")))
    t.setStyle(TableStyle(st))
    return t


def num(v, d=3):
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return "-"


def footer(c, doc):
    c.saveState()
    c.setFont("Helvetica", 7)
    c.setFillColor(SOFT)
    c.drawString(15 * mm, 9 * mm, "DCLab R&D - the problems DCLab solves, and how it went")
    c.drawRightString(195 * mm, 9 * mm, f"page {doc.page}")
    c.restoreState()


W = 180 * mm
story = []
now = datetime.now().strftime("%Y-%m-%d %H:%M")
kinds = D["kinds"]
played = [k for k in kinds if k["selfplay"]]
tried = sum(v["tried"] for k in played for v in k["selfplay"].values())
verified = sum(v["verified"] for k in played for v in k["selfplay"].values())

# ---------------------------------------------------------------- title and summary
story += [rich("<b>DCLab: every problem it solves, and how it went</b>", ParagraphStyle("t", parent=H1, fontSize=22, leading=26)),
          para(f"Report made {now} (local time). All runs of 2026-10-21 finished: 1,100 self-play problems, 16 solver runs, 16 Claude twin runs.", SM), Spacer(1, 8)]
story += [rich("<b>In one paragraph</b>", H2),
          para("DCLab learns to solve machine-learning problems on business tables without cheating. To train its own models, it needs many "
               "solved examples, each checked. This report lists every kind of problem DCLab knows, the invented practice problems it solves "
               "by itself (self-play), and the 16 real tables it was tested on, by its own rules and by Claude."), Spacer(1, 4)]
summary = [["What", "Count", "In plain words"],
           ["Kinds of problem DCLab knows", str(len(kinds)), "fraud, forecasting, churn, offers, images, maps, text, and more"],
           ["Kinds in self-play today", str(len(played)), "invented problems with planted traps, solved and graded automatically"],
           ["Self-play problems tried", str(tried), f"{verified} passed every check ({100 * verified / max(tried, 1):.0f}%)"],
           ["Real tables (studied samples)", str(len(D["samples"])), "public tables plus HyperAck and Telco, each with a known right answer"]]
story.append(table(summary, [55 * mm, 18 * mm, 107 * mm]))
story += [Spacer(1, 6), rich("<b>How to read the rest</b>", H3),
          para("Part 1 lists every kind of problem: the invented story, its table, the traps hidden in it, the right solution, the checks a "
               "solve must pass and the self-play results. Part 2 lists the 16 real tables: their known answer, what DCLab's rules did, and what "
               "Claude did on the same table. Part 3 explains the words and the limits.")]

# the data each model now has
import glob, os
STORE = "/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D/training_records/store/sets"
BEFORE = {"m1_students": (2461, 317), "m2_workflow_decisions": (349, 36), "m3_ml_decisions": (2282, 429), "m4_workflow_llm": (248, 173),
          "m5_solver_llm": (4, 335), "m6_stepwise_decisions": (1397, 477)}
NAMES = {"m1_students": "1. Student models", "m2_workflow_decisions": "2. Workflow decision models", "m3_ml_decisions": "3. ML decision models",
         "m4_workflow_llm": "4. Workflow LLM", "m5_solver_llm": "5. Solver LLM (MLE-bench)", "m6_stepwise_decisions": "6. Step-by-step decision model"}
def lines(model, name):
    return sum(sum(1 for _ in open(f)) for f in glob.glob(f"{STORE}/{model}/**/{name}", recursive=True))
story += [rich("<b>The data each of the 6 models has now</b>", H2),
          para("Rows in the training store, in its three parts: train (to learn from), dev (to tune on) and sealed (held back for the final exam, never trained on). "
               "'Before' is the morning of 2026-10-21.", SM)]
rows = [["Model", "Train before", "Train now", "Dev now", "Sealed before", "Sealed now"]]
for m, (tb, sb) in BEFORE.items():
    rows.append([NAMES[m], f"{tb:,}", f"{lines(m, 'train.jsonl'):,}", f"{lines(m, 'dev.jsonl'):,}", f"{sb:,}", f"{lines(m, 'sealed.jsonl'):,}"])
story.append(table(rows, [52 * mm, 25 * mm, 25 * mm, 25 * mm, 26 * mm, 27 * mm]))
m5 = [["M5 set", "Train", "Dev", "Sealed", "What it holds"]]
ABOUT = {"trajectories/studio/v2": "clean whole solves of real tables by the rules, scored and truth-checked",
         "trajectories/selfplay/v1": "verified self-play solves, decision by decision",
         "preferences/selfplay/v1": "pairs: the planted truth chosen over the engine's wrong answer",
         "trajectories/silver_frontier/v1": "Claude's solves: kept apart, NOT for training until the terms are checked",
         "trajectories/development/v2": "how DCLab itself is built (Claude Code sessions)"}
for rel, about in ABOUT.items():
    base = f"{STORE}/m5_solver_llm/{rel}"
    c = lambda n: sum(1 for _ in open(f"{base}/{n}")) if os.path.exists(f"{base}/{n}") else 0
    m5.append([rel, str(c("train.jsonl")), str(c("dev.jsonl")), str(c("sealed.jsonl") + c("sealed.pending.jsonl")), about])
story += [Spacer(1, 4), rich("<b>Inside model 5, the solver</b>", H3), table(m5, [48 * mm, 14 * mm, 12 * mm, 14 * mm, 92 * mm]),
          para("Most new rows of models 1 and 3 come from self-play's invented problems: plenty, but invented. A model trained on them should be "
               "checked on the real tables' sealed rows before it is trusted.", SM)]

# overview table of kinds
story += [rich("<b>All kinds at a glance</b>", H2)]
rows = [["Kind", "What it is", "Task", "Self-play: tried / passed", "Seconds each"]]
for k in kinds:
    sp = next(iter(k["selfplay"].values()), None)
    rows.append([k["key"], PLAIN.get(k["key"], k.get("description") or ""), str(k.get("base_task") or (k.get("case") or {}).get("task") or "-"),
                 f"{sp['tried']} / {sp['verified']}" if sp else "not yet", num(sp["mean_seconds"], 1) if sp and sp.get("mean_seconds") else "-"])
story.append(table(rows, [28 * mm, 80 * mm, 20 * mm, 32 * mm, 20 * mm]))
story.append(PageBreak())

# ---------------------------------------------------------------- part 1: kinds
story += [rich("<b>Part 1. The kinds of problem, one by one</b>", H1),
          para("Each section shows one invented example (seed 0). The generator makes a new table for each seed: the same story and traps, other "
               "rows. Column names and counts only: these tables are invented, no real person is in them.", SM)]
for k in kinds:
    case = k.get("case") or {}
    block = [rich(f"<b>{escape(k['key'])}</b>: {escape(str(k.get('title') or ''))}", H2), para(PLAIN.get(k["key"], "")),
             para(k.get("description") or "", SM)]
    facts = [["Field", "Value"]]
    if case.get("moment"):
        facts.append(["The question and its moment", case["moment"]])
    facts += [["Outcome column", str(case.get("target") or "inside the kind's own fields")], ["Task", str(case.get("task") or k.get("base_task") or "-")],
              ["Rows in the example", str(case.get("rows") or "-")]]
    if k.get("metrics"):
        facts.append(["Scored by", ", ".join(k["metrics"])])
    if k.get("baselines"):
        facts.append(["Must beat", ", ".join(map(str, k["baselines"]))])
    block.append(table(facts, [45 * mm, 135 * mm]))
    story.append(KeepTogether(block))
    if case.get("columns"):
        story.append(rich("<b>The table's columns</b>", H3))
        cols = case["columns"]
        cells = [f"{c['name']} ({c['dtype']})" for c in cols]
        rows = [cells[i:i + 4] + [""] * (4 - len(cells[i:i + 4])) for i in range(0, len(cells), 4)]
        story.append(table(rows, [45 * mm] * 4, header=False))
    for name, tab in (case.get("other_tables") or {}).items():
        story.append(para(f"Other table '{name}': {tab['rows']} rows; columns {', '.join(tab['columns'])}", SM))
    if case.get("facts"):
        story.append(rich("<b>The traps and facts the generator planted</b>", H3))
        story.append(table([["Fact", "Value"]] + [[a, str(b)] for a, b in case["facts"].items()], [40 * mm, 140 * mm]))
    if case.get("solution"):
        story.append(rich("<b>The right solution (what a careful data scientist sets)</b>", H3))
        story.append(table([["Setting", "Value"]] + [[a, str(b)] for a, b in case["solution"].items()], [40 * mm, 140 * mm]))
    if k.get("checks"):
        story.append(rich("<b>What the kind checks on every run</b>", H3))
        story.append(table([["Check"]] + [[str(c if isinstance(c, str) else (c.get("label") or c.get("name") or c))] for c in k["checks"]], [180 * mm]))
    sp = k["selfplay"]
    story.append(rich("<b>Self-play results</b>", H3))
    if sp:
        for g, v in sp.items():
            story.append(para(f"{v['tried']} problems tried, {v['verified']} passed every check "
                              f"({100 * v['verified'] / max(v['tried'], 1):.0f}%), about {num(v['mean_seconds'], 1)} seconds each."))
            if v["missed"]:
                story.append(table([["Check missed by the failed solves", "Times"]] + [[m, str(n)] for m, n in v["missed"]], [160 * mm, 20 * mm]))
    else:
        story.append(para(f"Not in self-play yet: {WHY_NOT.get(k['key'], 'not run by self-play yet')}."))
    story.append(Spacer(1, 6))
story.append(PageBreak())

# ---------------------------------------------------------------- part 2: real tables
story += [rich("<b>Part 2. The 16 real tables</b>", H1),
          para("Each table has a known right answer from the R&D (its outcome and the columns that leak). Two kinds of run were made blind (the "
               "known leak columns hidden): the solver run, where DCLab's rules got the outcome named as a person would name it; and the twin "
               "run, where the rules and Claude got the same task without the outcome named. 'Passed' means both the outcome and the leak "
               "columns matched the known answer.", SM)]
MET = {"roc_auc": "AUC", "average_precision": "AP", "mae": "MAE", "rmse": "RMSE", "log_loss": "log loss", "macro_f1": "macro F1", "accuracy": "acc."}
rows = [["Table", "Rows x cols", "Outcome", "Known leaks", "Rules (solver run)", "Rules score", "Twin: who was right", "Claude score", "Claude EUR"]]
for s in D["samples"]:
    sv = (s["solver_runs"] or [{}])[-1]
    tw = (s["twin_runs"] or [{}])[-1]
    v = (sv.get("verdict") or {}).get("system") or {}
    rl = (sv.get("lanes") or {}).get("workflow") or {}
    al = (tw.get("lanes") or {}).get("ai_first") or {}
    winner = {"both": "both", "system": "rules only", "ai_first": "Claude only", "neither": "neither"}.get((tw.get("verdict") or {}).get("winner"), "-")
    rows.append([s["key"], f"{(s['table'] or {}).get('rows', '-')} x {(s['table'] or {}).get('columns', '-')}", str(s.get("target")), str(len(s.get("known_leaks") or [])),
                 ("passed" if v and all(v.values()) else "failed") if v else "-", f"{num(rl.get('score'))} {MET.get(rl.get('metric'), rl.get('metric') or '')}", winner,
                 f"{num(al.get('score'))} {MET.get(al.get('metric'), al.get('metric') or '')}" if al.get("score") is not None else "-",
                 num(tw.get("eur"), 2) if tw else "-"])
story.append(table(rows, [26 * mm, 18 * mm, 20 * mm, 12 * mm, 18 * mm, 22 * mm, 22 * mm, 24 * mm, 18 * mm]))
story.append(para("Scores are each run's holdout number on its own metric, read once: AUC and AP (average precision) are better when higher, MAE and log loss when lower; "
                  "so scores of different tables cannot be compared. "
                  "In the twin runs the rules often stopped at 'name the outcome first', since the task named none: that is DCLab's rule, not a mistake.", SM))
for s in D["samples"]:
    sv = (s["solver_runs"] or [{}])[-1]
    tw = (s["twin_runs"] or [{}])[-1]
    block = [rich(f"<b>{escape(s['key'])}</b>: {escape(str(s.get('name') or ''))}", H2)]
    rows = [["Field", "Value"], ["Industry", str(s.get("industry") or "-")],
            ["Table", f"{(s['table'] or {}).get('rows', '-')} rows, {(s['table'] or {}).get('columns', '-')} columns"],
            ["Outcome and task", f"{s.get('target')} ({s.get('task') or '-'}; metric {s.get('metric') or 'default'})"],
            ["Prediction moment", s.get("prediction_moment") or "-"],
            ["Identifiers / time / group", f"{', '.join(s.get('identifiers') or []) or 'none'} / {s.get('time_column') or 'none'} / {s.get('group_column') or 'none'}"]]
    block.append(table(rows, [45 * mm, 135 * mm]))
    if s.get("known_leaks"):
        block.append(rich("<b>The known leak columns, and why</b>", H3))
        block.append(table([["Column", "Why it leaks"]] + [[c, s["known_leak_reasons"].get(c) or "-"] for c in s["known_leaks"]], [50 * mm, 130 * mm]))
    else:
        block.append(para("No known leak column: the right answer is to forbid nothing (and still keep identifiers out)."))
    rl = (sv.get("lanes") or {}).get("workflow") or {}
    v = (sv.get("verdict") or {}).get("system") or {}
    res = [["Run", "Leak columns it forbade", "Verdict", "Holdout (interval)", "Model"]]
    if sv:
        res.append(["Rules, solver run (outcome named)", ", ".join(sv.get("rules_leaks") or []) or "none", ("passed" if v and all(v.values()) else "failed") if v else "-",
                    f"{num(rl.get('score'))} ({num(rl.get('low'))} to {num(rl.get('high'))})", str(rl.get("model") or "-")])
    if tw:
        tv = tw.get("verdict") or {}
        for lane, name, leaks in (("workflow", "Rules, twin run (outcome not named)", tw.get("rules_leaks")), ("ai_first", f"Claude, twin run (EUR {num(tw.get('eur'), 2)})", tw.get("claude_leaks"))):
            ln = (tw.get("lanes") or {}).get(lane) or {}
            arm = (tv.get("system" if lane == "workflow" else "ai_first") or {})
            res.append([name, ", ".join(leaks or []) or ("none" if leaks is not None else "stopped before choosing"),
                        ("passed" if arm and all(arm.values()) else "failed") if arm else "-",
                        f"{num(ln.get('score'))} ({num(ln.get('low'))} to {num(ln.get('high'))})" if ln.get("score") is not None else "-", str(ln.get("model") or "-")])
    block.append(rich("<b>What happened</b>", H3))
    block.append(table(res, [52 * mm, 50 * mm, 16 * mm, 34 * mm, 28 * mm]))
    story.append(KeepTogether(block))
    story.append(Spacer(1, 4))
story.append(PageBreak())

# ---------------------------------------------------------------- part 3: words and limits
story += [rich("<b>Part 3. The words, and the limits</b>", H1)]
words = [["Word", "Meaning, with an example"],
         ["Leak", "A column known only after the moment of prediction. Example: a 'final refund amount' in a churn table exists only after the customer left."],
         ["Holdout", "Rows kept aside and scored once at the end, like an exam nobody saw before. Reading it twice makes the score look better than it is."],
         ["Planted truth", "The right answer the generator (or the R&D) knows in advance, so a solve can be graded automatically."],
         ["Blind run", "A run where the known answer is hidden, so the solver must find the traps itself."],
         ["Self-play", "DCLab inventing a problem, solving it with its own rules, and keeping it only if every check passes."],
         ["Verified", "A self-play solve that passed every check of its kind."],
         ["Interval", "The range the true score probably lies in. A wide interval (fraud: 0.30 to 1.00) means few test cases: the number is not precise."],
         ["Silver", "Examples made by a hosted model (Claude): kept apart, lower weight, not used for training until the provider's terms are checked."]]
story.append(table(words, [30 * mm, 150 * mm]))
story += [rich("<b>The limits, said plainly</b>", H2)]
for line in ["Self-play tables are invented. They teach the habits (when is each column known, keep groups together, read the holdout once), "
             "but they are cleaner than real business tables, which have missing values, odd names and traps nobody planted.",
             "There are 16 real tables only. Running them again adds copies, not new problems. More real tables would help more than more runs.",
             "Seven kinds are not in self-play yet (see Part 1), so the models see no practice examples of them.",
             "The rules' leak finder missed 8 of 9 activity columns on online_shoppers and took fraud's time column for a leak: a known weakness.",
             "Claude's twin runs are a comparison on these 16 tables with these caps; they are not a claim about other problems.",
             "No model is trained yet. These examples pay off only when a model trained on them beats the rules on sealed data."]:
    story.append(para("- " + line))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=16 * mm,
                        title="DCLab problems report", author="DCLab R&D")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
