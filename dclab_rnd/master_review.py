"""Build the outsider-readable and machine-readable DCLab model-building guide."""
import argparse
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaigns/model_building_50_v1/results"
KNOWLEDGE = ROOT / "knowledge"

STAGES = ("data_understanding", "leakage_audit", "feature_engineering", "model_selection", "optimization_reliability")

def read(path): return json.loads(Path(path).read_text())

def campaign_records():
    records = {}
    for path in sorted(CAMPAIGN.glob("EXP-*.json")):
        item = read(path)
        stage = next(s for s in STAGES if path.stem.endswith(s))
        records.setdefault(item["dataset"], {})[stage] = {"id": path.name.split("_")[0], "path": str(path.relative_to(ROOT)), **item}
    return records

def build_pack():
    registry = read(KNOWLEDGE / "evidence.json")
    churn = read(ROOT / "churn_exp/results/summary.json")
    records = campaign_records()
    datasets = []
    for dataset, stages in records.items():
        data, leak, feat, model, opt = (stages[s] for s in STAGES)
        de, le, fe, me, oe = (data["evidence"], leak["evidence"], feat["evidence"], model["evidence"], opt["evidence"])
        datasets.append({
            "dataset": dataset,
            "evidence_ids": {s: stages[s]["id"] for s in STAGES},
            "source_paths": {s: stages[s]["path"] for s in STAGES},
            "data": {k: de.get(k) for k in ("source_rows", "analyzed_rows", "feature_count", "positive_rate_train", "missing_cell_rate_train", "duplicate_row_rate_train", "risk_feature_count")},
            "leakage": {"declared": le.get("declared_leakage_features", []), "review_candidates": le.get("heuristic_review_candidates", []), "policy_rationale": le.get("policy_rationale"), "safe_vs_unsafe": le.get("safe_vs_unsafe_training_cv")},
            "feature_engineering": {k: fe.get(k) for k in ("raw_auc_mean", "selected_stage", "selected_auc_mean", "selected_vs_raw_auc", "selection_rule")},
            "model_selection": {k: me.get(k) for k in ("selected_model", "selected_auc_mean", "selected_auc_std", "selection_rule")},
            "optimization": {"accepted": oe.get("selected_optimization"), "cv_lift": oe.get("optimized_vs_baseline_cv_auc"), "holdout_metrics": oe.get("holdout_metrics"), "holdout_auc_ci": oe.get("holdout_roc_auc_bootstrap_ci"), "holdout_consumed": oe.get("holdout_consumed"), "top_features": oe.get("top_final_features", [])[:8]},
        })
    return {
        "schema_version": 1,
        "scope": {"registry_experiments": 579, "registry_datasets": 11, "campaign_experiments": 50, "campaign_datasets": 10, "hyperack_historical_experiments": 83, "churn_experiments": churn["completed"], "distinct_datasets_including_churn": 12},
        "boundaries": ["Historical registry records and the 50-experiment campaign use different protocols and must not be pooled as independent replications.", "HyperAck's 83 experiments are included in the 579-record registry inventory.", "The churn campaign is adaptive development CV with no independent holdout.", "Public benchmark performance is not production validation."],
        "campaign": datasets,
        "cross_dataset_models": registry["model_evidence"],
        "registry_leakage_gaps": registry["leakage_gaps"],
        "hyperack": {"safe_champion_auc": 0.9454609956777585, "unsafe_auc": 0.9801821330000522, "unsafe_lift": 0.03472113732229365, "blocked_features": ["final_customer_fare", "final_biker_fare"], "safe_source": "optimized_safe_model/results/53_softvote_etbag_lgbmwinner_xgb.json", "unsafe_source": "general_pipeline/results/unsafe_baseline_hist_gradient_boosting.json"},
        "churn": churn,
        "registry_quality": {"errors": 0, "warnings": 2, "full_provenance_records": 10, "source": "knowledge/DATA_QUALITY.md"},
    }

def rules():
    def rule(i, category, statement, why, evidence, confidence="methodological", exceptions="None; apply judgment to the prediction contract.", test="Record the decision and test it on identical folds."):
        return {"rule_id": f"DCLAB-R{i:02d}", "category": category, "statement": statement, "why": why, "evidence": evidence, "confidence": confidence, "exceptions": exceptions, "failure_signal": "The result changes materially under a valid ablation, later-time slice, missing-input simulation, or independent evaluation.", "next_test": test, "production_approved": False}
    return [
        rule(1,"problem_contract","Write the prediction moment, target window, action, unit of prediction and error costs before inspecting target associations.","Leakage is defined relative to what is knowable at a specific decision moment, not by a column name.",["campaigns/model_building_50_v1/AGENT_CONTEXT.md"],"high methodological"),
        rule(2,"splitting","Split by the real independence unit and time direction before preprocessing or feature selection.","Random rows leak entity history and future regimes when observations are related.",["campaigns/model_building_50_v1/MODEL_BUILDING_WORKFLOW.md"],"high methodological"),
        rule(3,"eda","Fit target-aware EDA, imputers, encoders, scalers, selectors and learned aggregations on training partitions only.","Any statistic that sees evaluation labels or distributions can make validation optimistic.",["campaigns/model_building_50_v1/CAMPAIGN_REPORT.md"],"high methodological"),
        rule(4,"leakage","Treat post-outcome, future, target-derived, cross-split aggregate, duplicate/entity and preprocessing leakage as separate failure modes.","One correlation detector cannot identify all leakage mechanisms.",["EXP-007","EXP-042","knowledge/KNOWLEDGE_BASE.md"],"high methodological"),
        rule(5,"leakage","A suspicious name or high univariate AUC starts a semantic review; it does not prove leakage.","Availability requires source lineage and timestamps; useful real predictors can also be highly associated.",["campaigns/model_building_50_v1/CAMPAIGN_REPORT.md"],"high methodological"),
        rule(6,"leakage","A large safe-versus-unsafe score gap is a severity demonstration, not the primary proof of leakage.","Bank marketing and online shoppers showed +0.1328 to +0.1575 registry gaps, while semantics identified the forbidden fields.",["knowledge/evidence.json","EXP-007","EXP-042"],"strong empirical"),
        rule(7,"feature_engineering","Begin with a strong raw baseline and promote the smallest defensible feature set within a predeclared tolerance.","Seven of ten campaign datasets selected raw features; extra features often added dilution without repeatable lift.",["EXP-003","EXP-013","EXP-023","EXP-033","EXP-038","EXP-043"],"moderate empirical","Domain transformations may still be essential when raw representation hides a known relationship."),
        rule(8,"feature_engineering","Generate features from domain invariances and mechanisms before blind combinatorial expansion.","Ratios helped bank marketing and wine quality; interactions helped German credit; churn charge/tenure features improved logistic AUC by +0.0050.",["EXP-008","EXP-028","EXP-048","CHURN-014"],"moderate empirical"),
        rule(9,"feature_engineering","Every generated feature needs explicit lineage, availability time, units, null behavior and a reproducible formula.","A mathematically valid feature can still be unavailable, unstable or impossible to reproduce online.",["campaigns/model_building_50_v1/MODEL_BUILDING_WORKFLOW.md","churn_exp/results/CHURN-014/recipe.json"],"high methodological"),
        rule(10,"feature_reliability","Importance is only a nomination signal; production reliability additionally requires availability, fold stability, coverage, missing-input resilience and drift monitoring.","The campaign explicitly rejects importance-only promotion and the agentic worker measures missing-input stress.",["EXP-005","dclab_rnd/agentic/worker.py"],"high methodological"),
        rule(11,"categoricals","Preserve raw categorical semantics and fit encoders inside folds; do not treat arbitrary category codes as ordered numbers.","Legacy factorized caches lose labels and can introduce false ordinal structure.",["campaigns/model_building_50_v1/CAMPAIGN_REPORT.md","dclab_rnd/agentic/catalog.py"],"high methodological"),
        rule(12,"missingness","Model missingness as an operational state: measure coverage, add indicators when justified, simulate absence, and define fallback behavior.","A feature that disappears in production can dominate offline ranking yet make the system brittle.",["dclab_rnd/agentic/worker.py","churn_exp/results/CHURN-014/result.json"],"high methodological"),
        rule(13,"model_selection","There is no universal best tabular algorithm; screen a transparent linear baseline and several nonlinear families on identical folds.","LightGBM had best mean rank across the registry, Extra Trees won 7/10 in the controlled campaign, logistic won churn, and an ensemble won HyperAck.",["knowledge/evidence.json","campaigns/model_building_50_v1/CAMPAIGN_REPORT.md","CHURN-014"],"strong empirical"),
        rule(14,"model_selection","Use mean performance, fold stability, probability quality, runtime and operational constraints—not ROC-AUC alone.","Small AUC wins can coincide with worse log loss, calibration, recall or latency.",["EXP-004","campaigns/model_building_50_v1/CAMPAIGN_REPORT.md","CHURN-010"],"high methodological"),
        rule(15,"optimization","Optimize only after data, leakage, features, metric and candidate family are locked.","Early broad tuning increases adaptive overfitting and obscures what caused improvement.",["campaigns/model_building_50_v1/MODEL_BUILDING_WORKFLOW.md"],"high methodological"),
        rule(16,"optimization","Compare tuning and feature changes with a paired baseline on the same folds and demand a predeclared useful margin.","The campaign used a 0.001 CV gate; this is a project convention, not a universal constant.",["EXP-005","EXP-030"],"high methodological","Use a larger gate when business value, noise or retraining cost demands it."),
        rule(17,"evaluation","Consume the final holdout once, after all choices are locked, and never tune from its result.","Repeated consultation converts a holdout into development data.",["campaigns/model_building_50_v1/CAMPAIGN_REPORT.md"],"high methodological"),
        rule(18,"evaluation","Choose metrics and thresholds from the real decision cost; report ranking, probability quality and operating-point errors together.","AUC does not specify calibration or the false-positive/false-negative tradeoff.",["campaigns/model_building_50_v1/CAMPAIGN_REPORT.md","churn_exp/CHURN_BENCHMARK.md"],"high methodological"),
        rule(19,"generalization","Use temporal, geographic, source-system and out-of-domain slices when those shifts exist.","Random CV estimates interpolation under the historical mixture, not future operational performance.",["knowledge/NEXT_EXPERIMENTS.md"],"high methodological"),
        rule(20,"llm_governance","Let LLM agents propose hypotheses, challenge semantics and curate cited lessons; deterministic executors own splits, metrics and artifacts.","This preserves creativity without letting generated text become measurement.",["dclab_rnd/agentic/agents.py","dclab_rnd/agentic/engine.py"],"high system-design"),
        rule(21,"knowledge","Store claims with scope, evidence IDs, counterevidence, uncertainty and the next discriminating test.","A score without context cannot teach a future person or model when the lesson applies.",["campaigns/model_building_50_v1/agent_memory.jsonl","dclab_rnd/agentic/store.py"],"high methodological"),
        rule(22,"promotion","A candidate is not production-ready until provenance, independent confirmation, feature contracts, monitoring, fairness/privacy and rollback gates are reviewed.","Benchmark excellence does not establish operational safety.",["campaigns/model_building_50_v1/CAMPAIGN_REPORT.md","knowledge/DATA_QUALITY.md"],"high methodological"),
    ]

def workflow_blocks():
    return [
        {"id":"WF-01","name":"Prediction contract","flow":["define entity and decision","define prediction moment","define target window","define action and costs","freeze allowable information"]},
        {"id":"WF-02","name":"Source and lineage gate","flow":["inventory sources","hash immutable snapshot","record event timestamps","map raw-to-modeled fields","label owner and SLA"]},
        {"id":"WF-03","name":"Split design","flow":["identify independence unit","choose group/time/domain split","lock test IDs","verify no overlap","version split manifest"]},
        {"id":"WF-04","name":"Train-only EDA","flow":["schema and type audit","target and missingness profile","duplicates/entities","distribution slices","association hypotheses without promotion"]},
        {"id":"WF-05","name":"Leakage audit","flow":["semantic availability matrix","lineage/timestamp check","target-proxy review","split/preprocessing audit","safe-vs-unsafe severity ablation"]},
        {"id":"WF-06","name":"Feature ladder","flow":["raw baseline","single domain transform","ratios/differences","bounded interactions","external point-in-time features","retain only reproducible lift"]},
        {"id":"WF-07","name":"Algorithm screen","flow":["dummy floor","regularized linear model","bagged trees","boosting families","domain-appropriate specialist","rank on identical folds"]},
        {"id":"WF-08","name":"Conservative optimization","flow":["freeze objective","small explicit search","paired fold comparison","proper scoring and runtime","accept or keep baseline"]},
        {"id":"WF-09","name":"Reliability challenge","flow":["calibration","threshold/cost analysis","missing-input stress","subgroup/slice floors","temporal/domain holdout","failure and rollback rule"]},
        {"id":"WF-10","name":"Knowledge capture","flow":["observed result","narrow interpretation","limitations/counterexample","evidence paths and hashes","next discriminating test","human review status"]},
    ]

def f(value, digits=4): return "—" if value is None else f"{value:.{digits}f}"
def md_table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"]*len(headers)) + "|\n" + "\n".join("| " + " | ".join(map(str,row)) + " |" for row in rows)

def render_markdown(pack, ruleset, workflows, llm_review=None):
    campaign = pack["campaign"]
    data_rows = [[d["dataset"], d["data"]["source_rows"], d["data"]["feature_count"], f(d["data"]["positive_rate_train"],3), f(d["data"]["duplicate_row_rate_train"],3), d["data"]["risk_feature_count"]] for d in campaign]
    feature_rows = [[d["dataset"], f(d["feature_engineering"]["raw_auc_mean"]), d["feature_engineering"]["selected_stage"], f(d["feature_engineering"]["selected_vs_raw_auc"]), d["model_selection"]["selected_model"], f(d["optimization"]["holdout_metrics"].get("roc_auc"))] for d in campaign]
    leakage_rows = [[x["dataset"], f(x["safe_roc_auc"]), f(x["unsafe_roc_auc"]), f(x["apparent_leakage_lift"])] for x in pack["registry_leakage_gaps"]]
    model_rows = [[x["model_family"],x["dataset_count"],x["wins"],f(x["mean_rank"],2),f(x["mean_roc_auc"])] for x in pack["cross_dataset_models"]]
    lines = ["# DCLab Model-Building Field Guide", "", "**Audience:** data scientists, ML engineers, reviewers, product owners, and future LLM agents who were not present for these experiments.", "", "> This is a research guide, not a production approval document. Rules are either methodological safeguards or scoped empirical defaults; every empirical claim includes evidence and an exception path.", "", "## Executive orientation", "", "DCLab's evidence should be read as four connected but non-independent bodies: a 579-record historical registry across 11 datasets (including 83 HyperAck experiments), a controlled 50/50 campaign on ten public datasets with one final holdout per dataset, a 15-run Telco Churn development benchmark, and live agentic trials. Do not add their experiment counts as if they were independent replications.", "", "The central lesson is simple: **the best model is usually determined earlier than training**. A precise prediction contract, honest split, leakage-safe feature set and meaningful objective can matter more than the last round of hyperparameter tuning.", "", "### What the evidence says", "", "- No algorithm won everywhere. LightGBM has the strongest mean registry rank; Extra Trees was selected on 7/10 controlled campaign datasets; engineered logistic regression led churn; a safe soft-voting ensemble led HyperAck.", "- Feature engineering was selective, not automatically beneficial: raw features were retained on 7/10 controlled datasets; ratios helped two, interactions helped one, and charge-domain features helped churn.", "- Leakage produced larger apparent improvements than most legitimate optimization: +0.1328 for bank marketing, +0.1575 for online shoppers and +0.0347 for HyperAck in the historical registry.", "- A feature's importance does not prove that it exists at prediction time, remains available, is causal, or survives production missingness.", "- Final holdouts, once consumed, are historical evidence. They must not be reused for further selection.", "", "## 1. Start with the prediction contract", "", "Before EDA, write one sentence: `For [entity], at [time], predict [outcome during window], so [decision] can be made, with [error costs].` This sentence determines whether a feature is legal. The same column can be safe for a retrospective diagnosis model and leakage for an early-warning model.", "", "Minimum contract:", "", "1. unit of prediction and grouping identity;", "2. scoring time and latency budget;", "3. target definition, observation window and label delay;", "4. permitted source systems and data cutoff;", "5. action taken from the prediction;", "6. false-positive, false-negative, abstention and delay costs;", "7. populations, exclusions, privacy and fairness constraints.", "", "## 2. Data understanding and EDA", "", md_table(["Dataset","Source rows","Features","Positive rate","Duplicate rate","Risk flags"],data_rows), "", "### An intuitive EDA sequence", "", "1. **Shape and schema:** one row per intended entity-event, unique keys, types, units and category meanings.", "2. **Target construction:** independently reproduce the label and check that its time window follows the prediction moment.", "3. **Missingness:** distinguish not-collected, not-applicable, delayed, failed-system and true unknown states.", "4. **Duplicates and entities:** exact duplicates are only the easiest case; repeated people, devices, households or orders need grouped validation.", "5. **Distributions:** inspect ranges, tails, impossible values, cardinality and slices. Random-split PSI is only a smoke test.", "6. **Target associations:** use them to generate questions, never to approve features. Target-aware summaries belong inside training partitions.", "7. **Representativeness:** compare time, geography, source system and deployment population—not only random rows.", "", "### EDA questions an LLM agent should ask", "", "- What real-world event creates each row and each column?", "- Which fields can be revised after the event?", "- Are nulls generated by the business process?", "- Does one entity appear more than once?", "- Which categories or ranges could appear after deployment but not in training?", "- Which high associations are plausible mechanisms, and which look like outcome recording?", "", "## 3. Leakage: the practical taxonomy", "", md_table(["Dataset","Safe AUC","Unsafe AUC","Apparent lift"],leakage_rows), "", "### Leakage types", "", "- **Direct target leakage:** a target copy or deterministic encoding.", "- **Post-outcome leakage:** information recorded because the outcome already happened, such as HyperAck final fares.", "- **Future leakage:** later measurements used to predict an earlier event.", "- **Window leakage:** aggregates whose calculation window crosses the prediction time.", "- **Entity leakage:** the same person/order/device appears across train and evaluation.", "- **Preprocessing leakage:** imputers, encoders, scalers or selectors fitted before splitting.", "- **Selection leakage:** repeated model/feature decisions made from the same test or holdout.", "- **External-data leakage:** a join uses current values instead of the historically available vintage.", "", "### How to decide whether a feature is leakage", "", "Use four tests together: (1) semantic availability at the exact prediction moment, (2) source lineage and event timestamps, (3) split/pipeline isolation, and (4) a removal or time-shift ablation to measure severity. The ablation measures impact; it does not replace the semantic proof.", "", "## 4. Feature engineering", "", md_table(["Dataset","Raw CV AUC","Selected recipe","Δ vs raw","Selected model","Final holdout AUC"],feature_rows), "", "### Generation ladder", "", "Move from interpretable, low-search transformations toward higher-risk ones:", "", "1. correct units, data types, category semantics and missing states;", "2. monotonic transforms for skew (`log1p`, clipping only with a domain reason);", "3. domain ratios and differences with explicit denominator-zero behavior;", "4. calendar/cyclical features derived from timestamps available at scoring time;", "5. bounded interactions representing a mechanism;", "6. history/count/recency aggregates computed strictly before the cutoff;", "7. external features with licensed, point-in-time joins;", "8. learned representations only after simpler baselines and leakage controls.", "", "### Feature reliability scorecard", "", "A feature should answer **yes** to: Is it known at scoring time? Is the formula reproducible online? Is its unit and category mapping stable? Does it appear often enough? Does it help on paired folds? Is the gain stable across time/slices? Can the model tolerate its absence? Can drift, coverage and latency be monitored? Is its use lawful and fair? A `no` does not always mean remove it; it means define fallback, redesign or collect better evidence.", "", "### What not to do", "", "Do not create every pairwise interaction, use feature importance as approval, arithmetic-combine categorical codes, compute customer history across the cutoff, tune feature recipes against the final holdout, or keep a feature merely because a complex model can use it.", "", "## 5. Model-family selection", "", md_table(["Family","Datasets","Wins","Mean rank","Mean ROC-AUC"],model_rows), "", "### Practical defaults by data behavior", "", "- **Dummy + logistic regression:** mandatory floors; often competitive on additive effects and smaller data, and easiest to calibrate/explain.", "- **Extra Trees / Random Forest:** strong when nonlinear interactions and mixed thresholds matter; regularization via depth/leaf size is often more valuable than more trees.", "- **Histogram boosting / LightGBM / XGBoost / CatBoost:** strong general tabular candidates. Prefer native categorical handling only when raw semantics are preserved and fold isolation is verified.", "- **Ensembles:** use only when members make complementary errors and the gain survives paired evaluation; HyperAck's safe ensemble gain is narrow.", "- **Tabular transformers:** not a default. They need data scale, appropriate regularization and evidence that they beat strong boosted-tree/linear baselines for the same cost.", "", "The correct conclusion is not 'LightGBM is best.' It is: 'LightGBM is a strong default challenger in this registry, while the winner changes with sample size, representation, objective and protocol.'", "", "## 6. Optimization without fooling yourself", "", "1. Freeze the metric, folds, feature recipe and candidate family.", "2. Search a small explicit space based on model behavior.", "3. Compare every candidate to its paired baseline on identical folds.", "4. Check mean delta, direction across folds, variability, log loss/Brier/calibration, runtime and complexity.", "5. Accept tuning only if it clears a predeclared useful margin; otherwise keep the baseline.", "6. Select thresholds from OOF predictions and decision costs—not from the final test.", "7. Evaluate the locked pipeline once on the untouched holdout.", "", "The campaign's +0.001 CV gate and 0.002 feature tolerance are DCLab conventions for screening, not statistical laws. A high-risk or noisy use case should demand a larger margin and stronger replication.", "", "## 7. Evaluation and production reliability", "", "Report discrimination (ROC-AUC and average precision), probability quality (log loss, Brier and calibration), operating metrics at cost-selected thresholds, uncertainty, runtime, and slice/stress results. For imbalance, accuracy can be actively misleading. For intervention systems, calibration and decision cost may matter more than rank.", "", "A production challenge set should include later time, unseen entities, source outages, missing critical features, rare categories, geographic/source shifts and protected/relevant subgroups. Define a performance floor and rollback rule before deployment.", "", "## 8. Case studies", "", "### HyperAck", "", "The unsafe historical ceiling reached about 0.9802 ROC-AUC, while the optimized safe champion reached 0.9455. Final customer and biker fares are forbidden for pre-dispatch prediction. The 0.0347 gap shows how a plausible business field can create a large false improvement. The safe champion is still historical random-split evidence; temporal, zone and demand-regime backtests remain necessary.", "", "### Telco Churn", "", "Across 15 repeated development-CV comparisons, logistic regression with tenure/charge features led at 0.8499 ROC-AUC and improved +0.0050 over plain logistic regression. Regularized histogram boosting was second at 0.8478. The lesson is not that logistic always wins; it is that domain representation can outperform complexity when the signal is mostly additive and the data are modest. There is no independent churn holdout yet.", "", "### Controlled 50-experiment campaign", "", "Raw features were selected on 7/10 datasets, ratios on two and interactions on one. Extra Trees led the model screen on seven datasets, but the historical registry gives LightGBM the best average rank across eleven. These results differ because protocols, candidate sets and selection rules differ. That disagreement is useful evidence against universal model rules.", "", "## 9. Reusable workflow blocks", ""]
    for block in workflows:
        lines += [f"### {block['id']} · {block['name']}", "", " → ".join(block["flow"]), ""]
    lines += ["## 10. Rules for future use cases", ""]
    for r in ruleset:
        lines += [f"### {r['rule_id']} · {r['category'].replace('_',' ').title()}", "", f"**Rule:** {r['statement']}", "", f"**Why:** {r['why']}", "", f"**Confidence:** {r['confidence']}. **Exceptions:** {r['exceptions']}", "", f"**Evidence:** {', '.join(f'`{e}`' for e in r['evidence'])}", "", f"**Next test:** {r['next_test']}", ""]
    lines += ["## 11. LLM-agent operating model", "", "Use agents as a scientific team with constrained responsibilities:", "", "- **Planner:** converts the business goal into falsifiable questions and budgets.", "- **Data/lineage reviewer:** audits schema, timing, missingness and leakage hypotheses.", "- **Experiment designer:** proposes one bounded, typed experiment and cites prior evidence.", "- **Deterministic executor:** owns data, splits, transformations, training and metrics; it never executes generated code.", "- **Results critic:** separates observation from interpretation and requests counterevidence.", "- **Knowledge curator:** stores narrow claims, evidence paths, scope, exceptions and next tests.", "", "LLM outputs are suggestions, not measurements. Never allow an agent to invent rows, metrics, citations, production availability or approvals. The strongest future training record is not a chat transcript; it is a structured decision with an evidence pointer and a known counterexample.", "", "## 12. Future-LLM training format", "", "Each durable example should contain: problem contract, dataset fingerprint, split protocol, hypothesis, workflow block IDs, typed configuration, measured outputs, observation, interpretation, limitations, counterevidence, failure mode, next test, human-review status and `training_ready=false` until curated. Include negative results and rejected experiments so the model learns when not to add complexity.", "", "## 13. Remaining evidence gaps", "", "1. Reproduce historical champions with complete code/data/environment provenance; only 10 registry records currently contain it.", "2. Restore raw category labels and compare fold-fitted/native categorical handling to legacy codes.", "3. Add temporal and operational holdouts for HyperAck and an untouched confirmation set for churn.", "4. Repeat close choices with more resamples or nested validation.", "5. Add cost matrices, threshold selection, fairness/privacy review and monitoring contracts.", "6. Treat 1.0 benchmark scores as audit triggers for duplicates, target proxies and overly easy splits—not automatic success.", ""]
    if llm_review:
        lines += ["## 14. GPT‑6 Astra advisory review", "", "> Agent-generated critique. Deterministic measurements above remain authoritative.", ""]
        for section, review in llm_review.get("sections", {}).items():
            lines += [f"### {section.replace('_',' ').title()}", "", review.get("audience_summary", ""), ""]
            for finding in review.get("findings", []): lines.append(f"- {finding}")
            lines += [""]
            for recommendation in review.get("recommendations", []): lines.append(f"- **Recommendation:** {recommendation}")
            lines += [""]
    lines += ["## Evidence map", "", "- Controlled campaign: `campaigns/model_building_50_v1/CAMPAIGN_REPORT.md`", "- Historical registry: `knowledge/KNOWLEDGE_BASE.md` and `knowledge/evidence.json`", "- Churn campaign: `churn_exp/CHURN_BENCHMARK.md`", "- HyperAck safe report: `optimized_safe_model/OPTIMIZED_SAFE_REPORT.md`", "- Machine rules: `knowledge/model_building_rules.jsonl`", "- Evidence pack: `knowledge/master_evidence_pack.json`", "- Typed GPT-6 Astra workflow: `dclab_rnd/agentic/guide_review.py` (advisory review appears when API credits are available)", "- Reviewer lifecycle status: `knowledge/gpt6_astra_master_review_status.json`", ""]
    return "\n".join(lines)

def render_html(markdown):
    # Small, dependency-free renderer for the local studio; tables are real HTML.
    def inline(value):
        value = html.escape(value)
        value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
        value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
        return value

    out=[]; list_open=False; table_rows=[]
    def close_list():
        nonlocal list_open
        if list_open: out.append("</ul>"); list_open=False
    def flush_table():
        nonlocal table_rows
        if not table_rows: return
        header=table_rows[0]; body_rows=table_rows[1:]
        out.append("<div class='table-wrap'><table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in header) + "</tr></thead><tbody>")
        for row in body_rows:
            out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
        out.append("</tbody></table></div>"); table_rows=[]
    for raw in markdown.splitlines():
        stripped=raw.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells=[c.strip() for c in stripped.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells): table_rows.append(cells)
            continue
        flush_table()
        if not stripped:
            close_list(); continue
        if stripped.startswith("# "):
            close_list(); out.append(f"<h1>{inline(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            close_list(); out.append(f"<h2>{inline(stripped[3:])}</h2>")
        elif stripped.startswith("### "):
            close_list(); out.append(f"<h3>{inline(stripped[4:])}</h3>")
        elif stripped.startswith("> "):
            close_list(); out.append(f"<blockquote>{inline(stripped[2:])}</blockquote>")
        elif stripped.startswith("- "):
            if not list_open: out.append("<ul>"); list_open=True
            out.append(f"<li>{inline(stripped[2:])}</li>")
        elif re.match(r"^\d+\. ", stripped):
            close_list(); out.append(f"<p class='step'>{inline(stripped)}</p>")
        else:
            close_list(); out.append(f"<p>{inline(stripped)}</p>")
    flush_table(); close_list()
    body="\n".join(out)
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>DCLab Model-Building Field Guide</title><style>body{margin:0;background:#f6f7f9;color:#1c2937;font:16px/1.7 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}main{max-width:1060px;margin:auto;background:white;padding:56px 8vw;box-shadow:0 0 40px #17222c0a}h1{font-size:42px;line-height:1.15;color:#142b3d}h2{margin-top:52px;border-top:1px solid #e5e9ee;padding-top:28px;color:#18384a}h3{margin-top:30px;color:#16796c}p,li{color:#526272}.step{margin:5px 0 5px 18px}blockquote{border-left:4px solid #16796c;background:#eaf5f1;padding:16px 20px;margin:24px 0}.table-wrap{overflow:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}th,td{border:1px solid #dde4e7;padding:9px 11px;text-align:left;vertical-align:top}th{background:#eef5f3;color:#173f3a;font-weight:700}tr:nth-child(even) td{background:#fafcfc}code{background:#eef2f2;padding:2px 5px;border-radius:4px;font-size:.9em}strong{color:#263f4d}@media(max-width:700px){main{padding:28px 20px}h1{font-size:32px}th,td{padding:7px;font-size:12px}}</style></head><body><main>"""+body+"</main></body></html>"

def build():
    KNOWLEDGE.mkdir(exist_ok=True)
    pack, ruleset, workflows = build_pack(), rules(), workflow_blocks()
    (KNOWLEDGE / "master_evidence_pack.json").write_text(json.dumps(pack, indent=2, allow_nan=False))
    (KNOWLEDGE / "model_building_rules.jsonl").write_text("\n".join(json.dumps(x, allow_nan=False) for x in ruleset)+"\n")
    (KNOWLEDGE / "workflow_blocks.json").write_text(json.dumps(workflows, indent=2))
    review_path = KNOWLEDGE / "gpt6_astra_master_review.json"
    review = read(review_path) if review_path.exists() else None
    markdown=render_markdown(pack,ruleset,workflows,review)
    (KNOWLEDGE / "MODEL_BUILDING_FIELD_GUIDE.md").write_text(markdown)
    (KNOWLEDGE / "MODEL_BUILDING_FIELD_GUIDE.html").write_text(render_html(markdown))
    return {"guide":str(KNOWLEDGE / "MODEL_BUILDING_FIELD_GUIDE.md"),"rules":len(ruleset),"workflows":len(workflows),"llm_review_included":bool(review)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["build"],default="build",nargs="?");args=parser.parse_args()
    print(json.dumps(build(),indent=2))

if __name__=="__main__": main()
