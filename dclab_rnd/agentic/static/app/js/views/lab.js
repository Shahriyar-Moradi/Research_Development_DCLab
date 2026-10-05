DC.view('lab', {
  init(el) {
    const { $, esc, chip, chips } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chips:([A-Za-z0-9_,-]+)\}/g, (m, ids) => chips(ids.split(','))).replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id));
    const CH = [['adult', 'catboost', 'catboost', 0.9274], ['bank_marketing', 'lightgbm', 'baseline_lightgbm', 0.8029], ['breast_cancer', 'lightgbm', 'fe_ratios', 1.0], ['credit_default', 'ensemble', 'stacking_ensemble', 0.8005], ['german_credit', 'catboost', 'catboost', 0.8112], ['heart_disease', 'hist_gradient_boosting', 'hist_gradient_boosting', 0.9740], ['hyperack', 'ensemble', 'softvote_etbag_lgbmwinner_xgb', 0.9455], ['mushroom', 'catboost', 'catboost', 1.0], ['online_shoppers', 'xgboost', 'tuned_xgboost', 0.7737], ['spambase', 'lightgbm', 'lightgbm', 0.9878], ['wine_quality', 'extra_trees', 'extra_trees', 0.9160]];
    $('#champ-table tbody', el).innerHTML = CH.map(([d, f, r, v]) => `<tr><td><span class="cell-main">${d}</span></td><td>${f}</td><td class="mono small">${r}</td><td class="num mono">${v.toFixed(4)}</td></tr>`).join('');
    const TR = [
      ['tabular-classification', 'Tabular classification', 'active', 'the most mature track'], ['ml-methodology', 'ML methodology (evidence campaigns)', 'active', 'campaigns and rules'],
      ['agentic-ml-copilot', 'Agentic ML copilot', 'active', 'code lives in dclab_rnd/'], ['churn-prediction', 'Churn prediction', 'active', 'Telco'],
      ['llm-fine-tuning', 'LLM fine-tuning', 'active', 'data ready, no training run yet'], ['tabular-foundation-models', 'Tabular foundation models', 'active', 'needs a leakage-safe rerun'],
      ['workflow-model', 'Focused model for ML workflows', 'proposed', 'the policy model'], ['workflow-actions', 'Workflow-sized action units', 'proposed', 'the "cake" idea'],
      ['evaluation-and-trust', 'Evaluation and trust', 'proposed', 'end-to-end value unproven'], ['cross-industry-workflows', 'Workflows beyond data science', 'proposed', 'software bug-fixing first'],
      ['vision-scene-graphs', 'Vision and scene graphs', 'proposed', 'no implementation yet'], ['temporal-gnn', 'Temporal GNNs', 'proposed', 'object interactions over frames'],
      ['driving-maps', 'Driving and road-map graphs', 'proposed', 'safety-critical'], ['computer-vision', 'Computer vision', 'planned', ''], ['graph-neural-networks', 'Graph neural networks', 'planned', ''],
      ['time-series-forecasting', 'Time-series forecasting', 'planned', ''], ['nlp-and-text', 'NLP and text', 'planned', ''], ['anomaly-and-fraud-detection', 'Anomaly and fraud detection', 'planned', ''],
      ['recommender-systems', 'Recommender systems', 'planned', ''], ['causal-inference-and-experimentation', 'Causal inference', 'planned', ''], ['mlops-and-deployment', 'MLOps and deployment', 'planned', ''], ['data-science-foundations', 'Data science foundations', 'planned', ''],
    ];
    const cls = { active: 'ok', proposed: 'proof', planned: 'outline' };
    $('#tracks', el).innerHTML = TR.map(([k, t, s, n]) => `<div class="inset stack tight"><div class="spread"><b class="small">${esc(t)}</b><span class="pill ${cls[s]}">${s}</span></div><span class="mono xs muted">research/${k}</span>${n ? `<span class="xs muted">${esc(n)}</span>` : ''}</div>`).join('');
    $('#lab-new', el).addEventListener('click', () => { DC.reveal($('#cd-h', el)); $('#cd-h', el).focus(); });
    DC.hydrate(el);
  },
});
