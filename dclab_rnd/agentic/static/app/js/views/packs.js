DC.view('packs', {
  init(el) {
    const { $, $$, esc, icon, int } = DC;
    const U = ['Objective solution', 'Source and lineage', 'Boundary (split)', 'Observation', 'Risk audit', 'Option generation', 'Controlled comparison', 'Optimization', 'Reliability challenge', 'Knowledge capture'];
    /* Product documentation: how each pack maps the ten steps. Keys follow dclab_rnd/draft/pack.py (GET /api/packs);
       names, descriptions and maturity come from the server. "engine" says plainly what runs today. */
    const TAB_ENGINE = 'Engine today: a project in this pack runs the same five tabular stages, on a table. The pack-specific steps, tools and validators below are design; they do not run yet.';
    const D = {
      tabular: { f: 'packs.tabular',
        steps: ['Prediction moment, target window, action, unit, costs', 'Snapshot hash, licence, time range', 'By time, group or stratified; holdout sealed', 'Train-only profile and risks', 'Availability matrix, heuristics, severity ablation', 'Feature ladder within a declared tolerance', 'Families on identical folds; declared rule', 'Paired tuning above a declared margin', 'Holdout once; calibration; slices; gate', 'Brief, lessons, examples, trajectory'],
        checks: ['solution signed', 'hash recorded', 'unit and direction declared', 'no target-aware fit outside folds', 'forbidden columns blocked', 'tolerance declared before the run', 'same folds for all', 'margin declared', 'holdout read once', 'lessons reviewed'],
        rules: ['DCLAB-R01 … DCLAB-R22 apply as written'], tools: ['propose_solution', 'run_stage', 'review_code', 'export_notebook'], metrics: ['ROC-AUC, average precision, Brier, ECE', 'Net value at the operating point'],
        engine: 'Runs end to end today: the five stages cover WF-04 to WF-09 on a table (binary, multiclass, regression); the solution and the split cover WF-01 to WF-03.',
        foot: 'The pack the engine was built for; its rules come from the campaigns in evidence/campaigns (see the Research lab).' },
      imbalanced: { f: 'packs.imbalanced',
        steps: ['Decision point; cost of a missed positive against a false alarm', 'Snapshot hash, time range, how late labels arrive', 'Time-ordered; the holdout is the latest period', 'Positive rate, drift and duplicates on training rows', 'Dataset-relative counters and post-event fields', 'Feature ladder on time-ordered folds', 'Families ranked by PR-AUC on identical folds, never by accuracy', 'Tuning kept only above a declared margin', 'Holdout once; PR-AUC with an interval; threshold from costs', 'Brief with the random-ranking baseline'],
        checks: ['costs declared', 'label delay recorded', 'no future row in a fold', 'computed on train only', 'forbidden columns blocked', 'tolerance declared before the run', 'same folds; PR-AUC', 'margin declared', 'holdout read once', 'baseline reported'],
        rules: ['IMB-01 Rank by average precision, not accuracy, when positives are rare', 'IMB-02 The holdout is the latest period'], tools: ['cost_curve', 'threshold_from_costs'], metrics: ['Average precision (PR-AUC) with an interval', 'Recall at the operating threshold'],
        engine: 'Engine today: the tabular stages, on a table. Below 10% positives they rank by average precision, and a declared time column orders the split. Cost curves and a cost-based threshold do not run yet.',
        foot: 'Evidence: card fraud average precision 0.8139 (95% interval 0.7317–0.8847) on the latest 20% of rows, against 0.0013 for random ranking (EXP-055); the elapsed Time counter was blocked (LEAK-credit_card_fraud).' },
      timeseries: { f: 'packs.timeseries',
        steps: ['Horizon, decision time, granularity, cost of under- and over-forecasting', 'Series lineage; late revisions recorded', 'Rolling-origin backtests; never shuffled', 'Seasonality and trend on the training window', 'Future covariates, revised values, same-day counts', 'Lags and rolling stats computed causally', 'Naive, seasonal naive, boosting, others on the same origins', 'Tuning on backtests only', 'Errors by horizon, holidays, regime shifts; block bootstrap', 'Brief with the baseline it beats'],
        checks: ['horizon declared', 'revisions tracked', 'no row from the future in a fold', 'stats from the window only', 'covariates known at decision time', 'lags shifted', 'baselines always included', 'margin declared', 'block bootstrap', 'baseline gap reported'],
        rules: ['TS-01 Always report the naive baseline next to the model', 'TS-02 Covariates must be known at forecast time', 'TS-03 Intervals use block bootstrap'], tools: ['backtest', 'lag_audit', 'seasonal_baseline'], metrics: ['MAE and RMSE against lag and seasonal baselines', 'Interval coverage per horizon'],
        engine: 'Engine today: the tabular regression stages, on a table; a declared time column orders the split. Horizons, rolling-origin backtests and the tools listed do not run yet.',
        foot: 'Evidence: bike demand MAE 680.0 (95% interval 484.0–890.1) against 1,156.4 for the lag-2 baseline (EXP-065); casual and registered were blocked (LEAK-bike_sharing_daily).' },
      text: { f: 'packs.text',
        steps: ['Prediction moment; which text exists at that moment', 'Text source, language, snapshot hash', 'Grouped by item or author so one product is never on both sides', 'Text length, language and vocabulary next to the structured fields', 'Ratings and outcome fields written with or after the text', 'TF-IDF fitted inside folds, joined with the table', 'Text recipes and families on identical folds', 'Tuning above a declared margin', 'Holdout once on unseen groups; cluster bootstrap', 'Brief with the tabular-only baseline'],
        checks: ['text availability declared', 'hash recorded', 'groups disjoint', 'train only', 'post-outcome fields blocked', 'vectorizer fitted in folds', 'same folds for all', 'margin declared', 'holdout read once', 'baseline gap reported'],
        rules: ['TXT-01 Fit the vectorizer inside training folds only', 'TXT-02 A rating written with the review describes the label; it is not a feature'], tools: ['text_profile', 'embedding_ladder'], metrics: ['ROC-AUC against a tabular-only baseline', 'Score by text-length slice'],
        engine: 'Engine today: the tabular stages, on a table; text columns become TF-IDF recipes for binary targets only. Embeddings and the tools listed do not run yet.',
        foot: 'Evidence: clothing reviews ROC-AUC 0.9459 (95% interval 0.9391–0.9525) on unseen clothing items, against 0.5480 for the tabular-only reference (EXP-070); Rating and Positive Feedback Count were blocked (LEAK-ecommerce_clothing_reviews).' },
      vision: { f: 'packs.cv',
        steps: ['Task, classes, operating conditions, error costs', 'Licence, annotators, labelling guide version', 'By scene, camera or drive; near-duplicates grouped', 'Class balance, box sizes, conditions per split', 'Near-duplicate frames across splits; hints in file names', 'Data recipe and augmentation ladder', 'Detector families on identical splits', 'Input size and schedule above a margin', 'Slices: night, rain, occlusion, small objects', 'Failure gallery, lessons, examples'],
        checks: ['conditions listed', 'guide version recorded', 'no drive in two splits', 'computed on train only', 'dedupe hash across splits', 'augmentations declared', 'same split for all', 'margin declared', 'every slice has an interval', 'failures reviewed'],
        rules: ['CV-01 Split by drive or scene, never by frame', 'CV-02 Report every operating condition as its own slice', 'CV-03 Near-duplicate frames count as one'], tools: ['frame_dedupe', 'slice_eval', 'detector_screen', 'failure_gallery'], metrics: ['mAP@0.5 per slice', 'Miss rate at the operating threshold'],
        engine: TAB_ENGINE + ' Images cannot enter the engine.',
        foot: 'Planned in research/computer-vision; no experiment has run.' },
      driving: { f: 'packs.driving',
        steps: ['ODD: weather, light, road type, speed; safety metric', 'Sensor logs, calibration, labelling', 'By trip, route and time; geographic holdout', 'ODD coverage per split', 'Same route or adjacent time in train and test', 'Sensor and fusion choices', 'Detectors on identical trips', 'Latency-aware tuning', 'ODD slices, rare events, latency budget', 'Evidence for a safety case, never an approval'],
        checks: ['ODD signed', 'calibration versions', 'routes disjoint', 'coverage gaps listed', 'route and time leakage', 'sensor set declared', 'identical trips', 'latency budget', 'rare-event recall', 'no deployment claim'],
        rules: ['DRV-01 A benchmark result is never a safety claim', 'DRV-02 Out-of-ODD inputs must be detected, not guessed'], tools: ['odd_coverage', 'route_split', 'latency_probe'], metrics: ['Miss rate by distance and ODD slice', 'End-to-end latency'],
        engine: TAB_ENGINE + ' Sensor logs cannot enter the engine.',
        foot: 'Safety-critical research, proposed in research/driving-maps; no experiment has run. The pack would produce evidence for a safety case; it never approves a vehicle.' },
      maps: { f: 'packs.maps',
        steps: ['Target per segment, horizon, decision time', 'Road graph version, sensor coverage', 'By time; held-out districts', 'Coverage and gaps per segment', 'Future incidents, neighbours\' future values', 'Graph neighbourhood and time lags', 'Seasonal baseline, boosting with graph features, temporal GNN', 'Tuning on backtests', 'Incidents, holidays, unseen segments', 'Lessons per road type'],
        checks: ['graph version pinned', 'coverage map', 'districts disjoint', 'gaps listed', 'neighbour values lagged', 'graph features causal', 'same backtests', 'margin declared', 'unseen-segment slice', 'reviewed'],
        rules: ['MAP-01 Neighbour features must be lagged like the target'], tools: ['graph_snapshot', 'district_split', 'gnn_screen'], metrics: ['MAE per segment and horizon', 'Error during incidents'],
        engine: TAB_ENGINE + ' Road graphs cannot enter the engine.',
        foot: 'Proposed in research/driving-maps and research/temporal-gnn. No experiments yet.' },
      scenegraph: { f: 'packs.scene',
        steps: ['Objects, relations, events; when the system must know', 'Video licence, relation annotation guide', 'By video and scene', 'Relation frequencies', 'Same scene or adjacent clips across splits', 'Detector outputs, tracks, predicted graphs', 'A: video→VLM · B: pairwise classifier · C: temporal GNN · D: video + graph→VLM', 'Tuning on validation videos', 'Rare relations, occlusion; asks for evidence when unsure', 'Lessons and examples'],
        checks: ['event moment declared', 'guide version', 'videos disjoint', 'train only', 'clip adjacency', 'track lineage', 'same videos for A–D', 'margin declared', 'uncertainty triggers evidence', 'reviewed'],
        rules: ['SG-01 When uncertain, gather evidence instead of guessing a relation'], tools: ['track_objects', 'relation_eval', 'vlm_compare'], metrics: ['Relation recall@k', 'Event F1'],
        engine: TAB_ENGINE + ' Video cannot enter the engine.',
        foot: 'Proposed in research/vision-scene-graphs; no implementation or result yet. Goal: a vision-language agent that keeps a temporal scene graph and gathers evidence when unsure.' },
      llm: { f: 'packs.llm',
        steps: ['Task, success metric, refusal behaviour', 'Data licence and provenance', 'By source and document; near-duplicates grouped', 'Lengths, task mix', 'Benchmark contamination; prompt near-duplicates', 'Data recipe ladder', 'Base models and LoRA ranks on the same eval', 'Hyperparameters on dev only', 'Held-out tasks, safety, regressions', 'Model card and lessons'],
        checks: ['metric declared', 'licences', 'sources disjoint', 'train only', 'contamination scan', 'recipe declared', 'same eval', 'dev only', 'held-out tasks', 'card written'],
        rules: ['LLM-01 Hold whole datasets out, not random examples', 'LLM-02 Facts that change stay in retrieval, not weights'], tools: ['build_sft', 'train_lora', 'eval_sft', 'contamination_scan'], metrics: ['Held-out task score', 'Judgment benchmark'],
        engine: TAB_ENGINE + ' Training an LLM is not a notebook stage: the corpus builder, train_lora.py and eval_sft.py are research scripts run by hand.',
        foot: 'Corpus v3, train_lora.py and eval_sft.py exist in research/llm-fine-tuning; the Policy model page shows whether any run exists.' },
      software: { f: 'packs.software',
        steps: ['Issue → what "fixed" means; acceptance tests', 'Repository, commit, issue link', 'By repository and time', 'Reproduce the bug', 'Fix visible in tests? flaky tests?', 'Candidate patches', 'Same test runs for every patch', 'Smallest diff that passes', 'Full regression suite, review', 'Lesson and example'],
        checks: ['acceptance tests named', 'commit pinned', 'repos disjoint', 'failing test first', 'flakiness check', 'patches isolated', 'identical runs', 'diff size', 'full suite green', 'reviewed'],
        rules: ['SW-01 Reproduce before fixing', 'SW-02 Never weaken a test to pass it'], tools: ['reproduce', 'run_tests', 'diff_review'], metrics: ['Tests passing', 'Reviewer acceptance'],
        engine: 'Research only: not a machine-learning pack, so it cannot be picked on Home or in a solution draft, and nothing runs for it.',
        foot: 'Proposed in research/cross-industry-workflows: the same ten steps outside machine learning. No implementation yet.' },
    };
    /* Research packs the page documents but the server does not offer (not in dclab_rnd/draft/pack.py, so never detected or picked). */
    const PAGE_ONLY = [{ key: 'software', name: 'Software bug-fixing', desc: 'The first non-ML pack: fix an issue so the tests pass.', maturity: 'Research', cls: '', icon: 'bug', page_only: true }];
    const NO_DOC = { steps: U.map(() => 'Not mapped yet'), checks: U.map(() => '—'), rules: [], tools: [], metrics: [], engine: TAB_ENGINE, foot: 'This pack has no step map written yet.' };
    const S = this.S = { packs: [], usage: null, current: 'tabular' };
    this.PAGE_ONLY = PAGE_ONLY;
    const doc = k => D[k] || NO_DOC;
    const use = k => (S.usage && S.usage.packs && S.usage.packs[k]) || { drafts: 0, open_drafts: 0, projects: 0, project_list: [] };
    const n = (v, one, many) => `${int(v)} ${v === 1 ? one : (many || one + 's')}`;

    function cards() {
      $('#pack-cards', el).innerHTML = S.packs.map(p => {
        const u = use(p.key), d = doc(p.key);
        const tag = p.page_only ? 'research only' : S.usage ? (u.projects || u.drafts ? [u.projects ? n(u.projects, 'project') : '', u.drafts ? n(u.drafts, 'draft') : ''].filter(Boolean).join(' · ') : 'not used yet') : '…';
        return `<button type="button" class="pack-card" data-pack="${esc(p.key)}"${d.f ? ` data-f="${d.f}"` : ''} aria-pressed="${p.key === S.current}"><span class="pc-top"><span class="pack-ic">${icon(p.icon)}</span><span class="pill ${esc(p.cls)}">${esc(p.maturity)}</span></span><span class="pc-name">${esc(p.name)}</span><span class="pc-desc">${esc(p.desc)}</span><span class="pc-foot"><span class="tag">${esc(tag)}</span></span></button>`;
      }).join('') || '<div class="empty">The server returned no packs.</div>';
    }
    function show(k) {
      const p = S.packs.find(x => x.key === k);
      if (!p) return;
      S.current = k; const d = doc(k), u = use(k);
      $$('[data-pack]', el).forEach(b => b.setAttribute('aria-pressed', String(b.dataset.pack === k)));
      $('#pk-eyebrow', el).textContent = 'Selected pack';
      $('#pk-title', el).textContent = p.name;
      $('#pk-steps-name', el).textContent = 'Pack · ' + p.name + ' · ' + p.maturity;
      $('#pk-steps-sub', el).textContent = k === 'tabular' ? 'Pick a pack on the Overview; this map follows it. It describes the pack\'s design; the project pages show what each run recorded.' : 'Pick a pack on the Overview; this map follows it. For this pack the map is design: the engine runs the tabular stages instead.';
      $('#pk-sub', el).textContent = p.desc;
      $('#pk-maturity', el).textContent = p.maturity; $('#pk-maturity', el).className = 'pill ' + p.cls;
      const list = u.project_list.slice(0, 6).map(x => `<a href="#project" data-open-project="${esc(x.id)}">${esc(x.name)}</a>`).join(', ');
      const usage = p.page_only ? '<b>In use:</b> not offered in projects.' : !S.usage ? 'Loading usage…'
        : `<b>In use:</b> ${n(u.projects, 'project')}${list ? ` (${list}${u.project_list.length > 6 ? ', …' : ''})` : ''} · ${n(u.drafts, 'draft')}${u.drafts ? `, ${int(u.open_drafts)} still open` : ''}.`;
      $('#pk-usage', el).innerHTML = `<div>${usage}</div><div class="${k === 'tabular' ? '' : 'muted'}">${esc(d.engine)}</div>`;
      $('#pk-table tbody', el).innerHTML = U.map((s, i) => `<tr><td><span class="tag">WF-${String(i + 1).padStart(2, '0')}</span> <b class="small">${s}</b></td><td>${esc(d.steps[i])}</td><td class="small muted">${esc(d.checks[i])}</td></tr>`).join('');
      $('#pk-rules', el).innerHTML = d.rules.map(r => `<div>${DC.linkIds(r)}</div>`).join('') || '<div class="muted">None written yet.</div>';
      $('#pk-tools-h', el).textContent = k === 'tabular' ? 'Tools it uses (built)' : 'Tools it would add (not built yet)';
      $('#pk-tools', el).innerHTML = d.tools.map(t => `<span class="chip mono">${esc(t)}</span>`).join('') || '<span class="muted small">None listed.</span>';
      $('#pk-metrics', el).innerHTML = d.metrics.map(m => `<div>${esc(m)}</div>`).join('') || '<div class="muted">None listed.</div>';
      $('#pk-foot', el).innerHTML = DC.linkIds(d.foot);
      DC.hydrate(el);
    }
    this.render = () => {
      const counts = {};
      const served = S.packs.filter(p => !p.page_only), extra = S.packs.length - served.length;
      served.forEach(p => { counts[p.maturity] = (counts[p.maturity] || 0) + 1; });
      $('#pk-n', el).textContent = served.length;
      $('#pk-n-l', el).textContent = 'packs on one engine · ' + Object.entries(counts).map(([m, c]) => `${c} ${m.toLowerCase() === 'ga' ? 'GA' : m.toLowerCase()}`).join(' · ') + (extra ? ` · plus ${extra} research-only` : '');
      if (S.usage) {
        const withPack = S.usage.projects - S.usage.projects_without_pack;
        $('#pk-use', el).textContent = int(withPack);
        $('#pk-use-l', el).textContent = `of ${n(S.usage.projects, 'project')} carry a pack from their draft` + (S.usage.projects_without_pack ? `; ${S.usage.projects_without_pack === 1 ? 'the one without runs' : `the ${int(S.usage.projects_without_pack)} without run`} as tabular` : '') + ` · ${n(S.usage.drafts - S.usage.drafts_without_pack, 'draft')} with a pack`;
      } else { $('#pk-use', el).textContent = '—'; $('#pk-use-l', el).textContent = 'projects with a pack (usage could not be loaded)'; }
      if (!S.packs.some(p => p.key === S.current) && S.packs.length) S.current = S.packs[0].key;
      cards(); show(S.current);
    };
    el.addEventListener('click', e => {
      const open = e.target.closest('[data-open-project]');
      if (open) { DC.state.project = 'p:' + open.dataset.openProject; DC.currentProject.clear(); return; }
      const b = e.target.closest('[data-pack]'); if (b) show(b.dataset.pack);
    });
    $('#pack-builder-btn', el).addEventListener('click', () => DC.modal.open({
      eyebrow: '<span class="eyebrow">Pack builder</span>', title: 'Packs are defined in code today', hideConfirm: true, cancel: 'Close',
      html: `<p>There is no pack builder yet. A pack is an entry in <code>dclab_rnd/draft/pack.py</code> (<code>PACKS</code>: key, name, description, maturity), with keywords that detect it from the problem sentence, and step labels in <code>dclab_rnd/draft/workflow.py</code> (<code>PACK_CHANGES</code>) that change how the solution workflow is drawn.</p>
        <p>Whatever the pack, the engine runs the same five tabular stages today. Pack-specific rules, tools, metrics and validators, as described on this page, are design.</p>
        <p class="muted small">To add a pack: add it to both files, run <code>make rd-check</code>, and it appears here, on Home and in the solution draft.</p>`,
    }));
  },
  async enter() {
    DC.markSample(false);
    const [packs, usage] = await Promise.all([DC.client.listPacks().catch(() => null), DC.client.learnPacks().catch(() => null)]);
    if (packs) this.S.packs = packs.concat(this.PAGE_ONLY.filter(x => !packs.some(p => p.key === x.key)));
    this.S.usage = usage;
    if (!packs) { DC.$('#pack-cards', DC.$('#view-packs')).innerHTML = '<div class="empty">Could not load the packs from the server.</div>'; return; }
    this.render();
  },
});
