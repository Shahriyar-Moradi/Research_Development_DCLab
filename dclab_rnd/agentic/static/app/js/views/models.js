DC.view('models', {
  init(el) {
    const { $, $$, esc, chip, icon, charts, toast } = DC;
    const view = this;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id)).replace(/\$\{icon:([a-z]+)\}/g, (m, n) => icon(n));
    /* ---------- sample mode (no real project open): the term-deposit demo, unchanged ---------- */
    const F = [['logistic_regression', 0.7189, 0.0301, 0.6], ['extra_trees', 0.7288, 0.0295, 2.0], ['hist_gradient_boosting', 0.7123, 0.0147, 1.2], ['lightgbm', 0.7172, 0.0165, 0.8], ['xgboost', 0.7047, 0.0235, 0.8]];
    function rank(k) { return F.map(f => ({ name: f[0], mean: f[1], sd: f[2], t: f[3], score: f[1] - k * f[2] })).sort((a, b) => b.score - a.score || a.t - b.t); }
    function draw() {
      const k = Number($('#kpen', el).value);
      $('#kpen-val', el).textContent = k.toFixed(2);
      const r = rank(k);
      $('#screen-table tbody', el).innerHTML = r.map((f, i) => `<tr class="${i === 0 ? 'selected' : ''}"><td><span class="cell-main">${f.name}</span>${i === 0 ? ' <span class="pill accent">selected</span>' : ''}</td><td class="num mono">${f.mean.toFixed(4)}</td><td class="num mono">±${f.sd.toFixed(4)}</td><td class="num mono">${f.score.toFixed(4)}</td><td class="num mono">${f.t.toFixed(1)} s</td><td>${i + 1}</td></tr>`).join('');
      $('#screen-chart', el).innerHTML = charts.barsH(F.map(f => ({ label: f[0], value: f[1], lo: f[1] - f[2], hi: f[1] + f[2], valueText: `${f[1].toFixed(4)} ± ${f[2].toFixed(4)}`, cls: f[0] === r[0].name ? '' : 'soft', strong: f[0] === r[0].name })),
        { width: 640, labelW: 160, valW: 128, min: 0.66, max: 0.77, rowH: 30, ticks: [0.66, 0.69, 0.72, 0.75], tickFmt: v => v.toFixed(2), aria: 'Mean ROC-AUC with fold spread' });
      const declared = Math.abs(k - 0.25) < 1e-9;
      $('#rule-out', el).innerHTML = `<div class="callout ${declared ? 'ok' : 'warn'}"><span class="ic">${icon(declared ? 'check' : 'alert')}</span><span>${declared
        ? `With the declared k = 0.25 the rule selects <b>extra_trees</b> (0.7288 − 0.25 × 0.0295 = ${(0.7288 - 0.25 * 0.0295).toFixed(4)}). The runner-up, lightgbm, is kept for the reliability step.`
        : `With k = ${k.toFixed(2)} the rule would select <b>${r[0].name}</b>${r[0].name !== 'extra_trees' ? ', a different model' : ''}. The penalty is part of the protocol; changing it after the screen is a protocol change with a written reason.`}</span></div>`;
      DC.hydrate(el);
    }
    let real = null;  // the open real project's view model; null in sample mode
    $('#kpen', el).addEventListener('input', () => (real ? drawRule(real) : draw()));
    draw();
    $('#fold-chart', el).innerHTML = charts.folds(F.map(f => ({ label: f[0], folds: [f[1] - f[2], f[1], f[1] + f[2]], mean: f[1], strong: f[0] === 'extra_trees' })), { width: 640, labelW: 160, min: 0.66, max: 0.78, aria: 'Fold scores per family' });
    $('#tune-chart', el).innerHTML = charts.barsH([
      { label: 'baseline', value: 0.7288, valueText: '0.7288', cls: 'muted' }, { label: 'C01', value: 0.7430, valueText: '0.7430', cls: 'soft' },
      { label: 'C02 · kept', value: 0.7523, valueText: '0.7523', strong: true }, { label: 'C03', value: 0.7383, valueText: '0.7383', cls: 'soft' },
    ], { width: 380, labelW: 84, valW: 56, min: 0.70, max: 0.76, rowH: 28, ref: 0.7338, refLabel: 'baseline + margin', ticks: [0.70, 0.72, 0.74, 0.76], tickFmt: v => v.toFixed(2), aria: 'Tuning candidates' });
    $('#family-chart', el).innerHTML = charts.barsH([
      { label: 'lightgbm', value: 2.55, valueText: '2.55 · 3 wins' }, { label: 'xgboost', value: 3.55, valueText: '3.55 · 1 win' }, { label: 'ensemble', value: 3.64, valueText: '3.64 · 2 wins' },
      { label: 'hist_gb', value: 4.36, valueText: '4.36 · 1 win' }, { label: 'extra_trees', value: 4.55, valueText: '4.55 · 1 win' }, { label: 'logistic', value: 7.64, valueText: '7.64 · 0 wins', cls: 'muted' },
    ], { width: 340, labelW: 82, valW: 92, min: 0, max: 8, rowH: 24, ticks: [0, 2, 4, 6, 8], tickFmt: v => v, aria: 'Mean rank across datasets' });
    const sampleOverride = () => DC.modal.open({
      eyebrow: '<span class="eyebrow">Override the rule</span>', title: 'Choose a different model', confirm: 'Log the override',
      html: `<p>The rule picked <b>extra_trees</b>. You can choose another candidate; the log records your reason, and the brief says the choice was made by a person.</p>
        <div class="field"><label for="ov-model">Model</label><select id="ov-model"><option>lightgbm · 0.7172 ± 0.0165 · faster scoring</option><option>logistic_regression · 0.7189 ± 0.0301 · easiest to explain</option></select></div>
        <div class="field"><label for="ov-why">Reason (required)</label><textarea id="ov-why">The call centre wants reasons per client; a linear model is easier to explain and is within the fold spread of the leader.</textarea></div>`,
      onConfirm: m => { if (!$('#ov-why', m).value.trim()) { toast('Write a reason first.', { ok: false }); return false; } toast('Override logged. The holdout is already used, so the new model needs fresh data or a new split for its estimate. The brief will say so.'); },
    });
    $('#override-btn', el).addEventListener('click', () => (real ? overrideReal(real) : sampleOverride()));
    // every element real mode rewrites, as the sample left it
    const SLOTS = $$('[data-slot]', el).map(n => [n, n.innerHTML, n.className]);
    this.restoreSample = () => {
      real = null;
      SLOTS.forEach(([n, html, cls]) => { n.innerHTML = html; n.className = cls; });
      $('#kpen', el).disabled = false; $('#kpen', el).value = '0.25';
      $('#override-btn', el).disabled = false; $('#override-btn', el).title = '';
      $('#mo-improve-go', el).hidden = true;
      const n = $('.ptab[data-ptab="improvements"] .n', el); if (n) n.textContent = '0';
      draw();
    };

    /* ---------- real mode: the open project's models and final records ---------- */
    const LABEL = { roc_auc: 'ROC-AUC', average_precision: 'average precision', macro_f1: 'macro-F1', mae: 'MAE', rmse: 'RMSE', r2: 'R²', accuracy: 'accuracy', balanced_accuracy: 'balanced accuracy', log_loss: 'log loss', brier: 'Brier score' };
    const LOWER = new Set(['mae', 'rmse', 'mape', 'log_loss', 'brier']);
    const UNIT = new Set(['roc_auc', 'average_precision', 'macro_f1', 'accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']);
    // the other metrics every fold already measured, shown beside the one the rule ranks by (the owner, 2026-10-20); a binary
    // model's precision, recall and F1 are read at the 0.5 cut: the operating threshold is chosen later, in the final stage
    const EXTRA = { binary: [['precision', 'Precision @0.5'], ['recall', 'Recall @0.5'], ['f1', 'F1 @0.5'], ['average_precision', 'Avg precision'], ['roc_auc', 'ROC-AUC']],
                    multiclass: [['macro_f1', 'Macro-F1'], ['balanced_accuracy', 'Balanced acc.'], ['accuracy', 'Accuracy']],
                    regression: [['mae', 'MAE'], ['rmse', 'RMSE'], ['r2', 'R²']] };
    const WORDS = ['No', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten'];
    const words = n => WORDS[n] || String(n);
    const ok = v => v != null && isFinite(v);
    const num = (v, d) => { if (!ok(v)) return '—'; const a = Math.abs(v); const dd = d != null ? d : a >= 1000 ? 0 : a >= 100 ? 1 : a >= 10 ? 2 : 4; return Number(v).toLocaleString('en-US', { minimumFractionDigits: dd, maximumFractionDigits: dd }); };
    const signed = v => (ok(v) ? (v >= 0 ? '+' : '−') + num(Math.abs(v)) : '—');
    const secs = v => (ok(v) ? (v < 0.1 ? v.toFixed(3) : v < 10 ? v.toFixed(1) : Math.round(v)) + ' s' : '—');
    const empty = text => `<div class="empty">${esc(text)}</div>`;
    const evChips = (rec, n = 2, prefer = []) => {
      const rank = id => (prefer.includes(id) ? prefer.indexOf(id) : prefer.length);
      return [...new Set(((rec && rec.notes) || []).flatMap(x => x.proof || []))].filter(id => DC.REC[id]).map((id, i) => [id, rank(id), i]).sort((a, b) => a[1] - b[1] || a[2] - b[2]).slice(0, n).map(x => chip(x[0])).join('');
    };
    /* An axis that fits the values: a nice step, rounded ends, ticks and their format. */
    function scale(vals, unit, pad = 0.2, parts = 4) {
      const v = vals.filter(ok);
      let lo = Math.min(...v), hi = Math.max(...v);
      const span = hi - lo || Math.abs(hi) * 0.1 || 0.01;
      lo -= span * pad; hi += span * pad;
      if (unit) { lo = Math.max(0, lo); hi = Math.min(1, hi); }
      const raw = (hi - lo) / parts, mag = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / mag;
      const step = (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
      const min = +(Math.floor(lo / step + 1e-9) * step).toFixed(10), max = +(Math.ceil(hi / step - 1e-9) * step).toFixed(10);
      const ticks = []; for (let t = min; t <= max + step * 1e-6; t += step) ticks.push(+t.toFixed(10));
      const dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
      return { min, max, ticks, fmt: t => t.toFixed(dec) };
    }
    function stageNote(p, stage, what) {
      const st = ((p.stages || {})[stage] || {}).status || 'pending', running = p.running && (p.running.stage || p.running);
      if (running === stage || st === 'running' || st === 'queued') return `The ${stage} stage is running now. This page fills in when it finishes.`;
      if (st === 'failed') return `The ${stage} stage failed. Run it again from the Workflow page${what ? ' to see ' + what : ''}.`;
      if (stage === 'final' && Number(p.holdout_uses) > 0) return 'The earlier holdout result was cleared by a later change; running the final stage again reuses the holdout and needs a written reason.';
      return `Not measured for this project yet: run the ${stage} stage${what ? ' to see ' + what : ''}.`;
    }
    function build(p) {
      const R = p.records || {}, m = R.models, f = R.final, d = R.data, fx = R.features;
      const ev = m && m.evidence, fe = f && f.evidence;
      const metric = (m && m.primary_metric) || (f && f.primary_metric) || (p.solution || {}).metric || 'roc_auc';
      const rule = (ev && ev.selection_rule) || '';
      const km = rule.match(/([0-9]*\.?[0-9]+)\s*×\s*fold std/);
      const results = (ev && ev.model_results) || [];
      const measured = Object.assign({}, ...results.map(x => x.metrics || {}));
      const task = 'f1' in measured && 'precision' in measured ? 'binary' : 'macro_f1' in measured ? 'multiclass' : 'mae' in measured ? 'regression' : '';
      const cols = (EXTRA[task] || []).filter(([k]) => k !== metric && k in measured).slice(0, 4);
      const rows = ((ev && ev.ranking) || []).map(r => {
        const mr = results.find(x => x.model === r.model) || {};
        const fit = mr.metrics && mr.metrics.fit_seconds;
        return { name: r.model, mean: r.mean, sd: r.std, t: r.elapsed_seconds, adjusted: r.adjusted_score, feats: r.feature_count_mean,
          folds: (mr.fold_metrics || []).map(x => x[metric]).filter(ok), fit: fit ? fit.mean : null,
          extra: Object.fromEntries(cols.map(([k]) => [k, mr.metrics && mr.metrics[k] ? mr.metrics[k].mean : null])) };
      });
      const dec = (m && m.decision) || {};
      const selected = (ev && ev.selected_model) || dec.selected || (rows[0] || {}).name;
      const chosen = dec.chosen || selected;
      const cv = (ev && ev.cv_protocol) || (fx && fx.evidence && fx.evidence.cv_protocol) || (d && d.evidence && d.evidence.cv_protocol) || '';
      const nf = (results[0] && results[0].folds) || Number((cv.match(/\((\d+)/) || [])[1]) || null;
      const kind = /Group/.test(cv) ? 'group' : /TimeSeries|time/i.test(cv) ? 'time-ordered' : /Stratified/.test(cv) ? 'stratified' : '';
      return { p, m, f, ev, fe, metric, label: LABEL[metric] || metric, unit: UNIT.has(metric), rule, cols, task,
        k: km ? Number(km[1]) : 0.25, higher: rule ? !/lower is better/i.test(rule) : !LOWER.has(metric),
        rows, selected, chosen, overridden: !!chosen && chosen !== selected, nf, kind, recipe: (ev && ev.feature_recipe) || (fe && fe.feature_recipe) || (fx && ((fx.decision || {}).chosen || (fx.evidence || {}).selected_recipe)) || '',
        trainRows: (d && d.evidence && d.evidence.train_rows) || (fe && fe.train_rows) || null, ...improvements(p, metric) };
    }
    /* The Experimenter's runs (the owner, 2026-11-03: keep every improvement, show it here and in the models table): every run kept,
       oldest first (a project improved before runs were kept has its one latest run); in the table, each way's latest result. */
    /* A run on the project's current setup (its model, features, folds and metric): its ways compare on the folds the table shows.
       An earlier setup's run stays in the tab, marked, never in the table. */
    function current(r, p, metric) {
      const R = p.records || {}, ev = (R.models || {}).evidence || {}, fx = R.features || {};
      const family = ((R.models || {}).decision || {}).chosen || ev.selected_model;
      const recipe = ev.feature_recipe || (fx.decision || {}).chosen || (fx.evidence || {}).selected_recipe;
      const folds = ((ev.model_results || [])[0] || {}).folds;
      const c = r.current || {};
      return (!r.metric || r.metric === metric) && c.family === family && c.recipe === recipe && (!folds || !r.folds || r.folds === folds);
    }
    function improvements(p, metric) {
      let runs = (p.improvement_runs || []).filter(r => r && Array.isArray(r.candidates));
      if (!runs.length && p.improvements && Array.isArray(p.improvements.candidates)) runs = [{ ...p.improvements, run: p.improvements.run || 1 }];
      runs = runs.map((r, i) => ({ ...r, run: r.run || i + 1 }));
      const latest = new Map();
      runs.filter(r => current(r, p, metric)).forEach(r => (r.candidates || []).forEach(c => latest.set(`${c.id}|${c.label}`, { ...c, run: r.run, nfolds: r.folds, adopted: r.adopted === c.id, current: r.current || {} })));
      runs.forEach(r => { r.onCurrent = current(r, p, metric); });
      return { runs, imp: [...latest.values()].sort((a, b) => (b.gain == null ? -Infinity : b.gain) - (a.gain == null ? -Infinity : a.gain)) };
    }
    const rankK = (V, k) => V.rows.map(r => ({ ...r, score: V.higher ? r.mean - k * r.sd : r.mean + k * r.sd }))
      .sort((a, b) => (V.higher ? b.score - a.score : a.score - b.score) || a.t - b.t);
    const shortId = (V, id) => String(id || '').replace((V.fe ? V.fe.model : '') + '_', '');

    function paint(V) {
      const { p, ev, fe } = V;
      const pillTitle = !ev ? '<span class="pill outline">not screened yet</span>'
        : `<span class="pill ${V.overridden ? 'warn' : 'ok'}">${esc(V.chosen)} · ${fe ? (fe.tuning_decision && fe.tuning_decision.accepted ? esc(shortId(V, fe.tuning_decision.selected_config)) + ' locked' : 'defaults locked') : 'not tuned yet'}</span>`;
      $('#mo-eyebrow', el).textContent = `${p.name} · WF-07 and WF-08`;
      $('#mo-title', el).innerHTML = `Models ${pillTitle}`;
      $('#mo-lede', el).textContent = `${ev ? words(V.rows.length) + ' families' : 'Several model families'} on identical folds, ranked by a rule declared before the run. Tuning stays only if it clears a margin set in advance.`;
      // overview cards
      const td = fe && fe.tuning_decision;
      const runner = rankK(V, V.k).find(r => r.name !== V.chosen);
      const big = name => (String(name).length > 14 ? ' data-style="font-size:22px;overflow-wrap:anywhere"' : '');
      const chosenRow = V.rows.find(r => r.name === V.chosen) || {};
      $('#mo-stats', el).innerHTML = ev ? `
        <div class="stat"><span class="v"${big(V.chosen)}>${esc(V.chosen)}</span><span class="l">${V.overridden ? `chosen by you · the declared rule picked ${esc(V.selected)}` : `selected by the declared rule${runner ? ' · runner-up ' + esc(runner.name) : ''}`}</span></div>
        <div class="stat"><span class="v">${num(chosenRow.mean)} <small>± ${num(chosenRow.sd)}</small></span><span class="l">mean CV ${esc(V.label)} on ${V.nf || '?'} folds${V.trainRows ? ' · ' + Number(V.trainRows).toLocaleString('en-US') + ' training rows' : ''}</span></div>
        ${td ? `<div class="stat"><span class="v ${td.accepted ? 'ok' : ''}">${signed(td.gain)}</span><span class="l">gain from tuning candidate ${esc(shortId(V, td.best_candidate))} · declared margin ${num(td.required_gain)}${td.accepted ? '' : ' · not enough, defaults kept'}</span></div>`
          : `<div class="stat"><span class="v">—</span><span class="l">gain from tuning · ${esc(stageNote(p, 'final').replace(/\.$/, '').replace(/^./, c => c.toLowerCase()))}</span></div>`}
        <div class="stat"><span class="v">${V.rows.length}</span><span class="l">model families screened on the same ${V.nf || ''} folds</span></div>`
        : `<div class="stat"><span class="v">—</span><span class="l">model selected by the declared rule · ${esc(stageNote(p, 'models').replace(/^./, c => c.toLowerCase()))}</span></div>
        <div class="stat"><span class="v">—</span><span class="l">mean CV ${esc(V.label)}${V.nf ? ' on ' + V.nf + ' folds' : ''}</span></div>
        <div class="stat"><span class="v">—</span><span class="l">gain from tuning · after the screen</span></div>
        <div class="stat"><span class="v">—</span><span class="l">model families screened${V.nf ? ' on the same ' + V.nf + ' folds' : ''}</span></div>`;
      // screen
      const feats = V.rows.length ? Math.round(V.rows[0].feats) : null;
      $('#mo-screen-title', el).innerHTML = `<h2>Screen on the ${esc(V.recipe || 'chosen')} recipe</h2><span class="sub">${ev ? `${V.nf} ${V.kind} folds · ${feats != null ? feats + ' features · ' : ''}ranked by mean ${V.higher ? '−' : '+'} ${V.k} × fold spread, then runtime` : `${V.nf ? V.nf + ' ' + V.kind + ' folds · ' : ''}ranked by a rule declared when the models stage runs`}</span>`;
      $('#mo-screen-ev', el).innerHTML = evChips(V.m);
      $('#mo-screen-head', el).innerHTML = `<tr><th>Family</th><th class="num">Mean ${esc(V.label)}</th><th class="num">Fold spread</th>`
        + `<th class="num" id="mo-adj-th">Mean ${V.higher ? '−' : '+'} k×spread</th>${V.cols.map(([, l]) => `<th class="num">${esc(l)}</th>`).join('')}`
        + `<th class="num" id="mo-fit-th">Fit time (${V.nf || 'all'} folds)</th><th>Rank</th></tr>`;
      const note = $('#mo-screen-note', el);
      note.hidden = !(ev && V.cols.length);
      note.textContent = !V.cols.length ? '' : `The other columns are each family's mean over the same folds; the rank uses ${V.label} alone.`
        + (V.task === 'binary' ? ' Precision, recall and F1 count a score of 0.5 or more as a case: the operating threshold is chosen later, on the training rows (the final stage), so they can look low for a rare outcome.' : '');
      $('#kpen', el).value = String(V.k); $('#kpen', el).disabled = !ev;
      $('#override-btn', el).disabled = !ev; $('#override-btn', el).title = ev ? '' : 'Run the models stage first';
      $('#mo-rule-sub', el).textContent = ev ? `Declared in the protocol before the screen ran: ${V.rule}` : 'The rule is declared when the models stage runs.';
      drawRule(V);
      // folds
      $('#mo-fold-sub', el).textContent = `The same ${V.nf ? words(V.nf).toLowerCase() : ''} folds for every family, so differences are paired.`;
      $('#mo-fold-pill', el).textContent = 'stored fold scores';
      const withFolds = V.rows.filter(r => r.folds.length);
      if (withFolds.length) {
        const sc = scale(withFolds.flatMap(r => r.folds), V.unit, 0.1);
        $('#fold-chart', el).innerHTML = charts.folds(withFolds.map(r => ({ label: r.name, folds: r.folds, mean: r.mean, strong: r.name === V.chosen })), { width: 640, labelW: 160, min: sc.min, max: sc.max, aria: `Fold ${V.label} per family` });
        $('#mo-fold-note', el).textContent = `Each dot is one fold's ${V.label} from the screen; bars mark the means.${V.higher ? '' : ' Lower is better.'}`;
      } else { $('#fold-chart', el).innerHTML = empty(stageNote(p, 'models', 'the fold scores')); $('#mo-fold-note', el).textContent = ''; }
      paintTuning(V);
      paintImprovements(V);
      // cost to run
      $('#mo-cost-title', el).innerHTML = `<h3>Cost to run</h3><span class="sub">Measured during the screen: the whole ${V.nf || ''}-fold run and the mean fit per fold</span>`;
      $('#mo-cost-table', el).innerHTML = V.rows.length ? `<table class="data compact"><thead><tr><th>Family</th><th class="num">Run (${V.nf} folds)</th><th class="num">Fit / fold</th></tr></thead><tbody>${V.rows.map(r =>
        `<tr class="${r.name === V.chosen ? 'selected' : ''}"><td>${esc(r.name)}</td><td class="num mono">${secs(r.t)}</td><td class="num mono">${secs(r.fit)}</td></tr>`).join('')}</tbody></table>` : empty(stageNote(p, 'models'));
      $('#mo-cost-foot', el).textContent = V.rows.length ? 'Scoring time is not measured separately yet. Runtime only breaks ties in the declared rule.' : '';
      DC.hydrate(el);
    }
    function drawRule(V) {
      const k = Number($('#kpen', el).value);
      $('#kpen-val', el).textContent = k.toFixed(2);
      if (!V.ev) {
        $('#screen-table tbody', el).innerHTML = `<tr><td colspan="${6 + V.cols.length}">${empty(stageNote(V.p, 'models'))}</td></tr>`;
        $('#screen-chart', el).innerHTML = '';
        $('#rule-out', el).innerHTML = empty(stageNote(V.p, 'models', 'how the rule ranks the families'));
        return;
      }
      const r = rankK(V, k), top = r[0], sign = V.higher ? '−' : '+';
      $('#screen-table tbody', el).innerHTML = r.map((f, i) => `<tr class="${i === 0 ? 'selected' : ''}"><td><span class="cell-main">${esc(f.name)}</span>${i === 0 ? ' <span class="pill accent">selected</span>' : ''}${V.overridden && f.name === V.chosen ? ' <span class="pill warn">chosen by you</span>' : ''}</td><td class="num mono">${num(f.mean)}</td><td class="num mono">±${num(f.sd)}</td><td class="num mono">${num(f.score)}</td>${V.cols.map(([k]) => `<td class="num mono">${num(f.extra[k])}</td>`).join('')}<td class="num mono">${secs(f.t)}</td><td>${i + 1}</td></tr>`).join('') + improveRows(V);
      const sc = scale(V.rows.flatMap(f => [f.mean - f.sd, f.mean + f.sd]), V.unit);
      $('#screen-chart', el).innerHTML = charts.barsH(V.rows.map(f => ({ label: f.name, value: f.mean, lo: f.mean - f.sd, hi: f.mean + f.sd, valueText: `${num(f.mean)} ± ${num(f.sd)}`, cls: f.name === top.name ? '' : 'soft', strong: f.name === top.name })),
        { width: 640, labelW: 160, valW: 128, min: sc.min, max: sc.max, rowH: 30, ticks: sc.ticks, tickFmt: sc.fmt, aria: `Mean ${V.label} with fold spread` });
      const declared = Math.abs(k - V.k) < 1e-9;
      const runner = r[1];
      const gap = (V.m.claims || []).find(c => /gap between leader and runner-up/i.test(c.statement || ''));
      $('#rule-out', el).innerHTML = `<div class="callout ${declared ? 'ok' : 'warn'}"><span class="ic">${icon(declared ? 'check' : 'alert')}</span><span>${declared
        ? `With the declared k = ${V.k} the rule selects <b>${esc(top.name)}</b> (${num(top.mean)} ${sign} ${V.k} × ${num(top.sd)} = ${num(top.score)}).${runner ? ` The runner-up is ${esc(runner.name)} at ${num(runner.score)}.` : ''}${V.overridden ? ` You chose <b>${esc(V.chosen)}</b> instead; the brief says a person made that choice.` : ''}`
        : `With k = ${k.toFixed(2)} the rule would select <b>${esc(top.name)}</b>${top.name !== V.selected ? ', a different model' : ''}. The penalty is part of the protocol; changing it after the screen is a protocol change with a written reason.`}</span></div>${declared && gap ? `<p class="xs muted">${esc(gap.statement)}</p>` : ''}`;
      DC.hydrate($('#rule-out', el));
    }
    /* Each way the Experimenter tried, its latest result, under the ranked families: the same folds, never ranked by the declared rule. */
    function improveRows(V) {
      if (!V.imp.length) return '';
      const span = 6 + V.cols.length;
      return `<tr class="mo-imp-head"><td colspan="${span}">From the Experimenter's improvement runs · the same folds, scored against the model each run improved, not ranked by the declared rule · <button type="button" class="link-btn small" data-mo-runs>see every run</button></td></tr>`
        + V.imp.map(c => `<tr class="mo-imp${c.clears ? ' clears' : ''}"><td><span class="cell-main">${esc(c.label)}</span><div class="mo-tags"><span class="pill proof">improvement · run ${esc(String(c.run))}</span><span class="mo-verdict ${c.adopted || c.clears ? 'ok' : ''}">${c.adopted ? 'adopted' : c.clears ? 'clears the margin' : 'within the noise'}</span></div>`
          + `<div class="cell-sub">${esc(c.aspect)} · ${signed(c.gain)} vs ${esc(c.current.family || 'the model')}${c.folds_better != null ? ` · ${c.folds_better}/${c.nfolds} folds better` : ''}</div></td>`
          + `<td class="num mono">${num(c.mean)}</td><td class="num mono">${ok(c.std) ? '±' + num(c.std) : '—'}</td><td class="num mono">—</td>${V.cols.map(() => '<td class="num mono">—</td>').join('')}`
          + `<td class="num mono">${secs(c.seconds)}</td><td><span class="faint">—</span></td></tr>`).join('');
    }
    /* The Improvements tab: every run, newest first, each way with what can be done with it here. */
    function paintImprovements(V) {
      const { p } = V, box = $('#mo-improve', el), go = $('#mo-improve-go', el);
      const n = $('.ptab[data-ptab="improvements"] .n', el); if (n) n.textContent = String(V.runs.length);
      const refused = (p.activity || []).filter(a => a.kind === 'improve_run' && a.payload && a.payload.state === 'refused').pop();
      if (view.waitImprove && refused && refused.at >= view.waitImprove.since) { toast(refused.payload.why, { ok: false, ms: 8000 }); view.waitImprove = null; }
      if (view.waitImprove && V.runs.length > view.waitImprove.count) view.waitImprove = null;  // the new run is here
      const running = p.running === 'improve' || (p.running && p.running.stage === 'improve') || !!view.waitImprove;
      go.hidden = !V.fe; go.disabled = !!running; go.textContent = running ? 'Improving…' : V.runs.length ? 'Try again' : 'Try ways to improve';
      if (!V.runs.length) {
        box.innerHTML = `<div class="empty">${V.fe ? 'No improvement run yet. Ask on Home "improve the model for me", or press Try ways to improve: every way is tried on the training folds only, and every run is kept here.' : esc(stageNote(p, 'final', 'a model to improve'))}</div>`;
        return;
      }
      const options = new Set(((V.m || {}).decision || {}).options ? V.m.decision.options.map(o => o.id) : []);
      const last = V.runs[V.runs.length - 1];
      box.innerHTML = V.runs.slice().reverse().map(r => {
        const cur = r.current || {}, cands = r.candidates || [], clear = cands.filter(c => c.clears).length, isLast = r === last;
        const use = c => {
          if (c.id === r.adopted) return '<span class="pill ok">adopted</span>';
          if (!c.clears) return '';
          if (c.adoptable) {
            if (!isLast || !r.onCurrent) return `<span class="xs muted">${r.onCurrent ? 'from an earlier run' : 'for an earlier model'}: run again to adopt</span>`;
            if (running) return '<span class="xs muted">after the run going on</span>';
            return r.adopted ? '' : `<button type="button" class="btn sm primary" data-mo-adopt="${esc(c.id)}">Adopt</button>`;
          }
          if (c.id !== 'blend' && c.family && c.family !== cur.family && c.family !== V.chosen && options.has(c.family)) return `<button type="button" class="btn sm" data-mo-choose="${esc(c.family)}">Choose ${esc(c.family)}</button>`;
          if (c.family === V.chosen && c.family !== cur.family) return '<span class="xs muted">now the chosen model</span>';
          return '<span class="xs muted">cannot be adopted yet</span>';
        };
        const rows = cands.map(c => `<tr><td><b>${esc(c.label)}</b><div class="cell-sub">${esc(c.aspect)}</div></td><td class="num mono">${num(c.mean)}</td><td class="num mono">${signed(c.gain)}</td>`
          + `<td class="num mono">${c.folds_better == null ? '—' : c.folds_better + ' / ' + r.folds}</td><td>${c.clears ? '<span class="pill ok">clears the margin</span>' : '<span class="pill outline">within the noise</span>'}</td><td>${use(c)}</td></tr>`).join('');
        const when = r.ran_at ? new Date(r.ran_at).toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : '';
        return `<div class="inset stack tight mo-run"><div class="spread"><div><b>Run ${esc(String(r.run))}</b>${when ? ` <span class="muted small">· ${esc(when)}</span>` : ''}${isLast ? ' <span class="pill outline">latest</span>' : ''}${r.onCurrent ? '' : ' <span class="pill warn">earlier setup · not in the table</span>'}</div>`
          + `<span class="small muted">${cands.length} tried · ${clear} cleared the margin${r.seconds != null ? ' · ' + num(r.seconds, 0) + ' s' : ''}</span></div>`
          + `<p class="small">Improving <b>${esc(cur.family || '?')}</b> on the <code>${esc(cur.recipe || '?')}</code> features: ${num(cur.mean)} ± ${num(cur.std)} ${esc(LABEL[r.metric] || r.metric || '')} on ${r.folds} training folds. Margin set in advance: ${num(r.margin)}.</p>`
          + `<div class="table-wrap"><table class="data compact"><thead><tr><th>Way</th><th class="num">Score on the folds</th><th class="num">Gain</th><th class="num">Folds better</th><th>Verdict</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`
          + ((r.skipped || []).length ? `<div class="xs muted">Not run: ${esc(r.skipped.join('; '))}</div>` : '') + '</div>';
      }).join('') + `<p class="xs muted">${esc(last.note || '')}</p>`;
    }
    el.addEventListener('click', async e => {
      if (!real) return;
      const V = real;
      const go = e.target.closest('#mo-improve-go');
      if (go) {
        go.disabled = true;
        try {
          view.waitImprove = { count: V.runs.length, since: new Date(Date.now() - 5000).toISOString(), until: Date.now() + 20 * 60 * 1000 };
          await DC.client.improveProject(V.p.id);
          toast('The Experimenter is trying ways to improve the model, on the training folds only. The new run appears here when it is done.', { ms: 6000 });
          DC.currentProject.clear(); view.enter(el);
        }
        catch (err) { view.waitImprove = null; go.disabled = false; toast(err.message, { ok: false }); }
        return;
      }
      const a = e.target.closest('[data-mo-adopt]');
      if (a) {
        a.disabled = true;
        try { const got = await DC.client.adoptImprovement(V.p.id, { body: { candidate: a.dataset.moAdopt } }); toast(got.next, { ms: 9000 }); DC.currentProject.clear(); view.enter(el); }
        catch (err) { a.disabled = false; toast(err.message, { ok: false }); }
        return;
      }
      if (e.target.closest('[data-mo-runs]')) { DC.selectPane(el, 'improvements'); return; }
      const c = e.target.closest('[data-mo-choose]');
      if (c) overrideReal(V, c.dataset.moChoose);
    });
    function paintTuning(V) {
      const { p, fe } = V;
      const td = fe && fe.tuning_decision, configs = (fe && fe.configuration_results) || [];
      if (!td || !configs.length) {
        const note = V.ev ? stageNote(p, 'final', 'the tuning candidates') : stageNote(p, 'models');
        $('#mo-tune-sub', el).textContent = 'Explicit candidates for the chosen model on the training folds. A candidate replaces the baseline only if it gains at least the declared margin.';
        $('#mo-tune-ev', el).innerHTML = chip('DCLAB-R15');
        $('#tune-chart', el).innerHTML = empty(note);
        $('#mo-tune-side', el).innerHTML = '<p class="muted">The final stage tunes on the training folds first, then opens the holdout once.</p>';
        return;
      }
      const m = V.metric, mean = c => c && c.metrics && c.metrics[m] ? c.metrics[m].mean : null;
      const base = configs.find(c => /_baseline$/.test(c.config_id)) || configs[0];
      const best = configs.find(c => c.config_id === td.best_candidate);
      const kept = td.selected_config;
      const cands = (fe.optimization_search_space || []);
      $('#mo-tune-sub', el).textContent = `${words(cands.length)} explicit candidate${cands.length === 1 ? '' : 's'} for ${fe.model} on the training folds. A candidate replaces the baseline only if it gains at least the declared margin.`;
      $('#mo-tune-ev', el).innerHTML = evChips(V.f, 1, ['DCLAB-R15', 'DCLAB-R16']);
      const ref = mean(base) + (V.higher ? 1 : -1) * td.required_gain;
      const items = configs.map(c => {
        const id = c.config_id === base.config_id ? 'baseline' : shortId(V, c.config_id), isKept = c.config_id === kept;
        return { label: id + (isKept ? ' · kept' : ''), value: mean(c), valueText: num(mean(c)), strong: isKept, cls: isKept ? '' : c === base ? 'muted' : 'soft' };
      });
      const sc = scale([...items.map(i => i.value), ref], V.unit, 0.3, 3);
      $('#tune-chart', el).innerHTML = charts.barsH(items, { width: 380, labelW: 100, valW: 56, min: sc.min, max: sc.max, rowH: 28, ref, refLabel: V.higher ? '+ margin' : '− margin', ticks: sc.ticks, tickFmt: sc.fmt, aria: 'Tuning candidates' });
      const bf = (base.fold_metrics || []).map(x => x[m]), cf = ((best || {}).fold_metrics || []).map(x => x[m]);
      const diffs = cf.map((v, i) => (V.higher ? v - bf[i] : bf[i] - v)).filter(ok);
      const params = cands.map(c => `${shortId(V, c.candidate_id)}: ${Object.entries(c.params || {}).map(([k, v]) => `${k.replace(/^model__/, '')}=${v}`).join(', ') || 'defaults'}`).join(' · ');
      $('#mo-tune-side', el).innerHTML = `
        <div class="inset stack tight">
          <div class="spread"><span>Baseline (defaults)</span><b class="mono">${num(mean(base))}</b></div>
          <div class="spread"><span>Best candidate ${esc(shortId(V, td.best_candidate))}</span><b class="mono">${num(mean(best))}</b></div>
          <div class="spread"><span>Gain</span><b class="mono"${td.accepted ? ' data-style="color:var(--ok)"' : ''}>${signed(td.gain)}</b></div>
          <div class="spread"><span>Declared margin</span><b class="mono">${num(td.required_gain)}</b></div>
          <div class="spread"><span>Decision</span><span class="pill ${td.accepted ? 'ok' : 'outline'}">${td.accepted ? 'keep ' + esc(shortId(V, kept)) : 'keep the defaults'}</span></div>
        </div>
        <div class="callout info"><span class="ic">${icon('info')}</span><span>${diffs.length ? `Paired on the same ${diffs.length} folds, ${esc(shortId(V, td.best_candidate))} against the baseline (positive is better): ${diffs.map(signed).join(', ')}. ` : ''}Candidates tried: ${esc(params)} ${DC.REC['DCLAB-R16'] ? chip('DCLAB-R16') : ''}</span></div>`;
    }
    /* ---------- "Choose a different model": the stage approval with your choice, checked by the graph first ---------- */
    function overrideReal(V, pre) {
      if (!V.ev) { toast('Run the models stage first; then there are candidates to choose from.', { ok: false }); return; }
      const opts = ((V.m.decision || {}).options || []).filter(o => o.id !== V.chosen);
      if (!opts.length) { toast('There is no other candidate to choose.', { ok: false }); return; }
      if (pre && !opts.some(o => o.id === pre)) { toast(pre === V.chosen ? `${pre} is already the chosen model.` : `${pre} is not among the screened models, so it cannot be chosen here.`, { ok: false }); return; }
      DC.modal.open({
        eyebrow: '<span class="eyebrow">Override the rule</span>', title: 'Choose a different model', confirm: 'Approve this choice',
        html: `<p>The rule picked <b>${esc(V.selected)}</b>${V.overridden ? `; the current choice is <b>${esc(V.chosen)}</b>, made by a person` : ''}. Choosing another candidate approves the models stage with your choice; the workflow log records that a person made it.</p>${pre ? `<p class="small muted">From an improvement run: the final stage tunes ${esc(pre)} again on its own candidates, so that way's own settings are not carried over and its score on the folds is not guaranteed.</p>` : ''}
          <div class="field"><label for="ov-model">Model</label><select id="ov-model">${opts.map(o => `<option value="${esc(o.id)}">${esc(o.label || o.id)} · ${num(o.mean)} ± ${num(o.std)}</option>`).join('')}</select></div>
          <div id="ov-check"><div class="xs muted">Checking the move against the workflow graph…</div></div>`,
        onOpen: mo => { const sel = $('#ov-model', mo), check = () => checkMove(V, sel.value, $('#ov-check', mo)); if (pre && opts.some(o => o.id === pre)) sel.value = pre; sel.addEventListener('change', check); check(); },
        onConfirm: mo => { approve(V, $('#ov-model', mo).value); return false; },
      });
    }
    async function checkMove(V, choice, box) {
      try {
        const v = await DC.client.checkMove(V.p.id, { body: { move: 'approve_stage', stage: 'models', choice } });
        const bad = v.status === 'blocked';
        const failed = (v.checks || []).filter(c => !c.ok).map(c => c.detail || c.name);
        const said = [...failed, ...(v.side_effects || [])];
        box.innerHTML = `<div class="callout ${bad ? 'bad' : said.length ? 'warn' : 'ok'}"><span class="ic">${icon(bad ? 'alert' : said.length ? 'info' : 'check')}</span><span><b>${bad ? 'Blocked by the workflow graph.' : 'Allowed.'}</b> ${said.map(s => DC.linkIds(s)).join(' ')}</span></div>`;
        DC.hydrate(box);
      } catch (e) { box.innerHTML = `<div class="callout bad"><span class="ic">${icon('alert')}</span><span>${esc(e.message)}</span></div>`; }
    }
    async function approve(V, choice) {
      const btn = $('#modal-confirm');
      btn.disabled = true;
      try {
        await DC.client.approveStage(V.p.id, 'models', { body: { choice } });
        DC.modal.close();
        DC.currentProject.clear();
        toast(Number(V.p.holdout_uses) > 0 ? `${choice} is now your choice. The final stage was cleared; the holdout is already used, so a new estimate on it needs a written reason.` : `${choice} is now your choice. Run the final stage to tune it and open the holdout.`);
        view.enter(el);
      } catch (e) { btn.disabled = false; toast(e.message, { ok: false }); }
    }
    this.renderReal = p => { real = build(p); paint(real); };
  },
  /* the Evaluation tab: the same panels as the Evaluation page, for the open project (its final stage's record) */
  async loadEvaluation(el, p) {
    await DC.evaluation.mount(DC.$('#mo-eval', el), p);
  },
  async enter(el) {
    let p = null;
    try { p = await DC.currentProject.get(); } catch (e) { p = null; }
    DC.markSample(!p);
    if (p) this.renderReal(p); else this.restoreSample();
    this.loadEvaluation(el, p);
    clearTimeout(this.timer);
    if (this.waitImprove && Date.now() > this.waitImprove.until) this.waitImprove = null;
    if (p && (p.running || this.waitImprove)) this.timer = setTimeout(() => { if (DC.state.view === 'models') { DC.currentProject.clear(); this.enter(el); } }, 3000);
  },
});
