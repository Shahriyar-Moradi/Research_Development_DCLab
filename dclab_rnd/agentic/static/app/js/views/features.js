DC.view('features', {
  init(el) {
    const { $, $$, esc, chip, charts, icon, fmt, pct, int } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id));
    const SNAP = ['feat-head', 'feat-stats', 'feat-ladder', 'feat-eng-foot'];
    const SAMPLE = Object.fromEntries(SNAP.map(id => [id, $('#' + id, el).innerHTML]));
    const SAMPLE_LADDER = [['raw', 15, 0.7130, 0.0129], ['logs', 23, 0.7130, 0.0129], ['ratios', 38, 0.7172, 0.0165], ['interactions', 46, 0.7136, 0.0169], ['full_fe', 51, 0.7130, 0.0216], ['selected', 25, 0.7029, 0.0106]];
    const SAMPLE_ENG = [
      { name: 'poutcome=success', family: 'onehot', label: 'One-hot', how: 'poutcome = success (fitted per fold)', timing: 'Before call', imp: 0.24, status: 'clean', note: 'Known prior campaign result' },
      { name: 'pdays_never_contacted', family: 'engineered', label: 'State flag', how: 'indicator(pdays == -1)', timing: 'Before call', imp: 0.18, status: 'clean', note: '81.8% of rows share -1 state' },
      { name: 'balance_to_age_ratio', family: 'engineered', label: 'Ratio', how: 'balance / max(age, 1)', timing: 'Before call', imp: 0.14, status: 'clean', note: 'Normalized per fold' },
      { name: 'month=mar', family: 'onehot', label: 'One-hot', how: 'month = mar (scheduled date)', timing: 'At scheduling', imp: 0.12, status: 'watch', note: 'Seasonal lift without year column' },
      { name: 'log1p_poutcome', family: 'engineered', label: 'Transform', how: 'log1p(category_code(poutcome))', timing: 'Before call', imp: 0.05, status: 'fix', note: 'Log on category code is not meaningful; use one-hot' },
      { name: 'housing', family: 'raw', label: 'Raw binary', how: 'housing loan flag as recorded', timing: 'Before call', imp: 0.09, status: 'clean', note: 'Point-in-time snapshot' },
    ];
    const STAGES = ['data', 'leakage', 'features', 'models', 'final'];
    const isDone = (p, s) => ['completed', 'approved'].includes(((p.stages || {})[s] || {}).status);
    const busy = p => !!p.running || Object.values(p.stages || {}).some(s => s && (s.status === 'queued' || s.status === 'running'));
    const rc = id => (DC.REC[id] ? chip(id) : '');
    const ML = { roc_auc: 'ROC-AUC', average_precision: 'average precision', macro_f1: 'macro-F1', mae: 'MAE', rmse: 'RMSE', r2: 'R²' };
    const LOWER = new Set(['mae', 'rmse', 'log_loss', 'brier']);
    const mlabel = m => ML[m] || (m ? String(m).replace(/_/g, ' ') : 'score');
    const signed = v => (v >= 0 ? '+' : '−') + fmt(Math.abs(v));
    let real = null, mode = 'sample', LAD = null, filter = 'all', currentRows = SAMPLE_ENG, stopPoll = null;

    function runBox(p, stage, what) {
      const st = ((p.stages || {})[stage] || {}).status || 'pending';
      if (st === 'queued' || st === 'running') return `<div class="empty">Running the ${esc(stage)} stage… this page updates when it finishes.</div>`;
      const prev = STAGES[STAGES.indexOf(stage) - 1];
      const ready = prev ? isDone(p, prev) : !!p.solution;
      const move = ((p.graph || {}).moves || []).find(m => m.move === 'run_stage' && m.stage === stage);
      const allowed = ready && !busy(p) && (!move || move.status === 'allowed');
      const action = allowed ? `<span><button type="button" class="btn sm primary" data-run-stage="${esc(stage)}">${icon('play')}Run the ${esc(stage)} stage${st === 'failed' ? ' again' : ''}</button></span>`
        : !ready ? `<span class="xs">${prev ? `Run the ${esc(prev)} stage first.` : 'Save the solution first.'}</span>` : move && move.message && move.status !== 'allowed' ? `<span class="xs">${esc(move.message)}</span>` : '';
      return `<div class="empty stack tight"><span>${what}</span>${action}</div>`;
    }

    function drawSampleLadder() {
      const tol = Number($('#feat-tol', el).value);
      $('#feat-tol-val', el).textContent = tol.toFixed(4);
      const best = Math.max(...SAMPLE_LADDER.map(r => r[2]));
      const ok = SAMPLE_LADDER.filter(r => r[2] >= best - tol - 1e-9);
      const pick = ok.reduce((a, b) => (b[1] < a[1] ? b : a));
      $('#feat-ladder-table tbody', el).innerHTML = SAMPLE_LADDER.map(r => `<tr class="${r === pick ? 'selected' : ''}"><td><span class="cell-main">${r[0]}</span>${r === pick ? ' <span class="pill accent">chosen</span>' : ''}</td><td class="num">${r[1]}</td><td class="num mono">${r[2].toFixed(4)}</td><td class="num mono">±${r[3].toFixed(4)}</td><td>${ok.includes(r) ? '<span class="yes">✓</span>' : '<span class="no">—</span>'}</td></tr>`).join('');
      const declared = Math.abs(tol - 0.002) < 1e-9;
      $('#feat-tol-out', el).innerHTML = `<div class="callout ${declared ? 'info' : 'warn'}"><span class="ic">${icon(declared ? 'info' : 'alert')}</span><span>${declared
        ? `With declared tolerance 0.0020 the rule picks <b>ratios</b> (38 features, +0.0042 over raw).`
        : `With ${tol.toFixed(4)} the rule would pick <b>${pick[0]}</b> (${pick[1]} features). Changing tolerance after seeing results requires a protocol reason ${chip('DCLAB-R07')}.`}</span></div>`;
      DC.hydrate(el);
    }

    function renderEngTable() {
      const list = currentRows.filter(r => filter === 'all' || r.family === filter);
      $('#feat-eng-table tbody', el).innerHTML = list.length ? list.map(r => `<tr>
        <td><code>${esc(r.name)}</code></td>
        <td><span class="pill outline">${esc(r.label)}</span></td>
        <td class="small">${esc(r.how)}${r.note ? `<div class="xs muted">${esc(r.note)}</div>` : ''}</td>
        <td class="small">${esc(r.timing)}</td>
        <td class="num mono">${r.imp != null ? pct(r.imp) : '—'}</td>
        <td><span class="pill ${r.status === 'clean' ? 'ok' : r.status === 'fix' ? 'bad' : 'warn'}">${esc(r.status)}</span></td>
      </tr>`).join('') : `<tr><td colspan="6"><div class="empty">No features match this filter.</div></td></tr>`;
    }

    function renderSignals(items, subHtml) {
      $('#feat-imp-sub', el).textContent = subHtml;
      $('#feat-imp-body', el).innerHTML = items.length ? charts.barsH(items.slice(0, 8).map(t => ({
        label: String(t.name).slice(0, 22), value: t.imp || 0, valueText: pct(t.imp || 0), cls: t.status === 'clean' ? '' : 'warn',
      })), { width: 400, labelW: 150, valW: 54, min: 0, max: Math.max(0.1, ...items.map(t => t.imp || 0)) * 1.15, rowH: 26, aria: 'Top features by importance' })
        : '<div class="empty">Run the features stage to inspect normalized importance.</div>';
      $('#feat-sig-body', el).innerHTML = charts.barsH([
        { label: 'poutcome = success', value: 0.647, valueText: '64.7%', cls: 'proof' },
        { label: 'age band 61+', value: 0.421, valueText: '42.1%', cls: 'proof' },
        { label: 'age band 18–30', value: 0.164, valueText: '16.4%' },
        { label: 'never contacted', value: 0.092, valueText: '9.2%' },
      ], { width: 400, labelW: 150, valW: 58, min: 0, max: 0.75, rowH: 26, ticks: [0, 0.35, 0.7], tickFmt: v => Math.round(v * 100) + '%', aria: 'Train-only outcome rates' });
    }

    function drawSample() {
      currentRows = SAMPLE_ENG;
      drawSampleLadder();
      renderEngTable();
      renderSignals(SAMPLE_ENG, 'Normalized importance · ratios recipe (sample)');
    }

    function drawRealLadder() {
      const L = LAD;
      if (!L || !$('#feat-tol', el)) return;
      const tol = Number($('#feat-tol', el).value);
      const show = v => (L.rel ? (v * 100).toFixed(2) + '%' : v.toFixed(4));
      $('#feat-tol-val', el).textContent = show(tol);
      const means = L.opts.map(o => o.mean), best = L.higher ? Math.max(...means) : Math.min(...means);
      const slack = L.rel ? Math.abs(best) * tol : tol;
      const ok = L.opts.filter(o => (L.higher ? o.mean >= best - slack - 1e-12 : o.mean <= best + slack + 1e-12));
      const pick = ok.reduce((a, b) => (b.features < a.features || (b.features === a.features && (b.elapsed_seconds || 0) < (a.elapsed_seconds || 0)) ? b : a));
      $('#feat-ladder-table tbody', el).innerHTML = L.opts.map(o => `<tr class="${o === pick ? 'selected' : ''}"><td><span class="cell-main">${esc(o.label || o.id)}</span>${o === pick ? ' <span class="pill accent">chosen</span>' : ''}${o.description ? `<div class="cell-sub">${esc(o.description)}</div>` : ''}</td><td class="num">${int(Math.round(o.features || 0))}</td><td class="num mono">${fmt(o.mean)}</td><td class="num mono">±${fmt(o.std)}</td><td>${ok.includes(o) ? '<span class="yes">✓</span>' : '<span class="no">—</span>'}</td></tr>`).join('');
      const declared = Math.abs(tol - L.tol) < 1e-9;
      const refO = L.opts.find(o => o.id === L.ref) || L.opts[0];
      const gain = L.higher ? pick.mean - refO.mean : refO.mean - pick.mean;
      $('#feat-tol-out', el).innerHTML = `<div class="callout ${declared ? 'info' : 'warn'}"><span class="ic">${icon(declared ? 'info' : 'alert')}</span><span>${declared
        ? `Declared ${show(L.tol)} picks <b>${esc(pick.label || pick.id)}</b> (${int(Math.round(pick.features || 0))} features, ${signed(gain)} over ${esc(refO.label || refO.id)}).`
        : `With ${show(tol)} the rule would pick <b>${esc(pick.label || pick.id)}</b> (${int(Math.round(pick.features || 0))} features). ${rc('DCLAB-R07')}`}</span></div>`;
      DC.hydrate(el);
    }

    function renderReal(p) {
      real = p; mode = 'real';
      const R = p.records || {}, sol = p.solution || {}, fe = R.features || null, fin = (R.final || {}).evidence || null;
      const dev = (R.data || {}).evidence || null, prof = ((p.data || {}).profile || {}).columns || [];
      const ev = fe ? (fe.evidence || {}) : {}, dec = fe ? (fe.decision || {}) : {}, m = (fe && fe.primary_metric) || sol.metric || 'roc_auc';
      const feDone = isDone(p, 'features');
      const selRecipe = ev.selected_recipe || dec.selected || '—';
      const stageRows = (ev.stage_results || []);
      const selObj = stageRows.find(x => x.recipe === selRecipe) || {};
      const selMean = selObj.metrics && selObj.metrics[m] ? selObj.metrics[m].mean : null;
      const refObj = stageRows.find(x => x.recipe === (ev.reference_recipe || 'raw')) || stageRows[0];
      const refMean = refObj && refObj.metrics && refObj.metrics[m] ? refObj.metrics[m].mean : null;
      const gain = (selMean != null && refMean != null) ? (!LOWER.has(m) ? selMean - refMean : refMean - selMean) : null;

      $('#feat-head', el).innerHTML = `<div class="eyebrow accent">${esc(p.name)} · WF-06</div>
        <h1>Features <span class="pill ${feDone ? 'ok' : 'outline'}">${feDone ? esc(selRecipe) : 'pending stage'}</span></h1>
        <p class="lede">Every transformation is fit strictly inside training folds. A recipe stays only if it clears the predeclared tolerance.</p>`;

      $('#feat-stats', el).innerHTML = `<div class="stat"><span class="v">${selObj.feature_count_mean != null ? int(Math.round(selObj.feature_count_mean)) : prof.length}</span><span class="l">${fe ? `active features (${esc(selRecipe)})` : 'source columns before feature engineering'}</span></div>
        <div class="stat"><span class="v ${selMean != null ? 'ok' : ''}">${selMean != null ? fmt(selMean) : '—'}</span><span class="l">mean CV ${esc(mlabel(m))} on training folds</span></div>
        <div class="stat"><span class="v">${gain != null ? signed(gain) : '—'}</span><span class="l">gain over ${esc((refObj && refObj.recipe) || 'raw')} baseline</span></div>
        <div class="stat"><span class="v ok">Holdout sealed</span><span class="l">zero holdout rows inspected in feature engineering</span></div>`;

      if (!fe) {
        LAD = null;
        $('#feat-ladder', el).innerHTML = `<div class="panel-head"><div class="panel-title"><h2>Feature ladder</h2><span class="sub">Run WF-06 to compare candidate feature recipes safely inside training folds.</span></div></div><div class="panel-body">${runBox(p, 'features', 'Feature engineering has not run for this project yet.')}</div>`;
      } else {
        const opts = (dec.options && dec.options.length ? dec.options : stageRows.map(x => ({ id: x.recipe, label: x.recipe, description: x.description, features: x.feature_count_mean, mean: (x.metrics[m] || {}).mean, std: (x.metrics[m] || {}).std, elapsed_seconds: x.elapsed_seconds }))).filter(o => o.mean != null);
        const rule = ev.selection_rule || dec.rule || '';
        const rel = /\(relative\)/.test(rule), mt = rule.match(rel ? /within ([\d.]+)%/ : /within ([\d.]+) of/);
        const tol = mt ? (rel ? +mt[1] / 100 : +mt[1]) : 0.002;
        const max = rel ? 0.05 : Math.max(0.01, tol * 5), step = rel ? 0.0025 : max / 20;
        LAD = { opts, m, higher: !LOWER.has(m), tol, rel, selected: selRecipe, chosen: dec.chosen || selRecipe, ref: ev.reference_recipe };
        $('#feat-ladder', el).innerHTML = `<div class="panel-head"><div class="panel-title"><h2>Feature ladder</h2><span class="sub">${opts.length} recipes fit inside training folds.</span></div>${rc('DCLAB-R07')}</div>
          <div class="table-wrap"><table class="data compact" id="feat-ladder-table"><thead><tr><th>Recipe</th><th class="num">Features</th><th class="num">Mean ${esc(mlabel(m))}</th><th class="num">Fold spread</th><th>Within tolerance</th></tr></thead><tbody></tbody></table></div>
          <div class="panel-body stack tight"><div class="row"><label for="feat-tol" class="small strong" data-style="color:var(--ink)">Declared tolerance</label><input type="range" id="feat-tol" min="0" max="${max}" step="${step}" value="${tol}" data-style="flex:1;min-width:160px"><span class="mono small" id="feat-tol-val"></span></div><div id="feat-tol-out"></div></div>`;
        if (LAD.opts.length) drawRealLadder();
      }

      const rawFeats = fin && (fin.top_final_features || []).length
        ? fin.top_final_features.map(t => ({ f: t.feature, imp: t.normalized_importance }))
        : (selObj.top_features || []).map(t => ({ f: t.feature, imp: t.mean_normalized_importance }));
      const kinds = Object.fromEntries(prof.map(c => [c.name, c.kind]));
      const names = prof.map(c => c.name).sort((a, b) => b.length - a.length);
      const forb = new Set((sol.forbidden || []).map(f => f.column));
      const dprof = Object.fromEntries(((dev || {}).feature_profiles || []).map(x => [x.feature, x]));
      const parseFeat = n => {
        const eq = n.indexOf('=');
        if (eq > 0 && kinds[n.slice(0, eq)]) return { family: 'onehot', label: 'One-hot', how: `category ${n.slice(0, eq)} = ${n.slice(eq + 1)}`, base: [n.slice(0, eq)] };
        if (kinds[n]) return { family: 'raw', label: kinds[n], how: 'input column as recorded', base: [n] };
        const base = names.filter(c => n.includes(c)).slice(0, 2);
        return { family: 'engineered', label: 'Engineered', how: base.length ? `derived from ${base.join(', ')}` : 'derived fold transform', base };
      };
      currentRows = rawFeats.length ? rawFeats.map(({ f, imp }) => {
        const parsed = parseFeat(f);
        const bad = parsed.base.some(b => forb.has(b));
        const risk = parsed.base.map(b => dprof[b]).filter(Boolean).find(d => (d.train_vs_holdout_psi || 0) > 0.2 || d.missing_rate > 0.05);
        return {
          name: f, family: parsed.family, label: parsed.label, how: parsed.how,
          timing: bad ? 'Blocked by moment' : 'Before moment', imp,
          status: bad ? 'fix' : risk ? 'watch' : 'clean',
          note: bad ? 'Derived from a forbidden column' : risk ? `Missing ${pct(risk.missing_rate)} / PSI ${fmt(risk.train_vs_holdout_psi || 0, 2)}` : '',
        };
      }) : prof.filter(c => c.name !== sol.target && !(sol.identifiers || []).includes(c.name)).map(c => ({
        name: c.name, family: 'raw', label: c.kind || 'column', how: 'base table column',
        timing: forb.has(c.name) ? 'Forbidden' : 'Before moment', imp: null,
        status: forb.has(c.name) ? 'fix' : 'clean', note: forb.has(c.name) ? 'Excluded in solution' : '',
      }));
      $('#feat-eng-sub', el).textContent = rawFeats.length ? `Showing top ${rawFeats.length} ranked features from ${selRecipe} recipe.` : `Base schema columns (${currentRows.length}). Run features stage for engineered recipes.`;
      renderEngTable();

      const obf = ((dev || {}).outcome_by_feature || []).slice(0, 5);
      $('#feat-imp-sub', el).textContent = rawFeats.length ? `Top features (${selRecipe} recipe)` : 'Awaiting features stage';
      $('#feat-imp-body', el).innerHTML = rawFeats.length ? charts.barsH(currentRows.slice(0, 8).map(t => ({
        label: String(t.name).slice(0, 22), value: t.imp || 0, valueText: pct(t.imp || 0), cls: t.status === 'clean' ? '' : 'warn',
      })), { width: 400, labelW: 150, valW: 54, min: 0, max: Math.max(0.1, ...currentRows.map(t => t.imp || 0)) * 1.15, rowH: 26, aria: 'Top features by importance' })
        : '<div class="empty">Run the features stage to view normalized feature importance.</div>';
      $('#feat-sig-body', el).innerHTML = obf.length ? charts.barsH(obf.map(f => ({
        label: String(f.feature).slice(0, 22), value: f.spread || 0, valueText: fmt(f.spread, 3), cls: '',
      })), { width: 400, labelW: 150, valW: 58, min: 0, max: Math.max(0.1, ...obf.map(f => f.spread || 0)) * 1.15, rowH: 26, aria: 'Outcome spread on training rows' })
        : '<div class="empty">Run the data stage to inspect train-only column spreads.</div>';

      DC.hydrate(el);
      if (busy(p)) watch();
    }

    function restoreSample() {
      if (stopPoll) { stopPoll(); stopPoll = null; }
      real = null; LAD = null;
      if (mode === 'sample') return;
      mode = 'sample';
      SNAP.forEach(id => { $('#' + id, el).innerHTML = SAMPLE[id]; });
      drawSample();
    }

    function watch() {
      if (stopPoll) return;
      stopPoll = DC.poll(async () => {
        if (DC.state.view !== 'features' || mode !== 'real') { stopPoll = null; return true; }
        const p = await DC.currentProject.get(true);
        if (!p) { stopPoll = null; return true; }
        const still = busy(p);
        if (!still) stopPoll = null;
        renderReal(p);
        return !still;
      }, 1500);
    }

    drawSample();
    el.addEventListener('input', e => { if (e.target.id === 'feat-tol') (mode === 'real' ? drawRealLadder() : drawSampleLadder()); });
    el.addEventListener('click', async e => {
      const chipBtn = e.target.closest('[data-feat-filter]');
      if (chipBtn) {
        filter = chipBtn.dataset.featFilter;
        $$('[data-feat-filter]', el).forEach(b => b.setAttribute('aria-pressed', String(b === chipBtn)));
        renderEngTable();
        return;
      }
      const run = e.target.closest('[data-run-stage]');
      if (!run || !real) return;
      run.disabled = true;
      try { await DC.client.runStage(real.id, run.dataset.runStage); DC.toast(`Started the ${run.dataset.runStage} stage.`); }
      catch (err) { DC.toast(err.message, { ok: false }); }
      DC.currentProject.clear(); watch();
    });

    this.renderReal = renderReal;
    this.restoreSample = restoreSample;
  },
  async enter(el) {
    let p = null;
    try { p = await DC.currentProject.get(); } catch (e) { p = null; }
    DC.markSample(!p);
    if (p) this.renderReal(p); else this.restoreSample();
  },
});
