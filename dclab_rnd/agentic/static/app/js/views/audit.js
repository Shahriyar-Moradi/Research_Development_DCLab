DC.view('audit', {
  init(el) {
    const { $, esc, chip, chips, charts, icon, fmt, pct, int } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id));
    /* Sample mode keeps the markup above; real mode writes the open project into the same panels and restores them on the way back. */
    const SNAP = ['aud-head', 'aud-stats', 'aud-matrix-foot', 'aud-leaks', 'aud-eda', 'aud-critic'];
    const SAMPLE = Object.fromEntries(SNAP.map(id => [id, $('#' + id, el).innerHTML]));
    let real = null, mode = 'sample', stopPoll = null, advCompare = {};

    /* ---------- sample: the term-deposit project ---------- */
    const COLS = [
      ['age', 'client profile', 'yes', '', 'allowed'], ['job', 'client profile', 'yes', '', 'allowed'], ['marital', 'client profile', 'yes', '', 'allowed'], ['education', 'client profile', 'yes', '', 'allowed'],
      ['default', 'account state', 'yes', '', 'allowed'], ['balance', 'account state', 'yes', '', 'allowed'], ['housing', 'account state', 'yes', '', 'allowed'], ['loan', 'account state', 'yes', '', 'allowed'],
      ['contact', 'call scheduled', 'yes', '', 'allowed'], ['day_of_week', 'call scheduled (day of month)', 'yes', '', 'allowed'], ['month', 'call scheduled', 'yes', '', 'allowed'],
      ['pdays', 'earlier campaigns · −1 means never', 'yes', '', 'allowed'], ['previous', 'earlier campaigns', 'yes', '', 'allowed'], ['poutcome', 'earlier campaigns', 'yes', '', 'allowed'],
      ['campaign', 'during this campaign', 'ask', '', 'ask owner'], ['duration', 'when the call ends', 'no', 'name · AUC 0.812 · declared', 'forbidden'],
    ];
    const known = { yes: '<span class="yes">✓ yes</span>', no: '<span class="pill bad">no · after</span>', ask: '<span class="pill warn">unclear</span>' };
    const verdict = v => v === 'forbidden' ? '<span class="pill bad">forbidden</span>' : v === 'ask owner' ? '<span class="pill warn">ask owner</span>' : `<span class="pill ok">${esc(v)}</span>`;
    const CRIT = [
      ['EXP-007', 'high', 'Calling duration a decision-time exclusion is plausible but not proven here; the evidence only shows a declared exclusion and a CV lift.', 'kept', 'Agrees with the numbers. Resolved by the solution: the call length is written when the call ends (timeline on the solution page).'],
      ['EXP-007', 'high', 'The +0.1842 AUC lift is not sufficient to label a feature as leakage or proxy; predictive lift can arise from legitimate signal.', 'kept', 'Agrees with the numbers and with DCLAB-R06. The lift is shown as severity only.'],
      ['EXP-007', 'medium', 'The canary detection only shows the detector can catch a synthetic canary. It does not validate detection of real leakage columns.', 'kept', 'Fair. The blind auditor replay measures it: 11 of 18 known leaks found across 7 datasets.'],
      ['EXP-008', 'high', 'The selection rule is not satisfied because the selected stage mean ROC-AUC is 0.7029, not within 0.002 of the best stage mean 0.7172.', 'dropped', 'Contradicted by the numbers. The chosen recipe is ratios (0.7172, the best mean). 0.7029 belongs to a different recipe that happens to be named "selected".'],
      ['EXP-008', 'high', 'Higher training-CV ROC-AUC is treated as sufficient evidence; the improvement may be within sampling noise.', 'kept', 'Agrees with the numbers: +0.0042 against a fold spread of ±0.0129 to ±0.0165. Next test: 5×2 repeated CV before trusting the ladder.'],
    ];
    function renderSampleAdvisor() {
      const box = $('#aud-advisor', el);
      if (!box) return;
      const cmp = advCompare.campaign || null;
      const excl = advCompare._excluded && advCompare._excluded.campaign;
      if (excl) {
        box.innerHTML = `<div class="panel"><div class="panel-body spread small"><span><b>✓ Confirmed in sample:</b> <code>campaign</code> marked forbidden against prediction moment.</span><button type="button" class="btn sm" data-adv-reset="campaign">Undo</button></div></div>`;
        return;
      }
      box.innerHTML = `<div class="panel leak-adv-card">
        <div class="panel-head"><div class="panel-title"><div class="row" data-style="gap:6px;flex-wrap:wrap"><span class="pill warn">Leakage Advisor · Very likely leakage</span><span class="pill outline mono">cell_values_sent: 0</span></div><h3>Column <code>campaign</code> may count future contacts</h3></div></div>
        <div class="panel-body stack tight small">
          <p class="m-0">Prediction moment: <b>Immediately before a marketing call</b>. Because <code>campaign</code> counts contacts performed during this campaign, it is very likely leakage. Choose how to proceed:</p>
          ${cmp ? `<div class="leak-contrast-grid">
            <div class="leak-contrast-box is-safe"><div class="xs muted">SAFE · Without campaign</div><div class="mono" data-style="font-size:20px;font-weight:600">0.7130</div><div class="xs muted">3-fold CV ROC-AUC (holdout sealed)</div></div>
            <div class="leak-contrast-box"><div class="xs bad">LEAKY · With campaign</div><div class="mono" data-style="font-size:20px;font-weight:600">0.8972</div><div class="xs muted">+0.1842 inflated lift on training folds</div></div>
          </div><div class="row" data-style="gap:8px;margin-top:8px"><button type="button" class="btn sm primary" data-adv-sample="exclude">Confirm: keep safe pipeline &amp; exclude campaign</button></div>`
            : `<div class="row" data-style="gap:8px;flex-wrap:wrap"><button type="button" class="btn sm primary" data-adv-sample="compare">Compare both on training folds (Recommended)</button><button type="button" class="btn sm" data-adv-sample="exclude">Exclude directly</button></div>`}
        </div>
      </div>`;
    }
    function drawSample() {
      renderSampleAdvisor();
      $('#avail-table tbody', el).innerHTML = COLS.map(([c, w, k, f, v]) => `<tr class="${v === 'forbidden' ? 'selected' : ''}"><td><code>${c}</code></td><td class="small">${esc(w)}</td><td>${known[k]}</td><td class="small">${f ? `<span class="pill warn">${esc(f)}</span> ${chip('LEAK-bank_marketing')}` : '<span class="faint">none</span>'}</td><td>${verdict(v)}</td></tr>`).join('');
      $('#ablation-chart', el).innerHTML = charts.barsH([
        { label: 'Safe (3,200 rows)', value: 0.7130, valueText: '0.7130', cls: '' },
        { label: 'With duration', value: 0.8972, valueText: '0.8972', cls: 'bad' },
        { label: 'Safe (full rows)', value: 0.8029, valueText: '0.8029', cls: '' },
        { label: 'With duration', value: 0.9357, valueText: '0.9357', cls: 'bad' },
      ], { width: 340, labelW: 112, valW: 52, min: 0.5, max: 1.0, rowH: 28, ticks: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], tickFmt: v => v.toFixed(1), aria: 'ROC-AUC with and without duration' });
      $('#critic-list', el).innerHTML = CRIT.map(([ev, sev, txt, gate, why]) => `<div class="list-item"><span class="sev ${sev}">${sev}</span><div class="li-main"><span class="small" data-style="color:var(--ink-2)">“${esc(txt)}”</span><span class="row small"><span class="pill ${gate === 'kept' ? 'ok' : 'bad'}">${gate === 'kept' ? 'kept' : 'dropped by the gate'}</span><span class="muted">${DC.linkIds(why)}</span></span></div>${chip(ev)}</div>`).join('');
      const p0 = v => (v * 100).toFixed(0) + '%';
      $('#eda-month', el).innerHTML = charts.vbars([
        ['J', 0.102], ['F', 0.168], ['M', 0.509], ['A', 0.196], ['M', 0.067], ['J', 0.104], ['J', 0.093], ['A', 0.109], ['S', 0.452], ['O', 0.438], ['N', 0.101], ['D', 0.472],
      ].map(([l, v]) => ({ label: l, value: v, cls: v > 0.3 ? 'proof' : '' })), { width: 300, height: 140, max: 0.6, yFmt: p0, aria: 'Subscription rate by month' });
      $('#eda-pout', el).innerHTML = charts.barsH([
        { label: 'never contacted', value: 0.092, valueText: '9.2% · 29,589' }, { label: 'failure', value: 0.123, valueText: '12.3% · 3,889' },
        { label: 'other', value: 0.165, valueText: '16.5% · 1,485' }, { label: 'success', value: 0.647, valueText: '64.7% · 1,205', cls: 'proof' },
      ], { width: 300, labelW: 98, valW: 92, min: 0, max: 0.7, rowH: 26, ticks: [0, 0.35, 0.7], tickFmt: p0, aria: 'Subscription rate by previous outcome' });
      $('#eda-age', el).innerHTML = charts.barsH([
        { label: '18–30', value: 0.164, valueText: '16.4%' }, { label: '31–40', value: 0.102, valueText: '10.2%' }, { label: '41–50', value: 0.091, valueText: '9.1%' },
        { label: '51–60', value: 0.099, valueText: '9.9%' }, { label: '61+', value: 0.421, valueText: '42.1%', cls: 'proof' },
      ], { width: 300, labelW: 60, valW: 52, min: 0, max: 0.5, rowH: 24, ticks: [0, 0.25, 0.5], tickFmt: p0, aria: 'Subscription rate by age band' });
      DC.hydrate(el);
    }
    drawSample();

    /* ---------- real project: the solution plus the data, leakage, features and final records ---------- */
    const STAGES = ['data', 'leakage', 'features', 'models', 'final'];
    const isDone = (p, s) => ['completed', 'approved'].includes(((p.stages || {})[s] || {}).status);
    const busy = p => !!p.running || Object.values(p.stages || {}).some(s => s && (s.status === 'queued' || s.status === 'running'));
    const rc = id => (DC.REC[id] ? chip(id) : '');
    const ML = { roc_auc: 'ROC-AUC', average_precision: 'average precision', macro_f1: 'macro-F1', mae: 'MAE', rmse: 'RMSE', r2: 'R²' };
    const LOWER = new Set(['mae', 'rmse', 'log_loss', 'brier']);
    const mlabel = m => ML[m] || (m ? String(m).replace(/_/g, ' ') : '—');
    const code = c => `<code>${esc(c)}</code>`;
    const short = (s, n = 16) => (String(s).length > n ? String(s).slice(0, n - 1) + '…' : String(s));
    const signed = v => (v >= 0 ? '+' : '−') + fmt(Math.abs(v));
    const ASK = /\bask\b|owner|unclear|unsure|confirm|\?/i;
    const RS = { suspicious_name: 'name', strong_univariate_signal: 'strong signal', extreme_univariate_signal: 'extreme signal', exact_target_proxy: 'exact copy of the target', arithmetic_target_identity: 'sums to the target', identifier_like: 'unique per row', declared_post_outcome_or_contested: 'declared' };
    const reasonText = f => (f.reasons || []).map(r => RS[r] || r.replace(/_/g, ' ')).join(' · ') + (f.univariate_auc != null ? ` · AUC ${fmt(f.univariate_auc, 3)}` : '');
    const day = at => { const d = new Date(at); return at && !isNaN(d) ? d.toLocaleString('en-US', { month: 'short', day: '2-digit' }) : ''; };

    function runBox(p, stage, what) {
      const st = ((p.stages || {})[stage] || {}).status || 'pending';
      if (st === 'queued' || st === 'running') return `<div class="empty">Running the ${esc(stage)} stage… this page updates when it finishes.</div>`;
      const prev = STAGES[STAGES.indexOf(stage) - 1];
      const ready = prev ? isDone(p, prev) : !!p.solution;
      const move = ((p.graph || {}).moves || []).find(m => m.move === 'run_stage' && m.stage === stage);
      const allowed = ready && !busy(p) && (!move || move.status === 'allowed');
      const failed = st === 'failed' ? `<span class="small">The last run failed: ${esc((p.stages[stage] || {}).error || 'unknown error')}</span>` : '';
      const action = allowed ? `<span><button type="button" class="btn sm primary" data-run-stage="${esc(stage)}">${icon('play')}Run the ${esc(stage)} stage${st === 'failed' ? ' again' : ''}</button></span>`
        : !ready ? `<span class="xs">${prev ? `Run the ${esc(prev)} stage first.` : 'Save the solution first.'}</span>` : move && move.message && move.status !== 'allowed' ? `<span class="xs">${esc(move.message)}</span>` : '';
      return `<div class="empty stack tight"><span>${what}</span>${failed}${action}</div>`;
    }
    function facts(p) {
      const sol = p.solution || {}, R = p.records || {}, prof = ((p.data || {}).profile || {}).columns || [];
      const ev = s => (R[s] ? R[s].evidence || {} : null);
      const forb = sol.forbidden || [], lev = ev('leakage');
      return {
        sol, R, prof, dev: ev('data'), lev, fev: ev('features'), fin: ev('final'), forb,
        askF: forb.filter(f => ASK.test(f.reason || '')),
        flagged: lev ? (lev.heuristic_review_candidates || []).filter(f => !(f.reasons || []).includes('declared_post_outcome_or_contested')) : [],
        cand: lev ? Object.fromEntries((lev.heuristic_review_candidates || []).map(f => [f.feature, f])) : {},
        metric: (R.final || R.models || R.features || R.leakage || R.data || {}).primary_metric || sol.metric,
        leakIds: R.leakage ? [...new Set((R.leakage.notes || []).flatMap(n => n.proof || []).filter(id => /^(LEAK|FINDING)-/.test(id) && DC.REC[id]))].slice(0, 2) : [],
      };
    }

    function headReal(p) {
      const L = isDone(p, 'leakage');
      const at = ((p.stages || {}).leakage || {}).finished;
      const pill = L ? `<span class="pill ok">WF-05 done · ${esc(day(at))}</span>` : '<span class="pill outline">WF-05 pending</span>';
      return `<div class="eyebrow accent">${esc(p.name)} · WF-05</div>
        <h1>Leakage ${pill}</h1>
        <p class="lede">Every column is judged by when its value is written relative to the prediction moment, not by its name or score alone.</p>`;
    }
    function statsReal(p, F) {
      const { lev, dev, forb, flagged, askF } = F;
      const roles = dev ? dev.column_roles || {} : null;
      const audited = roles ? Object.keys(roles).length : Math.max(0, F.prof.length - (F.sol.target ? 1 : 0));
      const inputs = roles ? Object.values(roles).filter(r => r === 'feature').length : null;
      const declared = lev ? lev.declared_leakage_features || [] : forb.map(f => f.column);
      const flags = lev ? (flagged.length ? `${flagged.length} other heuristic flag${flagged.length === 1 ? '' : 's'}` : 'no other heuristic flag') : 'not audited yet';
      const open = [...askF.map(f => f.column), ...flagged.map(f => f.feature)];
      const can = lev && lev.detector_canary;
      return `<div class="stat"><span class="v">${audited}</span><span class="l">${lev ? 'columns audited against the solution' : 'columns to audit · the leakage stage has not run'}${inputs != null ? ` · ${inputs} used as inputs` : ''}</span></div>`
        + (declared.length ? `<div class="stat"><span class="v bad">${declared.length} <small>forbidden</small></span><span class="l">${esc(declared.join(', '))} · ${flags}</span></div>`
          : `<div class="stat"><span class="v">0 <small>forbidden</small></span><span class="l">none declared in the solution · ${lev ? (flagged.length ? `${flagged.length} heuristic flag${flagged.length === 1 ? '' : 's'}` : 'no heuristic flag') : 'not audited yet'}</span></div>`)
        + `<div class="stat"><span class="v ${open.length ? 'warn' : 'ok'}">${open.length} <small>open</small></span><span class="l">${open.length ? esc(open.slice(0, 3).join(', ')) + (open.length > 3 ? '…' : '') + ' · waiting on the owner: written when?' : 'no column waits for the owner'}</span><div class="row"><button type="button" class="link-btn small" data-pane-go="leaks">See the checks →</button></div></div>`
        + (!lev ? '<div class="stat"><span class="v">—</span><span class="l">synthetic canary · runs with the leakage stage</span></div>'
          : can && can.passed ? '<div class="stat"><span class="v ok">caught</span><span class="l">synthetic canary · the detector is awake</span></div>'
            : '<div class="stat"><span class="v bad">missed</span><span class="l">synthetic canary · treat the heuristic flags as incomplete</span></div>');
    }
    function matrixReal(p, F) {
      const { sol, prof, cand } = F;
      if (!prof.length) return { rows: '<tr><td colspan="5"><div class="empty">No data attached to this project yet.</div></td></tr>', foot: 'Attach data to audit its columns.' };
      const roles = (F.dev && F.dev.column_roles) || {};
      const forbMap = Object.fromEntries(F.forb.map(f => [f.column, f]));
      const ids = new Set(sol.identifiers || []), texts = new Set(sol.text_columns || []);
      const rows = prof.map(c => {
        const n = c.name, f = forbMap[n], k = cand[n], ask = f && ASK.test(f.reason || '');
        const role = roles[n] || (ids.has(n) ? 'identifier' : n === sol.time_column ? 'time' : n === sol.group_column ? 'group' : texts.has(n) ? 'text' : 'feature');
        const flag = k ? `<span class="pill warn">${esc(reasonText(k))}</span>${f ? ' ' + F.leakIds.map(rc).join('') : ''}` : '<span class="faint">none</span>';
        if (n === sol.target) return `<tr><td><code>${esc(n)}</code></td><td class="small">the outcome being predicted</td><td><span class="pill proof">outcome</span></td><td class="small"><span class="faint">—</span></td><td><span class="pill proof">target</span></td></tr>`;
        const written = f ? (ask ? 'unclear · ' : 'after the prediction moment · ') + esc(f.reason || '') : role === 'identifier' ? 'a key, never an input' : role === 'time' ? 'time column · orders the split' : role === 'group' ? 'group key · keeps groups together' : role === 'text' ? 'free text' : `${esc(c.kind)} input · not marked forbidden`;
        const kn = f ? (ask ? known.ask : known.no) : k ? '<span class="pill warn">review</span>' : known.yes;
        const vd = f ? verdict(ask ? 'ask owner' : 'forbidden') : role === 'identifier' ? '<span class="pill outline">identifier · excluded</span>' : k ? '<span class="pill warn">review</span>' : verdict(role === 'feature' ? 'allowed' : 'allowed · ' + role);
        return `<tr class="${f && !ask ? 'selected' : ''}"><td><code>${esc(n)}</code></td><td class="small">${written}</td><td>${kn}</td><td class="small">${flag}</td><td>${vd}</td></tr>`;
      }).join('');
      const flaggedNames = F.flagged.map(f => code(f.feature));
      const foot = F.lev
        ? `Heuristics flagged ${flaggedNames.length ? flaggedNames.length + ' column' + (flaggedNames.length === 1 ? '' : 's') + ' beyond the solution: ' + flaggedNames.join(', ') : 'no column beyond the solution'}. A missing flag never means safe: only the prediction moment (“${esc(sol.prediction_moment || 'not written yet')}”) decides ${rc('DCLAB-R05')}.`
        : `Verdicts come from the solution. Heuristic flags appear after the leakage stage ${rc('DCLAB-R05')}. ${runBox(p, 'leakage', 'The heuristic scan has not run for this project yet.')}`;
      return { rows, foot };
    }
    function leaksReal(p, F) {
      const { lev, dev } = F;
      const declared = lev ? lev.declared_leakage_features || [] : F.forb.map(f => f.column);
      const sub = declared.length ? `Safe vs with ${esc(short(declared.join(', '), 28))}, identical folds` : 'Safe baseline, identical folds';
      let left;
      if (!lev) left = runBox(p, 'leakage', 'Not measured for this project yet: run the leakage stage.');
      else {
        const cmp = lev.safe_vs_unsafe_training_cv || {}, m = cmp.metric || F.metric;
        const st = r => (r && r.metrics && r.metrics[m]) || null;
        const s = st(cmp.safe), u = st(cmp.unsafe), train = dev ? dev.train_rows : (lev.heuristic_rules || {}).rows_scanned;
        if (!s) left = '<div class="empty">The leakage record has no safe baseline score.</div>';
        else {
          const items = [{ label: `Safe (${int(train || 0)} rows)`, value: s.mean, valueText: fmt(s.mean), lo: s.mean - s.std, hi: s.mean + s.std }];
          if (u) items.push({ label: short('With ' + (declared.length === 1 ? declared[0] : declared.length + ' columns')), value: u.mean, valueText: fmt(u.mean), cls: 'bad', lo: u.mean - u.std, hi: u.mean + u.std });
          Object.entries(cmp.per_column || {}).forEach(([c, v]) => items.push({ label: short('Only ' + c), value: v.metric_mean, valueText: fmt(v.metric_mean), cls: 'bad' }));
          const bounded = !LOWER.has(m) && m !== 'r2';
          const lows = items.map(i => (i.lo != null ? i.lo : i.value)), highs = items.map(i => (i.hi != null ? i.hi : i.value));
          const min = bounded ? Math.max(0, Math.floor((Math.min(...lows) - 0.05) * 10) / 10) : 0, max = bounded ? 1 : Math.max(...highs) * 1.1;
          const ticks = bounded ? Array.from({ length: Math.round((max - min) * 10) + 1 }, (_, i) => +(min + i / 10).toFixed(1)) : undefined;
          const chart = charts.barsH(items, { width: 340, labelW: 112, valW: 52, min, max, rowH: 28, ticks, tickFmt: v => (bounded ? v.toFixed(1) : String(+v.toFixed(2))), aria: `${mlabel(m)} with and without the forbidden columns` });
          const folds = ((cmp.safe || {}).fold_metrics || []).map(f => f[m]).filter(v => v != null);
          const rvt = lev.random_vs_time_cv;
          left = `<div id="ablation-chart">${chart}</div>
            ${u ? `<div class="small">Training CV on ${int(train || 0)} rows (${esc(cmp.model)}, ${esc(cmp.recipe)} recipe): <b class="mono">${fmt(s.mean)} → ${fmt(u.mean)}</b> (${signed(lev.apparent_lift || 0)}) ${F.leakIds.map(rc).join('')}</div>`
              : `<div class="small">Training CV on ${int(train || 0)} rows (${esc(cmp.model)}, ${esc(cmp.recipe)} recipe): <b class="mono">${fmt(s.mean)} ± ${fmt(s.std)}</b>${folds.length ? ` · folds ${folds.map(v => fmt(v)).join(' · ')}` : ''}</div>
                 <div class="small muted">No column is forbidden, so there is no unsafe run to compare: the apparent lift is 0 by construction.</div>`}
            ${Object.keys(cmp.per_column || {}).length ? `<div class="small">Per column: ${Object.entries(cmp.per_column).map(([c, v]) => `${code(c)} ${signed(v.apparent_lift)}`).join(' · ')}</div>` : ''}
            ${rvt ? `<div class="small">Random KFold would report <b class="mono">${fmt(st(rvt.random_kfold) ? st(rvt.random_kfold).mean : null)}</b> against <b class="mono">${fmt((rvt.time_ordered || {}).metric_mean)}</b> time-ordered: ${signed(rvt.optimism_gap || 0)} of optimism a random split would hide.</div>` : ''}
            <div class="rule-quote">A large safe-versus-unsafe gap shows how bad a leak would be. It is not the proof. The proof is when the value is written, measured against the prediction moment ${rc('DCLAB-R06')}.</div>`;
        }
      }
      const open = [...F.askF.map(f => ({ col: f.column, why: f.reason || 'Written when?', kind: 'stays out of the model' })),
        ...F.flagged.map(f => ({ col: f.feature, why: 'Flagged: ' + reasonText(f), kind: 'stays an input with a warning' }))];
      const top = lev ? (lev.top_univariate_signals || []).slice(0, 6) : [];
      const rules = lev ? lev.heuristic_rules || {} : {};
      const strong = rules.strong_univariate_signal;
      const sig = top.length ? charts.barsH(top.map(t => ({ label: short(t.feature), value: t.signal, valueText: fmt(t.signal, 3), cls: strong != null && t.signal >= strong ? 'bad' : '' })),
        { width: 340, labelW: 112, valW: 52, min: top.every(t => t.signal >= 0.5) ? 0.5 : 0, max: 1, rowH: 24, ref: strong, refLabel: strong != null ? 'review' : '', tickFmt: v => v.toFixed(1), aria: 'Strongest single-column signals' }) : '';
      const basis = top.some(t => t.auc_basis) ? 'Categories are scored by their in-sample positive rate. ' : '';
      const moment = F.sol.prediction_moment || 'the prediction moment';
      const right = `<div class="panel" data-f="audit.verdict">
        <div class="panel-head"><h3>${open.length ? 'Open question for the owner' : 'No open question'}</h3>${open.length ? `<span class="pill warn">${esc(short(open[0].col, 20))}${open.length > 1 ? ' +' + (open.length - 1) : ''}</span>` : '<span class="pill ok">nothing waits</span>'}</div>
        <div class="panel-body small stack tight">
          ${open.map(o => `<p>${code(o.col)} · ${esc(o.why)}${/[.?!]$/.test(o.why.trim()) ? '' : '.'} If its value is written after “${esc(moment)}”, the model learns from the future. Until the owner answers, it ${o.kind}.</p>`).join('')}
          ${!open.length && lev ? `<p>No column waits on the owner. A single column is flagged for review at a signal of ${strong != null ? fmt(strong, 2) : '—'}${F.forb.length ? '; no column outside the solution\'s forbidden list got there' : '; none got there'}.</p>` : ''}
          ${top.length ? `<div class="eyebrow">Strongest single columns</div><div>${sig}</div><div class="xs muted">${basis}A strong signal alone is not a leak: a real driver is expected to be predictive ${rc('DCLAB-R05')}.</div>` : ''}
          ${!lev ? '<div class="xs muted">The single-column scan and the synthetic canary run with the leakage stage.</div>' : ''}
          ${open.length ? '<a class="btn sm" href="#solution/owner">Answer in the owner worksheet</a>' : ''}
        </div>
      </div>`;
      return `<div class="panel" data-f="audit.ablation" data-tour="ablation">
          <div class="panel-head"><div class="panel-title"><h3>How much the leak lies</h3><span class="sub">${sub}</span></div></div>
          <div class="panel-body stack tight">${left}</div>
        </div>${right}`;
    }
    function edaReal(p, F) {
      const { dev, sol } = F;
      const head = `<div class="panel-head"><div class="panel-title"><h2>Train-only look at the data</h2><span class="sub">${dev ? `Profiles of the ${int(dev.train_rows)} training rows only; the holdout's labels stay sealed.` : 'Profiles of the training rows only.'} Patterns here are questions, never promoted features.</span></div><span class="pill outline">${dev ? 'computed by the data stage' : 'not computed yet'}</span></div>`;
      if (!dev) return head + `<div class="panel-body">${runBox(p, 'data', 'Not measured for this project yet: run the data stage.')}</div>`;
      const ts = dev.target_summary || {}, p0 = v => (v * 100).toFixed(0) + '%';
      const profs = (dev.feature_profiles || []).filter(f => f.role === 'feature');
      let c1, n1;
      if (ts.positives_train != null) {
        const tot = ts.positives_train + ts.negatives_train;
        c1 = charts.barsH([
          { label: short(`${sol.target || 'target'} = ${ts.positive_label != null ? ts.positive_label : 'yes'}`, 14), value: ts.positives_train / tot, valueText: `${pct(ts.positive_rate_train)} · ${int(ts.positives_train)}`, cls: 'proof' },
          { label: 'the rest', value: ts.negatives_train / tot, valueText: `${pct(1 - ts.positive_rate_train)} · ${int(ts.negatives_train)}` },
        ], { width: 300, labelW: 98, valW: 92, min: 0, max: 1, rowH: 26, ticks: [0, 0.5, 1], tickFmt: p0, aria: 'Outcome balance in the training rows' });
        n1 = `Predicting the majority class already scores ${pct(ts.majority_baseline_accuracy_train)} accuracy, so accuracy alone cannot judge this model.`;
      } else {
        const kv = Object.entries(ts).filter(([, v]) => typeof v === 'number').slice(0, 6);
        c1 = kv.length ? `<dl class="kv">${kv.map(([k, v]) => `<dt>${esc(k.replace(/_/g, ' '))}</dt><dd class="mono">${Math.abs(v) < 10 ? fmt(v, 3) : int(Math.round(v))}</dd>`).join('')}</dl>` : '<span class="faint">No target summary recorded.</span>';
        n1 = 'Summary of the target on the training rows.';
      }
      const tops = profs.slice().sort((a, b) => b.top_value_frequency - a.top_value_frequency).slice(0, 5);
      const c2 = tops.length ? charts.barsH(tops.map(f => ({ label: short(f.feature, 14), value: f.top_value_frequency, valueText: pct(f.top_value_frequency), cls: f.top_value_frequency > 0.95 ? 'proof' : '' })),
        { width: 300, labelW: 98, valW: 52, min: 0, max: 1, rowH: 24, ticks: [0, 0.5, 1], tickFmt: p0, aria: 'Share of the most common value per column' }) : '<span class="faint">No input profiles.</span>';
      const n2 = tops.length ? `${code(tops[0].feature)} holds one value on ${pct(tops[0].top_value_frequency)} of rows.${tops[0].top_value_frequency > 0.95 ? ' A near-constant column carries little signal.' : ''}` : '';
      const psi = profs.filter(f => f.train_vs_holdout_psi != null).sort((a, b) => b.train_vs_holdout_psi - a.train_vs_holdout_psi).slice(0, 5);
      let c3, n3, t3;
      if (psi.length) {
        const mx = Math.max(0.25, psi[0].train_vs_holdout_psi * 1.1);
        t3 = 'Train vs holdout drift (PSI)';
        c3 = charts.barsH(psi.map(f => ({ label: short(f.feature, 14), value: f.train_vs_holdout_psi, valueText: fmt(f.train_vs_holdout_psi, 3), cls: f.train_vs_holdout_psi > 0.2 ? 'bad' : '' })),
          { width: 300, labelW: 98, valW: 52, min: 0, max: mx, rowH: 24, ref: 0.2, refLabel: '0.20', tickFmt: v => v.toFixed(2), aria: 'Population stability between training and holdout inputs' });
        n3 = `Largest: ${code(psi[0].feature)} at ${fmt(psi[0].train_vs_holdout_psi, 3)}. Above 0.20 is flagged. Input distributions only; no holdout labels are read.`;
      } else {
        const miss = profs.slice().sort((a, b) => b.missing_rate - a.missing_rate).slice(0, 5);
        t3 = 'Missing values by column';
        c3 = miss.length ? charts.barsH(miss.map(f => ({ label: short(f.feature, 14), value: f.missing_rate, valueText: pct(f.missing_rate, 2) })), { width: 300, labelW: 98, valW: 52, min: 0, max: Math.max(0.05, miss[0].missing_rate * 1.1), rowH: 24, tickFmt: p0, aria: 'Missing share per column' }) : '';
        n3 = 'No drift score was recorded for these inputs.';
      }
      const dup = dev.duplicate_feature_rows_train || 0, risks = Object.keys(dev.risk_features || {});
      // 25.13: the outcome per category or band of each input, training rows only, strongest spread first
      const obf = dev.outcome_by_feature || [], rateTask = ts.positives_train != null;
      const obfBox = obf.length ? `<div class="panel-body stack tight"><b class="small">The outcome by column, on the training rows</b>
          <span class="xs muted">${rateTask ? 'The share of positives' : 'The mean outcome'} in each category or quantile band of every input, strongest difference first. A pattern here is a question to check against the prediction moment, never a promoted feature; a column far stronger than the rest is a leak suspect.</span>
          <div class="grid g3">${obf.slice(0, 12).map(f => `<div class="inset stack tight"><div class="spread small"><b>${esc(f.feature)}</b><span class="pill outline">spread ${rateTask ? pct(f.spread) : fmt(f.spread, 3)}</span></div>${charts.barsH(f.groups.slice(0, 8).map(g => ({ label: short(g.value, 16), value: g.outcome, valueText: (rateTask ? pct(g.outcome) : fmt(g.outcome, 3)) + ' · ' + int(g.rows) })), { width: 300, labelW: 110, valW: 80, rowH: 20, noAxis: true, min: 0, max: Math.max(...f.groups.map(g => g.outcome)) || 1, aria: 'Outcome by ' + f.feature })}</div>`).join('')}</div>
          ${obf.length > 12 ? `<details><summary class="xs muted">${obf.length - 12} more columns</summary><div class="grid g3">${obf.slice(12).map(f => `<div class="inset small"><b>${esc(f.feature)}</b> · spread ${rateTask ? pct(f.spread) : fmt(f.spread, 3)}</div>`).join('')}</div></details>` : ''}</div>` : '';
      return head + `<div class="panel-body grid g3">
          <div class="stack tight"><b class="small">Outcome in the training rows</b><div>${c1}</div><span class="xs muted">${n1}</span></div>
          <div class="stack tight"><b class="small">Most common value per column</b><div>${c2}</div><span class="xs muted">${n2}</span></div>
          <div class="stack tight"><b class="small">${t3}</b><div>${c3}</div><span class="xs muted">${n3}</span></div>
        </div>${obfBox}
        <div class="panel-foot"><span class="pill ${dup ? 'warn' : 'ok'}">${dup ? int(dup) + ' repeated input rows' : 'no repeated input rows'}</span> Missing cells: ${pct(dev.missing_cell_rate_train, 2)} of training cells. ${risks.length ? `Review rules fired on ${risks.map(code).join(', ')}.` : 'No column triggered a review rule.'} Missing values are a state to model, not only to fill ${rc('DCLAB-R12')}; every imputer and encoder is fitted inside the training folds ${rc('DCLAB-R03')}.</div>`;
    }
    function criticReal(p, F) {
      const notes = STAGES.flatMap(s => ((F.R[s] || {}).notes || []).map(n => Object.assign({ stage: s }, n)));
      const crit = notes.filter(n => ['warning', 'high', 'critical'].includes(n.severity) || n.source === 'llm');
      const ran = STAGES.filter(s => F.R[s]).length;
      const head = `<div class="panel-head"><div class="panel-title"><h2>Critic review, checked against the numbers</h2><span class="sub">Warnings each stage wrote from its recorded numbers. An LLM critique, when configured, stays advisory.</span></div>${rc('DCLAB-R20')}</div>`;
      if (!ran) return { n: 0, html: head + `<div class="panel-body">${runBox(p, 'data', 'No stage has run yet, so there is nothing to review.')}</div>` };
      if (!crit.length) return { n: 0, html: head + `<div class="empty">No warning or critical note in this project's ${ran} completed stage${ran === 1 ? '' : 's'}. ${notes.length} information note${notes.length === 1 ? ' was' : 's were'} written; the Notebook shows them next to each stage.</div>` };
      const SEV = { critical: 'high', high: 'high', warning: 'medium' };
      return {
        n: crit.length,
        html: head + `<div class="list" id="critic-list">${crit.map(n => `<div class="list-item"><span class="sev ${n.source === 'llm' ? 'low' : SEV[n.severity] || 'info'}">${n.source === 'llm' ? 'advisory' : esc(n.severity)}</span><div class="li-main"><span class="small" data-style="color:var(--ink-2)"><b>${esc(n.title)}${/[.?!:]$/.test(n.title || '') ? '' : '.'}</b> ${DC.linkIds(n.text || '')}</span><span class="row small"><span class="pill outline">${esc(n.stage)} stage</span>${n.action ? `<span class="muted">Next: ${esc(n.action)}</span>` : ''}</span></div>${chips((n.proof || []).filter(id => DC.REC[id]).slice(0, 3))}</div>`).join('')}</div>`,
      };
    }

    function renderRealAdvisor(p, F) {
      const box = $('#aud-advisor', el);
      if (!box) return;
      const lev = F.lev || {};
      const adv = lev.leakage_advisor || null;
      if (!adv) { box.innerHTML = ''; return; }
      const forbiddenSet = new Set((F.forb || []).map(f => f.column));
      const recs = (adv.recommendations || []).filter(r => r && r.column && !forbiddenSet.has(r.column) && (r.likely_leakage || r.action === 'block' || r.action === 'ask'));
      if (!recs.length) {
        const noted = adv.likely_leakage_columns || [];
        if (noted.length && noted.every(c => forbiddenSet.has(c))) {
          box.innerHTML = `<div class="callout ok small"><span class="ic">✓</span><span>Leakage Advisor suspects (${esc(noted.join(', '))}) are excluded in the active solution contract.</span></div>`;
        } else {
          box.innerHTML = '';
        }
        return;
      }
      const depMoment = (adv.depends_on || {}).prediction_moment || '';
      const liveMoment = (F.sol || {}).prediction_moment || '';
      const stale = Boolean(depMoment && liveMoment && depMoment.trim() !== liveMoment.trim());
      box.innerHTML = recs.map(r => {
        const col = r.column;
        const cmp = advCompare[col];
        return `<div class="panel leak-adv-card">
          <div class="panel-head"><div class="panel-title">
            <div class="row" data-style="gap:6px;flex-wrap:wrap">
              <span class="pill warn">Leakage Advisor · Very likely leakage</span>
              <span class="pill outline mono">cell_values_sent: ${int(adv.cell_values_sent || 0)}</span>
              ${stale ? '<span class="pill bad">Stale · prediction moment changed (re-run leakage)</span>' : ''}
            </div>
            <h3>Column <code>${esc(col)}</code> is likely post-outcome leakage</h3>
          </div></div>
          <div class="panel-body stack tight small">
            <p class="m-0">${esc(r.rationale || r.timing_assessment || 'This column may not exist at the prediction moment.')}</p>
            ${cmp ? `<div class="leak-contrast-grid">
              <div class="leak-contrast-box is-safe"><div class="xs muted">SAFE · Without ${esc(col)}</div><div class="mono" data-style="font-size:20px;font-weight:600">${fmt(cmp.clean_cv_mean)}</div><div class="xs muted">Training CV without suspect (holdout untouched)</div></div>
              <div class="leak-contrast-box"><div class="xs bad">LEAKY · With ${esc(col)}</div><div class="mono" data-style="font-size:20px;font-weight:600">${fmt(cmp.leaky_cv_mean)}</div><div class="xs muted">${signed(cmp.apparent_lift || 0)} inflated lift on training folds</div></div>
            </div>
            <div class="row" data-style="gap:8px;margin-top:8px"><button type="button" class="btn sm primary" data-adv-col="${esc(col)}" data-adv-choice="confirm_compare">Confirm: Keep safe pipeline &amp; exclude ${esc(col)}</button></div>`
            : `<div class="row" data-style="gap:8px;flex-wrap:wrap">
              <button type="button" class="btn sm primary" data-adv-col="${esc(col)}" data-adv-choice="compare_both" ${stale ? 'disabled' : ''}>Compare both on training folds (Recommended)</button>
              <button type="button" class="btn sm" data-adv-col="${esc(col)}" data-adv-choice="exclude_directly" ${stale ? 'disabled' : ''}>Exclude directly</button>
            </div>`}
          </div>
        </div>`;
      }).join('');
    }
    function renderReal(p) {
      real = p; mode = 'real';
      const F = facts(p);
      $('#aud-head', el).innerHTML = headReal(p);
      renderRealAdvisor(p, F);
      $('#aud-stats', el).innerHTML = statsReal(p, F);
      const mx = matrixReal(p, F);
      $('#avail-table tbody', el).innerHTML = mx.rows;
      $('#aud-matrix-foot', el).innerHTML = mx.foot;
      $('#aud-leaks', el).innerHTML = leaksReal(p, F);
      $('#aud-eda', el).innerHTML = edaReal(p, F);
      const cr = criticReal(p, F);
      $('#aud-critic', el).innerHTML = cr.html;
      const n = $('.ptab[data-ptab="critic"] .n', el);
      if (n) n.textContent = String(cr.n);
      DC.hydrate(el);
      if (busy(p)) watch();
    }
    function restoreSample() {
      if (stopPoll) { stopPoll(); stopPoll = null; }
      real = null;
      if (mode === 'sample') return;
      mode = 'sample';
      SNAP.forEach(id => { $('#' + id, el).innerHTML = SAMPLE[id]; });
      const n = $('.ptab[data-ptab="critic"] .n', el);
      if (n) n.textContent = '5';
      drawSample();
    }
    function watch() {
      if (stopPoll) return;
      stopPoll = DC.poll(async () => {
        if (DC.state.view !== 'audit' || mode !== 'real') { stopPoll = null; return true; }
        const p = await DC.currentProject.get(true);
        if (!p) { stopPoll = null; return true; }
        const still = busy(p);
        if (!still) stopPoll = null;
        renderReal(p);
        return !still;
      }, 1500);
    }

    el.addEventListener('click', async e => {
      const sampleBtn = e.target.closest('[data-adv-sample], [data-adv-reset]');
      if (sampleBtn && mode === 'sample') {
        if (sampleBtn.dataset.advReset) { delete advCompare[sampleBtn.dataset.advReset]; delete advCompare._excluded; }
        else if (sampleBtn.dataset.advSample === 'compare') { advCompare.campaign = true; }
        else if (sampleBtn.dataset.advSample === 'exclude') { advCompare._excluded = { campaign: true }; }
        renderSampleAdvisor();
        return;
      }
      const advBtn = e.target.closest('[data-adv-choice]');
      if (advBtn && real) {
        const col = advBtn.dataset.advCol, ch = advBtn.dataset.advChoice;
        advBtn.disabled = true;
        try {
          if (ch === 'compare_both') {
            const res = await DC.client.decideLeakageAdvice(real.id, { body: { choice: 'compare_both', columns: [col] } });
            advCompare[col] = res.comparison || {};
            DC.toast(`Compared ${col} on training folds (holdout sealed).`);
            renderReal(real);
          } else {
            const payload = ch === 'confirm_compare'
              ? { choice: 'compare_both', columns: [col], confirm_exclude_after_compare: true }
              : { choice: 'exclude_directly', columns: [col] };
            await DC.client.decideLeakageAdvice(real.id, { body: payload });
            delete advCompare[col];
            DC.currentProject.clear();
            const fresh = await DC.currentProject.get(true);
            DC.toast(`Excluded ${col} through validated solution save path.`);
            if (fresh) renderReal(fresh);
          }
        } catch (err) {
          advBtn.disabled = false;
          DC.toast(err.message || 'Could not update leakage decision', { ok: false });
        }
        return;
      }
      const run = e.target.closest('[data-run-stage]');
      if (!run || !real) return;
      run.disabled = true;
      try { await DC.client.runStage(real.id, run.dataset.runStage); DC.toast(`Started the ${run.dataset.runStage} stage. This page updates when it finishes.`); }
      catch (err) { DC.toast(err.message, { ok: false }); }
      DC.currentProject.clear(); watch();
    });
    this.renderReal = renderReal;
    this.restoreSample = restoreSample;
  },
  async enter(el) {
    let p = null;
    try { p = await DC.currentProject.get(); } catch (e) { console.warn('audit: could not load the project', e); }
    DC.markSample(!p);
    if (p) this.renderReal(p); else this.restoreSample();
  },
});
