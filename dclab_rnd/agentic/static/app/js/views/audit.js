DC.view('audit', {
  init(el) {
    const { $, esc, chip, chips, charts, icon } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id));
    const COLS = [
      ['age', 'client profile', 'yes', '', 'allowed'], ['job', 'client profile', 'yes', '', 'allowed'], ['marital', 'client profile', 'yes', '', 'allowed'], ['education', 'client profile', 'yes', '', 'allowed'],
      ['default', 'account state', 'yes', '', 'allowed'], ['balance', 'account state', 'yes', '', 'allowed'], ['housing', 'account state', 'yes', '', 'allowed'], ['loan', 'account state', 'yes', '', 'allowed'],
      ['contact', 'call scheduled', 'yes', '', 'allowed'], ['day_of_week', 'call scheduled (day of month)', 'yes', '', 'allowed'], ['month', 'call scheduled', 'yes', '', 'allowed'],
      ['pdays', 'earlier campaigns · −1 means never', 'yes', '', 'allowed'], ['previous', 'earlier campaigns', 'yes', '', 'allowed'], ['poutcome', 'earlier campaigns', 'yes', '', 'allowed'],
      ['campaign', 'during this campaign', 'ask', '', 'ask owner'], ['duration', 'when the call ends', 'no', 'name · AUC 0.812 · declared', 'forbidden'],
    ];
    const known = { yes: '<span class="yes">✓ yes</span>', no: '<span class="pill bad">no · after</span>', ask: '<span class="pill warn">unclear</span>' };
    const verdict = v => v === 'forbidden' ? '<span class="pill bad">forbidden</span>' : v === 'ask owner' ? '<span class="pill warn">ask owner</span>' : `<span class="pill ok">${esc(v)}</span>`;
    $('#avail-table tbody', el).innerHTML = COLS.map(([c, w, k, f, v]) => `<tr class="${v === 'forbidden' ? 'selected' : ''}"><td><code>${c}</code></td><td class="small">${esc(w)}</td><td>${known[k]}</td><td class="small">${f ? `<span class="pill warn">${esc(f)}</span> ${chip('LEAK-bank_marketing')}` : '<span class="faint">none</span>'}</td><td>${verdict(v)}</td></tr>`).join('');

    $('#ablation-chart', el).innerHTML = charts.barsH([
      { label: 'Safe (3,200 rows)', value: 0.7130, valueText: '0.7130', cls: '' },
      { label: 'With duration', value: 0.8972, valueText: '0.8972', cls: 'bad' },
      { label: 'Safe (full rows)', value: 0.8029, valueText: '0.8029', cls: '' },
      { label: 'With duration', value: 0.9357, valueText: '0.9357', cls: 'bad' },
    ], { width: 340, labelW: 112, valW: 52, min: 0.5, max: 1.0, rowH: 28, ticks: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], tickFmt: v => v.toFixed(1), aria: 'ROC-AUC with and without duration' });

    const CRIT = [
      ['EXP-007', 'high', 'Calling duration a decision-time exclusion is plausible but not proven here; the evidence only shows a declared exclusion and a CV lift.', 'kept', 'Agrees with the numbers. Resolved by the solution: the call length is written when the call ends (timeline on the solution page).'],
      ['EXP-007', 'high', 'The +0.1842 AUC lift is not sufficient to label a feature as leakage or proxy; predictive lift can arise from legitimate signal.', 'kept', 'Agrees with the numbers and with DCLAB-R06. The lift is shown as severity only.'],
      ['EXP-007', 'medium', 'The canary detection only shows the detector can catch a synthetic canary. It does not validate detection of real leakage columns.', 'kept', 'Fair. The blind auditor replay measures it: 11 of 18 known leaks found across 7 datasets.'],
      ['EXP-008', 'high', 'The selection rule is not satisfied because the selected stage mean ROC-AUC is 0.7029, not within 0.002 of the best stage mean 0.7172.', 'dropped', 'Contradicted by the numbers. The chosen recipe is ratios (0.7172, the best mean). 0.7029 belongs to a different recipe that happens to be named "selected".'],
      ['EXP-008', 'high', 'Higher training-CV ROC-AUC is treated as sufficient evidence; the improvement may be within sampling noise.', 'kept', 'Agrees with the numbers: +0.0042 against a fold spread of ±0.0129 to ±0.0165. Next test: 5×2 repeated CV before trusting the ladder.'],
    ];
    $('#critic-list', el).innerHTML = CRIT.map(([ev, sev, txt, gate, why]) => `<div class="list-item"><span class="sev ${sev}">${sev}</span><div class="li-main"><span class="small" data-style="color:var(--ink-2)">“${esc(txt)}”</span><span class="row small"><span class="pill ${gate === 'kept' ? 'ok' : 'bad'}">${gate === 'kept' ? 'kept' : 'dropped by the gate'}</span><span class="muted">${DC.linkIds(why)}</span></span></div>${chip(ev)}</div>`).join('');

    const pct = v => (v * 100).toFixed(0) + '%';
    $('#eda-month', el).innerHTML = charts.vbars([
      ['J', 0.102], ['F', 0.168], ['M', 0.509], ['A', 0.196], ['M', 0.067], ['J', 0.104], ['J', 0.093], ['A', 0.109], ['S', 0.452], ['O', 0.438], ['N', 0.101], ['D', 0.472],
    ].map(([l, v]) => ({ label: l, value: v, cls: v > 0.3 ? 'proof' : '' })), { width: 300, height: 140, max: 0.6, yFmt: pct, aria: 'Subscription rate by month' });
    $('#eda-pout', el).innerHTML = charts.barsH([
      { label: 'never contacted', value: 0.092, valueText: '9.2% · 29,589' }, { label: 'failure', value: 0.123, valueText: '12.3% · 3,889' },
      { label: 'other', value: 0.165, valueText: '16.5% · 1,485' }, { label: 'success', value: 0.647, valueText: '64.7% · 1,205', cls: 'proof' },
    ], { width: 300, labelW: 98, valW: 92, min: 0, max: 0.7, rowH: 26, ticks: [0, 0.35, 0.7], tickFmt: pct, aria: 'Subscription rate by previous outcome' });
    $('#eda-age', el).innerHTML = charts.barsH([
      { label: '18–30', value: 0.164, valueText: '16.4%' }, { label: '31–40', value: 0.102, valueText: '10.2%' }, { label: '41–50', value: 0.091, valueText: '9.1%' },
      { label: '51–60', value: 0.099, valueText: '9.9%' }, { label: '61+', value: 0.421, valueText: '42.1%', cls: 'proof' },
    ], { width: 300, labelW: 60, valW: 52, min: 0, max: 0.5, rowH: 24, ticks: [0, 0.25, 0.5], tickFmt: pct, aria: 'Subscription rate by age band' });

    const LADDER = [['raw', 15, 0.7130, 0.0129], ['logs', 23, 0.7130, 0.0129], ['ratios', 38, 0.7172, 0.0165], ['interactions', 46, 0.7136, 0.0169], ['full_fe', 51, 0.7130, 0.0216], ['selected', 25, 0.7029, 0.0106]];
    function drawLadder() {
      const tol = Number($('#tol', el).value);
      $('#tol-val', el).textContent = tol.toFixed(4);
      const best = Math.max(...LADDER.map(r => r[2]));
      const ok = LADDER.filter(r => r[2] >= best - tol - 1e-9);
      const pick = ok.reduce((a, b) => (b[1] < a[1] ? b : a));
      $('#ladder-table tbody', el).innerHTML = LADDER.map(r => `<tr class="${r === pick ? 'selected' : ''}"><td><span class="cell-main">${r[0]}</span>${r === pick ? ' <span class="pill accent">chosen</span>' : ''}</td><td class="num">${r[1]}</td><td class="num mono">${r[2].toFixed(4)}</td><td class="num mono">±${r[3].toFixed(4)}</td><td>${ok.includes(r) ? '<span class="yes">✓</span>' : '<span class="no">—</span>'}</td></tr>`).join('');
      const declared = Math.abs(tol - 0.002) < 1e-9;
      $('#tol-out', el).innerHTML = `<div class="callout ${declared ? 'info' : 'warn'}"><span class="ic">${icon(declared ? 'info' : 'alert')}</span><span>${declared
        ? `With the declared 0.0020 the rule picks <b>ratios</b> (38 features, +0.0042 over raw), which is what the campaign recorded. The gap is smaller than the fold spread, so the critic's call for repeated CV stands.`
        : `With ${tol.toFixed(4)} the rule would pick <b>${pick[0]}</b> (${pick[1]} features). Changing the tolerance after seeing the results is logged as a protocol change and needs a reason ${chip('DCLAB-R07')}.`}</span></div>`;
      DC.hydrate(el);
    }
    $('#tol', el).addEventListener('input', drawLadder);
    drawLadder();

    const LIN = [
      ['poutcome', 'outcome of the previous campaign', 'known before this campaign', 'ok', 'Category code · fine for trees'],
      ['log1p_poutcome', 'log of the poutcome code', 'known before', 'bad', 'A log of a category code has no meaning. Restore the raw labels and one-hot or target-encode inside folds.'],
      ['month', 'month of the scheduled call', 'known at scheduling', 'warn', 'Strong seasonality on few rows; no year, so drift cannot be tested.'],
      ['log1p_month', 'log of the month code', 'known at scheduling', 'bad', 'Same problem: the codes follow first appearance in the file, not the calendar.'],
      ['housing', 'housing loan', 'known at the snapshot', 'ok', 'Binary state · fine'],
    ];
    $('#lineage-list', el).innerHTML = LIN.map(([n, src, when, st, note]) => `<div class="list-item"><div class="li-main"><span class="li-title"><code>${n}</code></span><span class="li-sub">${esc(src)} · ${esc(when)}</span><span class="xs" data-style="color:var(--${st === 'ok' ? 'muted' : st})">${esc(note)}</span></div><span class="pill ${st}">${st === 'ok' ? 'clean' : st === 'bad' ? 'fix' : 'watch'}</span></div>`).join('') +
      `<div class="list-item"><div class="li-main"><span class="xs muted">Top features come from the final model ${chip('EXP-010')}. Importance only nominates; a feature is promoted when its availability, stability and meaning check out ${chip('DCLAB-R10')} ${chip('DCLAB-R11')}.</span></div></div>`;
    DC.hydrate(el);
  },
});
