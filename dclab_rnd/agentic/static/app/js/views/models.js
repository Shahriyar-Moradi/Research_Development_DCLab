DC.view('models', {
  init(el) {
    const { $, esc, chip, icon, charts, toast } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id)).replace(/\$\{icon:([a-z]+)\}/g, (m, n) => icon(n));
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
    $('#kpen', el).addEventListener('input', draw);
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
    $('#override-btn', el).addEventListener('click', () => DC.modal.open({
      eyebrow: '<span class="eyebrow">Override the rule</span>', title: 'Choose a different model', confirm: 'Log the override',
      html: `<p>The rule picked <b>extra_trees</b>. You can choose another candidate; the log records your reason, and the brief says the choice was made by a person.</p>
        <div class="field"><label for="ov-model">Model</label><select id="ov-model"><option>lightgbm · 0.7172 ± 0.0165 · faster scoring</option><option>logistic_regression · 0.7189 ± 0.0301 · easiest to explain</option></select></div>
        <div class="field"><label for="ov-why">Reason (required)</label><textarea id="ov-why">The call centre wants reasons per client; a linear model is easier to explain and is within the fold spread of the leader.</textarea></div>`,
      onConfirm: m => { if (!$('#ov-why', m).value.trim()) { toast('Write a reason first.', { ok: false }); return false; } toast('Override logged. The holdout is already used, so the new model needs fresh data or a new split for its estimate. The brief will say so.'); },
    }));
  },
});
