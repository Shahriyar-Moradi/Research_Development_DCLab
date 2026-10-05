DC.view('reliability', {
  init(el) {
    const { $, chip, icon, charts, binormal, int } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id)).replace(/\$\{icon:([a-z]+)\}/g, (m, n) => icon(n));
    const B = DEMO.bank;
    $('#holdout-ci', el).innerHTML = charts.barsH([{ label: 'holdout ROC-AUC', value: B.holdoutAuc, lo: B.holdoutLo, hi: B.holdoutHi, valueText: '0.7718', strong: true }, { label: 'training CV (C02)', value: 0.7523, valueText: '0.7523', cls: 'soft' }],
      { width: 520, labelW: 120, valW: 56, min: 0.5, max: 0.9, rowH: 26, ref: 0.5, ticks: [0.5, 0.6, 0.7, 0.8, 0.9], tickFmt: v => v.toFixed(1), aria: 'Holdout ROC-AUC with interval' });
    const m = binormal(B.holdoutAuc, B.positive);
    const rows = [0.025, 0.05, 0.10, 0.15, 0.20, 0.30].map(q => { const r = m.atFraction(q); return { q, calls: Math.round(q * 20000), prec: r.precision, rec: r.tpr }; });
    $('#op-table tbody', el).innerHTML = `<tr class="selected"><td>cut-off 0.5 (≈ top 2.9%)</td><td class="num mono">~580</td><td class="num mono">56.5%</td><td class="num mono">13.8%</td><td><span class="pill ok">measured</span></td></tr>` +
      rows.map(r => `<tr><td>top ${(r.q * 100).toFixed(r.q < 0.05 ? 1 : 0)}%</td><td class="num mono">${int(r.calls)}</td><td class="num mono">${(r.prec * 100).toFixed(1)}%</td><td class="num mono">${(r.rec * 100).toFixed(1)}%</td><td>${r.q === 0.10 ? '<span class="pill accent">capacity</span>' : '<span class="pill outline">estimated</span>'}</td></tr>`).join('');
    const calib = [[0.02, 0.025], [0.06, 0.055], [0.10, 0.09], [0.15, 0.17], [0.22, 0.25], [0.32, 0.36], [0.45, 0.43], [0.6, 0.55], [0.75, 0.66]];
    $('#calib-chart', el).innerHTML = charts.line([{ pts: calib, dots: true }], { width: 360, height: 220, xMin: 0, xMax: 0.8, yMin: 0, yMax: 0.8, diag: true, xLabel: 'predicted probability', yLabel: 'observed rate', xFmt: v => v.toFixed(1), yFmt: v => v.toFixed(1), xTicks: [0, 0.2, 0.4, 0.6, 0.8], yTicks: [0, 0.2, 0.4, 0.6, 0.8], aria: 'Reliability curve' });
    $('#slice-chart', el).innerHTML = charts.barsH([
      { label: 'contact: cellular', value: 0.76, lo: 0.69, hi: 0.83, valueText: '0.76 · ~520 rows' },
      { label: 'contact: unknown', value: 0.71, lo: 0.58, hi: 0.83, valueText: '0.71 · ~230 rows' },
      { label: 'contact: telephone', value: 0.74, lo: 0.52, hi: 0.93, valueText: '0.74 · ~50 rows', cls: 'soft' },
      { label: 'age 61+', value: 0.70, lo: 0.42, hi: 0.94, valueText: 'not enough rows', cls: 'muted' },
      { label: 'past success', value: 0.66, lo: 0.40, hi: 0.90, valueText: 'not enough rows', cls: 'muted' },
    ], { width: 380, labelW: 120, valW: 104, min: 0.4, max: 1.0, rowH: 26, ref: 0.7718, ticks: [0.4, 0.6, 0.8, 1.0], tickFmt: v => v.toFixed(1), aria: 'ROC-AUC by slice' });
    $('#seal-preview', el).addEventListener('click', () => DC.modal.open({
      eyebrow: '<span class="eyebrow">WF-09 · holdout</span>', title: 'Open the holdout, once', confirm: 'Already used for this project', confirmDisabled: true,
      html: `<p>This reads 800 rows nobody has used. The number you get is final for this project: it cannot be used to choose or tune anything afterwards ${chip('PIT-006')}.</p>
        <ul class="plan-list"><li class="done"><span class="pi">✓</span><span>Contract v2 is signed</span></li><li class="done"><span class="pi">✓</span><span>Feature recipe locked: ratios</span></li><li class="done"><span class="pi">✓</span><span>Model locked: extra_trees C02</span></li><li class="done"><span class="pi">✓</span><span>Tuning decided on training folds only</span></li><li class="done"><span class="pi">✓</span><span>No earlier read of these rows</span></li></ul>
        <div class="field"><label for="seal-type">Type the project name to confirm</label><input id="seal-type" type="text" value="Term-deposit calls" disabled></div>
        <div class="callout info"><span class="ic">${icon('info')}</span><span>In this demo the holdout was opened on Oct 2. The dialog is shown read-only.</span></div>`,
    }));
    DC.hydrate(el);
  },
});
