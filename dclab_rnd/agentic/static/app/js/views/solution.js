DC.view('solution', {
  init(el) {
    const { $, chip, icon, charts, binormal, int } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id)).replace(/\$\{icon:([a-z]+)\}/g, (m, n) => icon(n));
    const B = DEMO.bank;
    const models = [binormal(B.holdoutLo, B.positive), binormal(B.holdoutAuc, B.positive), binormal(B.holdoutHi, B.positive)];
    function value(model, calls, pool, cost, val) { if (calls <= 0) return { net: 0, subs: 0, prec: 0 }; const r = model.atFraction(Math.min(1, calls / pool)); const subs = calls * r.precision; return { net: subs * val - calls * cost, subs, prec: r.precision }; }
    function draw() {
      const cost = Math.max(0.5, +$('#c-cost', el).value || 8), val = Math.max(1, +$('#c-value', el).value || 110);
      const pool = Math.max(1000, +$('#c-pool', el).value || 20000), cap = Math.max(1, Math.min(pool, +$('#c-cap', el).value || 2000));
      const steps = 40, xs = Array.from({ length: steps + 1 }, (_, i) => Math.round(pool * i / steps));
      const mid = xs.map(c => [c, value(models[1], c, pool, cost, val).net / 1000]);
      const lo = xs.map(c => [c, value(models[0], c, pool, cost, val).net / 1000]);
      const hi = xs.map(c => [c, value(models[2], c, pool, cost, val).net / 1000]);
      const rnd = xs.map(c => [c, (c * B.positive * val - c * cost) / 1000]);
      let best = xs[0], bestNet = -Infinity;
      for (let c = 0; c <= pool; c += Math.max(50, Math.round(pool / 400))) { const n = value(models[1], c, pool, cost, val).net; if (n > bestNet) { bestNet = n; best = c; } }
      const atCap = value(models[1], cap, pool, cost, val), atCapLo = value(models[0], cap, pool, cost, val), atCapHi = value(models[2], cap, pool, cost, val);
      const breakeven = cost / val;
      const ys = mid.concat(lo, hi, rnd).map(p => p[1]);
      $('#c-chart', el).innerHTML = charts.line([
        { pts: rnd, cls: 's3' }, { pts: mid },
      ], { width: 520, height: 230, range: [lo, hi], yMin: Math.min(0, ...ys), yMax: Math.max(...ys) * 1.05, xLabel: 'calls made', yLabel: 'net value (€ thousand)', xFmt: v => v >= 1000 ? (v / 1000) + 'k' : v, yFmt: v => Math.round(v), markers: [{ x: cap, label: 'capacity ' + int(cap) }], points: [{ x: best, y: bestNet / 1000, label: 'best ' + int(best) }] });
      $('#c-out', el).innerHTML = `
        <div class="spread"><span>Break-even hit rate</span><b class="mono">${(breakeven * 100).toFixed(1)}%</b></div>
        <div class="spread"><span>Calling at random reaches</span><b class="mono">${(B.positive * 100).toFixed(1)}% subscribers</b></div>
        <div class="spread"><span>At capacity (${int(cap)} calls), hit rate</span><b class="mono">${(atCap.prec * 100).toFixed(1)}% <span class="muted">[${(atCapLo.prec * 100).toFixed(0)}–${(atCapHi.prec * 100).toFixed(0)}%]</span></b></div>
        <div class="spread"><span>Expected subscriptions</span><b class="mono">${int(Math.round(atCap.subs))} <span class="muted">[${int(Math.round(atCapLo.subs))}–${int(Math.round(atCapHi.subs))}]</span></b></div>
        <div class="spread"><span>Expected net value</span><b class="mono">€${int(Math.round(atCap.net))}</b></div>
        <div class="spread"><span>Volume with the best value</span><b class="mono">${int(best)} calls</b></div>
        <div class="${best > cap ? 'callout info' : 'callout ok'}" data-style="margin-top:4px"><span class="ic">${icon('info')}</span><span>${best > cap ? `Capacity is the limit: more calls would still pay while the hit rate stays above ${(breakeven * 100).toFixed(1)}%. Recommendation: <b>call the top ${int(cap)}</b>.` : `Recommendation: <b>call the top ${int(best)}</b>; beyond that a call costs more than it earns.`}</span></div>`;
    }
    ['#c-cost', '#c-value', '#c-pool', '#c-cap'].forEach(s => $(s, el).addEventListener('input', draw));
    draw();
    $('#split-chart', el).innerHTML = charts.barsH([
      { label: 'Source rows', value: 45211, valueText: '45,211', cls: 'muted' },
      { label: 'Analyzed (quick)', value: 4000, valueText: '4,000' },
      { label: 'Training · 3 folds', value: 3200, valueText: '3,200', cls: 'soft' },
      { label: 'Holdout · sealed', value: 800, valueText: '800', cls: 'proof' },
    ], { width: 420, labelW: 128, valW: 60, min: 0, max: 46000, rowH: 26, ticks: [0, 15000, 30000, 45000], tickFmt: v => v ? (v / 1000) + 'k' : '0', aria: 'Rows in each split' });
    $('#solution-history', el).addEventListener('click', () => DC.modal.open({
      eyebrow: '<span class="eyebrow">Solution</span>', title: 'Version history', hideConfirm: true, cancel: 'Close',
      html: `<div class="timeline">
        <div class="tl-item"><span class="tl-mark ok">2</span><div class="tl-body"><span class="tl-title">v2 · signed by Shahriyar · Sep 27 09:21</span><span class="tl-meta">+ capacity 2,000 calls · + €8 per call · + €110 per subscription · metric changed from ROC-AUC to net value at capacity</span></div></div>
        <div class="tl-item"><span class="tl-mark">1</span><div class="tl-body"><span class="tl-title">v1 · drafted by the intern · Sep 27 09:12</span><span class="tl-meta">from the column audit and the R&amp;D's solution for bank_marketing · duration forbidden · campaign flagged for the owner</span></div></div>
      </div><p class="small muted">Each version is immutable. Results always say which solution version they were produced under.</p>`,
    }));
  },
});
