DC.view('brief', {
  init(el) {
    const { $, chip, charts, binormal, int } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id));
    const B = DEMO.bank, pool = 20000;
    const M = [binormal(B.holdoutLo, B.positive), binormal(B.holdoutAuc, B.positive), binormal(B.holdoutHi, B.positive)];
    const out = (m, calls, cost, val) => { const r = m.atFraction(calls / pool); const subs = calls * r.precision; return { subs, net: subs * val - calls * cost, prec: r.precision }; };
    function draw() {
      const calls = +$('#v-calls', el).value, cost = Math.max(1, +$('#v-cost', el).value || 8), val = Math.max(1, +$('#v-val', el).value || 110);
      $('#v-calls-out', el).textContent = int(calls);
      const [lo, mid, hi] = M.map(m => out(m, calls, cost, val));
      const rnd = { subs: calls * B.positive, net: calls * B.positive * val - calls * cost };
      $('#v-kpis', el).innerHTML = `
        <div class="kpi"><span class="label">Subscriptions</span><span class="value">${int(Math.round(mid.subs))}</span><span class="delta">likely ${int(Math.round(lo.subs))}–${int(Math.round(hi.subs))} · random ${int(Math.round(rnd.subs))}</span></div>
        <div class="kpi"><span class="label">Hit rate</span><span class="value">${(mid.prec * 100).toFixed(0)}%</span><span class="delta">break-even ${(cost / val * 100).toFixed(1)}% · random 11.7%</span></div>
        <div class="kpi"><span class="label">Net value</span><span class="value">€${int(Math.round(mid.net / 100) * 100)}</span><span class="delta ${mid.net > rnd.net ? 'up' : 'down'}">random €${int(Math.round(rnd.net / 100) * 100)}</span></div>`;
      const xs = Array.from({ length: 41 }, (_, i) => 250 + i * 243.75);
      const f = (m) => xs.map(c => [c, out(m, c, cost, val).net / 1000]);
      const loPts = f(M[0]), hiPts = f(M[2]);
      $('#v-chart', el).innerHTML = charts.line([
        { pts: xs.map(c => [c, (c * B.positive * val - c * cost) / 1000]), cls: 's3' }, { pts: f(M[1]) },
      ], { width: 480, height: 240, xMin: 0, xMax: 10000, range: [loPts, hiPts], yMax: Math.max(...hiPts.map(p => p[1])) * 1.05, yMin: Math.min(0, ...loPts.map(p => p[1])), xLabel: 'calls next month', yLabel: 'net value (€ thousand)', xFmt: v => v ? (v / 1000) + 'k' : '0', yFmt: v => Math.round(v), markers: [{ x: calls, label: int(calls) + ' calls' }], aria: 'Net value by call volume' });
    }
    ['#v-calls', '#v-cost', '#v-val'].forEach(s => $(s, el).addEventListener('input', draw));
    draw();
    $('#brief-lang', el).addEventListener('segchange', e => { $('#brief-en', el).hidden = e.detail !== 'en'; $('#brief-fa', el).hidden = e.detail !== 'fa'; });
    $('#so-approve', el).addEventListener('click', () => DC.modal.open({
      eyebrow: '<span class="eyebrow">Sign-off</span>', title: 'Approve the pilot?', confirm: 'Approve the pilot',
      html: '<p>This approves one campaign with a random control group of 200 clients. It does not approve production use: the production gate still has four open items.</p><label class="check"><input type="checkbox" checked> Notify Ava and Shahriyar</label><label class="check"><input type="checkbox" checked> Schedule the comparison report after the campaign</label>',
      onConfirm: () => DC.toast('Pilot approved. WF-10 is complete; the project moves to "pilot running".'),
    }));
  },
});
