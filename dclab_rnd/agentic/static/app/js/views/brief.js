DC.view('brief', {
  init(el) {
    const { $, $$, esc, chip, chips, icon, charts, binormal, int } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id));

    /* ---------- the value calculator: one draw function, fed by the sample or by the open project ---------- */
    const B = DEMO.bank;
    const SAMPLE_CALC = { key: 'sample', pool: 20000, positive: B.positive, auc: [B.holdoutLo, B.holdoutAuc, B.holdoutHi], min: 250, max: 10000, step: 250, def: 2000, cost: 8, val: 110, cur: '€' };
    let calc = SAMPLE_CALC, M = [];
    const out = (m, calls, cost, val) => { const r = m.atFraction(calls / calc.pool); const subs = calls * r.precision; return { subs, net: subs * val - calls * cost, prec: r.precision }; };
    const money = v => (calc.cur === '€' ? '€' : '') + int(Math.round(v / 100) * 100);
    function draw() {
      if (!M.length) return;
      const calls = +$('#v-calls', el).value, cost = Math.max(0.01, +$('#v-cost', el).value || calc.cost), val = Math.max(0.01, +$('#v-val', el).value || calc.val);
      $('#v-calls-out', el).textContent = int(calls);
      const [lo, mid, hi] = M.map(m => out(m, calls, cost, val));
      const rnd = { subs: calls * calc.positive, net: calls * calc.positive * val - calls * cost };
      const real = calc !== SAMPLE_CALC;
      $('#v-kpis', el).innerHTML = `
        <div class="kpi"><span class="label">${real ? 'Positives found' : 'Subscriptions'}</span><span class="value">${int(Math.round(mid.subs))}</span><span class="delta">likely ${int(Math.round(lo.subs))}–${int(Math.round(hi.subs))} · random ${int(Math.round(rnd.subs))}</span></div>
        <div class="kpi"><span class="label">Hit rate</span><span class="value">${(mid.prec * 100).toFixed(0)}%</span><span class="delta">break-even ${(cost / val * 100).toFixed(1)}% · random ${(calc.positive * 100).toFixed(1)}%</span></div>
        <div class="kpi"><span class="label">Net value</span><span class="value">${money(mid.net)}</span><span class="delta ${mid.net > rnd.net ? 'up' : 'down'}">random ${money(rnd.net)}</span></div>`;
      const xs = Array.from({ length: 41 }, (_, i) => calc.min + i * (calc.max - calc.min) / 40);
      const f = (m) => xs.map(c => [c, out(m, c, cost, val).net / 1000]);
      const loPts = f(M[0]), hiPts = f(M[2]);
      $('#v-chart', el).innerHTML = charts.line([
        { pts: xs.map(c => [c, (c * calc.positive * val - c * cost) / 1000]), cls: 's3' }, { pts: f(M[1]) },
      ], { width: 480, height: 240, xMin: 0, xMax: calc.max, range: [loPts, hiPts], yMax: Math.max(0.001, ...hiPts.map(p => p[1])) * 1.05, yMin: Math.min(0, ...loPts.map(p => p[1])), xLabel: real ? 'cases acted on' : 'calls next month', yLabel: real ? 'net value (thousands)' : 'net value (€ thousand)', xFmt: v => v >= 1000 ? (v / 1000) + 'k' : String(Math.round(v)), yFmt: v => Math.round(v * 10) / 10, markers: [{ x: calls, label: int(calls) + (real ? ' cases' : ' calls') }], aria: 'Net value by volume' });
    }
    function setCalc(c) {
      const changed = !calc || calc.key !== c.key;
      calc = c;
      M = c.auc.map(a => binormal(a, c.positive));
      const r = $('#v-calls', el);
      if (changed) {
        r.min = c.min; r.max = c.max; r.step = c.step; r.value = c.def;
        $('#v-cost', el).value = c.cost; $('#v-val', el).value = c.val;
      }
      draw();
    }
    ['#v-calls', '#v-cost', '#v-val'].forEach(s => $(s, el).addEventListener('input', draw));
    setCalc(SAMPLE_CALC);
    $('#brief-lang', el).addEventListener('segchange', e => { $('#brief-en', el).hidden = e.detail !== 'en'; $('#brief-fa', el).hidden = e.detail !== 'fa'; });
    $('#so-approve', el).addEventListener('click', () => DC.modal.open({
      eyebrow: '<span class="eyebrow">Sign-off</span>', title: 'Approve the pilot?', confirm: 'Approve the pilot',
      html: '<p>This approves one campaign with a random control group of 200 clients. It does not approve production use: the production gate still has four open items.</p><label class="check"><input type="checkbox" checked> Notify Ava and Shahriyar</label><label class="check"><input type="checkbox" checked> Schedule the comparison report after the campaign</label>',
      onConfirm: () => DC.toast('Pilot approved. WF-10 is complete; the project moves to "pilot running".'),
    }));

    /* ---------- sample mode keeps the original markup; real mode rewrites the same panels ---------- */
    const KEEP = ['brief-eyebrow', 'brief-status', 'brief-stats', 'brief-version', 'brief-en', 'brief-fa', 'brief-foot', 'brief-reading', 'v-title', 'v-sub', 'v-calls-label', 'v-cost-label', 'v-val-label', 'v-legend-rnd', 'brief-risks', 'so-pill', 'cap-sub', 'cap-list'];
    const SAMPLE = {};
    KEEP.forEach(id => { const n = $('#' + id, el); SAMPLE[id] = { html: n.innerHTML, cls: n.className }; });
    const SAMPLE_COPY = $('#brief-copy', el).dataset.copy;
    const set = (id, html, cls) => { const n = $('#' + id, el); n.innerHTML = html; if (cls != null) n.className = cls; };
    const riskCount = n => { const b = $('.ptab[data-ptab="risks"] .n', el); if (b) b.textContent = String(n); };

    function showSample() {
      KEEP.forEach(id => set(id, SAMPLE[id].html, SAMPLE[id].cls));
      $('#brief-explain-panel', el).hidden = true;
      $('#brief-copy', el).dataset.copy = SAMPLE_COPY;
      $('#brief-export-sample', el).hidden = false; $('#brief-export-real', el).hidden = true;
      $('#v-body', el).hidden = false; $('#v-empty', el).hidden = true; $('#v-hint', el).hidden = true; $('#risks-sub', el).hidden = true;
      $('#so-sample', el).hidden = false; $('#so-real', el).hidden = true;
      riskCount(5);
      setCalc(SAMPLE_CALC);
      DC.hydrate(el);
    }

    /* ---------- real data ---------- */
    const STAGES = ['data', 'leakage', 'features', 'models', 'final'];
    const VIEW_OF = { data: 'solution', leakage: 'audit', features: 'audit', models: 'models', final: 'reliability' };
    const LOWER = new Set(['mae', 'rmse', 'brier', 'log_loss']);
    const UNIT = new Set(['roc_auc', 'average_precision', 'macro_f1', 'accuracy', 'f1', 'precision', 'recall', 'r2']);
    const ML = { roc_auc: 'ROC-AUC', average_precision: 'average precision', macro_f1: 'macro-F1', mae: 'MAE', rmse: 'RMSE', r2: 'R²', accuracy: 'accuracy' };
    const ml = m => ML[m] || String(m || 'score').replace(/_/g, ' ');
    const FA = '۰۱۲۳۴۵۶۷۸۹';
    const fa = s => String(s).replace(/(\d)\.(?=\d)/g, '$1٫').replace(/(\d),(?=\d)/g, '$1٬').replace(/\d/g, d => FA[d]);
    const num = (v, m) => v == null || isNaN(v) ? '—' : UNIT.has(m) ? Number(v).toFixed(2) : Math.abs(v) >= 100 ? int(Math.round(v)) : Math.abs(v) >= 1 ? Number(v).toFixed(2) : Number(v).toFixed(3);
    const gapNum = (v, m) => UNIT.has(m) ? Math.abs(v).toFixed(3) : num(Math.abs(v), m);
    const code = s => `<code dir="ltr">${esc(s)}</code>`;
    const quote = s => `<bdi>${esc(s)}</bdi>`;
    const known = ids => (ids || []).filter((x, i, a) => DC.REC[x] && a.indexOf(x) === i);
    const rich = s => DC.linkIds(s).replace(/`([^`]+)`/g, '<code>$1</code>');
    const day = iso => { try { return new Date(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }); } catch (e) { return String(iso || '').slice(0, 10); } };
    const niceStep = x => { const mag = Math.pow(10, Math.floor(Math.log10(Math.max(1, x)))); const n = x / mag; return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag; };

    function facts(p) {
      const R = p.records || {}, sol = p.solution || {}, data = p.data || {};
      const fin = R.final && R.final.evidence && R.final.evidence.holdout_metrics ? R.final : null;
      const ev = fin ? fin.evidence : {};
      const metric = (fin || R.models || R.features || R.leakage || R.data || {}).primary_metric || sol.metric || (sol.task === 'regression' ? 'mae' : 'roc_auc');
      const ci = ev.holdout_primary_metric_ci || null;
      const point = fin ? (ev.holdout_metrics[metric] != null ? ev.holdout_metrics[metric] : ci && ci.point) : null;
      const dEv = (R.data && R.data.evidence) || {}, lEv = (R.leakage && R.leakage.evidence) || null;
      const gate = ((p.graph && p.graph.gates) || []).find(g => g.gate === 'solution');
      const signed = gate ? !!gate.approved : !!p.signoff;
      const meta = (p.stage_meta || []).reduce((o, s) => (o[s.key] = s, o), {});
      const done = STAGES.filter(s => R[s]);
      const missing = STAGES.find(s => !R[s]) || null;
      return {
        p, R, sol, data, fin, ev, metric, ci, point, dEv, lEv, gate, signed, meta, done, missing, task: sol.task || 'binary',
        sampling: dEv.sampling || {}, source: dEv.source_rows || data.rows, quick: !!(p.settings && p.settings.quick),
        synthetic: !!(data.synthetic), usable: dEv.usable_feature_count, raw: dEv.raw_feature_count != null ? dEv.raw_feature_count : Math.max(0, (data.columns || []).length - 1),
        running: !!p.running || STAGES.some(s => ['queued', 'running'].includes(((p.stages || {})[s] || {}).status)),
      };
    }
    function calcFor(F) {
      if (!F.fin || F.task !== 'binary' || F.ev.holdout_metrics.roc_auc == null) return null;
      const auc = F.ev.holdout_metrics.roc_auc, ci = F.metric === 'roc_auc' ? F.ci : null;
      const positive = F.ev.holdout_metrics.positive_rate != null ? F.ev.holdout_metrics.positive_rate : (F.dEv.target_summary || {}).positive_rate_train;
      if (positive == null || !(positive > 0 && positive < 1)) return null;
      const pool = Math.max(20, Math.round(F.source || 0) || 1000), step = niceStep(pool / 40);
      const min = step, max = Math.max(step * 2, Math.floor(pool / step) * step);
      const def = Math.min(max, Math.max(min, Math.round(pool * Math.min(0.5, positive) / step) * step));
      const clamp = a => Math.min(0.995, Math.max(0.505, a));
      return { key: F.p.id + ':' + F.fin.completed_at, pool, positive, auc: (ci ? [ci.low, auc, ci.high] : [auc, auc, auc]).map(clamp), hasCi: !!ci, min, max, step, def, cost: 10, val: 100, cur: '' };
    }
    /* What each step needs, so the brief can say the next step in both languages. */
    function nextSteps(F) {
      const en = [], fa_ = [];
      if (F.quick && F.sampling.rows_used && F.source && F.sampling.rows_used < F.source) { en.push(`rerun the stages without quick mode on all ${int(F.source)} rows`); fa_.push(`مراحل را بدون حالت سریع روی همه‌ی ${fa(int(F.source))} ردیف دوباره اجرا کنیم`); }
      if (!F.signed) { en.push('the owner signs the solution'); fa_.push('مالک راه‌حل را امضا کند'); }
      en.push('test the model in a small pilot with a random control group before any production use');
      fa_.push('پیش از هر استفاده‌ی واقعی، مدل را در یک پایلوت کوچک با یک گروه کنترل تصادفی بیازماییم');
      const joinEn = en.map((s, i) => (i === en.length - 1 && i ? 'then ' : '') + s).join('; ');
      const joinFa = fa_.map((s, i) => (i === fa_.length - 1 && i ? 'سپس ' : '') + s).join('؛ ');
      return { en: joinEn.charAt(0).toUpperCase() + joinEn.slice(1) + '.', fa: joinFa + '.', short: en[0] };
    }

    function stats(F, C) {
      const p = F.p, m = F.metric;
      if (!F.fin) {
        const best = F.R.models ? { v: F.R.models.evidence.selected_metric_mean, at: 'models stage' } : F.R.features ? { v: F.R.features.evidence.selected_metric_mean, at: 'features stage' } : null;
        const next = F.missing ? (F.meta[F.missing] || {}).title || F.missing : '';
        return `<div class="stat"><span class="v">—</span><span class="l">the holdout has not been scored yet · no honest ${esc(ml(m))} to report</span></div>
          <div class="stat"><span class="v">${F.done.length}<small> of 5</small></span><span class="l">stages finished${next ? ' · next: ' + esc(next.toLowerCase()) : ''}</span></div>
          <div class="stat"><span class="v">${best ? esc(num(best.v, m)) : '—'}</span><span class="l">${best ? `best training-CV ${esc(ml(m))} so far (${best.at}) · an estimate, not the honest number` : 'no model has been scored yet'}</span></div>
          <div class="stat"><span class="v warn">Not ready</span><span class="l">the brief needs the holdout · run the remaining stages</span><div class="row"><button type="button" class="link-btn small" data-run-rest>Run the remaining stages →</button></div></div>`;
      }
      const ev = F.ev, ci = F.ci;
      const base = m === 'roc_auc' ? 'a random ranking scores 0.50' : (() => { const b = Object.entries(ev.baselines || {}).find(([, x]) => x && typeof x[m] === 'number'); return b ? `${b[0].replace(/_/g, ' ')} baseline ${num(b[1][m], m)}` : 'no trivial baseline recorded'; })();
      const c1 = `<div class="stat"><span class="v">${esc(num(F.point, m))}${ci ? ` <small>${esc(num(ci.low, m))}–${esc(num(ci.high, m))}</small>` : ''}</span><span class="l">${esc(ml(m))} on ${int(ev.holdout_rows)} rows the model never saw · ${esc(base)}</span></div>`;
      const gap = ev.cv_to_holdout_gap, cv = ev.cv_selected_metric_mean;
      const cGap = gap == null ? '' : `<div class="stat"><span class="v ${gap < -0.02 ? 'warn' : ''}">${gap >= 0 ? '+' : '−'}${esc(gapNum(gap, m))}</span><span class="l">holdout ${gap >= 0 ? 'better' : 'worse'} than the training-CV mean (${esc(num(cv, m))}) · ${gap >= -0.02 ? 'the cross-validation estimate held' : 'cross-validation was optimistic'}</span></div>`;
      let c2;
      if (C) {
        const r = (a => { const x = binormal(a, C.positive).atFraction(C.def / C.pool); return C.def * x.precision; });
        const [lo, mid, hi] = C.auc.map(r);
        c2 = `<div class="stat"><span class="v ok">~${int(Math.round(mid))}</span><span class="l">positives expected in the top ${int(C.def)} of ${int(C.pool)} cases · likely ${int(Math.round(lo))}–${int(Math.round(hi))} · about ${int(Math.round(C.def * C.positive))} at random · an estimate</span><div class="row"><button type="button" class="link-btn small" data-pane-go="value">Try other volumes →</button></div></div>`;
      } else {
        c2 = `<div class="stat"><span class="v">${int(ev.train_rows || 0)}</span><span class="l">training rows · the holdout was used ${int(ev.holdout_uses_in_this_project || p.holdout_uses || 1)} time(s)</span></div>`;
      }
      const ns = nextSteps(F);
      const c4 = `<div class="stat"><span class="v warn">${ev.production_approved ? 'Approved' : 'Research'}</span><span class="l">${ev.production_approved ? 'production-approved' : 'not production-approved'} · next: ${esc(ns.short)}</span><div class="row"><button type="button" class="link-btn small" data-pane-go="signoff">Go to sign-off →</button></div></div>`;
      return c1 + c2 + (cGap || `<div class="stat"><span class="v">${int(ev.holdout_rows)}</span><span class="l">holdout rows, scored once</span></div>`) + c4;
    }

    function runBox(F, lang) {
      const next = F.missing ? (F.meta[F.missing] || {}).title || F.missing : '';
      if (F.running) return lang === 'fa'
        ? `<div class="callout info" dir="rtl"><span class="ic">${icon('clock')}</span><span>مراحل در حال اجرا هستند. این صفحه پس از پایان، خودش به‌روز می‌شود.</span></div>`
        : `<div class="callout info"><span class="ic">${icon('clock')}</span><span>The stages are running. This page updates when they finish.</span></div>`;
      return lang === 'fa'
        ? `<div class="callout warn"><span class="ic">${icon('alert')}</span><span><b>مجموعه‌ی نگه‌داشته هنوز ارزیابی نشده است.</b> ${fa(F.done.length)} مرحله از ۵ مرحله تمام شده است. بدون آن، عدد صادقانه‌ای برای این گزارش وجود ندارد.</span></div><p><button type="button" class="btn primary" data-run-rest>${icon('play')}اجرای مراحل باقی‌مانده</button></p>`
        : `<div class="callout warn"><span class="ic">${icon('alert')}</span><span><b>The holdout has not been scored yet.</b> ${F.done.length} of 5 stages have finished${next ? `; the next is “${esc(next)}”` : ''}. Without it there is no honest number to put in this brief.</span></div><p><button type="button" class="btn primary" data-run-rest>${icon('play')}Run the remaining stages</button></p>`;
    }

    function docEn(F) {
      const s = F.sol, ev = F.ev, m = F.metric, bin = F.task === 'binary';
      const taskEn = { binary: 'yes/no (binary)', multiclass: 'multiclass', regression: 'regression' }[F.task] || F.task;
      const h = bin ? `Rank cases by how likely ${code(s.target)} is ${s.positive_label != null ? code(s.positive_label) : 'positive'}` : F.task === 'regression' ? `Estimate ${code(s.target)}` : `Predict the class of ${code(s.target)}`;
      let html = `<h2>${h}</h2><h3>What the model predicts</h3><p>A ${esc(taskEn)} model that predicts ${code(s.target)}${bin && s.positive_label != null ? ` (the positive class is ${code(s.positive_label)})` : ''}. It predicts at this moment: “${quote(s.prediction_moment || 'not written yet')}”.${F.p.goal ? ` The goal as the owner wrote it: “${quote(F.p.goal)}”` : ''} It uses ${F.usable != null ? `${int(F.usable)} of ${int(F.raw)}` : int(F.raw)} input columns${F.fin ? `; the chosen model is ${code(ev.model)} on the ${code(ev.feature_recipe)} feature recipe` : ''}.</p>`;
      if (!F.fin) return html + `<h3>How good it is, honestly</h3>${runBox(F, 'en')}` + refusedEn(F);
      const ci = F.ci;
      html += '<h3>How good it is, honestly</h3><p>';
      html += `On ${int(ev.holdout_rows)} rows the model never saw (the holdout, scored once), ${esc(ml(m))} is <b>${esc(num(F.point, m))}</b>${ci ? `, most likely between ${esc(num(ci.low, m))} and ${esc(num(ci.high, m))} (95% interval from ${int(ci.draws)} ${esc(ci.method || 'bootstrap')} draws)` : ''}.`;
      if (m === 'roc_auc') html += ` In plain words: it ranks a random positive case above a random negative one about ${Math.round(F.point * 100)}% of the time; a random ranking does that 50% of the time.`;
      else { const bs = Object.entries(ev.baselines || {}).filter(([, b]) => b && typeof b[m] === 'number'); if (bs.length) html += ` For comparison, ${bs.map(([k, b]) => `the ${esc(k.replace(/_/g, ' '))} baseline scores ${esc(num(b[m], m))}`).join('; ')}.`; }
      const hm = ev.holdout_metrics, apBase = ((ev.baselines || {}).majority_class || {}).average_precision_random_ranking;
      if (bin && m !== 'average_precision' && hm.average_precision != null && apBase != null) html += ` Average precision is ${num(hm.average_precision, 'average_precision')}, against ${num(apBase, 'average_precision')} for a random ranking.`;
      if (ev.cv_to_holdout_gap != null) html += ` Training cross-validation had estimated ${esc(num(ev.cv_selected_metric_mean, m))}; the holdout came out ${esc(gapNum(ev.cv_to_holdout_gap, m))} ${ev.cv_to_holdout_gap >= 0 ? 'better' : 'worse'} than that.`;
      html += '</p>' + refusedEn(F);
      const np = [];
      if (!ev.production_approved) np.push('This is research evidence, not production approval.');
      np.push(`The estimate rests on one holdout of ${int(ev.holdout_rows)} rows.`);
      if (F.quick && F.sampling.rows_used && F.source && F.sampling.rows_used < F.source) np.push(`The stages ran in quick mode on ${int(F.sampling.rows_used)} of ${int(F.source)} rows (${esc(String(F.sampling.rule || 'a sample').replace(/\s*\(([^)]*)\)/g, ', $1'))}).`);
      if (F.synthetic) np.push('The data is synthetic, so these numbers say nothing about real cases yet.');
      if (!s.time_column) np.push('No time column was declared, so we could not check whether the pattern holds over time.');
      html += `<h3>What is not proven yet</h3><p>${np.join(' ')}</p><h3>The next step</h3><p>${esc(nextSteps(F).en)}</p>`;
      return html;
    }
    function leakFacts(F) {
      const l = F.lEv;
      if (!l) return null;
      const review = (l.heuristic_review_candidates || []).map(x => x.feature || x).filter(Boolean);
      const sv = l.safe_vs_unsafe_training_cv || {}, mm = sv.metric || F.metric;
      const g = x => x && x.metrics && x.metrics[mm] ? (typeof x.metrics[mm] === 'object' ? x.metrics[mm].mean : x.metrics[mm]) : null;
      return { review, safe: g(sv.safe), unsafe: g(sv.unsafe), mm };
    }
    function refusedEn(F) {
      const s = F.sol, forb = s.forbidden || [], ids = s.identifiers || [], L = leakFacts(F);
      let h = '<h3>What we refused to use</h3>';
      h += forb.length ? `<p>The solution forbids these columns at the prediction moment, so the model never sees them:</p>${forb.slice(0, 8).map(f => `<p>${code(f.column)} — ${quote(f.reason || 'no reason written')}</p>`).join('')}${forb.length > 8 ? `<p>…and ${forb.length - 8} more.</p>` : ''}` : '<p>No column was declared forbidden in the solution.';
      const tail = [];
      if (ids.length) tail.push(`Identifiers are never inputs: ${ids.map(code).join(', ')}.`);
      if (L) tail.push(L.review.length ? `The leakage audit flagged ${L.review.length} more column(s) for a person to review: ${L.review.slice(0, 8).map(code).join(', ')}.` : 'The leakage audit flagged no other column for review.');
      if (L && L.unsafe != null && L.safe != null) tail.push(`With the forbidden columns, training cross-validation would have scored ${esc(num(L.unsafe, L.mm))} instead of ${esc(num(L.safe, L.mm))}; that gap is inflation, not skill.`);
      return h + (forb.length ? (tail.length ? `<p>${tail.join(' ')}</p>` : '') : ` ${tail.join(' ')}</p>`);
    }

    function docFa(F) {
      const s = F.sol, ev = F.ev, m = F.metric, bin = F.task === 'binary';
      const taskFa = { binary: 'دودویی (بله/خیر)', multiclass: 'چندکلاسه', regression: 'رگرسیون' }[F.task] || F.task;
      const h = bin ? `رتبه‌بندی موارد بر اساس احتمال اینکه ${code(s.target)} ${s.positive_label != null ? 'برابر ' + code(s.positive_label) : 'مثبت'} باشد` : F.task === 'regression' ? `برآورد مقدار ${code(s.target)}` : `پیش‌بینی کلاس ${code(s.target)}`;
      let html = `<h2>${h}</h2><h3>مدل چه چیزی را پیش‌بینی می‌کند</h3><p>یک مدل ${taskFa} که ${code(s.target)} را پیش‌بینی می‌کند${bin && s.positive_label != null ? ` (کلاس مثبت ${code(s.positive_label)} است)` : ''}. لحظه‌ی پیش‌بینی، به تعریف راه‌حل: «${quote(s.prediction_moment || '—')}».${F.p.goal ? ` هدف، به نوشته‌ی مالک: «${quote(String(F.p.goal).trim().replace(/[.!?]+$/, ''))}».` : ''} از ${F.usable != null ? `${fa(int(F.usable))} ستون از ${fa(int(F.raw))}` : fa(int(F.raw))} ستون ورودی استفاده می‌کند${F.fin ? `؛ مدل انتخاب‌شده ${code(ev.model)} با دستور ویژگی ${code(ev.feature_recipe)} است` : ''}.</p>`;
      if (!F.fin) return html + `<h3>صادقانه، چقدر خوب است</h3>${runBox(F, 'fa')}` + refusedFa(F);
      const ci = F.ci;
      html += '<h3>صادقانه، چقدر خوب است</h3><p>';
      html += `روی ${fa(int(ev.holdout_rows))} ردیفی که مدل هرگز ندیده بود (مجموعه‌ی نگه‌داشته، که فقط یک بار ارزیابی شد)، ${esc(ml(m))} برابر <b>${fa(num(F.point, m))}</b> است${ci ? ` و به احتمال زیاد بین ${fa(num(ci.low, m))} و ${fa(num(ci.high, m))} قرار دارد (بازه‌ی ۹۵ درصد از ${fa(int(ci.draws))} بار بازنمونه‌گیری)` : ''}.`;
      if (m === 'roc_auc') html += ` به زبان ساده: در حدود ${fa(Math.round(F.point * 100))} درصد موارد، یک مورد مثبت تصادفی را بالاتر از یک مورد منفی تصادفی قرار می‌دهد؛ یک رتبه‌بندی تصادفی این کار را در ۵۰ درصد موارد انجام می‌دهد.`;
      else { const bs = Object.entries(ev.baselines || {}).filter(([, b]) => b && typeof b[m] === 'number'); if (bs.length) html += ` برای مقایسه، ${bs.map(([k, b]) => `خط پایه‌ی ${code(k)} ${fa(num(b[m], m))} می‌گیرد`).join('؛ ')}.`; }
      const hm = ev.holdout_metrics, apBase = ((ev.baselines || {}).majority_class || {}).average_precision_random_ranking;
      if (bin && m !== 'average_precision' && hm.average_precision != null && apBase != null) html += ` میانگین دقت (average precision) برابر ${fa(num(hm.average_precision, 'average_precision'))} است، در برابر ${fa(num(apBase, 'average_precision'))} برای رتبه‌بندی تصادفی.`;
      if (ev.cv_to_holdout_gap != null) html += ` اعتبارسنجی متقاطع روی داده‌ی آموزش ${fa(num(ev.cv_selected_metric_mean, m))} را تخمین زده بود و نتیجه‌ی مجموعه‌ی نگه‌داشته ${fa(gapNum(ev.cv_to_holdout_gap, m))} ${ev.cv_to_holdout_gap >= 0 ? 'بهتر' : 'بدتر'} از آن بود.`;
      html += '</p>' + refusedFa(F);
      const np = [];
      if (!ev.production_approved) np.push('این یک شاهد پژوهشی است، نه تأیید برای استفاده در محیط واقعی.');
      np.push(`این برآورد فقط بر یک مجموعه‌ی نگه‌داشته با ${fa(int(ev.holdout_rows))} ردیف تکیه دارد.`);
      if (F.quick && F.sampling.rows_used && F.source && F.sampling.rows_used < F.source) np.push(`مراحل در حالت سریع روی ${fa(int(F.sampling.rows_used))} ردیف از ${fa(int(F.source))} ردیف اجرا شدند.`);
      if (F.synthetic) np.push('داده مصنوعی است، پس این اعداد هنوز چیزی درباره‌ی موارد واقعی نمی‌گویند.');
      if (!s.time_column) np.push('ستون زمانی تعریف نشده است، پس نتوانستیم بررسی کنیم که الگو در طول زمان پایدار می‌ماند یا نه.');
      html += `<h3>چه چیزی هنوز ثابت نشده</h3><p>${np.join(' ')}</p><h3>قدم بعدی</h3><p>${nextSteps(F).fa}</p>`;
      return html;
    }
    function refusedFa(F) {
      const s = F.sol, forb = s.forbidden || [], ids = s.identifiers || [], L = leakFacts(F);
      let h = '<h3>از چه استفاده نکردیم</h3>';
      h += forb.length ? `<p>راه‌حل این ستون‌ها را در لحظه‌ی پیش‌بینی ممنوع کرده است، پس مدل هرگز آن‌ها را نمی‌بیند:</p>${forb.slice(0, 8).map(f => `<p>${code(f.column)} — ${quote(f.reason || '—')}</p>`).join('')}${forb.length > 8 ? `<p>و ${fa(forb.length - 8)} ستون دیگر.</p>` : ''}` : '<p>هیچ ستونی در راه‌حل ممنوع اعلام نشده است.';
      const tail = [];
      if (ids.length) tail.push(`شناسه‌ها هرگز ورودی مدل نیستند: ${ids.map(code).join('، ')}.`);
      if (L) tail.push(L.review.length ? `ممیزی نشت داده ${fa(L.review.length)} ستون دیگر را برای بررسی انسانی علامت زد: ${L.review.slice(0, 8).map(code).join('، ')}.` : 'ممیزی نشت داده ستون دیگری را برای بررسی علامت نزد.');
      if (L && L.unsafe != null && L.safe != null) tail.push(`با ستون‌های ممنوع، اعتبارسنجی متقاطع روی داده‌ی آموزش ${fa(num(L.unsafe, L.mm))} می‌شد به‌جای ${fa(num(L.safe, L.mm))}؛ این فاصله تورم است، نه مهارت.`);
      return h + (forb.length ? (tail.length ? `<p>${tail.join(' ')}</p>` : '') : ` ${tail.join(' ')}</p>`);
    }

    function risks(F) {
      const rows = STAGES.flatMap(st => (((F.R[st] || {}).notes) || []).filter(n => n.severity === 'warning' || n.severity === 'high').map(n => Object.assign({ stage: st }, n)));
      const infos = STAGES.reduce((a, st) => a + (((F.R[st] || {}).notes) || []).filter(n => n.severity === 'info').length, 0);
      riskCount(rows.length);
      if (!rows.length) return `<tr><td colspan="3"><div class="empty">No stage note raised a warning for this project${F.done.length < 5 ? ' so far' : ''}. ${infos} information note${infos === 1 ? '' : 's'} sit on the stage pages; what is not proven yet is written in the brief.</div></td></tr>`;
      return rows.map(n => `<tr><td class="strong">${rich(n.title)} <span class="pill ${n.severity === 'high' ? 'bad' : 'warn'}">${esc(n.severity)}</span></td><td>${rich(n.text)}</td><td><div class="stack tight">${n.action ? `<span>${rich(n.action)}</span>` : ''}<span class="row">${known(n.proof).map(i => chip(i)).join('')}<a class="link-btn small" href="#${VIEW_OF[n.stage]}">${esc(((F.meta[n.stage] || {}).title) || n.stage)} →</a></span></div></td></tr>`).join('');
    }

    function signoff(F) {
      const s = F.sol, so = F.p.signoff || {};
      if (F.signed) return `<div class="callout ok"><span class="ic">${icon('check')}</span><span>The owner signed the solution${so.at ? ' on ' + esc(day(so.at)) : ''}${so.reason ? `: “${esc(so.reason)}”` : ''}.</span></div>
        <p>Signing approves the problem definition behind this brief: the target, the prediction moment and the forbidden columns. It does not approve production use.</p>
        <div class="row"><a class="btn" href="#solution">Review the solution</a>${F.fin ? `<a class="btn" href="${esc(DC.client.href.exportReport(F.p.id))}" download>Export the report</a>` : ''}</div>
        <span class="xs muted">Approving a brief approves a pilot, not a launch ${chip('DCLAB-R22')}.</span>`;
      return `<p>Sign the solution behind this brief: target ${code(s.target)}, prediction moment “${quote(s.prediction_moment || 'not written yet')}”, ${(s.forbidden || []).length} forbidden column(s)${F.gate && F.gate.required ? '. The workspace policy requires this signature' : ''}. The signature is logged with your reason; it does not approve production use.</p>
        <label class="sr-only" for="so-reason">Reason</label><textarea id="so-reason" rows="2" placeholder="Why you sign (kept in the project log)"></textarea>
        <div class="row"><button type="button" class="btn primary" data-so-sign>Sign the solution</button><a class="btn" href="#solution">Review the solution</a></div>
        <span class="xs muted">Approving a brief approves a pilot, not a launch ${chip('DCLAB-R22')}.</span>`;
    }
    function lessons(F) {
      const KIND = { fact: 'ok', decision: 'proof', recommendation: 'accent', risk: 'warn' };
      const items = STAGES.flatMap(st => (((F.R[st] || {}).claims) || []).map(c => Object.assign({ stage: st }, c)));
      const sft = `<div class="list-item"><span class="pill proof">data</span><div class="li-main"><span>Training examples from this project (the solution and each finished stage) for the policy model.</span><a class="link-btn small" href="${esc(DC.client.href.exportSft(F.p.id))}" download>Download the training examples (.jsonl) →</a></div></div>`;
      if (!items.length) return `<div class="empty">No stage has finished, so there is nothing to capture yet.</div>`;
      return items.map(c => `<div class="list-item"><span class="pill ${KIND[c.kind] || 'outline'}">${esc(c.kind)}</span><div class="li-main"><span>${rich(c.statement)}</span><span class="li-sub">${esc(((F.meta[c.stage] || {}).title) || c.stage)} · ${esc(c.claim_id || c.id || '')}</span></div></div>`).join('') + sft;
    }

    function showReal(p) {
      const F = facts(p), C = calcFor(F);
      set('brief-eyebrow', `${esc(p.name)} · WF-10 · for the business`);
      set('brief-status', F.fin ? (F.signed ? 'solution signed' : 'awaiting sign-off') : F.running ? 'stages running' : 'holdout not scored', 'pill ' + (F.fin ? (F.signed ? 'ok' : 'warn') : 'info'));
      $('#brief-copy', el).dataset.copy = location.origin + location.pathname + '#brief';
      const ex = $('#brief-export-real', el);
      ex.href = DC.client.href.exportReport(p.id); ex.hidden = false; $('#brief-export-sample', el).hidden = true;
      set('brief-stats', stats(F, C));
      set('brief-version', F.fin ? `Brief · from the stage records · holdout scored ${esc(day(F.fin.completed_at))} · generated by DCLab` : 'Brief draft · the holdout has not been scored');
      set('brief-en', docEn(F));
      set('brief-fa', docFa(F));
      const proofIds = known([].concat(...STAGES.map(st => (((F.R[st] || {}).notes) || []).flatMap(n => n.proof || []))));
      set('brief-foot', `Every number in this brief comes from the project's stage records${F.fin ? ` (${esc(F.fin.experiment_id || 'final stage')})` : ''}. Rules and evidence behind them: ${proofIds.slice(0, 10).map(i => chip(i)).join(' ') || 'none yet'}`);
      set('brief-reading', `<div class="eyebrow">Reading the numbers</div>` + (F.fin && F.metric === 'roc_auc'
        ? `<p><b>ROC-AUC ${esc(num(F.point, 'roc_auc'))}</b> means the model puts a real positive case above a negative one about ${Math.round(F.point * 100)} times out of 100. Random guessing gives 50.</p><p><b>Hit rate</b> is the share of the cases you act on that turn out positive.</p>`
        : F.fin ? `<p><b>${esc(ml(F.metric))} ${esc(num(F.point, F.metric))}</b> is the score on rows the model never saw. The interval shows how much it could move on another sample of the same size.</p>`
        : '<p>The honest number comes from the holdout, which has not been scored yet. Training-CV numbers are estimates.</p>'));
      // value calculator
      if (C) {
        set('v-title', 'Choose how many cases to act on');
        set('v-sub', `Expected result for ${int(C.pool)} cases like this project's data. The curve is a binormal model fitted to the holdout ROC-AUC${C.hasCi ? '; the band comes from its interval' : ' (no interval: the primary metric is ' + esc(ml(F.metric)) + ')'}.`);
        set('v-calls-label', 'Cases to act on'); set('v-cost-label', 'Cost per action'); set('v-val-label', 'Value per true positive'); set('v-legend-rnd', 'acting at random');
        const h = $('#v-hint', el); h.hidden = false;
        h.textContent = `Cost and value are placeholders: enter your own. The positive rate (${(C.positive * 100).toFixed(1)}%) and ROC-AUC (${num(C.auc[1], 'roc_auc')}${C.hasCi ? ', ' + num(C.auc[0], 'roc_auc') + '–' + num(C.auc[2], 'roc_auc') : ''}) come from the holdout.`;
        $('#v-body', el).hidden = false; $('#v-empty', el).hidden = true;
        setCalc(C);
      } else {
        set('v-title', 'Choose how many cases to act on');
        set('v-sub', 'Not available for this project yet.');
        $('#v-body', el).hidden = true; $('#v-hint', el).hidden = true;
        const e = $('#v-empty', el); e.hidden = false;
        e.innerHTML = !F.fin ? 'The value calculator needs the holdout ROC-AUC and positive rate. The holdout has not been scored yet: run the remaining stages first.'
          : F.task !== 'binary' ? `The value calculator prices a yes/no ranking. This project is a ${esc(F.task)} task, so there is no ranking to price.`
          : 'The final record has no holdout ROC-AUC or positive rate, so the value cannot be estimated honestly.';
      }
      set('brief-risks', risks(F));
      const said = STAGES.map(st => DC.explanation(F.R[st], ((F.meta[st] || {}).title) || st)).filter(Boolean);
      $('#brief-explain', el).innerHTML = said.join('');
      $('#brief-explain-panel', el).hidden = !said.length;  // only when a model wrote something that passed the checks
      const rs = $('#risks-sub', el); rs.hidden = false; rs.textContent = 'From the stage notes with severity warning or high.';
      set('so-pill', F.signed ? 'signed' : 'not signed', 'pill ' + (F.signed ? 'ok' : 'warn'));
      $('#so-sample', el).hidden = true;
      const so = $('#so-real', el); so.hidden = false; so.innerHTML = signoff(F);
      set('cap-sub', `WF-10 · ${STAGES.reduce((a, st) => a + (((F.R[st] || {}).claims) || []).length, 0)} claims from the stage records`);
      set('cap-list', lessons(F));
      DC.hydrate(el);
      if (F.running) watch(p.id);
    }

    /* Run the stages that have no record yet, then redraw when the job ends. */
    let stopWatch = null;
    function watch(id) {
      if (stopWatch) return;
      stopWatch = DC.poll(async () => {
        const p = await DC.currentProject.get(true);
        if (!p || p.id !== id) { stopWatch = null; return true; }
        const F = facts(p);
        if (F.running) return false;
        stopWatch = null;
        if (DC.state.view === 'brief') { showReal(p); DC.toast(F.fin ? 'The holdout is scored. The brief now shows the honest numbers.' : 'The run stopped before the holdout. See the Workflow page.', { ok: !!F.fin }); }
        return true;
      }, 2000);
    }
    el.addEventListener('click', async e => {
      const run = e.target.closest('[data-run-rest]');
      if (run && this.s.p) {
        const F = facts(this.s.p);
        if (!F.missing) return;
        run.disabled = true;
        try {
          await DC.client.runAll(F.p.id, { query: { start: F.missing } });
          DC.toast('Running the remaining stages. This page updates when they finish.');
          DC.currentProject.clear();
          const p = await DC.currentProject.get(true);
          this.s.p = p; showReal(p);
        } catch (err) { run.disabled = false; DC.toast(err.message, { ok: false }); }
        return;
      }
      const sign = e.target.closest('[data-so-sign]');
      if (sign && this.s.p) {
        const id = this.s.p.id, reason = ($('#so-reason', el) || {}).value || '';
        DC.modal.open({
          eyebrow: '<span class="eyebrow">Sign-off</span>', title: 'Sign the solution?', confirm: 'Sign the solution',
          html: `<p>This records that you, the owner, accept the solution behind this brief${reason.trim() ? `, with the reason “${esc(reason.trim())}”` : ''}. It is logged in the project and does not approve production use.</p>`,
          onConfirm: () => {
            DC.client.approveGate(id, { body: { gate: 'solution', reason: reason.trim() } })
              .then(() => { DC.modal.close(); DC.toast('Solution signed. The approval is in the project log.'); DC.currentProject.clear(); return DC.currentProject.get(true); })
              .then(p => { if (p) { this.s.p = p; showReal(p); } })
              .catch(err => DC.toast(err.message, { ok: false }));
            return false;
          },
        });
      }
    });
    this.s = { tok: 0, p: null, showReal, showSample };
  },
  async enter(el) {
    const s = this.s, tok = ++s.tok;
    let p = null;
    try { p = await DC.currentProject.get(); } catch (e) { p = null; }
    if (tok !== s.tok) return;
    DC.markSample(!p);
    s.p = p;
    try { if (p) s.showReal(p); else s.showSample(); } catch (e) { console.error('brief', e); }
  },
});
