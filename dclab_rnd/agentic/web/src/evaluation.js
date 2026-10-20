/* The Evaluation panels (DC.evaluation.render): one builder for the Evaluation page and the Models page's Evaluation tab.
   It draws a model's measures, a confusion matrix at a cutoff the person moves, and the curves (ROC, precision and recall,
   what each cutoff gives, cumulative gains, lift, calibration, the scores of real cases and of the others), from the
   numbers GET /api/projects/{id}/evaluation returns: counts, rates and curve points, never a row. Legends are written
   beside each chart, never inside it, so no label lies over a line. */
DC.evaluation = (function () {
  const { $, $$, esc, charts } = DC;
  const f = (v, d = 3) => (v == null || !isFinite(v) ? '—' : Number(v).toFixed(d));
  const N = v => Number(v).toLocaleString('en-US');
  const pct = v => (v == null || !isFinite(v) ? '—' : (v * 100).toFixed(v < 0.1 ? 1 : 0) + '%');

  const WORDS = {
    precision: ['Precision', 'of the cases flagged, the share that are real'],
    recall: ['Recall', 'of the real cases, the share that were flagged'],
    f1: ['F1 score', 'precision and recall together: low if either one is low'],
    accuracy: ['Accuracy', 'the share of all cases called right; high even when the rare case is never flagged'],
    balanced_accuracy: ['Balanced accuracy', 'recall and specificity averaged: fair when one class is rare'],
    specificity: ['Specificity', 'of the cases that are not real, the share left alone'],
    mcc: ['MCC', 'one number from −1 to 1 that uses all four counts of the matrix'],
    flagged_share: ['Share flagged', 'the share of rows the cutoff flags'],
    positive_rate: ['Share of real cases', 'how common the case is in these rows'],
    roc_auc: ['ROC-AUC', 'how well the scores rank real cases above the others; 0.5 is chance'],
    average_precision: ['Average precision', 'precision over every cutoff: the honest score for a rare case'],
    brier: ['Brier score', 'the squared error of the probabilities; lower is better'],
    log_loss: ['Log loss', 'punishes confident wrong scores; lower is better'],
    macro_precision: ['Macro precision', 'precision averaged over the classes, each counting the same'],
    macro_recall: ['Macro recall', 'recall averaged over the classes, each counting the same'],
    macro_f1: ['Macro F1', 'F1 averaged over the classes, each counting the same'],
    weighted_precision: ['Weighted precision', 'precision averaged by class size'],
    weighted_recall: ['Weighted recall', 'recall averaged by class size'],
    weighted_f1: ['Weighted F1', 'F1 averaged by class size'],
    top_2_accuracy: ['Top-2 accuracy', 'the true class is one of the two the model rates highest'],
    mae: ['MAE', 'the average size of the error, in the units of the answer'],
    rmse: ['RMSE', 'like MAE but large errors count more'],
    median_absolute_error: ['Median error', 'half the errors are smaller than this'],
    r2: ['R²', 'the share of the answer\'s variation the model explains; 0 is no better than the average'],
    mape: ['MAPE', 'the average error as a share of the true answer'],
    mean_error: ['Mean error', 'above zero: the model predicts too high on average; below: too low'],
  };
  const ORDER = {
    binary: ['precision', 'recall', 'f1', 'accuracy', 'balanced_accuracy', 'specificity', 'mcc', 'flagged_share', 'roc_auc', 'average_precision', 'brier', 'log_loss'],
    multiclass: ['accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1', 'top_2_accuracy', 'log_loss'],
    regression: ['mae', 'rmse', 'median_absolute_error', 'r2', 'mape', 'mean_error'],
  };
  const AT_CUTOFF = ['precision', 'recall', 'f1', 'accuracy', 'balanced_accuracy', 'specificity', 'mcc', 'flagged_share'];

  /* the measures a cutoff gives, from its four counts (as dclab_core.model_evaluation.at_cutoff) */
  function atCutoff(c) {
    const { tp, fp, fn, tn } = c, n = tp + fp + fn + tn;
    const precision = tp + fp ? tp / (tp + fp) : null, recall = tp + fn ? tp / (tp + fn) : null, specificity = tn + fp ? tn / (tn + fp) : null;
    const f1 = precision && recall ? 2 * precision * recall / (precision + recall) : (precision != null && recall != null ? 0 : null);
    const d = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    return { precision, recall, f1, specificity, accuracy: n ? (tp + tn) / n : null, flagged_share: n ? (tp + fp) / n : null,
      balanced_accuracy: recall != null && specificity != null ? (recall + specificity) / 2 : null, mcc: d ? (tp * tn - fp * fn) / d : null };
  }
  const tile = (key, value, d = 3) => {
    const w = WORDS[key] || [key, ''];
    return `<div class="stat"><span class="v">${key === 'flagged_share' || key === 'positive_rate' ? pct(value) : f(value, d)}</span><span class="l"><b>${esc(w[0])}</b> · ${esc(w[1])}</span></div>`;
  };
  const legend = items => `<div class="legend">${items.map(([cls, text]) => `<span><i class="${cls}"></i>${esc(text)}</span>`).join('')}</div>`;
  const panel = (title, note, body, wide) => `<div class="panel${wide ? ' span-2' : ''}"><div class="panel-head"><div class="panel-title"><h2>${esc(title)}</h2></div></div>`
    + `<div class="panel-body stack tight">${body}<p class="xs muted">${esc(note)}</p></div></div>`;

  /* ---- the confusion matrices ---- */
  const lv = share => `lv${Math.max(0, Math.min(5, Math.round((share || 0) * 5)))}`;
  function binaryMatrix(c) {
    const neg = c.tn + c.fp, pos = c.tp + c.fn;
    const cell = (name, say, n, row, good) => `<td class="cm-c ${good ? 'ok' : 'bad'} ${lv(row ? n / row : 0)}"><b>${N(n)}</b><span>${esc(name)}</span><small>${esc(say)}</small></td>`;
    return `<table class="cm" aria-label="Confusion matrix"><thead><tr><th></th><th scope="col">Not flagged</th><th scope="col">Flagged</th></tr></thead><tbody>`
      + `<tr><th scope="row">Not a real case<small>${N(neg)} rows</small></th>${cell('True negative', 'left alone, rightly', c.tn, neg, true)}${cell('False positive', 'flagged, wrongly', c.fp, neg, false)}</tr>`
      + `<tr><th scope="row">A real case<small>${N(pos)} rows</small></th>${cell('False negative', 'missed', c.fn, pos, false)}${cell('True positive', 'flagged, rightly', c.tp, pos, true)}</tr></tbody></table>`;
  }
  function classMatrix(conf) {
    const labels = conf.labels, m = conf.matrix;
    const head = labels.map(l => `<th scope="col">${esc(l)}</th>`).join('');
    const rows = m.map((row, i) => {
      const total = row.reduce((a, b) => a + b, 0);
      return `<tr><th scope="row">${esc(labels[i])}<small>${N(total)} rows</small></th>${row.map((n, j) => `<td class="cm-c ${i === j ? 'ok' : 'bad'} ${lv(total ? n / total : 0)}"><b>${N(n)}</b><small>${pct(total ? n / total : 0)}</small></td>`).join('')}</tr>`;
    }).join('');
    return `<div class="table-wrap"><table class="cm cm-wide" aria-label="Confusion matrix by class"><thead><tr><th>Actual ↓ · Predicted →</th>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  /* ---- charts ---- */
  function histogram(h) {
    const W = 360, H = 190, L = 34, R = 8, T = 10, B = 26, n = h.negative.length, bw = (W - L - R) / n;
    const sn = h.negative.reduce((a, b) => a + b, 0) || 1, sp = h.positive.reduce((a, b) => a + b, 0) || 1;
    const neg = h.negative.map(v => v / sn), pos = h.positive.map(v => v / sp);
    const max = Math.max(0.05, ...neg, ...pos) * 1.08;
    let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" data-style="max-width:${Math.round(W * 1.12)}px" role="img" aria-label="Scores of real cases and of the others">`;
    [0, 0.25, 0.5, 0.75, 1].map(t => t * max).forEach(t => { const y = H - B - (t / max) * (H - T - B); s += `<line class="grid-line" x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/><text class="tick" x="${L - 5}" y="${y + 3}" text-anchor="end">${Math.round(t * 100)}%</text>`; });
    for (let i = 0; i < n; i++) {
      const x = L + i * bw, hn = (neg[i] / max) * (H - T - B), hp = (pos[i] / max) * (H - T - B);
      s += `<rect class="bar muted" x="${x + 0.5}" y="${H - B - hn}" width="${Math.max(1, bw / 2 - 1)}" height="${hn}" rx="1.5"/><rect class="bar" x="${x + bw / 2}" y="${H - B - hp}" width="${Math.max(1, bw / 2 - 1)}" height="${hp}" rx="1.5"/>`;
    }
    [0, 0.5, 1].forEach(t => { s += `<text class="tick" x="${L + t * (W - L - R)}" y="${H - 8}" text-anchor="${t === 0 ? 'start' : t === 1 ? 'end' : 'middle'}">${t}</text>`; });
    return s + '</svg>';
  }
  const axis = { xMin: 0, xMax: 1, yMin: 0, yMax: 1, width: 360, height: 220, xFmt: v => v.toFixed(1), yFmt: v => v.toFixed(1) };

  function binaryCharts(ev, row) {
    const c = ev.curves || {}, out = [];
    const rate = ev.metrics.positive_rate, grid = ev.cutoffs || [];
    const at = grid.map(r => ({ t: r.t, ...atCutoff(r) }));
    const rd = row ? atCutoff(row) : null;
    if (at.length > 1) {
      const pts = key => at.filter(a => a[key] != null).map(a => [a.t, a[key]]);
      out.push({ id: 'cut', title: 'What each cutoff gives', note: 'Moving the cutoff trades catching more real cases against flagging more rows. The dashed line is the cutoff chosen above.',
        html: () => legend([['', 'precision'], ['s2', 'recall'], ['s3', 'F1']]) + charts.line([{ pts: pts('precision') }, { pts: pts('recall'), cls: 's2' }, { pts: pts('f1'), cls: 's3' }],
          { ...axis, xLabel: 'cutoff (score)', yLabel: 'share', markers: row ? [{ x: row.t, label: '' }] : [], aria: 'Precision, recall and F1 by cutoff' }) });
    }
    if (c.roc) {
      const pt = row && rd ? [{ x: row.fp / Math.max(1, row.fp + row.tn), y: rd.recall, r: 4.5 }] : [];
      out.push({ id: 'roc', title: `ROC curve · AUC ${f(ev.metrics.roc_auc)}`, note: 'How well scores rank real cases above the others: the diagonal is chance, the top-left corner perfect. The dot is the chosen cutoff.',
        html: () => charts.line([{ pts: c.roc }], { ...axis, diag: true, points: pt, xLabel: 'share of others flagged', yLabel: 'share of real cases caught', aria: 'ROC curve' }) });
    }
    if (c.pr) {
      const pt = row && rd && rd.precision != null ? [{ x: rd.recall, y: rd.precision, r: 4.5 }] : [];
      out.push({ id: 'pr', title: `Precision and recall · AP ${f(ev.metrics.average_precision)}`, note: `Of the flagged, how many are real, as more real cases are caught. The grey line is flagging at random (${pct(rate)}).`,
        html: () => legend([['', 'model'], ['s4', 'flagging at random']]) + charts.line([{ pts: c.pr }, { pts: [[0, rate], [1, rate]], cls: 's4' }], { ...axis, points: pt, xLabel: 'share of real cases caught (recall)', yLabel: 'share of flags that are real', aria: 'Precision and recall' }) });
    }
    if (c.gains) {
      out.push({ id: 'gains', title: 'Cumulative gains', note: 'Checking the highest scores first: the share of real cases found as more rows are checked. The diagonal is checking at random.',
        html: () => charts.line([{ pts: c.gains }], { ...axis, diag: true, xLabel: 'share of rows checked', yLabel: 'share of real cases found', aria: 'Cumulative gains' }) });
      const lift = c.gains.filter(p => p[0] > 0.01).map(p => [p[0], p[1] / p[0]]);
      const top = Math.max(2, ...lift.map(p => p[1]));
      out.push({ id: 'lift', title: 'Lift', note: 'How many times better than random, at each share of rows checked.',
        html: () => legend([['', 'lift'], ['s4', 'random']]) + charts.line([{ pts: lift }, { pts: [[0, 1], [1, 1]], cls: 's4' }], { ...axis, yMax: Math.ceil(top), yFmt: v => v.toFixed(0) + '×', xLabel: 'share of rows checked', yLabel: 'times better than random', aria: 'Lift' }) });
    }
    if ((c.calibration || []).length) {
      const pts = c.calibration.filter(b => b.predicted != null && b.observed != null).map(b => [b.predicted, b.observed]).sort((a, b) => a[0] - b[0]);
      const top = Math.min(1, Math.max(0.2, ...pts.map(p => Math.max(p[0], p[1]))) * 1.1);
      out.push({ id: 'cal', title: 'Calibration', note: 'When it says 30%, do 30% turn out real? The diagonal is perfect: above it the model is too cautious, below it too sure.',
        html: () => charts.line([{ pts, dots: true }], { ...axis, xMax: top, yMax: top, diag: true, xLabel: 'predicted probability', yLabel: 'observed rate', aria: 'Calibration' }) });
    }
    if (c.histogram) {
      out.push({ id: 'hist', title: 'Scores of real cases and of the others', note: 'The further apart the two groups, the easier to tell them apart. Each group is shown as its own share, so a rare case stays visible.',
        html: () => legend([['sq', 'real cases'], ['sq s4', 'the others']]) + histogram(c.histogram) });
    }
    return out;
  }

  function regressionCharts(ev) {
    const c = ev.curves || {}, out = [];
    if (c.by_prediction && c.by_prediction.length) {
      const pts = c.by_prediction.map(b => [b.predicted, b.actual]);
      const lo = Math.min(...pts.map(p => Math.min(p[0], p[1]))), hi = Math.max(...pts.map(p => Math.max(p[0], p[1])));
      out.push({ id: 'pred', title: 'Predicted against actual', note: 'Rows grouped by their prediction, ten groups: the average prediction and the average true answer. On the diagonal is perfect.',
        html: () => charts.line([{ pts: pts.slice().sort((a, b) => a[0] - b[0]), dots: true }], { width: 360, height: 220, xMin: lo, xMax: hi, yMin: lo, yMax: hi, diag: true, xLabel: 'predicted', yLabel: 'actual', aria: 'Predicted against actual' }) });
    }
    if (c.residuals) {
      const e = c.residuals.edges, items = c.residuals.counts.map((v, i) => ({ value: v, label: ((e[i] + e[i + 1]) / 2).toPrecision(2) }));
      out.push({ id: 'resid', title: 'Errors', note: 'Prediction minus the true answer: centred on zero is unbiased, narrow is accurate.', html: () => charts.vbars(items, { width: 360, height: 190, aria: 'Distribution of the errors' }) });
    }
    return out;
  }

  function classTable(ev) {
    const rows = ev.per_class || [];
    if (!rows.length) return '';
    const best = charts.barsH(rows.slice(0, 12).map(r => ({ label: r.label, value: r.f1 || 0, valueText: `F1 ${f(r.f1)}` })), { width: 520, labelW: 120, valW: 80, min: 0, max: 1, rowH: 26, ticks: [0, 0.5, 1], tickFmt: v => v.toFixed(1), aria: 'F1 by class' });
    const body = rows.map(r => `<tr><td>${esc(r.label)}</td><td class="num mono">${N(r.support)}</td><td class="num mono">${f(r.precision)}</td><td class="num mono">${f(r.recall)}</td><td class="num mono">${f(r.f1)}</td></tr>`).join('');
    return `<div class="table-wrap"><table class="data"><thead><tr><th>Class</th><th class="num">Rows</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th></tr></thead><tbody>${body}</tbody></table></div>${best}`;
  }
  function mostConfused(ev) {
    const pairs = (ev.curves || {}).most_confused || [];
    if (!pairs.length) return '';
    return '<ul class="plain">' + pairs.map(p => `<li>${N(p.rows)} rows of <b>${esc(p.true)}</b> were called <b>${esc(p.predicted)}</b> (${pct(p.share_of_true)} of that class)</li>`).join('') + '</ul>';
  }

  /* ---- the whole panel ---- */
  function render(box, data, opts = {}) {
    if (!data || !data.holdout) { DC.states.empty(box, 'This project has no evaluation yet: the final stage has not scored the held-back cases.', { href: '#project', action: 'Open the workflow' }); return; }
    const sets = { holdout: data.holdout, validation: data.validation || null };
    let which = opts.rows && sets[opts.rows] ? opts.rows : 'holdout';
    box.innerHTML = `<div class="stack ev"><div class="row spread"><div class="seg" data-seg="ev-rows" role="group" aria-label="Which rows">
        <button type="button" data-v="holdout" aria-pressed="true">Held-back cases</button>${sets.validation ? '<button type="button" data-v="validation" aria-pressed="false">Validation folds</button>' : ''}</div>
        <span class="row"><span class="pill outline" data-ev-n></span><span class="pill" data-ev-source></span></span></div>
      <div data-ev-body class="stack"></div></div>`;
    const body = $('[data-ev-body]', box);
    function paint() {
      const ev = sets[which];
      $('[data-ev-n]', box).textContent = `${N(ev.n)} rows`;
      const src = $('[data-ev-source]', box);
      src.textContent = ev.source === 'scores' ? 'from the model\'s scores' : 'read back from the record';
      src.className = 'pill ' + (ev.source === 'scores' ? 'ok' : 'warn');
      const why = which === 'holdout'
        ? `The held-back cases (${N(ev.n)}), read once by the final stage${data.model ? ' for ' + esc(data.model) : ''}. This is the honest estimate; it says nothing about the future by itself, and it is not production approval.`
        : `The training rows, each scored by a model that did not see it (the folds the model was chosen on). Compare with the held-back cases: a big gap means the choice was optimistic.`;
      const notes = (ev.notes || []).map(t => `<div class="callout info small"><span>${esc(t)}</span></div>`).join('');
      let html = `<p class="small muted">${why}</p>${notes}`;
      const keys = (ORDER[ev.kind] || []).filter(k => ev.metrics[k] != null || (ev.kind === 'binary' && AT_CUTOFF.includes(k) && ev.cutoffs && ev.cutoffs.length));
      html += '<div class="stats" data-ev-cards></div>';
      if (ev.kind === 'binary') {
        html += `<div class="grid g2"><div class="panel"><div class="panel-head"><div class="panel-title"><h2>Confusion matrix</h2><span class="sub" data-ev-cut-text></span></div></div>`
          + `<div class="panel-body stack tight">${ev.cutoffs && ev.cutoffs.length > 1 ? '<label class="small strong stack tight">Cutoff: flag a row when its score is at least this<input type="range" class="ev-cut" min="0" step="1" data-style="width:100%"></label>' : ''}`
          + `<div data-ev-matrix></div><p class="xs muted">Rows are what is true, columns what the model said. Green cells are right, the other two are the two kinds of mistake.</p></div></div>`
          + `<div data-ev-cutchart></div></div>`;
      } else if (ev.kind === 'multiclass') {
        html += `<div class="panel"><div class="panel-head"><div class="panel-title"><h2>Confusion matrix</h2></div></div><div class="panel-body stack tight">${ev.confusion ? classMatrix(ev.confusion) : mostConfused(ev) || '<div class="empty">No confusion counts for these rows.</div>'}`
          + `<p class="xs muted">Rows are the true class, columns what the model said; each row sums to 100%. The diagonal is right.</p></div></div>`
          + `<div class="panel"><div class="panel-head"><div class="panel-title"><h2>Each class</h2></div></div><div class="panel-body stack tight">${classTable(ev)}</div></div>`;
      }
      html += '<div class="grid g3" data-ev-charts></div>';
      if (which === 'holdout' && sets.validation) html += compare(sets);
      body.innerHTML = html;

      let idx = 0;
      if (ev.kind === 'binary') {
        const grid = ev.cutoffs || [];
        idx = Math.max(0, grid.findIndex(r => Math.abs(r.t - (ev.cutoff == null ? 0.5 : ev.cutoff)) < 1e-6));
        const slider = $('.ev-cut', body);
        if (slider) { slider.max = String(grid.length - 1); slider.value = String(idx); }
      }
      function cutting() {
        if (ev.kind !== 'binary') return;
        const row = (ev.cutoffs || [])[idx] || ev.confusion && { t: ev.cutoff, ...ev.confusion };
        const m = row ? { ...ev.metrics, ...atCutoff(row) } : ev.metrics;
        if (row) { $('[data-ev-matrix]', body).innerHTML = binaryMatrix(row); $('[data-ev-cut-text]', body).textContent = `at cutoff ${f(row.t, 2)}`; }
        else $('[data-ev-matrix]', body).innerHTML = '<div class="empty">No confusion counts for these rows.</div>';
        $('[data-ev-cards]', body).innerHTML = keys.map(k => tile(k, m[k])).join('');
        const all = binaryCharts(ev, row);
        const cut = all.find(c => c.id === 'cut');
        $('[data-ev-cutchart]', body).innerHTML = cut ? panel(cut.title, cut.note, cut.html()) : '';
        const holder = $('[data-ev-charts]', body);
        holder.innerHTML = all.filter(c => c.id !== 'cut').map(c => panel(c.title, c.note, c.html())).join('');
        DC.applyDataStyles(body);
      }
      if (ev.kind === 'binary') {
        cutting();
        const slider = $('.ev-cut', body);
        if (slider) slider.addEventListener('input', () => { idx = Number(slider.value); cutting(); });
      } else {
        $('[data-ev-cards]', body).innerHTML = keys.map(k => tile(k, ev.metrics[k])).join('');
        if (ev.kind === 'regression') $('[data-ev-charts]', body).innerHTML = regressionCharts(ev).map(c => panel(c.title, c.note, c.html())).join('');
        else $('[data-ev-charts]', body).remove();
        DC.applyDataStyles(body);
      }
      if (!$('[data-ev-charts]', body) || !$('[data-ev-charts]', body).children.length) { const h = $('[data-ev-charts]', body); if (h) h.remove(); }
    }
    function compare(s) {
      const h = s.holdout, v = s.validation, rows = (ORDER[h.kind] || []).filter(k => k !== 'flagged_share' && h.metrics[k] != null && v.metrics[k] != null);
      if (!rows.length) return '';
      const body = rows.map(k => `<tr><td>${esc((WORDS[k] || [k])[0])}</td><td class="num mono">${f(h.metrics[k])}</td><td class="num mono">${f(v.metrics[k])}</td><td class="num mono">${f(h.metrics[k] - v.metrics[k])}</td></tr>`).join('');
      return `<div class="panel"><div class="panel-head"><div class="panel-title"><h2>Held-back cases against validation folds</h2><span class="sub">a large gap means the model was chosen on luck the held-back cases did not share</span></div></div>`
        + `<div class="table-wrap"><table class="data"><thead><tr><th>Measure</th><th class="num">Held back</th><th class="num">Validation</th><th class="num">Gap</th></tr></thead><tbody>${body}</tbody></table></div></div>`;
    }
    box.addEventListener('segchange', e => { if (e.target.closest('[data-seg="ev-rows"]') && sets[e.detail]) { which = e.detail; paint(); } });
    paint();
  }
  /* The panels for a project in a box, with the page's usual states: no project, no final stage yet, a failed read. */
  async function mount(box, p) {
    if (!box) return;
    if (!p) { DC.states.empty(box, 'No project is open, so there is no model to evaluate. Start one on Home.', { href: '#home', action: 'Go to Home' }); return; }
    if (!(p.records || {}).final) { DC.states.empty(box, 'The final stage has not run yet: the held-back cases are scored once, there, and the evaluation is read from that record.', { href: '#project', action: 'Open the workflow' }); return; }
    DC.states.loading(box, { text: 'Reading the evaluation…' });
    let data;
    try { data = await DC.client.projectEvaluation(p.id); }
    catch (e) { DC.states.failed(box, e, { lead: 'Could not read the evaluation: ', retry: () => mount(box, p) }); return; }
    render(box, data);
  }
  /* A chart a notebook cell showed as data (chart_line, chart_bars; the server has already checked it): drawn in the app's own style. */
  const SERIES = ['', 's2', 's3', 's4', 'bad', 'ok'];
  function drawChart(spec) {
    const title = spec.title ? `<h3 class="cell-chart-title">${esc(spec.title)}</h3>` : '';
    if (spec.kind === 'bars') {
      const items = spec.labels.map((l, i) => ({ label: l, value: spec.values[i], valueText: f(spec.values[i], 4) }));
      const lo = Math.min(0, ...spec.values), hi = Math.max(0.0001, ...spec.values) * 1.08;
      return title + charts.barsH(items, { width: 620, labelW: 130, valW: 90, min: lo, max: hi, rowH: 26, tickFmt: v => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)), aria: spec.title || 'bar chart' })
        + (spec.ylabel ? `<p class="xs muted">${esc(spec.ylabel)}</p>` : '');
    }
    const named = spec.series.filter(se => se.name);
    const all = spec.series.flatMap(se => se.pts);
    const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
    const pad = (a, b) => (a === b ? 1 : (b - a) * 0.06);
    const opts = { width: 560, height: 260, xMin: Math.min(...xs), xMax: Math.max(...xs), yMin: Math.min(...ys) - pad(Math.min(...ys), Math.max(...ys)),
      yMax: Math.max(...ys) + pad(Math.min(...ys), Math.max(...ys)), xLabel: spec.xlabel, yLabel: spec.ylabel, diag: spec.diag, aria: spec.title || 'line chart',
      xFmt: v => (Math.abs(v) >= 100 ? v.toFixed(0) : +v.toPrecision(3)), yFmt: v => (Math.abs(v) >= 100 ? v.toFixed(0) : +v.toPrecision(3)) };
    if (opts.xMin === opts.xMax) { opts.xMin -= 1; opts.xMax += 1; }
    return title + (named.length > 1 || (named.length === 1 && spec.series.length > 1) ? legend(spec.series.map((se, i) => [SERIES[i % SERIES.length], se.name || 'series ' + (i + 1)])) : '')
      + charts.line(spec.series.map((se, i) => ({ pts: se.pts, cls: SERIES[i % SERIES.length], dots: spec.dots || se.pts.length <= 12 })), opts);
  }
  return { render, mount, drawChart, atCutoff };
})();
