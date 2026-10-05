DC.view('evidence', {
  init(el) {
    const { $, $$, esc, chip, chips, RECORDS, TYPE_LABEL, TYPE_CLS, charts } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id));
    $('#ev-total', el).textContent = RECORDS.length + ' records';
    const types = ['rule', 'workflow', 'experiment', 'leakage_precedent', 'pitfall', 'finding', 'dataset'];
    $('#ev-types', el).innerHTML = types.map(t => `<button type="button" class="chip" data-toggle data-type="${t}" aria-pressed="false">${esc(TYPE_LABEL[t])} <span class="faint">${RECORDS.filter(r => r.type === t).length}</span></button>`).join('');
    const datasets = [...new Set(RECORDS.map(r => r.meta && r.meta.dataset).filter(Boolean))].sort();
    $('#ev-ds', el).innerHTML = '<option value="">All datasets</option>' + datasets.map(d => `<option>${esc(d)}</option>`).join('');
    const tok = s => (s.toLowerCase().match(/[a-z0-9_]+/g) || []).filter(w => w.length > 1 && !['the', 'is', 'of', 'to', 'and', 'in', 'on', 'for', 'a', 'it', 'be', 'do', 'should', 'with', 'what', 'my', 'are', 'when', 'fine'].includes(w));
    const docs = RECORDS.map(r => ({ r, title: r.title.toLowerCase(), text: (r.title + ' ' + r.text).toLowerCase() }));
    const df = {};
    docs.forEach(d => new Set(tok(d.text)).forEach(w => { df[w] = (df[w] || 0) + 1; }));
    const idf = w => Math.log(1 + docs.length / (1 + (df[w] || 0)));
    let showAll = false;
    function search() {
      const q = tok($('#ev-q', el).value), ds = $('#ev-ds', el).value;
      const on = $$('#ev-types [aria-pressed="true"]', el).map(b => b.dataset.type);
      const cand = docs.filter(d => (!on.length || on.includes(d.r.type)) && (!ds || (d.r.meta && d.r.meta.dataset === ds)));
      const scored = cand.map(d => { let s = 0; q.forEach(w => { const c = d.text.split(w).length - 1; if (c) s += idf(w) * (Math.min(c, 5) + (d.title.includes(w) ? 3 : 0)); }); return { d, s }; })
        .filter(x => !q.length || x.s > 0).sort((a, b) => b.s - a.s).slice(0, showAll ? 40 : 12);
      $('#ev-results', el).innerHTML = scored.length ? scored.map(({ d }) => {
        const text = d.r.text; const low = text.toLowerCase();
        let at = q.map(w => low.indexOf(w)).filter(i => i >= 0).sort((a, b) => a - b)[0] || 0;
        const start = Math.max(0, at - 90), snip = (start ? '…' : '') + text.slice(start, start + 260) + (start + 260 < text.length ? '…' : '');
        let html = esc(snip);
        q.forEach(w => { html = html.replace(new RegExp('(' + w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi'), '<mark>$1</mark>'); });
        return `<button type="button" class="list-item" data-record="${esc(d.r.id)}"><div class="li-main"><span class="row"><span class="pill ${TYPE_CLS[d.r.type] || ''}">${esc(TYPE_LABEL[d.r.type] || d.r.type)}</span><span class="tag">${esc(d.r.id)}</span>${d.r.meta && d.r.meta.dataset ? `<span class="tag">${esc(d.r.meta.dataset)}</span>` : ''}</span><span class="li-title">${esc(d.r.title)}</span><span class="li-sub">${html}</span></div></button>`;
      }).join('') + (!showAll && scored.length === 12 ? '<div class="list-item"><button type="button" class="link-btn small" id="ev-more">Show more results</button></div>' : '') : '<div class="empty">No record matches. Try fewer words, or clear the filters.</div>';
    }
    $('#ev-q', el).addEventListener('input', () => { showAll = false; search(); });
    $('#ev-results', el).addEventListener('click', e => { if (e.target.closest('#ev-more')) { showAll = true; search(); } });
    $('#ev-ds', el).addEventListener('change', search);
    $('#ev-types', el).addEventListener('togglechange', search);
    search();
    $('#ev-mix', el).innerHTML = charts.barsH(types.map(t => ({ label: TYPE_LABEL[t], value: RECORDS.filter(r => r.type === t).length, valueText: String(RECORDS.filter(r => r.type === t).length), cls: t === 'leakage_precedent' ? 'bad' : t === 'rule' ? 'proof' : '' })),
      { width: 320, labelW: 132, valW: 34, min: 0, max: 70, rowH: 24, ticks: [0, 35, 70], tickFmt: v => v, aria: 'Records by type' });

    const cites = id => RECORDS.filter(r => r.id !== id && r.text.includes(id)).length;
    $('#rules-table tbody', el).innerHTML = RECORDS.filter(r => r.type === 'rule').map(r => `<tr data-record="${esc(r.id)}"><td>${chip(r.id)}</td><td>${esc(r.title)}</td><td class="small">${esc(((r.meta && r.meta.category) || '').replace(/_/g, ' '))}</td><td class="num">${cites(r.id)}</td></tr>`).join('');
    const WFR = { 'WF-01': ['DCLAB-R01', 'DCLAB-R18'], 'WF-02': ['DCLAB-R09', 'DCLAB-R21'], 'WF-03': ['DCLAB-R02', 'DCLAB-R17'], 'WF-04': ['DCLAB-R03', 'DCLAB-R12'], 'WF-05': ['DCLAB-R04', 'DCLAB-R05', 'DCLAB-R06'], 'WF-06': ['DCLAB-R07', 'DCLAB-R08', 'DCLAB-R10', 'DCLAB-R11'], 'WF-07': ['DCLAB-R13', 'DCLAB-R14'], 'WF-08': ['DCLAB-R15', 'DCLAB-R16'], 'WF-09': ['DCLAB-R17', 'DCLAB-R19', 'DCLAB-R22'], 'WF-10': ['DCLAB-R20', 'DCLAB-R21'] };
    $('#wf-cards', el).innerHTML = RECORDS.filter(r => r.type === 'workflow').map(r => `<div class="panel"><div class="panel-head"><div class="panel-title"><span class="eyebrow">${esc(r.id)}</span><h3>${esc(r.title)}</h3></div>${chip(r.id)}</div><div class="panel-body small stack tight"><p>${esc(r.text.replace(/^Workflow block WF-\d+ — [^:]+: /, ''))}</p>${chips(WFR[r.id] || [])}</div></div>`).join('');
    $('#ds-cards', el).innerHTML = RECORDS.filter(r => r.type === 'dataset').map(r => {
      const name = r.id.replace('DATASET-', '');
      const rel = RECORDS.filter(x => x.meta && x.meta.dataset === name && x.type !== 'dataset').map(x => x.id).slice(0, 6);
      return `<div class="panel"><div class="panel-head"><div class="panel-title"><h3>${esc(r.title)}</h3><span class="sub">${esc((r.meta && r.meta.task_type) || '')}</span></div></div><div class="panel-body small stack tight"><p class="muted">${esc(r.text.split('Decision-time contract:')[0].replace(/^Dataset `[^`]+`: [^.]+\.\s*/, '').slice(0, 220))}</p>${r.text.includes('Decision-time contract:') ? `<p><b>Solution:</b> ${esc(r.text.split('Decision-time contract:')[1].split('Blocked')[0].trim().slice(0, 200))}</p>` : ''}<div class="ev-list">${chip(r.id)}${chips(rel)}</div></div></div>`;
    }).join('');
    const ANS = {
      smote: `<div class="inset stack tight"><p><b>No. Resample only inside the training data.</b> Oversampling before the split put copies of the same rows in train and test and inflated holdout ROC-AUC by up to +0.3003 on average (online_shoppers: 0.9019 reported, 0.6016 honest) ${chip('PIT-003')}.</p><p>Use an imblearn Pipeline so it happens inside each fold, and evaluate on untouched, naturally imbalanced rows with average precision ${chip('DCLAB-R03')} ${chip('DCLAB-R18')}.</p></div>`,
      best: `<div class="inset stack tight"><p><b>There is no single best one.</b> Across 11 datasets lightgbm had the best mean rank (2.55) but won only 3 times; logistic regression won none on average, yet it won the Telco churn campaign ${chip('FINDING-model-families')} ${chip('FINDING-churn-linear')}.</p><p>Screen a linear baseline and several tree ensembles on identical folds, then keep the simplest model within noise of the best ${chip('DCLAB-R13')} ${chip('DCLAB-R14')}.</p></div>`,
      time: `<div class="inset stack tight"><p><b>It answers a different question.</b> When the model will score future rows, validate forward in time and report that number ${chip('DCLAB-R02')}.</p><p>On HyperAck the random split happened to score lower (0.9332) than the forward-in-time test (0.9371), so do not assume the direction; measure it ${chip('PIT-005')}.</p></div>`,
      gnn: `<div class="inset stack tight"><p><b>The evidence does not cover this yet.</b> The temporal-GNN and driving-maps tracks are proposals with no experiments, so any answer would be a guess.</p><p class="muted">Suggested next step: a road-speed forecasting campaign that compares a seasonal baseline, gradient boosting with graph features, and a temporal GNN on the same time-ordered splits.</p></div>`,
    };
    $('#ask-chips', el).addEventListener('click', e => { const b = e.target.closest('[data-ask]'); if (!b) return; $$('#ask-chips .chip', el).forEach(c => c.setAttribute('aria-pressed', String(c === b))); $('#ask-answer', el).innerHTML = ANS[b.dataset.ask]; });
    DC.hydrate(el);
  },
});
