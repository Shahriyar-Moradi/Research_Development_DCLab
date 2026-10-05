DC.view('new', {
  init(el) {
    const { $, $$, esc, chip, icon } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id)).replace(/\$\{icon:([a-z]+)\}/g, (m, n) => icon(n));
    const packs = [
      ['Tabular', 'Binary, multiclass and regression on tables.', 'GA', 'ok', 'models', true],
      ['Imbalanced and fraud', 'Rare positives: PR-AUC, cost curves, time splits.', 'GA', 'ok', 'alert'],
      ['Time series', 'Forecasts with horizons and backtests against naive baselines.', 'Beta', 'info', 'trend'],
      ['Text + tabular', 'Reviews, tickets and notes next to structured fields.', 'Beta', 'info', 'text'],
      ['Computer vision', 'Detection and classification, split by scene or drive.', 'Preview', 'warn', 'image'],
      ['Driving perception', 'Operating domain (ODD) as the solution; split by trip and route.', 'Research', '', 'car'],
      ['Maps and road flow', 'Road graphs, speed and flow forecasts.', 'Research', '', 'route'],
      ['Scene graphs', 'Objects, relations and events in video.', 'Research', '', 'nodes'],
      ['LLM and SLM fine-tuning', 'Training data, LoRA runs and evaluation.', 'Beta', 'info', 'brain'],
    ];
    $('#pack-grid', el).innerHTML = packs.map(([n, d, m, c, ic, on]) => `<button type="button" class="pack-card" data-toggle aria-pressed="${on ? 'true' : 'false'}"><span class="pc-top"><span class="pack-ic">${icon(ic)}</span><span class="pill ${c}">${m}</span></span><span class="pc-name">${n}</span><span class="pc-desc">${d}</span>${on ? '<span class="pc-foot"><span class="pill accent">recommended</span></span>' : ''}</button>`).join('');
    $('#pack-grid', el).addEventListener('togglechange', e => { $$('.pack-card', el).forEach(c => { if (c !== e.target) c.setAttribute('aria-pressed', 'false'); }); });

    const samples = [
      ['bank_marketing', 'binary', '45,211', 'Immediately before a call; duration forbidden'],
      ['hyperack', 'binary', '11,118', 'At order time; final fares forbidden'],
      ['telco_churn', 'binary', '7,043', 'Snapshot; customerID forbidden'],
      ['credit_card_fraud', 'binary · imbalanced', '284,807', 'At authorization; elapsed Time forbidden'],
      ['online_shoppers', 'binary', '12,330', 'Before purchase; PageValues forbidden'],
      ['bike_sharing_daily', 'regression · time', '728', 'Day ahead; casual and registered forbidden'],
      ['letter_recognition', 'multiclass', '20,000', 'Image features only'],
      ['ecommerce_clothing_reviews', 'text + tabular', '23,486', 'At review time; Rating forbidden'],
    ];
    $('#sample-table tbody', el).innerHTML = samples.map(([k, t, r, c], i) => `<tr class="${i === 0 ? 'selected' : ''}"><td><span class="cell-main">${k}</span></td><td>${t}</td><td class="num">${r}</td><td class="small">${c}</td><td>${chip('DATASET-' + k)}</td></tr>`).join('');
    $('#sample-table tbody', el).addEventListener('click', e => { const tr = e.target.closest('tr'); if (!tr || e.target.closest('[data-record]')) return; $$('#sample-table tr', el).forEach(r => r.classList.toggle('selected', r === tr)); if (tr.rowIndex !== 1) DC.toast('In the demo the column profile stays on bank_marketing.'); });

    const cols = [
      ['age', 'number', '77', '58', 'input'], ['job', 'category (coded)', '12', '4', 'input'], ['marital', 'category (coded)', '3', '1', 'input'],
      ['education', 'category (coded)', '4', '2', 'input'], ['default', 'category (coded)', '2', '0', 'input'], ['balance', 'number', '7,168', '2143', 'input'],
      ['housing', 'category (coded)', '2', '1', 'input'], ['loan', 'category (coded)', '2', '0', 'input'], ['contact', 'category (coded)', '3', '2', 'input'],
      ['day_of_week', 'number (day of month)', '31', '5', 'calendar, no year'], ['month', 'category (coded)', '12', '8', 'calendar, no year'],
      ['duration', 'number', '1,573', '261', '<span class="pill bad">post-outcome</span>'], ['campaign', 'number', '48', '1', '<span class="pill warn">ask owner</span>'],
      ['pdays', 'number', '559', '-1', 'input · −1 means never'], ['previous', 'number', '41', '0', 'input'], ['poutcome', 'category (coded)', '4', '3', 'input'],
      ['y', 'binary', '2', '0', '<span class="pill accent">target</span>'],
    ];
    $('#profile-table tbody', el).innerHTML = cols.map(([n, k, u, ex, g]) => `<tr><td><code>${n}</code></td><td>${k}</td><td class="num">${u}</td><td class="mono small">${ex}</td><td>${g}</td></tr>`).join('');

    const show = n => {
      $$('[data-new-step]', el).forEach(p => { p.hidden = p.dataset.newStep !== String(n); });
      $$('#new-steps button', el).forEach(b => { const s = Number(b.dataset.step); b.classList.toggle('done', s < n); if (s === n) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current'); });
      DC.hydrate(el);
    };
    el.addEventListener('click', e => { const b = e.target.closest('[data-next]'); if (b) { show(Number(b.dataset.next)); window.scrollTo({ top: 0, behavior: 'smooth' }); } const s = e.target.closest('#new-steps button'); if (s) show(Number(s.dataset.step)); });
    this.show = show;
  },
  enter() { if (DC.state.tour) this.show(3); },
});
