DC.view('benchmark', {
  init(el) {
    const { $, esc, chip, chips, icon } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id)).replace(/\$\{icon:([a-z]+)\}/g, (m, n) => icon(n));
    const B = [
      ['Standard plan (no model)', '100%', '61%', '4%', '100%', '0%', '0', '€0.00', ''],
      ['General LLM (open model, router)', '91%', '78%', '11%', '86%', '58%', '3', '€0.09', ''],
      ['DCLab policy model v0.1', '98%', '84%', '6%', '95%', '81%', '0', '€0.01', 'accent'],
      ['Human reviewer', '—', '89%', '5%', '—', '93%', '0', '€14.00', ''],
    ];
    $('#board tbody', el).innerHTML = B.map(r => `<tr class="${r[8] ? 'selected' : ''}"><td><span class="cell-main">${esc(r[0])}</span>${r[8] ? ' <span class="pill accent">target</span>' : ''}</td>${r.slice(1, 8).map((v, i) => `<td class="num mono" data-style="${i === 5 && v !== '0' ? 'color:var(--bad);font-weight:600' : ''}">${v}</td>`).join('')}</tr>`).join('');
    const TRAJ = {
      std: [['run_stage · leakage', 'The column auditor sees no suspicious column.', 'warn'], ['run_stage · models', 'Continues; the fitted clustering is not a column, so nothing flags it.', 'warn'], ['verdict', 'Missed the trap. This is the pilot\'s real miss, now a test case.', 'bad']],
      llm: [['review_code', 'Notices KMeans but calls it "probably fine for an unsupervised step".', 'warn'], ['ask_owner', 'Asks the owner whether zones are allowed. A question nobody needed.', 'warn'], ['verdict', 'Partial: found it, did not act, spent a question.', 'warn']],
      pol: [['review_code', 'Flags KMeans fitted on all rows before the split.', 'ok'], ['propose · fix', 'Moves the clustering into the pipeline so it fits inside each fold.', 'ok'], ['validator', 'Fix accepted; evidence cited: DCLAB-R03, PIT-001.', 'ok'], ['verdict', 'Pass: caught, fixed, no unnecessary question.', 'ok']],
    };
    function traj(p) {
      $('#case-traj', el).innerHTML = `<div class="timeline">${TRAJ[p].map(([t, d, c], i) => `<div class="tl-item"><span class="tl-mark ${c === 'ok' ? 'ok' : c === 'bad' ? 'bad' : 'warn'}">${i + 1}</span><div class="tl-body"><span class="tl-title mono small">${esc(t)}</span><span class="small">${DC.linkIds(d)}</span></div></div>`).join('')}</div><div class="xs muted">Sample trajectories; the first is the pilot's real miss.</div>`;
      DC.hydrate(el);
    }
    $('#case-pol', el).addEventListener('segchange', e => traj(e.detail));
    traj('std');
  },
});
