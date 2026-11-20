/* A project's dashboard (v6, the owner 2026-11-20: "Home must move as the dashboard section of each project"). It only reads the open
   project (GET /api/projects/{id}): its table, its solution, the stage records and the workflow graph. Every number links to the
   page that made it; nothing is computed here that a stage did not record. */
DC.view('dashboard', {
  async enter(el) {
    const { $, esc, int } = DC;
    const box = $('#db-box', el);
    const p = await DC.currentProject.get();
    DC.markSample(false);
    if (!p) {
      $('#db-eyebrow', el).textContent = 'Project · dashboard';
      $('#db-title', el).textContent = 'Dashboard';
      DC.states.empty(box, 'No project is open yet. Start one from a sentence or a table on Home; its dashboard appears here.', { href: '#home', action: 'Go to Home' });
      return;
    }
    const LABEL = { roc_auc: 'ROC-AUC', average_precision: 'Avg. precision', f1: 'F1', macro_f1: 'Macro F1', accuracy: 'Accuracy', mae: 'MAE', rmse: 'RMSE', r2: 'R²', pr_auc: 'PR-AUC', log_loss: 'Log loss' };
    const R = p.records || {}, ev = (R.models || {}).evidence || {}, sol = p.solution || {}, d = p.data || {};
    const metric = (R.models || {}).primary_metric || sol.metric || 'roc_auc', mlabel = LABEL[metric] || metric;
    const ranking = (ev.ranking || []).filter(r => r && r.mean != null);
    const best = ranking[0], uses = Number(p.holdout_uses) || 0;
    const forbidden = (sol.forbidden || []).length;
    const g = p.graph || {}, nodes = g.nodes || [], here = nodes.find(n => n.id === g.current);
    const doneN = nodes.filter(n => n.state === 'done').length;
    $('#db-eyebrow', el).textContent = `${p.name} · dashboard`;
    $('#db-title', el).textContent = p.name || 'Dashboard';
    $('#db-lede', el).textContent = p.goal || 'Where this project stands: its numbers, the current step and its files.';

    const stat = (k, v, sub, href) => `<a class="stat dash-stat" href="${href}"><div class="l">${esc(k)}</div><div class="v num">${esc(v)}</div>${sub ? `<div class="s">${sub}</div>` : ''}</a>`;
    const stats = [
      stat('Rows', d.rows ? int(d.rows) : '—', d.filename ? esc(d.filename) : 'No table yet', '#solution'),
      stat('Columns left out', String(forbidden), forbidden ? '<span class="pill warn">known too late or an ID</span>' : 'none so far', '#audit'),
      stat(`Best validation ${mlabel}`, best ? Number(best.mean).toFixed(3) : '—', best ? esc(best.model) : 'after the models stage', '#models'),
      stat('Final test', uses === 0 ? 'Sealed' : uses === 1 ? 'Read once' : `Read ${uses} times`,
        uses === 0 ? '<span class="pill outline">runs once, at the end</span>' : uses === 1 ? '<span class="pill ok">used once</span>' : '<span class="pill warn">reopened: see Reliability</span>', '#reliability'),
    ].join('');

    /* Model ranking: one bar per candidate, on the stage's own scale (0 to 1 for a score, from 0 for an error). */
    let chart = '<p class="muted small">The models stage has not run yet. Its candidates and their validation scores appear here.</p>';
    if (ranking.length) {
      const rows = ranking.slice(0, 8), W = 560, H = 230, L = 44, B = 46, T = 10, R2 = 8;
      const hi = Math.max(1, ...rows.map(r => r.mean)) > 1 ? Math.max(...rows.map(r => r.mean)) * 1.1 : 1;
      const y = v => T + (H - T - B) * (1 - v / hi), slot = (W - L - R2) / rows.length, bw = Math.min(30, slot * 0.5);
      const ticks = [0, hi / 2, hi];
      chart = `<svg class="dash-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Validation ${esc(mlabel)} by model">`
        + ticks.map(t => `<line class="gl" x1="${L}" x2="${W - R2}" y1="${y(t)}" y2="${y(t)}"/><text x="${L - 8}" y="${y(t) + 4}" text-anchor="end">${t.toFixed(hi > 1 ? 0 : 1)}</text>`).join('')
        + rows.map((r, i) => { const cx = L + slot * i + slot / 2, top = y(r.mean), nm = String(r.model).replace(/_/g, ' ');
          return `<rect class="${i === 0 ? 'bar best' : 'bar'}" x="${cx - bw / 2}" y="${top}" width="${bw}" height="${Math.max(0, y(0) - top)}" rx="${bw / 2}"><title>${esc(nm)}: ${Number(r.mean).toFixed(3)}</title></rect>`
            + `<text x="${cx}" y="${H - 26}" text-anchor="middle">${esc(nm.length > 12 ? nm.slice(0, 11) + '…' : nm)}</text><text x="${cx}" y="${H - 10}" text-anchor="middle" class="v">${Number(r.mean).toFixed(3)}</text>`; }).join('')
        + '</svg>';
    }

    const stateCls = { done: 'ok', current: 'info', waiting: 'warn', running: 'info', failed: 'bad', todo: 'outline' };
    const stateTxt = { done: 'Done', current: 'Now', waiting: 'Waiting for you', running: 'Running', failed: 'Failed', todo: 'Not started' };
    const next = nodes.filter(n => n.state !== 'done').slice(0, 3);
    const work = nodes.length
      ? `<div class="dash-kv"><span>Current step</span><span>${here ? esc(here.name) : 'All steps done'}</span></div>`
        + next.map(n => `<div class="dash-kv"><span>${esc(n.name)}</span><span class="pill ${stateCls[n.state] || 'outline'}">${esc(stateTxt[n.state] || n.state)}</span></div>`).join('')
        + `<div class="dash-prog" role="progressbar" aria-valuemin="0" aria-valuemax="${nodes.length}" aria-valuenow="${doneN}" aria-label="Steps done"><span data-style="width:${Math.round(100 * doneN / nodes.length)}%"></span></div>`
        + `<p class="xs muted">${doneN} of ${nodes.length} steps done</p>`
      : '<p class="muted small">The workflow graph has not been drawn for this project yet.</p>';

    const st = p.stages || {}, sOf = k => (st[k] || {}).status;
    const ready = k => sOf(k) === 'completed' || sOf(k) === 'approved';
    const files = [
      ['db', d.filename || 'The table', d.rows ? `${int(d.rows)} rows` : 'not attached', '#solution'],
      ['shield', 'Leakage review', ready('leakage') ? `${forbidden} left out` : 'not run', '#audit'],
      ['layers', 'Features', ready('features') ? 'ready' : 'not run', '#features'],
      ['models', 'Model comparison', ranking.length ? `${ranking.length} candidates` : 'not run', '#models'],
      ['notebook', 'Notebook', 'Python', '#notebook'],
      ['brief', 'Decision brief', ready('final') ? 'ready to sign' : 'after the final stage', '#brief'],
    ];
    const fileList = `<ul class="dash-list">${files.map(f => `<li><a href="${f[3]}">${DC.icon(f[0])}<span class="dash-li-main">${esc(f[1])}</span><span class="muted small">${esc(f[2])}</span></a></li>`).join('')}</ul>`;

    const moves = (p.transitions || []).slice(-6).reverse();
    const when = x => { const t = Date.parse(x || ''); return isNaN(t) ? '' : new Date(t).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }); };
    const moveTxt = m => { const a = m.args || {}; return String(m.move || '').replace(/_/g, ' ') + (a.stage ? ' · ' + a.stage : ''); };
    const activity = moves.length
      ? `<ul class="dash-list">${moves.map(m => `<li><span class="dash-li-main">${esc(moveTxt(m))}<small>${esc(m.actor || '')}${m.at ? ' · ' + esc(when(m.at)) : ''}</small></span><span class="pill ${m.status === 'blocked' ? 'bad' : 'outline'}">${esc(m.status || '')}</span></li>`).join('')}</ul>`
      : '<p class="muted small">Nothing has been run yet.</p>';

    box.innerHTML = `<div class="dash-stats">${stats}</div>`
      + `<div class="dash-row"><div class="panel"><div class="panel-head"><div class="panel-title"><h2>Model quality</h2></div><a class="link-btn small" href="#models">Models</a></div><div class="panel-body">${chart}</div></div>`
      + `<div class="panel"><div class="panel-head"><div class="panel-title"><h2>Current work</h2></div><a class="link-btn small" href="#project">Workflow</a></div><div class="panel-body">${work}</div></div></div>`
      + `<div class="dash-row"><div class="panel"><div class="panel-head"><div class="panel-title"><h2>Project files</h2></div></div><div class="panel-body">${fileList}</div></div>`
      + `<div class="panel"><div class="panel-head"><div class="panel-title"><h2>Recent activity</h2></div></div><div class="panel-body">${activity}</div></div></div>`;
    DC.applyDataStyles(box);
  },
});
