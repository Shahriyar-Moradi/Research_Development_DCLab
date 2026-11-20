/* Projects (v6, the owner 2026-11-20): the list every project opens from. Reads GET /api/workspace; opening one makes it the
   project every project page follows, then shows its dashboard. Nothing here changes a project. */
DC.view('projects', {
  init(el) {
    const { $ } = DC;
    $('#pj-new', el).addEventListener('click', () => { const b = $('#ds-new-chat'); if (b) b.click(); else location.hash = '#home'; });
    $('#pj-q', el).addEventListener('input', () => this.draw(el));
    el.addEventListener('click', e => {
      const q = e.target.closest('[data-pj-draft]');
      if (q) { e.preventDefault(); DC.state.resumeDraft = q.dataset.pjDraft; location.hash = '#home'; return; }
      const b = e.target.closest('[data-pj-open]');
      if (!b) return;
      const id = b.dataset.pjOpen, S = DC.state;
      if ((DC.projectSwitch.list || []).some(p => p.id === id)) DC.projectSwitch.select(id);
      else { S.project = 'p:' + id; S.projectId = null; DC.currentProject.clear(); try { localStorage.setItem('dclab-project', id); } catch (err) { /* storage blocked */ } }
      location.hash = b.dataset.pjTo || '#dashboard';
      e.preventDefault();
    });
  },
  async enter(el) {
    const { $, esc } = DC;
    const ws = await DC.client.workspace();
    this.ws = ws;
    const list = ws.projects || [], needs = ws.needs || [];
    const steps = p => String(p.state || '');
    const done = list.filter(p => steps(p) && !/[^d]/.test(steps(p))).length;
    const running = list.filter(p => p.running).length;
    $('#pj-stats', el).innerHTML = [
      ['Projects', list.length], ['Running now', running], ['Waiting for you', needs.length], ['All steps done', done],
    ].map(([k, v]) => `<div class="stat"><div class="l">${esc(k)}</div><div class="v num">${esc(v)}</div></div>`).join('');
    $('#pj-needs', el).innerHTML = needs.length ? `<div class="panel"><div class="panel-head"><div class="panel-title"><h2>Waiting for you</h2></div></div><div class="panel-body"><ul class="dash-list">${needs.slice(0, 5).map(n =>
      `<li><a href="${n.project ? '#solution' : '#home'}"${n.project ? ` data-pj-open="${esc(n.project)}" data-pj-to="#solution"` : n.draft ? ` data-pj-draft="${esc(n.draft)}"` : ''}><span class="dash-li-main">${esc(n.title)}<small>${esc(n.sub || '')}</small></span>${n.tag ? `<span class="pill outline">${esc(n.tag)}</span>` : ''}</a></li>`).join('')}</ul></div></div>` : '';
    $('#pj-count', el).textContent = String(list.length);
    this.draw(el);
  },
  draw(el) {
    const { $, esc, int } = DC;
    const ws = this.ws || {}, q = ($('#pj-q', el).value || '').trim().toLowerCase();
    const list = (ws.projects || []).filter(p => !q || (String(p.name || '') + ' ' + String(p.goal || '') + ' ' + String((p.data || {}).filename || '')).toLowerCase().includes(q));
    const box = $('#pj-box', el);
    if (!(ws.projects || []).length) {
      DC.states.empty(box, 'No project yet. Describe a problem on Home, or attach a table there, and DCLab builds the project with you.', { href: '#home', action: 'Go to Home' });
      return;
    }
    if (!list.length) { box.innerHTML = '<p class="muted small">No project matches that search.</p>'; return; }
    const stage = p => {
      const st = String(p.state || ''), n = st.length, d = (st.match(/d/g) || []).length;
      if (p.running) return ['Running', 'info'];
      if (/f/.test(st)) return ['A step failed', 'bad'];
      if (n && d === n) return ['All steps done', 'ok'];
      if (/w/.test(st)) return ['Waiting for you', 'warn'];
      return [n ? `Step ${Math.min(d + 1, n)} of ${n}` : 'Not started', 'outline'];
    };
    const when = x => { const t = Date.parse(x || ''); return isNaN(t) ? '' : new Date(t).toLocaleDateString(undefined, { day: 'numeric', month: 'short' }); };
    box.innerHTML = `<div class="table-wrap"><table class="data"><thead><tr><th>Project</th><th>Goal</th><th>Where it is</th><th>Updated</th><th class="r"></th></tr></thead><tbody>${list.map(p => {
      const d = p.data || {}, [label, cls] = stage(p);
      return `<tr><td><div class="cell-main">${esc(p.name || p.id)}</div><div class="cell-sub">${esc(d.filename || 'No table yet')}${d.rows ? ' · ' + int(d.rows) + ' rows' : ''}${d.synthetic ? ' · synthetic' : ''}</div></td>`
        + `<td class="dash-goal">${esc(p.goal || '—')}</td><td><span class="pill ${cls}">${esc(label)}</span></td><td class="muted">${esc(when(p.updated))}</td>`
        + `<td class="r"><button type="button" class="btn sm" data-pj-open="${esc(p.id)}">Open</button></td></tr>`;
    }).join('')}</tbody></table></div>`;
  },
});
