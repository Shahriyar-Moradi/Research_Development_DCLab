DC.view('admin', {
  init(el) {
    const { $, esc, chip, toast } = DC;
    const POL = [
      ['The holdout opens once, after every choice is locked', 'DCLAB-R17', true, true],
      ['Forbidden columns are blocked at the tool level', 'DCLAB-R04', true, true],
      ['LLM critique is advisory; numbers come from code', 'DCLAB-R20', true, true],
      ['A research result is never labelled production-ready', 'DCLAB-R22', true, true],
      ['The solution must be signed before WF-04', 'DCLAB-R01', true, false],
      ['Tuning needs a margin declared before the run', 'DCLAB-R16', true, false],
      ['Ask the owner when a column\'s timing is unclear', 'DCLAB-R05', true, false],
      ['Allow quick mode (row sampling) for exploration', '', true, false],
      ['Let the intern run code in the sandbox', '', true, false],
      ['Let the intern start GPU jobs under the cap without asking', '', false, false],
    ];
    $('#policy-list', el).innerHTML = POL.map(([t, r, on, locked], i) => `<div class="list-item"><label class="switch" data-style="flex:1"><input type="checkbox" data-pol="${i}" ${on ? 'checked' : ''} ${locked ? 'disabled' : ''}><span class="track"></span><span>${esc(t)}</span></label>${r ? chip(r) : ''}${locked ? '<span class="pill">locked</span>' : ''}</div>`).join('');
    const ROLES = ['Owner', 'ML engineer', 'Reviewer', 'Business viewer', 'Intern'];
    const ACT = [
      ['Create a project', ['y', 'y', '-', '-', 'y']], ['Sign a solution', ['y', '-', '-', '-', '-']], ['Approve a stage', ['y', 'y', 'y', '-', 'a']],
      ['Open the holdout', ['y', 'a', '-', '-', 'a']], ['Override the rule\'s pick (with a reason)', ['y', 'y', '-', '-', '-']], ['Approve a brief', ['y', '-', '-', 'y', '-']],
      ['Start a GPU job over the cap', ['y', 'a', '-', '-', 'a']], ['Change policies', ['y', '-', '-', '-', '-']], ['See code and tool calls', ['y', 'y', 'y', '-', 'y']],
    ];
    const cell = v => `<td>${v === 'y' ? '<span class="yes">✓</span>' : v === 'a' ? '<span class="ask">asks</span>' : '<span class="no">—</span>'}</td>`;
    $('#role-matrix', el).innerHTML = `<thead><tr><th>Action</th>${ROLES.map(r => `<th>${r}</th>`).join('')}</tr></thead><tbody>${ACT.map(([a, v]) => `<tr><td>${esc(a)}</td>${v.map(cell).join('')}</tr>`).join('')}</tbody>`;

    /* Real mode: GET /api/platform/policies, /api/platform/limits, /api/intern and /api/platform/audit. */
    const S = this.S = { real: false, offset: 0, limit: 50 };
    $('#policy-list', el).addEventListener('change', async e => {
      const box = e.target;
      if (!S.real) { if (box.dataset.pol != null) toast('Sample policy list: nothing is saved.'); return; }
      const { project, key } = box.dataset;
      if (!project || !key) return;
      box.disabled = true;
      try {
        await DC.api(`/projects/${project}`, { method: 'PATCH', body: { policy: { [key]: box.checked } } });
        DC.currentProject.clear();
        toast(`Saved on this project: the gate is now ${box.checked ? 'required' : 'off'}. Switch changes are not written to the audit log yet.`);
        await this.loadPolicies(el);
      } catch (err) { box.checked = !box.checked; box.disabled = false; toast(err.message, { ok: false }); }
    });
    el.addEventListener('click', e => {
      const op = e.target.closest('[data-open-project]');
      if (op) { DC.state.projectId = op.dataset.openProject; DC.state.project = 'p:' + op.dataset.openProject; DC.currentProject.clear(); }
    });
    $('#audit-newer', el).addEventListener('click', () => { S.offset = Math.max(0, S.offset - S.limit); this.loadAudit(el); });
    $('#audit-older', el).addEventListener('click', () => { S.offset += S.limit; this.loadAudit(el); });
    DC.hydrate(el);
  },

  async enter(el) {
    const [pol, lim, intern, gw] = await Promise.allSettled([DC.api('/platform/policies'), DC.api('/platform/limits'), DC.api('/intern'), DC.api('/models')]);
    if (pol.status !== 'fulfilled') { DC.markSample(true); return; }
    this.S.real = true;
    DC.markSample(false);
    this.paintPolicies(el, pol.value);
    if (lim.status === 'fulfilled') this.paintLimits(el, lim.value);
    if (intern.status === 'fulfilled') this.paintModels(el, intern.value, gw.status === 'fulfilled' ? gw.value : null);
    this.paintStats(el, pol.value, lim.status === 'fulfilled' ? lim.value : null, intern.status === 'fulfilled' ? intern.value : null);
    this.S.offset = 0;
    await this.loadAudit(el);
  },

  async loadPolicies(el) {
    const pol = await DC.api('/platform/policies');
    this.paintPolicies(el, pol);
    this.paintStats(el, pol, this.S.limits, this.S.intern);
  },

  paintStats(el, pol, lim, intern) {
    const { $, esc } = DC;
    this.S.limits = lim; this.S.intern = intern;
    const stat = (v, l) => `<div class="stat"><span class="v">${esc(v)}</span><span class="l">${l}</span></div>`;
    const checked = pol.invariants.filter(i => i.checked), holding = checked.filter(i => i.holds).length;
    const sw = pol.switches.map(s => `${esc(s.gate)} gate on in ${s.on} of ${s.total}`).join(' · ');
    const b = lim && lim.intern_session.defaults;
    $('#admin-stats', el).innerHTML =
      stat(pol.invariants.length + pol.switches.length, `policies · ${pol.invariants.length} locked, ${holding} of ${checked.length} checked against the validator just now${holding === checked.length ? '' : ' · <b>some do not hold</b>'} · ${pol.switches.length} per-project gate switches`) +
      stat('0', 'user accounts or roles · one local workspace; the validator tells a person from the intern') +
      stat(intern ? (intern.available ? '1' : '0') : '—', intern ? (intern.available ? `model configured: ${esc(intern.model)} via ${esc(intern.endpoint)}` : 'models configured · the intern follows the standard DCLab plan') : 'model status unavailable') +
      stat(b ? b.max_steps : '—', b ? `tool calls and ${b.max_minutes} min per intern session by default · model requests are counted by the gateway` : 'budget defaults unavailable');
    if (pol.switches.length) $('#policy-sub', el).textContent = `Locked rules hold in every project; the two gate switches are set per project (${sw}).`;
  },

  paintPolicies(el, pol) {
    const { $, esc, linkIds } = DC;
    const ids = list => list.length ? `<span class="ev-list">${list.map(i => linkIds(i)).join('')}</span>` : '';
    const tags = html => `<span class="row">${html}</span>`;
    const locked = pol.invariants.map(i => `<div class="list-item"><div class="li-main">
        <div class="spread"><label class="switch"><input type="checkbox" checked disabled><span class="track"></span><span>${esc(i.title)}</span></label>
        ${tags(ids([...i.rules, ...i.evidence]) + '<span class="pill">locked</span>' + (i.checked ? `<span class="pill ${i.holds ? 'ok' : 'bad'}" title="Asked the validator just now with a probe project">${i.holds ? 'holds now' : 'does not hold'}</span>` : '<span class="pill outline" title="Enforced in the stage engine; not probed by this page">in the engine</span>'))}</div>
        <span class="li-sub">${esc(i.detail)} <span class="mono xs">${esc(i.where)}</span></span></div></div>`).join('');
    const switches = pol.switches.map(s => {
      const rows = s.projects.length ? s.projects.map(p => `<div class="row"><label class="switch"><input type="checkbox" data-project="${esc(p.id)}" data-key="${esc(s.key)}" ${p.on ? 'checked' : ''}><span class="track"></span><span></span></label><a href="#project" data-open-project="${esc(p.id)}">${esc(p.name)}</a></div>`).join('')
        : '<span class="muted">No projects yet.</span>';
      return `<div class="list-item"><div class="li-main">
        <div class="spread"><span class="li-title">${esc(s.title)}</span>${tags(ids(s.rules) + '<span class="pill accent">per project</span>')}</div>
        <span class="li-sub">${esc(s.applies)} New projects start with it ${s.default ? 'on' : 'off'}; there is no workspace default to change. On in ${s.on} of ${s.total} project${s.total === 1 ? '' : 's'}:</span>
        <div class="stack tight" data-style="margin-top:6px">${rows}</div></div></div>`;
    }).join('');
    $('#policy-list', el).innerHTML = locked + switches;

    /* Roles: no accounts today; the actor matrix is the validator's own answer */
    $('#roles-real', el).hidden = false;
    $('#roles-sub', el).textContent = 'No user accounts yet: one local workspace';
    const inv = $('#invite-btn', el);
    inv.dataset.toast = 'Not built: DCLab has no user accounts yet. It runs as one local workspace on this machine.';
    $('#roles-note', el).innerHTML = `<span class="ic">${DC.icon('info')}</span><span><b>No users, roles or sign-in today.</b> DCLab runs as one local workspace on this machine; the server issues a CSRF token at start and every write request must carry it. The one real control is <b>View as</b> in the top bar, and it is ${esc(pol.accounts.view_as)}. What is enforced is the difference between a person and the intern, below; the roles table after it is a plan.</span>`;
    const sym = c => {
      const reason = c.failed.includes('Holdout reuse confirmed');
      const t = c.status === 'allowed' ? '<span class="yes">✓</span>' : c.status === 'needs_approval' ? `<span class="ask">${reason ? 'with a reason' : 'asks'}</span>` : '<span class="no">—</span>';
      return `<td title="${esc(c.message)}">${t}</td>`;
    };
    $('#actor-matrix', el).innerHTML = `<thead><tr><th>Move</th><th>A person</th><th>The intern</th></tr></thead><tbody>${pol.actors.map(a => `<tr><td>${esc(a.action)}</td>${sym(a.person)}${sym(a.intern)}</tr>`).join('')}</tbody>`;
    $('#roles-foot', el).textContent = '"Asks" means the move waits for a person (a signature, an approval or a written reason); "—" means it is refused. Hover a cell for the validator\'s message.';
    DC.hydrate(el);
  },

  paintModels(el, m, gw) {
    const { $, esc } = DC;
    $('#models-sub', el).textContent = gw ? 'Every request goes through one gateway: each purpose uses a tier, and a local model can serve the cheap one' : 'Model status';
    const pill = $('#tier3-pill', el);
    const live = gw ? gw.tiers.some(t => t.available) : m.available;
    pill.className = 'pill ' + (live ? 'ok' : 'warn');
    pill.textContent = live ? 'configured' : 'not configured';
    if (!gw) { $('#tier3-body', el).innerHTML = `<span class="muted">${m.available ? esc(m.model) + ' via ' + esc(m.endpoint) : 'No model is configured: every part of DCLab uses its deterministic path.'}</span>`; return; }
    const tiers = gw.tiers.map(t => `<tr><td><b>${esc(t.name)}</b></td><td class="mono small">${esc(t.endpoint)}${t.local ? ' <span class="pill ok">this machine</span>' : ''}</td><td class="mono small">${esc(t.model)}</td><td><span class="pill ${t.available ? 'ok' : 'outline'}">${t.available ? 'ready' : (t.key_configured ? 'SDK missing' : 'no key')}</span></td></tr>`).join('');
    const purposes = gw.purposes.map(p => `<tr><td class="mono small">${esc(p.purpose)}</td><td>${esc(p.tier)}</td><td class="small muted">${esc(p.may_see)}</td></tr>`).join('');
    const failing = (gw.failing || []).map(f => `<div class="callout warn small"><span><b>${esc(f.model)}</b> failed the ${esc(f.purpose)} output check ${f.failed} of ${f.checked} times this month (${esc(f.tier)} tier)${f.last_reason ? ': ' + esc(f.last_reason) : ''}. Consider another model for this tier.</span></div>`).join('');
    $('#tier3-body', el).innerHTML = `${failing}
      <div class="table-wrap"><table class="data compact"><thead><tr><th>Tier</th><th>Endpoint</th><th>Model</th><th></th></tr></thead><tbody>${tiers}</tbody></table></div>
      <div class="table-wrap"><table class="data compact"><thead><tr><th>Purpose</th><th>Tier</th><th>What it may be shown</th></tr></thead><tbody>${purposes}</tbody></table></div>
      <span class="xs muted">Keys stay on the server. To change a tier set <code>DCLAB_TIER_&lt;TIER&gt;_BASE_URL</code>, <code>_MODEL</code> and <code>_API_KEY</code> in <code>.env</code> (unset tiers use <code>OPENAI_*</code>), then restart the server.</span>`;
  },

  paintLimits(el, l) {
    const { $, esc } = DC;
    const n = v => Number(v).toLocaleString('en-US');
    const box = (label, value, sub) => `<div class="field"><span class="label">${esc(label)}</span><div class="inset small">${value}${sub ? `<div class="xs muted">${sub}</div>` : ''}</div></div>`;
    const i = l.intern_session, d = l.draft_settings;
    $('#spend-pill', el).hidden = l.spend.tracked;
    $('#budget-body', el).innerHTML =
      box('Per intern session', `<b>${i.defaults.max_steps} tool calls · ${i.defaults.max_minutes} min</b> <span class="pill ok">enforced</span>`, `Caps: ${i.caps.max_steps[0]}–${i.caps.max_steps[1]} calls, ${i.caps.max_minutes[0]}–${i.caps.max_minutes[1]} min. ${esc(i.note)}`) +
      box('Per project (new-project wizard)', `${d.defaults.calls} calls · ${d.defaults.minutes} min · €${d.defaults.eur}`, `Caps: ${d.caps.calls[0]}–${d.caps.calls[1]} calls, ${d.caps.minutes[0]}–${d.caps.minutes[1]} min, €${d.caps.eur[0]}–${n(d.caps.eur[1])}. ${esc(d.note)}`) +
      box('Rows and folds', `${n(d.defaults.max_rows)} rows · ${d.defaults.folds} folds by default`, `Quick mode uses ${n(l.rows.quick)} rows; full runs ${n(l.rows.caps[0])}–${n(l.rows.caps[1])}. Folds ${d.caps.folds[0]}–${d.caps.folds[1]}.`) +
      box('Model spend this month', l.spend.tracked
        ? `<b>€${Number(l.spend.eur_this_month || 0).toFixed(2)}</b>${l.spend.unpriced_this_month ? ` <span class="pill warn">${n(l.spend.unpriced_this_month)} request${l.spend.unpriced_this_month === 1 ? '' : 's'} without a price</span>` : ''}${l.spend.workspace_cap != null ? ` · workspace cap €${l.spend.workspace_cap}` : ''}`
        : '<span class="muted">Not tracked</span>', esc(l.spend.note));
    const mb = v => Math.round(v / 1048576) + ' MB';
    $('#privacy-body', el).innerHTML = l.privacy.map(p => `<div class="inset stack tight"><div class="spread"><b>${esc(p.title)}</b><span class="pill ${p.on ? 'ok' : 'outline'}">${p.on ? 'yes' : 'no'}</span></div><span class="muted">${esc(p.text)}</span></div>`).join('') +
      `<dl class="kv"><dt>Upload limit</dt><dd>${mb(l.upload_max_bytes)} per file · ${mb(l.connector_max_bytes)} per connector import</dd><dt>Workspace folder</dt><dd class="mono xs">${esc(l.storage)}</dd></dl>`;
  },

  async loadAudit(el) {
    const { $, esc, linkIds } = DC;
    const S = this.S;
    let a;
    try { a = await DC.api(`/platform/audit?limit=${S.limit}&offset=${S.offset}`); } catch (e) { $('#audit-body', el).innerHTML = `<tr><td colspan="3"><div class="empty">${esc(e.message)}</div></td></tr>`; return; }
    const when = s => { const t = new Date(s); return isNaN(t) ? esc(s || '') : t.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }); };
    const what = e => {
      const g = e.args || {};
      const verb = {
        run_stage: `${e.status === 'allowed' ? 'Ran' : 'Asked to run'} stage <code>${esc(g.stage || '')}</code>${g.reuse_reason ? ' (holdout reuse: ' + esc(g.reuse_reason) + ')' : ''}`,
        set_solution: `${e.status === 'allowed' ? 'Saved' : 'Asked to change'} the solution${g.target ? ' · target <code>' + esc(g.target) + '</code>' : ''}`,
        approve_stage: `${e.status === 'allowed' ? 'Approved' : 'Asked to approve'} stage <code>${esc(g.stage || '')}</code>${g.choice ? ' · chose <code>' + esc(g.choice) + '</code>' : ''}`,
        approve_gate: `${e.status === 'allowed' ? 'Approved' : 'Asked to approve'} the ${esc(g.gate || '')} gate`,
        capture: 'Exported the notebook and report',
        use_approval: `Used the ${esc(g.gate || '')} approval`,
        set_policy: `Turned ${g.on ? 'on' : 'off'} the ${esc(g.label || g.policy || '')} switch`,
      }[e.move] || esc(e.move || '');
      const failed = (e.outcome || '').startsWith('failed');
      const pill = e.status === 'blocked' ? '<span class="pill bad">refused</span>' : e.status === 'needs_approval' ? '<span class="pill warn">waited for a person</span>' : failed ? '<span class="pill bad">failed</span>' : '';
      const detail = e.status !== 'allowed' ? e.message : e.kind === 'approval' && e.message ? 'Reason: ' + e.message : (e.outcome || '').replace(/^done: /, '');
      return `<div class="row">${verb} ${pill}<span class="muted">· <a href="#project" data-open-project="${esc(e.project.id)}">${esc(e.project.name)}</a></span></div>${detail ? `<div class="xs muted">${linkIds(detail.length > 220 ? detail.slice(0, 220) + '…' : detail)}</div>` : ''}`;
    };
    $('#audit-sub', el).textContent = `Every move the workflow validator checked, allowed or refused, every gate approval and every gate switch change, across ${a.projects} project${a.projects === 1 ? '' : 's'}. Written by the app to each project's transition and activity logs.`;
    $('#audit-body', el).innerHTML = a.items.length ? a.items.map(e => `<tr><td class="mono small" data-style="white-space:nowrap">${when(e.at)}</td><td>${esc(e.who)}</td><td class="small">${what(e)}</td></tr>`).join('')
      : `<tr><td colspan="3"><div class="empty">${a.total ? 'No more entries.' : 'Nothing logged yet: the log starts when a project saves its solution.'}</div></td></tr>`;
    $('#audit-foot', el).hidden = !a.total;
    $('#audit-range', el).textContent = a.total ? `${a.offset + 1}–${Math.min(a.offset + a.items.length, a.total)} of ${a.total} · ${a.counts.blocked} refused · ${a.counts.waiting} waited for a person · ${a.counts.approvals} approvals` : '';
    $('#audit-newer', el).disabled = a.offset === 0;
    $('#audit-older', el).disabled = a.offset + a.items.length >= a.total;
    const tab = $('.ptab[data-ptab="audit"]', el);
    if (tab) tab.dataset.total = a.total;
    DC.hydrate(el);
  },
});
