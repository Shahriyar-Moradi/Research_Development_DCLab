DC.view('blueprint', {
  init(el) {
    const { $, $$, esc, chip, FEATURES, STATUS_LABEL, Decisions, decideButtons, charts, copyText, toast } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id));
    const VIEW_LABEL = { shell: 'Everywhere', home: 'Home', new: 'New project', project: 'Workflow graph', solution: 'Solution & data', notebook: 'Notebook', audit: 'Leakage & features', models: 'Models', reliability: 'Reliability', brief: 'Decision brief', intern: 'Intern', compute: 'Compute & jobs', evidence: 'Evidence', lab: 'Research lab', benchmark: 'Benchmark', policy: 'Policy model', packs: 'Domain packs', integrations: 'Integrations', admin: 'Admin' };
    const PH = [
      { k: 'Now', t: 'Exists today', g: 'The evidence engine, the notebook stages, the intern, MCP and Chat UI.', exit: 'Already in the repository and tested.' },
      { k: 'P1', t: 'Proof core', g: 'Make every move checkable: the graph, the validator, the log, questions and approvals, the brief, and the first benchmark suites.', exit: 'Term-deposit calls and HyperAck run end to end inside the graph; suites A–D score the standard plan and one LLM.' },
      { k: 'P2', t: 'Reach', g: 'Code in a sandbox, live notebook cells with fixes, GPU jobs, connectors, VS Code, roles.', exit: 'A new project from Kaggle or a warehouse runs with code cells; a GPU job runs under a cap.' },
      { k: 'P3', t: 'Domain packs', g: 'Vision first, then complete time series and text; the pack builder.', exit: 'The pedestrian experiment runs the ten steps with drive-level splits and slice reports.' },
      { k: 'P4', t: 'Own model', g: 'Trajectories, expert labels, training, the benchmark gates, then serving as the intern\'s model.', exit: 'The policy model passes the gates and serves 10% of intern sessions.' },
      { k: 'R', t: 'Research', g: 'Macro-actions, maps, scene graphs, driving, software: experiments before designs.', exit: 'Each idea has a controlled experiment and a record in the index.' },
    ];
    const counts = s => FEATURES.filter(f => f.status === s).length;
    $('#bp-total', el).textContent = FEATURES.length + ' features';
    function statusPanel() {
      ['built', 'partial', 'todo'].forEach(s => { $('#bp-n-' + s, el).innerHTML = `${counts(s)} <small>of ${FEATURES.length}</small>`; });
      $('#bp-status', el).innerHTML = charts.barsH(['built', 'partial', 'todo', 'research'].map(s => ({ label: STATUS_LABEL[s], value: counts(s), valueText: String(counts(s)), cls: s === 'built' ? 'ok' : s === 'partial' ? 'warn' : s === 'todo' ? 'proof' : 'muted' })),
        { width: 300, labelW: 74, valW: 32, min: 0, max: Math.max(...['built', 'partial', 'todo', 'research'].map(counts)) + 5, rowH: 26, noAxis: true, ticks: [], aria: 'Features by status' });
    }
    function decisionsPanel() {
      const c = Decisions.counts(), n = c.must + c.later + c.cut;
      $('#bp-decisions', el).innerHTML = `<div class="stack tight small">
        <div class="meter-row"><span><b>${n}</b> of ${FEATURES.length} decided</span><span class="mono">${Math.round(n / FEATURES.length * 100)}%</span><div class="meter"><span data-style="width:${n / FEATURES.length * 100}%"></span></div></div>
        <div class="row"><span class="pill accent">${c.must} must</span><span class="pill info">${c.later} later</span><span class="pill bad">${c.cut} cut</span></div>
        <div class="xs muted">${Decisions.backend === 'shared' ? 'Saved to the shared plan. Claude can read these decisions.' : 'Saved in this browser only. Copy the Markdown to share it.'}</div></div>`;
      $('#bp-store', el).textContent = Decisions.backend === 'shared' ? 'shared plan' : 'this browser';
      $('#bp-n-decided', el).innerHTML = `${n} <small>of ${FEATURES.length}</small>`;
    }
    function pagesPanel() {
      const views = Object.keys(VIEW_LABEL);
      $('#bp-pages', el).innerHTML = views.map(v => {
        const fs = FEATURES.filter(f => f.view === v); if (!fs.length) return '';
        const seg = s => fs.filter(f => f.status === s).length;
        const bar = ['built', 'partial', 'todo', 'research'].map(s => seg(s) ? `<span class="mini-bar ${s === 'built' ? '' : s === 'partial' ? 'warn' : s === 'todo' ? 'proof' : 'muted'}" data-style="width:${seg(s) * 9}px;${s === 'built' ? 'background:var(--ok)' : ''}" title="${seg(s)} ${STATUS_LABEL[s]}"></span>` : '').join('');
        return `<div class="spread small" data-style="padding:4px 4px"><a href="#${v === 'shell' ? 'home' : v}">${esc(VIEW_LABEL[v])}</a><span class="row" data-style="gap:2px;flex-wrap:nowrap">${bar}</span></div>`;
      }).join('');
    }
    function phasesPanel() {
      $('#bp-phases', el).innerHTML = PH.map(p => {
        const fs = FEATURES.filter(f => f.phase === p.k && (p.k === 'Now' ? f.status === 'built' : f.status !== 'built'));
        const must = fs.filter(f => Decisions.get(f.id) === 'must').length, cut = fs.filter(f => Decisions.get(f.id) === 'cut').length;
        return `<div class="panel"><div class="panel-head"><div class="panel-title"><span class="eyebrow">${p.k === 'Now' ? 'Now' : p.k === 'R' ? 'Research' : 'Phase ' + p.k.slice(1)}</span><h3>${esc(p.t)}</h3></div><span class="pill ${p.k === 'Now' ? 'ok' : p.k === 'P1' ? 'accent' : p.k === 'R' ? '' : 'outline'}">${fs.length} features</span></div>
          <div class="panel-body small stack tight"><p>${esc(p.g)}</p><p class="muted"><b>Done when:</b> ${esc(p.exit)}</p>
          <details><summary class="link-btn small" data-style="cursor:pointer">List the features</summary><ul class="small" data-style="margin-top:6px">${fs.map(f => `<li>${esc(f.name)} <span class="faint">· ${esc(VIEW_LABEL[f.view] || f.view)}</span></li>`).join('')}</ul></details>
          ${p.k !== 'Now' ? `<div class="row"><span class="pill accent">${must} must</span>${cut ? `<span class="pill bad">${cut} cut</span>` : ''}</div>` : ''}</div></div>`;
      }).join('');
    }
    const openNotes = new Set();
    $('#bpf-view', el).innerHTML = '<option value="">Any page</option>' + Object.entries(VIEW_LABEL).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join('');
    function table() {
      const q = $('#bpf-q', el).value.toLowerCase(), v = $('#bpf-view', el).value, s = $('#bpf-status', el).value, p = $('#bpf-phase', el).value, d = $('#bpf-dec', el).value;
      const rows = FEATURES.filter(f => (!q || (f.name + ' ' + f.where + ' ' + f.note).toLowerCase().includes(q)) && (!v || f.view === v) && (!s || f.status === s) && (!p || f.phase === p) && (!d || (d === 'none' ? !Decisions.get(f.id) : Decisions.get(f.id) === d)));
      $('#bp-count', el).textContent = `${rows.length} of ${FEATURES.length} shown`;
      $('#bp-table tbody', el).innerHTML = rows.map(f => `<tr data-bp="${f.status}">
        <td><div class="cell-main">${esc(f.name)}</div>${f.where || f.note ? `<div class="cell-sub">${f.where ? `<span class="mono">${esc(f.where)}</span>` : ''}${f.where && f.note ? ' · ' : ''}${esc(f.note)}</div>` : ''}
          ${Decisions.note(f.id) || openNotes.has(f.id) ? `<input type="text" class="bp-note" data-note="${esc(f.id)}" placeholder="Your note" value="${esc(Decisions.note(f.id))}" aria-label="Note for ${esc(f.name)}" data-style="margin-top:6px;padding:4px 8px;font-size:12px">` : `<button type="button" class="link-btn xs" data-open-note="${esc(f.id)}">+ note</button>`}</td>
        <td class="small"><a href="#${f.view === 'shell' ? 'home' : f.view}">${esc(VIEW_LABEL[f.view] || f.view)}</a></td>
        <td><span class="bp-status">${STATUS_LABEL[f.status]}</span></td>
        <td class="mono small">${esc(f.phase)}</td>
        <td>${decideButtons(f.id)}</td></tr>`).join('') || '<tr><td colspan="5"><div class="empty">No feature matches these filters.</div></td></tr>';
    }
    ['#bpf-q', '#bpf-view', '#bpf-status', '#bpf-phase', '#bpf-dec'].forEach(sel => $(sel, el).addEventListener(sel === '#bpf-q' ? 'input' : 'change', table));
    $('#bp-table', el).addEventListener('click', e => { const b = e.target.closest('[data-open-note]'); if (b) { openNotes.add(b.dataset.openNote); table(); const i = $(`[data-note="${b.dataset.openNote}"]`, el); if (i) i.focus(); } });
    $('#bp-table', el).addEventListener('change', e => { const n = e.target.closest('[data-note]'); if (n) { Decisions.set(n.dataset.note, Decisions.get(n.dataset.note), n.value.slice(0, 500)); toast('Note saved.'); } });
    function md() {
      const groups = { must: 'Must', later: 'Later', cut: 'Cut', '': 'Undecided' };
      let out = '# DCLab R&D · product decisions\n\n';
      Object.entries(groups).forEach(([k, label]) => {
        const fs = FEATURES.filter(f => (Decisions.get(f.id) || '') === k);
        if (!fs.length) return;
        out += `## ${label} (${fs.length})\n\n` + fs.map(f => `- [${f.phase}] ${f.name} — ${VIEW_LABEL[f.view] || f.view}, ${STATUS_LABEL[f.status].toLowerCase()}${Decisions.note(f.id) ? ` — note: ${Decisions.note(f.id)}` : ''}`).join('\n') + '\n\n';
      });
      return out;
    }
    $('#bp-copy', el).addEventListener('click', () => copyText(md(), 'Decisions'));
    $('#bp-overlay', el).addEventListener('click', () => { DC.setBlueprint(true); toast('Blueprint on. Each page now shows what exists and what is missing.'); });
    function all() { statusPanel(); decisionsPanel(); pagesPanel(); phasesPanel(); table(); DC.hydrate(el); }
    document.addEventListener('decisionschange', () => { decisionsPanel(); phasesPanel(); if (!$('#bp-table', el).contains(document.activeElement)) table(); });
    all();
  },
});
