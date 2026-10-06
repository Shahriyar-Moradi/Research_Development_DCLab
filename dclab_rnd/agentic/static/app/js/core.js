/* DCLab — app core: shell, router, page tabs, server API, helpers, charts, blueprint layer, decisions, tours, palette. */
(function () {
  'use strict';
  const DEMO = window.DEMO || {};
  const FEATURES = window.FEATURES || [];
  const RECORDS = window.DEMO_RECORDS || [];
  const REC = Object.fromEntries(RECORDS.map(r => [r.id, r]));

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (v, d = 4) => (v == null || isNaN(v) ? '—' : Number(v).toFixed(d));
  const pct = (v, d = 1) => (v == null || isNaN(v) ? '—' : (v * 100).toFixed(d) + '%');
  const int = v => Number(v).toLocaleString('en-US');

  /* ---------------- icons ---------------- */
  const IC = {
    home: '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/><path d="M10 21v-6h4v6"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    graph: '<circle cx="5" cy="6" r="2.5"/><circle cx="19" cy="6" r="2.5"/><circle cx="12" cy="18" r="2.5"/><path d="M7.5 6h9M6.3 8.2l4.4 7.6M17.7 8.2l-4.4 7.6"/>',
    solution: '<path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8z"/><path d="M14 3v5h5M8 13h8M8 17h5"/>',
    notebook: '<rect x="4" y="3" width="16" height="18" rx="2"/><path d="m9 10-2 2 2 2M15 10l2 2-2 2"/>',
    shield: '<path d="M12 3 4 6v6c0 4.5 3.4 8.2 8 9 4.6-.8 8-4.5 8-9V6z"/><path d="m9 12 2 2 4-4"/>',
    models: '<path d="M4 20V10M10 20V4M16 20v-7M21 20H3"/>',
    pulse: '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    brief: '<rect x="3" y="7" width="18" height="13" rx="2"/><path d="M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2M3 13h18"/>',
    intern: '<path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3"/><rect x="7" y="7" width="10" height="10" rx="3"/><path d="M10 12h.01M14 12h.01"/>',
    cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    book: '<path d="M4 4h6a2 2 0 0 1 2 2v14a2 2 0 0 0-2-2H4zM20 4h-6a2 2 0 0 0-2 2v14a2 2 0 0 1 2-2h6z"/>',
    flask: '<path d="M9 3h6M10 3v6L4.5 18.5A1.7 1.7 0 0 0 6 21h12a1.7 1.7 0 0 0 1.5-2.5L14 9V3"/><path d="M7 15h10"/>',
    target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    brain: '<path d="M9.5 3.5a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 1.5 5 3 3 0 0 0 3.5 4 3 3 0 0 0 2.5-1V5a2.5 2.5 0 0 0-2.5-1.5z"/><path d="M14.5 3.5a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-1.5 5 3 3 0 0 1-3.5 4 3 3 0 0 1-2.5-1"/>',
    layers: '<path d="M12 3 3 7.5l9 4.5 9-4.5z"/><path d="m3 12 9 4.5 9-4.5M3 16.5 12 21l9-4.5"/>',
    plug: '<path d="M9 2v6M15 2v6M7 8h10v4a5 5 0 0 1-10 0zM12 17v5"/>',
    gear: '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M2 12h3M19 12h3M4.9 19.1 7 17M17 7l2.1-2.1"/>',
    map: '<path d="M3 6l6-3 6 3 6-3v15l-6 3-6-3-6 3z"/><path d="M9 3v15M15 6v15"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
    close: '<path d="M6 6l12 12M18 6 6 18"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
    bell: '<path d="M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.9 1.9 0 0 0 3.4 0"/>',
    play: '<path d="M7 4v16l13-8z"/>',
    check: '<path d="m5 12 5 5L20 7"/>',
    alert: '<path d="M12 3 2 20h20z"/><path d="M12 10v4M12 17h.01"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    lock: '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    unlock: '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 7.5-2"/>',
    copy: '<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/>',
    ext: '<path d="M14 4h6v6M20 4l-9 9M19 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h5"/>',
    arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
    back: '<path d="M19 12H5M11 6l-6 6 6 6"/>',
    spark: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
    upload: '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
    db: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    git: '<circle cx="6" cy="6" r="2.5"/><circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="9" r="2.5"/><path d="M6 8.5v7M18 11.5c0 4-6 3-10.5 5"/>',
    users: '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/><path d="M16 4.5a3.5 3.5 0 0 1 0 7M18 14a6 6 0 0 1 3.5 6"/>',
    eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    route: '<circle cx="6" cy="19" r="2.5"/><circle cx="18" cy="5" r="2.5"/><path d="M8.5 19H17a3.5 3.5 0 0 0 0-7H7a3.5 3.5 0 0 1 0-7h8.5"/>',
    image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/>',
    car: '<path d="M5 16h14M6.5 16l1.5-5h8l1.5 5M5 16v3M19 16v3"/><circle cx="8" cy="16" r="1.5"/><circle cx="16" cy="16" r="1.5"/>',
    text: '<path d="M4 6h16M4 10h16M4 14h10M4 18h7"/>',
    trend: '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    alertc: '<circle cx="12" cy="12" r="9"/><path d="M12 7v6M12 16.5h.01"/>',
    bug: '<rect x="7" y="7" width="10" height="13" rx="5"/><path d="M12 7V4M9 4l1 3M15 4l-1 3M3 12h4M17 12h4M4 18l3-2M20 18l-3-2M4 7l3 2M20 7l-3 2"/>',
    nodes: '<circle cx="12" cy="5" r="2.5"/><circle cx="5" cy="19" r="2.5"/><circle cx="19" cy="19" r="2.5"/><path d="M12 7.5v4M12 11.5l-5.5 5.5M12 11.5l5.5 5.5"/>',
    scale: '<path d="M12 3v18M5 21h14M6 7h12M6 7l-3 7a3 3 0 0 0 6 0zM18 7l-3 7a3 3 0 0 0 6 0z"/>',
    file: '<path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8z"/><path d="M14 3v5h5"/>',
    seal: '<circle cx="12" cy="10" r="6"/><path d="M9 15.5 8 22l4-2 4 2-1-6.5"/>',
    send: '<path d="M22 2 11 13M22 2l-7 20-4-9-9-4z"/>',
    refresh: '<path d="M21 12a9 9 0 1 1-2.6-6.4L21 8"/><path d="M21 3v5h-5"/>',
    filter: '<path d="M3 5h18l-7 8v6l-4 2v-8z"/>',
    stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    tour: '<path d="M4 20 10 4l4 10 2-4 4 10"/>',
  };
  const icon = (name, cls = '') => `<svg viewBox="0 0 24 24" class="${cls}" aria-hidden="true">${IC[name] || ''}</svg>`;

  /* ---------------- navigation model ---------------- */
  const PROJECT_VIEWS = ['project', 'solution', 'notebook', 'audit', 'models', 'reliability', 'brief'];
  const VIEW_META = {
    home: { label: 'Home', crumbs: ['Workspace'] },
    new: { label: 'New project', crumbs: ['Workspace'] },
    project: { label: 'Workflow', crumbs: ['Term-deposit calls'] },
    solution: { label: 'Solution & data', crumbs: ['Term-deposit calls'] },
    notebook: { label: 'Notebook', crumbs: ['Term-deposit calls'] },
    audit: { label: 'Leakage & features', crumbs: ['Term-deposit calls'] },
    models: { label: 'Models', crumbs: ['Term-deposit calls'] },
    reliability: { label: 'Reliability & holdout', crumbs: ['Term-deposit calls'] },
    brief: { label: 'Decision brief', crumbs: ['Term-deposit calls'] },
    intern: { label: 'Intern', crumbs: ['Workspace'] },
    compute: { label: 'Compute & jobs', crumbs: ['Workspace'] },
    evidence: { label: 'Evidence library', crumbs: ['Knowledge'] },
    lab: { label: 'Research lab', crumbs: ['Knowledge'] },
    benchmark: { label: 'Judgment benchmark', crumbs: ['Knowledge'] },
    policy: { label: 'Policy model', crumbs: ['Knowledge'] },
    packs: { label: 'Domain packs', crumbs: ['Platform'] },
    integrations: { label: 'Integrations', crumbs: ['Platform'] },
    admin: { label: 'Admin', crumbs: ['Platform'] },
    blueprint: { label: 'Blueprint & roadmap', crumbs: ['Platform'] },
  };
  const NAV = [
    { group: 'Workspace', items: [
      { id: 'home', label: 'Home', icon: 'home' },
      { id: 'new', label: 'New project', icon: 'plus' },
      { id: 'intern', label: 'Intern', icon: 'intern', count: '1' },
      { id: 'compute', label: 'Compute & jobs', icon: 'cpu', count: '2' },
    ] },
    { group: 'Project · Term deposits', items: [
      { id: 'project', label: 'Workflow', icon: 'graph' },
      { id: 'solution', label: 'Solution & data', icon: 'solution' },
      { id: 'notebook', label: 'Notebook', icon: 'notebook', count: '10' },
      { id: 'audit', label: 'Leakage & features', icon: 'shield' },
      { id: 'models', label: 'Models', icon: 'models' },
      { id: 'reliability', label: 'Reliability', icon: 'pulse' },
      { id: 'brief', label: 'Decision brief', icon: 'brief' },
    ] },
    { group: 'Knowledge', items: [
      { id: 'evidence', label: 'Evidence library', icon: 'book', count: String(RECORDS.length || 133) },
      { id: 'lab', label: 'Research lab', icon: 'flask' },
      { id: 'benchmark', label: 'Benchmark', icon: 'target' },
      { id: 'policy', label: 'Policy model', icon: 'brain' },
    ] },
    { group: 'Platform', items: [
      { id: 'packs', label: 'Domain packs', icon: 'layers' },
      { id: 'integrations', label: 'Integrations', icon: 'plug' },
      { id: 'admin', label: 'Admin', icon: 'gear' },
      { id: 'blueprint', label: 'Blueprint & roadmap', icon: 'map', count: '' },
    ] },
  ];

  function renderNav() {
    const nav = $('#nav');
    if (!nav) return;
    nav.innerHTML = NAV.map(g => `<div class="nav-group"><div class="nav-label">${esc(g.group)}</div>${g.items.map(it =>
      `<a class="nav-item" href="#${it.id}" data-nav="${it.id}">${icon(it.icon)}<span>${esc(it.label)}</span>${it.count != null ? `<span class="count">${esc(it.count)}</span>` : ''}</a>`).join('')}</div>`).join('');
  }

  /* The sidebar's project group follows the project that is open on the project pages. */
  /* A sidebar count ("" hides it): pages set their own once they know the real number. */
  function setNavCount(id, text) {
    const c = $(`#nav [data-nav="${id}"] .count`);
    if (c) c.textContent = text == null ? '' : String(text);
  }
  function setProjectLabel(name) {
    setNavCount('notebook', '');  // the sample's "10 findings" does not belong to a real project
    const label = $$('#nav .nav-label').find(l => l.textContent.startsWith('Project'));
    if (label) label.textContent = 'Project · ' + (name || 'none');
    PROJECT_VIEWS.forEach(v => { if (VIEW_META[v]) VIEW_META[v].crumbs = [name || 'Project']; });
    const meta = VIEW_META[state.view];
    if (meta && PROJECT_VIEWS.includes(state.view)) $('#crumbs').innerHTML = meta.crumbs.map(c => `<span>${esc(c)}</span><span class="sep">›</span>`).join('') + `<b>${esc(meta.label)}</b>`;
  }

  /* ---------------- toast / drawer / modal ---------------- */
  function toast(text, opts = {}) {
    const box = $('#toasts');
    if (!box) return;
    const el = document.createElement('div');
    el.className = 'toast';
    el.setAttribute('role', 'status');
    el.innerHTML = (opts.ok === false ? icon('alert') : icon('check', 't-ok')).replace('<svg', '<svg width="15" height="15" data-style="stroke:currentColor;fill:none;stroke-width:2"') + `<span>${esc(text)}</span>`;
    box.appendChild(el);
    setTimeout(() => el.remove(), opts.ms || 3200);
  }

  const drawer = {
    open({ eyebrow = '', title = '', html = '', actions = '' }) {
      $('#drawer-eyebrow').innerHTML = eyebrow;
      $('#drawer-title').textContent = title;
      $('#drawer-body').innerHTML = html;
      $('#drawer-actions').innerHTML = actions;
      $('#drawer').classList.add('open');
      $('#drawer').setAttribute('aria-hidden', 'false');
      $('#scrim').classList.add('open');
      setTimeout(() => $('#drawer-close').focus(), 50);
      hydrate($('#drawer'));
    },
    close() {
      $('#drawer').classList.remove('open');
      $('#drawer').setAttribute('aria-hidden', 'true');
      if (!$('#modal').classList.contains('open')) $('#scrim').classList.remove('open');
    },
  };

  let modalConfirm = null;
  const modal = {
    open({ eyebrow = '', title = '', html = '', confirm = 'Confirm', cancel = 'Cancel', danger = false, onConfirm = null, onOpen = null, confirmDisabled = false, hideConfirm = false }) {
      $('#modal-eyebrow').innerHTML = eyebrow;
      $('#modal-title').textContent = title;
      $('#modal-body').innerHTML = html;
      const ok = $('#modal-confirm');
      ok.textContent = confirm;
      ok.className = 'btn ' + (danger ? 'danger solid' : 'primary');
      ok.disabled = !!confirmDisabled;
      ok.hidden = !!hideConfirm;
      $('#modal-cancel').textContent = cancel;
      modalConfirm = onConfirm;
      $('#modal').classList.add('open');
      $('#modal').setAttribute('aria-hidden', 'false');
      $('#scrim').classList.add('open');
      hydrate($('#modal'));
      if (onOpen) onOpen($('#modal'));
      setTimeout(() => (ok.hidden || ok.disabled ? $('#modal-cancel') : ok).focus(), 60);
    },
    close() {
      $('#modal').classList.remove('open');
      $('#modal').setAttribute('aria-hidden', 'true');
      if (!$('#drawer').classList.contains('open')) $('#scrim').classList.remove('open');
      modalConfirm = null;
    },
  };

  /* ---------------- evidence records ---------------- */
  const TYPE_LABEL = { rule: 'Rule', workflow: 'Workflow block', dataset: 'Dataset card', experiment: 'Experiment', leakage_precedent: 'Leakage precedent', pitfall: 'Pitfall experiment', finding: 'Cross-dataset finding', workspace_lesson: 'Workspace lesson' };
  const TYPE_CLS = { rule: 'proof', workflow: 'accent', dataset: 'info', experiment: '', leakage_precedent: 'bad', pitfall: 'warn', finding: 'ok', workspace_lesson: 'info' };
  function chip(id, label) {
    const cls = id.startsWith('DCLAB-R') ? ' rule' : id.startsWith('WF-') ? ' wf' : '';
    return `<button type="button" class="ev-chip${cls}" data-record="${esc(id)}" title="${esc((REC[id] && REC[id].title) || id)}">${esc(label || id)}</button>`;
  }
  const chips = ids => `<span class="ev-list">${ids.map(i => chip(i)).join('')}</span>`;
  /* A model's explanation of a stage result (package A3.3): its sentences with what each cites, labelled as model text.
     The deterministic notes stay where they are; this never replaces them. */
  function explanation(record, title) {
    const x = record && record.explanation;
    if (!x || !(x.sentences || []).length) return '';
    const cite = id => REC[id] ? chip(id) : `<span class="tag" title="${esc(id)}">${esc(id.replace(/^PRJ-[^-]+-/, ''))}</span>`;
    return `<div class="explain"><span class="row"><span class="eyebrow">${esc(title || 'Explained')}</span><span class="pill outline">written by a model from these records</span></span>`
      + `<p>${x.sentences.map(s => `${esc(s.text)} ${(s.cites || []).map(cite).join('')}`).join(' ')}</p>`
      + `<span class="xs muted">${esc(x.label || '')}${x.dropped ? ` ${x.dropped} sentence${x.dropped === 1 ? ' was' : 's were'} removed.` : ''} The deterministic notes are unchanged.</span></div>`;
  }
  const ID_RE = /\b(DCLAB-R\d{2}|WF-\d{2}|EXP-\d{3}|PIT-\d{3}|LEAK-[a-z_]+|FINDING-[a-z-]+|DATASET-[a-z_]+)\b/g;
  function linkIds(text) {
    return esc(text).replace(ID_RE, m => (REC[m] ? chip(m) : m));
  }
  /* The evidence index, live: RECORDS starts as the snapshot bundled at build time and is replaced in place by
     GET /api/evidence, so every chip, drawer and search reads the index the server has now. */
  let recordsLoad = null;
  function loadRecords(force) {
    if (recordsLoad && !force) return recordsLoad;
    recordsLoad = api('/evidence').then(d => {
      if (!d || !Array.isArray(d.records) || !d.records.length) return null;
      RECORDS.splice(0, RECORDS.length, ...d.records);
      Object.keys(REC).forEach(k => { delete REC[k]; });
      RECORDS.forEach(r => { REC[r.id] = r; });
      setNavCount('evidence', String(RECORDS.length));
      return d;
    }).catch(() => { recordsLoad = null; return null; });
    return recordsLoad;
  }
  function openRecord(id) {
    const r = REC[id];
    if (!r) {
      drawer.open({ eyebrow: '<span class="pill">Not in the index</span>', title: id, html: `<p>The record <code>${esc(id)}</code> is not in the evidence index.</p>` });
      return;
    }
    const meta = r.meta || {};
    const metaRows = Object.entries(meta).filter(([k, v]) => v !== null && v !== '' && typeof v !== 'object').slice(0, 10)
      .map(([k, v]) => `<dt>${esc(k.replace(/_/g, ' '))}</dt><dd>${esc(typeof v === 'number' ? (Math.abs(v) < 10 ? Number(v).toFixed(4).replace(/0+$/, '').replace(/\.$/, '') : v) : v)}</dd>`).join('');
    const listMeta = Object.entries(meta).filter(([k, v]) => Array.isArray(v)).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v.map(x => `<span class="tag">${esc(x)}</span>`).join(' ')}</dd>`).join('');
    const usedBy = RECORDS.filter(x => x.id !== id && x.text && x.text.includes(id)).slice(0, 8).map(x => x.id);
    drawer.open({
      eyebrow: `<span class="pill ${TYPE_CLS[r.type] || ''}">${esc(TYPE_LABEL[r.type] || r.type)}</span> <span class="tag">${esc(r.id)}</span>`,
      title: r.title,
      html: `<div class="record-text">${linkIds(r.text)}</div>
        ${metaRows || listMeta ? `<div class="inset"><dl class="kv">${metaRows}${listMeta}</dl></div>` : ''}
        ${r.citations && r.citations.length ? `<div class="stack tight"><div class="eyebrow">Citations</div>${r.citations.map(c => `<code class="small" data-style="overflow-wrap:anywhere">${linkIds(c)}</code>`).join('')}</div>` : ''}
        ${usedBy.length ? `<div class="stack tight"><div class="eyebrow">Cited by</div>${chips(usedBy)}</div>` : ''}
        <div class="callout proof"><span class="ic">${icon('info')}</span><span>This is a real record from the repository's evidence index (<code>evidence/knowledge/rag/records.jsonl</code>). Every note, flag and decision in the product links to records like this one.</span></div>`,
      actions: `<button class="btn sm" data-toast="In the product: asks the intern to explain ${esc(r.id)} in plain words, with its scope and limits.">${icon('spark')}Explain simply</button>`,
    });
  }

  /* ---------------- charts (SVG strings, drawn to scale) ---------------- */
  function niceTicks(min, max, n = 5) {
    const span = max - min || 1;
    const step0 = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const t = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) t.push(+v.toFixed(10));
    return t;
  }
  const charts = {
    barsH(items, o = {}) {
      const W = o.width || 620, rowH = o.rowH || 30, labelW = o.labelW || 150, valW = o.valW || 118, top = 8, bottom = o.noAxis ? 6 : 26;
      const H = top + bottom + rowH * items.length;
      const min = o.min != null ? o.min : 0, max = o.max != null ? o.max : Math.max(...items.map(i => i.hi != null ? i.hi : i.value));
      const x0 = labelW, x1 = W - valW;
      const sx = v => x0 + ((v - min) / (max - min)) * (x1 - x0);
      const ticks = o.ticks || niceTicks(min, max, 5);
      const tf = o.tickFmt || (v => v.toFixed(2));
      let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" data-style="max-width:${Math.round(W * 1.12)}px" role="img" aria-label="${esc(o.aria || 'bar chart')}">`;
      ticks.forEach(t => { if (t < min || t > max) return; const x = sx(t); s += `<line class="grid-line" x1="${x}" x2="${x}" y1="${top}" y2="${H - bottom}"/>`; if (!o.noAxis) s += `<text class="tick" x="${x}" y="${H - 8}" text-anchor="middle">${tf(t)}</text>`; });
      if (o.ref != null) { const x = sx(o.ref); s += `<line class="ref" x1="${x}" x2="${x}" y1="${top - 4}" y2="${H - bottom}"/>`; if (o.refLabel) s += `<text class="val-muted" x="${x + 4}" y="${top + 6}">${esc(o.refLabel)}</text>`; }
      items.forEach((it, i) => {
        const y = top + i * rowH, bh = Math.min(16, rowH - 10), by = y + (rowH - bh) / 2;
        const xv = sx(Math.max(min, Math.min(max, it.value)));
        s += `<text class="${it.strong ? 'lbl-strong' : 'lbl'}" x="${labelW - 10}" y="${y + rowH / 2 + 4}" text-anchor="end">${esc(it.label)}</text>`;
        s += `<rect class="bar ${it.cls || ''}" x="${x0}" y="${by}" width="${Math.max(1, xv - x0)}" height="${bh}" rx="3"/>`;
        if (it.lo != null && it.hi != null) {
          const a = sx(Math.max(min, it.lo)), b = sx(Math.min(max, it.hi)), cy = by + bh / 2;
          s += `<line class="whisker" x1="${a}" x2="${b}" y1="${cy}" y2="${cy}"/><line class="whisker" x1="${a}" x2="${a}" y1="${cy - 5}" y2="${cy + 5}"/><line class="whisker" x1="${b}" x2="${b}" y1="${cy - 5}" y2="${cy + 5}"/>`;
        }
        s += `<text class="val" x="${x1 + 8}" y="${y + rowH / 2 + 4}">${esc(it.valueText != null ? it.valueText : it.value.toFixed(4))}</text>`;
      });
      return s + '</svg>';
    },
    line(series, o = {}) {
      const W = o.width || 560, H = o.height || 240, L = o.left || 46, R = o.right || 16, T = o.top || 12, B = o.bottom || 36;
      const xs = series.flatMap(s => s.pts.map(p => p[0])), ys = series.flatMap(s => s.pts.map(p => p[1]));
      const xMin = o.xMin != null ? o.xMin : Math.min(...xs), xMax = o.xMax != null ? o.xMax : Math.max(...xs);
      const yMin = o.yMin != null ? o.yMin : Math.min(...ys), yMax = o.yMax != null ? o.yMax : Math.max(...ys);
      const sx = v => L + ((v - xMin) / (xMax - xMin || 1)) * (W - L - R);
      const sy = v => H - B - ((v - yMin) / (yMax - yMin || 1)) * (H - T - B);
      const xt = o.xTicks || niceTicks(xMin, xMax, 5), yt = o.yTicks || niceTicks(yMin, yMax, 4);
      const xf = o.xFmt || (v => String(v)), yf = o.yFmt || (v => String(v));
      let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" data-style="max-width:${Math.round(W * 1.12)}px" role="img" aria-label="${esc(o.aria || 'line chart')}">`;
      yt.forEach(t => { if (t < yMin - 1e-9 || t > yMax + 1e-9) return; const y = sy(t); s += `<line class="grid-line" x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/><text class="tick" x="${L - 6}" y="${y + 3}" text-anchor="end">${yf(t)}</text>`; });
      xt.forEach(t => { if (t < xMin - 1e-9 || t > xMax + 1e-9) return; const x = sx(t); s += `<text class="tick" x="${x}" y="${H - B + 15}" text-anchor="middle">${xf(t)}</text>`; });
      s += `<line class="axis-line" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>`;
      if (o.bands) o.bands.forEach(b => { s += `<rect class="band" x="${sx(b[0])}" y="${T}" width="${Math.max(0, sx(b[1]) - sx(b[0]))}" height="${H - T - B}"/>`; });
      if (o.range) { const [lo, hi] = o.range; s += `<path class="area" d="${lo.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join('')}${hi.slice().reverse().map(p => `L${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join('')}Z"/>`; }
      if (o.diag) s += `<line class="ref" x1="${sx(Math.max(xMin, yMin))}" y1="${sy(Math.max(xMin, yMin))}" x2="${sx(Math.min(xMax, yMax))}" y2="${sy(Math.min(xMax, yMax))}"/>`;
      series.forEach(se => {
        const d = se.pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join('');
        if (se.area) s += `<path class="area ${se.cls || ''}" d="${d}L${sx(se.pts[se.pts.length - 1][0]).toFixed(1)},${sy(yMin)}L${sx(se.pts[0][0]).toFixed(1)},${sy(yMin)}Z"/>`;
        s += `<path class="line ${se.cls || ''}" d="${d}"/>`;
        if (se.dots) se.pts.forEach(p => { s += `<circle class="pt ${se.cls || ''}" cx="${sx(p[0])}" cy="${sy(p[1])}" r="3.2"/>`; });
        if (se.endLabel) { const p = se.pts[se.pts.length - 1]; s += `<circle class="pt ${se.cls || ''}" cx="${sx(p[0])}" cy="${sy(p[1])}" r="3.6"/><text class="val" x="${sx(p[0]) - 6}" y="${sy(p[1]) - 8}" text-anchor="end">${esc(se.endLabel)}</text>`; }
      });
      if (o.markers) o.markers.forEach(m => { const x = sx(m.x); s += `<line class="marker" x1="${x}" x2="${x}" y1="${T}" y2="${H - B}"/><text class="val" x="${x + 5}" y="${T + 10}">${esc(m.label)}</text>`; });
      if (o.points) o.points.forEach(p => { s += `<circle class="pt ${p.cls || ''}" cx="${sx(p.x)}" cy="${sy(p.y)}" r="${p.r || 4}"/>${p.label ? `<text class="val" x="${sx(p.x) + 7}" y="${sy(p.y) - 6}">${esc(p.label)}</text>` : ''}`; });
      if (o.xLabel) s += `<text class="val-muted" x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${esc(o.xLabel)}</text>`;
      if (o.yLabel) s += `<text class="val-muted" x="12" y="${(T + H - B) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(T + H - B) / 2})">${esc(o.yLabel)}</text>`;
      return s + '</svg>';
    },
    folds(rows, o = {}) {
      const W = o.width || 620, rowH = 30, labelW = o.labelW || 170, R = 70, T = 8, B = 26, H = T + B + rowH * rows.length;
      const all = rows.flatMap(r => r.folds.concat([r.mean]));
      const min = o.min != null ? o.min : Math.min(...all) - 0.01, max = o.max != null ? o.max : Math.max(...all) + 0.01;
      const sx = v => labelW + ((v - min) / (max - min)) * (W - labelW - R);
      let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" data-style="max-width:${Math.round(W * 1.12)}px" role="img" aria-label="${esc(o.aria || 'fold scores')}">`;
      niceTicks(min, max, 5).forEach(t => { if (t < min || t > max) return; const x = sx(t); s += `<line class="grid-line" x1="${x}" x2="${x}" y1="${T}" y2="${H - B}"/><text class="tick" x="${x}" y="${H - 8}" text-anchor="middle">${t.toFixed(2)}</text>`; });
      rows.forEach((r, i) => {
        const y = T + i * rowH + rowH / 2;
        s += `<text class="${r.strong ? 'lbl-strong' : 'lbl'}" x="${labelW - 10}" y="${y + 4}" text-anchor="end">${esc(r.label)}</text>`;
        s += `<line class="grid-line" x1="${sx(Math.min(...r.folds))}" x2="${sx(Math.max(...r.folds))}" y1="${y}" y2="${y}" data-style="stroke:var(--line-2);stroke-width:2"/>`;
        r.folds.forEach(f => { s += `<circle class="pt muted" cx="${sx(f)}" cy="${y}" r="3.4"/>`; });
        s += `<rect class="bar ${r.strong ? '' : 'proof'}" x="${sx(r.mean) - 1.5}" y="${y - 8}" width="3" height="16" rx="1"/>`;
        s += `<text class="val" x="${W - R + 8}" y="${y + 4}">${r.mean.toFixed(4)}</text>`;
      });
      return s + '</svg>';
    },
    vbars(items, o = {}) {
      const W = o.width || 360, H = o.height || 150, L = o.left || 30, R = 8, T = 10, B = 24;
      const max = o.max || Math.max(...items.map(i => i.value)) * 1.08;
      const bw = (W - L - R) / items.length;
      let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" data-style="max-width:${Math.round(W * 1.12)}px" role="img" aria-label="${esc(o.aria || 'histogram')}">`;
      niceTicks(0, max, 3).forEach(t => { const y = H - B - (t / max) * (H - T - B); if (t > max) return; s += `<line class="grid-line" x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/><text class="tick" x="${L - 5}" y="${y + 3}" text-anchor="end">${o.yFmt ? o.yFmt(t) : t}</text>`; });
      items.forEach((it, i) => {
        const h = (it.value / max) * (H - T - B), x = L + i * bw + 1.5;
        s += `<rect class="bar ${it.cls || ''}" x="${x}" y="${H - B - h}" width="${Math.max(1, bw - 3)}" height="${h}" rx="2"/>`;
        if (it.label != null && (items.length <= 14 || i % Math.ceil(items.length / 10) === 0)) s += `<text class="tick" x="${x + bw / 2 - 1.5}" y="${H - 8}" text-anchor="middle">${esc(it.label)}</text>`;
      });
      return s + '</svg>';
    },
    spark(values, o = {}) {
      const W = o.width || 96, H = o.height || 26, min = Math.min(...values), max = Math.max(...values);
      const sx = i => 2 + (i / (values.length - 1)) * (W - 4), sy = v => H - 3 - ((v - min) / (max - min || 1)) * (H - 6);
      const d = values.map((v, i) => `${i ? 'L' : 'M'}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join('');
      return `<svg class="spark" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" aria-hidden="true"><path class="area" d="${d}L${sx(values.length - 1)},${H}L${sx(0)},${H}Z"/><path class="line" d="${d}"/><circle class="end" cx="${sx(values.length - 1)}" cy="${sy(values[values.length - 1])}" r="2.4"/></svg>`;
    },
  };

  /* ---------------- workflow graphs (the project graph and the solution workflow on Home) ----------------
     A workflow is {nodes:[{id, wf, label, detail, state}], revisits:[{from, to, why}], gates:[{after, label}]}.
     Nodes run in rows of five, left to right then right to left, like the project graph. */
  const graph = {
    W: 150, H: 64,
    render(wf, opts = {}) {
      const W = this.W, H = this.H, per = opts.perRow || 5, GX = 190, GY = 160, X0 = 20, Y0 = 64;
      const nodes = (wf && wf.nodes) || [];
      const rows = Math.max(1, Math.ceil(nodes.length / per));
      const width = X0 * 2 + (per - 1) * GX + W, height = Y0 + (rows - 1) * GY + H + 64;
      const pos = {};
      nodes.forEach((n, i) => {
        const r = Math.floor(i / per), c = r % 2 ? per - 1 - (i % per) : i % per;
        const x = X0 + c * GX, y = Y0 + r * GY;
        pos[n.id] = { x, y, cx: x + W / 2, cy: y + H / 2, i };
      });
      const mk = (id, cls) => `<marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="${cls}" d="M0,0 L10,5 L0,10 z"/></marker>`;
      let s = `<svg class="graph-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(opts.aria || 'Solution workflow')}"><defs>${mk('gah', 'ah')}${mk('gah-ok', 'ah-ok')}${mk('gah-allowed', 'ah-allowed')}</defs>`;
      const fwd = (a, b) => {
        const p = pos[a], q = pos[b];
        if (p.y === q.y && q.x > p.x) return `M${p.x + W},${p.cy} L${q.x - 2},${q.cy}`;
        if (p.y === q.y) return `M${p.x},${p.cy} L${q.x + W + 2},${q.cy}`;
        return `M${p.cx},${p.y + H} L${q.cx},${q.y - 2}`;
      };
      nodes.slice(1).forEach((b, i) => {
        const a = nodes[i];
        const taken = a.state === 'done' && b.state && b.state !== 'todo';
        s += `<path class="edge ${taken ? 'taken' : ''}" d="${fwd(a.id, b.id)}" marker-end="url(#${taken ? 'gah-ok' : 'gah'})"/>`;
      });
      ((wf && wf.revisits) || []).forEach((r, k) => {
        const a = pos[r.from], b = pos[r.to];
        if (!a || !b) return;
        const lift = 44 + 10 * k;
        const d = a.y === b.y ? `M${a.cx},${a.y} C${a.cx},${a.y - lift} ${b.cx},${b.y - lift} ${b.cx},${b.y - 2}`
          : `M${a.cx},${a.y} C${a.cx},${a.y - lift} ${b.cx},${b.y + H + lift} ${b.cx},${b.y + H + 2}`;
        const lx = (a.cx + b.cx) / 2, ly = a.y === b.y ? a.y - lift * 0.72 : (a.y + b.y + H) / 2;
        s += `<path class="edge allowed" d="${d}" marker-end="url(#gah-allowed)"/><text class="lane-label" x="${lx}" y="${ly}" text-anchor="middle">${esc(('revisit · ' + r.why).toUpperCase())}</text>`;
      });
      ((wf && wf.gates) || []).forEach(g => {
        const a = pos[g.after], next = nodes[(a ? a.i : -2) + 1];
        if (!a || !next) return;
        const b = pos[next.id];
        let x, y;
        if (a.y === b.y) { x = (a.x < b.x ? a.x + W + b.x : b.x + W + a.x) / 2; y = a.cy; } else { x = a.cx; y = (a.y + H + b.y) / 2; }
        const firstRow = a.y === Y0;
        const ly = a.y === b.y ? (firstRow ? a.y - 10 : a.y + H + 16) : y + 4;
        const lx = a.y === b.y ? x : x + 16;
        s += `<path class="gate" d="M${x},${y - 9} L${x + 9},${y} L${x},${y + 9} L${x - 9},${y} Z"/><text class="gate-text" x="${lx}" y="${ly}" text-anchor="${a.y === b.y ? 'middle' : 'start'}">${esc(g.label)}</text>`;
      });
      const stateText = { done: 'done', current: 'in progress', waiting: 'waiting for data', blocked: 'blocked', todo: '' };
      const stateCls = { done: 'done', current: 'current', waiting: 'blocked', blocked: 'blocked', todo: 'todo' };
      nodes.forEach(n => {
        const p = pos[n.id], label = n.label.length > 21 ? n.label.slice(0, 20) + '…' : n.label;
        s += `<g class="node ${stateCls[n.state] || 'todo'}" data-node="${esc(n.id)}" tabindex="0" role="button" aria-label="${esc(n.wf + ' ' + n.label + (n.state ? ': ' + (stateText[n.state] || n.state) : ''))}">
          <title>${esc(n.label + (n.detail ? ' — ' + n.detail : ''))}</title>
          <rect x="${p.x}" y="${p.y}" width="${W}" height="${H}" rx="9"${n.id === opts.selected ? ' data-style="stroke-width:2.6"' : ''}/>
          <text class="id" x="${p.x + 10}" y="${p.y + 17}">${esc(n.wf)}</text>
          <text x="${p.x + 10}" y="${p.y + 36}">${esc(label)}</text>
          <text class="st" x="${p.x + 10}" y="${p.y + 53}">${esc(stateText[n.state] || '')}</text>
        </g>`;
      });
      return s + '</svg>';
    },
  };

  /* ---------------- maths used by the cost tools ---------------- */
  function erf(x) { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; }
  const Phi = z => 0.5 * (1 + erf(z / Math.SQRT2));
  function PhiInv(p) { let lo = -8, hi = 8; for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (Phi(m) < p) lo = m; else hi = m; } return (lo + hi) / 2; }
  /* Binormal ranking model fitted to a reported ROC-AUC: positives ~ N(mu,1), negatives ~ N(0,1). */
  function binormal(auc, prevalence) {
    const mu = Math.SQRT2 * PhiInv(auc);
    const atFraction = q => { let lo = -8, hi = 10; for (let i = 0; i < 70; i++) { const t = (lo + hi) / 2; const share = prevalence * (1 - Phi(t - mu)) + (1 - prevalence) * (1 - Phi(t)); if (share > q) lo = t; else hi = t; } const t = (lo + hi) / 2; const tpr = 1 - Phi(t - mu), fpr = 1 - Phi(t); return { t, tpr, fpr, precision: prevalence * tpr / Math.max(1e-9, prevalence * tpr + (1 - prevalence) * fpr) }; };
    return { mu, atFraction };
  }

  /* ---------------- code highlighting ---------------- */
  const KW = 'from|import|def|return|for|in|if|elif|else|None|True|False|and|or|not|with|as|class|lambda|while|try|except|raise|assert|yield|pass|break|continue|is';
  const TOK = new RegExp(`(#[^\\n]*)|(f?"(?:[^"\\\\\\n]|\\\\.)*"|f?'(?:[^'\\\\\\n]|\\\\.)*')|\\b(\\d+(?:\\.\\d+)?(?:e-?\\d+)?)\\b|\\b(${KW})\\b|\\b([A-Za-z_][A-Za-z0-9_]*)(?=\\()`, 'g');
  function highlightPy(src) {
    let out = '', last = 0, m;
    TOK.lastIndex = 0;
    while ((m = TOK.exec(src))) {
      out += esc(src.slice(last, m.index));
      const [t] = m;
      out += m[1] ? `<span class="c">${esc(t)}</span>` : m[2] ? `<span class="s">${esc(t)}</span>` : m[3] ? `<span class="n">${esc(t)}</span>` : m[4] ? `<span class="k">${esc(t)}</span>` : `<span class="f">${esc(t)}</span>`;
      last = m.index + t.length;
    }
    return out + esc(src.slice(last));
  }
  function codeBlock(src, opts = {}) {
    const hl = new Set(opts.hl || []), ok = new Set(opts.ok || []);
    const lines = highlightPy(src).split('\n');
    return `<pre class="code">${lines.map((l, i) => `<span class="${hl.has(i + 1) ? 'hl' : ok.has(i + 1) ? 'hl-ok' : ''}">${l || ' '}</span>`).join('\n')}</pre>`;
  }

  /* ---------------- decisions (Must / Later / Cut on the blueprint, kept in this browser) ---------------- */
  const LS_KEY = 'dclab-decisions-v1';
  const Decisions = {
    map: {}, backend: 'memory', readOnly: false, db: null, listeners: new Set(), chain: {},
    init() {
      try { const raw = localStorage.getItem(LS_KEY); if (raw) this.map = JSON.parse(raw) || {}; this.backend = 'local'; } catch (e) { this.backend = 'memory'; }
      this.emit();
    },
    get(id) { return (this.map[id] && this.map[id].decision) || ''; },
    note(id) { return (this.map[id] && this.map[id].note) || ''; },
    set(id, decision, note) {
      if (this.readOnly) { toast('You can see these decisions but cannot change them.', { ok: false }); return; }
      const cur = this.map[id] || {};
      const next = { decision: decision || '', note: note != null ? String(note) : (cur.note || ''), updated: new Date().toISOString() };
      this.map = Object.assign({}, this.map, { [id]: next });
      try { localStorage.setItem(LS_KEY, JSON.stringify(this.map)); } catch (e) { /* storage unavailable */ }
      this.emit();
      if (this.db) {
        const prev = this.chain[id] || Promise.resolve();
        this.chain[id] = prev.then(() => this.db.doc('decisions/' + id).set(next)).catch(err => {
          if (err && err.code === 'invalid_argument') { this.readOnly = true; toast('Your access to this page is read-only, so the decision stayed in this browser.', { ok: false }); }
          else toast('Could not save to the shared plan. The decision is kept in this browser.', { ok: false });
        });
      }
    },
    counts() { const c = { must: 0, later: 0, cut: 0 }; Object.values(this.map).forEach(v => { if (c[v.decision] != null) c[v.decision]++; }); return c; },
    on(fn) { this.listeners.add(fn); },
    emit() { this.listeners.forEach(fn => { try { fn(this); } catch (e) { console.error(e); } }); },
  };
  const decideButtons = id => {
    const d = Decisions.get(id);
    return `<span class="decide" data-decide="${esc(id)}"><button type="button" data-d="must" aria-pressed="${d === 'must'}">Must</button><button type="button" data-d="later" aria-pressed="${d === 'later'}">Later</button><button type="button" data-d="cut" aria-pressed="${d === 'cut'}">Cut</button></span>`;
  };

  /* ---------------- blueprint layer ---------------- */
  const FMAP = Object.fromEntries(FEATURES.map(f => [f.id, f]));
  const STATUS_LABEL = { built: 'Built', partial: 'Partial', todo: 'To build', research: 'Research' };
  function applyBlueprintAttrs(root = document) {
    $$('[data-f]', root).forEach(el => {
      const f = FMAP[el.dataset.f];
      if (!f) { el.dataset.bp = 'todo'; el.dataset.bpLabel = '? ' + el.dataset.f; return; }
      el.dataset.bp = f.status;
      el.dataset.bpLabel = `${STATUS_LABEL[f.status].toUpperCase()}${f.phase && f.status !== 'built' ? ' · ' + f.phase : ''}`;
      if (!el.title) el.title = `${f.name} — ${STATUS_LABEL[f.status]}${f.where ? ' · ' + f.where : ''}`;
    });
  }
  function renderBpPanel() {
    const panel = $('#bp-panel');
    if (!panel || panel.hidden) return;
    const view = state.view;
    const list = FEATURES.filter(f => f.view === view || (view === 'home' && f.view === 'shell'));
    const counts = { built: 0, partial: 0, todo: 0, research: 0 };
    list.forEach(f => counts[f.status]++);
    $('#bp-summary').innerHTML = `<b>${list.length}</b> features on this page · ${Object.entries(counts).filter(([, n]) => n).map(([k, n]) => `<span class="bp-status" data-bp="${k}">${n} ${STATUS_LABEL[k]}</span>`).join(' ')}`;
    $('#bp-list').innerHTML = list.length ? list.map(f => `<div class="bp-item" data-bp="${f.status}">
        <div class="spread"><span class="bp-name">${esc(f.name)}</span><span class="bp-status">${STATUS_LABEL[f.status]}${f.phase && f.status !== 'built' ? ' · ' + esc(f.phase) : ''}</span></div>
        ${f.where ? `<div class="bp-where">${esc(f.where)}</div>` : ''}
        ${f.note ? `<div class="small muted">${esc(f.note)}</div>` : ''}
        <div class="spread"><button type="button" class="link-btn small" data-bp-find="${esc(f.id)}">Show on page</button>${decideButtons(f.id)}</div>
      </div>`).join('') : '<div class="empty">No product features are tagged on this page.</div>';
    const be = Decisions.readOnly ? 'You can read the decisions but not change them.' : Decisions.backend === 'shared' ? 'Decisions save to the shared plan, so Claude can read them.' : 'Decisions save in this browser only.';
    $('#bp-backend').textContent = be;
  }
  function setBlueprint(on) {
    state.bp = !!on;
    document.body.classList.toggle('bp', state.bp);
    const sw = $('#bp-switch');
    if (sw) sw.checked = state.bp;
    $('#bp-panel').hidden = !state.bp;
    if (state.bp) { applyBlueprintAttrs(); renderBpPanel(); }
    try { localStorage.setItem('dclab-demo-bp', state.bp ? '1' : '0'); } catch (e) { /* ignore */ }
  }

  /* ---------------- page tabs ----------------
     A view lists its sections as <div class="pane" data-pane="id" data-label="Label" [data-count="n"]>.
     The first pane is the Overview; the tab bar is built here and the choice lives in the URL (#view/pane). */
  function buildPaneTabs(el) {
    const panes = $$(':scope > .pane', el);
    if (!panes.length || $(':scope > .ptabs', el)) return;
    const bar = document.createElement('div');
    bar.className = 'ptabs';
    bar.setAttribute('role', 'tablist');
    bar.setAttribute('aria-label', 'Sections of this page');
    bar.innerHTML = panes.map(p => {
      const roles = ['dev-only', 'biz-only', 'research-only', 'admin-only'].filter(c => p.classList.contains(c)).join(' ');
      return `<button type="button" class="ptab ${roles}" role="tab" data-ptab="${esc(p.dataset.pane)}" aria-selected="false">${esc(p.dataset.label || p.dataset.pane)}${p.dataset.count ? `<span class="n">${esc(p.dataset.count)}</span>` : ''}</button>`;
    }).join('');
    panes[0].before(bar);
    panes.forEach(p => { p.setAttribute('role', 'tabpanel'); p.hidden = true; });
  }
  function selectPane(el, id, opts = {}) {
    const panes = $$(':scope > .pane', el);
    if (!panes.length) return;
    const usable = panes.filter(p => !roleHidden(p));
    let target = panes.find(p => p.dataset.pane === id && !roleHidden(p)) || usable[0] || panes[0];
    panes.forEach(p => { p.hidden = p !== target; });
    $$(':scope > .ptabs .ptab', el).forEach(b => b.setAttribute('aria-selected', String(b.dataset.ptab === target.dataset.pane)));
    state.panes[el.id] = target.dataset.pane;
    hydrate(target);
    if (opts.url !== false) {
      const name = el.id.replace(/^view-/, '');
      const want = '#' + name + (target === usable[0] ? '' : '/' + target.dataset.pane);
      if (location.hash !== want) history.replaceState(null, '', want);
    }
    if (state.bp) renderBpPanel();
    target.dispatchEvent(new CustomEvent('paneshow', { bubbles: true, detail: target.dataset.pane }));
  }
  function roleHidden(p) {
    const r = state.role;
    return (p.classList.contains('dev-only') && r === 'business') || (p.classList.contains('biz-only') && r !== 'business') ||
      (p.classList.contains('research-only') && r !== 'researcher') || (p.classList.contains('admin-only') && r !== 'admin');
  }
  /* Open whichever pane holds an element, so tours and "Show on page" can reach it. */
  function reveal(target) {
    if (!target) return;
    const pane = target.closest('.pane');
    if (pane && pane.hidden) { const v = pane.closest('.view'); if (v) selectPane(v, pane.dataset.pane); }
    let d = target.closest('details:not([open])');
    while (d) { d.open = true; d = d.parentElement && d.parentElement.closest('details:not([open])'); }
  }

  /* ---------------- router ---------------- */
  const VIEWS = {};
  const state = { view: 'home', role: 'developer', bp: false, tour: null, step: 0, panes: {} };
  function view(name, def) { VIEWS[name] = Object.assign({ inited: false }, def); }
  function parseHash() { const [v, p] = (location.hash || '').replace(/^#/, '').split('/'); return { view: VIEW_META[v] ? v : 'home', pane: p || '' }; }
  function currentHash() { return parseHash().view; }
  function show(name, paneId) {
    const changed = state.view !== name;
    state.view = name;
    $$('.view').forEach(v => { v.hidden = v.id !== 'view-' + name; });
    const el = $('#view-' + name);
    const def = VIEWS[name];
    if (el) buildPaneTabs(el);
    if (def && el) {
      if (!def.inited) { def.inited = true; try { def.init && def.init(el); } catch (e) { console.error('init ' + name, e); } buildPaneTabs(el); hydrate(el); }
      try { def.enter && def.enter(el); } catch (e) { console.error('enter ' + name, e); }
    }
    if (el) selectPane(el, paneId || '', { url: !!paneId });
    // nav highlight
    $$('.nav-item').forEach(a => a.classList.remove('active'));
    const main = $(`[data-nav="${name}"]`); if (main) main.classList.add('active');
    const meta = VIEW_META[name] || { label: name, crumbs: [] };
    $('#crumbs').innerHTML = meta.crumbs.map(c => `<span>${esc(c)}</span><span class="sep">›</span>`).join('') + `<b>${esc(meta.label)}</b>`;
    document.body.classList.remove('nav-open');
    if (state.bp) { applyBlueprintAttrs(el || document); renderBpPanel(); }
    if (!state.tour && changed) window.scrollTo({ top: 0 });
    updateTourBar();
  }

  /* ---------------- roles ---------------- */
  const ROLE_NOTE = {
    developer: '',
    business: 'Business view: code, folds and tool calls are hidden. Numbers come with plain-language meaning.',
    researcher: 'Researcher view: evidence IDs, critic reviews and protocol details are expanded.',
    admin: 'Admin view: policies, budgets and approvals are editable.',
  };
  function setRole(role) {
    state.role = role;
    document.body.dataset.role = role;
    const rs = $('#role-select'); if (rs) rs.value = role;
    const cur = $('#view-' + state.view);
    if (cur && $(':scope > .ptabs', cur)) selectPane(cur, state.panes[cur.id] || '', { url: false });
    const n = $('#role-note');
    if (n) { n.hidden = !ROLE_NOTE[role]; n.textContent = ROLE_NOTE[role]; }
    document.dispatchEvent(new CustomEvent('rolechange', { detail: role }));
  }

  /* ---------------- tours ---------------- */
  const TOURS = DEMO.tours || {};
  function startTour(key) {
    if (!TOURS[key]) return;
    state.tour = key; state.step = 0;
    if (TOURS[key].role) setRole(TOURS[key].role);
    gotoStep();
  }
  function gotoStep() {
    const t = TOURS[state.tour];
    if (!t) return;
    const st = t.steps[state.step];
    $$('.tour-focus').forEach(e => e.classList.remove('tour-focus'));
    if (parseHash().view !== st.view) location.hash = st.view; else show(st.view);
    setTimeout(() => {
      const target = st.target ? $(`[data-tour="${st.target}"]`) : null;
      reveal(target);
      if (target) { target.classList.add('tour-focus'); target.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
      else window.scrollTo({ top: 0, behavior: 'smooth' });
    }, 120);
    updateTourBar();
  }
  function updateTourBar() {
    const bar = $('#tour-bar');
    if (!bar) return;
    if (!state.tour) { bar.hidden = true; $$('.tour-focus').forEach(e => e.classList.remove('tour-focus')); return; }
    const t = TOURS[state.tour], st = t.steps[state.step];
    bar.hidden = false;
    $('#tb-step').textContent = `${t.name} · step ${state.step + 1} of ${t.steps.length}`;
    $('#tb-title').textContent = st.title;
    $('#tb-text').textContent = st.text;
    $('#tb-dots').innerHTML = t.steps.map((_, i) => `<i class="${i === state.step ? 'on' : ''}"></i>`).join('');
    $('#tb-back').disabled = state.step === 0;
    $('#tb-next').textContent = state.step === t.steps.length - 1 ? 'Finish tour' : 'Next';
  }
  function endTour() { state.tour = null; updateTourBar(); }

  /* ---------------- command palette ---------------- */
  let palItems = [], palIndex = 0;
  function paletteSource(q) {
    const items = [];
    Object.entries(VIEW_META).forEach(([id, m]) => items.push({ group: 'Go to', label: m.label, hint: (m.crumbs || []).join(' / '), run: () => { location.hash = id; } }));
    $$('.view > .pane').forEach((p, i, all) => {
      const v = p.closest('.view').id.replace(/^view-/, ''), m = VIEW_META[v];
      if (!m || p === all.find(x => x.closest('.view') === p.closest('.view'))) return;
      items.push({ group: 'Sections', label: `${m.label} › ${p.dataset.label || p.dataset.pane}`, hint: (m.crumbs || []).join(' / '), run: () => { location.hash = v + '/' + p.dataset.pane; } });
    });
    items.push({ group: 'Actions', label: 'Start a new project', hint: 'N', run: () => { location.hash = 'new'; } });
    items.push({ group: 'Actions', label: state.bp ? 'Hide the blueprint layer' : 'Show the blueprint layer', hint: 'B', run: () => setBlueprint(!state.bp) });
    items.push({ group: 'Actions', label: 'View as business', hint: 'role', run: () => setRole('business') });
    items.push({ group: 'Actions', label: 'View as developer', hint: 'role', run: () => setRole('developer') });
    Object.entries(TOURS).forEach(([k, t]) => items.push({ group: 'Tours', label: 'Start the ' + t.name.toLowerCase(), hint: t.steps.length + ' steps', run: () => startTour(k) }));
    RECORDS.forEach(r => items.push({ group: 'Evidence', label: `${r.id} · ${r.title}`, hint: TYPE_LABEL[r.type] || r.type, run: () => openRecord(r.id) }));
    const ql = q.trim().toLowerCase();
    if (!ql) return items.filter(i => i.group !== 'Evidence' && i.group !== 'Sections').concat(items.filter(i => i.group === 'Evidence').slice(0, 6));
    const words = ql.split(/\s+/);
    return items.filter(i => words.every(w => (i.label + ' ' + i.hint + ' ' + i.group).toLowerCase().includes(w))).slice(0, 40);
  }
  function renderPalette() {
    const q = $('#pal-input').value;
    palItems = paletteSource(q);
    palIndex = Math.min(palIndex, Math.max(0, palItems.length - 1));
    let html = '', g = null;
    palItems.forEach((it, i) => {
      if (it.group !== g) { g = it.group; html += `<div class="palette-group">${esc(g)}</div>`; }
      html += `<button type="button" class="palette-item ${i === palIndex ? 'active' : ''}" data-pal="${i}"><span>${esc(it.label)}</span><span class="pk">${esc(it.hint || '')}</span></button>`;
    });
    $('#pal-list').innerHTML = html || '<div class="empty">Nothing matches. Try a record ID such as EXP-007 or a page name.</div>';
  }
  function openPalette() { $('#palette').hidden = false; $('#pal-input').value = ''; palIndex = 0; renderPalette(); setTimeout(() => $('#pal-input').focus(), 20); }
  function closePalette() { $('#palette').hidden = true; }
  function runPal(i) { const it = palItems[i]; closePalette(); if (it) it.run(); }

  /* ---------------- delegated behaviours ---------------- */
  function hydrate(root) {
    $$('pre[data-py]', root).forEach(pre => {
      if (pre.dataset.done) return;
      const hl = (pre.dataset.hl || '').split(',').filter(Boolean).map(Number);
      const okl = (pre.dataset.ok || '').split(',').filter(Boolean).map(Number);
      const tmp = document.createElement('div');
      tmp.innerHTML = codeBlock(pre.textContent.replace(/^\n/, '').replace(/\s+$/, ''), { hl, ok: okl });
      const fresh = tmp.firstChild;
      fresh.dataset.done = '1';
      pre.replaceWith(fresh);
    });
    $$('[data-chart]', root).forEach(el => {
      if (el.dataset.drawn) return;
      const fn = window.DEMO_CHARTS && window.DEMO_CHARTS[el.dataset.chart];
      if (fn) { el.innerHTML = fn(charts); el.dataset.drawn = '1'; }
    });
    if (state.bp) applyBlueprintAttrs(root);
  }
  function selectTab(btn) {
    const list = btn.closest('[data-tabs]');
    if (!list) return;
    const group = list.dataset.tabs;
    $$('[data-tab]', list).forEach(b => b.setAttribute('aria-selected', String(b === btn)));
    $$(`[data-tab-panel^="${group}:"]`).forEach(p => { p.hidden = p.dataset.tabPanel !== group + ':' + btn.dataset.tab; if (!p.hidden) hydrate(p); });
    list.dispatchEvent(new CustomEvent('tabchange', { detail: btn.dataset.tab, bubbles: true }));
  }
  async function copyText(text, label) {
    try { await navigator.clipboard.writeText(text); toast((label || 'Text') + ' copied'); }
    catch (e) {
      const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy'); toast((label || 'Text') + ' copied'); } catch (e2) { toast('Copy is blocked here. Select the text and copy it by hand.', { ok: false }); }
      ta.remove();
    }
  }
  document.addEventListener('click', e => {
    const t = e.target;
    const rec = t.closest('[data-record]');
    if (rec) { e.preventDefault(); openRecord(rec.dataset.record); return; }
    const ptab = t.closest('.ptab[data-ptab]');
    if (ptab) { selectPane(ptab.closest('.view'), ptab.dataset.ptab); return; }
    const tab = t.closest('[data-tab]');
    if (tab && tab.closest('[data-tabs]')) { selectTab(tab); return; }
    const segBtn = t.closest('.seg[data-seg] button');
    if (segBtn) {
      const seg = segBtn.closest('.seg');
      $$('button', seg).forEach(b => b.setAttribute('aria-pressed', String(b === segBtn)));
      seg.dispatchEvent(new CustomEvent('segchange', { detail: segBtn.dataset.v, bubbles: true }));
      return;
    }
    const tog = t.closest('[data-toggle]');
    if (tog) { tog.setAttribute('aria-pressed', String(tog.getAttribute('aria-pressed') !== 'true')); tog.dispatchEvent(new CustomEvent('togglechange', { bubbles: true })); }
    const go = t.closest('[data-go]');
    if (go) { e.preventDefault(); location.hash = go.dataset.go; return; }
    const pg = t.closest('[data-pane-go]');
    if (pg) { e.preventDefault(); selectPane(pg.closest('.view'), pg.dataset.paneGo); window.scrollTo({ top: 0, behavior: 'smooth' }); return; }
    const ts = t.closest('[data-toast]');
    if (ts) { toast(ts.dataset.toast); }
    const cp = t.closest('[data-copy]');
    if (cp) { const src = cp.dataset.copy.startsWith('#') ? ($(cp.dataset.copy) || {}).textContent : cp.dataset.copy; copyText(src || '', cp.dataset.copyLabel); }
    const dec = t.closest('[data-decide] button');
    if (dec) { const id = dec.closest('[data-decide]').dataset.decide; const d = dec.dataset.d; Decisions.set(id, Decisions.get(id) === d ? '' : d); return; }
    const find = t.closest('[data-bp-find]');
    if (find) { const el = $(`#view-${state.view} [data-f="${find.dataset.bpFind}"]`) || $(`[data-f="${find.dataset.bpFind}"]`); reveal(el); if (el && el.offsetParent !== null) { el.scrollIntoView({ behavior: 'smooth', block: 'center' }); el.classList.add('tour-focus'); setTimeout(() => el.classList.remove('tour-focus'), 1600); } else toast('That feature sits in a tab or panel that is closed right now.'); return; }
    const pal = t.closest('[data-pal]');
    if (pal) { runPal(Number(pal.dataset.pal)); return; }
    const tourStart = t.closest('[data-tour-start]');
    if (tourStart) { startTour(tourStart.dataset.tourStart); return; }
    const roleBtn = t.closest('[data-set-role]');
    if (roleBtn) { setRole(roleBtn.dataset.setRole); return; }
  });
  document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); $('#palette').hidden ? openPalette() : closePalette(); return; }
    if (!$('#palette').hidden) {
      if (e.key === 'Escape') closePalette();
      else if (e.key === 'ArrowDown') { e.preventDefault(); palIndex = Math.min(palItems.length - 1, palIndex + 1); renderPalette(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); palIndex = Math.max(0, palIndex - 1); renderPalette(); }
      else if (e.key === 'Enter') { e.preventDefault(); runPal(palIndex); }
      return;
    }
    if (e.key === 'Escape') { if ($('#modal').classList.contains('open')) modal.close(); else if ($('#drawer').classList.contains('open')) drawer.close(); else if (state.tour) endTour(); }
  });
  Decisions.on(() => {
    $$('[data-decide]').forEach(el => { const id = el.dataset.decide, d = Decisions.get(id); $$('button', el).forEach(b => b.setAttribute('aria-pressed', String(b.dataset.d === d))); });
    const c = Decisions.counts(), n = c.must + c.later + c.cut;
    const navBp = $('[data-nav="blueprint"] .count');
    if (navBp) navBp.textContent = n ? n + ' decided' : '';
    if (state.bp) renderBpPanel();
    document.dispatchEvent(new CustomEvent('decisionschange'));
  });

  /* ---------------- per-element styles under a strict CSP ----------------
     The server sends style-src 'self', which drops inline style attributes. Markup and templates write
     data-style="" instead; each distinct value becomes one rule in a constructed stylesheet (CSSOM is not
     blocked by CSP). !important keeps inline-style priority; :not([hidden]) keeps hidden elements hidden. */
  const STYLE_SHEET = typeof CSSStyleSheet === 'function' ? new CSSStyleSheet() : null;
  const styled = new Set();
  if (STYLE_SHEET) document.adoptedStyleSheets = [...document.adoptedStyleSheets, STYLE_SHEET];
  function applyDataStyles(root) {
    if (!STYLE_SHEET) return;
    const els = root.nodeType === 1 && root.hasAttribute('data-style') ? [root] : [];
    if (root.querySelectorAll) els.push(...root.querySelectorAll('[data-style]'));
    els.forEach(el => {
      const v = el.getAttribute('data-style');
      if (!v || styled.has(v)) return;
      styled.add(v);
      const decls = v.split(';').map(d => d.trim()).filter(Boolean).map(d => {
        const i = d.indexOf(':');
        return i > 0 ? `${d.slice(0, i).trim()}:${d.slice(i + 1).trim().replace(/\s*!important$/, '')} !important` : '';
      }).filter(Boolean).join(';');
      const sel = `[data-style="${v.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"]:not([hidden])`;
      try { STYLE_SHEET.insertRule(`${sel}{${decls}}`, STYLE_SHEET.cssRules.length); } catch (e) { console.warn('data-style rule skipped', v); }
    });
  }
  applyDataStyles(document);
  new MutationObserver(list => list.forEach(m => {
    if (m.type === 'attributes') applyDataStyles(m.target);
    else m.addedNodes.forEach(n => { if (n.nodeType === 1) applyDataStyles(n); });
  })).observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ['data-style'] });

  /* ---------------- server API (CSRF token from /api/config, JSON in and out) ---------------- */
  const server = { config: null };
  async function config() {
    if (!server.config) server.config = await fetch('/api/config', { headers: { Accept: 'application/json' } }).then(r => r.json());
    return server.config;
  }
  async function api(path, opts = {}, retried = false) {
    const cfg = await config();
    const headers = Object.assign({ Accept: 'application/json', 'X-DCLab-Token': cfg.csrf }, opts.headers || {});
    let body = opts.body;
    if (body != null && typeof body === 'object' && !(body instanceof Blob) && !(body instanceof ArrayBuffer) && !(body instanceof FormData)) {
      body = JSON.stringify(body); headers['Content-Type'] = 'application/json';
    }
    const res = await fetch('/api' + path, Object.assign({}, opts, { headers, body }));
    if (res.status === 403 && !retried && (opts.method || 'GET') !== 'GET') {
      // The server restarted and issued a new request token: fetch it once and try again. A role refusal (package 10.2)
      // is a 403 too, but saying so again would not change it: only the token's refusal is retried.
      const detail = await res.clone().json().then(d => d.detail).catch(() => null);
      if (detail === 'Missing local UI request token') { server.config = null; return api(path, opts, true); }
    }
    if (res.status === 204) return null;
    const type = res.headers.get('content-type') || '';
    const data = type.includes('json') ? await res.json() : await res.text();
    if (!res.ok) {
      const msg = data && data.detail ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) : `Request failed (${res.status})`;
      const err = new Error(msg); err.status = res.status; err.data = data; throw err;
    }
    return data;
  }
  /* Server-sent events: onEvent(type, data) for every event; returns a function that closes the stream. */
  function stream(path, onEvent) {
    const src = new EventSource('/api' + path);
    const handler = e => { let data = e.data; try { data = JSON.parse(e.data); } catch (err) { /* plain text */ } onEvent(e.type, data); };
    ['message', 'chat', 'token', 'pipeline', 'analysis', 'workflow', 'question', 'status', 'error', 'done'].forEach(t => src.addEventListener(t, handler));
    return () => src.close();
  }
  /* Repeat fn every ms until it returns true or the returned stop function is called. */
  function poll(fn, ms = 1500) {
    let stopped = false, timer = null;
    const tick = async () => { if (stopped) return; let done = false; try { done = await fn(); } catch (e) { console.error(e); } if (!done && !stopped) timer = setTimeout(tick, ms); };
    tick();
    return () => { stopped = true; clearTimeout(timer); };
  }

  /* ---------------- the open project (project pages share it) ----------------
     DC.state.project is "p:<id>" for a real project (set by the Workflow page) or a sample key. With no choice yet,
     the most recent real project opens; with none at all, the pages keep their sample. */
  const currentProject = {
    _cache: null,
    async id() {
      const key = state.project;
      if (typeof key === 'string' && key.startsWith('p:')) return key.slice(2);
      if (key) return null;  // a sample was chosen on purpose
      try { const ws = await api('/workspace'); if (ws.projects.length) { state.project = 'p:' + ws.projects[0].id; return ws.projects[0].id; } } catch (e) { /* offline */ }
      return null;
    },
    /* The project with its stage records, transitions and graph; cached for two seconds across pages. */
    async get(force) {
      const id = await this.id();
      if (!id) return null;
      const c = this._cache;
      if (!force && c && c.id === id && Date.now() - c.at < 2000) return c.project;
      const project = await api(`/projects/${id}`);
      this._cache = { id, at: Date.now(), project };
      setProjectLabel(project.name);
      return project;
    },
    clear() { this._cache = null; },
  };
  /* Shows or hides the "Sample data" pill in the top bar and a page's own sample note. */
  function markSample(isSample) {
    const pill = $('.demo-pill');
    if (pill) pill.textContent = isSample ? 'Sample data · Demo' : 'Your data';
  }

  /* ---------------- synthetic templates (the Synthetic tabs of Home and the wizard) ----------------
     Without a model the rows come from a built-in template, so the tab shows a template choice and says so.
     With a model the field stays hidden: the table is designed from the description. */
  const synthetic = {
    _load: null,
    templates() { return this._load || (this._load = api('/synthetic/templates').catch(() => { this._load = null; return null; })); },
    async setup(field) {
      const info = await this.templates();
      if (!field || !info || info.model || !(info.templates || []).length) return;
      const select = $('select', field);
      select.innerHTML = '<option value="">Closest to my description</option>' + info.templates.map(t => `<option value="${esc(t.key)}">${esc(t.name)}${t.target ? ' · predicts ' + esc(t.target) : ''}</option>`).join('');
      const about = () => { const t = info.templates.find(x => x.key === select.value); $('.hint', field).textContent = t ? `${t.description} ${t.columns} columns.` : 'No model is configured, so the rows come from a built-in template, not from your description. With a model, DCLab designs the table from your words.'; };
      select.addEventListener('change', about);
      about();
      field.hidden = false;
    },
    chosen(field) { return field && !field.hidden ? ($('select', field).value || null) : null; },
  };

  /* ---------------- data connectors (Home and the wizard share them) ----------------
     Credentials and connection strings live on the server; the page only names a configured connection. */
  const connectors = {
    _status: null,
    async status() { if (!this._status) { try { this._status = await api('/connectors'); } catch (e) { this._status = {}; } } return this._status; },
    async kaggleSearch(query, listEl) {
      const st = await this.status();
      if (st.kaggle && !st.kaggle.configured) { listEl.innerHTML = `<div class="empty">${esc(st.kaggle.note)}</div>`; return; }
      if (!query.trim()) { toast('Type what to search for.', { ok: false }); return; }
      listEl.innerHTML = '<div class="empty">Searching Kaggle…</div>';
      try {
        const rows = await api('/connectors/kaggle/search', { method: 'POST', body: { query } });
        listEl.innerHTML = rows.length ? rows.slice(0, 8).map(r => `<div class="list-item"><div class="li-main"><span class="li-title">${esc(r.ref)}</span><span class="li-sub">${esc(r.title || '')}${r.size_bytes ? ' · ' + (r.size_bytes / 1048576).toFixed(1) + ' MB' : ''}${r.license ? ' · licence ' + esc(r.license) : ''}</span></div><button type="button" class="btn sm" data-kaggle-ref="${esc(r.ref)}">Import</button></div>`).join('')
          : '<div class="empty">No dataset matched. Try other words.</div>';
      } catch (e) { listEl.innerHTML = `<div class="empty">${esc(e.message)}</div>`; }
    },
    importKaggle(draftId, ref) { return api(`/drafts/${draftId}/data/kaggle`, { method: 'POST', body: { ref } }); },
    importHF(draftId, f) {
      if (!f.dataset) { toast('Give the dataset ID, for example scikit-learn/adult-census-income.', { ok: false }); return Promise.reject(new Error('no dataset')); }
      return api(`/drafts/${draftId}/data/hf`, { method: 'POST', body: { dataset: f.dataset, revision: f.revision || null, split: f.split || 'train', config: f.config || null } });
    },
    async database(draftId, done) {
      const st = await this.status(), conns = (st.database && st.database.connections) || [];
      if (!conns.length) {
        modal.open({ eyebrow: '<span class="eyebrow">Warehouse</span>', title: 'No database connection yet', hideConfirm: true, cancel: 'Close',
          html: `<p>${esc((st.database && st.database.note) || '')}</p><pre class="code">DCLAB_DB_WAREHOUSE=postgresql+psycopg2://reader:…@host/db</pre><p class="small muted">Use a read-only database user. The connection string stays on the server; this page only sees the name.</p>` });
        return;
      }
      modal.open({
        eyebrow: '<span class="eyebrow">Warehouse · read-only</span>', title: 'Import from a database', confirm: 'Import',
        html: `<div class="field"><label for="cx-conn">Connection</label><select id="cx-conn">${conns.map(c => `<option>${esc(c)}</option>`).join('')}</select></div>
          <div class="field"><label for="cx-table">Table</label><input id="cx-table" type="text" placeholder="schema.table"></div>
          <div class="field"><label for="cx-query">…or a SELECT query</label><textarea id="cx-query" placeholder="SELECT * FROM orders WHERE created_at >= '2025-01-01'"></textarea><span class="hint">One SELECT (or WITH) statement. Nothing that writes is accepted.</span></div>
          <div class="field"><label for="cx-limit">Row limit</label><input id="cx-limit" type="number" value="200000" min="1" max="1000000"></div>`,
        onConfirm: m => {
          const table = $('#cx-table', m).value.trim(), query = $('#cx-query', m).value.trim();
          if (!table === !query) { toast('Give a table or a query, not both.', { ok: false }); return false; }
          api(`/drafts/${draftId}/data/database`, { method: 'POST', body: { connection: $('#cx-conn', m).value, table: table || null, query: query || null, limit: Number($('#cx-limit', m).value) } })
            .then(a => { modal.close(); toast('Import started.'); done && done(a); }).catch(e => toast(e.message, { ok: false }));
          return false;
        },
      });
    },
    async cloud(draftId, done) {
      const st = await this.status();
      modal.open({
        eyebrow: '<span class="eyebrow">Cloud storage</span>', title: 'Import a file from S3 or Google Cloud Storage', confirm: 'Import',
        html: `<div class="field"><label for="cx-uri">Object</label><input id="cx-uri" type="text" placeholder="s3://bucket/path/data.parquet or gs://bucket/path/data.csv"><span class="hint">${esc((st.cloud && st.cloud.note) || '')}</span></div>`,
        onConfirm: m => {
          const uri = $('#cx-uri', m).value.trim();
          if (!/^(s3|gs):\/\/./.test(uri)) { toast('Give an s3:// or gs:// path.', { ok: false }); return false; }
          api(`/drafts/${draftId}/data/cloud`, { method: 'POST', body: { uri } }).then(a => { modal.close(); toast('Import started.'); done && done(a); }).catch(e => toast(e.message, { ok: false }));
          return false;
        },
      });
    },
  };

  /* ---------------- boot ---------------- */
  function boot() {
    renderNav();
    $('#menu-btn').addEventListener('click', () => document.body.classList.toggle('nav-open'));
    $('#side-scrim').addEventListener('click', () => document.body.classList.remove('nav-open'));
    $('#search-btn').addEventListener('click', openPalette);
    $('#palette').addEventListener('click', e => { if (e.target.id === 'palette') closePalette(); });
    $('#pal-input').addEventListener('input', () => { palIndex = 0; renderPalette(); });
    $('#scrim').addEventListener('click', () => { drawer.close(); modal.close(); });
    $('#drawer-close').addEventListener('click', drawer.close);
    $('#modal-cancel').addEventListener('click', modal.close);
    $('#modal-confirm').addEventListener('click', () => { const fn = modalConfirm; if (fn && fn($('#modal')) === false) return; modal.close(); });
    $('#bp-switch').addEventListener('change', e => setBlueprint(e.target.checked));
    $('#bp-close').addEventListener('click', () => setBlueprint(false));
    $('#bp-expand').addEventListener('click', () => { const p = $('#bp-panel'); const min = p.classList.toggle('min'); $('#bp-expand').textContent = min ? 'Show list' : 'Hide list'; $('#bp-expand').setAttribute('aria-expanded', String(!min)); renderBpPanel(); });
    $('#role-select').addEventListener('change', e => setRole(e.target.value));
    $('#theme-btn').addEventListener('click', () => {
      const root = document.documentElement;
      const dark = root.dataset.theme ? root.dataset.theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.dataset.theme = dark ? 'light' : 'dark';
      $('#theme-btn').innerHTML = icon(dark ? 'moon' : 'sun');
    });
    $('#theme-btn').innerHTML = icon(window.matchMedia('(prefers-color-scheme: dark)').matches ? 'sun' : 'moon');
    $('#side-tours').addEventListener('click', () => {
      modal.open({
        eyebrow: '<span class="eyebrow accent">Guided tours</span>', title: 'Walk through the product', hideConfirm: true, cancel: 'Close',
        html: `<p>Each tour opens the pages in order and points at the part that matters. Use the bar at the bottom to move.</p><div class="stack tight">${Object.entries(TOURS).map(([k, t]) => `<button type="button" class="list-item panel" data-tour-start="${k}" data-style="border-radius:10px"><div class="li-main"><span class="li-title">${esc(t.name)}</span><span class="li-sub">${esc(t.blurb)}</span></div><span class="pill outline">${t.steps.length} steps</span></button>`).join('')}</div>`,
      });
    });
    $('#tb-next').addEventListener('click', () => { const t = TOURS[state.tour]; if (!t) return; if (state.step < t.steps.length - 1) { state.step++; gotoStep(); } else { endTour(); toast('Tour finished. Turn on the blueprint layer to see what exists today.'); } });
    $('#tb-back').addEventListener('click', () => { if (state.step > 0) { state.step--; gotoStep(); } });
    $('#tb-exit').addEventListener('click', endTour);
    document.addEventListener('click', e => { if (e.target.closest('[data-tour-start]')) modal.close(); }, true);
    Decisions.init();
    let bpOn = false;
    try { bpOn = localStorage.getItem('dclab-demo-bp') === '1'; } catch (e) { /* ignore */ }
    setRole('developer');
    window.addEventListener('hashchange', () => { const h = parseHash(); show(h.view, h.pane); });
    const h0 = parseHash();
    show(h0.view, h0.pane);
    if (bpOn) setBlueprint(true);
    loadRecords();
  }

  window.DC = { explanation, synthetic, loadRecords, setNavCount, currentProject, markSample, connectors, setProjectLabel, graph, api, stream, poll, config, applyDataStyles, selectPane, reveal, $, $$, esc, fmt, pct, int, icon, chip, chips, linkIds, openRecord, toast, drawer, modal, charts, binormal, Phi, PhiInv, highlightPy, codeBlock, view, hydrate, Decisions, decideButtons, FEATURES, FMAP, STATUS_LABEL, RECORDS, REC, state, setRole, setBlueprint, startTour, applyBlueprintAttrs, copyText, TYPE_LABEL, TYPE_CLS };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else setTimeout(boot, 0);
})();
