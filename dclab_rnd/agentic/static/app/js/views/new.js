DC.view('new', {
  /* The wizard works on a draft (the same object Home fills) and turns it into a project only at "Review and
     start": goal → data → solution draft → split and budget → review. Every step saves to /api/drafts/{id}. */
  init(el) {
    const { $, $$, esc, chip, icon, api, stream, toast, graph } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id)).replace(/\$\{icon:([a-z]+)\}/g, (m, n) => icon(n));
    const KEY = 'dclab-home-draft';
    const W = { draft: null, close: null, packs: [], proposal: null, step: 1, samples: null };
    const fmtN = v => (v == null ? '—' : Number(v).toLocaleString('en-US'));
    const pctTxt = v => (v == null ? '—' : (v * 100).toFixed(v < 0.1 ? 1 : 0) + '%');
    const remember = id => { try { id ? localStorage.setItem(KEY, id) : localStorage.removeItem(KEY); } catch (e) { /* blocked */ } };

    /* ---------- steps ---------- */
    const show = n => {
      W.step = n;
      $$('[data-new-step]', el).forEach(p => { p.hidden = p.dataset.newStep !== String(n); });
      $$('#new-steps button', el).forEach(b => { const s = Number(b.dataset.step); b.classList.toggle('done', s < n); if (s === n) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current'); });
      if (n === 3) enterSolution();
      if (n === 4) enterSettings();
      if (n === 5) enterReview();
      DC.hydrate(el);
    };
    el.addEventListener('click', e => {
      const b = e.target.closest('[data-next]'); if (b) { show(Number(b.dataset.next)); window.scrollTo({ top: 0, behavior: 'smooth' }); }
      const s = e.target.closest('#new-steps button'); if (s) { const n = Number(s.dataset.step); if (n === 1 || W.draft) show(n); }
    });

    /* ---------- loading the draft ---------- */
    async function load(id) {
      const d = await api(`/drafts/${id}`);
      if (d.status !== 'open') { remember(null); throw new Error('built'); }
      W.draft = d;
      remember(id);
      if (W.close) W.close();
      W.close = stream(`/drafts/${id}/events`, onEvent);
      render();
    }
    async function refresh() { if (W.draft) { W.draft = await api(`/drafts/${W.draft.id}`); render(); } }
    function onEvent(kind, data) {
      if (kind === 'pipeline' && ['ready', 'failed', 'queued', 'structured', 'cleaned'].includes(data.step)) refresh();
      if (kind === 'workflow' && W.draft) { W.draft.workflow = data.workflow; drawGraph(); }
      if (kind === 'status' && data.pack && W.draft) { W.draft.pack = data.pack; drawPacks(); drawStats(); }
      if (kind === 'plan' && W.draft) { W.draft.plan = data.plan; W.draft.understanding = data.understanding || W.draft.understanding; drawUnderstood(); }
    }
    function render() {
      const d = W.draft; if (!d) return;
      if (document.activeElement !== $('#new-goal', el)) $('#new-goal', el).value = d.problem;
      drawUnderstood(); drawAssets(); drawStats(); drawProfile(); drawPacks(); drawGraph();
    }

    /* ---------- step 1: goal ---------- */
    /* the agent's plan (package A3.1): how each line is known, and the user's words behind it */
    const PLAN_PILL = { problem: ['ok', 'from your sentence'], answer: ['ok', 'from your answer'], model: ['ok', 'read from your words'], data: ['ok', 'data attached'] };
    function planPill(entry, next, field, v) {
      if (!entry) return v ? '<span class="pill ok">from the chat</span>' : '<span class="pill warn">open</span>';
      if (entry.status === 'inferred') return '<span class="pill warn" title="Implied by your words, not said outright">inferred: check it</span>';
      if (entry.status === 'unknown') return next === field ? '<span class="pill accent">asked next</span>' : '<span class="pill warn">open</span>';
      const [tone, label] = PLAN_PILL[entry.source] || ['ok', 'from the chat'];
      return `<span class="pill ${tone}">${label}</span>`;
    }
    function drawUnderstood() {
      const u = (W.draft && W.draft.understanding) || {}, plan = (W.draft && W.draft.plan) || {};
      const rows = [['Target', 'target', 'What exactly is predicted'], ['Prediction moment', 'prediction_moment', 'When the prediction is made'],
        ['Action', 'action', 'What happens with each prediction'], ['Costs', 'costs', 'What a wrong prediction costs'], ['Data', 'data_plan', 'Upload, connect or simulate']];
      $('#wz-understood', el).innerHTML = rows.map(([k, f, hint]) => {
        const v = u[f], entry = plan[f], quote = entry && entry.quote && entry.quote !== v ? `<span class="small muted"> · your words: “${esc(entry.quote)}”</span>` : '';
        return `<div class="cs-row"><span class="cs-key">${k}</span><span class="cs-val">${v ? esc(v) + quote : `<span class="muted">${hint}: not stated yet</span>`}</span>${planPill(entry, plan.next, f, v)}</div>`;
      }).join('');
    }
    $('#wz-goal-next', el).addEventListener('click', async () => {
      const problem = $('#new-goal', el).value.trim();
      if (problem.length < 8) { toast('Describe the problem in one sentence first.', { ok: false }); return; }
      try {
        if (!W.draft) { const d = await api('/drafts', { method: 'POST', body: { problem } }); await load(d.id); }
        else if (problem !== W.draft.problem) { W.draft = await api(`/drafts/${W.draft.id}`, { method: 'PATCH', body: { problem } }); render(); }
        show(2);
      } catch (e) { toast(e.message, { ok: false }); }
    });

    /* ---------- step 2: data ---------- */
    function drawAssets() {
      const d = W.draft, list = $('#wz-assets', el);
      list.innerHTML = (d.assets || []).map(a => `<div class="list-item panel"><div class="li-main"><span class="li-title">${esc(a.name)}${a.synthetic ? ' <span class="pill warn">synthetic</span>' : ''}</span><span class="li-sub">${a.status === 'ready' ? `${fmtN(a.rows)} rows · ${fmtN(a.columns)} columns${a.format ? ' · read as ' + esc(a.format.replace('_', ' ')) : ''}${a.template ? ' · built-in template, not designed from the description' : ''}` : a.status === 'failed' ? esc(a.error || 'failed') : 'Processing: ' + esc(a.status)}</span></div><span class="pill ${a.status === 'ready' ? 'ok' : a.status === 'failed' ? 'bad' : 'accent'}">${esc(a.status)}</span></div>`).join('');
      const ready = (d.assets || []).find(a => a.id === d.active_asset && a.status === 'ready');
      $('#wz-data-next', el).disabled = !ready;
      $('#wz-data-summary', el).hidden = !ready;
      if (ready) {
        const s = (d.analysis || {}).summary || {};
        $('#wz-data-text', el).innerHTML = `<b>${esc(ready.name)}</b> · ${fmtN(s.rows)} rows · ${fmtN(s.columns)} columns · missing ${pctTxt(s.missing_cell_rate)} · ${(d.cleaning_log || []).length} cleaning steps`;
      }
    }
    async function uploadFile(file) {
      if (!file || !W.draft) return;
      try { toast(`Uploading ${file.name}…`); await api(`/drafts/${W.draft.id}/data?filename=${encodeURIComponent(file.name)}`, { method: 'PUT', body: file, headers: { 'Content-Type': 'application/octet-stream' } }); refresh(); }
      catch (e) { toast(e.message, { ok: false }); }
    }
    $('#wz-file', el).addEventListener('change', e => uploadFile(e.target.files[0]));
    const drop = $('#wz-drop', el);
    drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('over'));
    drop.addEventListener('drop', e => { e.preventDefault(); drop.classList.remove('over'); uploadFile(e.dataTransfer.files[0]); });
    async function loadSamples() {
      if (W.samples) return;
      try { W.samples = await api('/samples'); } catch (e) { W.samples = []; }
      $('#wz-sample-count', el).textContent = W.samples.length;
      $('#sample-table tbody', el).innerHTML = W.samples.map(s => `<tr data-sample="${esc(s.key)}"><td><span class="cell-main">${esc(s.key)}</span><div class="cell-sub">${esc(s.name || '')}</div></td><td>${esc(s.task || '')}</td><td class="num">${fmtN(s.rows)}</td><td class="small">${esc(s.decision || '')}</td><td><button type="button" class="btn sm">Use</button></td></tr>`).join('');
    }
    $('[data-tabs="wsrc"]', el).addEventListener('tabchange', e => { if (e.detail === 'samples') loadSamples(); });
    $('#sample-table', el).addEventListener('click', async e => {
      const tr = e.target.closest('tr[data-sample]'); if (!tr || !W.draft || e.target.closest('[data-record]')) return;
      try { await api(`/drafts/${W.draft.id}/data/sample`, { method: 'POST', body: { key: tr.dataset.sample } }); refresh(); } catch (err) { toast(err.message, { ok: false }); }
    });
    DC.synthetic.setup($('#wz-syn-template-field', el));
    $('#wz-syn-go', el).addEventListener('click', async () => {
      if (!W.draft) return;
      try { await api(`/drafts/${W.draft.id}/data/synthetic`, { method: 'POST', body: { prompt: $('#wz-syn-prompt', el).value || W.draft.problem, rows: Number($('#wz-syn-rows', el).value), template: DC.synthetic.chosen($('#wz-syn-template-field', el)) } }); toast('Generating the synthetic data…'); }
      catch (e) { toast(e.message, { ok: false }); }
    });
    el.addEventListener('click', async e => {
      const X = DC.connectors, k = e.target.closest('[data-kaggle-ref]');
      if (k && W.draft) { try { await X.importKaggle(W.draft.id, k.dataset.kaggleRef); k.disabled = true; k.textContent = 'Importing…'; refresh(); } catch (err) { toast(err.message, { ok: false }); } return; }
      const c = e.target.closest('[data-connect]'); if (!c || !W.draft) return;
      const kind = c.dataset.connect;
      if (kind === 'kaggle') X.kaggleSearch($('#wz-kaggle-q', el).value, $('#wz-kaggle-list', el));
      if (kind === 'hf') X.importHF(W.draft.id, { dataset: $('#hf-id', el).value.trim(), revision: $('#hf-rev', el).value.trim(), split: $('#hf-split', el).value.trim(), config: $('#hf-cfg', el).value.trim() }).then(refresh).catch(err => { if (err.message !== 'no dataset') toast(err.message, { ok: false }); });
      if (kind === 'database') X.database(W.draft.id, refresh);
      if (kind === 'cloud') X.cloud(W.draft.id, refresh);
    });
    $('#wz-data-next', el).addEventListener('click', () => { show(3); window.scrollTo({ top: 0, behavior: 'smooth' }); });

    /* ---------- profile pane ---------- */
    function drawProfile() {
      const d = W.draft, a = d.analysis;
      if (!a) return;
      const asset = (d.assets || []).find(x => x.id === d.active_asset) || {};
      const s = a.summary || {}, prof = a.profile || {};
      $('#wz-profile-title', el).textContent = 'Profile · ' + (asset.name || 'data');
      $('#wz-profile-sub', el).textContent = `${fmtN(s.rows)} rows · ${fmtN(s.columns)} columns · missing ${pctTxt(s.missing_cell_rate)} · duplicate rows left ${fmtN(s.duplicate_rows)}`;
      const pill = $('#wz-profile-pill', el); pill.hidden = false; pill.textContent = asset.synthetic ? 'synthetic' : 'profiled';
      pill.className = 'pill ' + (asset.synthetic ? 'warn' : 'ok');
      const guess = c => (prof.target_candidates || []).includes(c.name) ? '<span class="pill accent">target?</span>' : c.id_like || c.name_id_like ? '<span class="pill outline">identifier?</span>' : c.time_like ? 'time' : c.text_like ? 'text' : 'input';
      $('#profile-table tbody', el).innerHTML = (prof.columns || []).map(c => `<tr><td><code>${esc(c.name)}</code></td><td>${esc(c.kind)}</td><td class="num">${fmtN(c.unique)}</td><td class="num">${pctTxt(c.missing_rate)}</td><td class="mono small">${esc(String((c.preview || [])[0] ?? ''))}</td><td>${guess(c)}</td></tr>`).join('');
      const log = d.cleaning_log || [], st = d.structure || {};
      $('#wz-lineage', el).innerHTML = [
        `<div class="tl-item"><span class="tl-mark ok">1</span><div class="tl-body"><span class="tl-title">${esc(asset.name || 'data')}</span><span class="tl-meta">${esc(asset.kind || '')}${asset.synthetic ? ' · synthetic' : ''} · read as ${esc((st.format || '').replace('_', ' '))}${st.parse_rate != null && st.parse_rate < 1 ? ' · ' + pctTxt(st.parse_rate) + ' of lines parsed' : ''}</span></div></div>`,
        ...log.map((l, i) => `<div class="tl-item"><span class="tl-mark ok">${i + 2}</span><div class="tl-body"><span class="tl-title">${esc(l.title)}</span><span class="tl-meta">${esc(l.detail)}</span></div></div>`),
        `<div class="tl-item"><span class="tl-mark accent">${log.length + 2}</span><div class="tl-body"><span class="tl-title">Cleaned table</span><span class="tl-meta">${fmtN(asset.rows)} rows · ${fmtN(asset.columns)} columns · ready for the solution</span></div></div>`,
      ].join('') + (st.notes && st.notes.length ? `<div class="callout info"><span class="ic">${icon('info')}</span><span>${esc(st.notes.join(' '))}</span></div>` : '');
    }

    /* ---------- step 3: solution draft ---------- */
    function columns() { return (((W.draft || {}).analysis || {}).profile || {}).columns || []; }
    function enterSolution() {
      const cols = columns().map(c => c.name);
      const u = W.draft.understanding || {}, sol = W.draft.solution, cands = ((W.draft.analysis || {}).profile || {}).target_candidates || [];
      const own = (((W.draft.assets || []).find(a => a.id === W.draft.active_asset) || {}).suggestion || {}).target;  // a sample or a synthetic table knows its outcome
      const pick = sol ? sol.target : cols.includes(u.target) ? u.target : cols.includes(own) ? own : cands[0] || '';
      if (W.proposalFor !== W.draft.id) { W.proposal = null; W.proposalFor = W.draft.id; }  // never show another draft's audit
      $('#wz-target', el).innerHTML = '<option value="">choose the outcome column</option>' + cols.map(c => `<option ${c === pick ? 'selected' : ''}>${esc(c)}</option>`).join('');
      if (sol) { W.proposal = W.draft.proposal || W.proposal; drawSheet(sol); }
      else if (pick) propose(false);
      else $('#wz-sheet', el).innerHTML = '<div class="empty">Choose the outcome column, then DCLab audits every other column against the prediction moment.</div>';
    }
    /* keepTask: the user set the Task box and asked for the audit again. Otherwise the task is detected from the
       target, so a task chosen for another column (or another draft) is never forced onto this one. */
    /* what the person chose on the sheet, so a review against a new moment never undoes it */
    function choices() {
      return {
        ticked: $$('[data-forbid]:checked', el).map(i => ({ column: i.dataset.forbid, reason: i.dataset.reason || '' })),
        unticked: new Set($$('[data-forbid]:not(:checked)', el).map(i => i.dataset.forbid)),
        idents: new Set($$('[data-ident]:checked', el).map(i => i.dataset.ident)), identsOff: new Set($$('[data-ident]:not(:checked)', el).map(i => i.dataset.ident)),
        time: ($('#wz-time', el) || {}).value, group: ($('#wz-group', el) || {}).value, metric: ($('#wz-metric', el) || {}).value,
      };
    }
    function reapply(c) {
      $$('[data-forbid]', el).forEach(i => { if (c.ticked.some(f => f.column === i.dataset.forbid)) i.checked = true; else if (c.unticked.has(i.dataset.forbid)) i.checked = false; });
      $$('[data-ident]', el).forEach(i => { if (c.idents.has(i.dataset.ident)) i.checked = true; else if (c.identsOff.has(i.dataset.ident)) i.checked = false; });
      [['#wz-time', c.time], ['#wz-group', c.group], ['#wz-metric', c.metric]].forEach(([id, v]) => { const sel = $(id, el); if (sel && v != null && [...sel.options].some(o => o.value === v)) sel.value = v; });
    }
    async function propose(keepTask, keepChoices) {
      const target = $('#wz-target', el).value; if (!target) return;
      const kept = keepChoices ? choices() : null, accepted = W.draft.solution;
      $('#wz-sol-status', el).textContent = 'Auditing the columns…';
      try {
        const moment = (($('#wz-moment', el) || {}).value || '').trim() || undefined;  // the reviewer judges the columns against this moment
        const proposal = await api(`/drafts/${W.draft.id}/solution/proposal`, { method: 'POST', body: { target, task: (keepTask === true && $('#wz-task', el).value) || undefined, prediction_moment: moment } });
        $('#wz-sol-status', el).textContent = '';
        if (W.draft.solution !== accepted) return;  // accepted while the review ran: the accepted sheet stays
        if (kept) {  // a column the person ticked stays listed (and ticked) even if the new review no longer names it
          const listed = new Set((proposal.forbidden || []).concat((proposal.review || {}).items || []).map(f => f.column));
          proposal.forbidden = (proposal.forbidden || []).concat(kept.ticked.filter(f => !listed.has(f.column)));
        }
        W.proposal = proposal;
        drawSheet(null);
        if (kept) reapply(kept);
      } catch (e) { $('#wz-sol-status', el).textContent = e.message; }
    }
    $('#wz-propose', el).addEventListener('click', () => propose(true));
    $('#wz-target', el).addEventListener('change', () => propose(false));
    $('#wz-sheet', el).addEventListener('change', e => { if (e.target.id === 'wz-moment' && e.target.value.trim().length >= 12) propose(true, true); });  // a new moment: review again, keep the choices
    /* the leakage reviewer (package A3.2): what the moment or the reviewer model says is written after the moment.
       Shown unticked, never applied: the person decides. A flag the reviewer disputes stays ticked. */
    const REVIEW_SOURCE = { moment: 'from the moment', model: 'reviewer model' };
    function reviewRow(p, flagged, saved) {
      const r = p.review || {}, taken = new Set(flagged.map(f => f.column).concat(saved ? saved.forbidden.map(f => f.column) : []));
      const extra = (r.items || []).filter(i => !i.apply && !taken.has(i.column));
      const disputes = r.disagreements || [];
      if (!extra.length && !disputes.length) return '';
      return `<div class="cs-row"><span class="cs-key">The reviewer also asks</span><span class="cs-val stack tight">${extra.map(i => `<label class="check"><input type="checkbox" data-forbid="${esc(i.column)}" data-reason="${esc(i.reason || '')}"><span><code>${esc(i.column)}</code> <span class="pill outline">${REVIEW_SOURCE[i.source] || esc(i.source)}</span> <span class="muted small">${esc(i.reason || '')}</span> ${(i.records || []).map(id => chip(id)).join('')}</span></label>`).join('')}${disputes.map(d => `<div class="small muted">The reviewer thinks <code>${esc(d.column)}</code> is known at the moment (${esc(d.reason)}); the audit's flag stays until you untick it.</div>`).join('')}<div class="small muted">Not applied: tick a column only if it is written after the prediction moment.</div></span>${chip('DCLAB-R01')}</div>`;
    }
    function drawSheet(saved) {
      const p = W.proposal || {}, cols = columns().map(c => c.name).filter(c => c !== (saved ? saved.target : p.target));
      const forb = saved ? saved.forbidden : (p.forbidden || []).map(f => ({ column: f.column, reason: f.reason }));
      const forbSet = new Set(forb.map(f => f.column));
      const flagged = (p.forbidden || []).concat(saved ? saved.forbidden.filter(f => !(p.forbidden || []).some(x => x.column === f.column)) : []);
      const ids = new Set(saved ? saved.identifiers : p.identifiers || []);
      const task = saved ? saved.task : p.task;
      $('#wz-task', el).value = task || '';
      const sel = (id, opts, cur) => `<select id="${id}" data-style="width:auto;min-width:180px"><option value="">none</option>${opts.map(o => `<option ${o === cur ? 'selected' : ''}>${esc(o)}</option>`).join('')}</select>`;
      $('#wz-sheet', el).innerHTML = `
        <div class="cs-row"><span class="cs-key">Target</span><span class="cs-val"><code>${esc(saved ? saved.target : p.target)}</code> · ${esc(task || '')}${(saved ? saved.positive_label : p.positive_label) != null ? ` · positive label <code>${esc(saved ? saved.positive_label : p.positive_label)}</code>` : ''}${p.detected && p.detected.note ? `<div class="small muted">${esc(p.detected.note)}</div>` : ''}</span><span></span></div>
        <div class="cs-row"><span class="cs-key">Prediction moment</span><span class="cs-val"><textarea id="wz-moment" rows="2" placeholder="For example: when the order is created, before dispatch.">${esc(saved ? saved.prediction_moment : p.prediction_moment || '')}</textarea><div class="small muted">${!saved && !p.prediction_moment && p.prediction_moment_hint ? esc(p.prediction_moment_hint) + ' ' : ''}Anything written after this moment must stay out of the model.</div></span>${chip('DCLAB-R01')}</div>
        <div class="cs-row"><span class="cs-key">Forbidden</span><span class="cs-val stack tight">${flagged.length ? flagged.map(f => `<label class="check"><input type="checkbox" data-forbid="${esc(f.column)}" data-reason="${esc(f.reason || '')}" ${forbSet.has(f.column) ? 'checked' : ''}><span><code>${esc(f.column)}</code> <span class="muted small">${esc(f.reason || '')}</span> ${(f.proof || []).map(id => chip(id)).join('')}</span></label>`).join('') : '<span class="muted">The audit flagged no column. That is a heuristic scan, not proof: confirm against the prediction moment.</span>'}</span><span class="ev-list">${chip('DCLAB-R04')}</span></div>${reviewRow(p, flagged, saved)}
        <div class="cs-row"><span class="cs-key">Identifiers</span><span class="cs-val">${cols.filter(c => ids.has(c) || columns().find(x => x.name === c && (x.id_like || x.name_id_like))).map(c => `<label class="check"><input type="checkbox" data-ident="${esc(c)}" ${ids.has(c) ? 'checked' : ''}><code>${esc(c)}</code></label>`).join('') || '<span class="muted">None found.</span>'}</span><span class="pill outline">never a feature</span></div>
        <div class="cs-row"><span class="cs-key">Time column</span><span class="cs-val">${sel('wz-time', cols, saved ? saved.time_column : (p.time_candidates || [])[0])} <span class="small muted">orders the split; nothing from the future enters a fold</span></span>${chip('PIT-005')}</div>
        <div class="cs-row"><span class="cs-key">Group column</span><span class="cs-val">${sel('wz-group', cols, saved ? saved.group_column : null)} <span class="small muted">rows of one group stay on one side</span></span><span></span></div>
        <div class="cs-row"><span class="cs-key">Metric</span><span class="cs-val"><select id="wz-metric" data-style="width:auto">${(p.metric_options || [saved && saved.metric].filter(Boolean)).map(m => { const [key, label] = Array.isArray(m) ? m : [m, m]; return `<option value="${esc(key)}" ${key === (saved ? saved.metric : p.metric) ? 'selected' : ''}>${esc(label)}</option>`; }).join('')}</select></span>${chip('DCLAB-R18')}</div>`;
      const n = forb.length;
      $('#wz-sol-pill', el).textContent = saved ? 'accepted' : `${n} forbidden · draft`;
      $('#wz-sol-pill', el).className = 'pill ' + (saved ? 'ok' : 'outline');
      DC.hydrate($('#wz-sheet', el));
      drawStats();
    }
    $('#wz-accept', el).addEventListener('click', async () => {
      const p = W.proposal || {}, target = $('#wz-target', el).value;
      if (!target) { toast('Choose the outcome column first.', { ok: false }); return; }
      const body = {
        target, task: $('#wz-task', el).value || p.task, positive_label: p.positive_label != null ? String(p.positive_label) : null,
        prediction_moment: ($('#wz-moment', el) || {}).value || '',
        forbidden: $$('[data-forbid]:checked', el).map(i => ({ column: i.dataset.forbid, reason: i.dataset.reason || 'declared unavailable at the prediction moment' })),
        identifiers: $$('[data-ident]:checked', el).map(i => i.dataset.ident),
        time_column: ($('#wz-time', el) || {}).value || null, group_column: ($('#wz-group', el) || {}).value || null,
        text_columns: p.text_columns || [], metric: ($('#wz-metric', el) || {}).value || null, notes: '',
      };
      if (body.task !== 'binary') body.positive_label = null;
      if (body.prediction_moment.trim().length < 12) { toast('Write the prediction moment first: when is the prediction made, and what is known then?', { ok: false }); ($('#wz-moment', el) || {}).focus?.(); return; }
      try { W.draft = await api(`/drafts/${W.draft.id}/solution`, { method: 'PUT', body }); drawSheet(W.draft.solution); show(4); window.scrollTo({ top: 0, behavior: 'smooth' }); }
      catch (e) { toast(e.message, { ok: false }); }
    });

    /* ---------- packs and the workflow ---------- */
    async function loadPacks() { try { W.packs = await api('/packs'); } catch (e) { W.packs = []; } drawPacks(); }
    function drawPacks() {
      const cur = (W.draft && W.draft.pack) || null;
      $('#pack-grid', el).innerHTML = W.packs.map(p => `<button type="button" class="pack-card" data-pack="${esc(p.key)}" aria-pressed="${cur && cur.key === p.key ? 'true' : 'false'}"><span class="pc-top"><span class="pack-ic">${icon(p.icon)}</span><span class="pill ${esc(p.cls)}">${esc(p.maturity)}</span></span><span class="pc-name">${esc(p.name)}</span><span class="pc-desc">${esc(p.desc)}</span>${cur && cur.key === p.key ? `<span class="pc-foot"><span class="pill accent">${cur.source === 'user' ? 'your choice' : 'detected'}</span></span>` : ''}</button>`).join('');
      $('#wz-pack-why', el).textContent = cur ? cur.why : '';
    }
    $('#pack-grid', el).addEventListener('click', async e => {
      const c = e.target.closest('[data-pack]'); if (!c || !W.draft) return;
      try { W.draft = await api(`/drafts/${W.draft.id}/pack`, { method: 'POST', body: { key: c.dataset.pack } }); drawPacks(); drawGraph(); drawStats(); } catch (err) { toast(err.message, { ok: false }); }
    });
    function drawGraph() { const wf = W.draft && W.draft.workflow; $('#wz-graph', el).innerHTML = wf ? graph.render(wf) : '<div class="empty">The workflow appears once the problem is described.</div>'; }

    /* ---------- step 4: split and budget ---------- */
    function enterSettings() {
      const sol = W.draft.solution || {}, rows = ((W.draft.analysis || {}).summary || {}).rows || 0, st = W.draft.settings || {};
      /* The solution decides the split (engine.split_for), so the control shows it and offers nothing the run would not do. */
      const split = sol.time_column ? 'time' : sol.group_column ? 'group' : 'stratified';
      $$('#wz-split button', el).forEach(b => { b.setAttribute('aria-pressed', String(b.dataset.v === split)); b.disabled = b.dataset.v !== split; });
      const change = ' To change the split, change that column in the Solution draft step.';
      $('#wz-split-hint', el).textContent = sol.time_column ? `Time column ${sol.time_column}: the holdout is the latest period.${change}` : sol.group_column ? `Group column ${sol.group_column}: a group never lands on both sides.${change}` : 'No time or group column in the solution, so stratified random is the honest choice. The brief will say so. To split by time or by group, name that column in the Solution draft step.';
      const quick = Math.min(3000, rows), full = Math.min(rows, 200000);
      $('#new-rows', el).innerHTML = `<option value="quick" ${st.quick !== false ? 'selected' : ''}>Quick · ${fmtN(quick)} rows</option><option value="full" ${st.quick === false ? 'selected' : ''}>Full · ${fmtN(full)} rows${rows > 200000 ? ' (cap)' : ''}</option>`;
      if (st.folds) $('#new-folds', el).value = String(st.folds);
      $$('#wz-where button', el).forEach(b => { b.disabled = b.dataset.v !== 'local'; b.setAttribute('aria-pressed', String(b.dataset.v === 'local')); });
      $('#wz-where-hint', el).textContent = 'Sandboxes and GPU jobs are not switched on in this build; tabular work runs on this machine.';
      if (st.budget) { $('#b-calls', el).value = st.budget.max_steps; $('#b-min', el).value = st.budget.max_minutes; $('#b-eur', el).value = st.budget.eur; }
    }
    $('#wz-settings-next', el).addEventListener('click', async () => {
      const rows = ((W.draft.analysis || {}).summary || {}).rows || 20000, quick = $('#new-rows', el).value === 'quick';
      const body = { quick, max_rows: Math.max(200, Math.min(rows, 200000)), folds: Number($('#new-folds', el).value),
        where: 'local', budget: { calls: Number($('#b-calls', el).value), minutes: Number($('#b-min', el).value), eur: Number($('#b-eur', el).value) }, ask_over_eur: $('#b-ask', el).checked };
      try { W.draft = await api(`/drafts/${W.draft.id}/settings`, { method: 'PUT', body }); show(5); window.scrollTo({ top: 0, behavior: 'smooth' }); } catch (e) { toast(e.message, { ok: false }); }
    });

    /* ---------- step 5: review and start ---------- */
    function enterReview() {
      const d = W.draft, sol = d.solution, st = d.settings || {}, a = (d.assets || []).find(x => x.id === d.active_asset) || {};
      const pack = W.packs.find(p => p.key === (d.pack || {}).key) || {};
      $('#wz-review', el).innerHTML = `
        <dt>Problem</dt><dd>${esc(d.problem)}</dd>
        <dt>Pack</dt><dd>${esc(pack.name || '—')}${sol ? ' · ' + esc(sol.task) : ''}</dd>
        <dt>Data</dt><dd>${a.name ? esc(a.name) + ` · ${fmtN(a.rows)} rows` + (a.synthetic ? ' · <span class="pill warn">synthetic</span>' : '') : '<span class="pill warn">no data</span>'}</dd>
        <dt>Solution</dt><dd>${sol ? `target <code>${esc(sol.target)}</code> · ${sol.forbidden.length} forbidden · metric ${esc(sol.metric || 'default')}` : '<span class="pill warn">not accepted yet</span>'}</dd>
        <dt>Split</dt><dd>${esc({ time: 'by time', group: 'by group', stratified: 'stratified random' }[st.split] || 'stratified random')} · holdout sealed · ${st.folds || 3} folds · ${st.quick === false ? 'full rows' : 'quick mode'}</dd>
        <dt>Budget</dt><dd>${st.budget ? `${st.budget.max_steps} tool calls · ${st.budget.max_minutes} minutes · €${st.budget.eur}` : '—'}</dd>`;
      const ready = !!(sol && a.status === 'ready');
      $('#wz-go-notebook', el).disabled = !ready; $('#wz-go-intern', el).disabled = !ready;
    }
    async function start(mode) {
      try {
        const project = await api(`/drafts/${W.draft.id}/build`, { method: 'POST' });
        remember(null);
        DC.state.projectId = project.id;
        if (mode === 'intern') {
          const st = W.draft.settings || {};
          const s = await api('/intern/sessions', { method: 'POST', body: { task: W.draft.problem, project_id: project.id, budget: st.budget ? { max_steps: st.budget.max_steps, max_minutes: st.budget.max_minutes } : undefined } });
          DC.state.internSession = s.id;
          toast('Handed to the intern with its budget. It asks before anything only you can decide.');
          location.hash = 'intern';
        } else {
          toast('Project created with its data, solution and settings. Run the stages from the workflow.');
          location.hash = 'project';
        }
      } catch (e) { toast(e.message, { ok: false }); }
    }
    $('#wz-go-notebook', el).addEventListener('click', () => start('notebook'));
    $('#wz-go-intern', el).addEventListener('click', () => start('intern'));

    /* ---------- stats ---------- */
    function drawStats() {
      const d = W.draft; if (!d) return;
      const st = $$('#wz-stats .stat', el), a = (d.assets || []).find(x => x.id === d.active_asset) || null;
      const pack = W.packs.find(p => p.key === (d.pack || {}).key);
      const sol = d.solution, forb = sol ? sol.forbidden : (W.proposal ? W.proposal.forbidden : null);
      const open = (d.questions || []).filter(q => !q.answered).length;
      st[0].innerHTML = `<span class="v">${esc(pack ? pack.name.split(' ')[0] : '—')}${sol ? ` <small>${esc(sol.task)}</small>` : ''}</span><span class="l">domain pack · ${esc(d.pack ? (d.pack.source === 'user' ? 'your choice' : 'detected') : 'not chosen')}</span>`;
      st[1].innerHTML = a && a.status === 'ready' ? `<span class="v">${fmtN(a.rows)} <small>rows</small></span><span class="l">${esc(a.name)} · ${fmtN(a.columns)} columns${a.synthetic ? ' · synthetic' : ''}</span>` : `<span class="v">— <small>rows</small></span><span class="l">no data yet</span>`;
      st[2].innerHTML = forb ? `<span class="v ${forb.length ? 'bad' : ''}">${forb.length ? esc(forb[0].column) : '0'}${forb.length > 1 ? ` <small>+${forb.length - 1}</small>` : ''}</span><span class="l">${forb.length ? 'forbidden in the draft · ' + esc((forb[0].reason || '').slice(0, 60)) : 'no column flagged yet'}</span>` : `<span class="v">—</span><span class="l">columns forbidden in the solution draft</span>`;
      st[3].innerHTML = `<span class="v ${open ? 'warn' : ''}">${open} <small>question${open === 1 ? '' : 's'}</small></span><span class="l">${open ? 'open from the conversation on Home' : 'nothing open for you'}</span>`;
    }

    this.show = show; this.load = load;
    this.reset = () => { W.draft = null; W.proposal = null; $('#new-goal', el).value = ''; $('#wz-assets', el).innerHTML = ''; drawUnderstood(); show(1); };
    loadPacks();
  },
  async enter() {
    DC.markSample(false);
    let id = DC.state.draftId || null;
    try { id = id || localStorage.getItem('dclab-home-draft'); } catch (e) { /* blocked */ }
    if (id) { try { await this.load(id); this.show(DC.state.draftId ? 2 : 1); DC.state.draftId = null; return; } catch (e) { /* gone */ } }
    this.reset();
  },
});
