DC.view('home', {
  /* Home: start from one sentence. A draft is created on "Draft the solution"; its events (chat, data pipeline,
     analysis, solution workflow) stream from /api/drafts/{id}/events and are rendered here. The user stays on
     Home until "Let's build the solution" turns the draft into a project. */
  init(el) {
    const { $, $$, esc, chip, charts, icon, api, stream, toast, graph } = DC;
    el.innerHTML = el.innerHTML
      .replace(/\$\{chip:([A-Z0-9_-]+)\}/g, (m, id) => chip(id))
      .replace(/\$\{icon:([a-z]+)\}/g, (m, n) => icon(n))
      .replace('${spark}', charts.spark([2.1, 3.4, 4.0, 6.2, 7.9, 9.4, 12.6, 14.1, 16.0, 18.4], { width: 110, height: 22 }));
    const KEY = 'dclab-home-draft';
    const S = { draft: null, close: null, packs: [], pack: null, analysis: {}, assets: {}, samples: null, thinking: false };
    const remember = id => { try { id ? localStorage.setItem(KEY, id) : localStorage.removeItem(KEY); } catch (e) { /* storage blocked */ } };
    const recalled = () => { try { return localStorage.getItem(KEY); } catch (e) { return null; } };
    const fmtN = v => (v == null ? '—' : Number(v).toLocaleString('en-US'));
    const pctTxt = v => (v == null ? '—' : (v * 100).toFixed(v < 0.1 ? 1 : 0) + '%');

    /* ---------- workspace: stats, needs, projects ---------- */
    const segClass = { d: 'd', c: 'c', r: 'c', w: 'b', f: 'b', '-': '' };
    async function loadWorkspace() {
      let ws;
      try { ws = await DC.client.workspace(); } catch (e) { return; }
      const st = ws.stats, stats = $$('.stats .stat', el);
      // the project the person is on (the green button): Home shows that project, not the workspace's sum
      const hereId = ws.projects.length ? await DC.currentProject.id() : null;
      const here = ws.projects.find(p => p.id === hereId) || null;
      let doc = null;
      if (here) { try { doc = await DC.currentProject.get(); } catch (e) { doc = null; } }
      S.here = here;
      if (stats.length >= 4 && here && doc) {
        const sol = doc.solution || {}, moves = doc.transitions || [], blocked = moves.filter(m => m.status === 'blocked').length;
        const keys = ['data', 'leakage', 'features', 'models', 'final'], done = keys.filter(k => ['completed', 'approved'].includes((doc.stages || {})[k] && (doc.stages || {})[k].status || (here.stages || {})[k])).length;
        const uses = Number(doc.holdout_uses || 0);
        stats[0].innerHTML = `<span class="v">${fmtN((sol.forbidden || []).length)}</span><span class="l">columns forbidden at the prediction moment in ${esc(here.name)}</span>`;
        stats[1].innerHTML = `<span class="v ${uses === 1 ? 'ok' : ''}">${uses ? fmtN(uses) + '×' : 'sealed'}</span><span class="l">${uses === 1 ? 'the holdout was opened exactly once · no reuse for selection' : uses ? 'the holdout was opened more than once: its number is no longer a fresh one' : 'the holdout is still sealed: nothing has read it yet'}</span>`;
        stats[2].innerHTML = `<span class="v">${fmtN(blocked)}</span><span class="l">unsafe moves stopped by the workflow graph in this project, each logged with its rule</span>`;
        stats[3].innerHTML = `<span class="v">${done} <small>/ ${keys.length}</small></span><span class="l">stages done · ${esc(here.current || 'all steps done')}</span>`;
      } else if (stats.length >= 4) {
        stats[0].innerHTML = `<span class="v">${fmtN(st.forbidden)}</span><span class="l">columns forbidden at the prediction moment, across ${fmtN(st.projects)} project${st.projects === 1 ? '' : 's'}</span>`;
        stats[1].innerHTML = `<span class="v ok">${fmtN(st.holdouts_once)} <small>/ ${fmtN(st.holdout_projects)}</small></span><span class="l">holdouts opened exactly once · no reuse for selection</span>`;
        stats[2].innerHTML = `<span class="v">${fmtN(st.blocked_moves)}</span><span class="l">unsafe moves stopped by the workflow graph, each logged with its rule</span>`;
        stats[3].innerHTML = `<span class="v">${fmtN(st.drafts)}</span><span class="l">open draft${st.drafts === 1 ? '' : 's'} on Home · ${fmtN(st.projects)} project${st.projects === 1 ? '' : 's'}</span>`;
      }
      const hereDraft = doc && (doc.draft || {}).id;  // the chat this project was built from
      S.hereDraft = hereDraft || null;
      const inbox = $('[data-f="home.inbox"]', el);
      if (inbox) {
        let off = false;
        try { off = localStorage.getItem('dclab-notes-off') === '1'; } catch (e) { off = false; }
        const mine = n => !here || (n.project ? n.project === here.id : n.draft ? n.draft === hereDraft : false);  // only what waits on the selected project
        const needs = ws.needs.filter(n => !(off && n.kind === 'disagreements') && mine(n));  // 29.10: a person who turned the notes off
        $('.pill', inbox).textContent = needs.length;
        const label = { question: 'Question', solution: 'Solution', disagreements: 'Which is right?' };
        $('.list', inbox).innerHTML = needs.length ? needs.map(n => `<a class="list-item" href="${n.draft ? '#home' : n.kind === 'disagreements' ? '#solution' : '#project'}" ${n.draft ? `data-resume="${esc(n.draft)}"` : n.kind === 'disagreements' ? `data-open-notes="${esc(n.project)}"` : `data-open-project="${esc(n.project)}"`}>
            <span class="avatar ai">AI</span>
            <div class="li-main"><span class="li-title">${esc(n.title)}</span><span class="li-sub">${esc(n.sub || '')}</span><span class="row"><span class="pill warn">${esc(label[n.kind] || 'Solution')}</span><span class="tag">${esc(n.tag)}</span></span></div></a>`).join('')
          : `<div class="empty">${here ? 'Nothing waits for you in ' + esc(here.name) + '.' : 'Nothing waits for you.'} Start from one sentence above.</div>`;
      }
      const count = $(`.ptab[data-ptab="projects"] .n`, el);
      if (count) count.textContent = ws.projects.length;  // the sample's count goes, also when there is no project yet
      if (ws.projects.length) {
        const pane = $('[data-pane="projects"]', el);
        S.projects = ws.projects;
        const shown = here && !S.allProjects ? ws.projects.filter(p => p.id === here.id) : ws.projects;
        let note = $('#home-projects-note', pane);
        if (!note) { note = document.createElement('div'); note.id = 'home-projects-note'; note.className = 'small muted'; note.setAttribute('data-style', 'padding:8px 16px'); $('.panel-head', pane).after(note); }
        note.hidden = !here || ws.projects.length < 2;
        note.innerHTML = S.allProjects ? `All ${fmtN(ws.projects.length)} projects. <button type="button" class="link-btn small" data-projects-all="0">Only ${esc((here || {}).name || 'this one')}</button>`
          : `Only ${esc((here || {}).name || '')}, the project you are on. <button type="button" class="link-btn small" data-projects-all="1">Show all ${fmtN(ws.projects.length)} projects</button>`;
        $('#home-projects tbody', el).innerHTML = shown.map(p => `
          <tr data-real="${esc(p.id)}">
            <td><div class="cell-main">${esc(p.name)}${p.data && p.data.synthetic ? ' <span class="pill warn">synthetic</span>' : ''}</div><div class="cell-sub">${esc(p.data ? p.data.filename + ' · ' + fmtN(p.data.rows) + ' rows' : 'no data yet')}${p.pack ? ' · ' + esc(p.pack) : ''}</div></td>
            <td><span class="wf-mini" aria-label="${esc(p.current || 'done')}">${p.state.split('').map(c => `<i class="${segClass[c] || ''}"></i>`).join('')}</span><div class="cell-sub">${esc(p.current || 'all steps done')}</div></td>
            <td><div class="score">${p.stages && p.stages.final === 'completed' ? 'holdout scored' : '—'}</div><div class="cell-sub">${p.holdout_uses ? 'holdout opened ' + p.holdout_uses + '×' : 'holdout sealed'}</div></td>
            <td class="num">—</td>
            <td><span class="pill ${p.running ? 'accent' : p.has_solution ? 'outline' : 'warn'}">${p.running ? 'running' : p.has_solution ? 'in progress' : 'solution draft'}</span><div class="cell-sub">${esc((p.updated || '').slice(0, 10))} · <button type="button" class="link-btn small" data-project-chat="${esc(p.id)}">Open its chat</button></div></td>
          </tr>`).join('');
        const tab = $(`.ptab[data-ptab="projects"] .n`, el);
        if (tab) tab.textContent = ws.projects.length;
        $('.panel-foot', pane).innerHTML = '<span class="dot ok"></span>done <span class="dot accent"></span>current <span class="dot warn"></span>waiting on a person <span class="muted">· Click a project to open its workflow.</span>';
      }
    }
    el.addEventListener('click', e => {
      const all = e.target.closest('[data-projects-all]');
      if (all) { S.allProjects = all.dataset.projectsAll === '1'; loadWorkspace(); }
    });
    $('#home-projects tbody', el).addEventListener('click', e => {
      const chat = e.target.closest('[data-project-chat]');
      if (chat) {  // the project's Home chat (its draft), shown again to read and go on with
        e.stopPropagation(); e.preventDefault();
        openProjectChat(chat.dataset.projectChat, chat);
        return;
      }
      const real = e.target.closest('tr[data-real]');
      if (real) { e.stopPropagation(); DC.state.projectId = real.dataset.real; location.hash = 'project'; return; }
    }, true);

    async function openProjectChat(id, button) {
      button.disabled = true;
      try {
        const project = await DC.client.getProject(id);
        const draft = (project.draft || {}).id;
        if (!draft) { toast('This project was not started from a Home chat, so it has none to open. Ask about it in the Intern.', { ok: false }); return; }
        if (await resume(draft)) { DC.selectPane(el, 'overview'); $('#home-chat', el).scrollIntoView({ behavior: 'smooth' }); }
        else toast('This project\'s chat could not be opened.', { ok: false });
      } catch (err) { toast(err.message, { ok: false }); }
      finally { button.disabled = false; }
    }

    /* ---------- domain packs (optional) ---------- */
    function packCard(p, on, detected) {
      return `<button type="button" class="pack-card" data-pack="${esc(p.key)}" aria-pressed="${on ? 'true' : 'false'}"><span class="pc-top"><span class="pack-ic">${icon(p.icon)}</span><span class="pill ${esc(p.cls)}">${esc(p.maturity)}</span></span><span class="pc-name">${esc(p.name)}</span><span class="pc-desc">${esc(p.desc)}</span>${detected ? '<span class="pc-foot"><span class="pill accent">detected</span></span>' : ''}</button>`;
    }
    async function loadPacks() {
      try { S.packs = await DC.client.listPacks(); } catch (e) { S.packs = []; }
      drawPacks();
    }
    function drawPacks() {
      const chosen = S.draft && S.draft.pack ? S.draft.pack : null;
      const key = S.pack || (chosen && chosen.source === 'user' ? chosen.key : null);
      $('#home-packs', el).innerHTML = S.packs.map(p => packCard(p, p.key === key, chosen && chosen.source !== 'user' && chosen.key === p.key)).join('');
      const name = key ? (S.packs.find(p => p.key === key) || {}).name : chosen ? `Auto · ${(S.packs.find(p => p.key === chosen.key) || {}).name || chosen.key}` : 'Auto-detect';
      $('#home-pack-btn', el).textContent = name + ' ▾';
    }
    $('#home-pack-btn', el).addEventListener('click', () => {
      const g = $('#home-packs', el), open = g.hidden;
      g.hidden = !open; $('#home-pack-btn', el).setAttribute('aria-expanded', String(open));
    });
    $('#home-packs', el).addEventListener('click', async e => {
      const card = e.target.closest('[data-pack]'); if (!card) return;
      const key = card.dataset.pack;
      S.pack = S.pack === key ? null : key;
      drawPacks();
      if (S.draft) {
        try { S.draft = await DC.client.choosePack(S.draft.id, { body: { key: S.pack || 'auto' } }); drawPacks(); }
        catch (err) { toast(err.message, { ok: false }); }
      }
    });

    /* ---------- the draft ---------- */
    function showDraftPanels(on) {
      $('#home-solution', el).hidden = !on;
      $('#home-chat', el).hidden = !on;
      $('#home-understand', el).hidden = !on;
      // dclab:closed-begin  (the understanding team: the full edition's)
      $('#home-team', el).hidden = !on;
      // dclab:closed-end
      if (!on) { $('#home-twin', el).hidden = true; $('#home-lanes', el).className = 'stack'; }
    }
    const CHATS = ['open', 'built'];  // a built project's chat stays open: the person reads it and goes on
    async function ensureDraft(opts = {}) {
      if (S.draft && (S.draft.status === 'open' || (S.draft.status === 'built' && !opts.data))) return S.draft;
      if (S.draft && S.draft.status === 'built' && opts.data) {  // a built project took its table when it was built: new data is a new chat
        toast('This chat\'s project is built, so new data starts a new chat with the same problem. To change the project\'s data, use its Solution & data page.', { ms: 6000 });
        S.draft = null; remember(null);
      }
      const problem = $('#home-task', el).value.trim();
      if (problem.length < 8) { toast('Describe the problem in one sentence first.', { ok: false }); $('#home-task', el).focus(); throw new Error('no problem'); }
      S.draft = await DC.client.createDraft({ body: { problem, pack: S.pack } }); S.fresh = false;
      remember(S.draft.id);
      openDraft(S.draft);
      return S.draft;
    }
    function openDraft(draft) {
      S.draft = draft; S.analysis = {}; S.assets = {};
      $('#home-thread', el).innerHTML = '';
      $('#home-task', el).value = draft.problem;
      $('#home-chat-sub', el).textContent = draft.project_id
        ? 'The chat this project was built from: read it and go on asking. It does not change the project: change the project on its own pages (Solution & data), or start a new project for new data.'
        : 'A few questions to understand the problem, and the data as it gets ready.';
      showDraftPanels(true);
      creating(false);
      DC.client.internStatus().then(i => { const m = $('#home-chat-mode', el); m.textContent = i.mode === 'llm' ? 'model · ' + i.model : 'standard questions'; m.hidden = false; }).catch(() => {});
      if (S.close) S.close();
      S.close = stream(DC.client.path.events(draft.id), onEvent);
      drawPacks();
      drawIntent(draft.intent);
      updateDataPill();
      drawUnderstanding(draft);
      loadTwin();
      // dclab:closed-begin  (the understanding team: the full edition's)
      loadTeam();
      // dclab:closed-end
    }
    // dclab:closed-begin  (the understanding team: the full edition's)
    // 28.8: the team's walk: its board and the pause it waits on, read without running; a step only when the person asks
    async function loadTeam() {
      const box = $('#home-team-board', el);
      if (!S.draft) return;
      try { drawTeam(await DC.client.teamBoard(S.draft.id, { quiet: true })); }
      catch (e) { DC.states.failed(box, e, { lead: 'Could not read the team: ', retry: loadTeam }); }
    }
    function drawTeam(v) {
      const box = $('#home-team-board', el);
      const b = (v && v.board) || {}, wait = v && v.waiting;
      if (!v || (!v.board && !wait)) { DC.states.empty(box, 'The team has not walked this draft yet.'); return; }
      const fields = Object.entries(b.problem || {}).filter(([, x]) => x).map(([k, x]) => `<span class="tag">${esc(k)}: ${esc(x)}</span>`).join(' ');
      const asks = !wait ? `<span class="muted small">${v.walk ? 'The walk ' + esc(v.walk.status || '') + '.' : 'Working.'}</span>`
        : wait.kind === 'question'
          ? `<div class="stack"><span>${esc(wait.text || '')}</span><span class="row"><label class="sr-only" for="home-team-answer">Your answer</label><input id="home-team-answer" type="text" placeholder="Your answer (or leave it empty to skip)"><button type="button" class="btn primary sm" id="home-team-answer-go">Answer</button></span></div>`
          : `<div class="stack" data-team-kind="${esc(wait.kind || '')}" data-team-gate="${esc(wait.gate || '')}"><span>${esc(wait.kind === 'confirmation' ? 'Confirm the map of the problem?' : 'Approve: ' + (wait.gate || ''))}</span>${teamSummary(wait.summary)}<div class="row"><button type="button" class="btn primary sm" data-team-approve="1">Yes</button><button type="button" class="btn sm" data-team-approve="0">No</button></div></div>`;
      const spent = Number(b.spent_eur || 0) > 0 ? ` · €${Number(b.spent_eur).toFixed(3)}` : '';
      box.innerHTML = `<div class="row"><span class="pill outline">${esc(b.state || 'starting')}</span><span class="muted small">${v.node ? esc(v.node) + ' · ' + esc(v.what || '') + ' · ' : ''}${v.handoffs} steps${spent}</span></div>
        <div class="row">${fields}</div>${(b.blockers || []).length ? `<div class="muted small">Blocked by: ${(b.blockers || []).map(esc).join(', ')}</div>` : ''}${asks}`;
      const go = $('#home-team-answer-go', box);
      if (go) go.onclick = () => teamStep({ answer: $('#home-team-answer', box).value });
      box.querySelectorAll('[data-team-approve]').forEach(btn => {
        const buildYes = wait && wait.kind === 'approval' && wait.gate === 'build' && btn.dataset.teamApprove === '1';
        btn.onclick = () => buildYes ? teamBuild() : teamStep({ approve: btn.dataset.teamApprove === '1' });  // 28.8: Build starts the stages too
      });
    }
    async function teamBuild() {
      const box = $('#home-team-board', el);
      DC.states.loading(box, { text: 'Building through the team…' });
      try { await DC.client.buildAndRun(S.draft.id, { body: { team: true } }); const d = await DC.client.readDraft(S.draft.id, { quiet: true }); S.draft = d; openDraft(d); }
      catch (e) { DC.states.failed(box, e, { lead: 'The team could not build: ', retry: loadTeam }); }
    }
    function teamSummary(sm) {  // what a confirmation or an approval is about: the critic's concerns, the map's parts, the build's state
      if (!sm || typeof sm !== 'object') return '';
      const line = (k, v) => `<li><span class="muted small">${esc(k)}:</span> ${Array.isArray(v) ? v.map(x => esc(typeof x === 'object' ? Object.values(x).join(' · ') : String(x))).join('; ') : esc(String(v))}</li>`;
      return `<ul class="small">${Object.entries(sm).filter(([, v]) => v !== null && v !== undefined && v !== '').map(([k, v]) => line(k, v)).join('')}</ul>`;
    }
    async function teamStep(body) {
      const box = $('#home-team-board', el);
      if (!S.draft) return;
      DC.states.loading(box, { text: 'The team is working…' });
      try {
        const out = await DC.client.teamWalk(S.draft.id, { body }); drawTeam(out.view);
        if (out.project_id || (out.view && out.view.board && out.view.board.state === 'built')) {  // 28.8: the walk built: the page follows
          const d = await DC.client.readDraft(S.draft.id, { quiet: true });
          if (d.project_id) { S.draft = d; openDraft(d); }
        }
      }
      catch (e) { DC.states.failed(box, e, { lead: 'The team could not go on: ', retry: loadTeam }); }
    }
    // dclab:closed-end
    /* The kind of help Home read (package H2): its label, the person's words behind it, and a Change control. Only a
       prediction shows the solution workflow and the build button. Labels mirror dclab_rnd/draft/intent.py LABELS. */
    const INTENTS = { prediction: 'A prediction (a model)', analysis: 'Numbers from my data', preparation: 'Cleaning or combining data',
      reporting: 'A report or dashboard', business: 'A business question', advice: 'Advice or an explanation' };
    function drawIntent(intent) {
      const box = $('#home-intent', el), sel = $('#home-intent-change', el);
      if (!intent) { box.hidden = true; $('#home-create', el).hidden = false; return; }  // a draft from before H2: a prediction
      if (S.draft) S.draft.intent = intent;
      const asked = intent.status === 'asked';
      $('#home-intent-label', el).textContent = asked ? 'not decided yet' : INTENTS[intent.track] || intent.track;
      $('#home-intent-why', el).textContent = asked ? '— answer below, or choose' : intent.source === 'user' ? '(you chose)'
        : intent.quote ? `— from “${intent.quote}”` : intent.source === 'default' ? '(the usual start; change it if not)' : '';
      sel.innerHTML = '<option value="">Change…</option>' + Object.entries(INTENTS).filter(([k]) => asked || k !== intent.track)
        .map(([k, v]) => `<option value="${esc(k)}">${esc(v)}</option>`).join('');
      box.hidden = false;
      const prediction = !asked && intent.track === 'prediction';
      $('#home-solution', el).hidden = !prediction;
      $('#home-create', el).hidden = !prediction;
    }
    $('#home-intent-change', el).addEventListener('change', async e => {
      const track = e.target.value; if (!track || !S.draft) return;
      e.target.disabled = true;
      try { S.draft = await DC.client.chooseIntent(S.draft.id, { body: { track } }); drawIntent(S.draft.intent); }
      catch (err) { toast(err.message, { ok: false }); }
      finally { e.target.disabled = false; e.target.value = ''; }
    });
    async function resume(id) {
      try { const d = await DC.client.readDraft(id, { quiet: true }); if (CHATS.includes(d.status)) { remember(d.id); openDraft(d); return true; } }
      catch (e) { if (e.status !== 404) return false; }  // a slow or failed read keeps the draft remembered; only a draft that is gone is forgotten
      remember(null); return false;
    }
    $('#home-start', el).addEventListener('click', async () => {
      if (S.draft && CHATS.includes(S.draft.status) && $('#home-task', el).value.trim() !== S.draft.problem) { S.draft = null; remember(null); }
      try { await ensureDraft(); $('#home-chat', el).scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch (e) { if (e.message !== 'no problem') toast(e.message, { ok: false }); }
    });
    el.addEventListener('click', async e => {
      const ex = e.target.closest('[data-example]');
      if (ex) { $('#home-task', el).value = ex.dataset.example; $('#home-task', el).focus(); }
      const mb = e.target.closest('[data-map-build]');
      if (mb && S.draft) {  // 25.13: the person confirms the map as drawn, then the build goes on as asked
        mb.disabled = true;
        try { await DC.client.editMap(S.draft.id, { body: { action: 'confirm' } }); await DC.client.message(S.draft.id, { body: { text: "Let's build the solution" } }); }
        catch (err) { mb.disabled = false; toast(err.message, { ok: false }); }
        return;
      }
      const rw = e.target.closest('[data-rows]'); if (rw) { showRows(rw.dataset.rows, Number(rw.dataset.offset || 0)); return; }
      const op = e.target.closest('[data-open-project]');
      if (op) { DC.state.projectId = op.dataset.openProject; if (op.dataset.openView) { location.hash = '#' + op.dataset.openView; return; } }
      const bringing = e.target.closest('[data-bring]');
      if (bringing && S.draft) {
        bringing.disabled = true;
        try { await DC.client.bringTable(S.draft.id, { body: { draft: bringing.dataset.bring } }); DC.toast('Bringing that table here: it is read with the cases.'); }
        catch (err) { bringing.disabled = false; DC.toast(err.message, { ok: false }); }
        return;
      }
      const rated = e.target.closest('[data-rate]');
      if (rated && S.draft) {
        rated.parentElement.querySelectorAll('[data-rate]').forEach(b => { b.disabled = true; });
        try { await DC.client.giveFeedback({ body: { on: 'answer', ref: rated.dataset.ref, kind: 'thumbs', rating: rated.dataset.rate, draft: S.draft.id } }); toast('Thanks: noted.'); }
        catch (err) { toast(err.message, { ok: false }); }
      }
      /* 23.6: the person's own version of a reply. It is kept beside the original and becomes the example's reply. */
      const editing = e.target.closest('[data-edit]');
      if (editing && S.draft) {
        const body = editing.closest('.msg-body');
        if (!body.querySelector('[data-edit-box]')) {
          const shown = body.querySelector('p, .q');
          body.insertAdjacentHTML('beforeend', `<div class="stack tight" data-edit-box><textarea rows="3" maxlength="4000" aria-label="Your version of this reply">${esc(shown ? shown.textContent : '')}</textarea>`
            + `<div class="row"><button type="button" class="btn sm primary" data-edit-save="${esc(editing.dataset.edit)}">Save my version</button><button type="button" class="btn sm" data-edit-cancel="1">Cancel</button></div></div>`);
        }
        return;
      }
      if (e.target.closest('[data-edit-cancel]')) { e.target.closest('[data-edit-box]').remove(); return; }
      const saving = e.target.closest('[data-edit-save]');
      if (saving && S.draft) {
        const box = saving.closest('[data-edit-box]');
        const edited = box.querySelector('textarea').value.trim();
        if (!edited) { toast('Write your version first.', { ok: false }); return; }
        saving.disabled = true;
        try {
          const r = await DC.client.giveFeedback({ body: { on: 'answer', ref: saving.dataset.editSave, kind: 'edited_output', edited, draft: S.draft.id } });
          if (!r || !r.recorded) throw new Error('Not saved: try again in a moment.');
          box.remove(); toast('Kept: your version is saved beside the original.');
        }
        catch (err) { saving.disabled = false; toast(err.message, { ok: false }); }
        return;
      }
      const res = e.target.closest('[data-resume]');
      if (res) { e.preventDefault(); if (await resume(res.dataset.resume)) { DC.selectPane(el, 'overview'); $('#home-chat', el).scrollIntoView({ behavior: 'smooth' }); } }
    });

    /* ---------- events from the draft ---------- */
    function onEvent(kind, data) {
      if (kind === 'chat') renderMessage(data);
      else if (kind === 'token') renderToken(data);
      else if (kind === 'pipeline') renderPipeline(data);
      else if (kind === 'analysis') { S.analysis[data.asset] = data.analysis; }
      else if (kind === 'workflow') drawWorkflow(data.workflow, data.rejected);
      else if (kind === 'status') onStatus(data);
      else if (kind === 'plan') { if (S.draft) { S.draft.plan = data.plan; S.draft.understanding = data.understanding; } drawUnderstanding(S.draft); }
      else if (kind === 'twin') onTwin(data);
      else if (kind === 'board') renderBoard(data);
      else if (kind === 'fit' && S.draft) {  // 30.13: a fit edit accepted: the person's answers changed, the panel shows them
        DC.client.readDraft(S.draft.id).then(d => { S.draft = d; drawUnderstanding(S.draft); }).catch(() => {});
      }
    }

    /* ---------- 25.5: what DCLab understood, editable ---------- */
    const FIELD_LABELS = { target: 'Outcome', prediction_moment: 'Moment of prediction', action: 'What is done with it', data_plan: 'The data', costs: 'Costs', entity: 'Entity', labels: 'Labels', budget: 'Review budget' };
    function drawUnderstanding(draft) {
      const box = $('#home-und-fields', el); if (!box || !draft) return;
      const plan = draft.plan || {}, und = draft.understanding || {};
      const fields = Object.keys(FIELD_LABELS).filter(f => f in plan || f in und || ['target', 'prediction_moment', 'action', 'data_plan'].includes(f));
      box.innerHTML = fields.map(f => {
        const p = plan[f] || {}, v = und[f];
        const status = p.status || (v ? 'stated' : 'unknown');
        const pill = status === 'stated' ? 'ok' : status === 'inferred' ? 'info' : 'warn';
        return `<div class="stack tight" data-und="${esc(f)}"><div class="spread"><b class="small">${esc(FIELD_LABELS[f])}</b><span class="pill ${pill}">${esc(status)}</span></div>`
          + `<div class="small">${v ? esc(v) : '<span class="muted">not known yet</span>'}</div>`
          + (p.quote ? `<div class="xs muted">“${esc(String(p.quote).slice(0, 160))}”</div>` : '')
          + `<div class="row"><button type="button" class="link-btn small" data-und-edit="${esc(f)}">Correct</button></div></div>`;
      }).join('');
      const next = $('#home-und-next', el);
      if (plan.next) { next.textContent = 'next: ' + (FIELD_LABELS[plan.next] || plan.next); next.hidden = false; } else next.hidden = true;
    }
    $('#home-und-fields', el).addEventListener('click', async e => {
      const ed = e.target.closest('[data-und-edit]');
      if (ed) {
        const card = ed.closest('[data-und]');
        if (!card.querySelector('[data-und-box]')) {
          const cur = (S.draft.understanding || {})[ed.dataset.undEdit] || '';
          card.insertAdjacentHTML('beforeend', `<div class="stack tight" data-und-box><textarea rows="2" maxlength="400" aria-label="${esc(FIELD_LABELS[ed.dataset.undEdit])}, in your words">${esc(cur)}</textarea>`
            + `<div class="row"><button type="button" class="btn sm primary" data-und-save="${esc(ed.dataset.undEdit)}">Save</button><button type="button" class="btn sm" data-und-cancel="1">Cancel</button></div></div>`);
        }
        return;
      }
      if (e.target.closest('[data-und-cancel]')) { e.target.closest('[data-und-box]').remove(); return; }
      const sv = e.target.closest('[data-und-save]');
      if (sv && S.draft) {
        const value = sv.closest('[data-und-box]').querySelector('textarea').value.trim();
        if (!value) { toast('Write it in your words first.', { ok: false }); return; }
        sv.disabled = true;
        try { S.draft = await DC.client.editUnderstanding(S.draft.id, { body: { field: sv.dataset.undSave, value } }); drawUnderstanding(S.draft); toast('Recorded in your words.'); }
        catch (err) { sv.disabled = false; toast(err.status === 409 ? 'DCLab is still answering; try again in a moment.' : err.message, { ok: false }); }
      }
    });

    /* ---------- 25.5: Claude beside DCLab ---------- */
    const STEP_LABELS = { 'home.understanding': 'What the problem is', 'home.next_question': 'What to ask next', 'home.pack': 'Domain pack', 'home.map': 'Parts of the problem',
                          'home.workflow': 'Solution workflow', 'home.ready': 'Ready to build' };
    const show = v => (v == null ? '—' : Array.isArray(v) ? (v.length ? v.join(', ') : 'none') : typeof v === 'object' ? JSON.stringify(v) : String(v));
    async function loadTwin() {
      if (!S.draft) return;
      let v;
      try { v = await DC.client.twinView(S.draft.id, { quiet: true }); } catch (e) { return; }
      const on = !!(v.settings && v.settings.on);
      $('#home-twin', el).hidden = !on;
      $('#home-to', el).hidden = !on;
      $('#home-lanes', el).className = on ? 'grid g-chat' : 'stack';
      if (!on) return;
      S.twin = v;
      drawTwinStatus(v.status);
      const spent = v.session ? v.session.spent_eur || 0 : 0;
      const cap = v.cap_eur != null ? v.cap_eur : v.settings.session_eur;
      $('#home-twin-spent', el).textContent = `€${spent.toFixed(2)} of €${Number(cap).toFixed(2)}`;
      // dclab:closed-begin  (the person's month of the twin: the full edition's)
      DC.client.myBudget({ quiet: true }).then(b => {
        const btn = $('#home-twin-budget', el);
        btn.title = b.on ? `Your month: €${Number(b.spent_eur).toFixed(2)} of €${Number(b.monthly_eur).toFixed(2)} used; €${Number(b.left_eur).toFixed(2)} left` : 'Add to this draft\'s Claude budget';
      }).catch(() => {});
      // dclab:closed-end
      const thread = $('#home-twin-thread', el);
      thread.innerHTML = (v.conversation || []).length ? v.conversation.map(m => m.role === 'claude'
        ? `<div class="msg"><span class="avatar b">C</span><div class="msg-body"><div class="who">Claude</div><p>${esc(m.text)}</p></div></div>`
        : `<div class="msg me"><div class="msg-body"><div class="who">You, to Claude only</div><p>${esc(m.text)}</p></div></div>`).join('')
        : '<div class="empty">Claude answers after DCLab\'s first turn.</div>';
      let n = 0;
      const rows = (v.steps || []).map(st => {
        const diffs = (st.differences || []).filter(d => d.kind !== 'same' && d.kind !== 'neither');
        n += diffs.length ? 1 : 0;
        const picked = st.truth && st.truth.source === 'person_pick';
        return `<div class="list-item"><div class="li-main"><span class="li-title">${esc(STEP_LABELS[st.step] || st.step)}${diffs.length ? '' : ' <span class="pill ok">same</span>'}</span>`
          + diffs.map(d => `<span class="li-sub"><b>${esc(d.field)}</b> · DCLab: ${esc(show(d.system))} · Claude: ${esc(show(d.ai_first))}</span>`).join('')
          + (diffs.length ? (picked ? `<span class="row"><span class="pill info">you chose ${st.verdict.winner === 'ai_first' ? 'Claude' : st.verdict.winner === 'system' ? 'DCLab' : st.verdict.winner}</span></span>`
            : `<span class="row">${diffs.some(d => d.system != null) ? `<button type="button" class="btn sm" data-pick="${esc(st.step)}" data-arm="system">DCLab is right</button>` : ''}`
              + `${diffs.some(d => d.ai_first != null) ? `<button type="button" class="btn sm" data-pick="${esc(st.step)}" data-arm="ai_first">Claude is right</button>` : ''}</span>`) : '')
          + '</div></div>';
      });
      $('#home-twin-diff', el).innerHTML = rows.length ? rows.join('') : '<div class="empty">No step answered by both yet.</div>';
      $('#home-twin-ndiff', el).textContent = n;
      const r = v.record;
      $('#home-twin-record', el).innerHTML = r ? [
        ['Problem', r.problem && r.problem.restatement], ['One prediction is about', r.problem && r.problem.unit], ['Moment', r.problem && r.problem.moment],
        ['Outcome', r.problem && r.problem.outcome], ['Kind of problem', r.problem_type && [r.problem_type.family, r.problem_type.task].filter(Boolean).join(' · ')],
        ['Why', r.problem_type && r.problem_type.why],
        ['Algorithms', ((r.algorithms || {}).ranked || []).slice().sort((a, b) => (a.rank || 99) - (b.rank || 99)).map(a => a.family).join(' → ')],
        ['Split and metric', r.evaluation && [r.evaluation.split, r.evaluation.metric].filter(Boolean).join(' · ')],
        ['Workflow', (r.solution_workflow || []).map(s => s.wf).filter(Boolean).join(' → ')],
        ['Questions it would ask', (r.questions || []).join(' · ')], ['Risks', (r.risks || []).join(' · ')]]
        .filter(([, val]) => val).map(([k, val]) => `<div><b>${esc(k)}</b>: ${esc(val)}</div>`).join('')
        + ((v.thinking || []).length ? `<details><summary class="small"><b>Its thinking</b> (summarized by the model, last turn)</summary>${v.thinking.map(t => `<p class="small muted" data-style="white-space:pre-wrap">${esc(t)}</p>`).join('')}</details>` : '')
        : '<div class="empty">No reading yet.</div>';
    }
    /* where Claude's lane stands, kept on the session: thinking, answered, capped (add budget), unavailable (no model), failed */
    function drawTwinStatus(st) {
      const box = $('#home-twin-status', el);
      if (!st || st.state === 'answered') { box.hidden = true; return; }
      box.hidden = false;
      box.className = 'callout small ' + (st.state === 'thinking' ? 'info' : st.state === 'capped' ? 'warn' : 'bad');
      $('span:last-child', box).textContent = st.state === 'thinking' ? 'Claude is thinking: it reads the conversation and the data (about half a minute)…' : (st.note || st.state);
    }
    function onTwin(d) {
      drawTwinStatus({ state: d.status, note: d.note });
      if (d.status !== 'thinking') loadTwin();
    }
    $('#home-twin-budget', el).addEventListener('click', async e => {
      if (!S.draft) return;
      e.target.disabled = true;
      try { const r = await DC.client.twinBudget(S.draft.id, { body: { add_eur: 1.5 } }); toast(`This draft's Claude budget is now €${Number(r.session_eur).toFixed(2)}.`); loadTwin(); }
      catch (err) { toast(err.message, { ok: false }); }
      finally { e.target.disabled = false; }
    });
    $('#home-twin', el).addEventListener('click', async e => {
      const pk = e.target.closest('[data-pick]');
      if (!pk || !S.draft) return;
      pk.disabled = true;
      try { await DC.client.twinPick(S.draft.id, { body: { step: pk.dataset.pick, arm: pk.dataset.arm } }); toast('Noted: kept as the truth of this step.'); loadTwin(); }
      catch (err) { pk.disabled = false; toast(err.message, { ok: false }); }
    });
    function onStatus(d) {
      if (d.intent) drawIntent(d.intent);
      if (d.pack) { if (S.draft) S.draft.pack = d.pack; drawPacks(); $('#home-sol-pack', el).textContent = (S.packs.find(p => p.key === d.pack.key) || {}).name || d.pack.key; }
      if ('thinking' in d) setThinking(d.thinking);
      if (d.note) addSystem(d.note);
      if (d.built) { S.draft.status = 'built'; }
    }
    function setThinking(on) {
      S.thinking = on;
      let t = $('#home-typing', el);
      if (on && !t) { $('#home-thread', el).insertAdjacentHTML('beforeend', `<div class="msg" id="home-typing"><span class="avatar ai">AI</span><div class="msg-body"><div class="who">DCLab</div><div class="typing muted small">Thinking</div></div></div>`); }
      if (!on && t) t.remove();
    }
    function addSystem(text) {
      $('#home-thread', el).insertAdjacentHTML('beforeend', `<div class="callout info small"><span class="ic">${icon('info')}</span><span>${esc(text)}</span></div>`);
    }
    function scrollThread() { const t = $('#home-thread', el); t.lastElementChild && t.lastElementChild.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }

    /* A reply being written: text grows in a bubble keyed by the message id; the chat event with that id replaces it,
       and a drop event removes it (the model ended its step with tool calls, or failed). Text goes in as text, never as HTML. */
    function renderToken(d) {
      const thread = $('#home-thread', el);
      let live = $(`[data-live="${d.id}"]`, thread);
      if (d.drop) { if (live) live.remove(); return; }
      setThinking(false);
      if (!live) {
        thread.insertAdjacentHTML('beforeend', `<div class="msg" data-live="${esc(d.id)}"><span class="avatar ai">AI</span><div class="msg-body"><div class="who">DCLab</div><p></p></div></div>`);
        live = $(`[data-live="${d.id}"]`, thread);
      }
      const p = $('p', live);
      p.textContent += d.delta || '';
      scrollThread();
    }

    function renderMessage(m) {
      setThinking(false);
      if (m.role === 'agent' && S.draft) {
        if (m.kind === 'build' && m.project_id) S.draft.project_id = m.project_id;
        creating(false);
      }
      const thread = $('#home-thread', el);
      const live = m.id && $(`[data-live="${m.id}"]`, thread);
      if (live) live.remove();
      if (m.role === 'user') {
        $$('.ask-card:not([data-done])', thread).forEach(card => { card.dataset.done = '1'; $$('.ask-option', card).forEach(b => { b.disabled = true; if (b.dataset.answer === m.text) b.setAttribute('aria-pressed', 'true'); }); });
        thread.insertAdjacentHTML('beforeend', `<div class="msg user"><span class="avatar">SM</span><div class="msg-body"><div class="who">You <span>${esc((m.at || '').slice(11, 16))}</span></div><div class="bubble">${esc(m.text)}</div></div></div>`);
        return scrollThread();
      }
      let body = '';
      if (m.kind === 'question') {
        const q = m.question || {};
        body = `<div class="ask-card"><div class="q">${esc(m.text)}</div>${q.why ? `<div class="small muted">${esc(q.why)}</div>` : ''}${(q.options || []).length ? `<div class="ask-options">${q.options.map(o => `<button type="button" class="ask-option" data-answer="${esc(o)}">${esc(o)}</button>`).join('')}</div>` : '<div class="small muted">Answer below in your own words.</div>'}</div>`;
      } else if (m.kind === 'build') {  // 25.13: the project built from "build it", the agent carrying out its plan
        body = `<p data-style="white-space:pre-line">${esc(m.text)}</p><div class="row">${m.session_id ? `<button type="button" class="btn primary sm" data-follow="${esc(m.session_id)}" data-project="${esc(m.project_id || '')}">Follow the agent</button>` : ''}${m.project_id ? `<button type="button" class="btn sm" data-open-project="${esc(m.project_id)}">Open the project</button>` : ''}</div>`;
      } else if (m.kind === 'analysis') {
        body = `<p>${esc(m.text)}</p>${analysisCard(S.analysis[m.asset], m.asset)}`;
      } else if (m.kind === 'summary') {
        const c = m.card;  // 20.2: the problem as DCLab understood it, to confirm before the build
        const fam = c ? (c.family ? c.family.replace(/_/g, ' ') : (c.in_play || []).map(f => f.replace(/_/g, ' ')).join(' or ') || 'not settled') : '';
        const card = c ? `<div class="inset stack tight small"><div class="eyebrow">The problem, as understood</div><dl class="kv">`
          + `<dt>Kind</dt><dd>${esc(fam)}${c.family_how === 'stated' ? ' <span class="muted">(you said)</span>' : ''}</dd>`
          + `<dt>Labels</dt><dd>${esc({ column: 'a column marks the cases', file: 'a file lists the known cases', none: 'nothing is labelled' }[c.labels] || 'not said')}</dd>`
          + `<dt>Outcome</dt><dd>${esc(c.outcome || 'not named')}</dd><dt>Moment</dt><dd>${esc(c.moment || 'not said')}</dd></dl>`
          + (c.ready ? '' : `<div class="callout warn small">${esc((c.blocking || [])[0] || 'Something is still open.')}</div>`) + '</div>' : '';
        body = `<p>${esc(m.text)}</p>${card}<div class="stack tight" data-map-box="1"></div><div class="row"><button type="button" class="btn primary sm" data-build="1">Build this chat as a project</button><button type="button" class="btn sm" data-open-bring="1">Add data first</button></div>`;
      } else if (m.kind === 'synthetic_way') {
        const offer = esc(JSON.stringify(m.offer || {}));
        body = `<p>${esc(m.text)}</p><div class="row"><button type="button" class="btn primary sm" data-syn-ai='${offer}'>1 · Build it with AI</button><button type="button" class="btn sm" data-syn-describe="1">2 · I'll describe it, then approve</button></div>`;
      } else if (m.kind === 'synthetic_spec' && m.proposal) {
        const pr = m.proposal, cols = (pr.columns || []).map(c => `<tr><td>${esc(c.name)}</td><td class="mono">${esc(c.type)}${c.categories ? ': ' + esc(c.categories.join(', ')) : ''}</td><td class="num">${c.missing ? Math.round(c.missing * 100) + '% missing' : ''}</td></tr>`).join('');
        const t = pr.target ? `<p class="small"><b>Outcome:</b> ${esc(pr.target.name)} (${esc(pr.target.type)}${pr.target.positive_rate != null ? ', about ' + Math.round(pr.target.positive_rate * 100) + '% positive' : ''}${(pr.target.classes || []).length ? ': ' + esc(pr.target.classes.join(', ')) : ''})</p>` : '<p class="small muted">No outcome column in this design.</p>';
        body = `<p>${esc(m.text)}</p><div class="inset stack tight"><div class="eyebrow">${esc(pr.name)} · ${Number(pr.rows).toLocaleString('en-US')} rows · synthetic</div>${t}`
          + `<div class="table-wrap"><table class="data"><thead><tr><th>Column</th><th>Kind</th><th></th></tr></thead><tbody>${cols}</tbody></table></div>`
          + (pr.note ? `<div class="callout warn small"><span>${esc(pr.note)}</span></div>` : '') + '</div>'
          + `<div class="row"><button type="button" class="btn primary sm" data-syn-approve="${esc(pr.id)}">Approve and create the data</button><button type="button" class="btn sm" data-syn-change="${esc(pr.id)}">Change it</button><button type="button" class="btn sm ghost" data-syn-drop="${esc(pr.id)}">Not now</button></div>`;
      } else if (m.kind === 'improvement_report' && m.report) {
        const r = m.report, cur = r.current || {}, sign = v => (v == null ? '—' : (v > 0 ? '+' : '') + Number(v).toFixed(4));
        const rows = (r.candidates || []).map(c => `<tr><td><b>${esc(c.label)}</b><div class="cell-sub">${esc(c.aspect)}</div></td><td class="num mono">${Number(c.mean).toFixed(4)}</td><td class="num mono">${sign(c.gain)}</td>`
          + `<td class="num mono">${c.folds_better == null ? '—' : c.folds_better + ' / ' + r.folds}</td><td>${c.clears ? '<span class="pill ok">clears the margin</span>' : '<span class="pill outline">within the noise</span>'}</td>`
          + `<td>${c.id === r.adopted ? '<span class="pill ok">adopted</span>' : c.clears && c.adoptable && !r.adopted ? `<button type="button" class="btn sm primary" data-improve-adopt="${esc(c.id)}" data-improve-project="${esc(r.project)}">Adopt</button>` : ''}</td></tr>`).join('');
        const aspects = (r.aspects || []).map(a => `<li><b>${esc(a.aspect)}</b>: ${a.tried} tried, ${a.clear} cleared the margin${a.best_gain == null ? '' : ', best ' + sign(a.best_gain)}</li>`).join('');
        body = `<p>${esc(m.text)}</p><div class="inset stack tight"><div class="eyebrow">Improvement run · ${esc(r.metric)}</div>`
          + `<p class="small">Current model: <b>${esc(cur.family)}</b> on the <code>${esc(cur.recipe)}</code> features, ${Number(cur.mean).toFixed(4)} ± ${Number(cur.std).toFixed(4)} on ${r.folds} training folds.</p>`
          + `<ul class="plain">${aspects}</ul>`
          + `<div class="table-wrap"><table class="data compact"><thead><tr><th>Way</th><th class="num">Score on the folds</th><th class="num">Gain</th><th class="num">Folds better</th><th>Verdict</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`
          + ((r.skipped || []).length ? `<div class="small muted">Not run: ${esc(r.skipped.join('; '))}</div>` : '') + `<div class="small muted">${esc(r.note || '')}</div></div>`;
      } else if (m.kind === 'project_answer') {
        body = `${mdLite(m.text)}${(m.lines || []).length ? `<ul class="plain">${m.lines.map(l => `<li>${esc(l)}</li>`).join('')}</ul>` : ''}`
          + ((m.proof || []).length ? `<div class="row small"><span class="pill proof">cites</span>${DC.chips(m.proof)}</div>` : '')
          + ((m.team || []).length ? `<div class="row xs muted team-line">${m.team.map(a => `<span class="pill outline">${esc(a)}</span>`).join('')}</div>` : '')
          + chipsRow(m.chips);
      } else if (m.kind === 'project_card' && m.card) {
        body = projectCard(m.card);
      } else if (m.kind === 'task_board') {
        body = `<div class="inset stack tight" data-board="${esc(m.id)}">${boardHtml(m.tasks)}</div>`;
      } else if (m.kind === 'notebook_cell' && m.cell) {
        body = notebookCellCard(m.cell, m.project);
      } else {
        body = `<p>${esc(m.text)}</p>${(m.actions || []).length ? `<div class="row">${m.actions.includes('upload') ? '<button type="button" class="btn sm" data-open-bring="upload">Upload a sample</button>' : ''}${m.actions.includes('connect') ? '<button type="button" class="btn sm" data-open-bring="kaggle">Connect a source</button>' : ''}${m.actions.includes('simulate') ? '<button type="button" class="btn sm" data-open-bring="synthetic">Simulate data</button>' : ''}${m.actions.includes('confirm_map') ? '<button type="button" class="btn primary sm" data-map-build="1">Confirm the map and build</button>' : ''}</div>${m.actions.includes('show_map') ? '<div class="stack tight" data-map-box="1"></div>' : ''}` : ''}`;
      }
      // 19.9: was this answer helpful? It joins the session record, never changes the conversation
      const rate = m.id ? `<div class="row xs muted" data-style="gap:6px;margin-top:4px"><button type="button" class="link-btn" data-rate="helpful" data-ref="${esc(m.id)}" aria-label="Helpful">👍</button><button type="button" class="link-btn" data-rate="unhelpful" data-ref="${esc(m.id)}" aria-label="Not helpful">👎</button><button type="button" class="link-btn" data-edit="${esc(m.id)}" aria-label="Write my own version of this reply">✏</button></div>` : '';
      thread.insertAdjacentHTML('beforeend', `<div class="msg"><span class="avatar ai">AI</span><div class="msg-body"><div class="who">DCLab <span>${esc((m.at || '').slice(11, 16))}</span></div>${body}${rate}</div></div>`);
      DC.hydrate(thread);
      const mapBox = (m.kind === 'summary' || (m.actions || []).includes('show_map')) && thread.querySelector('.msg:last-child [data-map-box]');
      if (mapBox && S.draft) loadMap(mapBox);
      scrollThread();
    }

    /* ---------- the Home orchestrator's team on a built project: tables, the task board, a notebook cell, suggestions ---------- */
    const numText = (v, digits) => (typeof v === 'number' ? (Number.isInteger(v) && digits == null ? v.toLocaleString('en-US') : v.toFixed(digits != null ? digits : Math.abs(v) >= 100 ? 1 : 4)) : v == null ? '—' : String(v));
    /* A reply's light Markdown, escaped first: paragraphs, "- " and "1. " lists, **bold** and `code`; nothing else becomes HTML. */
    function mdLite(text) {
      const inline = t => esc(t).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`([^`]+)`/g, '<code>$1</code>');
      const out = []; let list = null;
      const close = () => { if (list) { out.push(`<${list.tag} class="plain-list">${list.items.join('')}</${list.tag}>`); list = null; } };
      String(text || '').split('\n').forEach(line => {
        const bullet = line.match(/^\s*[-*•]\s+(.*)$/), numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
        const item = bullet || numbered;
        if (item) {
          const tag = numbered ? 'ol' : 'ul';
          if (!list || list.tag !== tag) { close(); list = { tag, items: [] }; }
          list.items.push(`<li>${inline(item[1])}</li>`);
        } else { close(); if (line.trim()) out.push(`<p>${inline(line)}</p>`); }
      });
      close();
      return out.join('') || '<p></p>';
    }
    const PAGES = { models: 'Models', audit: 'Audit', reliability: 'Reliability', evaluation: 'Evaluation', project: 'Project', notebook: 'Notebook' };
    function chipsRow(chips) {
      return (chips || []).length ? `<div class="row chips-row">${chips.map(c => `<button type="button" class="btn sm ghost" data-say="${esc(c)}">${esc(c)}</button>`).join('')}</div>` : '';
    }
    function projectCard(c) {
      const kv = list => `<dl class="kv small">${list.map(f => `<dt>${esc(f[0])}</dt><dd>${esc(numText(f[1]))}</dd>`).join('')}</dl>`;
      const facts = ((c.facts || []).length ? kv(c.facts) : '') + ((c.more || []).length ? `<details class="small"><summary>More measures (${c.more.length})</summary>${kv(c.more)}</details>` : '');
      const cols = c.columns || [];
      const table = cols.length && (c.rows || []).length ? `<div class="table-wrap"><table class="data compact"><thead><tr>${cols.map(k => `<th${k.num ? ' class="num"' : ''}>${esc(k.label)}</th>`).join('')}</tr></thead><tbody>`
        + c.rows.map(r => `<tr${c.mark && r[c.mark] ? ' class="row-chosen"' : ''}>${cols.map((k, i) => `<td${k.num ? ' class="num mono"' : ''}>${i === 0 && c.mark && r[c.mark] ? `<b>${esc(numText(r[k.key], k.digits))}</b> <span class="pill ok">chosen</span>` : esc(numText(r[k.key], k.digits))}</td>`).join('')}</tr>`).join('')
        + '</tbody></table></div>' : '';
      let matrix = '';
      if (c.matrix && (c.matrix.matrix || []).length) {
        const L = c.matrix.labels || [];
        matrix = `<div class="table-wrap"><table class="data compact matrix-mini"><thead><tr><th>actual \\ predicted</th>${L.map(l => `<th class="num">${esc(l)}</th>`).join('')}</tr></thead><tbody>`
          + c.matrix.matrix.map((row, i) => `<tr><th>${esc(L[i] || '')}</th>${row.map(v => `<td class="num mono">${esc(numText(v))}</td>`).join('')}</tr>`).join('') + '</tbody></table></div>';
      }
      const lines = (c.lines || []).length ? `<ul class="plain">${c.lines.map(l => `<li>${esc(l)}</li>`).join('')}</ul>` : '';
      const proof = (c.proof || []).length ? `<div class="row small"><span class="pill proof">cites</span>${DC.chips(c.proof)}</div>` : '';
      const notes = (c.notes || []).map(n => `<div class="small muted">${esc(n)}</div>`).join('') + (c.note ? `<div class="small muted">${esc(c.note)}</div>` : '');
      const open = c.page && c.project ? `<button type="button" class="btn sm" data-open-project="${esc(c.project)}" data-open-view="${esc(c.page)}">Open ${esc(PAGES[c.page] || c.page)}</button>` : '';
      const href = c.download && c.project && DC.client.href[c.download.route];  // a route the API client knows, never a path from the card
      const download = href ? `<a class="btn sm" href="${esc(href(c.project))}" download>${esc(c.download.label || 'Download')}</a>` : '';
      return `<div class="inset stack tight project-card"><div class="row between"><div><div class="eyebrow">${esc(c.agent || '')}</div><b>${esc(c.title || '')}</b>${c.subtitle ? `<div class="small muted">${esc(c.subtitle)}</div>` : ''}</div>${open}${download}</div>`
        + facts + table + matrix + lines + proof + notes + (c.offer ? `<div class="small">${esc(c.offer)}</div>` : '') + '</div>';
    }
    function boardHtml(tasks) {
      const pill = st => st === 'done' ? '<span class="pill ok">done</span>' : st === 'failed' ? '<span class="pill bad">failed</span>' : '<span class="pill info"><span class="dot live"></span> running</span>';
      return '<div class="eyebrow">The team on this request</div>' + (tasks || []).map(t => `<div class="row between board-task"><div><b>${esc(t.agent)}</b> · ${esc(t.title)}${t.detail ? `<div class="small muted">${esc(t.detail)}</div>` : ''}</div>${pill(t.status)}</div>`).join('');
    }
    function renderBoard(d) {
      const box = $(`[data-board="${d.id}"]`, el);
      if (box) box.innerHTML = boardHtml(d.tasks);
    }
    function notebookCellCard(c, project) {
      const ok = c.status === 'proposed' || c.status === 'accepted';
      return `<div class="inset stack tight" data-nb-cell="${esc(c.id)}"><div><div class="eyebrow">Notebook coder</div><b>A cell for: ${esc(c.request || '')}</b></div>`
        + (c.explanation ? `<p class="small">${esc(c.explanation)}</p>` : '')
        + (c.code ? `<pre class="code">${esc(c.code)}</pre>` : '')
        + ((c.problems || []).length ? `<div class="callout warn small">${esc(c.problems.join('; '))}</div>` : '')
        + (ok ? `<div class="row"><button type="button" class="btn primary sm" data-nb-run="${esc(c.id)}" data-nb-project="${esc(project || '')}" data-nb-status="${esc(c.status)}">Run it (training rows only)</button><button type="button" class="btn sm" data-open-project="${esc(project || '')}" data-open-view="notebook">Open in the notebook</button></div>` : '')
        + '<div data-nb-out></div></div>';
    }
    function cellOutput(r) {
      return (r.outputs || []).map(o => o.type === 'image'
        ? `<div class="cell-out"><img class="cell-img" alt="Output of the cell" src="data:image/png;base64,${esc(o.png)}"></div>`
        : o.type === 'chart' ? `<div class="cell-out cell-chart">${DC.evaluation.drawChart(o.spec)}</div>`
        : o.type === 'error' ? `<div class="cell-out cell-err">${esc(`${o.ename}: ${o.evalue}`)}</div>`
        : `<div class="cell-out">${esc(o.text)}</div>`).join('') || '<div class="cell-out faint">ran · no output</div>';
    }
    async function runChatCell(btn) {
      const card = btn.closest('[data-nb-cell]'), out = $('[data-nb-out]', card), pid = btn.dataset.nbProject, id = btn.dataset.nbRun;
      btn.disabled = true; btn.textContent = 'Running…';
      try {
        if (btn.dataset.nbStatus !== 'accepted') { await DC.client.answerNotebookCell(pid, id, { body: { accepted: true } }); btn.dataset.nbStatus = 'accepted'; }
        const r = await DC.client.runNotebookCell(pid, { body: { cell: id } });
        out.innerHTML = cellOutput(r);
        btn.textContent = 'Run again';
      } catch (err) { out.innerHTML = `<div class="callout warn small">${esc(err.message)}</div>`; btn.textContent = 'Run it (training rows only)'; }
      finally { btn.disabled = false; scrollThread(); }
    }

    /* ---------- the map of the problem (24.2): one problem, or parts that depend on each other; the person edits and confirms it ---------- */
    const FAMILIES = ['classification', 'regression', 'rare_event', 'anomaly', 'forecasting', 'time_to_event', 'ranking', 'segmentation', 'uplift', 'entity_tables', 'streams', 'text_or_images'];
    const famText = f => String(f || '').replace(/_/g, ' ');
    const LABELS_TEXT = { column: 'a column marks the outcome', file: 'a file lists the known cases', none: 'nothing is labelled', unknown: 'labels not said' };
    async function loadMap(box) {
      if (!S.draft) return;
      box.dataset.draft = S.draft.id;
      box.innerHTML = '<div class="muted small">Reading the problem as parts…</div>';
      let v;
      try { v = await DC.client.draftMap(S.draft.id, { quiet: true }); } catch (e) { DC.states.failed(box, e, { lead: 'Could not draw the map: ', retry: () => loadMap(box) }); return; }
      renderMap(box, v);
    }
    function renderMap(box, v) {
      const m = v.map, subs = m.sub_problems, titleOf = id => { const sp = subs.find(x => x.id === id); return sp ? sp.title : id; };
      const NEEDS = { labels: 'known outcomes', binary_target: 'an outcome of two values', many_classes: 'an outcome of several classes', number_target: 'a numeric outcome', text_column: 'a long-text column',
        place_columns: 'two columns of place', time_column: 'a time column', entity_column: 'an entity column', several_tables: 'several tables', treatment_column: 'a column that says who was treated',
        duration_column: 'a duration column', budget_step: 'a review budget', not_budget_step: 'not a review list' };
      const routeBox = sp => {
        const r = (v.routes || {})[sp.id];
        if (!r) return '';
        const others = (r.candidates || []).filter(c => c.approach !== r.chosen);
        const blocked = c => c.blocking.length ? `blocked: it needs ${c.blocking.map(b => NEEDS[b] || b).join(', ')}` : (c.unknown.length ? `possible: still to settle ${c.unknown.map(b => NEEDS[b] || b).join(', ')}` : 'possible');
        const mine = (r.candidates || []).find(c => c.approach === r.chosen);
        const warn = mine && mine.blocking.length ? `<div class="callout warn small">The data or the map refutes this approach: it needs ${mine.blocking.map(b => esc(NEEDS[b] || b)).join(', ')}.</div>` : '';
        return `<div class="small"><b>Approach:</b> ${r.chosen ? esc(r.title) : '<span class="muted">none fits yet</span>'} <span class="muted xs">${esc(r.reason || '')}</span></div>${warn}`
          + (others.length ? `<details class="xs"><summary>${others.length} other approach${others.length === 1 ? '' : 'es'}</summary><ul class="plan-list">${others.map(c => `<li class="${c.blocking.length ? 'blocked' : 'done'}"><span class="pi">${c.blocking.length ? '!' : '·'}</span><span>${esc(c.title)} <span class="muted">${esc(blocked(c))}</span></span></li>`).join('')}</ul></details>` : '');
      };
      const addForm = `<details class="small"><summary>Add a part</summary><div class="row" data-style="gap:6px;flex-wrap:wrap;margin-top:6px"><input type="text" data-map-title placeholder="What it finds, in a few words" aria-label="Title of the new part"><select data-map-family aria-label="Family of the new part">${FAMILIES.map(f => `<option value="${f}">${famText(f)}</option>`).join('')}</select><button type="button" class="btn sm" data-map-act="add">Add</button></div></details>`;
      if (m.size === 'single') {
        box.innerHTML = `<div class="inset stack tight small" data-map-draft="${esc(box.dataset.draft || '')}"><div class="eyebrow">The map of the problem</div><div>One problem, no parts to combine: ${esc(famText(m.sub_problems[0].family))}. <span class="muted">If it is really several that depend on each other, add the part you see.</span></div>${routeBox(m.sub_problems[0])}
          ${(v.review.questions || []).length ? `<ul class="plan-list">${v.review.questions.map(q => `<li class="${q.blocking ? 'blocked' : 'done'}"><span class="pi">${q.blocking ? '!' : '?'}</span><span>${esc(q.question)}</span></li>`).join('')}</ul>` : ''}
          ${addForm}<div class="row" data-style="gap:6px"><button type="button" class="btn sm" data-map-act="redraw">Draw it again</button></div></div>`;
        loadSolution(box);
        return;
      }
      const partCard = p => {
        const sp = subs.filter(x => x.part === p.id);
        const reasons = p.why ? esc(p.why) : p.quote ? `You said: “${esc(p.quote.length > 160 ? p.quote.slice(0, 160) + '…' : p.quote)}”` : '';
        const others = m.parts.filter(o => o.id !== p.id);
        return `<div class="inset stack tight" data-map-part="${esc(p.id)}">
          <div class="spread"><b>${esc(p.title)}</b><span class="row" data-style="gap:6px">${sp.map(x => `<span class="pill accent">${esc(famText(x.family))}</span>`).join('')}</span></div>
          ${p.question ? `<div class="small">${esc(p.question)}</div>` : ''}
          <dl class="kv small"><dt>One row is</dt><dd>${esc(p.unit || 'not said')}</dd><dt>Needed</dt><dd>${esc(p.moment || 'not said')}</dd>
          ${sp.map(x => `<dt>Labels</dt><dd>${esc(LABELS_TEXT[x.labels] || x.labels)}</dd>`).join('')}
          ${sp.some(x => (x.reads || []).length) ? `<dt>Uses</dt><dd>${esc(sp.flatMap(x => (x.reads || []).map(r => `${r.what === 'score' ? 'the score of' : 'the outcome of'} ${titleOf(r.sub_problem)}${r.via ? ` (by ${r.via})` : ''}`)).join('; '))}</dd>` : ''}</dl>
          ${sp.map(routeBox).join('')}
          <div class="xs muted">${reasons}</div>
          <div class="row" data-style="gap:6px;flex-wrap:wrap"><button type="button" class="btn sm" data-map-act="remove" data-part="${esc(p.id)}">Remove</button>
          ${others.length ? `<label class="xs muted">Merge into <select data-map-merge="${esc(p.id)}" aria-label="Merge ${esc(p.title)} into another part"><option value="">…</option>${others.map(o => `<option value="${esc(o.id)}">${esc(o.title)}</option>`).join('')}</select></label>` : ''}</div>
        </div>`;
      };
      const steps = (m.composition.steps || []).map(c => `<li class="done"><span class="pi">→</span><span>${esc(c.kind)} of ${esc(c.inputs.map(titleOf).join(', '))} <span class="muted">→ ${esc(c.output)}${c.joins ? ` · joined by ${esc(c.joins)}` : ''}</span></span></li>`).join('');
      const qs = (v.review.questions || []).map(q => `<li class="${q.blocking ? 'blocked' : 'done'}"><span class="pi">${q.blocking ? '!' : '?'}</span><span>${esc(q.question)} <span class="muted xs">${esc(q.why || '')}</span></span></li>`).join('');
      const issues = (v.issues || []).length ? `<div class="callout warn small">${v.issues.map(esc).join('<br>')}</div>` : '';
      box.innerHTML = `<div class="inset stack tight" data-map-draft="${esc(box.dataset.draft || '')}">
        <div class="spread"><div class="eyebrow">The map of the problem</div><span class="pill ${v.confirmed ? 'ok' : 'warn'}">${v.confirmed ? 'confirmed' : 'to confirm'}</span></div>
        <div class="small">This is ${m.parts.length} parts that depend on each other, not one problem. Check each part, change what is wrong, then confirm.${v.stale ? ' <b>You said more since it was drawn.</b>' : ''}</div>
        ${issues}
        <div class="stack tight">${m.parts.map(partCard).join('')}</div>
        ${steps ? `<div class="eyebrow">How the results combine</div><ul class="plan-list">${steps}</ul>${m.composition.budget ? `<div class="small muted">Limit you gave: ${esc(m.composition.budget)}</div>` : ''}` : ''}
        ${qs ? `<div class="eyebrow">Still open</div><ul class="plan-list">${qs}</ul><div class="xs muted">Answer these in the chat below, in your own words.</div>` : ''}
        ${addForm}
        <div class="row" data-style="gap:6px;flex-wrap:wrap"><button type="button" class="btn primary sm" data-map-act="confirm"${v.confirmed ? ' disabled' : ''}>${v.confirmed ? 'Confirmed' : 'Confirm the map'}</button><button type="button" class="btn sm" data-map-act="redraw">Draw it again</button></div>
        <div class="xs muted">${v.source === 'advisor' ? 'Drawn by the model from your words' : v.source === 'person' ? 'Changed by you' : 'Drawn by rules from your words'}; every change is checked, and refused with its reason when it would not hold together.</div>
      </div>`;
      loadSolution(box);
    }
    /* the solution map (24.4): the problem map answered, beside it; one holdout for the whole */
    const SPLIT = { stratified: 'a random split that keeps the classes in proportion', time: 'by time: the latest period is the holdout', group: 'by entity: no entity on both sides', time_group: 'by time and by entity: the latest period, no entity on both sides' };
    async function loadSolution(box) {
      const id = box.dataset.draft;
      if (!id) return;
      let slot = $('[data-solution-box]', box);
      if (!slot) { box.insertAdjacentHTML('beforeend', '<div class="inset stack tight small" data-solution-box="1"><div class="muted">Drawing the solution…</div></div>'); slot = $('[data-solution-box]', box); }
      let v;
      try { v = await DC.client.draftSolutionMap(id, { quiet: true }); } catch (e) { DC.states.failed(slot, e, { lead: 'Could not draw the solution map: ', retry: () => loadSolution(box) }); return; }
      const h = v.holdout, split = h.split;
      const keys = split.keys.length ? ` on ${split.keys.map(esc).join(', ')}` : '';
      const blocks = v.blocks.map(b => `<div class="inset stack tight" data-block="${esc(b.id)}">
          <div class="spread"><b>${esc(b.title)}</b><span class="pill ${b.approach ? 'accent' : 'warn'}">${esc(b.approach_title || 'no approach yet')}</span></div>
          <div class="xs muted">Answers: ${esc(b.part_title)}${b.track ? ` · measured in ${esc(b.track)}` : ''}</div>
          <div class="xs">${b.final === 'deferred' ? 'Hands its predictions to the composition: it never scores the holdout itself.' : 'Scores its own holdout, once, after your approval.'}</div>
          <details class="xs"><summary>${b.workflow.length} steps of its workflow</summary><ol class="plan-list">${b.workflow.map(w => `<li class="${w.gate ? 'blocked' : 'done'}"><span class="pi">${w.gate ? '✋' : '·'}</span><span><b>${esc(w.name)}</b>${w.deferred ? ' <span class="pill outline">run by the composition</span>' : ''} <span class="muted">${esc(w.purpose)}</span></span></li>`).join('')}</ol></details></div>`).join('');
      const steps = v.steps.map(c => `<li class="${c.reads_holdout ? 'blocked' : 'done'}"><span class="pi">${c.reads_holdout ? '◆' : '→'}</span><span>${esc(c.kind)} of ${esc((c.inputs || []).join(', '))} <span class="muted">→ ${esc(c.output)}${c.joins ? ` · joined by ${esc(c.joins)}` : ''}${c.reads_holdout ? ' · the only reading of the holdout' : ''}</span></span></li>`).join('');
      slot.innerHTML = `<div class="eyebrow">The solution, beside the problem</div>
        <div class="small">One holdout for the whole. Split: ${esc(SPLIT[split.strategy] || split.strategy)}${keys}${split.moment ? `, following ${esc(split.moment)}` : ''}. ${h.read_by === 'composition' ? 'It is read once, by the last step, and nothing is chosen on it: weights, thresholds and the budget are chosen on validation first.' : 'It is read once, after your approval.'}${(v.issues || []).length ? ' <b>Not while the problems below stand.</b>' : ''}</div>
        ${(v.issues || []).length ? `<div class="callout warn small">${v.issues.map(esc).join('<br>')}</div>` : ''}
        ${(v.blockers || []).length ? `<div class="callout warn small">${v.blockers.map(esc).join('<br>')}</div>` : ''}
        ${(v.notes || []).length ? `<div class="callout info small">${v.notes.map(esc).join('<br>')}</div>` : ''}
        <div class="stack tight">${blocks}</div>
        ${steps ? `<div class="eyebrow">How the results combine, and the one reading of the holdout</div><ul class="plan-list">${steps}</ul>` : ''}`;
    }
    async function mapAct(btn) {
      const box = btn.closest('[data-map-box]'), id = box && box.dataset.draft;
      if (!id) return;
      const act = btn.dataset.mapAct, body = { action: act };
      if (act === 'redraw' && !confirm('Draw the map again from your words? The changes you made to it, and your confirmation, are dropped.')) return;
      if (act === 'remove') body.part = btn.dataset.part;
      if (act === 'add') {
        body.title = ($('[data-map-title]', box).value || '').trim(); body.family = $('[data-map-family]', box).value;
        if (!body.title) { toast('Say in a few words what the new part finds.', { ok: false }); return; }
      }
      btn.disabled = true;
      try { const v = await DC.client.editMap(id, { body }); renderMap(box, v); if (act === 'confirm') toast('Map confirmed.'); }
      catch (err) { toast(err.message, { ok: false }); btn.disabled = false; }
    }
    async function mapMerge(sel) {
      const box = sel.closest('[data-map-box]'), id = box && box.dataset.draft;
      if (!id || !sel.value) return;
      try { renderMap(box, await DC.client.editMap(id, { body: { action: 'merge', keep: sel.value, other: sel.dataset.mapMerge } })); }
      catch (err) { toast(err.message, { ok: false }); sel.value = ''; }
    }
    el.addEventListener('change', e => { const sel = e.target.closest('[data-map-merge]'); if (sel) mapMerge(sel); });

    function renderPipeline(d) {
      const thread = $('#home-thread', el);
      const id = typeof d.asset === 'object' && d.asset ? d.asset.id : d.asset;
      if (d.step === 'designing') { addSystem(d.text); return; }
      if (typeof d.asset === 'object' && d.asset) { S.assets[id] = d.asset; updateDataPill(); }
      let box = id && $(`[data-asset-run="${id}"]`, thread);
      if (!box && id) {
        const a = S.assets[id] || {};
        thread.insertAdjacentHTML('beforeend', `<details class="tool-call" data-asset-run="${esc(id)}"><summary><span class="tname">${a.synthetic ? 'synthetic data' : 'prepare data'}</span><span class="tsum">${esc(a.name || 'data')}</span><span class="pill outline" data-asset-state>queued</span></summary><div class="tc-body small"><ol class="plan-list" data-asset-steps></ol></div></details>`);
        box = $(`[data-asset-run="${id}"]`, thread);
      }
      if (!box) return;
      const pill = $('[data-asset-state]', box), steps = $('[data-asset-steps]', box);
      const label = { queued: 'queued', structuring: 'reading', structured: 'cleaning', cleaned: 'describing', ready: 'ready', failed: 'failed' }[d.step] || d.step;
      pill.textContent = label;
      pill.className = 'pill ' + (d.step === 'ready' ? 'ok' : d.step === 'failed' ? 'bad' : 'accent');
      if (d.text && d.step !== 'queued') steps.insertAdjacentHTML('beforeend', `<li class="${d.step === 'failed' ? 'blocked' : 'done'}"><span class="pi">${d.step === 'failed' ? '!' : '✓'}</span><span class="t">${esc(d.text)}</span></li>`);
      if (d.log && d.log.length) steps.insertAdjacentHTML('beforeend', d.log.map(l => `<li class="done"><span class="pi">✓</span><span>${esc(l.title)}<span class="muted"> · ${esc(l.detail)}</span></span></li>`).join(''));
      if (d.step === 'ready' || d.step === 'failed') { if (S.assets[id]) S.assets[id].status = d.step; updateDataPill(); loadWorkspace(); }
      scrollThread();
    }
    function updateDataPill() {
      const ready = Object.values(S.assets).filter(a => a.status === 'ready' || a.status === 'queued' || !a.status);
      const last = ready[ready.length - 1];
      const pill = $('#home-data-pill', el);
      pill.hidden = !last;
      if (last) pill.textContent = `${last.name}${last.synthetic ? ' · synthetic' : ''}`;
    }

    function analysisCard(a, assetId) {
      if (!a) return '';
      const s = a.summary || {};
      const kinds = Object.entries(s.kinds || {}).map(([k, v]) => `${v} ${k}`).join(' · ');
      const cols = a.columns || [];
      const missing = cols.filter(c => c.missing_rate > 0).sort((x, y) => y.missing_rate - x.missing_rate).slice(0, 6);
      const num = cols.find(c => c.hist && c.hist.counts && c.hist.counts.length > 2);
      const ids = new Set(((a.profile || {}).id_candidates) || []);
      const cat = cols.find(c => c.top && c.top.length > 1 && !ids.has(c.name) && c.unique <= 50);
      let charts = '';
      if (missing.length) charts += `<div class="inset"><div class="eyebrow">Missing values</div>${DC.charts.barsH(missing.map(c => ({ label: c.name, value: c.missing_rate, valueText: pctTxt(c.missing_rate), cls: c.missing_rate > 0.3 ? 'warn' : '' })), { width: 420, labelW: 140, valW: 60, min: 0, max: 1, rowH: 24, ticks: [0, 0.5, 1], tickFmt: v => (v * 100) + '%', aria: 'Missing values by column' })}</div>`;
      if (num) charts += `<div class="inset"><div class="eyebrow">Distribution · ${esc(num.name)}</div>${DC.charts.vbars(num.hist.counts.map((v, i) => ({ value: v, label: i === 0 || i === num.hist.counts.length - 1 ? Number(num.hist.edges[i]).toPrecision(3) : null })), { width: 420, height: 140, aria: 'Histogram of ' + num.name })}</div>`;
      if (cat) charts += `<div class="inset"><div class="eyebrow">Top values · ${esc(cat.name)}</div>${DC.charts.barsH(cat.top.slice(0, 6).map(([v, n]) => ({ label: String(v).slice(0, 18), value: n, valueText: fmtN(n) })), { width: 420, labelW: 140, valW: 70, rowH: 22, aria: 'Top values of ' + cat.name })}</div>`;
      const personal = (((a.profile || {}).columns) || cols).filter(c => c.sensitivity && c.sensitivity !== 'none');  // 23.7: every column, not only the 60 charted
      const guard = personal.length ? `<div class="inset stack tight"><div class="eyebrow">Personal data: by name only</div><div class="row small" data-style="flex-wrap:wrap;gap:6px">${personal.slice(0, 12).map(c =>
        `<span class="pill outline">${esc(c.name)} · ${esc({ restricted: 'secret or id', pii: 'personal', sensitive: 'sensitive' }[c.sensitivity] || c.sensitivity)}</span>`).join('')}${personal.length > 12 ? `<span class="muted">and ${personal.length - 12} more</span>` : ''}</div>`
        + `<span class="xs muted">${personal.length === 1 ? 'This column is' : 'These columns are'} shown by name and class: no example, category name or prefix is kept or sent to a model. A sensitive column keeps its overall distribution; the outcome you choose to predict keeps its class names, since they define the task.</span></div>` : '';
      const hl = (a.highlights || []).slice(0, 5).map(h => `<li class="${h.severity === 'warning' ? 'blocked' : 'done'}"><span class="pi">${h.severity === 'warning' ? '!' : 'i'}</span><span><b>${esc(h.title)}</b> <span class="muted">${esc(h.text)}</span></span></li>`).join('');
      const ct = a.cases_table && (a.cases_table.tables || []).length ? a.cases_table : null;  // 21.3: known cases, and the table they label
      const bring = ct ? `<div class="inset stack tight"><div class="eyebrow">Known cases: the table they label</div>${ct.tables.map(t =>
        `<div class="spread small"><span><b>${esc(t.asset)}</b> in another draft holds ${fmtN(t.matched)} of these ${fmtN(t.of)} cases (key <code>${esc(t.key)}</code>)${t.key_repeats ? ` · ${fmtN(t.key_repeats)} of its rows repeat a key` : ''}</span>`
        + `<button type="button" class="btn sm" data-bring="${esc(t.draft_id)}">Bring it here</button></div>`).join('')}</div>` : '';
      return `<div class="stack tight">
        ${bring}
        <div class="stats"><div class="stat"><span class="v">${fmtN(s.rows)}</span><span class="l">rows${s.sampled_rows ? ' · charts on a sample' : ''}</span></div><div class="stat"><span class="v">${fmtN(s.columns)}</span><span class="l">${esc(kinds || 'columns')}</span></div><div class="stat"><span class="v">${pctTxt(s.missing_cell_rate)}</span><span class="l">missing cells</span></div><div class="stat"><span class="v">${fmtN(s.duplicate_rows)}</span><span class="l">duplicate rows left</span></div></div>
        ${hl ? `<ul class="plan-list">${hl}</ul>` : ''}
        ${guard}
        <div class="grid g2">${charts}</div>
        ${assetId ? `<div class="row small" data-style="gap:8px;flex-wrap:wrap"><a class="btn sm" href="${esc(DC.client.href.downloadAsset(S.draft.id, assetId))}" download>Download CSV</a><button type="button" class="btn sm" data-rows="${esc(assetId)}">View rows</button></div><div data-rows-box="${esc(assetId)}"></div>` : ''}
        ${edaSection(a, ids)}
        <div class="xs muted">Descriptive only: nothing here looks at the outcome yet. Its relations with each column are measured on training rows after the split, in the project's data stage (WF-04), so the holdout stays sealed.</div>
      </div>`;
    }
    /* 25.13 (the owner: "plots are not enough, the EDA is weak"): every column with its own chart and numbers, time trends and the
       strongest feature-to-feature correlations, from the analysis already computed (descriptive only, no outcome) */
    function edaSection(a, ids) {
      const cols = a.columns || [];
      if (!cols.length) return '';
      const num = v => (v == null || isNaN(v) ? '—' : Math.abs(v) >= 1000 ? fmtN(Math.round(v)) : Number(v).toPrecision(3));
      const card = c => {
        const head = `<div class="spread small"><b>${esc(c.name)}</b><span class="pill outline">${esc(c.kind)}</span></div>`
          + `<div class="xs muted">${pctTxt(c.missing_rate)} missing · ${fmtN(c.unique)} distinct</div>`;
        let body = '';
        if (ids.has(c.name)) body = '<div class="xs muted">Identifier-like: one value per row, never a feature.</div>';
        else if (c.hist && (c.hist.counts || []).length) {
          const q = c.quantiles || {};
          body = DC.charts.vbars(c.hist.counts.map((v, i) => ({ value: v, label: i === 0 ? num(c.hist.edges[0]) : i === c.hist.counts.length - 1 ? num(c.hist.edges[i + 1]) : null })), { width: 300, height: 110, aria: 'Histogram of ' + c.name })
            + `<div class="xs muted">mean ${num(c.mean)} · median ${num(q.p50)} · p01–p99 ${num(q.p01)} to ${num(q.p99)} · min ${num(c.min)} · max ${num(c.max)}</div>`;
        } else if (c.time && (c.time.by || []).length) {
          body = DC.charts.vbars(c.time.by.map(([t, n], i) => ({ value: n, label: i === 0 || i === c.time.by.length - 1 ? String(t).slice(0, 10) : null })), { width: 300, height: 110, aria: 'Rows per ' + (c.time.period || 'period') + ' of ' + c.name })
            + `<div class="xs muted">rows per ${esc(c.time.period || 'period')} · ${esc(String(c.time.min).slice(0, 10))} to ${esc(String(c.time.max).slice(0, 10))}</div>`;
        } else if ((c.top || []).length) {
          body = DC.charts.barsH(c.top.slice(0, 5).map(([v, n]) => ({ label: String(v).slice(0, 16), value: n, valueText: fmtN(n) })), { width: 300, labelW: 110, valW: 60, rowH: 20, noAxis: true, aria: 'Top values of ' + c.name })
            + (c.other_count ? `<div class="xs muted">and ${fmtN(c.other_count)} rows in other values</div>` : '');
        }
        return `<div class="inset stack tight">${head}${body}</div>`;
      };
      const corr = (a.correlations || []).slice(0, 8);
      const corrBox = corr.length ? `<div class="inset stack tight"><div class="eyebrow">Strongest correlations between columns</div>${DC.charts.barsH(corr.map(r => ({ label: `${r.a} · ${r.b}`.slice(0, 30), value: Math.abs(r.rho ?? 0), valueText: Number(r.rho ?? 0).toFixed(2) })), { width: 420, labelW: 200, valW: 50, min: 0, max: 1, rowH: 22, aria: 'Correlations' })}<div class="xs muted">Two columns that move together carry the same information; a model needs only one of them.</div></div>` : '';
      return `<details class="stack tight" open><summary class="eyebrow">Every column (${cols.length})</summary>${corrBox}<div class="grid g3">${cols.map(card).join('')}</div></details>`;
    }
    async function showRows(assetId, offset = 0) {
      const box = $(`[data-rows-box="${assetId}"]`, el); if (!box || !S.draft) return;
      try {
        const page = await DC.client.assetRows(S.draft.id, assetId, { query: { offset, limit: 50 } });
        box.innerHTML = `<div class="inset stack tight"><div class="spread small"><span>Rows ${fmtN(page.offset + 1)} to ${fmtN(page.offset + page.rows.length)} of ${fmtN(page.total)}${page.synthetic ? ' · <span class="pill warn">synthetic</span>' : ''}</span><span class="row" data-style="gap:6px">${page.offset > 0 ? `<button type="button" class="btn sm" data-rows="${esc(assetId)}" data-offset="${Math.max(0, page.offset - 50)}">Previous</button>` : ''}${page.offset + page.rows.length < page.total ? `<button type="button" class="btn sm" data-rows="${esc(assetId)}" data-offset="${page.offset + 50}">Next</button>` : ''}</span></div>`
          + `<div class="table-wrap" data-style="max-height:420px"><table class="data compact"><thead><tr>${page.columns.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${page.rows.map(r => `<tr>${r.map(v => `<td>${esc(v == null ? '' : String(v))}</td>`).join('')}</tr>`).join('')}</tbody></table></div></div>`;
        DC.hydrate(box);
      } catch (e) { toast(e.message, { ok: false }); }
    }

    /* ---------- solution workflow ---------- */
    function drawWorkflow(wf, rejected) {
      if (!wf) return;
      S.workflow = wf;
      $('#home-sol-title', el).textContent = wf.title || 'Solution workflow';
      $('#home-sol-version', el).textContent = 'v' + wf.version + (wf.source === 'model' ? ' · agent' : ' · template');
      $('#home-sol-graph', el).innerHTML = graph.render(wf, { selected: S.node });
      if (rejected && rejected.length) addSystem('DCLab kept the previous workflow: the proposed one ' + rejected.join('; ') + '.');
    }
    $('#home-sol-graph', el).addEventListener('click', e => {
      const g = e.target.closest('[data-node]'); if (!g || !S.workflow) return;
      const n = S.workflow.nodes.find(x => x.id === g.dataset.node); if (!n) return;
      S.node = n.id; drawWorkflow(S.workflow);
      $('#home-sol-node', el).textContent = `${n.wf} · ${n.label}: ${n.detail || ''}`;
    });

    /* ---------- talking ---------- */
    async function send(text) {
      text = (text || '').trim(); if (!text) return;
      try { await ensureDraft(); } catch (e) { return; }
      if (S.synthDescribe) { await describeSynthetic(text); return; }
      const to = $('#home-to', el).hidden ? 'both' : $('#home-to', el).value;
      if (to === 'claude') {
        try { await DC.client.twinMessage(S.draft.id, { body: { text } }); $('#home-reply', el).value = ''; toast('Sent to Claude only: DCLab does not see it.'); }
        catch (err) { toast(err.message, { ok: false }); }
        return;
      }
      try { await DC.client.message(S.draft.id, { body: { text } }); $('#home-reply', el).value = ''; }
      catch (err) { toast(err.status === 409 ? 'DCLab is still answering; send again in a moment.' : err.message, { ok: false }); }
    }
    /* ---------- synthetic data, the second way: describe in the chat, see the design, approve, then create ---------- */
    const APPROVE = /^(approve|approved|yes|yep|ok|okay|create it|create the data|looks good|go ahead|do it|generate it|build it|confirm)\b/i;
    function synthMode(on) {
      S.synthDescribe = on;
      $('#home-syn-banner', el).hidden = !on;
      $('#home-reply', el).placeholder = on ? 'Describe the table to simulate, or say “approve” when the design is right.' : 'Answer, ask, or say “create a project”.';
      if (on) { $('#home-chat', el).scrollIntoView({ behavior: 'smooth', block: 'start' }); $('#home-reply', el).focus(); }
    }
    async function approveSynthetic(id) {
      try { await DC.client.syntheticData(S.draft.id, { body: { proposal: id } }); synthMode(false); S.proposal = null; toast('Creating the table you approved.'); }
      catch (err) { toast(err.message, { ok: false }); }
    }
    async function describeSynthetic(text) {
      if (S.proposal && text.length <= 40 && APPROVE.test(text)) { $('#home-reply', el).value = ''; await approveSynthetic(S.proposal); return; }
      const rows = Number($('#home-syn-rows', el).value) || 5000;
      $('#home-reply', el).value = '';
      try { const got = await DC.client.designSynthetic(S.draft.id, { body: { description: text, rows } }); S.proposal = got.proposal; }
      catch (err) { toast(err.message, { ok: false }); }
    }
    $('#home-syn-way', el).addEventListener('segchange', e => {
      $('#home-syn-ai', el).hidden = e.detail !== 'ai'; $('#home-syn-describe', el).hidden = e.detail !== 'describe';
    });
    $('#home-syn-start', el).addEventListener('click', async () => {
      try { await ensureDraft({ data: true }); } catch (e) { return; }
      synthMode(true);
    });
    $('#home-syn-leave', el).addEventListener('click', async () => {
      synthMode(false);
      if (S.draft && S.proposal) { try { await DC.client.designSynthetic(S.draft.id, { body: { cancel: true } }); } catch (e) { /* kept: it can still be approved from its message */ } S.proposal = null; }
    });
    $('#home-thread', el).addEventListener('click', async e => {
      const ok = e.target.closest('[data-syn-approve]');
      if (ok) { ok.disabled = true; await approveSynthetic(ok.dataset.synApprove); return; }
      const chip = e.target.closest('[data-say]');
      if (chip) { send(chip.dataset.say); return; }
      const nbRun = e.target.closest('[data-nb-run]');
      if (nbRun) { await runChatCell(nbRun); return; }
      const adopt = e.target.closest('[data-improve-adopt]');
      if (adopt) {
        adopt.disabled = true;
        try { const got = await DC.client.adoptImprovement(adopt.dataset.improveProject, { body: { candidate: adopt.dataset.improveAdopt } }); adopt.textContent = 'Adopted'; toast(got.next, { ms: 9000 }); }
        catch (err) { adopt.disabled = false; toast(err.message, { ok: false, ms: 7000 }); }
        return;
      }
      const ai = e.target.closest('[data-syn-ai]');
      if (ai) {
        ai.disabled = true;
        let offer = {}; try { offer = JSON.parse(ai.dataset.synAi); } catch (err) { offer = {}; }
        try { await DC.client.syntheticData(S.draft.id, { body: { prompt: offer.prompt || S.draft.problem, rows: Number(offer.rows) || 5000 } }); toast('Building the synthetic table with AI.'); }
        catch (err) { ai.disabled = false; toast(err.message, { ok: false }); }
        return;
      }
      const describe = e.target.closest('[data-syn-describe]');
      if (describe) { synthMode(true); return; }
      const change = e.target.closest('[data-syn-change]');
      if (change) { S.proposal = change.dataset.synChange; synthMode(true); return; }
      const drop = e.target.closest('[data-syn-drop]');
      if (drop) { drop.disabled = true; synthMode(false); S.proposal = null; try { await DC.client.designSynthetic(S.draft.id, { body: { cancel: true } }); } catch (err) { toast(err.message, { ok: false }); } }
    });
    $('#home-send', el).addEventListener('click', () => send($('#home-reply', el).value));
    $('#home-reply', el).addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send($('#home-reply', el).value); } });
    $('#home-thread', el).addEventListener('click', e => {
      const opt = e.target.closest('.ask-option[data-answer]');
      if (opt && !opt.disabled) send(opt.dataset.answer);
    });

    /* ---------- bring the data ---------- */
    function openBring(tab) {
      const card = $('#home-bring', el);
      card.hidden = false; $('#home-connect', el).setAttribute('aria-expanded', 'true');
      if (tab && tab !== '1') { const b = $(`[data-tabs="hsrc"] [data-tab="${tab}"]`, el); if (b) b.click(); }
      card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    $('#home-connect', el).addEventListener('click', () => { const card = $('#home-bring', el); if (card.hidden) openBring(); else { card.hidden = true; $('#home-connect', el).setAttribute('aria-expanded', 'false'); } });
    el.addEventListener('click', async e => {
      const mp = e.target.closest('[data-map-act]'); if (mp) { mapAct(mp); return; }
      const ob = e.target.closest('[data-open-bring]'); if (ob) { openBring(ob.dataset.openBring); return; }
      const b = e.target.closest('[data-build]'); if (b) { build(); return; }
      const fo = e.target.closest('[data-follow]'); if (fo) { DC.state.internSession = fo.dataset.follow; if (fo.dataset.project) DC.state.projectId = fo.dataset.project; location.hash = 'intern'; return; }
      const op = e.target.closest('[data-open-project]'); if (op) { DC.state.projectId = op.dataset.openProject; location.hash = 'project'; return; }
      // 29.10: a note opens its own project's Solution page, whatever project was open before
      const on = e.target.closest('[data-open-notes]'); if (on) { e.preventDefault(); DC.state.projectId = null; DC.state.project = 'p:' + on.dataset.openNotes; DC.currentProject.clear(); location.hash = 'solution'; return; }
      const wz = e.target.closest('[data-wizard]'); if (wz) { wizard(); return; }
      const c = e.target.closest('[data-connect]');
      if (c) {
        const kind = c.dataset.connect, X = DC.connectors;
        if (kind === 'kaggle') { X.kaggleSearch($('#home-kaggle-q', el).value, $('#home-kaggle-list', el)); return; }
        try { await ensureDraft({ data: true }); } catch (err) { return; }
        const started = () => $('#home-chat', el).scrollIntoView({ behavior: 'smooth', block: 'start' });
        if (kind === 'hf') X.importHF(S.draft.id, { dataset: $('#home-hf-id', el).value.trim(), revision: $('#home-hf-rev', el).value.trim(), split: $('#home-hf-split', el).value.trim(), config: $('#home-hf-cfg', el).value.trim() }).then(started).catch(err => { if (err.message !== 'no dataset') toast(err.message, { ok: false }); });
        if (kind === 'database') X.database(S.draft.id, started);
        if (kind === 'cloud') X.cloud(S.draft.id, started);
        return;
      }
      const k = e.target.closest('[data-kaggle-ref]');
      if (k) {
        try { await ensureDraft({ data: true }); await DC.connectors.importKaggle(S.draft.id, k.dataset.kaggleRef); k.disabled = true; k.textContent = 'Importing…'; $('#home-chat', el).scrollIntoView({ behavior: 'smooth', block: 'start' }); }
        catch (err) { if (err.message !== 'no problem') toast(err.message, { ok: false }); }
      }
    });
    async function uploadFile(file) {
      if (!file) return;
      try { await ensureDraft({ data: true }); } catch (e) { return; }
      const t0 = Date.now();
      DC.ui.note('ui.upload_started', { view: 'home', file_kind: (file.name.split('.').pop() || 'file').toLowerCase().replace(/[^a-z0-9]/g, '').slice(0, 12) || 'file', bytes: file.size });
      try {
        toast(`Uploading ${file.name}…`);
        await DC.client.upload(S.draft.id, { query: { filename: file.name }, body: file, headers: { 'Content-Type': 'application/octet-stream' } });
        DC.ui.note('ui.upload_finished', { view: 'home', status: 'ok', seconds: (Date.now() - t0) / 1000 });
        $('#home-chat', el).scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch (err) { DC.ui.note('ui.upload_finished', { view: 'home', status: 'failed', seconds: (Date.now() - t0) / 1000 }); toast(err.message, { ok: false }); }
    }
    $('#home-file', el).addEventListener('change', e => uploadFile(e.target.files[0]));
    const drop = $('#home-drop', el);
    drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('over'));
    drop.addEventListener('drop', e => { e.preventDefault(); drop.classList.remove('over'); uploadFile(e.dataTransfer.files[0]); });

    async function loadSamples() {
      if (S.samples) return;
      try { S.samples = await DC.client.samples(); } catch (e) { S.samples = []; }
      $('#home-sample-count', el).textContent = S.samples.length;
      $('#home-samples tbody', el).innerHTML = S.samples.map(s => `<tr data-sample="${esc(s.key)}"><td><span class="cell-main">${esc(s.key)}</span><div class="cell-sub">${esc(s.name || '')}</div></td><td>${esc(s.task || '')}</td><td class="num">${fmtN(s.rows)}</td><td class="small">${esc(s.decision || s.moment || '')}</td><td><button type="button" class="btn sm">Use</button></td></tr>`).join('');
    }
    $('[data-tabs="hsrc"]', el).addEventListener('tabchange', e => { if (e.detail === 'samples') loadSamples(); });
    $('#home-samples', el).addEventListener('click', async e => {
      const tr = e.target.closest('tr[data-sample]'); if (!tr || e.target.closest('[data-record]')) return;
      try { await ensureDraft({ data: true }); } catch (err) { return; }
      try { await DC.client.sample(S.draft.id, { body: { key: tr.dataset.sample } }); $('#home-chat', el).scrollIntoView({ behavior: 'smooth', block: 'start' }); }
      catch (err) { toast(err.message, { ok: false }); }
    });
    DC.synthetic.setup($('#home-syn-template-field', el));
    $('#home-syn-go', el).addEventListener('click', async () => {
      try { await ensureDraft({ data: true }); } catch (e) { return; }
      try {
        await DC.client.syntheticData(S.draft.id, { body: { prompt: $('#home-syn-prompt', el).value || S.draft.problem, rows: Number($('#home-syn-rows', el).value), template: DC.synthetic.chosen($('#home-syn-template-field', el)) } });
        $('#home-chat', el).scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch (err) { toast(err.message, { ok: false }); }
    });

    /* ---------- build ---------- */
    async function build() {
      // 25.13: one step, the agentic way: the project is built from what the chat settled and DCLab's agent carries out the plan
      // (the chat shows the plan and links to follow it); what is still missing comes back as a message, never a dead end.
      // The button says it in the chat, so it goes the same way as typing "build it" (to DCLab, whatever the send-to choice).
      if (!S.draft) { toast('Start with one sentence first.', { ok: false }); return; }
      if (S.draft.project_id) { DC.state.projectId = S.draft.project_id; location.hash = 'project'; return; }  // built already: open it
      creating(true);
      try { await DC.client.message(S.draft.id, { body: { text: 'Create a project from this chat' } }); }
      catch (err) { creating(false); toast(err.status === 409 ? 'DCLab is still answering; press it again in a moment.' : err.message, { ok: false }); }
    }
    function creating(on) {  // the header button while DCLab builds: it answers with the plan, or with what is missing first
      const b = $('#home-create', el); if (!b) return;
      b.disabled = on; b.textContent = on ? 'Building…' : (S.draft && S.draft.project_id ? 'Open the project' : 'Build this chat as a project');
    }
    $('#home-create', el).addEventListener('click', build);
    // dclab:closed-begin  (the understanding team: the full edition's)
    $('#home-team-go', el).addEventListener('click', () => teamStep({}));  // 28.8: start the walk (a walk already waiting answers with its pause)
    // dclab:closed-end
    /* The header's "New project from chat" (the owner, 2026-10-16: a completely new project, never the old chat): the open
       conversation is put away (it stays in Projects and Needs you, to resume from there), the box is cleared and focused,
       and the page says what comes next */
    function newChat() {
      S.fresh = true;  // a new problem on purpose: Home does not reopen the current project's chat until a draft starts
      if (S.close) { S.close(); S.close = null; }
      S.draft = null; S.analysis = {}; S.assets = {};
      remember(null);
      $('#home-thread', el).innerHTML = '';
      $('#home-reply', el).value = '';
      showDraftPanels(false);
      creating(false);
      const box = $('#home-task', el);
      box.value = '';
      box.scrollIntoView({ behavior: 'smooth', block: 'center' });
      box.focus();
      toast('A new project: describe the problem in one sentence and press Draft the solution. Your earlier conversation is kept under Projects.');
    }
    el.addEventListener('click', e => { if (e.target.closest('[data-new-chat]')) newChat(); });
    function wizard() {  // the step-by-step path: data, solution draft, split and budget, then "Review and start"
      if (!S.draft) return;
      DC.state.draftId = S.draft.id;
      location.hash = 'new';
    }

    /* ---------- activity: what runs now (/api/ops/jobs) and what is new in the evidence (/api/ops/evidence-recent) ---------- */
    const ACT = { stop: null };
    const actSecs = v => (v == null ? '' : v < 90 ? Number(v).toFixed(0) + ' s' : v < 5400 ? (v / 60).toFixed(1) + ' min' : (v / 3600).toFixed(1) + ' h');
    function liveItem(j) {
      const state = j.status === 'running' && j.seconds != null ? `${j.label} · ${actSecs(j.seconds)} so far` : j.label;
      const main = { stage: `Stage · ${j.project || 'project'} · ${j.title}`, intern: `Intern · ${j.title}`, data: `Data · ${j.title}`, research: `Research run · ${j.title}` }[j.kind] || j.title;
      const link = j.kind === 'stage' ? `href="#project" data-live-project="${esc(j.project_id)}"`
        : j.kind === 'intern' ? `href="#intern" data-live-session="${esc(j.session_id)}"`
          : j.kind === 'data' && j.draft_open ? `href="#home" data-resume="${esc(j.draft_id)}"` : 'href="#compute"';
      return `<a class="list-item" ${link}><span class="dot ${j.status === 'running' ? 'live' : 'accent'}" data-style="margin-top:6px"></span><div class="li-main"><span class="li-title">${esc(main)}</span><span class="li-sub">${esc(`${state} · ${j.where}${j.detail ? ' · ' + j.detail : ''}`)}</span></div></a>`;
    }
    async function loadLive() {
      let d;
      try { d = await DC.client.opsJobs({ quiet: true }); } catch (e) { DC.states.failed($('#home-live', el), e, { lead: 'Could not read the jobs: ', retry: loadLive }); return false; }
      const here = S.here, hereDraft = S.hereDraft;
      const live = d.jobs.filter(j => (j.status === 'running' || j.status === 'queued')
        && (!here || j.project_id === here.id || (hereDraft && j.draft_id === hereDraft)));  // the selected project's work only
      $('#home-live', el).innerHTML = live.length ? live.slice(0, 6).map(liveItem).join('')
        : `<div class="empty">Nothing is running. Stage runs, intern sessions and data pipelines show here while they work; ${fmtN(d.totals.total)} job${d.totals.total === 1 ? '' : 's'} ran on this machine so far.</div>`;
      DC.setNavCount('compute', d.totals.running || '');
      return live.length > 0;
    }
    const EV_SHORT = { rule: 'rule', workflow: 'workflow', dataset: 'dataset', experiment: 'experiment', leakage_precedent: 'leak', pitfall: 'pitfall', finding: 'finding' };
    const evRef = id => (DC.REC[id] ? chip(id) : `<button type="button" class="ev-chip" data-ev-live="${esc(id)}" title="Open ${esc(id)}">${esc(id)}</button>`);
    async function loadEvidence() {
      const box = $('#home-ev-new', el);
      let d;
      try { d = await DC.client.opsEvidenceRecent({ query: { limit: '3' }, quiet: true }); } catch (e) { DC.states.failed(box, e, { lead: 'Could not read the evidence index: ', retry: loadEvidence }); return; }
      // The index has no dates of its own: the server dates a record by the campaign result it cites (see ops.py).
      box.innerHTML = d.records.length ? d.records.map(r => {
        const title = r.title.startsWith(r.id) ? r.title.slice(r.id.length).replace(/^\s*[·:—-]\s*/, '') : r.title;
        return `<div class="row top"><span class="pill ${DC.TYPE_CLS[r.type] || ''}">${esc(EV_SHORT[r.type] || r.type)}</span><span>${evRef(r.id)} ${esc(title)} <span class="muted">· ${r.date ? esc(r.date.slice(0, 10)) : 'undated'}${r.campaign ? ' · ' + esc(r.campaign) : ''}</span>${r.related.length ? ' ' + r.related.slice(0, 2).map(evRef).join(' ') : ''}</span></div>`;
      }).join('') + `<div class="xs muted">${esc(d.note)} ${fmtN(d.total)} records in the index.</div>`
        : `<div class="empty">${esc(d.note || 'The evidence index is empty.')}</div>`;
      DC.hydrate(box);
    }
    $('#home-live', el).addEventListener('click', e => {
      const p = e.target.closest('[data-live-project]');
      if (p) { DC.state.project = 'p:' + p.dataset.liveProject; DC.state.projectId = null; DC.currentProject.clear(); return; }
      const s = e.target.closest('[data-live-session]');
      if (s) DC.state.internSession = s.dataset.liveSession;  // a draft link uses data-resume, handled above
    });
    $('#home-ev-new', el).addEventListener('click', async e => {
      const b = e.target.closest('[data-ev-live]'); if (!b) return;  // records outside the demo snapshot come from /api/evidence/{id}
      try {
        const r = await DC.client.evidenceRecord(b.dataset.evLive);
        DC.drawer.open({ eyebrow: `<span class="pill ${DC.TYPE_CLS[r.type] || ''}">${esc(DC.TYPE_LABEL[r.type] || r.type)}</span> <span class="tag">${esc(r.record_id)}</span>`, title: r.title,
          html: `<div class="record-text">${DC.linkIds(r.text)}</div>${(r.citations || []).length ? `<div class="stack tight"><div class="eyebrow">Citations</div>${r.citations.map(c => `<code class="small" data-style="overflow-wrap:anywhere">${esc(c)}</code>`).join('')}</div>` : ''}` });
      } catch (err) { toast(err.message, { ok: false }); }
    });
    // While the Activity tab is open, the live list refreshes every 3 s as long as something runs.
    el.addEventListener('paneshow', e => {
      if (e.detail !== 'activity') return;
      if (ACT.stop) ACT.stop();
      ACT.stop = DC.poll(async () => DC.state.view !== 'home' || $('[data-pane="activity"]', el).hidden || !(await loadLive()), 3000);
    });
    this.loadActivity = () => Promise.all([loadLive(), loadEvidence()]);

    this.loadWorkspace = loadWorkspace;
    this.resume = resume;
    document.addEventListener('dclab:project', () => {  // the person chose another project: its chat, its numbers
      if (DC.state.view !== 'home') return;
      S.fresh = false; S.draft = null; S.here = null; S.hereDraft = null;
      if (S.close) { S.close(); S.close = null; }
      $('#home-thread', el).innerHTML = '';
      $('#home-reply', el).value = '';
      showDraftPanels(false);
      creating(false);
    });
    /* The chat of the project the person is on (chosen on another page), shown on Home to read and go on with. */
    this.projectChat = async () => {
      const key = DC.state.project;
      if (S.fresh || S.draft || typeof key !== 'string' || !key.startsWith('p:')) return false;  // a chat on screen is never replaced
      try {
        const project = await DC.currentProject.get();
        const draft = project && (project.draft || {}).id;
        if (!draft || S.draft || S.fresh) return false;  // the person started a chat while the project was read
        return await resume(draft);
      } catch (e) { return false; }
    };
    loadPacks();
  },
  enter(el) {
    DC.markSample(false);
    const asked = DC.state.resumeDraft;  // a draft opened from another page (Admin · Sessions)
    if (asked) { DC.state.resumeDraft = null; this.resume(asked); }
    else this.projectChat().then(shown => {  // the open project's chat first, else the chat last left open here
      if (shown) return;
      try { const id = localStorage.getItem('dclab-home-draft'); if (id && !(document.querySelector('#home-thread') || {}).childElementCount) this.resume(id); } catch (e) { /* storage blocked */ }
    });
    return this.loadWorkspace().then(() => this.loadActivity());  // the page's own load, the project first (the jobs are its own): Loading… and a failed read are shown (13.1)
  },
});
