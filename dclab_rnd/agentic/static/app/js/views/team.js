/* W8 of the collaborative workspace (2026-11-20): the team threads of the open project.
   A team thread (each team talks to its own agent; a message can ask another team), a task's page (requested output, done
   criteria, every action with its status, evidence, remaining work, who owes the next response), its report (the answer
   first, details to expand, "Simulated" marked, superseded revisions linked both ways), the problem's shared handoff
   history in the owner's words, and the agent's job description. Everything is read from the API: nothing here invents a
   status, a score or a next owner. #team/<thread id> opens a thread; #team/task:<task id> opens a task. */
DC.view('team', {
  init(el) {
    const T = this.T = { gen: 0, scope: 'project', threads: [], thread: null, seen: new Set(), seq: 0, close: null, teams: [], problems: [], pid: null, task: null, taskId: null, projects: [] };
    try { if (localStorage.getItem('dclab-team-scope') === 'all') T.scope = 'all'; } catch (e) { /* storage blocked */ }
    const { $, $$ } = DC;
    $$('[data-tm-scope]', el).forEach(b => b.addEventListener('click', () => {
      T.scope = b.dataset.tmScope;
      try { localStorage.setItem('dclab-team-scope', T.scope); } catch (e) { /* storage blocked */ }
      this.paintInbox(el);
    }));
    $('#tm-compose', el).addEventListener('submit', e => { e.preventDefault(); this.send(el); });
    $('#tm-text', el).addEventListener('keydown', e => {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); this.send(el); }
      else if (e.key === 'Escape' && T.replyTo) { e.preventDefault(); this.replyTo(el, null); }
    });
    $('#tm-open', el).addEventListener('submit', e => { e.preventDefault(); this.openThread(el); });
    $('#tm-open-problem', el).addEventListener('change', () => { $('#tm-open-name-field', el).hidden = $('#tm-open-problem', el).value !== '__new'; });
    el.addEventListener('click', e => {
      const open = e.target.closest('[data-tm-task]');
      if (open) { e.preventDefault(); this.openTask(el, open.dataset.tmTask, { focus: true }); return; }
      const reply = e.target.closest('[data-tm-reply]');
      if (reply) { e.preventDefault(); this.replyTo(el, reply.dataset.tmReply); return; }
      const jump = e.target.closest('[data-tm-report]');
      if (jump) { e.preventDefault(); this.jumpToReport(el, jump.dataset.tmReport); return; }
      const job = e.target.closest('[data-tm-job]');
      if (job) { e.preventDefault(); this.loadJob(el, job.dataset.tmJob, true); return; }
      const pick = e.target.closest('[data-tm-thread]');
      if (pick) { e.preventDefault(); this.pickThread(el, pick.dataset.tmThread); return; }
      if (e.target.closest('[data-tm-close-task]')) { this.closeTask(el); }
    });
    // leaving the page ends its stream, and a read still under way never opens one (its generation is gone)
    window.addEventListener('hashchange', () => { if (DC.state.view !== 'team') { this.T.gen++; this.stop(); } });
  },

  /* ---------------- words ---------------- */
  TEAM: { business: 'Business', data_science: 'Data Science', ml_engineering: 'ML Engineering', sales: 'Sales', marketing: 'Marketing',
    project: 'Project', problem: 'Problem', model: 'Model', home: 'Home', lead: 'Lead', advisor: 'Advisor' },
  label(name) {
    const n = String(name || '');
    if (this.TEAM[n]) return this.TEAM[n];
    const [base, q] = n.split(':');
    const words = base.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    return q ? `${words} (${q})` : words;
  },
  kindOf(team) {  // a thread's team is a team's id, or an agent's name when the workspace has no such team
    const found = (this.T.teams || []).find(t => t.id === team);
    return String((found && found.kind) || team || '').split(':')[0];
  },
  token(kind, text) {
    const k = ['business', 'data_science', 'ml_engineering', 'sales', 'marketing', 'model'].includes(kind) ? kind : 'person';
    return `<span class="tm-token tm-tone-${k}" aria-hidden="true">${DC.esc(text)}</span>`;
  },
  initials(words) { return String(words || '?').split(/[\s(]+/).filter(Boolean).slice(0, 2).map(w => w.charAt(0).toUpperCase()).join('') || '?'; },
  when(x) { const t = Date.parse(x || ''); return isNaN(t) ? '' : new Date(t).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }); },
  ACTION: { succeeded: ['ok', 'Succeeded'], simulated: ['proof', 'Simulated'], denied: ['bad', 'Denied'], failed: ['bad', 'Failed'], timed_out: ['bad', 'Timed out'],
    skipped: ['outline', 'Skipped'], planned: ['info', 'Planned'], running: ['info', 'Running'] },
  REPORT: { complete: ['ok', 'Complete'], partial: ['warn', 'Partial'], blocked: ['warn', 'Blocked'], failed: ['bad', 'Failed'], superseded: ['outline', 'Superseded'] },
  TASK: { open: ['info', 'Open'], accepted: ['info', 'Accepted'], waiting_for_person: ['warn', 'Waiting for a person'], waiting_for_team: ['warn', 'Waiting for a team'],
    partial: ['warn', 'Partial'], complete: ['ok', 'Complete'], blocked: ['warn', 'Blocked'], failed: ['bad', 'Failed'], cancelled: ['outline', 'Cancelled'], superseded: ['outline', 'Superseded'] },
  HANDOFF: { task: 'Question', clarification: 'Clarifying question', answer: 'Answer', reply: 'Reply with report', hint: 'What it needs', which_task: 'Which task?' },
  pill(map, status) { const [cls, words] = map[status] || ['outline', String(status || '').replace(/_/g, ' ')]; return `<span class="pill ${cls}" data-status="${DC.esc(status || '')}">${DC.esc(words)}</span>`; },
  simulatedPill: '<span class="pill proof tm-sim" title="A fixture result that illustrates the handoff: no real model ran">Simulated</span>',
  words(name) { return String(name || '').replace(/_/g, ' '); },

  /* ---------------- the page ---------------- */
  async enter(el) {
    const { $, esc } = DC;
    const T = this.T, gen = ++T.gen;
    this.stop();
    const p = await DC.currentProject.get();
    DC.markSample(false);
    if (gen !== T.gen) return;
    if (!p) {
      T.pid = null;
      $('#tm-eyebrow', el).textContent = 'Project · teams';
      $('#tm-title', el).textContent = 'Team threads';
      $('#tm-grid', el).hidden = true;
      const none = $('#tm-none', el);
      none.hidden = false;
      DC.states.empty(none, 'No project is open yet. Start one on Home; its teams talk here.', { href: '#home', action: 'Go to Home' });
      return;
    }
    $('#tm-none', el).hidden = true;
    $('#tm-grid', el).hidden = false;
    T.pid = p.id;
    $('#tm-eyebrow', el).textContent = `${p.name} · teams`;
    $('#tm-title', el).textContent = 'Team threads';
    DC.states.loading($('#tm-stream', el));
    const [threads, teams, problems, projects] = await Promise.all([
      DC.client.myTeamThreads(),
      DC.client.listTeams({ quiet: true }).catch(() => []),
      DC.client.projectProblemsList(p.id, { quiet: true }).catch(() => ({ problems: [] })),
      DC.projectSwitch.load().catch(() => []),
    ]);
    if (gen !== T.gen) return;
    T.threads = threads || []; T.teams = teams || []; T.problems = (problems && problems.problems) || []; T.projects = projects || [];
    DC.setNavCount('team', T.threads.length || '');
    this.paintOpenForm(el);
    const [, pane] = (location.hash || '').replace(/^#/, '').split('/');
    let saved = null;
    try { saved = localStorage.getItem('dclab-team-thread'); } catch (e) { saved = null; }
    const here = T.threads.filter(t => t.project_id === p.id);
    const wanted = pane && pane.startsWith('th_') ? pane : saved;
    const thread = here.find(t => t.id === wanted) || here[here.length - 1] || null;
    this.paintInbox(el, thread && thread.id);
    if (!thread) {
      T.thread = null;
      $('#tm-thread-title', el).textContent = 'No team thread yet';
      $('#tm-thread-eyebrow', el).textContent = 'Team conversation';
      $('#tm-compose', el).hidden = true; $('#tm-readonly', el).hidden = true; $('#tm-live', el).hidden = true;
      DC.states.empty($('#tm-stream', el), here.length ? '' : 'You have no thread on this project yet. Open your team\'s thread on one of its problems (Team inbox).');
      $('#tm-open-wrap', el).open = true;
      this.closeTask(el);
      await this.loadHistory(el, (problems && problems.default_problem_id) || (T.problems[0] || {}).problem_id);
      await this.loadJob(el, 'business');
      return;
    }
    await this.showThread(el, thread, gen);
    if (pane && pane.startsWith('task:')) await this.openTask(el, decodeURIComponent(pane.slice(5)));
    else this.closeTask(el);
  },

  stop() { const T = this.T; if (T && T.close) { T.close(); T.close = null; } },

  problemName(id) { const f = (this.T.problems || []).find(x => x.problem_id === id); return f ? f.name : 'Problem'; },
  projectName(id) { const f = (this.T.projects || []).find(x => x.id === id); return f ? f.name : 'Another project'; },

  paintInbox(el, current) {
    const { $, $$, esc } = DC;
    const T = this.T;
    $$('[data-tm-scope]', el).forEach(b => b.setAttribute('aria-pressed', String(b.dataset.tmScope === T.scope)));
    const shown = T.scope === 'all' ? T.threads : T.threads.filter(t => t.project_id === T.pid);
    const box = $('#tm-inbox', el);
    const active = current || (T.thread && T.thread.id);
    if (!shown.length) {
      const others = T.threads.length - shown.length;
      DC.states.empty(box, others ? `No thread on this project. ${others} in other projects (All).` : 'No team thread yet.');
      return;
    }
    box.innerHTML = shown.map(t => {
      const kind = this.kindOf(t.team), name = t.team_label || this.label(t.team);
      const where = t.project_id === T.pid ? this.problemName(t.problem_id) : this.projectName(t.project_id);
      return `<li><button type="button" class="tm-inbox-item" data-tm-thread="${esc(t.id)}"${t.id === active ? ' aria-current="true"' : ''}>`
        + `${this.token(kind, this.initials(name))}<span class="tm-inbox-main"><b>${esc(name)}</b><small>${esc(where)}</small></span>`
        + `<span class="tm-count" title="Messages">${esc(String(t.messages || 0))}</span>${t.access === 'read' ? '<span class="pill outline">Read only</span>' : ''}</button></li>`;
    }).join('');
  },

  paintOpenForm(el) {
    const { $, esc } = DC;
    const T = this.T;
    const probs = $('#tm-open-problem', el);
    probs.innerHTML = T.problems.map(x => `<option value="${esc(x.problem_id)}">${esc(x.name || x.problem_id)}</option>`).join('') + '<option value="__new">A new problem…</option>';
    $('#tm-open-name-field', el).hidden = probs.value !== '__new';
    const teams = T.teams.length ? T.teams.map(t => [t.id, t.name || this.label(t.kind)])
      : ['business', 'data_science', 'ml_engineering', 'sales', 'marketing'].map(k => [k, this.label(k)]);
    $('#tm-open-team', el).innerHTML = teams.map(([v, w]) => `<option value="${esc(v)}">${esc(w)}</option>`).join('');
  },

  async openThread(el) {
    const { $, toast } = DC;
    const T = this.T;
    const btn = $('#tm-open-go', el);
    let problem = $('#tm-open-problem', el).value;
    btn.disabled = true;
    try {
      if (problem === '__new' || !problem) {
        const name = $('#tm-open-name', el).value.trim();
        if (!name) { toast('Name the problem first.', { ok: false }); $('#tm-open-name', el).focus(); return; }
        problem = (await DC.client.projectProblemsCreate(T.pid, { body: { name } })).problem_id;
      }
      const thread = await DC.client.openTeamThread(T.pid, problem, { body: { team: $('#tm-open-team', el).value } });
      try { localStorage.setItem('dclab-team-thread', thread.id); } catch (e) { /* storage blocked */ }
      history.replaceState(null, '', '#team/' + thread.id);
      $('#tm-open-wrap', el).open = false;
      await this.enter(el);
    } catch (e) {
      toast(e.message || 'The thread could not be opened.', { ok: false });
    } finally { btn.disabled = false; }
  },

  async pickThread(el, id) {
    const T = this.T;
    const thread = T.threads.find(t => t.id === id);
    if (!thread) return;
    try { localStorage.setItem('dclab-team-thread', id); } catch (e) { /* storage blocked */ }
    if (thread.project_id !== T.pid) {  // a thread of another project: that project becomes the open one, as Recents does
      DC.state.project = 'p:' + thread.project_id; DC.currentProject.clear();
      try { localStorage.setItem('dclab-project', thread.project_id); } catch (e) { /* storage blocked */ }
      history.replaceState(null, '', '#team/' + id);
      await this.enter(el);
      return;
    }
    history.replaceState(null, '', '#team/' + id);
    this.paintInbox(el, id);
    this.closeTask(el);
    await this.showThread(el, thread, T.gen);
  },

  /* ---------------- the thread ---------------- */
  async showThread(el, thread, gen) {
    const { $ } = DC;
    const T = this.T;
    this.stop();
    T.thread = thread; T.seen = new Set(); T.seq = 0;
    const kind = this.kindOf(thread.team);
    $('#tm-thread-eyebrow', el).textContent = `${this.problemName(thread.problem_id)} · team conversation`;
    $('#tm-thread-title', el).textContent = `${thread.team_label || this.label(thread.team)} agent`;
    const stream = $('#tm-stream', el);
    DC.states.loading(stream);
    let view;
    try { view = await DC.client.readTeamThread(T.pid, thread.id, { quiet: true }); } catch (e) {
      DC.states.failed(stream, e, { retry: () => this.showThread(el, thread, T.gen) });
      return;
    }
    if (gen !== T.gen || T.thread !== thread) return;
    stream.innerHTML = '';
    T.replyTo = null; T.pending = null;  // a reply target belongs to the thread it was chosen in
    (view.messages || []).forEach(m => this.addMessage(el, m));
    if (!T.seen.size) DC.states.empty(stream, 'Nothing has been said here yet. Write to your agent below.');
    const can = view.access || thread.access;
    $('#tm-compose', el).hidden = can !== 'write';
    $('#tm-readonly', el).hidden = can === 'write';
    $('#tm-ask-wrap', el).hidden = kind !== 'business';  // the Business agent is the one that asks other teams
    this.composeNote(el);
    this.listen(el, thread);
    await Promise.all([this.loadHistory(el, thread.problem_id), this.loadJob(el, kind)]);
  },

  listen(el, thread) {
    if (DC.state.view !== 'team') return;  // the person left while the thread was read: no stream for a page not shown
    const T = this.T;
    this.stop();
    const live = DC.$('#tm-live', el);
    T.close = DC.stream(DC.client.path.teamThreadEvents(T.pid, thread.id, { query: { after: T.seq } }), (type, data) => {
      if (type !== 'message' || !data || T.thread !== thread) return;
      live.hidden = false;
      if (this.addMessage(el, data)) {
        this.composeNote(el);
        if (data.handoff) {
          clearTimeout(T.later);
          T.later = setTimeout(() => {
            this.loadHistory(el, thread.problem_id);
            if (T.taskId) this.openTask(el, T.taskId, { quiet: true });
          }, 300);
        }
      }
    });
  },

  addMessage(el, m) {
    const { $, esc } = DC;
    const T = this.T;
    const seq = Number(m.seq) || 0;
    if (T.seen.has(seq)) return false;
    T.seen.add(seq); T.seq = Math.max(T.seq, seq);
    const stream = $('#tm-stream', el);
    const empty = $(':scope > .empty', stream); if (empty) empty.remove();
    const a = m.author || {}, h = m.handoff || null;
    const agent = a.kind === 'agent';
    const team = agent ? (a.team || a.id) : '';
    const who = agent ? `${this.label(team)} agent` : (a.name || 'A member');
    const lines = String(m.text || '').split('\n');
    const sim = /^Simulated\b/.test(lines[0]);
    let body;
    if (h && h.kind === 'reply' && lines.length >= 4 && /^Why: /.test(lines[1])) {  // a reply: what was found, why, who acts next, what changed
      const four = lines.slice(1).map(l => { const i = l.indexOf(': '); return i > 0 ? [l.slice(0, i), l.slice(i + 2)] : ['', l]; });
      body = `<p class="tm-text tm-found">${esc(lines[0])}</p><dl class="tm-four">${four.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('')}</dl>`;
    } else body = `<p class="tm-text">${esc(m.text || '')}</p>`;
    const route = h ? `<span class="tm-route">${esc(this.label(h.from_team))} → ${esc(this.label(h.to_team))}</span>` : '';
    // T3: a question an agent asked here (or "which task do you mean?") carries a reply button: the answer names its task
    // only where the question is for this thread's team: never on an agent's own copy of a question it sent elsewhere
    const here = h && (h.from_team === h.to_team || String(h.to_team || '').split(':')[0] === this.kindOf(T.thread && T.thread.team));
    const replyIds = h && agent && here ? (h.kind === 'which_task' ? (h.task_ids || []) : ['clarification', 'hint'].includes(h.kind) && h.task_id ? [h.task_id] : []) : [];
    const replies = replyIds.map(id => `<button type="button" class="link-btn small" data-tm-reply="${esc(id)}">${replyIds.length > 1 ? `Answer ${esc(id)}` : 'Answer this'}</button>`).join('');
    const open = (h && h.task_id ? `<button type="button" class="link-btn small" data-tm-task="${esc(h.task_id)}">${h.kind === 'reply' ? 'Open the task and its report' : 'Open the task'}</button>` : '') + replies;
    const row = document.createElement('article');
    row.className = 'tm-msg' + (agent ? ' tm-agent' : ' tm-person');
    row.dataset.seq = String(seq);
    if (h && h.kind) row.dataset.handoff = h.kind;
    row.innerHTML = this.token(agent ? String(team).split(':')[0] : 'person', this.initials(agent ? this.label(team) : who))
      + `<div class="tm-msg-main"><div class="tm-msg-head"><b>${esc(who)}</b>${route}${h ? `<span class="pill outline">${esc(this.HANDOFF[h.kind] || h.kind)}</span>` : ''}${sim ? this.simulatedPill : ''}<time class="xs muted" datetime="${esc(m.created_at || '')}">${esc(this.when(m.created_at))}</time></div>`
      + body + (open ? `<div class="tm-msg-foot">${open}</div>` : '') + '</div>';
    const after = [...stream.children].find(c => Number(c.dataset.seq) > seq);
    stream.insertBefore(row, after || null);
    if (!after) stream.scrollTop = stream.scrollHeight;
    return true;
  },

  composeNote(el) {
    const { $ } = DC;
    const msgs = [...$('#tm-stream', el).querySelectorAll('.tm-msg')];
    const last = msgs.reverse().find(m => m.dataset.handoff);
    const note = $('#tm-compose-note', el);
    if (this.T.replyTo) { note.textContent = `Your next message answers ${this.T.replyTo}. Press Escape in the box to write a plain message instead.`; return; }
    note.textContent = last && ['clarification', 'which_task', 'hint'].includes(last.dataset.handoff)
      ? 'An agent asked a question: press "Answer this" on it (or just write, when only one question is open), and your agent takes it.'
      : 'Choose "A question for …" to ask another team; the answer comes back here. Ctrl or ⌘ + Enter sends.';
  },

  replyTo(el, taskId) {
    this.T.replyTo = taskId || null;
    this.composeNote(el);
    if (taskId) DC.$('#tm-text', el).focus();
  },

  async send(el) {
    const { $, toast } = DC;
    const T = this.T;
    const box = $('#tm-text', el), btn = $('#tm-send', el);
    const text = box.value;
    if (!text.trim() || !T.thread) { box.focus(); return; }
    const askSel = $('#tm-ask', el);
    const ask = $('#tm-ask-wrap', el).hidden ? '' : askSel.value;
    btn.disabled = true;
    const body = ask ? { text, ask } : { text };
    if (!ask && T.replyTo) body.task_id = T.replyTo;
    // the answer's identity, kept until it is sent: a resend after a failure is the same answer, recorded once
    if (!T.pending || T.pending.text !== text || T.pending.task !== (body.task_id || '') || T.pending.ask !== ask) T.pending = { text, ask, task: body.task_id || '', id: 'r' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8) };
    body.response_id = T.pending.id;
    try {
      const m = await DC.client.postTeamMessage(T.pid, T.thread.id, { body });
      box.value = ''; askSel.value = ''; T.pending = null; T.replyTo = null;
      this.addMessage(el, m);
      this.composeNote(el);
      T.thread.messages = Math.max(Number(T.thread.messages) || 0, Number(m.seq) || 0);
      this.paintInbox(el);
    } catch (e) {
      toast(e.message || 'The message was not sent.', { ok: false });
    } finally { btn.disabled = false; }
  },

  /* ---------------- the shared handoff history (the owner's form) ---------------- */
  async loadHistory(el, problemId) {
    const { $, esc } = DC;
    const box = $('#tm-history', el);
    if (!problemId) { DC.states.empty(box, 'Shared handoff history · 0'); return; }
    let h;
    try { h = await DC.client.problemHandoffs(this.T.pid, problemId, { quiet: true }); } catch (e) {
      DC.states.failed(box, e, { retry: () => this.loadHistory(el, problemId) });
      return;
    }
    const line = text => {
      const i = text.lastIndexOf(' · not delivered: ');
      return i < 0 ? esc(text) : `${esc(text.slice(0, i))} · <mark class="tm-undelivered">${esc(text.slice(i + 3))}</mark>`;
    };
    box.innerHTML = `<details class="tm-history" open><summary id="tm-history-summary">${esc(h.title || `Shared handoff history · ${h.count || 0}`)}</summary>`
      + ((h.lines || []).length ? `<ol class="tm-lines">${h.lines.map(l => `<li>${line(String(l))}</li>`).join('')}</ol>` : '<p class="small muted">Nothing has travelled between teams on this problem yet.</p>')
      + '</details>';
  },

  /* ---------------- an agent's job description ---------------- */
  async loadJob(el, name, focus) {
    const { $, esc } = DC;
    const body = $('#tm-job-body', el);
    const T = this.T;
    if (!name || T.jobFor === name) { if (focus) $('#tm-job', el).scrollIntoView({ block: 'start', behavior: 'smooth' }); return; }
    let j;
    try { j = await DC.client.teamAgentJob(name, { quiet: true }); } catch (e) {
      if (e.status === 404) { T.jobFor = name; DC.states.empty(body, 'This team has no agent job in the directory.'); return; }
      DC.states.failed(body, e, { retry: () => { T.jobFor = null; this.loadJob(el, name); } });
      return;
    }
    T.jobFor = name;
    $('#tm-job-title', el).textContent = `${this.label(name)} agent`;
    $('#tm-job-eyebrow', el).textContent = j.identity || 'Agent job';
    const acts = (j.actions || []).map(a => `<li><span>${esc(a.what)}</span><small class="muted">${esc(a.by)}${a.available ? '' : ' · not in this release'}</small></li>`).join('');
    body.innerHTML = `<p class="tm-resp"><span class="muted small">Responsible for</span><br>${esc(j.responsibility || '')}</p>`
      + `<details class="tm-fold" open><summary>Actions it may take · ${(j.actions || []).length}</summary><ul class="tm-plain">${acts}</ul></details>`
      + `<details class="tm-fold"><summary>The report it owes</summary><ul class="tm-plain">${(j.report_in_words || []).map(r => `<li>${esc(r)}</li>`).join('')}</ul></details>`
      + `<div class="tm-never"><div class="small"><b>It may never</b></div><ul class="tm-plain">${(j.may_never || []).map(r => `<li>${esc(r)}</li>`).join('')}</ul></div>`;
    if (focus) $('#tm-job', el).scrollIntoView({ block: 'start', behavior: 'smooth' });
  },

  /* ---------------- a task's page and its report ---------------- */
  closeTask(el) {
    const { $ } = DC;
    this.T.taskId = null; this.T.task = null;
    $('#tm-task', el).hidden = true; $('#tm-report', el).hidden = true;
    $('#tm-task', el).innerHTML = ''; $('#tm-report', el).innerHTML = '';
    const tid = this.T.thread && this.T.thread.id;
    if (/^#team\/task:/.test(location.hash) && tid) history.replaceState(null, '', '#team/' + tid);
  },

  async openTask(el, taskId, { focus, quiet } = {}) {
    const { $ } = DC;
    const T = this.T;
    const box = $('#tm-task', el);
    T.taskId = taskId;
    if (!quiet) { box.hidden = false; DC.states.loading(box); $('#tm-report', el).hidden = true; }
    let v;
    try { v = await DC.client.readTeamTask(T.pid, taskId, { quiet: true }); } catch (e) {
      if (T.taskId !== taskId) return;
      box.hidden = false;
      DC.states.failed(box, e, { lead: 'This task could not be opened. ', retry: () => this.openTask(el, taskId) });
      return;
    }
    if (T.taskId !== taskId) return;
    T.task = v;
    history.replaceState(null, '', '#team/task:' + encodeURIComponent(taskId));
    this.paintTask(el, v);
    this.paintReports(el, v);
    if (focus) { box.focus({ preventScroll: true }); box.scrollIntoView({ block: 'start', behavior: 'smooth' }); }
  },

  currentReport(v) {
    const live = (v.reports || []).filter(r => r.status !== 'superseded');
    return live.sort((a, b) => b.revision - a.revision)[0] || null;
  },

  paintTask(el, v) {
    const { $, esc } = DC;
    const t = v.task || {};
    const box = $('#tm-task', el);
    const acts = v.actions || [];
    const report = this.currentReport(v);
    const anySim = acts.some(a => a.status === 'simulated') || (report && report.simulated);
    const nxt = v.next_owner;
    const owes = nxt
      ? `<div class="callout info tm-next" role="status"><span><b>Next response owed by: ${esc(nxt.label)}</b>${report && report.next_task && report.next_task.unblocks ? ` · ${esc(report.next_task.unblocks)}` : ''}</span></div>`
      : `<div class="callout tm-next tm-closed" role="status"><span><b>Closed: nobody owes a response.</b></span></div>`;
    const crit = (t.done_criteria || []).map(c => `<li>${esc(c)}</li>`).join('');
    const actRows = acts.length ? acts.map(a => {
      const who = (a.responded_by && a.responded_by.id) ? `${this.label(a.actor && a.actor.id)} · answered by ${a.responded_by.id}` : this.label(a.actor && a.actor.id);
      const ref = a.output_ref || a.failure_ref || '';
      return `<li class="tm-act" data-action-status="${esc(a.status)}">${this.pill(this.ACTION, a.status)}<span class="tm-act-main"><b>${esc(this.words(a.action))}</b>`
        + `<small>${esc(who)} · ${esc(a.source)}${a.ended_at ? ' · ' + esc(this.when(a.ended_at)) : ''}</small>`
        + (ref ? `<small class="tm-ref">${esc(a.output_ref ? 'Output' : 'Failure')}: ${esc(ref)}</small>` : '')
        + (a.reason ? `<small>${esc(a.reason)}</small>` : '') + '</span></li>';
    }).join('') : '';
    const evidence = report ? (report.evidence || []) : [];
    const evRows = evidence.map(e => `<li><b>${esc(e.ref)}</b><small>${esc(e.source)} · input ${esc(e.input_revision)}${e.uncertainty ? ' · ' + esc(e.uncertainty) : ''}</small>`
      + ((e.assumptions || []).length ? `<small>Assumes: ${esc(e.assumptions.join('; '))}</small>` : '') + '</li>').join('');
    const children = v.children || [];
    const join = v.join || null;
    const remaining = [];
    if (join) (join.outstanding || []).forEach(id => { const c = children.find(x => x.id === id); remaining.push(`Waiting on ${esc(c ? c.owner_label : 'a specialist')}: <button type="button" class="link-btn small" data-tm-task="${esc(id)}">${esc(id)}</button>`); });
    if (report) (report.outstanding || []).filter(id => !(join && (join.outstanding || []).includes(id))).forEach(id => remaining.push(`Still out: ${esc(id)}`));
    if (join) (join.unresolved || []).forEach(id => remaining.push(`Returned without a result: <button type="button" class="link-btn small" data-tm-task="${esc(id)}">${esc(id)}</button>`));
    if (report && report.next_task && !report.next_task.closed && report.next_task.unblocks) remaining.push(`${esc(this.label(report.next_task.owner))}: ${esc(report.next_task.unblocks)}`);
    (v.deliveries || []).filter(d => d.status !== 'delivered').forEach(d => remaining.push(`${esc(d.from_team)} → ${esc(d.to_team)} ${esc(this.HANDOFF[d.kind] || d.kind)} not delivered: ${esc(d.status)}${d.reason ? ' (' + esc(d.reason) + ')' : ''}`));
    if (!report && !children.length && !(t.status in { complete: 1, failed: 1, cancelled: 1, superseded: 1 })) remaining.push(`${esc(t.owner_label)} has not reported yet.`);
    const childRows = children.map(c => `<li><button type="button" class="tm-inbox-item" data-tm-task="${esc(c.id)}">${this.token(String(c.owner_team).split(':')[0], this.initials(c.owner_label))}<span class="tm-inbox-main"><b>${esc(c.owner_label)}</b><small>${esc(c.question)}</small></span>${this.pill(this.TASK, c.status)}</button></li>`).join('');
    const parent = t.parent_task_id ? `<button type="button" class="link-btn small" data-tm-task="${esc(t.parent_task_id)}">Open the task it answers</button>` : '';
    box.innerHTML = `<div class="panel-head tm-head"><div class="panel-title"><div class="eyebrow">Task · ${esc(t.requester_label)} asked ${esc(t.owner_label)}</div><h2 id="tm-task-title" class="tm-q">${esc(t.question || '')}</h2></div>`
      + `<div class="tm-pills">${this.pill(this.TASK, t.status)}${anySim ? this.simulatedPill : ''}<button type="button" class="icon-btn" data-tm-close-task aria-label="Close the task">${DC.icon('close')}</button></div></div>`
      + `<div class="panel-body">${owes}`
      + `<dl class="tm-facts"><div><dt>Requested output</dt><dd>${esc(t.expected_output || '')}</dd></div><div><dt>Done when</dt><dd><ul class="tm-plain">${crit}</ul></dd></div>`
      + `<div><dt>Owner</dt><dd>${esc(t.owner_label)} <button type="button" class="link-btn small" data-tm-job="${esc(t.owner_team)}">its job</button></dd></div>`
      + `<div><dt>Asked by</dt><dd>${esc(t.requester_label)}${t.requester && t.requester.member ? ' · ' + esc(t.requester.member) : ''} · brief revision ${esc(String(t.brief_revision || ''))}</dd></div></dl>`
      + (parent ? `<p class="small">${parent}</p>` : '')
      + `<h3 class="tm-h3">Actions · ${acts.length}</h3>` + (actRows ? `<ul class="tm-acts">${actRows}</ul>` : '<p class="small muted">No action has been recorded yet.</p>')
      + `<h3 class="tm-h3">Evidence</h3>` + (evRows ? `<ul class="tm-plain tm-ev">${evRows}</ul>` : `<p class="small muted">${report ? 'The report names no evidence: nothing was measured or read for it.' : 'No evidence yet: nothing has been reported on this task.'}</p>`)
      + (children.length ? `<h3 class="tm-h3">Specialist tasks · ${children.length}</h3><ul class="tm-inbox">${childRows}</ul>` : '')
      + `<h3 class="tm-h3">Remaining work</h3>` + (remaining.length ? `<ul class="tm-plain tm-remaining">${remaining.map(r => `<li>${r}</li>`).join('')}</ul>` : '<p class="small muted">Nothing remains: the task is closed.</p>')
      + '</div>';
  },

  reportAnchor(id) { return 'tm-r-' + String(id || '').replace(/[^A-Za-z0-9_-]/g, '_'); },

  reportHtml(r, { all, child } = {}) {
    const { esc } = DC;
    const known = id => all.some(x => x.id === id);
    const link = (id, words) => known(id) ? `<button type="button" class="link-btn small" data-tm-report="${esc(id)}">${words} ${esc(id)}</button>` : `<span class="small muted">${words} ${esc(id)}</span>`;
    const lines = (list, empty) => (list || []).length ? `<ul class="tm-plain">${list.map(l => `<li>${this.pill(this.ACTION, l.status)} <b>${esc(this.words(l.action))}</b> · ${esc(this.label(l.actor))}${l.ref ? ` · ${esc(l.ref)}` : ''}${l.reason ? ` · ${esc(l.reason)}` : ''}</li>`).join('')}</ul>` : `<p class="small muted">${empty}</p>`;
    const a = r.actions || {}, n = r.next_task || {};
    const links = [r.supersedes ? link(r.supersedes, 'Replaces') : '', r.superseded_by ? link(r.superseded_by, 'Replaced by') : ''].filter(Boolean).join('');
    return `<article class="tm-report${r.status === 'superseded' ? ' tm-old' : ''}" id="${this.reportAnchor(r.id)}" tabindex="-1" data-report="${esc(r.id)}">`
      + `<div class="tm-report-head"><span class="eyebrow">${child ? esc(r.owner_label) + ' · ' : ''}Report r${esc(String(r.revision))}</span>${this.pill(this.REPORT, r.status)}${r.simulated ? this.simulatedPill : ''}</div>`
      + `<p class="tm-answer">${r.simulated ? '<span class="tm-sim-word">Simulated:</span> ' : ''}${esc(r.answer)}</p>`
      + (links ? `<div class="tm-links">${links}</div>` : '')
      + `<details class="tm-fold"><summary>Assignment and owner</summary><p class="small">${esc(r.assignment && r.assignment.question || '')}</p>`
      + `<p class="small"><b>Done when:</b> ${esc(((r.assignment && r.assignment.done_criteria) || []).join('; '))}</p><p class="small"><b>Work done by:</b> ${esc(r.owner_label || this.label(r.owner))}</p></details>`
      + `<details class="tm-fold"><summary>Actions and observations</summary><p class="small"><b>Completed</b></p>${lines(a.completed, 'None.')}`
      + `<p class="small"><b>Failed or denied</b></p>${lines(a.failed, 'None.')}<p class="small"><b>Skipped</b></p>${lines(a.skipped, 'None.')}</details>`
      + `<details class="tm-fold"><summary>Evidence and limits</summary>${(r.evidence || []).length ? `<ul class="tm-plain">${r.evidence.map(e => `<li><b>${esc(e.ref)}</b> · ${esc(e.source)} · input ${esc(e.input_revision)}${e.uncertainty ? ' · ' + esc(e.uncertainty) : ''}</li>`).join('')}</ul>` : '<p class="small muted">No evidence.</p>'}`
      + `${(r.limits || []).length ? `<p class="small"><b>Limits</b></p><ul class="tm-plain">${r.limits.map(l => `<li>${esc(l)}</li>`).join('')}</ul>` : ''}<p class="small muted">Answers brief revision ${esc(String(r.brief_revision))}.</p></details>`
      + `<details class="tm-fold"><summary>Next task</summary><p class="small">${n.closed ? 'Closed: nothing is owed.' : `${esc(this.label(n.owner))} acts next${n.unblocks ? ': ' + esc(n.unblocks) : '.'}`}</p>`
      + `${(r.outstanding || []).length ? `<p class="small">Still out: ${esc(r.outstanding.join(', '))}</p>` : ''}</details>`
      + (child ? `<div class="tm-msg-foot"><button type="button" class="link-btn small" data-tm-task="${esc(r.task_id)}">Open ${esc(r.owner_label)}'s task</button></div>` : '')
      + '</article>';
  },

  paintReports(el, v) {
    const { $, esc } = DC;
    const box = $('#tm-report', el);
    const reports = (v.reports || []).slice().sort((a, b) => b.revision - a.revision);
    const kids = v.child_reports || [];
    if (!reports.length && !kids.length) { box.hidden = true; box.innerHTML = ''; return; }
    const all = reports.concat(kids);
    const current = reports.filter(r => r.status !== 'superseded'), older = reports.filter(r => r.status === 'superseded');
    box.hidden = false;
    box.innerHTML = `<div class="panel-head tm-head"><div class="panel-title"><div class="eyebrow">The answer first; open a part for its details</div><h2 id="tm-report-title">Report</h2></div></div><div class="panel-body">`
      + (current.length ? current.map(r => this.reportHtml(r, { all })).join('') : '<p class="small muted">Every revision of this report has been replaced.</p>')
      + (older.length ? `<details class="tm-fold tm-older"><summary>Earlier revisions · ${older.length}</summary>${older.map(r => this.reportHtml(r, { all })).join('')}</details>` : '')
      + (kids.length ? `<h3 class="tm-h3">Specialist reports joined · ${kids.length}</h3>${kids.map(r => this.reportHtml(r, { all, child: true })).join('')}` : '')
      + '</div>';
  },

  jumpToReport(el, id) {
    const target = DC.$('#' + this.reportAnchor(id), el);
    if (!target) return;
    let d = target.closest('details:not([open])');
    while (d) { d.open = true; d = d.parentElement && d.parentElement.closest('details:not([open])'); }
    target.scrollIntoView({ block: 'center', behavior: 'smooth' });
    target.focus({ preventScroll: true });
  },
});
