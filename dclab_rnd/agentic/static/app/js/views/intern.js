DC.view('intern', {
  init(el) {
    const { $, $$, esc, chip, chips, icon, codeBlock, toast, int } = DC;
    const SESS = [
      ['HyperAck: an acceptance model we can trust', 'paused · question for you', 'warn', true],
      ['Card fraud: does hour of day help?', 'done · 14 calls · PR-AUC kept', 'ok'],
      ['Telco churn: rerun after the solution is fixed', 'waiting on the solution', 'warn'],
      ['Bike demand: beat the lag-2 baseline', 'done · MAE 680.0 vs 1,156.4', 'ok'],
      ['Review a notebook from Ava', 'done · 10 findings', 'ok'],
    ];
    $('#intern-sessions', el).innerHTML = SESS.map(([t, s, c, on]) => `<button type="button" class="list-item" ${on ? 'data-style="background:var(--accent-soft)" data-pane-go="overview"' : 'data-toast="In the demo only the HyperAck session opens."'}><div class="li-main"><span class="li-title">${esc(t)}</span><span class="li-sub"><span class="dot ${c}"></span> ${esc(s)}</span></div></button>`).join('');

    const TOOL = (name, sum, body, t, extra = '') => ({ k: 'tool', name, sum, body, t, extra });
    const PRE = [
      { k: 'user', text: 'Build an order-acceptance model for HyperAck that we can trust. Use what the R&D already learned. Ask me before anything irreversible.' },
      { k: 'agent', text: 'I will start from the evidence, draft the solution, and stop at the first thing only you can decide. Plan:' , plan: true },
      TOOL('search_evidence', '"HyperAck leakage final fare" · 3 records', `<div class="small">Found ${chips(['LEAK-hyperack', 'PIT-005', 'DATASET-hyperack'])}. Final fares are post-outcome; with them ROC-AUC reaches 0.9802, without 0.9455. On this data a forward-in-time test scored 0.9371 against 0.9332 for a random split.</div>`, '0.4 s'),
      TOOL('create_project · use_sample', 'hyperack · 11,118 orders · 2022-06-29 → 2022-11-14', `<pre class="term"><span class="t-dim">rows</span> 11,118   <span class="t-dim">columns</span> 14   <span class="t-dim">target</span> hyper_ack (0/1)
<span class="t-dim">time column</span> created_at   <span class="t-dim">range</span> 2022-06-29 → 2022-11-14</pre>`, '1.2 s'),
      TOOL('propose_solution', 'forbidden: 2 · review: 2 · split by time', `<div class="small stack tight"><div><span class="pill bad">forbidden</span> <code>final_customer_fare</code>, <code>final_biker_fare</code> · known after the dispatch outcome ${chip('LEAK-hyperack')}</div><div><span class="pill warn">review</span> <code>first_created_at</code> (timestamp, becomes calendar features) · <code>deliverey_category_id</code> (strong signal, legitimate?)</div><div><span class="pill accent">split</span> by time: last 20% of orders as holdout ${chip('PIT-005')}</div></div>`, '0.9 s'),
      { k: 'ask' },
    ];
    function POSTFOR(a) {
      const solution = a === 'create'
        ? TOOL('set_solution', 'solution v1 signed · prediction at order creation', `<div class="small">Same moment as the R&amp;D solution ${chip('DATASET-hyperack')}. Courier-side columns are not allowed. Final fares stay forbidden.</div>`, '0.2 s')
        : TOOL('set_solution', 'solution v1 signed · prediction at the courier offer', '<div class="small">Courier distance and offer price become allowed; anything written after the offer stays forbidden. This is a new solution: the R&amp;D has not run it yet.</div>', '0.2 s');
      return [
        solution,
        TOOL('run_stage · WF-02 → WF-05', 'lineage ok · split by time · train-only profile · leakage audit', `<div class="small">Final fares stay blocked ${chip('LEAK-hyperack')}. The canary was caught. <code>deliverey_category_id</code> stays flagged for your review.</div>`, '6.8 s'),
        TOOL('sandbox.run', 'calendar features built inside time-ordered folds', `${codeBlock(`hour = df.created_at.dt.hour
weekday = df.created_at.dt.weekday
# cyclic encoding; nothing is learned from the target
df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
df["is_weekend"] = weekday >= 5`)}<pre class="term" data-style="margin-top:6px"><span class="t-ok">✓</span> 3 features added · <span class="t-dim">sandbox sbx-41c · 2 vCPU · no network · 3.1 s</span></pre>`, '3.1 s', 'data-f="intern.sandbox"'),
        TOOL('run_stage · WF-06 → WF-08', 'recipe, model and tuning chosen by the declared rules', '<div class="small">All comparisons on the same time-ordered training folds. Tuning is kept only if it clears the declared margin. Details are in the stage records.</div>', '41 s'),
        { k: 'blocked' },
        { k: 'approve' },
      ];
    }
    function FINALFOR(a) {
      return a === 'create'
        ? [TOOL('open_holdout', 'approved by Shahriyar · used once', `<div class="small">Last 20% of orders by time (from 2022-10-16): <b class="mono">ROC-AUC 0.9371</b>, the protocol of the R&amp;D's forward-in-time test ${chip('PIT-005')}.</div>`, '2.2 s'), { k: 'report' }]
        : [TOOL('open_holdout', 'approved by Shahriyar · used once', '<div class="small">In the product the score appears here. This solution has no R&amp;D run yet, so the demo shows no number rather than an invented one.</div>', '2.2 s'), { k: 'report' }];
    }
    let POST = POSTFOR('create'), FINAL = FINALFOR('create');
    let phase = 'pre', answered = null, steps = 9, minutes = 3.1, spend = 0.42;
    const tin = { pre: '18,240', post: '31,880', final: '36,410' }, tout = { pre: '3,120', post: '5,470', final: '6,960' };

    function msgUser(t) { return `<div class="msg user"><span class="avatar">SM</span><div class="msg-body"><span class="who">Shahriyar <span>Oct 2, 15:10</span></span><div class="bubble">${esc(t)}</div></div></div>`; }
    function msgAgent(html) { return `<div class="msg"><span class="avatar ai">AI</span><div class="msg-body"><span class="who">Intern</span>${html}</div></div>`; }
    function planHtml() {
      const P = [['Check the evidence for HyperAck', 'done'], ['Load and profile the data', 'done'], ['Draft the solution', 'done'], ['Ask the owner what only the owner knows', answered ? 'done' : 'now'], ['Run WF-02 to WF-08 inside the graph', phase === 'final' || phase === 'done' ? 'done' : phase === 'post' ? 'now' : ''], ['Ask before opening the holdout', phase === 'done' ? 'done' : phase === 'final' ? 'now' : ''], ['Write the brief and the report', phase === 'done' ? 'done' : '']];
      return `<ul class="plan-list">${P.map(([t, s]) => `<li class="${s}"><span class="pi">${s === 'done' ? '✓' : ''}</span><span class="t">${esc(t)}</span></li>`).join('')}</ul>`;
    }
    function toolHtml(s, open) {
      return `<details class="tool-call reveal" ${open ? 'open' : ''} ${s.extra}><summary><span class="tname">${esc(s.name)}</span><span class="tsum">${esc(s.sum)}</span><span class="ttime">${esc(s.t)}</span></summary><div class="tc-body">${s.body}</div></details>`;
    }
    function askHtml() {
      return `<div class="ask-card reveal" data-f="intern.ask"><div class="spread"><span class="q">When is the HyperAck prediction made?</span><span class="pill warn">blocks WF-01</span></div>
        <div class="small">I cannot decide this from the data. It changes which columns are allowed.</div>
        <div class="ask-options">
          <button type="button" class="ask-option" data-answer="create" aria-pressed="${answered === 'create'}"><span><b>At order creation.</b> Before any courier sees it.</span></button>
          <button type="button" class="ask-option" data-answer="offer" aria-pressed="${answered === 'offer'}"><span><b>When the order is offered to a courier.</b> Distance and offer price become allowed.</span></button>
          <button type="button" class="ask-option" data-answer="assign" aria-pressed="${answered === 'assign'}"><span><b>After a courier is assigned.</b> Acceptance is then mostly known.</span></button>
        </div>
        <label class="check small"><input type="checkbox" checked> Sign solution v1 with this answer</label></div>`;
    }
    function blockedHtml() {
      return `<details class="tool-call blocked reveal" open><summary><span class="tname">open_holdout</span><span class="tsum">blocked by the validator</span><span class="ttime">0.0 s</span></summary><div class="tc-body small">The move exists, but its gate needs the owner's approval. Nothing was read. ${chips(['DCLAB-R17'])}</div></details>`;
    }
    function approveHtml() {
      const done = phase === 'final' || phase === 'done';
      return `<div class="ask-card reveal"><div class="spread"><span class="q">May I open the holdout? It can be opened once.</span><span class="pill warn">approval gate</span></div>
        <div class="small">Locked: solution v1, feature recipe, model and tuning decision. The 2,222 orders from 2022-10-16 on have never been read.</div>
        <div class="row"><button type="button" class="btn sm primary" data-approve="yes" ${done ? 'disabled' : ''}>Approve once</button><button type="button" class="btn sm" data-approve="no" ${done ? 'disabled' : ''}>Not yet</button>${done ? '<span class="pill ok">approved</span>' : ''}</div></div>`;
    }
    function reportHtml() {
      const body = answered === 'create'
        ? `<div>Solution v1 is signed: prediction at order creation, as the R&amp;D assumed. Final fares are blocked ${chip('LEAK-hyperack')}.</div>
           <div>On the last 20% of orders by time the protocol scores <b>ROC-AUC 0.9371</b> ${chip('PIT-005')}; the leakage-safe cross-validated ensemble reached 0.9455 ${chip('LEAK-hyperack')}. One time-ordered holdout; the interval is in the full report.</div>`
        : `<div>Solution v1 is signed: prediction at the courier offer. The R&amp;D has not run this solution yet, so I report no score until the stage records exist.</div>
           <div>Next: collect courier-side columns with their timestamps, then rerun WF-02 to WF-09.</div>`;
      return msgAgent(`<div class="inset stack tight reveal"><b>Report</b>${body}
        <div>Not proven: behaviour after November 2022, and whether <code>deliverey_category_id</code> is legitimate. I flagged it for your review.</div>
        <div class="row"><a class="btn sm" href="#project" data-project="hyperack">Open the project</a><a class="btn sm" href="#brief">See a finished brief</a></div></div>`);
    }
    function render() {
      if (R.mode !== 'sample') return;  // a real session owns the thread
      let h = msgUser(PRE[0].text) + msgAgent(`<div>${esc(PRE[1].text)}</div>${planHtml()}`);
      PRE.slice(2, 5).forEach(s => { h += toolHtml(s, false); });
      h += askHtml();
      if (pushback && !answered) h += msgUser('After a courier is assigned.') + msgAgent(`<div class="reveal">At that moment acceptance is mostly decided, so the model would predict something we already know. I suggest order creation (what the R&amp;D assumed) or the courier offer. Which one? ${chip('DCLAB-R01')}</div>`);
      if (answered) {
        h += msgUser({ create: 'At order creation.', offer: 'When the order is offered to a courier.' }[answered]);
        POST.slice(0, revealed).forEach(s => { h += s.k === 'blocked' ? blockedHtml() : s.k === 'approve' ? approveHtml() : toolHtml(s, s.name === 'sandbox.run'); });
      }
      if (phase === 'final' || phase === 'done') FINAL.slice(0, finalRevealed).forEach(s => { h += s.k === 'report' ? reportHtml() : toolHtml(s, true); });
      $('#thread', el).innerHTML = h;
      DC.hydrate(el);
      drawSide();
    }
    let revealed = 0, finalRevealed = 0, timer = null, pushback = false;
    function drawSide() {
      const st = { pre: ['paused · waiting for you', 'warn'], post: [revealed >= POST.length ? 'paused · needs approval' : 'working…', revealed >= POST.length ? 'warn' : 'accent'], final: ['working…', 'accent'], done: ['done', 'ok'] }[phase];
      $('#sess-status', el).textContent = st[0]; $('#sess-status', el).className = 'pill ' + st[1];
      const meter = (label, used, max, unit, cls) => `<div class="meter-row"><span>${label}</span><span class="mono">${used}${unit} / ${max}${unit}</span><div class="meter ${cls}"><span data-style="width:${Math.min(100, (parseFloat(String(used).replace(/,/g, '')) / max) * 100)}%"></span></div></div>`;
      $('#budget', el).innerHTML = meter('Tool calls', steps, 48, '', '') + meter('Minutes', minutes.toFixed(1), 30, '', '') + meter('Spend', '€' + spend.toFixed(2), 5, '', 'proof').replace('€' + spend.toFixed(2) + ' / 5', '€' + spend.toFixed(2) + ' / €5') +
        `<div class="small muted">Tokens: ${tin[phase === 'done' ? 'final' : phase === 'pre' ? 'pre' : phase === 'final' ? 'final' : 'post']} in · ${tout[phase === 'done' ? 'final' : phase === 'pre' ? 'pre' : phase === 'final' ? 'final' : 'post']} out</div>`;
      const pos = phase === 'pre' ? 0 : phase === 'post' ? Math.min(7, 1 + revealed * 2) : phase === 'final' ? 8 : 9;
      $('#intern-rail', el).innerHTML = DEMO.wf.map(([id, n], i) => `<div class="wf-step ${i < pos ? 'done' : i === pos ? (phase === 'pre' ? 'blocked' : 'current') : 'todo'}" data-style="min-height:58px;padding:6px 7px"><span class="wf-id">${id}</span><span class="wf-name" data-style="font-size:11px">${['Solution', 'Source', 'Split', 'EDA', 'Leakage', 'Features', 'Screen', 'Tuning', 'Holdout', 'Capture'][i]}</span></div>`).join('');
      const L = phase === 'pre' ? [21, 4, 3] : phase === 'done' ? [31, 6, 4] : [27, 5, 4];
      const tot = L[0] + L[1] + L[2];
      $('#ladder', el).innerHTML = [['Typed classifier', 'is this an ID? a timestamp? a target proxy?', L[0], ''], ['NOOA structured call', 'fill a solution field from evidence', L[1], 'proof'], ['Full model', 'plan, ask, explain', L[2], 'warn']]
        .map(([n, d, c, cls]) => `<div class="meter-row"><span><b>${n}</b> · <span class="muted">${d}</span></span><span class="mono">${c}</span><div class="meter ${cls}"><span data-style="width:${(c / tot) * 100}%"></span></div></div>`).join('');
    }
    function advance() {
      clearInterval(timer);
      timer = setInterval(() => {
        if (phase === 'post' && revealed < POST.length) { revealed++; steps += revealed === 3 ? 2 : 1; minutes += 0.6; spend += 0.05; render(); if (revealed >= POST.length) clearInterval(timer); }
        else if (phase === 'final' && finalRevealed < FINAL.length) { finalRevealed++; steps += 1; minutes += 0.4; spend += 0.04; if (finalRevealed >= FINAL.length) { phase = 'done'; clearInterval(timer); toast('Session finished inside its budget.'); } render(); }
        else clearInterval(timer);
      }, 900);
    }
    el.addEventListener('click', e => {
      const a = e.target.closest('[data-answer]');
      if (a && phase === 'pre') {
        if (a.dataset.answer === 'assign') { pushback = true; render(); return; }
        answered = a.dataset.answer; pushback = false; POST = POSTFOR(answered); FINAL = FINALFOR(answered); phase = 'post'; revealed = 0; render(); toast('Answer recorded and solution v1 signed. The intern continues.'); advance(); return;
      }
      const ap = e.target.closest('[data-approve]');
      if (ap && phase === 'post' && revealed >= POST.length) {
        if (ap.dataset.approve === 'yes') { phase = 'final'; finalRevealed = 0; render(); advance(); }
        else toast('Not yet. The intern waits; the holdout stays sealed.');
        return;
      }
      const c = e.target.closest('[data-cmd]'); if (c) { $('#intern-follow', el).value = c.dataset.cmd + ' '; $('#intern-follow', el).focus(); }
      const p = e.target.closest('[data-project]'); if (p) { DC.state.project = p.dataset.project; if (p.dataset.project.startsWith('p:')) DC.currentProject.clear(); }
      const sb = e.target.closest('[data-session]');
      if (sb) { DC.state.internSession = sb.dataset.session; R.sig = ''; DC.selectPane(el, 'overview'); window.scrollTo({ top: 0, behavior: 'smooth' }); sync(); }
    });
    $('#sess-replay', el).addEventListener('click', () => { clearInterval(timer); phase = 'pre'; answered = null; pushback = false; revealed = 0; finalRevealed = 0; steps = 9; minutes = 3.1; spend = 0.42; render(); });
    $('#intern-send', el).addEventListener('click', send);
    $('#intern-model', el).addEventListener('change', e => {
      if (R.mode === 'real') { toast('The model is set on the server (OPENAI_MODEL and OPENAI_BASE_URL). Keys never reach this page.'); modelReal(); return; }
      const m = { router: 'open model · via Hugging Face router', local: 'local model · Ollama on this machine', standard: 'no model · standard plan', policy: 'DCLab policy model · planned' }[e.target.value];
      $('#sess-mode', el).textContent = m;
      toast(e.target.value === 'policy' ? 'The policy model is planned (phase 4). Its training data is collected from logs like this one.' : 'Model changed for the next turn. Keys stay on the server.');
    });

    /* ---------------- real sessions ----------------
       Real mode: at least one intern session exists (or one is being started). The Overview shows the selected
       session (DC.state.internSession, else the most recent), polled every 2 s while it works. Sample mode keeps the
       scripted HyperAck conversation above. */
    const R = { mode: 'sample', sessions: [], cur: null, proj: null, projKey: '', info: null, timer: null, seq: 0, kick: 0, sig: '', openFor: null, seen: {}, follow: {}, starting: false, busy: false };
    const STATUS = { queued: ['queued', 'accent', 'live'], running: ['working…', 'accent', 'live'], completed: ['done', 'ok', 'ok'], failed: ['failed', 'bad', 'bad'], budget_exhausted: ['budget used up', 'warn', 'warn'], stopped: ['stopped', 'warn', 'warn'] };
    const isLive = s => !!s && (s.status === 'queued' || s.status === 'running');
    const stOf = s => STATUS[s.status] || [String(s.status || 'unknown'), '', ''];
    const statsEl = $('.pane[data-pane="overview"] .stats', el);
    const sessHead = $('.pane[data-pane="sessions"] .panel-head', el);
    const railLink = $('[data-f="intern.graphpos"] a[data-project]', el);
    const SAMPLE = {
      stats: statsEl.innerHTML, sessions: $('#intern-sessions', el).innerHTML, sessSub: $('.sub', sessHead).textContent, sessPill: $('.pill', sessHead).textContent,
      mode: $('#sess-mode', el).textContent, title: $('#sess-title', el).textContent, placeholder: $('#intern-follow', el).placeholder,
      ladderPill: $('[data-f="intern.ladder"] .pill', el).textContent, nav: null,
    };
    const when = iso => { const d = new Date(iso); return iso && !isNaN(d) ? d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }) : ''; };
    const clip = (t, n) => (t.length > n ? t.slice(0, n) + '… [' + (t.length - n) + ' more characters]' : t);
    const short = (t, n = 120) => { t = String(t == null ? '' : t).replace(/\s+/g, ' ').trim(); return t.length > n ? t.slice(0, n - 1) + '…' : t; };
    const grab = (text, key) => { const m = new RegExp('"' + key + '":\\s*(\\d+|"[^"]*")').exec(text || ''); return m ? m[1].replace(/^"|"$/g, '') : null; };

    function realUser(text, iso) { return `<div class="msg user"><span class="avatar">SM</span><div class="msg-body"><span class="who">You <span>${esc(when(iso))}</span></span><div class="bubble">${esc(text)}</div></div></div>`; }
    /* One line for a tool call: what it was asked, or what came back. The full arguments and result are in its body. */
    function shortSum(st) {
      const a = st.arguments || {}, r = st.summary || '';
      if (st.ok === false) return short(r.replace(/^error:\s*/, ''), 140);
      const rows = grab(r, 'rows'), cols = grab(r, 'column_count'), shape = rows ? `${int(rows)} rows${cols ? ' · ' + cols + ' columns' : ''}` : '';
      const f = {
        search_evidence: () => `"${a.query || ''}"`, get_record: () => a.record_id || a.id || '', get_rules: () => 'the R&D rules',
        list_samples: () => 'the datasets the R&D already studied', create_project: () => a.name || '', use_sample: () => [a.key, shape].filter(Boolean).join(' · '),
        describe_data: () => shape || 'the project table', propose_solution: () => `target ${a.target || '?'}${a.task ? ' · ' + a.task : ''}`,
        set_solution: () => `target ${a.target || '?'} · ${a.task || ''} · ${(a.forbidden || []).length} forbidden`, set_settings: () => (a.quick === false ? 'all rows (up to max_rows)' : a.quick ? 'quick mode' : '') + (a.max_rows ? ` · max ${int(a.max_rows)} rows` : ''),
        run_stage: () => `${a.stage || ''}${grab(r, 'title') ? ' · ' + grab(r, 'title') : ''}`, run_all: () => 'every remaining stage, in order',
        approve_stage: () => `${a.stage || ''}${a.choice ? ' · choice ' + a.choice : ''}`, get_results: () => a.stage || 'every finished stage',
        ask_project: () => `"${a.question || ''}"`, export_notebook: () => 'notebook and report written', get_graph: () => 'where the project is in the graph',
        check_move: () => `${a.move || ''}${a.stage ? ' · ' + a.stage : ''}${a.gate ? ' · ' + a.gate : ''}`, write_plan: () => r,
      }[st.tool];
      const text = f ? f() : Object.entries(a).filter(([k]) => k !== 'project_id').map(([k, v]) => `${k} ${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' · ');
      return short(text || r, 140);
    }
    function realStep(st, fresh) {
      const t = st.elapsed_seconds != null ? Number(st.elapsed_seconds).toFixed(1) + ' s' : '';
      const res = st.result != null ? (typeof st.result === 'string' ? st.result : JSON.stringify(st.result, null, 1)) : (st.summary || '');
      const bad = st.ok === false;
      return `<details class="tool-call${bad ? ' blocked' : ''}${fresh ? ' reveal' : ''}" data-step="${esc(st.n)}"><summary><span class="tname">${esc(st.tool)}</span><span class="tsum">${esc(shortSum(st))}</span><span class="ttime">${esc(t)}</span></summary>
        <div class="tc-body"><pre class="term" data-style="white-space:pre-wrap;overflow-wrap:anywhere"><span class="t-dim">arguments</span>\n${esc(clip(JSON.stringify(st.arguments || {}, null, 1), 1200))}</pre>
        <pre class="term" data-style="white-space:pre-wrap;overflow-wrap:anywhere"><span class="${bad ? 't-bad' : 't-dim'}">${bad ? 'error' : 'result'}</span>\n${esc(clip(res, 1200))}</pre></div></details>`;
    }
    /* The plan as a checklist. In the standard plan each item maps to its tools, so done / now is known; a model's own plan is shown without marks. */
    const PLAN_IDX = { list_samples: 0, create_project: 1, use_sample: 1, describe_data: 1, propose_solution: 2, set_solution: 2, set_settings: 3, run_stage: 3, run_all: 3, approve_stage: 3, export_notebook: 3 };
    const PLAN_END = { list_samples: 0, describe_data: 1, set_solution: 2, export_notebook: 3 };
    function realPlan(s) {
      const lines = String(s.plan || '').split('\n').map(l => l.trim()).filter(Boolean);
      if (!lines.length) return '';
      const item = /^(?:\d+[.)]|[-*•])\s+/;
      const intro = lines.filter(l => !item.test(l)).join(' '), items = lines.filter(l => item.test(l)).map(l => l.replace(item, ''));
      if (!items.length) return `<div>${DC.linkIds(lines.join('\n')).replace(/\n/g, '<br>')}</div>`;
      // A step is logged when its call returns, so after the last call of an item the next item is the one under way.
      let cur = -1, bad = -1;
      if (s.mode !== 'llm') {
        const steps = s.steps || [], last = steps[steps.length - 1];
        let reached = -1;
        steps.forEach(st => { if (PLAN_IDX[st.tool] != null) reached = Math.max(reached, PLAN_IDX[st.tool]); });
        cur = reached;
        if (isLive(s) && (reached < 0 || (last && PLAN_END[last.tool] === reached))) cur = reached + 1;
        if (steps.some(st => st.tool === 'export_notebook')) cur = items.length;
        if (!isLive(s) && last && last.ok === false && PLAN_IDX[last.tool] != null) bad = PLAN_IDX[last.tool];
      }
      const mark = i => (s.mode === 'llm' || cur < 0 ? '' : i === bad ? 'blocked' : i < cur ? 'done' : i > cur ? '' : isLive(s) ? 'now' : s.status === 'completed' ? 'done' : 'blocked');
      return `<div>${esc(intro || 'Plan:')}</div><ul class="plan-list">${items.map((t, i) => { const c = mark(i); return `<li class="${c}"><span class="pi">${c === 'done' ? '✓' : ''}</span><span class="t">${esc(t)}</span></li>`; }).join('')}</ul>`;
    }
    function realReport(text, s, title = 'Report') {
      const pid = s.project_id;
      const body = String(text).split('\n').filter(l => l.trim()).map(l => {
        const sub = /^\s+[·•-]/.test(l);
        const html = DC.linkIds(l.trim()).replace(/#project\/([0-9a-f]{12})/g, (m, id) => `<a class="link-btn" href="#project" data-project="p:${id}">project ${id}</a>`);
        return sub ? `<div class="small" data-style="padding-left:14px">${html}</div>` : `<div>${html}</div>`;
      }).join('');
      const links = pid && title === 'Report' ? `<div class="row"><a class="btn sm" href="#project" data-project="p:${esc(pid)}">Open the project</a><a class="btn sm" href="#brief" data-project="p:${esc(pid)}">Open its decision brief</a></div>` : '';
      return msgAgent(`<div class="inset stack tight"><b>${esc(title)}</b>${body}${links}</div>`);
    }
    /* Follow-ups the person sent. The server keeps no transcript of them, so the ones sent from this page are remembered here
       (with the report they replaced); in the standard plan each follow-up is also one ask_project / search_evidence call with the text. */
    function followUps(s) {
      const mem = R.follow[s.id] || [];
      if (s.mode === 'llm') return mem;
      const derived = (s.steps || []).map((st, i) => {
        const a = st.arguments || {}, text = st.tool === 'ask_project' ? a.question : st.tool === 'search_evidence' ? a.query : null;
        return text && !mem.some(m => m.at === i) ? { at: i, text, prevFinal: null, when: st.at } : null;
      }).filter(Boolean);
      return mem.concat(derived).sort((x, y) => x.at - y.at);
    }
    function realThread(s) {
      const fol = followUps(s), steps = s.steps || [], seen = R.seen[s.id];
      let h = realUser(s.task, s.created);
      const plan = realPlan(s);
      if (plan) h += msgAgent(plan);
      else if (isLive(s)) h += msgAgent('<div class="small muted"><span class="dot live"></span> Reading the task…</div>');
      let fi = 0;
      const follow = (f, k) => (f.prevFinal ? realReport(f.prevFinal, s, k ? 'Answer' : 'Report') : '') + realUser(f.text, f.when);
      steps.forEach((st, i) => { while (fi < fol.length && fol[fi].at <= i) { h += follow(fol[fi], fi); fi++; } h += realStep(st, seen != null && i >= seen); });
      while (fi < fol.length) { h += follow(fol[fi], fi); fi++; }
      if (s.error) h += `<div class="callout bad"><span class="ic">${icon('alert')}</span><span><b>The session stopped with an error.</b> ${esc(s.error)}</span></div>`;
      else if (s.status === 'budget_exhausted') h += `<div class="callout warn"><span class="ic">${icon('alert')}</span><span><b>The budget is used up.</b> The intern stopped at its limit of ${esc(s.budget.max_steps)} tool calls or ${esc(s.budget.max_minutes)} minutes; the work so far is kept.</span></div>`;
      if (s.final) h += realReport(s.final, s, fol.length ? 'Answer' : 'Report');
      if (isLive(s)) h += `<div class="small muted"><span class="dot live"></span> ${steps.length ? 'working…' : 'Starting…'}</div>`;
      return h;
    }

    function meterReal(label, used, max, text) {
      const pc = max ? Math.min(100, (used / max) * 100) : 0;
      return `<div class="meter-row"><span>${label}</span><span class="mono">${text}</span><div class="meter ${pc >= 100 ? 'bad' : pc >= 80 ? 'warn' : ''}"><span data-style="width:${pc.toFixed(1)}%"></span></div></div>`;
    }
    const RAIL = { done: 'done', current: 'current', running: 'current', waiting: 'blocked', failed: 'blocked', todo: 'todo' };
    const RAIL_TEXT = { done: 'done', current: 'next step', running: 'running', waiting: 'waiting for a person', failed: 'failed', todo: 'not yet' };
    const SHORT = ['Solution', 'Source', 'Split', 'EDA', 'Leakage', 'Features', 'Screen', 'Tuning', 'Holdout', 'Capture'];
    function modelReal() {
      const sel = $('#intern-model', el), i = R.info;
      let opt = $('option[value="configured"]', sel);
      if (R.mode === 'real' && i && i.mode === 'llm') {
        if (!opt) { opt = document.createElement('option'); opt.value = 'configured'; sel.prepend(opt); }
        opt.textContent = `Model · ${i.model} via ${i.endpoint}`; sel.value = 'configured';
      } else {
        if (opt) opt.remove();
        sel.value = R.mode === 'real' && i ? 'standard' : 'router';
      }
    }
    function sideReal(s, p) {
      const st = stOf(s), used = s.used || {}, b = s.budget || {};
      $('#sess-status', el).textContent = st[0]; $('#sess-status', el).className = 'pill ' + st[1];
      $('#sess-mode', el).textContent = s.mode === 'llm' ? 'model · ' + (s.model || (R.info && R.info.model) || 'set on the server') : 'no model · standard plan';
      $('#sess-title', el).textContent = short(s.task, 90);
      const tok = s.mode === 'llm' ? `Tokens: ${int(used.input_tokens || 0)} in · ${int(used.output_tokens || 0)} out` : 'Tokens: none · the standard plan calls no model';
      $('#budget', el).innerHTML = meterReal('Tool calls', used.steps || 0, b.max_steps, `${used.steps || 0} / ${b.max_steps}`) +
        meterReal('Minutes', used.minutes || 0, b.max_minutes, `${Number(used.minutes || 0).toFixed(1)} / ${b.max_minutes}`) + `<div class="small muted">${esc(tok)}</div>`;
      // Where the session's project is in the workflow graph.
      const rail = $('#intern-rail', el), nodes = (p && p.graph && p.graph.nodes) || [];
      railLink.hidden = !s.project_id;
      if (s.project_id) railLink.dataset.project = 'p:' + s.project_id;
      rail.innerHTML = !s.project_id ? '<div class="small muted" data-style="grid-column:1/-1">No project yet. The intern creates one when it loads the data.</div>'
        : !nodes.length ? `<div class="small muted" data-style="grid-column:1/-1">Project ${esc(s.project_id)} is not in this workspace any more.</div>`
        : nodes.map((n, i) => `<div class="wf-step ${RAIL[n.state] || 'todo'}" data-style="min-height:58px;padding:6px 7px" title="${esc(n.id + ' ' + n.name + ': ' + (RAIL_TEXT[n.state] || n.state))}"><span class="wf-id">${esc(n.id)}</span><span class="wf-name" data-style="font-size:11px">${esc(SHORT[i] || n.name)}</span></div>`).join('');
      // Model tiers: only what the session measured.
      $('[data-f="intern.ladder"] .pill', el).textContent = 'this session';
      $('#ladder', el).innerHTML = [['Deterministic tools', 'every stage, audit and evidence lookup', `${int(used.steps || 0)} calls`],
        ['Typed classifier', 'is this an ID? a timestamp? a target proxy?', 'not logged per call yet'], ['NOOA structured call', 'fill a solution field from evidence', 'not logged per call yet'],
        ['Full model', 'plan, ask, explain', s.mode === 'llm' ? `${int((used.input_tokens || 0) + (used.output_tokens || 0))} tokens` : 'none · no model configured']]
        .map(([n, d, v]) => `<div class="meter-row"><span><b>${n}</b> · <span class="muted">${d}</span></span><span class="mono">${esc(v)}</span></div>`).join('');
      // The four numbers on the Overview.
      const failed = (s.steps || []).filter(x => x.ok === false).length, all = R.sessions, n = all.length;
      const done = all.filter(x => x.status === 'completed').length, live = all.filter(isLive).length, other = n - done - live;
      const parts = [`${done} done`].concat(live ? [`${live} working`] : [], other ? [`${other} stopped or failed`] : []);
      let fourth;
      if (p && p.graph) {
        const uses = Number(p.holdout_uses || 0), gate = (p.graph.gates || []).find(g => g.gate === 'holdout');
        fourth = `<span class="v ${uses > 1 ? 'bad' : 'ok'}">${uses}</span><span class="l">holdout use${uses === 1 ? '' : 's'} in this session's project · ${gate && gate.required ? 'it opens once, after the owner approves' : 'it opens once; owner approval is off for this project'}</span>`;
      } else fourth = `<span class="v ok">${$$('[data-f="intern.guard"] .row.top', el).length}</span><span class="l">guardrails active · the same limits a person works under</span>`;
      statsEl.innerHTML = `<div class="stat"><span class="v ${failed ? 'warn' : ''}">${int(used.steps || 0)}</span><span class="l">tool calls in this session · ${failed ? failed + ' failed or blocked' : 'none failed or blocked'}</span></div>
        <div class="stat"><span class="v">${n}</span><span class="l">session${n === 1 ? '' : 's'} · ${parts.join(', ')}</span><div class="row"><button type="button" class="link-btn small" data-pane-go="sessions">See all sessions →</button></div></div>
        <div class="stat"><span class="v">${esc(b.max_steps)} <small>calls · ${esc(b.max_minutes)} min</small></span><span class="l">budget for this session · enforced on every call</span></div>
        <div class="stat">${fourth}</div>`;
      // Sessions tab, its count, and the sidebar count (sessions working now).
      $('#intern-sessions', el).innerHTML = all.map(x => {
        const xs = stOf(x), on = x.id === s.id;
        return `<button type="button" class="list-item" data-session="${esc(x.id)}" ${on ? 'data-style="background:var(--accent-soft)" aria-current="true"' : ''}><div class="li-main"><span class="li-title">${esc(short(x.task, 140))}</span><span class="li-sub"><span class="dot ${xs[2]}"></span> ${esc(xs[0])} · ${int(x.steps || 0)} tool call${x.steps === 1 ? '' : 's'}${x.project_id ? ' · project ' + esc(x.project_id) : ''}${x.mode === 'llm' ? ' · model' : ' · standard plan'} · ${esc(when(x.updated))}</span></div></button>`;
      }).join('');
      $('.sub', sessHead).textContent = 'Newest first. Click one to open it on the Overview.';
      $('.pill', sessHead).textContent = String(n);
      const tabN = $('.ptab[data-ptab="sessions"] .n', el); if (tabN) tabN.textContent = String(n);
      const nav = document.querySelector('[data-nav="intern"] .count');
      if (nav) { if (SAMPLE.nav == null) SAMPLE.nav = nav.textContent; nav.textContent = live ? String(live) : ''; }
    }
    function drawReal() {
      const s = R.cur;
      if (!s) return;
      const fol = followUps(s);
      const sig = [s.id, s.updated, s.status, (s.steps || []).length, !!s.final, s.error || '', fol.length].join('|');
      if (sig !== R.sig) {
        const thread = $('#thread', el);
        const keep = R.openFor === s.id ? $$('details[data-step][open]', thread).map(d => d.dataset.step) : [];
        thread.innerHTML = realThread(s);
        keep.forEach(n => { const d = $(`details[data-step="${n}"]`, thread); if (d) d.open = true; });
        R.openFor = s.id; R.seen[s.id] = (s.steps || []).length; R.sig = sig;
        DC.hydrate(thread);
      }
      sideReal(s, R.proj);
    }

    function toReal() {
      if (R.mode !== 'real') {
        R.mode = 'real'; R.sig = ''; clearInterval(timer);
        $('#sess-replay', el).hidden = true; $('#intern-new-cmd', el).hidden = false;
        $('#intern-follow', el).placeholder = 'Ask a follow-up when the turn is done, or type /new and a task to start another session';
        modelReal();
      }
      DC.markSample(false);
    }
    function toSample() {
      DC.markSample(true);
      if (R.mode === 'sample') return;
      R.mode = 'sample'; R.cur = null; R.proj = null; R.projKey = ''; clearTimeout(R.timer);
      statsEl.innerHTML = SAMPLE.stats; $('#intern-sessions', el).innerHTML = SAMPLE.sessions;
      $('.sub', sessHead).textContent = SAMPLE.sessSub; $('.pill', sessHead).textContent = SAMPLE.sessPill;
      const tabN = $('.ptab[data-ptab="sessions"] .n', el); if (tabN) tabN.textContent = SAMPLE.sessPill;
      const nav = document.querySelector('[data-nav="intern"] .count'); if (nav && SAMPLE.nav != null) nav.textContent = SAMPLE.nav;
      $('#sess-mode', el).textContent = SAMPLE.mode; $('#sess-title', el).textContent = SAMPLE.title;
      $('#sess-replay', el).hidden = false; $('#intern-new-cmd', el).hidden = true; $('#intern-follow', el).placeholder = SAMPLE.placeholder;
      railLink.hidden = false; railLink.dataset.project = 'hyperack';
      $('[data-f="intern.ladder"] .pill', el).textContent = SAMPLE.ladderPill;
      modelReal();
      render();
      if ((phase === 'post' && revealed < POST.length) || (phase === 'final' && finalRevealed < FINAL.length)) advance();  // resume a paused replay
    }
    /* Load the sessions, pick the one to show, and its project; then poll while it works. */
    async function sync() {
      const my = ++R.seq;
      let list = null;
      try { list = await DC.api('/intern/sessions'); } catch (e) { /* offline: keep what is shown */ }
      if (my !== R.seq) return;
      if (!list) { if (R.mode === 'sample') DC.markSample(true); return; }
      R.sessions = list;
      if (!list.length) { if (!R.starting) toSample(); return; }
      const want = DC.state.internSession, id = want && list.some(x => x.id === want) ? want : list[0].id;
      let s = null;
      try { s = await DC.api('/intern/sessions/' + id); } catch (e) { /* removed meanwhile */ }
      if (my !== R.seq || !s) return;
      let p = R.proj;
      const key = s.project_id ? s.project_id + '|' + s.updated : '';
      if (!s.project_id) p = null;
      else if (isLive(s) || key !== R.projKey) { try { p = await DC.api('/projects/' + s.project_id); } catch (e) { p = null; } }
      if (!R.info) { try { R.info = await DC.api('/intern'); } catch (e) { /* the model name stays unknown */ } }
      if (my !== R.seq) return;
      R.cur = s; R.proj = p; R.projKey = key;
      toReal(); drawReal(); schedule();
    }
    function schedule() {
      clearTimeout(R.timer); R.timer = null;
      if (R.mode !== 'real' || DC.state.view !== 'intern' || !(isLive(R.cur) || R.kick > 0)) return;
      if (R.kick > 0) R.kick--;
      R.timer = setTimeout(() => { if (DC.state.view === 'intern') sync(); }, 2000);
    }
    async function startSession(task) {
      if (task.length < 8) { toast('Describe the task in at least a sentence.', { ok: false }); return; }
      R.starting = true; R.busy = true; $('#intern-send', el).disabled = true;
      try {
        const s = await DC.api('/intern/sessions', { method: 'POST', body: { task } });
        DC.state.internSession = s.id; $('#intern-follow', el).value = '';
        toast('Session started. It works inside its budget and every move passes the graph.');
        DC.selectPane(el, 'overview');
        await sync();
      } catch (e) { toast(e.message, { ok: false }); }
      finally { R.starting = false; R.busy = false; $('#intern-send', el).disabled = false; }
    }
    async function send() {
      const box = $('#intern-follow', el), text = box.value.trim();
      if (!text || R.busy) return;
      const cmd = /^\/(\w+)\s*([\s\S]*)$/.exec(text);
      if (R.mode !== 'real') {
        // Sample mode: a command answers in the script; a sentence starts a real session.
        if (cmd) { toast(phase === 'pre' ? 'Noted. The intern answers after your decision on the open question.' : 'Sent. The intern replies inside the same budget.'); box.value = ''; return; }
        await startSession(text); return;
      }
      if (cmd) {
        const [, name, rest] = cmd, page = { solution: 'solution', audit: 'audit', brief: 'brief' }[name];
        if (name === 'new') { if (rest.trim()) await startSession(rest.trim()); else toast('Type the task after /new.', { ok: false }); return; }
        if (name === 'stop') { toast('A session cannot be stopped by hand yet: it stops at its budget.', { ok: false }); return; }
        if (page) {
          if (R.cur && R.cur.project_id) { box.value = ''; DC.state.project = 'p:' + R.cur.project_id; DC.currentProject.clear(); location.hash = page; }
          else toast('This session has no project yet.', { ok: false });
          return;
        }
      }
      const s = R.cur;
      if (!s) { await startSession(text); return; }
      if (isLive(s)) { toast('The intern is still working. Send this when the turn has finished.', { ok: false }); return; }
      R.busy = true; $('#intern-send', el).disabled = true;
      const entry = { at: (s.steps || []).length, text, prevFinal: s.final, when: new Date().toISOString() };
      try {
        let out = await DC.api(`/intern/sessions/${s.id}/message`, { method: 'POST', body: { text } });
        // The turn starts right after the reply, so the reply still shows the previous state.
        if (!isLive(out)) out = Object.assign({}, out, { status: 'queued', final: null });
        (R.follow[s.id] = R.follow[s.id] || []).push(entry);
        box.value = ''; R.cur = out; R.kick = 2; drawReal(); schedule();
      } catch (e) { toast(e.message, { ok: false }); }
      finally { R.busy = false; $('#intern-send', el).disabled = false; }
    }

    render();
    this.sync = sync;
  },
  enter() {
    this.sync();
  },
});
