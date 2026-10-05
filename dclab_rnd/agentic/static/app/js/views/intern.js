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
      const p = e.target.closest('[data-project]'); if (p) DC.state.project = p.dataset.project;
    });
    $('#sess-replay', el).addEventListener('click', () => { clearInterval(timer); phase = 'pre'; answered = null; pushback = false; revealed = 0; finalRevealed = 0; steps = 9; minutes = 3.1; spend = 0.42; render(); });
    $('#intern-send', el).addEventListener('click', () => { const v = $('#intern-follow', el).value.trim(); if (!v) return; toast(phase === 'pre' ? 'Noted. The intern answers after your decision on the open question.' : 'Sent. The intern replies inside the same budget.'); $('#intern-follow', el).value = ''; });
    $('#intern-model', el).addEventListener('change', e => {
      const m = { router: 'open model · via Hugging Face router', local: 'local model · Ollama on this machine', standard: 'no model · standard plan', policy: 'DCLab policy model · planned' }[e.target.value];
      $('#sess-mode', el).textContent = m;
      toast(e.target.value === 'policy' ? 'The policy model is planned (phase 4). Its training data is collected from logs like this one.' : 'Model changed for the next turn. Keys stay on the server.');
    });
    render();
  },
});
