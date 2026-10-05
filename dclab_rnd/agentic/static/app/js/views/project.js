DC.view('project', {
  init(el) {
    const { $, $$, esc, chip, chips, icon, toast } = DC;
    const NODES = [
      ['WF-01', 'Solution', 20, 64], ['WF-02', 'Source & lineage', 210, 64], ['WF-03', 'Split design', 400, 64], ['WF-04', 'Train-only EDA', 590, 64], ['WF-05', 'Leakage audit', 780, 64],
      ['WF-06', 'Feature ladder', 780, 224], ['WF-07', 'Algorithm screen', 590, 224], ['WF-08', 'Optimization', 400, 224], ['WF-09', 'Reliability', 210, 224], ['WF-10', 'Knowledge capture', 20, 224],
    ];
    const W = 150, H = 64;
    const RULES = { 'WF-01': ['DCLAB-R01', 'DCLAB-R18'], 'WF-02': ['DCLAB-R09', 'DCLAB-R21'], 'WF-03': ['DCLAB-R02', 'DCLAB-R17'], 'WF-04': ['DCLAB-R03', 'DCLAB-R12'], 'WF-05': ['DCLAB-R04', 'DCLAB-R05', 'DCLAB-R06'],
      'WF-06': ['DCLAB-R07', 'DCLAB-R08', 'DCLAB-R10', 'DCLAB-R11'], 'WF-07': ['DCLAB-R13', 'DCLAB-R14'], 'WF-08': ['DCLAB-R15', 'DCLAB-R16'], 'WF-09': ['DCLAB-R17', 'DCLAB-R19', 'DCLAB-R22'], 'WF-10': ['DCLAB-R20', 'DCLAB-R21'] };
    const EXIT = { 'WF-01': 'Signed solution: moment, target window, action, unit, costs, forbidden columns with reasons.', 'WF-02': 'Snapshot hash, licence, row count, time range, known transformations.',
      'WF-03': 'Split unit and direction declared; holdout sealed; fold assignment stored.', 'WF-04': 'Training-only profile; risks listed; no feature promoted from EDA.',
      'WF-05': 'Availability verdict per column; severity ablation on identical folds; canary result.', 'WF-06': 'Recipe chosen by the predeclared tolerance; lineage for every new feature.',
      'WF-07': 'Families screened on identical folds; pick by the declared rule; runner-up kept.', 'WF-08': 'Tuning kept only if it beats the paired baseline by the declared margin.',
      'WF-09': 'Holdout used once with an interval; calibration; operating points; slices; production gate.', 'WF-10': 'Brief, lessons for the evidence index, training examples, trajectory.' };
    const pos = Object.fromEntries(NODES.map(n => [n[0], { x: n[2], y: n[3], cx: n[2] + W / 2, cy: n[3] + H / 2, name: n[1] }]));
    const FWD = NODES.slice(0, 9).map((n, i) => [n[0], NODES[i + 1][0]]);
    function fwdPath(a, b) {
      const p = pos[a], q = pos[b];
      if (p.y === q.y && q.x > p.x) return `M${p.x + W},${p.cy} L${q.x - 2},${q.cy}`;
      if (p.y === q.y) return `M${p.x},${p.cy} L${q.x + W + 2},${q.cy}`;
      return `M${p.cx},${p.y + H} L${q.cx},${q.y - 2}`;
    }
    const BACK = {
      'WF-05>WF-01': { d: 'M855,64 C855,14 95,14 95,62', label: 'revisit · a leak changes the solution', lx: 475, ly: 28 },
      'WF-09>WF-03': { d: 'M285,224 C285,178 475,178 475,130', label: 'revisit · a shift needs a new split', lx: 380, ly: 172 },
      'WF-07>WF-06': { d: 'M665,288 C665,328 855,328 855,290', label: 'revisit · features', lx: 760, ly: 330 },
    };
    const BLOCK = { 'WF-07>WF-09': { d: 'M630,288 C630,352 285,352 285,290', label: 'blocked · open the holdout to choose', lx: 458, ly: 352 } };
    const GATES = [
      { x: 570, y: 96, label: 'solution signed', ly: 54 },
      { x: 380, y: 256, label: 'owner approves', ly: 304 },
    ];

    const P = {
      bank: {
        title: 'Term-deposit calls', sub: 'bank_marketing · tabular · binary · started Sep 27',
        lede: 'Nine of ten steps are done and the brief waits for Reza\'s sign-off. One unsafe move was blocked on the way; the log shows why.',
        actions: '<a class="btn" href="#notebook">Open notebook</a><a class="btn primary" href="#brief">Open the brief</a>',
        current: 'WF-10',
        log: [
          ['Sep 27 09:12', 'SM', 'human', 'Create project from the sentence', 'WF-01', [], 'ok', 'proceed', 'c---------'],
          ['Sep 27 09:12', 'AI', 'agent', 'Propose solution v1 from the column audit', 'WF-01', ['DATASET-bank_marketing', 'DCLAB-R01'], 'ok', 'run check', 'c---------'],
          ['Sep 27 09:13', 'AI', 'agent', 'Ask: does campaign include the call we are about to make?', 'WF-01', ['DATASET-bank_marketing'], 'ask', 'ask owner', 'c---------'],
          ['Sep 27 09:20', 'SM', 'human', 'Edit solution v2: capacity 2,000 calls, €8 per call, €110 per subscription', 'WF-01', ['DCLAB-R18'], 'ok', 'edit', 'c---------'],
          ['Sep 27 09:21', 'SM', 'human', 'Sign solution v2 (gate: solution signed)', 'WF-01', [], 'ok', 'approve', 'dc--------'],
          ['Sep 27 09:22', 'AI', 'agent', 'Record lineage: snapshot 9c1e…4b07, licence CC BY 4.0', 'WF-02', ['DCLAB-R09'], 'ok', 'proceed', 'ddc-------'],
          ['Sep 28 10:02', 'AI', 'agent', 'Design split: stratified 80/20, no time order, no client ID; seal holdout', 'WF-03', ['PIT-005', 'DCLAB-R02'], 'ok', 'proceed', 'dddc------'],
          ['Sep 28 10:04', 'AI', 'agent', 'Profile 3,200 training rows only', 'WF-04', ['EXP-006'], 'ok', 'run check', 'dddc------'],
          ['Sep 28 10:04', 'CR', 'critic', 'Critic: do not approve modelling readiness yet', 'WF-04', ['EXP-006'], 'note', 'advisory', 'dddc------'],
          ['Sep 28 10:09', 'SM', 'human', 'Approve WF-04 with the risk recorded', 'WF-04', [], 'ok', 'approve', 'ddddc-----'],
          ['Sep 28 10:11', 'AI', 'agent', 'Leakage audit: duration forbidden (+0.1842), canary caught', 'WF-05', ['EXP-007', 'LEAK-bank_marketing'], 'ok', 'proceed', 'dddddc----'],
          ['Sep 28 10:14', 'AI', 'agent', 'Feature ladder: ratios is the smallest set within 0.002', 'WF-06', ['EXP-008', 'DCLAB-R07'], 'ok', 'proceed', 'dddddc----'],
          ['Sep 28 10:14', 'CR', 'critic', 'Critic challenge contradicted by the numbers, dropped by the gate', 'WF-06', ['EXP-008', 'DCLAB-R20'], 'note', 'advisory', 'ddddddc---'],
          ['Sep 29 14:30', 'AI', 'agent', 'Screen 5 families on identical folds; extra_trees by mean − 0.25×std', 'WF-07', ['EXP-009', 'DCLAB-R13'], 'ok', 'proceed', 'dddddddc--'],
          ['Sep 29 14:31', 'AI', 'agent', 'Open the holdout to compare extra_trees with lightgbm', 'WF-07', ['PIT-006', 'DCLAB-R17'], 'blocked', 'proceed', 'dddddddc--'],
          ['Sep 29 14:36', 'AI', 'agent', 'Tune on training folds: C02 kept (+0.0235 over baseline)', 'WF-08', ['EXP-010', 'DCLAB-R16'], 'ok', 'proceed', 'ddddddddc-'],
          ['Oct 2 11:40', 'SM', 'human', 'Approve opening the holdout (gate: owner approves)', 'WF-08', ['DCLAB-R17'], 'ok', 'approve', 'ddddddddc-'],
          ['Oct 2 11:41', 'AI', 'agent', 'Holdout opened once: ROC-AUC 0.7718 [0.7126, 0.8290]', 'WF-09', ['EXP-010'], 'ok', 'proceed', 'dddddddddc'],
          ['Oct 2 11:42', 'AI', 'agent', 'Calibration and operating points: Brier 0.0870, ECE 0.0272', 'WF-09', ['EXP-010', 'DCLAB-R18'], 'ok', 'run check', 'dddddddddc'],
          ['Oct 2 11:44', 'AI', 'agent', 'Production gate: 2 of 6 items met; marked as a research result', 'WF-09', ['DCLAB-R22'], 'ok', 'run check', 'dddddddddc'],
          ['Oct 3 09:00', 'AI', 'agent', 'Draft the brief; propose 3 lessons; write 11 training examples', 'WF-10', ['DCLAB-R21'], 'ok', 'proceed', 'dddddddddc'],
          ['Oct 3 16:20', 'AV', 'human', 'Review lessons: 2 accepted, 1 sent back for scope', 'WF-10', ['DCLAB-R21'], 'ok', 'approve', 'dddddddddc'],
          ['Oct 4 10:05', 'RZ', 'human', 'Question on the brief: is €8 the full cost of a call?', 'WF-10', [], 'ask', 'ask', 'dddddddddc'],
          ['now', 'RZ', 'human', 'Waiting: business sign-off', 'WF-10', [], 'wait', 'approve', 'dddddddddc'],
        ],
      },
      hyperack: {
        title: 'HyperAck order acceptance', sub: 'hyperack · tabular · binary · 11,118 orders · 2022-06-29 to 2022-11-14',
        lede: 'Step 1 is waiting for an answer only the owner can give: the moment of prediction. The research runs exist, but none counts until the solution is signed.',
        actions: '<a class="btn" href="#intern">Open the intern session</a><button type="button" class="btn primary" data-scroll-ask>Answer the question</button>',
        current: 'WF-01',
        log: [
          ['Oct 2 15:10', 'SM', 'human', 'Create project, hand it to the intern', 'WF-01', [], 'ok', 'proceed', 'c---------'],
          ['Oct 2 15:11', 'AI', 'agent', 'Load the R&D precedent: final fares are post-outcome', 'WF-01', ['LEAK-hyperack', 'DATASET-hyperack'], 'ok', 'run check', 'c---------'],
          ['Oct 2 15:12', 'AI', 'agent', 'Ask: when is the prediction made?', 'WF-01', ['DCLAB-R01'], 'ask', 'ask owner', 'b---------'],
          ['now', 'SM', 'human', 'Waiting for the owner\'s answer', 'WF-01', [], 'wait', 'ask', 'b---------'],
        ],
      },
      telco: {
        title: 'Telco churn', sub: 'telco_churn · tabular · binary · 7,043 customers',
        lede: 'The screen finished, but the solution never said when the prediction is made or how long the outcome window is. It was reopened; later steps wait for it.',
        actions: '<a class="btn primary" href="#solution">Open the solution</a>',
        current: 'WF-01',
        log: [
          ['Sep 20 10:00', 'AV', 'human', 'Create project from the Telco sample', 'WF-01', ['DATASET-telco_churn'], 'ok', 'proceed', 'c---------'],
          ['Sep 20 10:30', 'AI', 'agent', 'Steps 2–7 run; logistic regression leads (ROC-AUC 0.8499)', 'WF-07', ['FINDING-churn-linear'], 'ok', 'proceed', 'dddddddc--'],
          ['Sep 21 09:15', 'CR', 'critic', 'Critic: adaptive development CV is not independent confirmation', 'WF-07', ['FINDING-churn-linear'], 'note', 'advisory', 'dddddddc--'],
          ['Sep 21 09:20', 'AV', 'human', 'Reopen the solution: the prediction moment and outcome window are missing', 'WF-01', ['DCLAB-R01'], 'back', 'revisit', 'bdddddd---'],
          ['now', 'SM', 'human', 'Waiting: 30-day or 90-day outcome window?', 'WF-01', [], 'wait', 'ask', 'bdddddd---'],
        ],
      },
    };
    const WHO = { SM: ['Shahriyar', ''], AV: ['Ava', 'b'], RZ: ['Reza', 'c'], AI: ['Intern', 'ai'], CR: ['Critic', 'ai'] };
    let proj = 'bank', step = 24, selected = 'WF-10';

    function nodeClass(s) { return { d: 'done', c: 'current', b: 'blocked', '-': 'todo' }[s] || 'todo'; }
    function statusText(s, id) {
      if (s === 'd') return 'done';
      if (s === 'c') return 'current';
      if (s === 'b') return proj === 'telco' ? 'reopened · waiting' : 'waiting on the owner';
      return 'not started';
    }
    function drawGraph() {
      const log = P[proj].log.slice(0, step);
      const states = (log[log.length - 1] || ['', '', '', '', '', [], '', '', 'c---------'])[8];
      const blocked = log.some(r => r[6] === 'blocked');
      const takenBack = proj === 'telco' && log.some(r => r[6] === 'back');
      let s = `<svg class="graph-svg" viewBox="0 0 950 362" role="img" aria-label="Workflow graph with ten steps">
        <defs>
          <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="ah" d="M0,0 L10,5 L0,10 z"/></marker>
          <marker id="ah-ok" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="ah-ok" d="M0,0 L10,5 L0,10 z"/></marker>
          <marker id="ah-back" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="ah-back" d="M0,0 L10,5 L0,10 z"/></marker>
          <marker id="ah-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="ah-bad" d="M0,0 L10,5 L0,10 z"/></marker>
          <marker id="ah-allowed" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="ah-allowed" d="M0,0 L10,5 L0,10 z"/></marker>
        </defs>`;
      FWD.forEach(([a, b]) => {
        const i = NODES.findIndex(n => n[0] === b);
        const taken = states[i] && states[i] !== '-' && states[i - 1] === 'd';
        s += `<path class="edge ${taken ? 'taken' : ''}" d="${fwdPath(a, b)}" marker-end="url(#${taken ? 'ah-ok' : 'ah'})"/>`;
      });
      Object.entries(BACK).forEach(([k, e]) => {
        s += `<path class="edge allowed" d="${e.d}" marker-end="url(#ah-allowed)"/><text class="lane-label" x="${e.lx}" y="${e.ly}" text-anchor="middle">${esc(e.label.toUpperCase())}</text>`;
      });
      if (takenBack) s += `<path class="edge back" d="M640,224 C640,150 140,150 95,130" marker-end="url(#ah-back)"/><text class="lane-label" x="380" y="146" text-anchor="middle" data-style="fill:var(--warn)">REOPENED FROM WF-07</text>`;
      if (blocked) Object.values(BLOCK).forEach(e => { s += `<path class="edge blocked" d="${e.d}" marker-end="url(#ah-bad)"/><text class="lane-label" x="${e.lx}" y="${e.ly - 4}" text-anchor="middle" data-style="fill:var(--bad)">${esc(e.label.toUpperCase())}</text>`; });
      GATES.forEach(g => { s += `<path class="gate" d="M${g.x},${g.y - 9} L${g.x + 9},${g.y} L${g.x},${g.y + 9} L${g.x - 9},${g.y} Z"/><text class="gate-text" x="${g.x}" y="${g.ly}" text-anchor="middle">${esc(g.label)}</text>`; });
      NODES.forEach(([id, name, x, y], i) => {
        const st = states[i] || '-';
        s += `<g class="node ${nodeClass(st)}" data-node="${id}" tabindex="0" role="button" aria-label="${id} ${esc(name)}: ${statusText(st, id)}">
          <rect x="${x}" y="${y}" width="${W}" height="${H}" rx="9" ${id === selected ? 'data-style="stroke-width:2.6"' : ''}/>
          <text class="id" x="${x + 10}" y="${y + 17}">${id}</text>
          <text x="${x + 10}" y="${y + 36}">${esc(name)}</text>
          <text class="st" x="${x + 10}" y="${y + 53}">${statusText(st, id)}</text>
        </g>`;
      });
      $('#graph-box', el).innerHTML = s + '</svg>';
      const last = log[log.length - 1];
      $('#replay-label', el).textContent = `${step}/${P[proj].log.length} · ${last ? last[0] : ''}`;
      // rail
      $('#wf-rail', el).innerHTML = DEMO.wf.map(([id, name], i) => { const st = states[i] || '-'; return `<button type="button" class="wf-step ${nodeClass(st)}" data-node="${id}"><span class="wf-id">${id}</span><span class="wf-name">${esc(name)}</span><span class="wf-state">${st === 'd' ? '✓ ' : ''}${statusText(st, id)}</span></button>`; }).join('');
    }
    function drawLog(filter = 'all') {
      const rows = P[proj].log.map((r, i) => ({ r, i })).filter(({ r }) => filter === 'all' || (filter === 'human' && r[2] === 'human') || (filter === 'blocked' && r[6] === 'blocked'));
      const res = { ok: '<span class="pill ok">done</span>', blocked: '<span class="pill bad">blocked</span>', ask: '<span class="pill warn">asked</span>', note: '<span class="pill outline">advisory</span>', wait: '<span class="pill warn">waiting</span>', back: '<span class="pill warn">revisit</span>' };
      $('#log-table tbody', el).innerHTML = rows.map(({ r, i }) => `<tr class="${i >= step ? 'muted-row' : ''}">
        <td class="nowrap mono small">${esc(r[0])}</td>
        <td><span class="row" data-style="flex-wrap:nowrap"><span class="avatar sm ${WHO[r[1]][1]}">${r[1]}</span><span class="small">${WHO[r[1]][0]}</span></span></td>
        <td><div>${esc(r[3])}</div><div class="cell-sub">${esc(r[7])}</div></td>
        <td class="mono small nowrap">${r[4]}</td>
        <td>${r[5].length ? chips(r[5]) : '<span class="faint">—</span>'}</td>
        <td>${res[r[6]] || ''}</td></tr>`).join('');
    }
    function drawNode(id) {
      selected = id;
      const i = NODES.findIndex(n => n[0] === id);
      const log = P[proj].log.slice(0, step);
      const states = (log[log.length - 1] || [])[8] || 'c---------';
      const st = states[i] || '-';
      const rec = DC.REC[id];
      const produced = [...new Set(P[proj].log.filter(r => r[4] === id).flatMap(r => r[5]).filter(x => !x.startsWith('DCLAB-')))];
      const next = [];
      if (i < 9) next.push(`<li>Proceed to <b>${NODES[i + 1][0]}</b> ${esc(NODES[i + 1][1])}${id === 'WF-03' ? ' · needs the signed solution' : id === 'WF-08' ? ' · needs the owner\'s approval' : ''}</li>`);
      Object.keys(BACK).filter(k => k.startsWith(id + '>')).forEach(k => next.push(`<li>Revisit <b>${k.split('>')[1]}</b> with a written reason</li>`));
      next.push('<li>Run a named check, ask the owner, or stop</li>');
      if (i > 0) next.push('<li>Reopen the solution (clears later results)</li>');
      $('#node-eyebrow', el).textContent = `Selected step · ${statusText(st, id)}`;
      $('#node-title', el).textContent = `${id} · ${DEMO.wf[i][1]}`;
      $('#node-body', el).innerHTML = `
        <p class="muted">${rec ? DC.linkIds(rec.text.replace(/^Workflow block WF-\d+ — [^:]+: /, '')) : ''}</p>
        <div><div class="eyebrow">To leave this step</div><p>${esc(EXIT[id])}</p></div>
        <div><div class="eyebrow">Rules</div>${chips(RULES[id])}</div>
        ${produced.length ? `<div><div class="eyebrow">Evidence produced here</div>${chips(produced)}</div>` : ''}
        <div><div class="eyebrow">Allowed moves from here</div><ul class="small">${next.join('')}</ul></div>
        ${proj === 'hyperack' && id === 'WF-01' ? askCard() : ''}
        ${proj === 'telco' && id === 'WF-01' ? telcoCard() : ''}`;
      $$('.node rect', el).forEach(r => r.removeAttribute('style'));
      const g = $(`.node[data-node="${id}"] rect`, el); if (g) g.style.strokeWidth = '2.6';
    }
    function askCard() {
      return `<div class="ask-card" id="ask-card">
        <div class="q">When is the HyperAck prediction made?</div>
        <div class="small">This decides which columns are allowed. The final fares are already blocked: with them ROC-AUC reaches 0.9802, the leakage-safe ensemble 0.9455 ${chip('LEAK-hyperack')}.</div>
        <div class="ask-options">
          <button type="button" class="ask-option" data-ans="a"><span><b>When the order is created.</b> Before any courier sees it. Courier-side columns are not allowed.</span></button>
          <button type="button" class="ask-option" data-ans="b"><span><b>When the order is offered to a courier.</b> Courier distance and offer price become allowed; anything after the offer is not.</span></button>
          <button type="button" class="ask-option" data-ans="c"><span><b>After a courier is assigned.</b> Then acceptance is mostly known; the model would answer a different question.</span></button>
        </div>
        <div class="small muted">Also needed: the target window (accepted within how many seconds?).</div>
      </div>`;
    }
    function telcoCard() {
      return `<div class="ask-card"><div class="q">Which outcome window should "churn" mean?</div>
        <div class="small">The dataset card says a verified prediction timestamp and outcome window are still required ${chip('DATASET-telco_churn')}. The screen result (${chip('FINDING-churn-linear')}) stays in the log but cannot be promoted until this is answered.</div>
        <div class="ask-options"><button type="button" class="ask-option" data-ans="t30"><span><b>Cancels within 30 days</b> of the snapshot</span></button><button type="button" class="ask-option" data-ans="t90"><span><b>Cancels within 90 days</b> of the snapshot</span></button></div></div>`;
    }
    const VALID = {
      holdout: { verdict: 'blocked', checks: [['Edge exists', false, 'WF-07 → WF-09 is not an edge. Optimization (WF-08) comes first.'], ['Prerequisites', false, 'No tuning decision recorded yet.'], ['Approval gate', false, 'Opening the holdout needs the owner.'], ['Rule check', false, 'DCLAB-R17: the holdout is consumed once, after every choice is locked.'], ['Evidence attached', true, 'EXP-009 attached.']], why: 'Choosing between models by holdout score makes that score optimistic: +0.0072 ROC-AUC on average, +0.0177 on german_credit.', ev: ['DCLAB-R17', 'PIT-006'] },
      stop: { verdict: 'needs approval', checks: [['Edge exists', true, 'WF-10 may end the project.'], ['Prerequisites', true, 'Brief drafted, lessons proposed, examples written.'], ['Approval gate', false, 'Business sign-off from Reza is still pending.'], ['Rule check', true, 'Marked as a research result (DCLAB-R22).']], why: 'The move is valid but waits at the sign-off gate.', ev: ['DCLAB-R22'] },
      revisit: { verdict: 'allowed', checks: [['Edge exists', true, 'Reopening the solution is allowed from any step.'], ['Reason given', true, 'A reason is required and logged.'], ['Side effects', true, 'Clears WF-02…WF-10 results; the holdout stays consumed.'], ['Approval gate', true, 'The owner signs the new version.']], why: 'Allowed. Expensive on purpose: the log shows what was cleared and why.', ev: ['DCLAB-R01'] },
      skip: { verdict: 'blocked', checks: [['Edge exists', false, 'WF-04 → WF-06 is not an edge; WF-05 cannot be skipped.'], ['Rule check', false, 'DCLAB-R04: leakage has several separate failure modes.']], why: 'No path through the graph skips the leakage audit.', ev: ['DCLAB-R04', 'FINDING-leakage-gaps'] },
      useduration: { verdict: 'blocked', checks: [['Solution', false, 'duration is forbidden in the signed solution v2.'], ['Tool guard', false, 'No tool accepts a forbidden column as input.'], ['Rule check', false, 'Post-outcome column: known only after the call ends.']], why: 'With duration the training-CV score rises by +0.1842. That is the size of the lie, not a gain.', ev: ['LEAK-bank_marketing', 'EXP-007'] },
    };
    function drawValidator() {
      const v = VALID[$('#move-select', el).value];
      const cls = v.verdict === 'blocked' ? 'bad' : v.verdict === 'allowed' ? 'ok' : 'warn';
      $('#validator-out', el).innerHTML = `<div class="callout ${cls}"><span class="ic">${icon(v.verdict === 'blocked' ? 'alert' : v.verdict === 'allowed' ? 'check' : 'info')}</span><span><b>${v.verdict === 'blocked' ? 'Blocked and logged' : v.verdict === 'allowed' ? 'Allowed' : 'Valid, waits for approval'}.</b> ${esc(v.why)}</span></div>
        <ul class="plan-list">${v.checks.map(([n, ok, d]) => `<li class="${ok ? 'done' : 'blocked'}"><span class="pi">${ok ? '✓' : '!'}</span><span><b>${esc(n)}</b> · <span class="muted">${DC.linkIds(d)}</span></span></li>`).join('')}</ul>
        <div class="row small"><span class="muted">Evidence returned to the intern:</span>${chips(v.ev)}</div>`;
    }
    function drawTeam() {
      if (isReal(proj)) { $('#team-list', el).innerHTML = '<div class="list-item"><span class="avatar sm">SM</span><div class="li-main"><span class="li-title">Shahriyar</span><span class="li-sub">Owner · the only member of this local workspace</span></div></div>'; return; }
      $('#team-list', el).innerHTML = DEMO.team.map(t => `<div class="list-item"><span class="avatar sm ${t.cls}">${t.id}</span><div class="li-main"><span class="li-title">${esc(t.name)}</span><span class="li-sub">${esc(t.role)}</span></div></div>`).join('');
    }
    function drawStats() {
      const log = P[proj].log, last = log[log.length - 1], states = last[8];
      const done = [...states].filter(c => c === 'd').length, cur = NODES.find(n => n[0] === P[proj].current);
      const people = log.filter(r => r[2] === 'human').length, blocked = log.filter(r => r[6] === 'blocked').length, back = log.filter(r => r[6] === 'back').length;
      $('#proj-stats', el).innerHTML = `
        <div class="stat"><span class="v">${done} <small>/ 10</small></span><span class="l">steps done · current ${cur[0]} ${esc(cur[1])}</span></div>
        <div class="stat"><span class="v">${log.length}</span><span class="l">moves in the transition log · ${people} by people</span></div>
        <div class="stat"><span class="v ${blocked ? 'bad' : back ? 'warn' : ''}">${blocked || back}</span><span class="l">${blocked ? 'unsafe move blocked by the graph' : back ? 'revisit: the solution was reopened' : 'blocked moves'}</span></div>
        <div class="stat"><span class="v warn">${esc(WHO[last[1]][0])}</span><span class="l">${esc(last[3])}</span></div>`;
    }
    /* ---------- real projects: the API's project, graph and transition log in the sample's shape ---------- */
    const { api } = DC;
    const REAL = {};
    let poller = null;
    const isReal = k => typeof k === 'string' && k.startsWith('p:');
    const st8 = s => (s || 'c---------').replace(/[wf]/g, 'b').replace(/r/g, 'c');
    const when = at => { const d = new Date(at); return isNaN(d) ? '' : d.toLocaleString('en-US', { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }); };
    function moveText(t) {
      const a = t.args || {};
      if (t.move === 'run_stage') return `Run the ${a.stage} stage`;
      if (t.move === 'set_solution') return 'Save the solution' + (a.target ? ` · target ${a.target}` : '');
      if (t.move === 'approve_gate') return `Approve the ${a.gate} gate`;
      if (t.move === 'approve_stage') return `Approve ${a.stage}${a.choice ? ' · ' + a.choice : ''}`;
      if (t.move === 'capture') return 'Capture the knowledge';
      return t.move;
    }
    function fromApi(p) {
      const g = p.graph || {}, moves = p.transitions || [];
      const ids = x => (x || []).filter(id => DC.REC[id] || /^DCLAB-R\d\d$/.test(id));
      let log = moves.map(t => [when(t.at), t.actor === 'agent' ? 'AI' : 'SM', t.actor === 'agent' ? 'agent' : 'human', moveText(t) + (t.status !== 'allowed' && t.message ? ' — ' + t.message : ''),
        t.to || t.from || '', ids([...(t.rules || []), ...(t.evidence || [])]), { allowed: 'ok', blocked: 'blocked', needs_approval: 'wait' }[t.status] || 'note',
        t.outcome || t.status, '']);
      if (!log.length) log = [[when(p.created), 'SM', 'human', 'Create the project', 'WF-01', [], 'ok', 'proceed', '']];
      // each row shows the state after its move: the state before the next move, and now for the last one
      log.forEach((r, i) => { r[8] = st8(i + 1 < moves.length ? moves[i + 1].state : g.state); });
      const d = p.data || {}, sol = p.solution || {};
      const next = (g.moves || []).find(m => m.move === 'run_stage' && m.status === 'allowed' && (p.stages[m.stage] || {}).status !== 'completed');
      const running = p.running && (p.running.stage || p.running);
      const actions = running ? `<button type="button" class="btn" disabled>Running ${esc(String(running))}…</button>`
        : `${next ? `<button type="button" class="btn primary" data-run-stage="${esc(next.stage)}">Run the ${esc(next.stage)} stage</button><button type="button" class="btn" data-run-all="1">Run all remaining</button>` : ''}<a class="btn" href="/classic#project/${esc(p.id)}">Open the notebook</a>`;
      return { title: p.name, sub: [d.filename, d.rows ? Number(d.rows).toLocaleString('en-US') + ' rows' : '', sol.task, d.synthetic ? 'synthetic data' : '', 'started ' + (p.created || '').slice(0, 10)].filter(Boolean).join(' · '),
        lede: p.goal || 'No goal written yet.', actions, current: g.current || 'WF-10', log, real: true, project: p };
    }
    function realValidator() {
      const p = REAL[proj], moves = (p.graph.moves || []);
      const opts = [...moves.map(m => ({ move: m.move, stage: m.stage, actor: 'human', label: `You: ${m.move === 'capture' ? 'capture the knowledge' : 'run the ' + m.stage + ' stage'}` })),
        { move: 'run_stage', stage: 'final', actor: 'agent', label: 'Intern: score the holdout now' },
        { move: 'set_solution', actor: 'agent', label: 'Intern: change the solution' },
        { move: 'approve_gate', gate: 'holdout', actor: 'agent', label: 'Intern: approve its own holdout gate' }];
      $('#move-select', el).innerHTML = opts.map((o, i) => `<option value="${i}">${esc(o.label)}</option>`).join('');
      $('#move-select', el).dataset.real = JSON.stringify(opts);
    }
    async function drawRealVerdict() {
      const opts = JSON.parse($('#move-select', el).dataset.real || '[]'), o = opts[Number($('#move-select', el).value)];
      if (!o) return;
      try {
        const v = await api(`/projects/${REAL[proj].id}/graph/check`, { method: 'POST', body: { move: o.move, actor: o.actor, stage: o.stage, gate: o.gate } });
        const cls = v.status === 'blocked' ? 'bad' : v.status === 'allowed' ? 'ok' : 'warn';
        $('#validator-out', el).innerHTML = `<div class="callout ${cls}"><span class="ic">${icon(v.status === 'blocked' ? 'alert' : v.status === 'allowed' ? 'check' : 'info')}</span><span><b>${v.status === 'blocked' ? 'Blocked' : v.status === 'allowed' ? 'Allowed' : 'Valid, waits for approval'}.</b> ${esc(v.message || '')}</span></div>
          <ul class="plan-list">${(v.checks || []).map(c => `<li class="${c.ok ? 'done' : 'blocked'}"><span class="pi">${c.ok ? '✓' : '!'}</span><span><b>${esc(c.name)}</b> · <span class="muted">${DC.linkIds(c.detail || '')}</span></span></li>`).join('')}</ul>
          ${(v.rules || []).length ? `<div class="row small"><span class="muted">Rules:</span>${chips(v.rules.filter(r => DC.REC[r]))}</div>` : ''}<div class="xs muted">A check only: nothing is run or logged from here.</div>`;
        DC.hydrate($('#validator-out', el));
      } catch (e) { $('#validator-out', el).innerHTML = `<div class="callout bad"><span class="ic">${icon('alert')}</span><span>${esc(e.message)}</span></div>`; }
    }
    async function openReal(key, keepStep) {
      const p = await api(`/projects/${key.slice(2)}`);
      REAL[key] = p; P[key] = fromApi(p);
      setProject(key, keepStep);
      clearTimeout(poller);
      if (p.running) poller = setTimeout(() => { if (proj === key && DC.state.view === 'project') openReal(key, true); }, 1500);
    }
    async function switcher() {
      let list = [];
      try { list = (await api('/workspace')).projects.slice(0, 3); } catch (e) { /* offline: samples only */ }
      $('#proj-switch', el).innerHTML = list.map(p => `<button type="button" data-v="p:${esc(p.id)}" aria-pressed="false">${esc(p.name)}</button>`).join('')
        + '<button type="button" data-v="bank" aria-pressed="false">Term-deposit calls · sample</button><button type="button" data-v="hyperack" aria-pressed="false">HyperAck · sample</button><button type="button" data-v="telco" aria-pressed="false">Telco churn · sample</button>';
      return list;
    }
    el.addEventListener('click', async e => {
      const run = e.target.closest('[data-run-stage], [data-run-all]');
      if (!run || !isReal(proj)) return;
      const id = REAL[proj].id;
      try {
        if (run.dataset.runAll) await api(`/projects/${id}/run`, { method: 'POST' });
        else await api(`/projects/${id}/stages/${run.dataset.runStage}/run`, { method: 'POST' });
        toast('Started. The graph checked the move first; the log shows it.');
        openReal(proj, true);
      } catch (err) { toast(err.message, { ok: false }); openReal(proj, true); }
    });

    function setProject(p, keepStep) {
      proj = p; step = P[p].log.length;
      $$('#proj-switch button', el).forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === p)));
      $('#proj-title', el).textContent = P[p].title;
      $('#proj-sub', el).textContent = P[p].sub;
      $('#proj-lede', el).textContent = P[p].lede;
      $('#proj-actions', el).innerHTML = P[p].actions;
      const r = $('#replay', el); r.max = P[p].log.length; r.value = step;
      DC.state.project = p; DC.markSample(!isReal(p)); DC.currentProject.clear();
      if (isReal(p)) { realValidator(); drawRealVerdict(); DC.setProjectLabel && DC.setProjectLabel(P[p].title); }
      else { $('#move-select', el).innerHTML = SAMPLE_MOVES; delete $('#move-select', el).dataset.real; drawValidator(); DC.setProjectLabel && DC.setProjectLabel(P[p].title + ' · sample'); }
      drawStats(); drawGraph(); drawLog(); drawNode(keepStep && selected ? selected : P[p].current); drawTeam(); DC.hydrate(el);
    }
    this.setProject = setProject; this.openReal = openReal; this.switcher = switcher;
    const SAMPLE_MOVES = $('#move-select', el).innerHTML;
    $('#proj-switch', el).addEventListener('segchange', e => { DC.state.project = e.detail; if (isReal(e.detail)) openReal(e.detail); else setProject(e.detail); });
    $('#replay', el).addEventListener('input', e => { step = Number(e.target.value); drawGraph(); drawLog($('#log-filter button[aria-pressed="true"]', el).dataset.v); drawNode(selected); });
    let timer = null;
    $('#replay-play', el).addEventListener('click', () => {
      clearInterval(timer); step = 1; const max = P[proj].log.length;
      timer = setInterval(() => { $('#replay', el).value = step; drawGraph(); drawLog(); drawNode(selected); if (++step > max) { clearInterval(timer); step = max; } }, 450);
    });
    $('#log-filter', el).addEventListener('segchange', e => drawLog(e.detail));
    $('#move-select', el).addEventListener('change', () => (isReal(proj) ? drawRealVerdict() : drawValidator()));
    el.addEventListener('click', e => {
      const n = e.target.closest('[data-node]'); if (n) { drawNode(n.dataset.node); if (n.classList.contains('wf-step')) { DC.reveal($('#node-panel', el)); $('#graph-box', el).scrollIntoView({ behavior: 'smooth', block: 'start' }); } return; }
      const a = e.target.closest('[data-ans]');
      if (a) {
        $$('.ask-option', el).forEach(b => b.setAttribute('aria-pressed', String(b === a)));
        toast(a.dataset.ans.startsWith('t') ? 'Answer recorded. Solution v3 drafted; later steps rerun after you sign.' : 'Answer recorded. Solution v2 drafted with the allowed columns; it needs your signature.');
      }
      if (e.target.closest('[data-scroll-ask]')) { const c = $('#ask-card', el); if (c) { DC.reveal(c); c.scrollIntoView({ behavior: 'smooth', block: 'center' }); } }
    });
    el.addEventListener('keydown', e => { const n = e.target.closest('.node'); if (n && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); drawNode(n.dataset.node); } });
    drawValidator(); drawTeam(); setProject('bank');
  },
  async enter() {
    // a project just created by the wizard, else the most recent real project, else the bank sample
    const list = await this.switcher();
    const want = DC.state.projectId ? 'p:' + DC.state.projectId : DC.state.project || (list.length ? 'p:' + list[0].id : 'bank');
    DC.state.projectId = null; DC.state.project = want;
    if (want.startsWith('p:')) { try { await this.openReal(want); return; } catch (e) { DC.toast(e.message, { ok: false }); } }
    this.setProject(want.startsWith('p:') ? 'bank' : want);
  },
});
