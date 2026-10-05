DC.view('benchmark', {
  init(el) {
    const { $, esc, chip, icon, int } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id)).replace(/\$\{icon:([a-z]+)\}/g, (m, n) => icon(n));
    /* The leaderboard is planned: no policy has been scored, so every cell stays empty (no sample scores). */
    const board = names => {
      $('#board tbody', el).innerHTML = names.map(n => `<tr><td><span class="cell-main">${esc(n)}</span></td>${'<td class="num mono faint">—</td>'.repeat(7)}</tr>`).join('');
    };
    board(['Standard plan (no model)', 'General LLM (open model, router)', 'DCLab policy model', 'Human reviewer']);
    const TRAJ = {
      std: [['run_stage · leakage', 'The column auditor sees no suspicious column.', 'warn'], ['run_stage · models', 'Continues; the fitted clustering is not a column, so nothing flags it.', 'warn'], ['verdict', 'Missed the trap.', 'bad']],
      llm: [['review_code', 'Notices KMeans but calls it "probably fine for an unsupervised step".', 'warn'], ['ask_owner', 'Asks the owner whether zones are allowed. A question nobody needed.', 'warn'], ['verdict', 'Partial: found it, did not act, spent a question.', 'warn']],
      pol: [['review_code', 'Flags KMeans fitted on all rows before the split.', 'ok'], ['propose · fix', 'Moves the clustering into the pipeline so it fits inside each fold.', 'ok'], ['validator', 'Fix accepted; evidence cited: DCLAB-R03, PIT-001.', 'ok'], ['verdict', 'Pass: caught, fixed, no unnecessary question.', 'ok']],
    };
    function traj(p) {
      $('#case-traj', el).innerHTML = `<div class="timeline">${TRAJ[p].map(([t, d, c], i) => `<div class="tl-item"><span class="tl-mark ${c === 'ok' ? 'ok' : c === 'bad' ? 'bad' : 'warn'}">${i + 1}</span><div class="tl-body"><span class="tl-title mono small">${esc(t)}</span><span class="small">${DC.linkIds(d)}</span></div></div>`).join('')}</div><div class="xs muted">Illustrative trajectories of how each policy could behave; none of them is a scored run.</div>`;
      DC.hydrate(el);
    }
    $('#case-pol', el).addEventListener('segchange', e => traj(e.detail));
    traj('std');

    /* ---------- real mode: GET /api/lab/benchmark ---------- */
    const codes = xs => (xs.length ? xs.map(x => `<code>${esc(x)}</code>`).join(' ') : '<span class="faint">—</span>');
    const SUITE_CLS = ['bad', 'warn', 'info', 'proof', 'accent'];

    function paintAuditor(A) {
      if (!A) {
        $('#aud-body', el).innerHTML = '<tr><td colspan="4"><div class="empty">Not tracked yet: no stored blind auditor replay. Run <code>make verify-auditor</code>.</div></td></tr>';
        $('#aud-foot', el).textContent = 'The replay result is written to evidence/campaigns/agent_verification_v1/results/.';
        return;
      }
      $('#aud-body', el).innerHTML = A.cases.map(c => `<tr><td><span class="cell-main">${esc(c.dataset)}</span><div class="cell-sub">recall ${Math.round((c.recall || 0) * 100)}%${c.false_alarms.length ? ` · ${c.false_alarms.length} other column${c.false_alarms.length === 1 ? '' : 's'} flagged` : ''}</div></td>
        <td class="num">${c.known.length}</td><td class="small">${codes(c.found)}</td><td class="small">${codes(c.missed)}</td></tr>`).join('');
      const pct = Math.round(A.recall * 100);
      $('#aud-foot', el).innerHTML = `<b>${A.found} of ${A.known} found (${pct}%)</b> across ${A.datasets} datasets; every known leak was found in ${A.datasets_all_found} of them and at least one in ${A.datasets_any_found}. ${A.false_alarms} other columns were flagged for review. The misses look like ordinary numbers; only the solution's "when is this written?" catches them, so the agent must ask for the solution and never treat a clean scan as proof. Run on ${esc(String(A.completed_at || '').slice(0, 10))} · <code>${esc(A.command)}</code>${A.report ? ` · <code>${esc(A.report)}</code>` : ''}`;
    }

    function paintPilot(P) {
      const panel = $('#pilot-panel', el);
      if (P.status === 'scored') {
        const fails = P.cases.filter(c => !c.pass);
        panel.innerHTML = `<div class="panel-head"><div class="panel-title"><h2>Measured today: notebook pilot</h2><span class="sub">${P.total} labelled cells from real notebooks, frozen by hash</span></div><span class="pill ok">real</span></div>
          <div class="panel-body stack">
            <div class="row" data-style="gap:18px"><div class="score" data-style="font-size:34px;font-weight:600;color:var(--ink)">${P.passed} / ${P.total}</div><div class="small muted">Cases where the notebook companion's finding matched the label. Evidence rule cited correctly in ${P.proof_rules_passed} of ${P.proof_rules_scored} scored cases; availability wording ${P.availability_wording_safe ? 'stayed cautious' : 'was overconfident somewhere'}. ${esc(P.scope || '')}</div></div>
            ${fails.map(c => `<div class="callout warn"><span class="ic">${icon('alert')}</span><span><b>${esc(c.id)}</b> (${esc(c.signal || '')}): expected ${c.expected ? 'a flag' : 'no flag'}, got ${c.observed ? 'a flag' : 'none'}. ${DC.linkIds(c.why || '')}</span></div>`).join('')}
            <div class="xs muted">Manifest <code>${esc(P.manifest)}</code> · development cases, not a blinded test.</div>
          </div>`;
      } else {
        const why = P.status === 'error' ? `The pilot could not be scored: ${esc(P.note)}` : `Not tracked yet: the frozen pilot manifest (<code>${esc(P.manifest)}</code>) is not committed, so the pilot cannot be scored from the repository.`;
        panel.innerHTML = `<div class="panel-head"><div class="panel-title"><h2>Notebook pilot</h2><span class="sub">Labelled cells from real notebooks, frozen by hash</span></div><span class="pill outline">${P.status === 'error' ? 'not scored' : 'not tracked yet'}</span></div>
          <div class="panel-body stack tight"><div class="empty">${why}</div>
          <div class="xs muted">Once the manifest is committed, this page scores the notebook companion on it with <code>dclab_rnd.notebook_assist_eval</code> (development cases, not a blinded test). <code>tests/test_notebook_eval.py</code> skips until then.</div></div>`;
      }
    }

    function paintPlan(B) {
      const S = B.suites, Pol = B.policies;
      $('#suites-pill', el).textContent = `planned · ${S.planned_cases} cases`;
      $('#suites-list', el).innerHTML = S.items.map((s, i) => `<div class="list-item"><span class="pill ${SUITE_CLS[i % SUITE_CLS.length]}">${esc(s.id)}</span><div class="li-main"><span class="li-title">${esc(s.name)} · ${s.planned_cases}</span><span class="li-sub">${esc(s.covers)} · ${s.scored_cases} frozen</span></div></div>`).join('');
      $('#board-sub', el).textContent = `Same ${S.planned_cases} planned cases for every policy · higher is better except unsafe actions and cost`;
      board(Pol.items.map(p => p.name));
    }

    function paintStats(B) {
      const A = B.auditor, P = B.notebook_pilot, S = B.suites, Pol = B.policies;
      $('#bench-stats', el).innerHTML = `
        ${A ? `<div class="stat"><span class="v warn">${A.found} / ${A.known} <small>${Math.round(A.recall * 100)}%</small></span><span class="l">known leaks the blind auditor found · ${A.datasets} datasets</span></div>`
            : '<div class="stat"><span class="v faint">—</span><span class="l">blind auditor replay not stored yet</span></div>'}
        ${P.status === 'scored' ? `<div class="stat"><span class="v">${P.passed} / ${P.total}</span><span class="l">labelled problems matched in the notebook pilot</span></div>`
            : `<div class="stat"><span class="v faint">—</span><span class="l">notebook pilot not tracked yet · frozen manifest not committed</span><div class="row"><button type="button" class="link-btn small" data-pane-go="pilot">Why →</button></div></div>`}
        <div class="stat"><span class="v">${S.planned_cases}</span><span class="l">planned cases in ${S.items.length} suites, the same for every policy · none frozen yet</span></div>
        <div class="stat"><span class="v">${Pol.items.length}</span><span class="l">policies planned · ${Pol.items.map(p => esc(p.name.replace(/ \(.*\)$/, ''))).join(', ')} · ${Pol.scored ? Pol.scored + ' scored' : 'none scored yet'}</span></div>`;
      $('#bench-pill', el).textContent = `${P.status === 'scored' ? 'pilots real' : 'auditor replay real'} · suites planned`;
    }

    this.paint = B => { paintStats(B); paintAuditor(B.auditor); paintPilot(B.notebook_pilot); paintPlan(B); DC.hydrate(el); };
  },
  async enter(el) {
    let B = null;
    try { B = await DC.api('/lab/benchmark'); } catch (e) { B = null; }
    DC.markSample(!B);
    if (B) this.paint(B);
  },
});
