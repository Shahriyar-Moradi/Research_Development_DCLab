DC.view('compute', {
  /* Compute & jobs, real data only: /api/ops/jobs lists what ran on this machine (project stage runs, intern sessions,
     draft data pipelines, legacy research runs) and /api/ops/jobs/{id} gives one job's logs, metrics, artifacts and
     reproducibility record. Model spend comes from the gateway's usage log; sandboxes / GPU jobs are not switched on; the page says so. */
  init(el) {
    const { $, $$, esc, api, toast } = DC;
    const S = { data: null, sel: null, sig: '', stop: null };
    const KIND = { stage: 'Stage run', intern: 'Intern session', data: 'Data pipeline', research: 'Research run' };
    const PILL = { running: 'accent', queued: 'outline', done: 'ok', failed: 'bad' };
    const pillCls = j => (j.label === 'interrupted' || j.label === 'budget used up' || j.label === 'paused' ? 'warn' : PILL[j.status] || '');
    const int = v => Number(v).toLocaleString('en-US');
    const secs = v => (v == null ? '—' : v < 90 ? Number(v).toFixed(1) + ' s' : v < 5400 ? (v / 60).toFixed(1) + ' min' : (v / 3600).toFixed(1) + ' h');
    const when = iso => { const d = new Date(iso); return iso && !isNaN(d) ? d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }) : ''; };
    const clock = iso => { const d = new Date(iso); return iso && !isNaN(d) ? d.toLocaleTimeString('en-GB', { hour12: false }) : ''; };
    function shortId(j) {
      const parts = j.id.split('-');
      if (j.kind === 'stage') return `${(j.project_id || '').slice(0, 6)}·${j.stage}·${parts[parts.length - 1]}`;
      if (j.kind === 'data') return 'data·' + parts[parts.length - 1].slice(0, 7);
      if (j.kind === 'intern') return 'intern·' + (j.session_id || '').slice(0, 6);
      return 'run·' + (j.run_id || '').slice(0, 6);
    }
    function cells(j) {
      if (j.kind === 'stage') return [j.project || 'Project', `${j.title} · ${j.subtitle}`];
      if (j.kind === 'intern') return [j.title, `${j.subtitle}${j.project ? ' · ' + j.project : ''}`];
      if (j.kind === 'data') return [j.project || 'Draft on Home', `${j.subtitle} · ${j.title}`];
      return [j.title, `${j.subtitle}${j.project ? ' · ' + j.project : ''}`];
    }
    const who = by => (by === 'agent' ? '<span class="avatar sm ai" title="Started by the intern or an agent">AI</span>'
      : by === 'human' ? '<span class="avatar sm" title="Started by a person in this workspace">You</span>' : '<span class="muted small">—</span>');

    /* ---------- overview, spend, where ---------- */
    function drawStats(d) {
      const t = d.totals, sec = t.seconds_this_month, st = $$('#ops-stats .stat', el);
      const local = sec.stage + sec.data + sec.research;
      st[0].innerHTML = `<span class="v">${t.running}</span><span class="l">job${t.running === 1 ? '' : 's'} running now · this machine${t.queued ? ` · ${t.queued} queued` : ''}</span>`;
      st[1].innerHTML = `<span class="v${t.approvals ? ' warn' : ''}">${t.approvals}</span><span class="l">waiting for approval · ${t.failed} failed or interrupted run${t.failed === 1 ? '' : 's'} in total</span>${t.approvals ? '<div class="row"><button type="button" class="link-btn small" data-pane-go="spend">Review →</button></div>' : ''}`;
      st[2].innerHTML = `<span class="v">${esc(secs(local))}</span><span class="l">compute time this month · stages ${esc(secs(sec.stage))} · data ${esc(secs(sec.data))}${sec.research ? ' · research ' + esc(secs(sec.research)) : ''} · intern sessions ${esc(secs(sec.intern))} (their stages are counted above)</span>`;
      const sp = d.spend || {};
      st[3].innerHTML = sp.requests
        ? `<span class="v">€${Number(sp.eur || 0).toFixed(2)}</span><span class="l">model spend this month · ${int(sp.requests)} request${sp.requests === 1 ? '' : 's'} (${int(sp.input_tokens)} in · ${int(sp.output_tokens)} out tokens)${sp.unpriced ? ` · <b>${int(sp.unpriced)} without a price</b>, counted in tokens only` : ''}${sp.local ? ` · ${int(sp.local)} on a local model, free` : ''} · jobs on this machine are not billed</span>`
        : `<span class="v">€0.00</span><span class="l">model spend this month · no model request yet · jobs on this machine are not billed</span>`;
      $('#jobs-pill', el).textContent = `${t.total} job${t.total === 1 ? '' : 's'} · this machine`;
      $('#spend-sub', el).textContent = sp.requests ? (sp.metered ? 'model requests, this month' : 'model requests, this month · some without a price') : 'no model request this month';
      $('#spend-chart', el).innerHTML = (sp.by_project || []).length && sp.eur > 0
        ? DC.charts.barsH(sp.by_project.map(p => ({ label: p.name, value: p.eur, valueText: '€' + p.eur.toFixed(2) })), { width: 520, labelW: 160, valW: 70, min: 0, aria: 'Model spend by project this month' }) + `<p class="xs muted">${esc(sp.note || '')}</p>`
        : `<div class="empty">${esc(sp.note || 'No model request this month.')}</div>`;
      const ap = d.approvals;
      const pill = $('#approvals-pill', el);
      pill.textContent = ap.length; pill.className = 'pill ' + (ap.length ? 'warn' : 'outline');
      $('#approvals-list', el).innerHTML = ap.length ? ap.map(a => `<a class="list-item" href="#${a.page}" data-ops-project="${esc(a.project_id)}">
          <div class="li-main"><span class="li-title">${esc(a.project || a.project_id)}: ${a.gate === 'solution' ? 'sign the solution' : 'approve opening the holdout'}</span><span class="li-sub">${esc(a.what)} The project's policy requires it.</span></div>
          <div class="li-side"><span class="pill warn">${esc(a.gate)} gate</span><span>Open →</span></div></a>`).join('')
        : '<div class="empty">No approval is waiting. Gates are off by default; a project\'s policy can require the owner to sign the solution or to approve opening the holdout.</div>';
      const tab = $('.ptab[data-ptab="spend"]', el);
      if (tab) { let n = $('.n', tab); if (ap.length && !n) { tab.insertAdjacentHTML('beforeend', '<span class="n"></span>'); n = $('.n', tab); } if (n) { if (ap.length) n.textContent = ap.length; else n.remove(); } }
      const m = d.machine || {};
      $('#where-local', el).textContent = `${m.cpus ? m.cpus + ' CPU cores · ' : ''}${[m.system, m.arch].filter(Boolean).join(' ')}${m.python ? ' · Python ' + m.python : ''}. Runs project stages, intern sessions, data pipelines and notebook export: ${int(t.total)} job${t.total === 1 ? '' : 's'} so far. Not metered.`;
      DC.setNavCount('compute', t.running || '');
    }

    /* ---------- failures (package 12.4): the job table's failed and interrupted jobs, with Retry ---------- */
    function drawFailures(rows) {
      const panel = $('#failures-panel', el);
      panel.hidden = !rows.length;
      $('#failures-table tbody', el).innerHTML = rows.map(f => `<tr><td><div class="cell-main">${esc(f.what)}</div><div class="cell-sub mono">${esc(f.id)}${f.stages ? ' · ' + esc(f.stages.join(', ')) : ''}</div></td>
        <td class="small"><span class="pill ${f.status === 'interrupted' ? 'warn' : 'bad'}">${esc(f.status)}</span> ${esc(f.error || '')}</td>
        <td class="num mono">${esc(f.attempts == null ? '' : f.attempts)}</td><td class="small">${esc(when(f.finished))}</td>
        <td><button type="button" class="btn sm" data-job-act="retry" data-job-id="${esc(f.id)}" data-job-done="Queued again">Retry</button></td></tr>`).join('');
    }

    /* ---------- jobs table ---------- */
    function drawJobs(jobs) {
      const tbody = $('#jobs-table tbody', el);
      if (!jobs.length) {
        tbody.innerHTML = '<tr><td colspan="7"><div class="empty">No jobs yet. Stage runs, intern sessions and data pipelines on Home appear here as they run.</div></td></tr>';
        $('#job-detail', el).hidden = true;
        return;
      }
      tbody.innerHTML = jobs.map(j => {
        const [main, sub] = cells(j);
        const t = esc(secs(j.seconds)) + (j.status === 'running' ? '<div class="cell-sub">so far</div>' : '');
        return `<tr data-job="${esc(j.id)}" class="${j.id === S.sel ? 'selected' : ''}"><td class="mono small" title="${esc(j.id)}">${esc(shortId(j))}</td><td><div class="cell-main">${esc(main)}</div><div class="cell-sub">${esc(sub)}</div></td><td class="small">${esc(j.where)}</td><td><span class="pill ${pillCls(j)}">${esc(j.label)}</span><div class="cell-sub">${esc(when(j.started))}</div></td><td class="num mono">${t}</td><td class="num mono"><span class="muted" title="Not metered">—</span></td><td>${who(j.by)}</td></tr>`;
      }).join('');
    }
    $('#jobs-table tbody', el).addEventListener('click', e => { const tr = e.target.closest('tr[data-job]'); if (tr) select(tr.dataset.job, true); });

    /* ---------- one job ---------- */
    async function select(id, user) {
      S.sel = id;
      $$('#jobs-table tbody tr', el).forEach(r => r.classList.toggle('selected', r.dataset.job === id));
      let d;
      try { d = await DC.client.opsJob(id); } catch (err) { if (user) toast(err.message, { ok: false }); return; }
      if (S.sel !== id) return;
      drawDetail(d);
    }
    function drawDetail(d) {
      const j = d.job;
      $('#job-detail', el).hidden = false;
      $('#jd-eyebrow', el).textContent = `${KIND[j.kind] || j.kind} · ${j.where} · ${j.label}`;
      $('#jd-title', el).textContent = cells(j).join(' · ');
      const acts = [];
      if (j.kind === 'intern') acts.push(`<a class="btn sm" href="#intern" data-ops-session="${esc(j.session_id)}">Open the session →</a>`);
      if (j.project_id && (j.kind !== 'data' || !j.draft_open)) acts.push(`<a class="btn sm" href="#project" data-ops-project="${esc(j.project_id)}">Open the project →</a>`);
      if (j.kind === 'data' && j.draft_open) acts.push(`<a class="btn sm" href="#home" data-ops-draft="${esc(j.draft_id)}">Open the draft on Home →</a>`);
      if (j.kind === 'research') acts.push('<a class="btn sm" href="/classic">Open in the classic UI →</a>');
      // Stop and Retry act on the job table (package 10.3): a stop takes effect at the job's next checkpoint (between stages).
      if (d.stop) acts.push(d.stop.requested ? '<span class="pill outline">stopping…</span>'
        : `<button type="button" class="btn sm danger" data-job-act="stop" data-job-id="${esc(d.stop.job_id)}" data-job-done="Stop requested: it ends after the step it is on">Stop</button>`);
      if (d.retry) acts.push(`<button type="button" class="btn sm" data-job-act="retry" data-job-id="${esc(d.retry.job_id)}" data-job-done="Queued again">Retry</button>`);
      $('#jd-actions', el).innerHTML = acts.join('');
      const lvl = { dim: 't-dim', ok: 't-ok', warn: 't-warn', bad: 't-bad', acc: 't-acc' };
      $('#jd-logs', el).innerHTML = d.logs.length ? d.logs.map(l => `${l.at ? `<span class="t-dim">${esc(clock(l.at))}</span> ` : '         '}<span class="${lvl[l.level] || ''}">${esc(l.text)}</span>${l.proof ? ` <span class="t-dim">[${esc(l.proof.join(', '))}]</span>` : ''}`).join('\n')
        : '<span class="t-dim">No log lines for this job.</span>';
      const note = $('#jd-log-note', el); note.hidden = !d.log_note; note.textContent = d.log_note || '';
      $('#jd-metrics', el).innerHTML = d.metrics.length ? `<dl class="kv small">${d.metrics.map(m => `<dt>${esc(m.label)}</dt><dd class="mono">${esc(m.value)}</dd>`).join('')}</dl>
        <p class="xs muted">Measured on this machine, which is not billed. Model spend is on the Spend &amp; approvals tab.</p>` : '<div class="empty">No metrics for this job.</div>';
      $('#jd-artifacts', el).innerHTML = (d.artifacts.length ? `<div class="list small">${d.artifacts.map(a => a.href
        ? `<a class="list-item" href="${esc(a.href)}" download><code>${esc(a.label)}</code><span class="li-side">${esc(a.note || '')}</span></a>`
        : `<a class="list-item" href="#project" data-ops-project="${esc(a.project_id)}"><code>${esc(a.label)}</code><span class="li-side">${esc(a.note || '')} · open →</span></a>`).join('')}</div>` : '')
        + (d.artifact_note ? `<div class="empty">${esc(d.artifact_note)}</div>` : '');
      $('#jd-repro', el).innerHTML = (d.repro.length ? `<dl class="kv small">${d.repro.map(r => `<dt>${esc(r.label)}</dt><dd class="${r.mono ? 'mono' : ''}" data-style="overflow-wrap:anywhere">${esc(r.value)}</dd>`).join('')}</dl>` : '')
        + (d.repro_note ? `<p class="xs muted">${esc(d.repro_note)}</p>` : '');
      DC.hydrate($('#job-detail', el));
    }

    /* ---------- Stop and Retry ---------- */
    el.addEventListener('click', async e => {
      const b = e.target.closest('[data-job-act]');
      if (!b) return;
      b.disabled = true;
      try {
        await (b.dataset.jobAct === 'stop' ? DC.client.stopJob(b.dataset.jobId) : DC.client.retryJob(b.dataset.jobId));
        toast(b.dataset.jobDone);
        await load();
        if (S.sel) select(S.sel, false);
      } catch (err) { toast(err.message, { ok: false }); b.disabled = false; }
    });

    /* ---------- links to the pages that own each job ---------- */
    el.addEventListener('click', e => {
      const p = e.target.closest('[data-ops-project]');
      if (p) { DC.state.project = 'p:' + p.dataset.opsProject; DC.state.projectId = null; DC.currentProject.clear(); return; }
      const s = e.target.closest('[data-ops-session]');
      if (s) { DC.state.internSession = s.dataset.opsSession; return; }
      const d = e.target.closest('[data-ops-draft]');
      if (d) { try { localStorage.setItem('dclab-home-draft', d.dataset.opsDraft); } catch (err) { /* storage blocked */ } }
    });

    /* ---------- loading and polling while something runs ---------- */
    async function load() {
      let d;
      try { d = await DC.client.opsJobs({ quiet: true }); } catch (err) {  // quiet: the table says it and offers Try again
        DC.states.failed($('#jobs-table tbody', el), err, { lead: 'Could not read the jobs: ', retry: load, cols: 7 });
        return false;
      }
      S.data = d;
      DC.markSample(false);
      drawStats(d);
      drawFailures(d.failures || []);
      const jobs = d.jobs;
      const sig = jobs.map(j => `${j.id}:${j.status}:${j.label}:${j.seconds}`).join('|');
      if (sig !== S.sig) {
        S.sig = sig;
        drawJobs(jobs);
        const sel = jobs.find(j => j.id === S.sel) || jobs[0];
        if (sel) select(sel.id, false);
      }
      return d.totals.running + d.totals.queued > 0;
    }
    this.load = load;
    // Loads now, then every 2.5 s while a job runs or waits and the page is open. The promise ends with the first load,
    // so the page's own load is tracked (13.1: Loading… and a failed read) and a test waits for it (13.2).
    this.watch = () => {
      if (S.stop) S.stop();
      let firstDone;
      const first = new Promise(resolve => { firstDone = resolve; });
      S.stop = DC.poll(async () => {
        try { return DC.state.view !== 'compute' || !(await load()); } finally { firstDone(); }
      }, 2500);
      return first;
    };
  },
  enter() {
    return this.watch();
  },
});
