DC.view('lab', {
  init(el) {
    const { $, $$, esc, chip, chips, fmt, int } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chips:([A-Za-z0-9_,-]+)\}/g, (m, ids) => chips(ids.split(','))).replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id));
    const CH = [['adult', 'catboost', 'catboost', 0.9274], ['bank_marketing', 'lightgbm', 'baseline_lightgbm', 0.8029], ['breast_cancer', 'lightgbm', 'fe_ratios', 1.0], ['credit_default', 'ensemble', 'stacking_ensemble', 0.8005], ['german_credit', 'catboost', 'catboost', 0.8112], ['heart_disease', 'hist_gradient_boosting', 'hist_gradient_boosting', 0.9740], ['hyperack', 'ensemble', 'softvote_etbag_lgbmwinner_xgb', 0.9455], ['mushroom', 'catboost', 'catboost', 1.0], ['online_shoppers', 'xgboost', 'tuned_xgboost', 0.7737], ['spambase', 'lightgbm', 'lightgbm', 0.9878], ['wine_quality', 'extra_trees', 'extra_trees', 0.9160]];
    $('#champ-table tbody', el).innerHTML = CH.map(([d, f, r, v]) => `<tr><td><span class="cell-main">${d}</span></td><td>${f}</td><td class="mono small">${r}</td><td class="num mono">${v.toFixed(4)}</td></tr>`).join('');
    const TR = [
      ['tabular-classification', 'Tabular classification', 'active', 'the most mature track'], ['ml-methodology', 'ML methodology (evidence campaigns)', 'active', 'campaigns and rules'],
      ['agentic-ml-copilot', 'Agentic ML copilot', 'active', 'code lives in dclab_rnd/'], ['churn-prediction', 'Churn prediction', 'active', 'Telco'],
      ['llm-fine-tuning', 'LLM fine-tuning', 'active', 'data ready, no training run yet'], ['tabular-foundation-models', 'Tabular foundation models', 'active', 'needs a leakage-safe rerun'],
      ['workflow-model', 'Focused model for ML workflows', 'proposed', 'the policy model'], ['workflow-actions', 'Workflow-sized action units', 'proposed', 'the "cake" idea'],
      ['evaluation-and-trust', 'Evaluation and trust', 'proposed', 'end-to-end value unproven'], ['cross-industry-workflows', 'Workflows beyond data science', 'proposed', 'software bug-fixing first'],
      ['vision-scene-graphs', 'Vision and scene graphs', 'proposed', 'no implementation yet'], ['temporal-gnn', 'Temporal GNNs', 'proposed', 'object interactions over frames'],
      ['driving-maps', 'Driving and road-map graphs', 'proposed', 'safety-critical'], ['computer-vision', 'Computer vision', 'planned', ''], ['graph-neural-networks', 'Graph neural networks', 'planned', ''],
      ['time-series-forecasting', 'Time-series forecasting', 'planned', ''], ['nlp-and-text', 'NLP and text', 'planned', ''], ['anomaly-and-fraud-detection', 'Anomaly and fraud detection', 'planned', ''],
      ['recommender-systems', 'Recommender systems', 'planned', ''], ['causal-inference-and-experimentation', 'Causal inference', 'planned', ''], ['mlops-and-deployment', 'MLOps and deployment', 'planned', ''], ['data-science-foundations', 'Data science foundations', 'planned', ''],
    ];
    const cls = { active: 'ok', proposed: 'proof', planned: 'outline', open: 'info', paused: 'warn', concluded: 'accent', unknown: 'outline' };
    $('#tracks', el).innerHTML = TR.map(([k, t, s, n]) => `<div class="inset stack tight"><div class="spread"><b class="small">${esc(t)}</b><span class="pill ${cls[s]}">${s}</span></div><span class="mono xs muted">research/${k}</span>${n ? `<span class="xs muted">${esc(n)}</span>` : ''}</div>`).join('');
    $('#lab-new', el).addEventListener('click', () => { DC.reveal($('#cd-h', el)); $('#cd-h', el).focus(); });
    DC.hydrate(el);

    /* ---------- real mode: everything below comes from GET /api/lab/summary and GET /api/research ---------- */
    const ref = id => (DC.REC[id] ? chip(id) : `<span class="tag">${esc(id)}</span>`);
    const setCount = (pane, n) => { const b = $(`.ptab[data-ptab="${pane}"] .n`, el); if (b) b.textContent = String(n); };
    const plural = (n, w) => `${int(n)} ${w}${n === 1 ? '' : 's'}`;
    const day = iso => (iso ? String(iso).slice(0, 10) : '');
    const V = { crit: 0, designed: false };

    async function openFile(path, title) {
      if (!path) return;
      try {
        const f = await DC.api('/research/file?path=' + encodeURIComponent(path));
        DC.drawer.open({ eyebrow: `<span class="tag">${esc(path)}</span>`, title: title || path,
          html: `<pre class="code xs" data-style="white-space:pre-wrap;overflow-wrap:anywhere">${esc(f.text || (f.entries || []).join('\n'))}</pre>${f.truncated ? '<div class="xs muted">Truncated; open the file in the repository for the rest.</div>' : ''}` });
      } catch (e) { DC.toast(e.message, { ok: false }); }
    }
    el.addEventListener('click', e => {
      const f = e.target.closest('[data-open-file]');
      if (f) { e.preventDefault(); openFile(f.dataset.openFile, f.dataset.fileTitle); return; }
      const c = e.target.closest('[data-crit]');
      if (c && V.S) { const n = V.S.critic.examples.length; V.crit = (V.crit + Number(c.dataset.crit) + n) % n; paintCritic(V.S.critic); }
    });

    function paintStats(S) {
      const R = S.registry, C = S.critic, A = S.auditor;
      const camps = S.campaigns.length;
      $('#lab-stats', el).innerHTML = `
        <div class="stat"><span class="v">${int(R.experiments)}</span><span class="l">experiments in the registry · ${int(R.eligible)} eligible results · ${plural(R.datasets, 'dataset')}</span></div>
        <div class="stat"><span class="v">${camps}</span><span class="l">campaigns · ${int(S.index.campaign_records)} of ${int(S.index.records)} records in the evidence index</span></div>
        <div class="stat"><span class="v">${int(C.challenges)}</span><span class="l">critic challenges · ${int(C.kept)} kept · ${int(C.contradicted)} contradicted by the numbers</span><div class="row"><button type="button" class="link-btn small" data-pane-go="critic">How the gate works →</button></div></div>
        ${A ? `<div class="stat"><span class="v warn">${Math.round(A.recall * 100)}%</span><span class="l">blind auditor recall · ${A.found} of ${A.known} known leaks, no hints</span></div>`
            : '<div class="stat"><span class="v faint">—</span><span class="l">blind auditor replay not stored yet · <code>make verify-auditor</code></span></div>'}`;
    }

    function paintCampaigns(S) {
      const rows = S.campaigns.slice().sort((a, b) => String(a.last_completed_at || '9').localeCompare(String(b.last_completed_at || '9')));
      const pill = { done: 'ok', 'in progress': 'info', failed: 'bad', planned: 'outline' };
      $('#lab-camps', el).innerHTML = rows.map(c => {
        const indexed = c.record_ids.filter(id => DC.REC[id]);
        const shown = (indexed.length ? indexed : c.record_ids).slice(0, 3);
        const more = (indexed.length || c.record_ids.length) - shown.length;
        const ds = c.datasets.length;
        const sub = [ds ? `${plural(ds, 'dataset')}` : '', c.extends ? `extends ${esc(c.extends)}` : '', c.last_completed_at ? `last result ${day(c.last_completed_at)}` : ''].filter(Boolean).join(' · ');
        const count = c.completed === c.planned ? int(c.experiments) : `${int(c.completed)} / ${int(c.planned)}`;
        return `<tr><td><span class="cell-main">${esc(c.id)}</span><div class="cell-sub" title="${esc(c.datasets.join(', '))}">${sub}</div></td>
          <td class="small">${DC.linkIds(c.objective)}</td>
          <td class="num">${count}</td>
          <td><span class="pill ${pill[c.status] || 'outline'}">${esc(c.status)}</span></td>
          <td><div class="stack tight"><span class="ev-list">${shown.map(ref).join('')}${more > 0 ? `<span class="tag">+${more}</span>` : ''}</span>${c.report ? `<button type="button" class="link-btn small" data-style="align-self:flex-start" data-open-file="${esc(c.report)}" data-file-title="${esc(c.title || c.id)}">Report →</button>` : ''}</div></td></tr>`;
      }).join('') || '<tr><td colspan="5"><div class="empty">No campaign folder under evidence/campaigns/ yet.</div></td></tr>';
      const foot = $('#lab-camps-foot', el);
      foot.hidden = false;
      foot.innerHTML = `Read from <code>evidence/campaigns/*/</code> (immutable result files). ${int(S.index.campaign_records)} of their records are in the evidence index; a campaign without index records (such as the auditor replay) is cited by its result file.`;
    }

    function paintDesigner(S) {
      const cmds = S.commands.campaign || [], after = S.commands.after || [];
      if (!cmds.length) return;
      const box = $('#lab-designer', el);
      const keep = V.designed ? $('#cd-h', el).value : '';
      const pick = V.designed ? $('#cd-c', el).value : cmds[0].target;
      $('#cd-sub', el).textContent = 'Campaigns are fixed plans in code, launched from the command line today';
      box.innerHTML = `
        <div class="stack tight">
          <div class="field"><label for="cd-h">Hypothesis</label><textarea id="cd-h" rows="3" placeholder="Write the hypothesis and its decision rule before you look at any score."></textarea></div>
          <div class="xs muted">There is no launch button yet. A new hypothesis becomes a campaign module with a fixed plan; deterministic code runs and scores it, and the LLM critic only reviews the records, checked by the gate (<code>make critic-gate</code>).</div>
        </div>
        <div class="stack tight">
          <div class="field"><label for="cd-c">Campaign command</label><select id="cd-c">${cmds.map(c => `<option value="${esc(c.target)}">${esc(c.command)} — ${esc(c.help)}</option>`).join('')}</select></div>
          <pre class="code" id="cd-cmd"></pre>
          <div class="xs muted" id="cd-runs"></div>
          ${after.length ? `<div class="xs muted">After new results: ${after.map(a => `<code>${esc(a.command)}</code>`).join(' then ')}, so the registry, the index and this page pick them up.</div>` : ''}
          <div class="row"><button type="button" class="btn primary" data-copy="#cd-cmd" data-copy-label="Command">Copy command</button></div>
        </div>`;
      const sel = $('#cd-c', box);
      const show = () => { const c = cmds.find(x => x.target === sel.value) || cmds[0]; $('#cd-cmd', box).textContent = c.command; $('#cd-runs', box).innerHTML = c.runs ? `Runs <code>${esc(c.runs)}</code>` : ''; };
      sel.value = cmds.some(c => c.target === pick) ? pick : cmds[0].target;
      sel.addEventListener('change', show);
      $('#cd-h', box).value = keep;
      show();
      V.designed = true;
    }

    function paintChampions(S) {
      const rows = S.champions;
      $('#champ-table tbody', el).innerHTML = rows.map(c => `<tr><td><span class="cell-main">${esc(c.dataset)}</span>${c.record && DC.REC[c.record] ? ` ${chip(c.record, 'card')}` : ''}<div class="cell-sub mono" title="${esc(c.source)}">run ${esc(c.run_id)} · ${esc(c.suite)}</div></td><td>${esc(c.family)}</td><td class="mono small">${esc(c.recipe)}</td><td class="num mono">${fmt(c.value)}</td></tr>`).join('')
        || '<tr><td colspan="4"><div class="empty">No eligible result in the registry yet.</div></td></tr>';
      const perfect = rows.filter(c => c.value >= 0.99995).map(c => c.dataset);
      const R = S.registry;
      $('#champ-foot', el).innerHTML = `${perfect.length ? `A ROC-AUC of 1.0000 on ${perfect.map(esc).join(' and ')} says ${perfect.length === 1 ? 'that benchmark is' : 'those benchmarks are'} easy, not that the method is perfect. ` : ''}Champions come from ${int(R.experiments)} registry experiments (${int(R.eligible)} eligible) via <code>${esc(R.command)}</code>. Each value is a point estimate from its own evaluation with no interval shown here, so treat small differences with care.`;
    }

    function paintPitfalls(S) {
      const P = S.pitfalls;
      $('#pit-body', el).innerHTML = P.map(p => `<tr><td><span class="strong">${esc(p.name)}</span><div class="cell-sub">${esc(p.datasets.join(', '))}</div></td>
        <td><span class="mono">${esc(p.cost.text || '—')}</span>${p.cost.detail ? `<div class="cell-sub">${esc(p.cost.detail)}</div>` : ''}</td>
        <td class="small">${DC.linkIds(p.lesson || '')}</td><td>${ref(p.id)}</td></tr>`).join('')
        || '<tr><td colspan="4"><div class="empty">No pitfall result under evidence/campaigns/pitfalls_v1/results/ yet.</div></td></tr>';
      setCount('pitfalls', P.length);
      const foot = $('#pit-foot', el);
      foot.hidden = false;
      foot.innerHTML = 'Cost = reported minus honest ROC-AUC (mean over seeds) on the dataset where it was largest; each record lists its limits. Re-measure with <code>make pitfalls</code>.';
    }

    function paintCritic(C) {
      $('#crit-sofar', el).innerHTML = `<b>So far:</b> ${int(C.kept)} challenges kept, ${int(C.contradicted)} dropped as contradicted by the numbers, across ${int(C.experiments)} campaign experiments. ${int(C.recomputable_rules)} selection rules were recomputed from the recorded numbers; ${int(C.rule_inconsistencies)} disagreed with the recorded choice. Run <code>${esc(C.command)}</code> to see them.`;
      const box = $('#crit-example', el);
      const n = C.examples.length;
      if (!n) {
        box.innerHTML = '<div class="panel-head"><h3>A dropped challenge</h3></div><div class="panel-body"><div class="empty">No challenge has been contradicted by the numbers so far.</div></div>';
        return;
      }
      const x = C.examples[Math.min(V.crit, n - 1)];
      const chosen = x.recorded === x.recomputed ? `the record chose <code>${esc(x.recorded)}</code> too` : `the record chose <code>${esc(x.recorded)}</code>`;
      box.innerHTML = `<div class="panel-head"><h3>A dropped challenge</h3>${ref(x.experiment_id)}</div>
        <div class="panel-body small stack tight">
          <p class="muted">“${esc(x.issue)}”</p>
          <p><b>Gate:</b> contradicted. Recomputing the stage's own rule (${esc(x.rule || 'its recorded selection rule')}) from the recorded numbers selects <code>${esc(x.recomputed)}</code>; ${chosen}. The challenge disputes a rule the numbers confirm.</p>
          <div class="xs muted">${esc(x.dataset)} · ${esc(String(x.kind || '').replace(/_/g, ' '))} · ${esc(x.claim_id || '')} · severity ${esc(x.severity || '—')}</div>
          <p class="muted">Why it matters: without the gate, a confident but wrong critique would have become a "lesson" in the training data.</p>
          ${n > 1 ? `<div class="row"><button type="button" class="btn sm" data-crit="-1">‹ Previous</button><span class="xs muted">${Math.min(V.crit, n - 1) + 1} of ${n}</span><button type="button" class="btn sm" data-crit="1">Next ›</button></div>` : ''}
        </div>`;
    }

    function paintMap(M, status) {
      const ORDER = { active: 0, open: 1, proposed: 2, paused: 3, planned: 4, concluded: 5, unknown: 6 };
      const tracks = Object.values(M.tracks || {}).map(t => Object.assign({}, t, { st: status[t.name] || { kind: 'unknown', note: t.status || '' } }))
        .sort((a, b) => (ORDER[a.st.kind] ?? 9) - (ORDER[b.st.kind] ?? 9) || a.title.localeCompare(b.title));
      $('#tracks', el).innerHTML = tracks.map(t => {
        const k = t.counts || {};
        const counts = [['experiment', k.experiments], ['campaign', k.campaigns], ['notebook', k.notebooks], ['report', k.reports]].filter(([, v]) => v).map(([w, v]) => plural(v, w)).join(' · ')
          || (k.files > 1 ? plural(k.files, 'tracked file') : 'README only');
        return `<div class="inset stack tight"><div class="spread"><b class="small">${esc(t.title)}</b><span class="pill ${cls[t.st.kind] || 'outline'}">${esc(t.st.kind)}</span></div>
          <span class="mono xs muted">${esc(t.path)}</span>${t.st.note ? `<span class="xs muted">${esc(t.st.note)}</span>` : ''}
          <div class="spread"><span class="xs faint">${counts}</span>${t.readme ? `<button type="button" class="link-btn small" data-open-file="${esc(t.readme)}" data-file-title="${esc(t.title)}">README →</button>` : ''}</div></div>`;
      }).join('') || '<div class="empty">No research track found under research/.</div>';
      const by = {};
      tracks.forEach(t => { by[t.st.kind] = (by[t.st.kind] || 0) + 1; });
      $('#map-pill', el).textContent = [plural(tracks.length, 'track')].concat(['active', 'open', 'proposed', 'planned', 'paused', 'concluded'].filter(s => by[s]).map(s => `${by[s]} ${s}`)).join(' · ');
      setCount('map', tracks.length);
    }

    this.paint = S => {
      V.S = S;
      paintStats(S); paintCampaigns(S); paintDesigner(S); paintChampions(S); paintPitfalls(S); paintCritic(S.critic);
      DC.hydrate(el);
    };
    this.paintMap = paintMap;
  },
  async enter(el) {
    let S = null;
    try { S = await DC.api('/lab/summary'); } catch (e) { S = null; }
    DC.markSample(!S);
    if (!S) return;  // offline demo: the sample stays
    this.paint(S);
    try { this.paintMap(await DC.api('/research'), S.track_status || {}); } catch (e) { /* the research map keeps its last content */ }
  },
});
