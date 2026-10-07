DC.view('integrations', {
  init(el) {
    const { $, esc, chip, icon } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id)).replace(/\$\{icon:([a-z]+)\}/g, (m, n) => icon(n));
    const T = [
      ['search_evidence', 'Filter-then-rank search over rules, workflows, datasets, experiments, precedents, pitfalls and findings', 'evidence'],
      ['get_record', 'Return one full evidence record by ID', 'evidence'], ['get_rules', 'List the model-building rules, optionally one category', 'evidence'],
      ['plan_next_stage', 'The next stage, its method, its rules and precedents for this dataset', 'evidence'], ['review_code', 'Static methodology review of one cell or script', 'evidence'],
      ['list_samples', 'Datasets the R&D studied, with their solutions', 'project'], ['create_project', 'Create a notebook project', 'project'],
      ['use_sample', 'Load a studied dataset with its profile and solution suggestion', 'project'], ['describe_data', 'Rows, columns, candidates for target, time, ID and text', 'project'],
      ['propose_solution', 'Audit columns for a target and propose a solution with proof', 'project'], ['set_solution', 'Save the solution; clears earlier stage results', 'project'],
      ['set_settings', 'How many rows the stages use', 'project'], ['run_stage', 'Run one stage; stages run in order', 'project'],
      ['run_all', 'Run the remaining stages; the final one uses the holdout once', 'project'], ['approve_stage', 'Approve a stage, optionally choosing another option', 'project'],
      ['get_results', 'Records of the finished stages', 'project'], ['ask_project', 'Ask the project; answers cite records', 'project'],
      ['export_notebook', 'Write the runnable notebook and the report', 'project'],
    ];
    $('#tool-table tbody', el).innerHTML = T.map(([n, d, g]) => `<tr><td class="mono small">${n}</td><td class="small">${esc(d)}</td><td><span class="pill ${g === 'evidence' ? 'proof' : 'accent'}">${g}</span></td></tr>`).join('');
    DC.hydrate(el);
  },

  /* Real mode: everything below comes from GET /api/platform/integrations (this server's own routes, tools and Makefile). */
  async enter(el) {
    const { $, esc } = DC;
    const md = t => esc(t || '').replace(/`([^`]+)`/g, '<code>$1</code>');
    let d;
    try { d = await DC.client.platformIntegrations(); } catch (e) { DC.markSample(true); return; }
    DC.markSample(false);
    const m = d.mcp, n = m.count, mi = m.ml_intern_tools || {};
    const stat = (v, l) => `<div class="stat"><span class="v">${esc(v)}</span><span class="l">${l}</span></div>`;
    const c = d.connectors, ready = c.ready || {}, readyN = Object.values(ready).filter(Boolean).length;
    const missing = [!ready.kaggle && 'Kaggle needs credentials', !ready.database && 'no database connection', !ready.cloud && 'no cloud storage library'].filter(Boolean);

    /* header and overview */
    const pill = $('#int-pill', el);
    pill.className = 'pill ' + (m.mounted ? 'ok' : 'warn');
    pill.textContent = m.mounted ? 'MCP mounted at /mcp' : 'MCP not mounted here';
    $('#int-stats', el).innerHTML =
      stat(n, `DCLab tools on the MCP endpoint · ${m.mounted ? 'mounted on this server' : 'not mounted on this server'}`) +
      stat(mi.verified_total != null ? mi.verified_total : '—', mi.verified_total != null
        ? `tools in ML Intern mode when last verified (${mi.verified_dclab} DCLab + ${mi.own} of its own)${mi.current ? '' : ` · DCLab now has ${n}, so expect ${mi.expected_now}; not re-verified`}`
        : 'tools in ML Intern mode: no recorded verification') +
      stat(d.rest.length, `REST routes in ${d.rest_groups.length} groups on this server`) +
      stat(readyN, `of 4 data connectors ready on this server${missing.length ? ' · ' + esc(missing.join(' · ')) : ''}`);

    /* MCP endpoint panel: real URL in every snippet */
    const url = m.mounted ? m.url : m.standalone.url;
    $('#mcp-pill', el).className = 'pill ' + (m.mounted ? 'ok' : 'warn');
    $('#mcp-pill', el).textContent = m.mounted ? 'mounted · tested' : 'not mounted';
    const note = $('#mcp-note', el);
    note.hidden = m.mounted;
    if (!m.mounted) note.innerHTML = `<span class="ic">${DC.icon('alert')}</span><span>${md(m.note)} The snippets below point at the standalone server (<code>${esc(m.standalone.make)}</code>).</span>`;
    $('#mcp-url', el).textContent = url;
    $('#mcp-standalone', el).innerHTML = m.mounted ? `or standalone: <code>${esc(m.standalone.command)}</code>` : `start it with <code>${esc(m.standalone.make)}</code>`;
    $('#cfg-chatui', el).textContent = `MCP_SERVERS=[{"name": "DCLab notebook", "url": ${JSON.stringify(url)}}]\nMCP_ALLOW_INSECURE_URLS=true`;
    const cn = $('#cfg-chatui-note', el);
    cn.hidden = false;
    cn.innerHTML = `<code>${esc(d.chat_ui.command)}</code> writes these two lines into Chat UI's <code>.env.local</code> for you.`;
    $('#cfg-claude', el).textContent = `claude mcp add --transport http dclab ${url}`;
    $('#cfg-generic', el).textContent = JSON.stringify({ mcpServers: { dclab: { type: 'http', url } } });
    $('#mcp-tools-link', el).textContent = `See the ${n} tools →`;

    /* MCP tools tab */
    const tab = $('.ptab[data-ptab="tools"] .n', el);
    if (tab) tab.textContent = n;
    const groups = m.tools.reduce((a, t) => (a[t.group] = (a[t.group] || 0) + 1, a), {});
    $('#tool-sub', el).textContent = `${groups.evidence || 0} evidence tools read the index · ${groups.project || 0} project tools drive the notebook stages · ${groups.graph || 0} graph tools ask the validator. Every project move is checked as the actor “agent”.`;
    const gcls = { evidence: 'proof', project: 'accent', graph: 'info' };
    $('#tool-table tbody', el).innerHTML = m.tools.map(t => `<tr><td class="mono small">${esc(t.name)}</td><td class="small">${esc(t.description)}${t.required.length ? `<div class="xs muted mono">needs: ${t.required.map(esc).join(', ')}</div>` : ''}</td><td><span class="pill ${gcls[t.group] || ''}">${esc(t.group)}</span></td></tr>`).join('');

    /* Chat UI & VS Code */
    $('#chatui-term', el).innerHTML = `<span class="t-dim">$</span> make notebook        <span class="t-dim"># DCLab + /mcp</span>\n<span class="t-dim">$</span> ${esc(d.chat_ui.command)}         <span class="t-dim"># Chat UI on ${esc(d.chat_ui.url)}</span>\n<span class="t-dim">$</span> ${esc(d.chat_ui.intern_command)}  <span class="t-dim"># with ML Intern mode: ${esc(d.chat_ui.intern_url)}</span>`;
    $('#chatui-verified', el).textContent = mi.verified_total != null
      ? `Last verified with the real Chat UI: plain mode offered the ${mi.verified_dclab} DCLab tools; ML Intern mode offered ${mi.verified_total} (${mi.breakdown || `${mi.own} of ML Intern's own`}).` + (mi.current ? '' : ` DCLab serves ${n} tools now, so ML Intern mode should offer ${mi.expected_now}; that has not been re-verified.`)
      : 'No recorded verification of ML Intern mode was found.';
    $('#chatui-where', el).innerHTML = `${md(mi.where)} Sign-in with a Hugging Face OAuth app enables Jobs (GPU) from ML Intern mode.`;
    $('#vscode-pill', el).textContent = 'not packaged yet';
    $('#vscode-pill', el).className = 'pill outline';
    const vn = $('#vscode-note', el);
    vn.hidden = false;
    vn.innerHTML = `The extension is not packaged yet. The engine it would call is built: ${md(d.vscode.engine)}. The editor below is an <b>illustration</b> of the plan, not a running extension.`;

    /* API & CLI */
    const byPath = [];
    d.rest.forEach(r => { const last = byPath[byPath.length - 1]; if (last && last.path === r.path) last.methods.push(r.method); else byPath.push({ path: r.path, group: r.group, summary: r.summary, methods: [r.method] }); });
    $('#rest-sub', el).textContent = `${d.rest.length} routes in ${d.rest_groups.length} groups, read from this server (hidden aliases left out)`;
    let group = null;
    $('#rest-body', el).innerHTML = byPath.map(r => {
      const head = r.group !== group ? `<tr><td colspan="2" class="xs muted"><b>${esc(r.group)}</b></td></tr>` : '';
      group = r.group;
      return head + `<tr><td class="mono small">${esc(r.methods.join(' · '))} ${esc(r.path)}</td><td class="small muted">${esc(r.summary || '')}</td></tr>`;
    }).join('');
    $('#rest-foot', el).hidden = false;
    let section = null;
    $('#cli-body', el).innerHTML = d.cli.map(t => {
      const head = t.section !== section && t.section ? `<tr><td colspan="2" class="xs muted"><b>${esc(t.section)}</b></td></tr>` : '';
      section = t.section;
      return head + `<tr><td class="mono small">make ${esc(t.target)}</td><td class="small">${esc(t.help)}</td></tr>`;
    }).join('');
    $('#cli-wrap', el).hidden = !d.cli.length;

    /* Connectors: configured or not, and what to set; names and booleans only */
    const item = (title, sub, ok, label) => `<div class="list-item"><div class="li-main"><span class="li-title">${title}</span><span class="li-sub">${sub}</span></div><span class="pill ${ok ? 'ok' : 'warn'}">${esc(label)}</span></div>`;
    const db = c.database || {}, cl = c.cloud || {};
    $('#conn-list', el).innerHTML =
      item('Kaggle', esc(c.kaggle.note), c.kaggle.configured, c.kaggle.configured ? 'ready' : 'not configured') +
      item('Hugging Face Hub', esc(c.hf.note), c.hf.configured, c.hf.token ? 'ready · token' : 'ready · public only') +
      item('Databases <span class="muted small">read-only SQL</span>', esc(db.note) + (db.connections && db.connections.length ? ` Connections: ${db.connections.map(x => `<code>${esc(x)}</code>`).join(', ')}.` : ''), db.configured, db.configured ? `ready · ${db.connections.length}` : 'not configured') +
      item('S3 · Google Cloud Storage', esc(cl.note), cl.s3 || cl.gcs, [cl.s3 && 'S3', cl.gcs && 'GCS'].filter(Boolean).join(' · ') + (cl.s3 || cl.gcs ? ' ready' : 'not available'));
    $('#conn-max', el).textContent = c.max_bytes ? `up to ${Math.round(c.max_bytes / 1048576)} MB per import` : '';
    DC.hydrate(el);
  },
});
