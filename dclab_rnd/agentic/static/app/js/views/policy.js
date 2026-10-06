DC.view('policy', {
  init(el) {
    const { $, esc, charts, codeBlock, int } = DC;
    const NAMES = { critique_claim: 'critique a claim', grounded_qa: 'grounded Q&A', explain_experiment: 'explain an experiment', apply_selection_rule: 'apply a selection rule', leakage_judgment: 'leakage judgment', rule_reasoning: 'rule reasoning', workflow_steps: 'workflow steps' };
    const STAGE = { data: 'data', leakage: 'leakage', features: 'features', models: 'models', final: 'final', solution: 'solution' };
    const plural = (n, one, many) => `${int(n)} ${n === 1 ? one : (many || one + 's')}`;
    const P = this.P = { data: null, ex: 0, full: false };

    function drawExample() {
      const list = (P.data && P.data.examples) || [];
      const x = list[P.ex % Math.max(1, list.length)];
      $('#sft-next', el).hidden = list.length < 2;
      $('#sft-more', el).hidden = !x;
      if (!x) { $('#sft-example', el).innerHTML = '<div class="empty">No example: the corpus files are missing.</div>'; $('#sft-ex-meta', el).textContent = ''; return; }
      const m = x.metadata || {};
      $('#sft-ex-meta', el).textContent = `· ${NAMES[x.task] || x.task} · ${x.split === 'val' ? 'validation' : 'train'} line ${x.line}` + (m.experiment_id ? ` · ${m.experiment_id}` : m.rule_id ? ` · ${m.rule_id}` : m.workflow_id ? ` · ${m.workflow_id}` : m.record_id ? ` · ${m.record_id}` : '') + (m.dataset ? ` · ${m.dataset}` : '');
      $('#sft-example', el).innerHTML = x.messages.map(msg => {
        const lim = msg.role === 'assistant' ? 700 : 360;
        const t = P.full ? msg.content : msg.content.slice(0, lim) + (msg.content.length > lim ? '…' : '');
        const cut = P.full && msg.truncated ? `<div class="xs muted" data-style="margin-top:4px">Shown up to ${int(msg.content.length)} of ${int(msg.length)} characters; the full record is in ${esc(x.file)}.</div>` : '';
        return `<div class="inset small"><div class="eyebrow ${msg.role === 'assistant' ? 'accent' : ''}" data-style="margin-bottom:4px">${esc(msg.role)}</div><div data-style="white-space:pre-wrap;overflow-wrap:anywhere">${DC.linkIds(t)}</div>${cut}</div>`;
      }).join('');
      $('#sft-more', el).textContent = P.full ? 'Show less' : 'Show the full example';
      DC.hydrate(el);
    }
    $('#sft-more', el).addEventListener('click', () => { P.full = !P.full; drawExample(); });
    $('#sft-next', el).addEventListener('click', () => { P.ex += 1; P.full = false; drawExample(); });
    $('#traj-sharing-list', el).addEventListener('change', e => {
      const box = e.target.closest('[data-share]'); if (!box) return;
      DC.api(`/projects/${encodeURIComponent(box.dataset.share)}`, { method: 'PATCH', body: { settings: { share_for_training: box.checked } } })
        .then(() => {
          DC.toast(box.checked ? 'This project\'s runs can now be exported for training, without cell values or free text.' : 'This project is no longer exported for training.');
          const boxes = [...el.querySelectorAll('#traj-sharing-list [data-share]')];
          $('#traj-count', el).textContent = `${boxes.filter(b => b.checked).length} of ${boxes.length} projects opted in.`;
        })
        .catch(err => { box.checked = !box.checked; DC.toast(err.message, { ok: false }); });
    });

    this.render = d => {
      P.data = d;
      const c = d.corpus || {}, q = c.quality_gates || {}, g = d.critic_gate || {}, runs = (d.training && d.training.runs) || [];
      const ready = d.curriculum.filter(s => s.ready).length;
      // Overview stats
      if (c.available) {
        $('#ps-total', el).textContent = int(c.total);
        $('#ps-total-l', el).textContent = `examples in SFT corpus v3 · ${int(c.train)} train · ${int(c.val)} validation`;
        $('#ps-kept', el).textContent = int(q.challenges_kept || 0);
        $('#ps-kept-l', el).textContent = `critic challenges kept · ${plural(q.contradicted_challenges_dropped || 0, 'contradicted one')} dropped`;
        $('#ps-held', el).textContent = int((c.held_out_datasets || []).length);
      } else {
        $('#ps-total', el).textContent = '0'; $('#ps-total-l', el).textContent = 'examples: the SFT corpus v3 is not built';
        $('#ps-kept', el).textContent = '—'; $('#ps-kept-l', el).textContent = 'critic challenges kept (no corpus)';
        $('#ps-held', el).textContent = '—';
      }
      $('#ps-ready', el).innerHTML = `${ready} <small>of ${d.curriculum.length}</small>`;
      $('#ps-ready-l', el).textContent = 'curriculum stages with data ready · ' + (runs.length ? plural(runs.length, 'training run') + ' found' : 'no training run yet');

      // Curriculum, readiness computed by the server from the corpus and the project logs
      const F = { sft: 'policy.sft', traj: 'policy.traj' };
      $('#pol-curriculum', el).innerHTML = d.curriculum.map(s => `<div class="inset stack tight"${F[s.key] ? ` data-f="${F[s.key]}"` : ''}><div class="spread"><span class="eyebrow">Stage ${s.stage}</span><span class="pill ${esc(s.cls)}">${esc(s.status)}</span></div><b>${esc(s.name)}</b><span class="small muted">${esc(s.detail)}</span></div>`).join('');
      const t = d.trajectories;
      /* A6.1: a project's runs are exported for training only when its owner turns this on (off by default) */
      $('#traj-sharing', el).hidden = !(t.sharing || []).length;
      $('#traj-sharing-list', el).innerHTML = (t.sharing || []).map(s => `<div class="list-item"><label class="switch"><input type="checkbox" data-share="${esc(s.id)}"${s.on ? ' checked' : ''}><span class="track"></span><span>${esc(s.name || s.id)}</span></label></div>`).join('');
      $('#traj-foot', el).innerHTML = `${esc(t.export || '')} <span id="traj-count">${t.opted_in || 0} of ${(t.sharing || []).length} projects opted in.</span> Blocked moves are as valuable as good ones: the model learns the boundary, and the recovery that followed.`;
      const u = t.usable || { trajectories: 0, target: t.target, projects: 0, min_projects: 10, min_decisions: t.min_moves };
      $('#pol-curriculum-foot', el).textContent = (u.error ? `The count could not be read (${u.error}); the numbers below are not the real count. ` : '') + `${u.trajectories} of ${u.target} distinct trajectories, from ${u.projects} of ${u.min_projects} opted-in projects. Stage 2 is data ready only at ${u.target} trajectories (each with at least ${u.min_decisions} decisions) from at least ${u.min_projects} projects: corpus v4 holds out whole projects, about one in five, as v3 holds out whole datasets. Until then python -m dclab_rnd.studio.corpus_v4 writes nothing.`;

      // Training data
      $('#sft-pill', el).textContent = c.available ? 'real' : 'not built'; $('#sft-pill', el).className = 'pill ' + (c.available ? 'ok' : 'warn');
      $('#sft-sub', el).textContent = c.available ? `Built from the evidence index on ${String(c.built).slice(0, 10)}; datasets held out entirely: ${(c.held_out_datasets || []).join(', ')}` : 'Not built: run make knowledge to write the corpus.';
      const by = c.by_task || {}, split = c.by_split_and_task || {};
      const rows = Object.entries(by).sort((a, b) => b[1] - a[1]);
      const max = Math.max(10, ...rows.map(r => r[1]));
      const top = Math.ceil(max / 20) * 20;
      $('#sft-chart', el).innerHTML = rows.length ? charts.barsH(rows.map(([k, v]) => ({ label: NAMES[k] || k, value: v, valueText: `${v} · ${(split.val || {})[k] || 0} val`, cls: k === 'leakage_judgment' || k === 'workflow_steps' ? 'proof' : '' })),
        { width: 380, labelW: 140, valW: 60, min: 0, max: top, rowH: 24, ticks: [0, top / 2, top], tickFmt: v => v, aria: 'Examples per task, with the validation share' }) : '<div class="empty">No corpus to chart.</div>';
      const linked = ids => ids.slice(0, 3).map(i => DC.linkIds(i)).join(', ') + (ids.length > 3 ? ` and ${ids.length - 3} more` : '');
      $('#sft-kv', el).innerHTML = c.available ? `<dt>Examples</dt><dd>${int(c.total)} · ${int(c.train)} train · ${int(c.val)} validation</dd>
        <dt>Files</dt><dd>${Object.entries(c.lines || {}).map(([f, n]) => `${esc(f)} ${int(n)} lines`).join(' · ')}${c.lines_match_manifest ? ' · match the manifest' : ' · <b>differ from the manifest</b>'}</dd>
        <dt>Formats</dt><dd>${Object.keys(c.formats || {}).length ? 'chat (TRL messages), Alpaca, ShareGPT' : '—'}</dd>
        <dt>Quality gates</dt><dd>${int(q.challenges_kept || 0)} critic challenges kept · ${int(q.contradicted_challenges_dropped || 0)} contradicted ones dropped · ${int(q.duplicates_removed || 0)} duplicates · ${int(q.rejected_by_path_or_hash_or_length || 0)} rejected for paths, hashes or length</dd>
        <dt>Critic gate now</dt><dd>${int(g.critic_challenges || 0)} challenges over ${int(g.experiments || 0)} experiments; ${int(g.contradicted_challenges || 0)} contradicted by recomputing the rule${(g.experiments_with_contradicted_challenges || []).length ? ` (in ${linked(g.experiments_with_contradicted_challenges)})` : ''}; ${int(g.rule_inconsistencies || 0)} recorded selections disagree with the rule</dd>
        <dt>Answer shape</dt><dd>Evidence · Interpretation · Decision · Risks · Next test</dd>`
        : `<dt>Corpus</dt><dd>${esc(c.reason || 'missing')}</dd>`;
      const pj = d.projects;
      const named = pj.items.filter(i => i.examples).slice(0, 4).map(i => `${esc(i.name)} ${i.examples}`).join(', ');
      const stages = Object.entries(pj.by_stage || {}).map(([k, v]) => `${STAGE[k] || k} ${v}`).join(', ');
      $('#sft-projects', el).innerHTML = `<b>From projects:</b> each project adds one example per completed stage, and one more when its solution forbids columns. ` +
        (pj.examples ? `Today ${plural(pj.with_examples, 'project gives', 'projects give')} ${plural(pj.examples, 'example')} (${named}${pj.with_examples > 4 ? ', …' : ''}; by stage: ${esc(stages)}). They are not in corpus v3 yet: export them with <code>python -m dclab_rnd.studio.sft</code>.`
          : pj.projects ? `None of the ${plural(pj.projects, 'project')} has a completed stage with a saved solution yet, so the notebook adds no example today.` : 'No project yet, so the notebook adds no example today.');
      drawExample();

      // Trajectory record: a real move from a transition log
      const s = t.sample;
      if (s) {
        const r = s.record;
        $('#traj-sub', el).textContent = `Move ${s.index} of ${s.of} in ${s.project.name}: ${r.move} by the ${r.actor === 'agent' ? 'agent' : 'person'}, ${String(r.status).replace('_', ' ')}${r.status === 'blocked' ? ' (the latest blocked move)' : ' (no move has been blocked yet, so this is the latest)'}`;
        $('#traj-pill', el).textContent = 'logged'; $('#traj-pill', el).className = 'pill ok';
        $('#traj-body', el).innerHTML = codeBlock(JSON.stringify(r, null, 2));
        const actors = Object.entries(t.by_actor).map(([a, v]) => `${a === 'agent' ? 'agent' : a === 'human' ? 'people' : esc(a)} ${v.moves} (${v.blocked} blocked, ${v.needs_approval} sent for approval)`).join(' · ');
        $('#traj-foot', el).innerHTML = `${plural(t.moves, 'move')} logged in ${plural(t.projects_with_log, 'project')}: ${actors}. Blocked moves are as valuable as good ones: the model learns the boundary, and the recovery that followed.`;
      } else {
        $('#traj-sub', el).textContent = 'One move from a project\'s transition log, as the model will see it';
        $('#traj-pill', el).textContent = 'none logged'; $('#traj-pill', el).className = 'pill outline';
        $('#traj-body', el).innerHTML = '<div class="empty">No project has logged a move yet. Every move on a project\'s Workflow page, and every move the intern makes, is written to the project\'s transition log (state → move → verdict).</div>';
        $('#traj-foot', el).textContent = 'Blocked moves are as valuable as good ones: the model learns the boundary, and the recovery that followed.';
      }

      // Expert review: no review store exists
      const rv = d.review || {};
      $('#review-queue', el).innerHTML = `<div class="empty">No examples queued for review. ${esc(rv.note || '')}${rv.awaiting_person ? ` Separately, ${plural(rv.awaiting_person, 'move')} in the transition logs waited for a person's approval; those approvals happen on each project's Workflow page.` : ''}</div>`;
      $('#rq-count', el).textContent = 'none queued';

      // Training runs found on disk
      const scripts = (d.training && d.training.scripts) || {};
      $('#runs-pill', el).textContent = runs.length ? plural(runs.length, 'run') + ' found' : 'none run yet';
      $('#runs-pill', el).className = 'pill ' + (runs.length ? 'ok' : 'outline');
      const runRows = runs.map(r => `<div class="list-item"><div class="li-main"><span class="li-title">${esc(r.name)}${r.base_model ? ' · ' + esc(r.base_model) : ''}${r.rank ? ' · LoRA r' + esc(r.rank) : ''}</span><span class="li-sub"><code>${esc(r.path)}</code> · ${esc(String(r.updated).slice(0, 10))}${r.epochs != null ? ` · ${Number(r.epochs).toFixed(1)} epochs` : ''}${r.eval_loss != null ? ` · last validation loss ${Number(r.eval_loss).toFixed(4)}` : ''}${r.unreadable ? ' · state file unreadable' : ''}</span></div><span class="pill ${r.adapter ? 'ok' : 'warn'}">${r.adapter ? 'adapter saved' : 'no adapter'}</span></div>`);
      if (!runs.length) runRows.push(`<div class="list-item"><div class="li-main"><span class="li-title">No training run on disk</span><span class="li-sub">Looked for an adapter or a trainer state under ${(d.training.searched || []).map(p => `<code>${esc(p)}</code>`).join(' and ')}.</span></div></div>`);
      runRows.push(`<div class="list-item"><div class="li-main"><span class="li-title">qwen2.5-1.5b · LoRA · corpus v3</span><span class="li-sub">${scripts['train_lora.py'] ? 'Script ready: <code>sft/train_lora.py</code>.' : 'Script missing: <code>sft/train_lora.py</code>.'} Teaches the answer format and the habit of reasoning from given evidence. Needs a GPU; not run in CI.</span></div><span class="pill ${scripts['train_lora.py'] ? 'ok' : 'warn'}">${scripts['train_lora.py'] ? 'ready' : 'missing'}</span></div>`);
      runRows.push(`<div class="list-item"><div class="li-main"><span class="li-title">open 8B · QLoRA · trajectories</span><span class="li-sub">Waits for stage 2 data: ${(t.usable || {}).trajectories || 0} of ${t.target} trajectories. No script, hardware or cost estimate yet.</span></div><span class="pill outline">plan</span></div>`);
      $('#runs-list', el).innerHTML = runRows.join('');
      $('#runs-foot', el).innerHTML = `Evaluation: <code>sft/eval_sft.py</code> on held-out datasets${scripts['eval_sft.py'] ? '' : ' (script missing)'}, then the judgment benchmark. ${runs.length ? '' : 'No evaluation has been run because no model has been trained.'}`;
      DC.hydrate(el);
    };
    this.fail = e => {
      const msg = `<div class="empty">Could not load the policy-model data: ${esc(e.message)}</div>`;
      ['#pol-curriculum', '#traj-body', '#sft-example', '#runs-list'].forEach(s => { $(s, el).innerHTML = msg; });
    };
  },
  async enter() {
    DC.markSample(false);
    try { this.render(await DC.api('/learn/policy')); } catch (e) { this.fail(e); }
  },
});
