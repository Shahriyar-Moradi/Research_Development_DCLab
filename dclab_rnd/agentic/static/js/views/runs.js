/* Agent research: project picker, composer, run list and live run detail. */
import {$, esc, number, state, labels, api, notice} from '../core.js';

export function updateSettings() {
  const count = Number($('experiment-limit').value);
  $('settings-summary').textContent = `${count} experiments · ${$('repeats').value} × 3-fold CV`;
  $('budget-copy').textContent = `Up to ${5+2*count} API calls · 6,000 output tokens per call. Input tokens also incur API charges. No automatic purchase or unlimited loop.`;
}
function inferProject(run) { return run.config.project || state.config?.datasets.find(d=>run.config.datasets.includes(d.key))?.project || 'general'; }
function chooseProject(key) {
  const config = state.config;
  state.campaignProject=key;
  document.querySelectorAll('[data-project]').forEach(b=>b.classList.toggle('active',b.dataset.project===key));
  const project=config.projects.find(p=>p.key===key), allowed=new Set(project.datasets);
  $('datasets').innerHTML=config.datasets.filter(d=>allowed.has(d.key)).map(d=>`<label class="dataset-option" title="${esc(d.decision)}"><input type="checkbox" value="${esc(d.key)}" ${d.key===project.default_dataset?'checked':''}>${esc(labels[d.key]||d.name)}</label>`).join('');
  const campaign=project.campaign||{};
  const evidence=campaign.completed!==undefined?`<strong>${esc(campaign.completed)} / ${esc(campaign.planned)} experiments</strong>`:'';
  const scores=key==='hyperack'?`<span>Safe champion AUC ${number(campaign.safe_champion_auc)} · unsafe ceiling ${number(campaign.unsafe_ceiling_auc)} (post-outcome leakage)</span>`:key==='telco_churn'&&campaign.best?`<span>Current development leader: ${esc(campaign.best.title)} · AUC ${number(campaign.best.metrics?.roc_auc)}</span>`:'';
  $('project-summary').innerHTML=`<div><b>${esc(project.status)}</b><p>${esc(project.summary)}</p></div><div>${evidence}${scores}</div>`;
}
export function renderProjects(){
  $('projects').innerHTML=state.config.projects.map(p=>`<button type="button" class="project-option ${p.key===state.campaignProject?'active':''}" data-project="${esc(p.key)}"><span>${esc(p.name)}</span><small>${esc(p.status)}</small></button>`).join('');
  document.querySelectorAll('[data-project]').forEach(b=>b.addEventListener('click',()=>chooseProject(b.dataset.project)));
  chooseProject(state.campaignProject);
}
export function renderRuns() {
  const runs = state.runs;
  $('run-list').innerHTML = runs.length ? runs.slice(0,10).map(r=>`<button class="run-nav ${state.selected===r.id?'active':''}" data-run="${esc(r.id)}"><strong>${esc(r.config.goal)}</strong><small>${esc(r.config.datasets.map(d=>labels[d]||d).join(' · '))} · ${esc(r.status)}</small></button>`).join('') : '<p class="muted small">Your research runs will appear here.</p>';
  document.querySelectorAll('[data-run]').forEach(b=>b.addEventListener('click',()=>{location.hash='#run/'+b.dataset.run;}));
}
export async function refreshRuns() {
  try {
    state.runs = await api('/runs'); renderRuns();
    if (state.selected && state.view === 'research') renderDetail(await api('/runs/'+state.selected));
  } catch(error) { notice(error.message); }
}
export function showComposer() {
  state.selected=null;
  $('composer').classList.remove('hidden');$('run-detail').classList.add('hidden');
  document.querySelector('#research-view .page-heading').classList.remove('hidden');renderRuns();
}
export async function showRun(id) {
  state.selected=id;
  $('composer').classList.add('hidden'); $('run-detail').classList.remove('hidden');
  document.querySelector('#research-view .page-heading').classList.add('hidden');
  await refreshRuns();
}
function list(items) { return `<ul>${(items||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`; }
function artifactLink(run,id,file,label) { return `<a href="/api/runs/${encodeURIComponent(run)}/trials/${encodeURIComponent(id.split(':').pop())}/${file}" download>${label} ↗</a>`; }
function renderDetail(run) {
  const config = state.config;
  const events=run.events, corrections=new Map(events.filter(e=>e.kind==='comparison_correction').map(e=>[e.payload.trial_id,e.payload]));
  const trials=[...new Map(events.filter(e=>e.kind==='trial').map(e=>[e.payload.id,e.payload])).values()];
  trials.forEach(t=>{if(corrections.has(t.id))t.paired_comparison=corrections.get(t.id).paired_comparison;});
  const completed=trials.filter(t=>t.status==='completed'), latest=completed.at(-1), synthesis=events.filter(e=>e.kind==='synthesis').at(-1)?.payload;
  const dataReview=events.find(e=>e.kind==='data_review')?.payload, critiques=events.filter(e=>e.kind==='critique');
  const phases=events.filter(e=>e.kind==='phase').slice(-7), running=['running','queued','pausing'].includes(run.status);
  const preserved = new Set([...$('run-detail').querySelectorAll('details[open]')].map(d=>d.dataset.key));
  $('run-detail').innerHTML=`
    <div class="run-heading"><div><div class="eyebrow">${esc((config.projects.find(p=>p.key===inferProject(run))||{}).name||'RESEARCH')} · RESEARCH IN MOTION</div><h2>${esc(run.config.datasets.map(d=>labels[d]||d).join(' + '))}</h2><p>${esc(run.config.model)} via NOOA + OpenAI Responses · ${esc(run.config.repeats)} × 3-fold group CV · ${run.config.max_rows.toLocaleString()} row cap</p></div><div class="actions"><span class="status ${esc(run.status)}">${esc(run.status)}</span>${running?'<button class="secondary" id="pause">Pause</button>':['paused','interrupted','failed'].includes(run.status)?'<button class="secondary" id="resume">Resume</button>':''}<a class="secondary" href="/api/runs/${esc(run.id)}/export">Export trace ↗</a></div></div>
    <div class="card panel"><p class="small">${esc(run.config.goal)}</p><div class="small muted">${running?'<span class="running-dot"></span>':''}${esc(run.error || run.phase || 'Queued')}</div><div class="progress"><div></div></div></div>
    <div class="stats"><div class="card stat"><small>EXPERIMENTS</small><strong>${completed.length}<span class="inline">of ${run.config.max_experiments} limit</span></strong><span>${trials.length-completed.length} rejected or failed</span></div><div class="card stat"><small>LATEST CV AUC</small><strong>${number(latest?.result.metrics.roc_auc)}</strong><span>${latest?esc(latest.result.dataset):'Waiting for measured evidence'}</span></div><div class="card stat"><small>LEARNED LESSONS</small><strong>${synthesis?.lessons.length || 0}</strong><span>Provisional, evidence-linked</span></div><div class="card stat"><small>LLM CALLS</small><strong>${run.llm_calls}</strong><span>${esc(run.usage.total_tokens?.toLocaleString() || '—')} total tokens</span></div></div>
    <div class="run-grid"><div>
      <div class="card panel evidence"><h2>Experiment evidence</h2><p class="small muted">Adaptive development scores. No untouched test set or production approval.</p>
      ${trials.length?`<div class="table-wrap"><table><thead><tr><th>EXPERIMENT</th><th>CV AUC</th><th>LOG LOSS</th><th>PAIRED Δ AUC</th></tr></thead><tbody>${trials.map((t,i)=>`<tr><td>${i+1}. ${esc(t.plan.title)}<small>${esc(t.plan.model)} · ${esc(t.plan.dataset)}</small></td><td>${number(t.result?.metrics.roc_auc)}</td><td>${number(t.result?.metrics.log_loss)}</td><td>${number(t.paired_comparison?.mean_deltas?.roc_auc)}</td></tr>`).join('')}</tbody></table></div>`:'<p class="muted small">The team is profiling data and designing its first experiment.</p>'}
      ${trials.map((t,i)=>`<details data-key="trial-${i}"><summary>${i+1}. Inspect inputs, outputs & recipe</summary><p><b>Hypothesis:</b> ${esc(t.plan.hypothesis)}</p>${t.error?`<p>${esc(t.error)}</p>`:`<p><b>Engineered features:</b> ${esc(t.plan.features.map(f=>f.name+' = '+f.operation+'('+f.inputs.join(', ')+')').join('; ') || 'None — baseline inputs')}</p><p><b>Policy exclusions:</b> ${esc(t.result.excluded_by_policy.join(', ') || 'None declared; availability remains an assumption')}</p><p><b>Most sensitive inputs:</b> ${esc(t.result.input_sensitivity.slice(0,5).map(f=>f.column+' ('+number(f.auc_drop)+' AUC drop)').join(', '))}</p><p><b>Missing-input stress:</b> ${esc(t.result.stress_tests.slice(0,3).map(s=>s.column+': '+number(s.metric_deltas_clean_minus_stressed?.roc_auc)+' AUC drop').join('; ') || 'Not requested')}</p><p><b>Output reliability:</b> Brier ${number(t.result.metrics.brier)} · Average precision ${number(t.result.metrics.average_precision)}. Repeat-specific calibration bins and OOF predictions in evidence.</p><p><b>Uncertainty:</b> AUC fold SD ${number(t.result.fold_standard_deviation.roc_auc)}; correlated folds, not a confidence interval.</p><div class="links">${artifactLink(run.id,t.id,'result.json','Evidence')}${artifactLink(run.id,t.id,'recipe.json','Recipe')}${artifactLink(run.id,t.id,'oof_predictions.jsonl','OOF predictions')}</div>`}</details>`).join('')}</div>
      ${critiques.length?`<div class="card panel evidence"><h2>Latest scientific critique</h2><p>${esc(critiques.at(-1).payload.observation)}</p><p>${esc(critiques.at(-1).payload.interpretation)}</p><p><b>Next question:</b> ${esc(critiques.at(-1).payload.next_question)}</p><details data-key="critique"><summary>Limitations & counterevidence</summary>${list(critiques.at(-1).payload.limitations)}</details></div>`:''}
      ${synthesis?`<div class="card panel evidence"><h2>Research synthesis</h2><p>${esc(synthesis.summary)}</p>${synthesis.lessons.map(l=>`<h3>${esc(l.claim)}</h3><p>${esc(l.scope)}</p><p><b>Counterevidence:</b> ${esc(l.counterevidence)}</p>`).join('')}<details data-key="theory"><summary>Theoretical principles</summary>${list(synthesis.theoretical_principles)}</details><details data-key="workflow"><summary>Reusable workflow blocks</summary>${list(synthesis.workflow_blocks)}</details><details data-key="unanswered"><summary>Still unanswered</summary>${list(synthesis.unanswered_questions)}</details></div>`:''}
    </div><div><div class="card panel"><h2>Agent activity</h2><ol class="timeline">${phases.map(e=>`<li>${esc(e.payload.name)}<small>${new Date(e.created).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</small></li>`).join('') || '<li>Waiting to start</li>'}</ol></div>${dataReview?`<div class="card panel evidence"><h2>Data & leakage review</h2>${list(dataReview.leakage_risks)}<details data-key="availability"><summary>Availability assumptions</summary>${list(dataReview.availability_assumptions)}</details><details data-key="external"><summary>External-data hypotheses</summary>${list(dataReview.external_data_hypotheses)}</details></div>`:''}</div></div>`;
  // Width is set as a DOM property because the strict CSP forbids inline styles.
  const progress=$('run-detail').querySelector('.progress div');progress.style.width=Math.round(trials.length/run.config.max_experiments*100)+'%';
  $('run-detail').querySelectorAll('details').forEach(d=>d.open=preserved.has(d.dataset.key));
  $('pause')?.addEventListener('click',()=>control('pause'));
  $('resume')?.addEventListener('click',()=>control('resume'));
}
async function control(action) { try { await api(`/runs/${state.selected}/${action}`,{method:'POST'});await refreshRuns(); }catch(e){notice(e.message);} }

export function initComposer() {
  $('goal').value=state.config.default_goal;$('model').value=state.config.default_model;state.campaignProject=state.config.default_project;renderProjects();
  $('runtime-copy').innerHTML=`<p><b>Active LLM:</b> ${esc(state.config.default_model)} · ${esc(state.config.frameworks.join(' · '))}</p>${Object.entries(state.config.commands).map(([name,command])=>`<div><span>${esc(name.replaceAll('_',' '))}</span><code>${esc(command)}</code></div>`).join('')}`;
  if(!state.config.api_key_configured){$('start').disabled=true;$('key-notice').textContent='Set OPENAI_API_KEY in the server environment or .env, then restart. Never paste your key into the goal.';$('key-notice').classList.remove('hidden');}
  ['experiment-limit','repeats'].forEach(id=>$(id).addEventListener('input',updateSettings));
  $('new-run').addEventListener('click',()=>{location.hash='#research';showComposer();});
  $('start').addEventListener('click',async()=>{
    $('start').disabled=true;notice('');
    try {
      const datasets=[...document.querySelectorAll('#datasets input:checked')].map(i=>i.value);
      if(!datasets.length)throw new Error('Choose at least one dataset.');
      const result=await api('/runs',{method:'POST',body:JSON.stringify({project:state.campaignProject,goal:$('goal').value,datasets,model:$('model').value,max_experiments:Number($('experiment-limit').value),max_rows:Number($('row-limit').value),repeats:Number($('repeats').value),max_minutes:Number($('minutes').value)})});
      location.hash='#run/'+result.id;
    } catch(error){notice(error.message);} finally{$('start').disabled=!state.config.api_key_configured;}
  });
}
