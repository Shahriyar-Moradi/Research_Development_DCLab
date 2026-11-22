'use strict';
let currentView='room', scenarioEdited=false;
const eligibleCount=data.cohort.filter(a=>a.eligible && a.consent).length;

function switchView(view){
  stop();
  currentView=view;
  render();
}
function resetScenario(){
  scenarioEdited=false;
  $('cap').value=s.capacity??18;
  $('money').value=s.budget??120;
}
function renderViews(){
  for(const view of ['room','evidence','decision'])$(view+'View').hidden=currentView!==view;
  for(const btn of $('views').querySelectorAll('button')){
    if(btn.dataset.view===currentView)btn.setAttribute('aria-current','page');
    else btn.removeAttribute('aria-current');
  }
  const stage=s.closed&&!s.rejected?3:s.version?2:s.tasks.biz==='brief'||s.tasks.ml==='idle'||s.rejected?0:1;
  $('phases').replaceChildren(...['Define the target','Validate the model','Review outreach','Record renewals'].map((name,i)=>{
    const item=el('li',name,i<stage?'past':'');
    if(i===stage)item.setAttribute('aria-current','step');
    return item;
  }));
  if(currentView==='evidence'){renderEvidence();renderRiskBands()}
  if(currentView==='decision')renderDecision();
}
function table(headings,records){
  const node=el('table'), head=node.createTHead().insertRow(), body=node.createTBody();
  for(const heading of headings){
    const th=el('th',heading);
    th.scope='col';
    head.append(th);
  }
  for(const record of records){
    const tr=body.insertRow();
    for(const value of record)tr.insertCell().textContent=value;
  }
  const wrap=el('div',null,'tablewrap');
  wrap.append(node);
  return wrap;
}
function renderEvidence(){
  $('dataMetrics').replaceChildren(fact('Training contracts',data.train.length),fact('Validation contracts',data.validation.length),fact('Due this week',data.cohort.length));
  $('contract').textContent=s.rejected?'Business requested explicit cancellation. This dataset only records renewal status, so training is on hold.':
    s.tasks.ml==='idle'?'Proposed target: non-renewal. Data Science must verify the label before ML can train.':'Confirmed target: non-renewal by the contract due date.';
  $('runs').replaceChildren(s.runs.length?table(['Run','Inputs','Model Brier','Baseline Brier','Ranking method'],s.runs.map((r,i)=>[
    i+1,r.tickets?'Usage + tickets':'Usage only',r.score.toFixed(3),r.baseline.toFixed(3),r.score<=r.baseline?'Logistic regression':'Constant rate'
  ])):el('p','Validation results appear after the ML engineer trains a model.','empty'));
}
function renderRiskBands(){
  const area=$('riskBands');
  if(!s.model){
    area.replaceChildren(el('p','Train a model to inspect how it scores customers who can receive an offer.','empty'));
    return;
  }
  const {allowed,model}=fittedModel(),counts=[0,0,0,0],bands=['Below 25%','25% to <50%','50% to <75%','75% or higher'];
  for(const account of allowed)counts[Math.min(3,Math.floor(model(account)*4))]++;
  area.replaceChildren();
  counts.forEach((count,i)=>{
    const row=el('div',null,'barrow'),track=el('span',null,'track'),fill=el('span');
    fill.style.width=(100*count/allowed.length)+'%';track.append(fill);
    row.append(el('span',bands[i]),track,el('b',count+' accounts'));area.append(row);
  });
}
function renderScenario(){
  const cap=$('cap').valueAsNumber, budget=$('money').valueAsNumber;
  const count=Math.min(cap,Math.floor(budget/10),Math.floor(eligibleCount/2));
  $('capOut').value=cap+' calls';
  $('moneyOut').value='€'+budget;
  $('scenario').textContent='Preview: '+count+' calls, '+count+' comparison customers and €'+count*10+' reserved. '+eligibleCount+' customers qualify for pairing.';
  $('requestLimits').disabled=!['approve','feedback'].includes(s.tasks.biz)||(cap===s.capacity && budget===s.budget);
}
function requestLimits(){
  if($('requestLimits').disabled)return;
  const limits={sales:$('cap').valueAsNumber,mkt:$('money').valueAsNumber};
  for(const [id,value] of Object.entries(limits)){
    const sales=id==='sales',field=sales?'capacity':'budget';
    if(s[field]===value)continue;
    s.requested[id]=value;s[field]=null;s.tasks[id]=sales?'capacity':'offer';
    send('biz',id,'Please review the proposed '+(sales?'capacity of '+value+' calls':'budget of €'+value)+' for the revision to v'+s.version+'.');
  }
  s.tasks.biz='join';
  say('biz',desks.biz.person,'Request revised team limits for proposal v'+s.version+'.','human');
  scenarioEdited=false;
  save();visit('biz');
}
function renderList(){
  let list=rows();
  if($('group').value!=='all')list=list.filter(r=>r.offer===($('group').value==='offer'));
  $('list').replaceChildren(list.length?table(['Account','Group','Risk','Days since use','Tickets','Renewal'],list.map(r=>[
    String(r.a.id),r.offer?'Offer':'Comparison',Math.round(fittedModel().model(r.a)*100)+'%',r.a.days,r.a.tickets,
    s.closed?(r.renew?'Renewed':'Lapsed'):'Pending approval'
  ])):el('p','No current customer list. Review the target, model and campaign limits in the team room.','empty'));
}
function decisionStatus(){
  if(s.rejected)return 'On hold. Business can reopen the case for non-renewal at its desk.';
  if(s.closed)return 'Approved by Mina, the Business decision owner.';
  if(s.tasks.biz==='feedback')return 'Blocked. The current limits cannot fund one call and a matched comparison customer.';
  if(s.tasks.biz==='approve')return 'Ready for Business approval. Review the groups and limits before revealing simulated renewals.';
  if(s.version)return 'Under revision. The previous proposal is in the history while the teams confirm new limits.';
  return 'Collecting evidence. The call list needs a risk ranking, a sales capacity and an offer budget.';
}
function renderDecision(){
  $('decisionState').textContent=decisionStatus();
  const count=rows().length/2,ready=['feedback','approve','closed'].includes(s.tasks.biz);
  $('metrics').replaceChildren(fact('Planned calls',ready?count:'—'),fact('Comparison group',ready?count:'—'),fact('Reserved credit',ready?'€'+count*10:'—'));
  if(!scenarioEdited){
    $('cap').value=s.capacity??s.requested.sales??s.proposals.at(-1)?.capacity??18;
    $('money').value=s.budget??s.requested.mkt??s.proposals.at(-1)?.budget??120;
  }
  renderScenario();
  $('versions').replaceChildren(s.proposals.length?table(['Version','Sales capacity','Offer budget','Planned calls'],
    s.proposals.slice().reverse().map(p=>['v'+p.version,p.capacity,'€'+p.budget,p.calls])):el('p','The first version appears when all three teams return their inputs.','empty'));
  renderList();
  $('result').replaceChildren();
  if(s.closed&&!s.rejected){
    const list=rows();
    $('result').append(el('h3','Simulated renewals'),el('p',outcome()));
    for(const offer of [true,false]){
      const group=list.filter(r=>r.offer===offer);
      const renewals=group.filter(r=>r.renew).length;
      const row=el('div',null,'barrow'),track=el('span',null,'track'),fill=el('span');
      fill.style.width=(100*renewals/group.length)+'%';track.append(fill);
      row.append(el('span',offer?'Offer group':'Comparison'),track,el('b',renewals+' / '+group.length));$('result').append(row);
    }
  }
}
function downloadBrief(){
  const lines=['# Customer renewal campaign','','Local DCLab simulation using synthetic customer records.','',decisionStatus(),'',
    '## Evidence','', 'Historical contracts: 900 training and 300 validation. Current cohort: 120. Eligible for pairing: '+eligibleCount+'.'];
  s.runs.forEach((r,i)=>lines.push('- Run '+(i+1)+' ('+(r.tickets?'usage and tickets':'usage only')+'). Model Brier '+r.score.toFixed(3)+', baseline '+r.baseline.toFixed(3)+'.'));
  lines.push('','## Proposals','');
  s.proposals.forEach(p=>lines.push('- v'+p.version+' — '+p.capacity+' call slots, €'+p.budget+' budget, '+p.calls+' planned calls, €'+p.calls*10+' reserved.'));
  if(!s.proposals.length)lines.push('No proposal recorded yet.');
  if(rows().length){
    lines.push('','## Assigned customers','','| Account | Group | Estimated risk | Renewal |','| --- | --- | --- | --- |');
    for(const {a,offer,renew} of rows()){
      const probability=(100*fittedModel().model(a)).toFixed(1)+'%';
      const observed=s.closed?(renew?'Renewed':'Lapsed'):'Pending approval';
      lines.push('| '+[a.id,offer?'Offer':'Comparison',probability,observed].join(' | ')+' |');
    }
  }
  if(s.closed&&!s.rejected)lines.push('','## Simulated renewals','',outcome(),'The simulator assumes the offer reduces underlying non-renewal risk by 24%.');
  lines.push('','## Handoffs','');
  s.log.forEach(m=>lines.push('- '+desks[m.from].team+' → '+desks[m.to].team+' — '+m.text));
  const url=URL.createObjectURL(new Blob([lines.join('\n')+'\n'],{type:'text/markdown;charset=utf-8'}));
  const link=el('a');link.href=url;link.download='renewal-case-report.md';link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
for(const btn of document.querySelectorAll('[data-view]'))btn.onclick=()=>switchView(btn.dataset.view);
for(const [id,d] of Object.entries(desks))$('filter').add(new Option(d.team,id));
$('filter').onchange=()=>{stop();render()};
$('group').onchange=renderList;
for(const id of ['cap','money'])$(id).oninput=()=>{scenarioEdited=true;renderScenario()};
$('requestLimits').onclick=requestLimits;
$('goBiz').onclick=()=>visit('biz');
$('goMl').onclick=()=>visit('ml');
$('export').onclick=downloadBrief;
$('step').onclick=()=>{stop();currentView='room';step()};
$('play').addEventListener('click',()=>{if(timer){currentView='room';render()}});
render();
