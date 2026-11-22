'use strict';
const {train,validation,cohort,fit}=createRetentionCohort();
const {score,baseScore,model,allowed}=fit();
const roles=['Business','Data Science','Machine Learning','Sales','Marketing'];
const $=id=>document.getElementById(id),fmt=n=>Number(n).toLocaleString('en-US');
const pct=n=>fmt(Math.round(n*100))+'%';
const storage='dclab-retention-sim-en-v1';
const fresh=()=>({stage:0,capacity:18,budget:120,version:1,closed:false,trail:[]});
let s=fresh(),focus=0,timer=null;
try{const value=JSON.parse(localStorage.getItem(storage));if(value && Number.isInteger(value.stage)&&value.stage>=0&&value.stage<=7&&Number.isInteger(value.capacity)&&value.capacity>=0&&value.capacity<=30&&Number.isInteger(value.budget)&&value.budget>=0&&value.budget<=300&&Array.isArray(value.trail))s=value}catch{}
const teams=[
  {from:0,to:1,title:'Business defined the campaign',text:()=> 'Give Sales a call list for this week. Predict which customers will miss renewal by the due date using only information available seven days beforehand.'},
  {from:1,to:2,title:'Data Science handed off the prediction task',text:()=> 'Use the earliest 900 completed contracts for training and the next 300 for validation. Renewal status supplies the outcome label.'},
  {from:2,to:1,title:'ML compared the model with a baseline',text:()=> 'On the next 300 contracts, the model’s Brier score was '+fmt(score.toFixed(3))+' and the constant baseline scored '+fmt(baseScore.toFixed(3))+'. Lower is better. '+(score<=baseScore?'Logistic regression':'The constant baseline')+' was selected for ranking.'},
  {from:1,to:0,title:'Business received the ranking',text:()=> 'All 120 customers due this week have a risk score. To plan outreach, Business still needs call capacity from Sales and offer terms from Marketing.'},
  {from:3,to:0,title:'Sales checked capacity and consent',text:()=> 'Sales can make '+fmt(s.capacity)+' calls this week. Of 120 customers, '+fmt(cohort.filter(a=>a.consent).length)+' have agreed to calls.'},
  {from:4,to:0,title:'Marketing confirmed the renewal offer',text:()=> 'Each offer reserves €10 from a budget of €'+fmt(s.budget)+'. '+fmt(allowed.length)+' customers qualify for the offer and have agreed to calls.'},
  {from:0,to:0,title:'The joint proposal is ready for approval',text:()=> 'Eligible customers were paired by risk. A reproducible random draw assigned one customer in each pair to the offer group and the other to the comparison group.'}
];
const duties=['Define churn, assemble the proposal, and record the outcome.','Set the prediction date, review evidence, and deliver the ranking.','Train a model and validate it on later contracts.','Check consent and confirm call capacity.','Set offer terms and approve renewal credit.'];
function plan(){
  const max=Math.min(s.capacity,Math.floor(s.budget/10),Math.floor(allowed.length/2));
  const result=[];
  for(let i=0;i<max;i++){
    const pair=allowed.slice(2*i,2*i+2);
    if(pair[0].coin<.5)pair.reverse();
    pair.forEach((a,j)=>{const offer=j===0;
      // The assumed offer effect is part of the simulator, never fitted on the evaluation labels.
      result.push({a,offer,renew:a.draw>=a.risk*(offer?.76:1)})});
  }
  return result;
}
function save(){try{localStorage.setItem(storage,JSON.stringify(s))}catch{}}
function halt(){clearInterval(timer);timer=null}
function advance(){if(s.stage===teams.length)return;const e=teams[s.stage++];focus=e.to;s.trail.push({actor:e.from,line:e.title,body:e.text()});if(s.stage===teams.length)halt();save();render()}
function revise(input){
  const property=input.id==='capacity'?'capacity':'budget';
  if(s.closed || s[property]===Number(input.value))return;
  const old=s[property];s[property]=Number(input.value);
  if(s.stage>= (property==='capacity'?5:6)){
    s.version++;const team=property==='capacity'?3:4;
    const line=roles[team]+' '+(team===3?'changed call capacity':'changed the budget')+' from '+fmt(old)+' to '+fmt(s[property])+'.';
    s.trail.push({actor:team,line,body:'Proposal v'+fmt(s.version)+' uses the revised limit and the existing risk ranking.'});focus=team;
  }
  save();render();
}
function addRow(parent,tag,text){const el=document.createElement(tag);el.textContent=text;parent.append(el)}
function outcome(list){
  const hit=list.filter(r=>r.offer&&r.renew).length,control=list.filter(r=>!r.offer&&r.renew).length;
  return 'Each group had '+fmt(list.length/2)+' customers. Renewals in the offer group: '+fmt(hit)+'. Renewals in the comparison group: '+fmt(control)+'. The small simulated sample gives limited evidence about the effect of the offer.';
}
function render(){
  const ready=s.stage===teams.length,list=ready?plan():[],count=list.length/2,latest=teams[s.stage-1];
  $('play').textContent=timer?'Pause':s.stage?'Resume':'Run simulation';$('play').disabled=$('step').disabled=ready;
  $('counter').textContent='Step '+fmt(s.stage)+' of '+fmt(teams.length);
  $('capacity').value=s.capacity;$('budget').value=s.budget;
  $('capacity').disabled=$('budget').disabled=s.closed;
  $('capLabel').textContent=fmt(s.capacity)+' calls';$('budgetLabel').textContent=fmt(s.budget)+' euros';
  $('nodes').replaceChildren();roles.forEach((r,i)=>{
    const b=document.createElement('button');b.className='node'+(focus===i?' selected':'')+(latest&&(latest.to===i||latest.from===i)?' active':'');
    b.setAttribute('aria-pressed',String(i===focus));addRow(b,'b',r);addRow(b,'span',s.trail.some(e=>e.actor===i)?'Report available':'Waiting');
    b.onclick=()=>{halt();focus=i;$('agent').open=true;render()};$('nodes').append(b);
  });
  $('agentName').textContent=roles[focus];const report=$('agentReport');report.replaceChildren();addRow(report,'p','Responsibility: '+duties[focus]);
  const own=s.trail.filter(e=>e.actor===focus);
  if(own.length)own.slice(-3).forEach(e=>addRow(report,'p',e.line+' — '+e.body));else addRow(report,'p','This team has yet to record an action.');
  $('route').textContent=s.closed?'Business · recorded outcome':latest?(latest.from===latest.to?'Business · shared case':roles[latest.from]+' → '+roles[latest.to]):'Business · opening question';
  $('subject').textContent=s.closed?'Renewal outcomes recorded':latest?.title||'Which customers might miss renewal?';
  $('detail').textContent=s.closed?outcome(list):latest?latest.text():'This week, 120 customers are due to renew. Run the simulation to see how the teams choose whom to call.';
  $('revision').textContent=s.version>1?'Proposal v'+fmt(s.version)+' · revised limits':'';
  $('state').textContent=s.closed?'Outcome recorded':ready?(count?'Awaiting approval':'Revise the campaign limits'):'Preparing';
  $('num').textContent=$('compare').textContent=ready?fmt(count):'—';$('spend').textContent=ready?fmt(count*10)+' €':'—';
  $('reason').textContent=ready?'Eligible customers: '+fmt(allowed.length)+'. Sales capacity: '+fmt(s.capacity)+' calls. The budget covers '+fmt(Math.floor(s.budget/10))+' offers. '+(count?'The tightest constraint determines the list size.':'At least one call slot and €10 are required.'):'The call list appears after the model, capacity, and budget are ready.';
  $('approve').hidden=!ready||s.closed||!count;$('observed').textContent=s.closed?outcome(list):'';
  $('customers').hidden=!ready||!count;$('rows').replaceChildren();
  for(const r of list){const row=document.createElement('tr');[fmt(r.a.id),pct(model(r.a)),fmt(r.a.days),r.offer?'Call and offer':'Comparison',s.closed?(r.renew?'Yes':'No'):'Waiting'].forEach(x=>addRow(row,'td',x));$('rows').append(row)}
  $('log').replaceChildren();[...s.trail].reverse().forEach(e=>addRow($('log'),'li',e.line));if(!s.trail.length)addRow($('log'),'li','Business is ready to send the campaign brief.');
}
$('play').onclick=()=>{if(timer){halt();render()}else{timer=setInterval(advance,2900);advance()}};
$('step').onclick=()=>{halt();advance()};
$('approve').onclick=()=>{if(s.closed||s.stage!==7||!plan().length)return;s.closed=true;focus=0;s.trail.push({actor:0,line:'Business approved proposal v'+fmt(s.version)+'.',body:outcome(plan())});save();render()};
for(const id of ['capacity','budget']){
  $(id).oninput=e=>{$(id==='capacity'?'capLabel':'budgetLabel').textContent=fmt(e.target.value)+(id==='capacity'?' calls':' euros')};
  $(id).onchange=e=>revise(e.target);
}
$('reset').onclick=()=>{halt();s=fresh();focus=0;save();$('agent').open=false;$('customers').open=false;render()};
render();
