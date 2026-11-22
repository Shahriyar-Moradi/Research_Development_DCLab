'use strict';
const desks={
  biz:{team:'Business',person:'Mina',job:'Decision owner',icon:'B',color:'#e5ebcb'},
  ds:{team:'Data Science',person:'Niloufar',job:'Data scientist',icon:'DS',color:'#d7e8ed'},
  ml:{team:'Machine Learning',person:'Kaveh',job:'ML engineer',icon:'ML',color:'#e5def0'},
  sales:{team:'Sales',person:'Raha',job:'Sales manager',icon:'S',color:'#f2e3d0'},
  mkt:{team:'Marketing',person:'Ava',job:'Campaign manager',icon:'M',color:'#efdee5'}
};
const ids=Object.keys(desks),$=id=>document.getElementById(id),fmt=n=>Number(n).toLocaleString('en-US');
const key='dclab-team-room-en-1',data=createRetentionCohort();
const title={brief:'Define this week’s decision',idle:'Waiting for the campaign brief',audit:'Check the outcome against the data',clarify:'Answer the data scientist’s question',blocked:'Waiting for a measurable outcome',spec:'Choose model inputs',evaluate:'Review model performance',verify:'Review the risk ranking',capacity:'Confirm call capacity',offer:'Confirm the offer budget',join:'Collect team responses',approve:'Review the call list',feedback:'Revise the proposal',done:'Report delivered',closed:'Case closed',unavailable:'Case on hold'};
function initial(){return {v:1,runs:[],proposals:[],requested:{},tasks:{biz:'brief',ds:'idle',ml:'idle',sales:'idle',mkt:'idle'},chat:Object.fromEntries(ids.map(i=>[i,[]])),log:[],capacity:null,budget:null,tickets:null,model:null,ranking:false,version:0,closed:false,rejected:false}}
let s=initial(),selected='biz',timer=null,cursor=0,fitted=null;
try{const saved=JSON.parse(localStorage.getItem(key));if(saved?.v===1&&saved.tasks&&saved.chat)s=saved}catch{}
function agent(id){return desks[id].team+' agent'}
function say(id,who,text,kind='agent'){s.chat[id].push({who,text,kind})}
function seed(){say('biz',desks.biz.person,'We have 120 customers coming up for renewal. Who should we call this week?','human');say('biz',agent('biz'),'I can ask Data Science to rank customers, Sales how many calls they can make, and Marketing what we can offer. Ready to start?')}
// Older demos persisted the current result but did not keep a structured revision history.
function proposalSnapshot(){
  return {version:s.version,capacity:s.capacity,budget:s.budget,calls:rows().length/2};
}
s.runs ??= s.model?[{...s.model,tickets:s.tickets}]:[];
s.proposals ??= s.version && s.capacity!==null && s.budget!==null?[proposalSnapshot()]:[];
s.requested ??= {};
if(!s.chat.biz.length)seed();
function send(from,to,text){
  s.log.push({from,to,text});say(from,agent(from),'Sent to the '+agent(to)+' — '+text,'handoff');
  say(to,agent(from),text,'handoff');
}
function save(){try{localStorage.setItem(key,JSON.stringify(s))}catch{}}
function stop(){clearInterval(timer);timer=null}
function fittedModel(){return fitted || (fitted=data.fit(s.tickets))}
function rows(){
  if(!s.ranking||s.capacity===null||s.budget===null)return [];
  const {allowed}=fittedModel(),size=Math.min(s.capacity,Math.floor(s.budget/10),Math.floor(allowed.length/2)),list=[];
  for(let i=0;i<size;i++){
    const pair=allowed.slice(i*2,i*2+2);if(pair[0].coin<.5)pair.reverse();
    pair.forEach((a,j)=>list.push({a,offer:j===0,renew:a.draw>=a.risk*(j===0?.76:1)}));
  }
  return list;
}
function gather(){
  if(!s.ranking||s.capacity===null||s.budget===null||s.closed)return;
  s.version++;s.proposals.push(proposalSnapshot());s.tasks.biz=rows().length?'approve':'feedback';
  say('biz',agent('biz'),'Proposal v'+fmt(s.version)+' is ready. Planned calls: '+fmt(rows().length/2)+'. The comparison group has the same number of customers with similar risk. '+(rows().length?'Review the list before approving it.':'Ask Sales to revise capacity or Marketing to revise the budget.'));
}
function outcome(){const list=rows(),n=list.length/2;return 'With '+fmt(n)+' customers in each group, '+fmt(list.filter(r=>r.offer&&r.renew).length)+' renewed in the offer group and '+fmt(list.filter(r=>!r.offer&&r.renew).length)+' renewed in the comparison group. These are simulated outcomes from a small sample.'}
// A person decides only at their own desk. Their agent computes or forwards a scoped response.
function choices(id){
  if(s.rejected && id==='biz')return [['reopen','Reopen with non-renewal as the target']];
  if(s.closed)return [];
  switch(s.tasks[id]){
  case 'brief': return [['start','Start the renewal campaign']];
  case 'audit':return [['confirm','Confirm non-renewal as the outcome'],['challenge','Ask about explicit cancellations']];
  case 'clarify':return [['resolve','Use non-renewal for this campaign'],['hold','Pause until we have cancellation records']];
  case 'spec':return [['full','Train with usage and support tickets'],['simple','Train with usage only']];
  case 'evaluate':return [['release','Send evaluation to Data Science'],['rerun','Change inputs and retrain']];
  case 'verify':return [['interpret','Send the findings to Business']];
  case 'capacity':return [['capacity','Send capacity to Business']];
  case 'offer':return [['offer','Confirm the offer budget']];
  case 'approve':return [['approve','Approve and reveal renewals'],['change','Ask Sales to revise capacity'],['terms','Ask Marketing to revise budget']];
  case 'feedback':return [['change','Ask Sales to revise capacity'],['terms','Ask Marketing to revise budget']];
  default:return [];
  }
}
function apply(id,action,value,note,automatic=false){
  const option=choices(id).find(c=>c[0]===action);if(!option)return;
  if((action==='capacity'||action==='offer')&&(!Number.isInteger(value)||value<0||value>(action==='capacity'?30:300)))return;
  say(id,desks[id].person+(automatic?' · simulated':''),option[1]+(value!=null?' · '+fmt(value):'')+(note?' — '+note:''),'human');
  if(action==='start'){
    s.tasks={biz:'join',ds:'audit',ml:'idle',sales:'capacity',mkt:'offer'};
    send('biz','ds','Can we estimate the risk of non-renewal seven days before a contract expires?');
    say('ds',agent('ds'),'The records tell us whether a contract renewed by its due date. Is that the outcome you want me to predict?');
    send('biz','sales','How many customers can we call this week?');say('sales',agent('sales'),'Of 120 customers, '+fmt(data.cohort.filter(a=>a.consent).length)+' have agreed to calls. How many can your team handle?');
    send('biz','mkt','How much credit can we offer customers this week?');say('mkt',agent('mkt'),'Each offer reserves €10 in renewal credit. What budget can I send to Business?');
  }else if(action==='challenge'){
    s.tasks.ds='blocked';s.tasks.biz='clarify';send('ds','biz','The table records renewals but has no cancellation events. Should we predict non-renewal, or wait for cancellation records?');
  }else if(action==='resolve'){
    s.tasks.biz='join';s.tasks.ds='audit';send('biz','ds','Mina confirmed non-renewal as the outcome. Please review the prediction task on that basis.');
  }else if(action==='hold'){
    send('biz','ds','Pause the case until we have cancellation records.');s.rejected=true;s.closed=true;ids.forEach(i=>s.tasks[i]='unavailable');
  }else if(action==='reopen'){
    s.closed=false;s.rejected=false;
    s.tasks={biz:'join',ds:'audit',ml:'idle',sales:s.capacity===null?'capacity':'done',mkt:s.budget===null?'offer':'done'};
    send('biz','ds','Business reopened the case to investigate non-renewal. Please check the revised target before training.');
  }else if(action==='confirm'){
    s.tasks.ds='join';s.tasks.ml='spec';send('ds','ml','Predict non-renewal seven days before expiry. Train on the earliest 900 contracts and validate on the next 300.');
    say('ml',agent('ml'),'Which inputs should I use? Both usage history and past ticket counts are available a week before expiry.');
  }else if(action==='full'||action==='simple'){
    s.tickets=action==='full';fitted=data.fit(s.tickets);s.model={score:fitted.score,baseline:fitted.baseScore};s.tasks.ml='evaluate';
    s.runs.push({...s.model,tickets:s.tickets});
    say('ml',agent('ml'),'Training finished. On the next 300 contracts, the model’s Brier score was '+fmt(fitted.score.toFixed(3))+' and the baseline scored '+fmt(fitted.baseScore.toFixed(3))+'. Lower is better. I’ll use the better prediction method for ranking.');
  }else if(action==='rerun'){
    s.tasks.ml='spec';s.model=null;fitted=null;say('ml',agent('ml'),'I’ve held back the report. Choose the inputs for the next run.');
  }else if(action==='release'){
    s.tasks.ml='done';s.tasks.ds='verify';send('ml','ds','Inputs used: '+fmt(s.tickets?2:1)+'. The model’s Brier score was '+fmt(s.model.score.toFixed(3))+' versus a baseline of '+fmt(s.model.baseline.toFixed(3))+' on 300 validation contracts.');
    say('ds',agent('ds'),'The ranking estimates who might miss renewal. To learn whether an offer helps, I suggest comparing customers with similar risk. Shall I send the plan?');
  }else if(action==='interpret'){
    s.ranking=true;s.tasks.ds='done';send('ds','biz','The ranking covers 120 customers. We’ll pair eligible customers by risk, then randomly assign one in each pair to receive the offer.');gather();
  }else if(action==='capacity'||action==='offer'){
    if(action==='capacity')s.capacity=value;else s.budget=value;
    delete s.requested[id];delete numericDrafts[id];
    s.tasks[id]='done';send(id,'biz',action==='capacity'?'Sales can make '+fmt(value)+' calls this week. We’ll only contact customers who consented.':'Marketing approved €'+fmt(value)+' for renewal offers. Each offer reserves €10.');gather();
  }else if(action==='change'||action==='terms'){
    const target=action==='change'?'sales':'mkt';s.tasks.biz='join';s.tasks[target]=target==='sales'?'capacity':'offer';
    if(target==='sales')s.capacity=null;else s.budget=null;
    send('biz',target,'Proposal v'+fmt(s.version)+' needs a revision. Please confirm the new limit at your desk.');
  }else if(action==='approve'){
    const count=rows().length/2;
    s.closed=true;s.tasks.biz='closed';
    send('biz','sales','Proposal v'+s.version+' approved. Release the '+count+' offer assignments in the simulation.');
    send('sales','biz','Simulated outreach completed for '+count+' customers. The '+count+' comparison customers received no offer.');
    send('biz','mkt','Record reserved credit for the approved offer group.');
    send('mkt','biz','€'+count*10+' reserved from the €'+s.budget+' budget. €'+(s.budget-count*10)+' remains available.');
    send('biz','ds','Compare simulated renewals across the two assigned groups.');
    send('ds','biz',outcome());
    say('biz',agent('biz'),'Business approved proposal v'+fmt(s.version)+'. Sales, Marketing and Data Science have returned their closing reports.');
  }
  save();if(s.closed)stop();render();
}

function el(tag,text,cls){
  const node=document.createElement(tag);
  if(text!=null)node.textContent=text;
  if(cls)node.className=cls;
  return node;
}
function button(label,fn,cls){
  const node=el('button',label,cls);
  node.type='button';
  node.onclick=fn;
  return node;
}
const drafts={},numericDrafts={};
function visit(id){
  stop();
  selected=id;
  currentView='room';
  render();
}
function stateText(id){
  if(choices(id).length)return 'Waiting for '+desks[id].person;
  if(id==='ml'&&s.tasks[id]==='idle'&&s.tasks.biz!=='brief')return 'Waiting for Data Science';
  if(s.tasks[id]==='join'&&id==='ds')return 'Waiting for the ML report';
  return title[s.tasks[id]];
}
function renderDesks(){
  $('desks').replaceChildren();
  for(const [id,d] of Object.entries(desks)){
    const b=button('',()=>visit(id),'deskbtn');
    b.dataset.team=id;
    b.setAttribute('aria-pressed',String(id===selected));
    const mark=el('span',d.icon,'symbol');
    mark.style.setProperty('--tone',d.color);
    const name=el('span');
    name.append(el('b',d.team),el('small',choices(id).length?d.person+' · decision needed':d.person+' · '+stateText(id)));
    const status=choices(id).length?'waiting':s.tasks[id]==='done'?'done':'';
    b.append(mark,name,el('i',null,'dot '+status));
    $('desks').append(b);
  }
}
function renderMessages(){
  const d=desks[selected];
  $('role').textContent=d.person+' · '+d.job;
  $('deskName').textContent=d.team+' desk';
  $('symbol').textContent=d.icon;
  $('symbol').style.setProperty('--tone',d.color);
  $('task').textContent=title[s.tasks[selected]];
  $('status').textContent=stateText(selected);
  const chat=$('chat');
  chat.replaceChildren();
  for(const m of s.chat[selected]){
    const msg=el('article',null,'message '+m.kind);
    msg.append(el('b',m.who),el('p',m.text));
    chat.append(msg);
  }
  if(!chat.childElementCount)chat.append(el('p','Incoming requests and agent replies will appear here.','empty'));
  chat.scrollTop=chat.scrollHeight;
}
function report(id){
  const sent=s.log.filter(m=>m.from===id).at(-1);
  const next=choices(id).map(c=>c[1]).join(' or ');
  return (sent?'Last handoff to '+desks[sent.to].team+' — '+sent.text:'This desk has yet to send a report.')+
    ' Current status: '+stateText(id)+'. '+(next?'Next decision: '+next+'.':'');
}
function renderActions(){
  const box=$('actions'),options=choices(selected),note=$('note');
  box.replaceChildren();
  note.value=drafts[selected]||'';
  note.hidden=!options.length;
  $('noteLabel').hidden=note.hidden;
  $('noteLabel').textContent='Reply as '+desks[selected].person;
  const numeric=options.some(([a])=>a==='capacity'||a==='offer');
  if(numeric){
    const capacity=selected==='sales',input=el('input');
    input.type='number';input.min='0';input.max=capacity?'30':'300';input.step='1';input.required=true;
    input.id='constraint';input.value=numericDrafts[selected] ?? s.requested[selected] ?? s.proposals.at(-1)?.[capacity?'capacity':'budget'] ?? (capacity?18:120);
    input.oninput=()=>numericDrafts[selected]=input.value;
    const label=el('label',capacity?'Calls available':'Budget (€)');
    label.htmlFor=input.id;
    box.append(label,input);
  }
  options.forEach(([action,label],i)=>{
    const b=button(label,()=>{
      const input=$('constraint');
      if(input&&!input.reportValidity())return;
      stop();
      const text=note.value.trim();
      delete drafts[selected];
      apply(selected,action,input?input.valueAsNumber:null,text);
    },i===0?'primary':'');
    b.dataset.action=action;
    box.append(b);
  });
  if(!options.length&&!s.closed){
    const next=ids.find(id=>choices(id).length);
    if(next)box.append(button('Go to '+desks[next].team,()=>visit(next),'primary'));
  }
}
$('agentReport').onclick=()=>{
    stop();
    say(selected,desks[selected].person,'What have you done, and what happens next?','human');
    say(selected,agent(selected),report(selected));
    save();render();
};
function fact(label,value){
  const row=el('div',null,'fact');
  row.append(el('small',label),el('strong',value));
  return row;
}
function renderCase(){
  const list=rows(),box=$('artifacts');
  $('brief').textContent=s.rejected?'The case is on hold until Business supplies cancellation records.':s.closed?outcome():'Which of the 120 customers approaching renewal should we call this week?';
  box.replaceChildren(
    fact('Prediction target',s.rejected?'Waiting for cancellation records':s.tasks.ml!=='idle'?'Non-renewal · seven days before expiry':'Awaiting agreement on the outcome'),
    fact('Model validation',s.model?'Brier score '+fmt(s.model.score.toFixed(3))+' · baseline '+fmt(s.model.baseline.toFixed(3)):'Awaiting training'),
    fact('Sales capacity',s.capacity==null?'Awaiting Sales':fmt(s.capacity)+' calls'),
    fact('Marketing budget',s.budget==null?'Awaiting Marketing':fmt(s.budget)+' euros')
  );
  if(s.version)box.append(fact('Proposal v'+fmt(s.version),s.tasks.biz==='join'?'Returned for revision':fmt(list.length/2)+' calls · '+(s.closed?'approved':'awaiting approval')));
  $('version').textContent=s.version?'v'+s.version:'';
  const events=s.log.filter(m=>!$('filter').value || [m.from,m.to].includes($('filter').value));
  $('history').replaceChildren();
  $('count').textContent=events.length+' of '+s.log.length+' handoffs';
  for(const m of events.slice().reverse()){
    const row=el('li');
    row.append(el('b',desks[m.from].team+' → '+desks[m.to].team),el('p',m.text));
    $('history').append(row);
  }
  if(!events.length)$('history').append(el('li',s.log.length?'No handoffs for this team yet.':'Start the campaign to dispatch the first three requests.','empty'));
}
function render(){
  $('play').textContent=timer?'Pause and take over':s.tasks.biz==='approve'?'Awaiting approval':s.closed?'Watch complete':'Watch the teams work';
  $('play').disabled=s.closed||s.tasks.biz==='approve';
  $('step').disabled=$('play').disabled;
  renderDesks();renderMessages();renderActions();renderCase();renderViews();
}
function step(){
  if(s.closed || s.tasks.biz==='approve'){stop();render();return}
  for(let i=0;i<ids.length;i++){
    const id=ids[cursor++%ids.length],options=choices(id);
    if(!options.length)continue;
    selected=id;
    let action=options[0][0];
    if(s.tasks[id]==='feedback')action=s.capacity===0?'change':'terms';
    const value=action==='capacity'?(s.requested.sales??18):action==='offer'?(s.requested.mkt??120):null;
    apply(id,action,value,null,true);
    if(s.tasks.biz==='approve'){stop();selected='biz';render()}
    return;
  }
  stop();render();
}
$('play').onclick=()=>{
  if(timer){stop();render();return}
  timer=setInterval(step,2700);
  step();
};
$('response').onsubmit=e=>e.preventDefault();
$('note').oninput=e=>drafts[selected]=e.target.value;
$('response').addEventListener('focusin',()=>{
  if(timer){stop();$('play').textContent='Watch the teams work'}
});
$('restart').onclick=()=>{stop();render();$('confirm').showModal()};
$('keep').onclick=()=>$('confirm').close();
$('clear').onclick=()=>{
  $('confirm').close();
  s=initial();selected='biz';cursor=0;fitted=null;currentView='room';
  $('filter').value='';$('group').value='all';resetScenario();
  for(const id of ids){delete drafts[id];delete numericDrafts[id]}
  seed();save();render();
};
