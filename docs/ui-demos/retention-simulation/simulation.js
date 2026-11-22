'use strict';
// Synthetic population and fixed random draws make scenario comparisons reproducible.
function generator(seed) {
  return () => { seed=(Math.imul(seed,1664525)+1013904223)>>>0; return seed/4294967296; };
}
const random=generator(1042);
const sigmoid=x=>1/(1+Math.exp(-x));
function account(id) {
  const days=Math.floor(random()*51),tickets=Math.floor(random()*7);
  const risk=sigmoid(-3.4+.069*days+.29*tickets);
  return {id,days,tickets,risk,lapsed:Number(random()<risk),eligible:random()>.18,
    consent:random()>.12,coin:random(),draw:random()};
}
const history=Array.from({length:1200},(_,i)=>account(i+1));
const train=history.slice(0,900),validation=history.slice(900);
const cohort=Array.from({length:120},(_,i)=>account(8001+i));
const features=a=>[1,a.days/50,a.tickets/6];
const weights=[0,0,0];
function prediction(a){return sigmoid(features(a).reduce((sum,x,i)=>sum+x*weights[i],0))}
for(let epoch=0;epoch<200;epoch++){
  const gradient=[0,0,0];
  for(const a of train){const error=prediction(a)-a.lapsed;features(a).forEach((x,i)=>gradient[i]+=error*x)}
  weights.forEach((w,i)=>weights[i]-=.8*gradient[i]/train.length);
}
const baseline=train.reduce((sum,a)=>sum+a.lapsed,0)/train.length;
const brier=fn=>validation.reduce((s,a)=>s+(fn(a)-a.lapsed)**2,0)/validation.length;
const score=brier(prediction),baseScore=brier(()=>baseline);
const model=score<=baseScore?prediction:()=>baseline;
const allowed=cohort.filter(a=>a.consent && a.eligible).sort((a,b)=>model(b)-model(a)||a.id-b.id);
const roles=['بیزینس','دیتا ساینس','یادگیری ماشین','فروش','مارکتینگ'];
const $=id=>document.getElementById(id),fa=n=>Number(n).toLocaleString('fa-IR');
const pct=n=>fa(Math.round(n*100))+'٪';
const storage='dclab-retention-sim-v1';
const fresh=()=>({stage:0,capacity:18,budget:120,version:1,closed:false,trail:[]});
let s=fresh(),focus=0,timer=null;
try{const value=JSON.parse(localStorage.getItem(storage));if(value && Number.isInteger(value.stage)&&value.stage>=0&&value.stage<=7&&Number.isInteger(value.capacity)&&value.capacity>=0&&value.capacity<=30&&Number.isInteger(value.budget)&&value.budget>=0&&value.budget<=300&&Array.isArray(value.trail))s=value}catch{}
const teams=[
  {from:0,to:1,title:'بیزینس مسئله را تعریف کرد',text:()=> 'بیزینس یک فهرست قابل‌اجرا برای این هفته می‌خواهد. ریزش یعنی تمدید نشدن تا تاریخ قرارداد. برای تشخیص ریسک فقط اطلاعات هفت روز پیش از تمدید را استفاده کنید.'},
  {from:1,to:2,title:'دیتا ساینس قرارداد پیش‌بینی را تحویل داد',text:()=> 'از ۱۲۰۰ قرارداد سررسیدشده، ۹۰۰ مورد قدیمی‌تر برای آموزش و ۳۰۰ مورد بعدی برای ارزیابی کنار گذاشته شد. برچسب تمدید فقط برای ارزیابی نتیجه استفاده می‌شود.'},
  {from:2,to:1,title:'مدل با یک خط مبنا مقایسه شد',text:()=> 'خطای بریر روی ۳۰۰ مشتری بعدی برای مدل '+fa(score.toFixed(3))+' و برای پیش‌بینی ثابت '+fa(baseScore.toFixed(3))+' است؛ کمتر بهتر است. '+(score<=baseScore?'مدل لجستیک':'خط مبنای ثابت')+' برای رتبه‌بندی انتخاب شد.'},
  {from:1,to:0,title:'بیزینس رتبه‌بندی را دریافت کرد',text:()=> 'برای ۱۲۰ مشتری هفتهٔ جدید نمرهٔ ریسک داریم. این نمره به‌تنهایی اثر تماس را نشان نمی‌دهد. بیزینس از فروش ظرفیت تماس و از مارکتینگ شرایط پیشنهاد را می‌گیرد.'},
  {from:3,to:0,title:'فروش ظرفیت و مجوز تماس را بررسی کرد',text:()=> 'فروش این هفته ظرفیت '+fa(s.capacity)+' تماس دارد. از ۱۲۰ مشتری، '+fa(cohort.filter(a=>a.consent).length)+' نفر مجوز تماس دارند.'},
  {from:4,to:0,title:'مارکتینگ پیشنهاد تمدید را تأیید کرد',text:()=> 'اعتبار هر پیشنهاد ۱۰ یورو و بودجه '+fa(s.budget)+' یورو است. '+fa(allowed.length)+' مشتری هم واجد شرایط پیشنهادند و هم مجوز تماس دارند.'},
  {from:0,to:0,title:'پیشنهاد مشترک برای تأیید آماده است',text:()=> 'مشتریان واجد شرایط به ترتیب ریسک جفت شدند. در هر جفت، قرعهٔ ثابت شبیه‌سازی تعیین کرد چه کسی پیشنهاد دریافت کند و چه کسی در گروه مقایسه باشد.'}
];
const duties=['تعریف ریزش، جمع‌بندی پیشنهاد و ثبت نتیجه.','تعیین زمان پیش‌بینی، بررسی شواهد و تحویل رتبه‌بندی.','آموزش یک مدل و ارزیابی آن روی دورهٔ بعدی.','بررسی رضایت مشتری و ظرفیت واقعی تماس.','کنترل شرایط پیشنهاد و بودجهٔ اعتبار تمدید.'];
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
    const line=roles[team]+' '+(team===3?'ظرفیت تماس':'بودجهٔ پیشنهاد')+' را از '+fa(old)+' به '+fa(s[property])+' تغییر داد.';
    s.trail.push({actor:team,line,body:'نسخهٔ '+fa(s.version)+' با قید جدید محاسبه شد. رتبه‌بندی مدل همان رتبه‌بندی قبلی است.'});focus=team;
  }
  save();render();
}
function addRow(parent,tag,text){const el=document.createElement(tag);el.textContent=text;parent.append(el)}
function outcome(list){
  const hit=list.filter(r=>r.offer&&r.renew).length,control=list.filter(r=>!r.offer&&r.renew).length;
  return 'در هر گروه '+fa(list.length/2)+' مشتری بود. '+fa(hit)+' نفر از گروه پیشنهاد و '+fa(control)+' نفر از گروه مقایسه تمدید کردند. اختلاف مشاهده‌شده در این اجرای کوچک برای اثبات اثر پیشنهاد کافی نیست.';
}
function render(){
  const ready=s.stage===teams.length,list=ready?plan():[],count=list.length/2,latest=teams[s.stage-1];
  $('play').textContent=timer?'Ⅱ مکث':s.stage?'▶ ادامه':'▶ اجرای شبیه‌سازی';$('play').disabled=$('step').disabled=ready;
  $('counter').textContent='قدم '+fa(s.stage)+' از '+fa(teams.length);
  $('capacity').value=s.capacity;$('budget').value=s.budget;
  $('capacity').disabled=$('budget').disabled=s.closed;
  $('capLabel').textContent=fa(s.capacity)+' تماس';$('budgetLabel').textContent=fa(s.budget)+' یورو';
  $('nodes').replaceChildren();roles.forEach((r,i)=>{
    const b=document.createElement('button');b.className='node'+(focus===i?' selected':'')+(latest&&(latest.to===i||latest.from===i)?' active':'');
    b.setAttribute('aria-pressed',String(i===focus));addRow(b,'b',r);addRow(b,'span',s.trail.some(e=>e.actor===i)?'گزارش دارد':'در انتظار');
    b.onclick=()=>{halt();focus=i;$('agent').open=true;render()};$('nodes').append(b);
  });
  $('agentName').textContent=roles[focus];const report=$('agentReport');report.replaceChildren();addRow(report,'p','مسئولیت: '+duties[focus]);
  const own=s.trail.filter(e=>e.actor===focus);
  if(own.length)own.slice(-3).forEach(e=>addRow(report,'p',e.line+' — '+e.body));else addRow(report,'p','هنوز اقدامی ثبت نشده است.');
  $('route').textContent=s.closed?'بیزینس · نتیجهٔ ثبت‌شده':latest?(latest.from===latest.to?'بیزینس · پروندهٔ مشترک':roles[latest.from]+' ← '+roles[latest.to]):'بیزینس · سؤال آغازین';
  $('subject').textContent=s.closed?'نتیجهٔ تمدید ثبت شد':latest?.title||'کدام مشتری‌ها ممکن است تمدید نکنند؟';
  $('detail').textContent=s.closed?outcome(list):latest?latest.text():'از ۱۲۰ مشتری که به تاریخ تمدید نزدیک‌اند، یک فهرست برای فروش می‌خواهیم. اجرای شبیه‌سازی را بزنید تا ببینید هر تیم روی این تصمیم چه اثری دارد.';
  $('revision').textContent=s.version>1?'نسخهٔ پیشنهاد: '+fa(s.version)+' · با قیود اصلاح‌شده':'';
  $('state').textContent=s.closed?'نتیجه ثبت شد':ready?(count?'در انتظار تأیید شما':'منتظر اصلاح شرایط'):'در حال آماده‌سازی';
  $('num').textContent=$('compare').textContent=ready?fa(count):'—';$('spend').textContent=ready?fa(count*10)+' €':'—';
  $('reason').textContent=ready?'از '+fa(allowed.length)+' مشتری واجد شرایط، ظرفیت فروش '+fa(s.capacity)+' و بودجه برای '+fa(Math.floor(s.budget/10))+' پیشنهاد است. '+(count?'کوچک‌ترین محدودیت اندازهٔ فهرست را تعیین کرد.':'برای ادامه ظرفیت و بودجهٔ حداقل یک تماس لازم است.'):'بعد از تحویل مدل و تأیید ظرفیت و بودجه، فهرست قابل بررسی می‌شود.';
  $('approve').hidden=!ready||s.closed||!count;$('observed').textContent=s.closed?outcome(list):'';
  $('customers').hidden=!ready||!count;$('rows').replaceChildren();
  for(const r of list){const row=document.createElement('tr');[fa(r.a.id),pct(model(r.a)),fa(r.a.days),r.offer?'تماس و پیشنهاد':'مقایسه',s.closed?(r.renew?'بله':'خیر'):'در انتظار'].forEach(x=>addRow(row,'td',x));$('rows').append(row)}
  $('log').replaceChildren();[...s.trail].reverse().forEach(e=>addRow($('log'),'li',e.line));if(!s.trail.length)addRow($('log'),'li','درخواست بیزینس آمادهٔ ارسال است.');
}
$('play').onclick=()=>{if(timer){halt();render()}else{timer=setInterval(advance,2900);advance()}};
$('step').onclick=()=>{halt();advance()};
$('approve').onclick=()=>{if(s.closed||s.stage!==7||!plan().length)return;s.closed=true;focus=0;s.trail.push({actor:0,line:'مالک بیزینس نسخهٔ '+fa(s.version)+' را تأیید کرد.',body:outcome(plan())});save();render()};
for(const id of ['capacity','budget']){
  $(id).oninput=e=>{$(id==='capacity'?'capLabel':'budgetLabel').textContent=fa(e.target.value)+(id==='capacity'?' تماس':' یورو')};
  $(id).onchange=e=>revise(e.target);
}
$('reset').onclick=()=>{halt();s=fresh();focus=0;save();$('agent').open=false;$('customers').open=false;render()};
render();
