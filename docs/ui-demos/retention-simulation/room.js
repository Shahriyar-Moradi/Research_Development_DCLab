'use strict';
const desks={
  biz:{team:'بیزینس',person:'مینا',job:'مالک تصمیم',icon:'ب',color:'#e5ebcb'},
  ds:{team:'دیتا ساینس',person:'نیلوفر',job:'دیتا ساینتیست',icon:'د',color:'#d7e8ed'},
  ml:{team:'یادگیری ماشین',person:'کاوه',job:'مهندس ML',icon:'م',color:'#e5def0'},
  sales:{team:'فروش',person:'رها',job:'مدیر فروش',icon:'ف',color:'#f2e3d0'},
  mkt:{team:'مارکتینگ',person:'آوا',job:'مدیر کمپین',icon:'ک',color:'#efdee5'}
};
const ids=Object.keys(desks),$=id=>document.getElementById(id),fa=n=>Number(n).toLocaleString('fa-IR');
const key='dclab-team-room-1',data=createRetentionCohort();
const title={brief:'تعریف تصمیم این هفته',idle:'منتظر درخواست بیزینس',audit:'بازبینی تعریف و ستون هدف',clarify:'پاسخ به سؤال دیتا ساینس',blocked:'انتظار برای تعریف قابل‌اندازه‌گیری',spec:'انتخاب ویژگی‌های مدل',evaluate:'بازبینی شواهد مدل',verify:'تفسیر رتبه‌بندی',capacity:'تأیید ظرفیت تماس',offer:'تأیید بودجهٔ پیشنهاد',join:'گردآوری پاسخ تیم‌ها',approve:'بررسی و تأیید فهرست',feedback:'بازبینی پیشنهاد',done:'گزارش تحویل شد',closed:'پرونده بسته شد',unavailable:'اجرای پرونده متوقف شد'};
function initial(){return {v:1,tasks:{biz:'brief',ds:'idle',ml:'idle',sales:'idle',mkt:'idle'},chat:Object.fromEntries(ids.map(i=>[i,[]])),log:[],capacity:null,budget:null,tickets:null,model:null,ranking:false,version:0,closed:false,rejected:false}}
let s=initial(),selected='biz',timer=null,cursor=0,fitted=null;
try{const saved=JSON.parse(localStorage.getItem(key));if(saved?.v===1&&saved.tasks&&saved.chat)s=saved}catch{}
function agent(id){return 'ایجنت '+desks[id].team}
function say(id,who,text,kind='agent'){s.chat[id].push({who,text,kind})}
function seed(){say('biz',desks.biz.person,'این هفته ۱۲۰ مشتری در آستانهٔ تمدید داریم. با چه کسانی تماس بگیریم؟','human');say('biz',agent('biz'),'برای اقدام، یک رتبه‌بندی، ظرفیت فروش و بودجهٔ پیشنهاد لازم داریم. هدف کمپین را تأیید می‌کنی؟')}
if(!s.chat.biz.length)seed();
function send(from,to,text){
  s.log.push({from,to,text});say(from,agent(from),'برای '+agent(to)+' فرستادم: '+text,'handoff');
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
  s.version++;s.tasks.biz=rows().length?'approve':'feedback';
  say('biz',agent('biz'),'پیشنهاد نسخهٔ '+fa(s.version)+' آماده است: '+fa(rows().length/2)+' تماس و به همین تعداد مشتری مشابه در گروه مقایسه. '+(rows().length?'مالک تصمیم می‌تواند فهرست را ببیند و تأیید کند.':'ظرفیت یا بودجه کافی نیست؛ از تیم مربوطه اصلاح بخواه.'));
}
function outcome(){const list=rows(),n=list.length/2;return 'از '+fa(n)+' نفر در هر گروه، '+fa(list.filter(r=>r.offer&&r.renew).length)+' نفر با پیشنهاد و '+fa(list.filter(r=>!r.offer&&r.renew).length)+' نفر در گروه مقایسه تمدید کردند. این یک مشاهدهٔ ساختگی با نمونهٔ کوچک است.'}
// A person decides only at their own desk. Their agent computes or forwards a scoped response.
function choices(id){
  if(s.closed)return [];
  switch(s.tasks[id]){
  case 'brief': return [['start','شروع کمپین تمدید']];
  case 'audit':return [['confirm','تأیید برچسب «تمدید نشدن»'],['challenge','درخواست بررسی «لغو صریح»']];
  case 'clarify':return [['resolve','هدف همین کمپین: تمدید نشدن'],['hold','توقف تا دریافت سابقهٔ لغو']];
  case 'spec':return [['full','آموزش با سابقهٔ استفاده و تیکت'],['simple','آموزش فقط با سابقهٔ استفاده']];
  case 'evaluate':return [['release','ارسال گزارش ارزیابی'],['rerun','تغییر ویژگی‌ها و آموزش مجدد']];
  case 'verify':return [['interpret','تأیید تفسیر و تحویل به بیزینس']];
  case 'capacity':return [['capacity','ثبت ظرفیت و پاسخ به بیزینس']];
  case 'offer':return [['offer','ثبت بودجه و شرایط پیشنهاد']];
  case 'approve':return [['approve','تأیید و مشاهدهٔ تمدیدها'],['change','بازگرداندن به فروش'],['terms','بازنگری بودجه با مارکتینگ']];
  case 'feedback':return [['change','بازگرداندن به فروش'],['terms','بازنگری بودجه با مارکتینگ']];
  default:return [];
  }
}
function apply(id,action,value,note,automatic=false){
  const option=choices(id).find(c=>c[0]===action);if(!option)return;
  if((action==='capacity'||action==='offer')&&(!Number.isInteger(value)||value<0||value>(action==='capacity'?30:300)))return;
  say(id,desks[id].person+(automatic?' · شبیه‌سازی':''),option[1]+(value!=null?' · '+fa(value):'')+(note?' — '+note:''),'human');
  if(action==='start'){
    s.tasks={biz:'join',ds:'audit',ml:'idle',sales:'capacity',mkt:'offer'};
    send('biz','ds','آیا می‌توان هفت روز پیش از سررسید، ریسک تمدید نشدن را تخمین زد؟');
    say('ds',agent('ds'),'در داده، وضعیت تمدید پس از سررسید ثبت شده است. آیا این همان خروجی مورد نظر است؟ پیش از آموزش، تعریف تو لازم است.');
    send('biz','sales','این هفته چند تماسِ مجاز می‌توانیم بگیریم؟');say('sales',agent('sales'),'از ۱۲۰ حساب، '+fa(data.cohort.filter(a=>a.consent).length)+' مورد مجوز تماس دارند. ظرفیت این هفته را ثبت کن.');
    send('biz','mkt','برای پیشنهاد تمدید چه بودجه‌ای در اختیار داریم؟');say('mkt',agent('mkt'),'هر پیشنهاد به ۱۰ یورو اعتبار نیاز دارد. بودجهٔ در دسترس را تأیید کن.');
  }else if(action==='challenge'){
    s.tasks.ds='blocked';s.tasks.biz='clarify';send('ds','biz','سابقهٔ لغو صریح در جدول نیست. آیا منظور همین کمپین تمدید نشدن است یا باید منتظر منبع لغو بمانیم؟');
  }else if(action==='resolve'){
    s.tasks.biz='join';s.tasks.ds='audit';send('biz','ds','مالک بیزینس تعریف تمدید نشدن را تأیید کرد. قرارداد را با همین هدف بازبینی کن.');
  }else if(action==='hold'){
    send('biz','ds','تا تأمین منبع رویداد لغو، پرونده را متوقف کن.');s.rejected=true;s.closed=true;ids.forEach(i=>s.tasks[i]='unavailable');
  }else if(action==='confirm'){
    s.tasks.ds='join';s.tasks.ml='spec';send('ds','ml','قرارداد تأیید شد: برچسب تمدید نشدن، زمان تصمیم هفت روز مانده به تمدید. ۹۰۰ ردیف قدیمی برای آموزش و ۳۰۰ ردیف بعدی برای ارزیابی.');
    say('ml',agent('ml'),'ویژگی‌های در دسترس همان روز را انتخاب کن. نتیجهٔ تمدید وارد ویژگی‌ها نمی‌شود.');
  }else if(action==='full'||action==='simple'){
    s.tickets=action==='full';fitted=data.fit(s.tickets);s.model={score:fitted.score,baseline:fitted.baseScore};s.tasks.ml='evaluate';
    say('ml',agent('ml'),'آموزش تمام شد. خطای بریر روی ۳۰۰ قرارداد بعدی: مدل '+fa(fitted.score.toFixed(3))+'، خط مبنا '+fa(fitted.baseScore.toFixed(3))+'؛ کمتر بهتر است. روش با خطای کمتر برای فهرست انتخاب می‌شود.');
  }else if(action==='rerun'){
    s.tasks.ml='spec';s.model=null;fitted=null;say('ml',agent('ml'),'گزارش به تیم دیگری فرستاده نشده است. مشخصات اجرای بعدی را تعیین کن.');
  }else if(action==='release'){
    s.tasks.ml='done';s.tasks.ds='verify';send('ml','ds','گزارش مدل با '+fa(s.tickets?2:1)+' ویژگی: خطای مدل '+fa(s.model.score.toFixed(3))+' و خطای مبنا '+fa(s.model.baseline.toFixed(3))+' روی ۳۰۰ قرارداد.');
    say('ds',agent('ds'),'این مدل ریسک را مرتب می‌کند. برای سنجش اثر تماس، یک گروه مقایسه با ریسک مشابه پیشنهاد می‌دهم. نتیجه را تحویل بدهم؟');
  }else if(action==='interpret'){
    s.ranking=true;s.tasks.ds='done';send('ds','biz','رتبه‌بندی ۱۲۰ مشتری آماده است. افراد مجاز را برحسب ریسک جفت می‌کنیم و از هر جفت یک نفر با قرعه وارد گروه پیشنهاد می‌شود.');gather();
  }else if(action==='capacity'||action==='offer'){
    if(action==='capacity')s.capacity=value;else s.budget=value;
    s.tasks[id]='done';send(id,'biz',action==='capacity'?'ظرفیت قطعی این هفته '+fa(value)+' تماس است. حساب‌های فاقد رضایت حذف می‌شوند.':'بودجهٔ تأییدشده '+fa(value)+' یورو است. هر تماس واجد شرایط ۱۰ یورو اعتبار می‌گیرد.');gather();
  }else if(action==='change'||action==='terms'){
    const target=action==='change'?'sales':'mkt';s.tasks.biz='join';s.tasks[target]=target==='sales'?'capacity':'offer';
    if(target==='sales')s.capacity=null;else s.budget=null;
    send('biz',target,'پیشنهاد نسخهٔ '+fa(s.version)+' برای بازنگری برگشت. محدودیت جدید را در میز خودت تأیید کن.');
  }else if(action==='approve'){
    s.closed=true;s.tasks.biz='closed';say('biz',agent('biz'),'تصمیم نسخهٔ '+fa(s.version)+' ثبت شد. '+outcome());
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
const drafts={};
function visit(id){
  stop();
  selected=id;
  render();
}
function stateText(id){
  if(choices(id).length)return 'نوبت '+desks[id].person;
  if(s.tasks[id]==='join'&&id==='ds')return 'منتظر گزارش ML';
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
    name.append(el('b',d.team),el('small',d.person+' · '+stateText(id)));
    const status=choices(id).length?'waiting':s.tasks[id]==='done'?'done':'';
    b.append(mark,name,el('i',null,'dot '+status));
    $('desks').append(b);
  }
}
function renderMessages(){
  const d=desks[selected];
  $('role').textContent=d.person+' · '+d.job;
  $('deskName').textContent='میز '+d.team;
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
  if(!chat.childElementCount)chat.append(el('p','درخواست‌های این تیم و پاسخ ایجنت اینجا ثبت می‌شود.','empty'));
  chat.scrollTop=chat.scrollHeight;
}
function report(id){
  const sent=s.log.filter(m=>m.from===id).at(-1);
  const next=choices(id).map(c=>c[1]).join(' یا ');
  return (sent?'آخرین تحویل به '+desks[sent.to].team+' — '+sent.text:'هنوز گزارشی از این میز به تیم دیگری ارسال نشده.')+
    ' وضعیت فعلی: '+stateText(id)+'. '+(next?'تصمیم بعدی: '+next+'.':'');
}
function renderActions(){
  const box=$('actions'),options=choices(selected),note=$('note');
  box.replaceChildren();
  note.value=drafts[selected]||'';
  note.hidden=!options.length;
  $('noteLabel').hidden=note.hidden;
  $('noteLabel').textContent='پاسخ در نقش '+desks[selected].person;
  const numeric=options.some(([a])=>a==='capacity'||a==='offer');
  if(numeric){
    const capacity=selected==='sales',input=el('input');
    input.type='number';input.min='0';input.max=capacity?'30':'300';input.step='1';input.required=true;
    input.id='constraint';input.value=capacity?'18':'120';
    const label=el('label',capacity?'تعداد تماس':'بودجه (یورو)');
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
    if(next)box.append(button('رفتن به میز '+desks[next].team,()=>visit(next),'primary'));
  }
  box.append(button('گزارش این ایجنت',()=>{
    stop();
    say(selected,desks[selected].person,'چه کاری انجام دادی و قدم بعدی چیست؟','human');
    say(selected,agent(selected),report(selected));
    save();render();
  }));
}
function fact(label,value){
  const row=el('div',null,'fact');
  row.append(el('small',label),el('strong',value));
  return row;
}
function renderCase(){
  const list=rows(),box=$('artifacts');
  $('brief').textContent=s.rejected?'پرونده تا دریافت سابقهٔ لغو متوقف شد.':s.closed?outcome():'از میان ۱۲۰ مشتری در آستانهٔ تمدید، این هفته به چه کسانی پیشنهاد بدهیم؟';
  box.replaceChildren(
    fact('قرارداد پیش‌بینی',s.rejected?'در انتظار منبع لغو':s.tasks.ml!=='idle'?'تمدید نشدن · هفت روز پیش از سررسید':'در انتظار تأیید تعریف'),
    fact('ارزیابی مدل',s.model?'بریر '+fa(s.model.score.toFixed(3))+' · مبنا '+fa(s.model.baseline.toFixed(3)):'در انتظار اجرای آموزش'),
    fact('ظرفیت فروش',s.capacity==null?'در انتظار فروش':fa(s.capacity)+' تماس'),
    fact('بودجهٔ مارکتینگ',s.budget==null?'در انتظار مارکتینگ':fa(s.budget)+' یورو')
  );
  if(s.version)box.append(fact('پیشنهاد نسخهٔ '+fa(s.version),s.tasks.biz==='join'?'برگشت برای بازنگری':fa(list.length/2)+' تماس · '+(s.closed?'تأیید شد':'در انتظار تصمیم')));
  $('list').replaceChildren();
  if(!list.length)$('list').append(el('p','هنوز فهرستی برای تماس آماده نیست.','empty'));
  for(const r of list){
    const parts=['حساب '+fa(r.a.id),r.offer?'پیشنهاد':'مقایسه','ریسک '+fa(Math.round(fittedModel().model(r.a)*100))+'٪'];
    if(s.closed)parts.push(r.renew?'تمدید شد':'تمدید نشد');
    $('list').append(el('div',parts.join(' · ')));
  }
  $('history').replaceChildren();
  $('count').textContent=fa(s.log.length)+' تحویل';
  for(const m of s.log.slice().reverse()){
    const row=el('li');
    row.append(el('b',desks[m.from].team+' به '+desks[m.to].team+' — '),document.createTextNode(m.text));
    $('history').append(row);
  }
  if(!s.log.length)$('history').append(el('li','با شروع کمپین، سه درخواست از بیزینس به تیم‌ها ارسال می‌شود.','empty'));
}
function render(){
  $('play').textContent=timer?'مکث و ادامهٔ دستی':'تماشای همکاری تیم‌ها';
  $('play').disabled=s.closed;
  renderDesks();renderMessages();renderActions();renderCase();
}
function step(){
  for(let i=0;i<ids.length;i++){
    const id=ids[cursor++%ids.length],options=choices(id);
    if(!options.length)continue;
    selected=id;
    let action=options[0][0];
    if(s.tasks[id]==='feedback')action=s.capacity===0?'change':'terms';
    const value=action==='capacity'?18:action==='offer'?120:null;
    apply(id,action,value,null,true);
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
  if(timer){stop();$('play').textContent='تماشای همکاری تیم‌ها'}
});
$('restart').onclick=()=>{stop();render();$('confirm').showModal()};
$('keep').onclick=()=>$('confirm').close();
$('clear').onclick=()=>{
  $('confirm').close();
  s=initial();selected='biz';cursor=0;fitted=null;
  for(const id of ids)delete drafts[id];
  seed();save();render();
};
render();
