'use strict';
function createRetentionCohort(){
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
function fit(useTickets=true){
const features=a=>useTickets?[1,a.days/50,a.tickets/6]:[1,a.days/50,0];
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
return {score,baseScore,model,allowed};
}
return {train,validation,cohort,fit};
}
