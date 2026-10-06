/* Shared helpers and state. No third-party scripts. Escape all user/model/file strings before DOM rendering. */
export const $ = id => document.getElementById(id);
export const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const number = value => Number.isFinite(value) ? value.toFixed(4) : '—';
export const state = {config: null, runs: [], selected: null, view: 'projects', project: null, projects: [], campaignProject: 'general'};
export const labels = {adult:'Adult income',bank_marketing:'Bank marketing',breast_cancer:'Breast cancer',heart_disease:'Heart disease',credit_default:'Credit default',german_credit:'German credit',mushroom:'Mushroom',spambase:'Spambase',online_shoppers:'Online shoppers',wine_quality:'Wine quality',hyperack:'HyperAck delivery acceptance',telco_churn:'Telco customer churn'};
export async function api(path, options = {}) {
  const response = await fetch('/api' + path, {...options, headers: {'Content-Type':'application/json','X-DCLab-Token':state.config?.csrf || '', ...options.headers}});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : (data.detail && data.detail.message) || JSON.stringify(data.detail));
  return data;
}
export function notice(message) { $('notice').textContent = message; $('notice').classList.toggle('hidden', !message); }
