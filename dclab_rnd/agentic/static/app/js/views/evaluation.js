DC.view('evaluation', {
  async enter(el) {
    const { $, esc } = DC;
    const box = $('#ev-box', el);
    let p = null;
    try { p = await DC.currentProject.get(); } catch (e) { p = null; }
    DC.markSample(!p);
    if (!p) {
      $('#ev-eyebrow', el).textContent = 'Sample · evaluation';
      DC.states.empty(box, 'No project is open, so there is no model to evaluate. Start one on Home; its evaluation appears here once its final stage has scored the held-back cases.', { href: '#home', action: 'Go to Home' });
      return;
    }
    $('#ev-eyebrow', el).textContent = `${p.name} · evaluation`;
    const final = (p.records || {}).final;
    if (!final) { DC.states.empty(box, 'The final stage has not run yet. The evaluation is read from its record: the held-back cases are scored once, there.', { href: '#project', action: 'Open the workflow' }); return; }
    DC.states.loading(box, { text: 'Reading the evaluation…' });
    let data;
    try { data = await DC.client.projectEvaluation(p.id); }
    catch (e) { DC.states.failed(box, e, { lead: 'Could not read the evaluation: ', retry: () => this.enter(el) }); return; }
    if (DC.state.view !== 'evaluation') return;
    DC.evaluation.render(box, data);
  },
});
