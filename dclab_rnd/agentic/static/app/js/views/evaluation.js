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
    await DC.evaluation.mount(box, p);
  },
});
