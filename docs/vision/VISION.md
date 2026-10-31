# DCLab vision

The owner's vision, 1 November 2026: **the goal DCLab must go for.** Every plan, phase and package should say how it
moves DCLab toward it.

Official documents (English; the PDFs are in `output/`):

- [`vision.html`](vision.html), **DCLab Vision**: the vision, mission, the problem, who we serve, the three layers,
  principles, where we are today and what success looks like (`output/DCLab_Vision.pdf`).
- [`plan.html`](plan.html), **The Plan to Reach the Vision**: the starting point, the gaps, phases 14 to 19 with their
  tests, risks and the owner's decisions (`output/DCLab_Vision_Plan.pdf`).

## The vision

DCLab is an AI-powered platform that takes companies from raw data to trusted models, and from models to better
decisions.

For data scientists and ML engineers, DCLab is a smart assistant. An AI agent first understands the problem, then
designs and runs the right ML pipeline. Before a model leaves DCLab, it is checked: no leakage, an honest score, and
the code that rebuilds it. Developers work from their own notebooks and tools, and spend their time on the real
problem instead of repetitive work.

For businesses, DCLab goes one step further. Soon every company will run many models, each making its own
predictions. But predictions alone are not enough. Companies need help turning them into the best decisions. DCLab
will give each company its own decision model, built on its own data, that grows together with the company and its
business.

Every business decision workflow starts with a data science and ML team. DCLab supports that team and the
decision-makers they serve, as one assistant for both.

## Mission

Help every data team build models that are right, help every business turn those models into its best decisions, and
keep learning from what happens next.

## Words

- **The company decision model** turns many models' predictions, plus the company's costs, budgets and rules, into one
  best action, and learns from outcomes. It is not the same thing as the small *decision models* of the ledger
  (`dclab_core/decision_models.py`), which answer DCLab's own technical choices, such as a column's role.
- **The order of work:** the data team first, then the decision layer on top of those models, then each company's
  decision model learning from its decisions and their outcomes.
