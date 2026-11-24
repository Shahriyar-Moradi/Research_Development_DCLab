# Review of the two Home intelligence documents

**24 November 2026 · Code read at `affefa45` · Reviewer: Claude**

This is the review of the owner's two documents in [originals/](originals/):

- Document 1: `DCLab_01_Home_Multi_Agent_Architecture_EN.md` (the Home Lead looks at a problem from several views).
- Document 2: `DCLab_02_All_Agent_Data_and_Fine_Tuning_EN.md` (collect every agent's experience and train a model).

The improved, simpler build plan that came out of this review is [BUILD_GUIDE.md](BUILD_GUIDE.md).

## The short verdict

- **Document 1 is a good idea with sound safety rules.** It is too big for a first step, and five of its statements
  about the code are not correct.
- **Document 2 describes a system that mostly already exists** in this repository. Building it again would create a
  second, competing record. Its reasoning policy is right and matches the owner's standing rule.
- **The main risk is not technical.** Our own measurements show that adding agents and structure has often made
  answers worse. So the plan must measure before it switches anything on.

## What our own results say

| Date | What was measured | Result |
|---|---|---|
| 2026-11-15 | The same model alone, and inside DCLab's tight harness (planning score) | 4.0 alone, 1.3 inside |
| 2026-11-16 | "Think first": room for the Lead to plan | 2.75 → 3.88 |
| 2026-11-19 | A loop where the model picks one small step and rules check it | 0.4–1.4 against 3.75 with it off |
| 2026-11-24 | Team race: rules, AI helper, AI agent (19 cases) | 9, 8 and 2 passed; the agent took 374 s a case |

**Lesson:** give the model room to think, keep code strict about facts and numbers, and never assume that more agents
means better answers.

## Document 1: what is good

- It is additive: a new module, one narrow connection to the Lead, and a switch that is off by default.
- In round one the specialists work independently. Agreement alone is not treated as proof.
- Every finding has a type (fact, hypothesis, inference, unknown), and a fact needs evidence.
- It separates four modes: no data, data, simulation, decision.
- It sets a budget and says "more agents are not themselves a success metric".
- It asks for a comparison of several ways before release.

## Document 1: what is wrong or missing

| # | What the document says | What the code shows | What the build guide does |
|---|---|---|---|
| 1 | `orchestra/` belongs to Workspace and must not be used. | `orchestra/` is mostly **Home's own engine**: Home's Build and its team panel run on it. Workspace is `teamspace/` plus two routers and two pages. | The forbidden import is `teamspace`. `orchestra/` is simply not needed. |
| 2 | With the flag off, Home is unchanged. | Adding a tool to the Lead changes a pinned hash (`prompts.fingerprints()` hashes the Lead's tool list), flag on or off. | The Lead is *offered* exactly today's tools when the flag is off. The hash changes once, and the two gate suites are re-run. |
| 3 | Specialists run in parallel. | Today they run one at a time. Parallel calls need per-thread clients, a split of the EUR 1.50 case cap, and time limits. | First version: one after another. Parallel is a later task. |
| 4 | Eight new roles beside the five existing specialists. | That gives 13 roles and two critics. | Three roles first. More only when a test shows each helps. |
| 5 | Four test arms: model alone, Lead, Lead + specialists, + follow-up. | These mix two things: new analysis tools and more agents. A win would not say which one helped. | A control arm: the Lead with the computed evidence only, no roles. |
| 6 | Release when there is "meaningful improvement", scored by a judge. | No number is fixed, and our judge agreed poorly with the owner before (0.66 on CEX-013). | Scores by code against planted causes. The rule is fixed and pinned before any live run. |
| 7 | The Data Scientist gets a Python sandbox. | None exists for a Home chat. The one used in evaluations is macOS-only and can read the whole disk. | Code runs a fixed set of typed analyses. No free code on real data. |
| 8 | A new case object with its own `verified_facts`. | The case file already keeps facts with their source (said, computed, assumed, confirmed). Two stores would disagree. | Facts stay in the case file. The new result holds evidence and reports only. |
| 9 | All new code in `dclab_rnd/home_intelligence/`. | The repository rule: contracts and rules in `dclab_core`, wiring in `dclab_rnd`. | `dclab_core/kpi_analysis.py` and `dclab_core/intelligence.py`. |
| 10 | A model "router" selects the roles. | That is one more model call and one more place to fail. | No router: the three roles always run. |
| 11 | Use synthetic e-commerce data with hidden truth. | Right idea. The existing sales replica has two tables, and the Lead reads one. | A new single-table orders replica with seven planted causes and a separate answer key. |
| 12 | Workspace is "entirely out of scope". | Home and Workspace are joined today: the Lead can ask the teams, and their answers appear in the Home chat. | The owner decided (2026-11-24): leave that bridge as it is; the new module never touches Workspace. |

## Document 2: what is good

- **The reasoning policy is right.** Keep the API's thinking summary when one is returned. Never write a made-up
  explanation and call it the model's reasoning. This is exactly the owner's standing rule.
- Explicit "absent" statuses instead of empty values.
- Rights and consent gates before any training export.
- Splits by group (case, customer, generator family), so near-copies cannot leak across the split.
- A sealed evaluation set that is never used for training.

## Document 2: what is wrong or missing

| # | What the document proposes | What already exists | What the build guide does |
|---|---|---|---|
| 1 | A new append-only event ledger | `dclab_rnd/experience/` with `dclab_core.experience.ExperienceEvent` (91 event kinds, visit, run and request ids) | Reuse. |
| 2 | Hooks that capture every model call | `dclab_rnd/models/gateway.py` `_keep_exchange`: every call is kept with its thinking summary, including failed calls, with `thinking_asked` and `thinking_dropped` | Reuse. A test proves the new role calls are kept. |
| 3 | Decision records | `dclab_rnd/ledger/` with `dclab_core.ledger.DecisionRecord` (84 decision points) | Reuse. |
| 4 | Raw, normalized and curated layers; a registry with statuses; quarantine; dedup | `dclab_rnd/store/` (raw → records → sets; one keeper per data kind; curators that keep, mark silver, drop or quarantine; sha256 dedup) and `dclab_rnd/lake/` | Reuse. |
| 5 | Privacy and rights gates | `dclab_rnd/consent.py`, `dclab_core/redact.py`, `store/provider_outputs.py` `may_train`, `store/writers.py` | Reuse. |
| 6 | Dataset registry and exporters | `dclab_rnd/datasets.py`, `dclab_core/datasets.py` (`DatasetManifest`, sealed partition), `store/export_chat.py`, `studio/sft.py`, `lake/train.py` | Reuse. |
| 7 | An 80/10/10 split | The store already splits by group: sealed 0.2, dev 0.2, train 0.6 | Use the existing split. |
| 8 | Deployment of a tuned model with rollback | `dclab_rnd/models/tuned.py` (shadow → promote → retire) | Reuse later. |

**Real gaps that Document 2 does not name:**

- `code_runs.record` has no draft reference, so code run in a Home chat without a project would not be captured. (Not
  needed yet: the first version runs no free code.)
- No test proves that every role in the registry leaves a model exchange and a run record.
- There is no DPO trainer and no QLoRA script in the repository, only a command in a README.
- The research campaign calls its model through another library and keeps no exchange (an old open gap).

**Fine-tuning (FT1 to FT6) has no entry condition.** There are no Home-intelligence cases yet, and the disk is nearly
full. The build guide leaves fine-tuning out and sets a condition: at least 50 verified Home-intelligence cases from
at least 10 different problems, a recorded rights review for the models that wrote them, and free disk for a pilot.

## What the improved plan keeps from both documents

- The idea itself: several independent views of one problem, then one answer from the Lead.
- Additive code behind a switch that is off by default.
- Typed findings, and facts that need evidence.
- No invented numbers, ever.
- Experience capture from the first version.
- A comparison before the switch is turned on, with "keep the flag off" as a valid result.
- The reasoning policy.

## What the improved plan adds

- A middle setting of the switch (`tools`) that returns computed evidence without roles. It is the control that tells
  us whether roles help.
- Seven planted causes, including three that punish over-confident answers: "nothing really changed", "the data has a
  gap" and "only the season changed".
- A penalty for naming a false cause, so an answer that lists every possible cause cannot win.
- A pass rule fixed and pinned by hash before any live run. A tie goes to the simpler way.
- Small tasks with exact files, function names, tests and commands, so a smaller model can build it and a reviewer can
  check it.

## Honest limits of the improved plan

- **The analyst has little freedom in the first version.** It reads fixed analyses instead of choosing its own. This
  trades the owner's "AI chooses its tools" principle for safety and simplicity. If the roles show promise, letting
  the analyst choose analyses is the next step.
- **Three roles one after another can take 1 to 3 minutes.** The comparison measures it.
- **The test data is invented.** A win on it does not prove a win on a real company's data.
- **Live measurement needs OpenRouter credit** (about EUR 3 to 5). The account was empty on 2026-11-24.
- **Roles may not help at all.** Then the result is still useful: the Lead keeps the analysis tool, and we add no
  roles.
