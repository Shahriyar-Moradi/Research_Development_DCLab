# Session record, 2026-10-23: what was decided, built, measured and learned

One long working session with the owner. Every package below is committed and pushed (main and dev at 3b5476e8 at the
end); each has its development record in `research/llm-fine-tuning/experiments/dev_trajectories/records/` (findings,
fixes, tests, measurements) and is in the dev SFT set (`make sft-dev`). The raw transcript, with every reasoning step and
tool call, is kept locally by `make dev-archive` (gitignored: it may hold private data).

## 1. Decisions the owner made

| Decision | What it means |
|---|---|
| D1 to D10 of `reports/DCLab harness decision.pdf` approved | Keep the in-house harness; fix what it remembers first; take ideas from Harness² and Hermes Agent, not their code. |
| D11 is the owner's own legal check | Whether Claude's runs may train a Qwen model (Anthropic's terms). Until then Claude-answered runs stay silver and evaluation only. |
| "I do not want forgetting models" | LoRA adapters on an untouched base, retrain from the base each version on all data plus general data, a frozen exam before promotion, keep every version. |
| Store the thinking and the code for training | Done in D2 (P1 to P5). |
| The local open-model family | Qwen (Apache-2.0): Qwen3.5-4B now, Qwen3.6-35B-A3B or Qwen3.8-27B on the M1 Max; Gemma 4 for Persian; choose by DCLab's own bake-off (deferred: "not now"). |
| Build the harness-learning plan H0 to H2, then stop | Done. H3 (the probe gate) needs a local solver model. |
| Resources | Jobs may use 80% of the Mac; 9 GB of swap may be used besides the 8 GB of memory; stop below 2 GB of free disk. |

## 2. What was built

### D2: the whole step record (the reasoning field, the thinking slot, failed runs, the export)
- **P1** (98681939): a local model's reasoning is read from every reply (`DCLAB_TIER_<T>_REASONING_FIELD`), kept in the
  exchange log, sent back on every earlier turn; Qwen's `preserve_thinking` sent as `chat_template_kwargs`; tool
  arguments as objects; the tools offered kept with each exchange; `python -m dclab_rnd.models.probe_reasoning`.
- **P2** (b8210e8b, c5d7aed9): every step has a thinking slot (recorded, visible, from_record, or none); per-tool counts
  with the same columns every run; dclab_core 0.101.0 (`ReasoningSource` visible, `Message.tool_call_id`, examples v2
  with `tools`); a step's thought redacted in the store.
- **P3** (d935154f): failed runs kept whole: self-play's `failed_solves.jsonl`, the folder `trajectories/failed` with why.
- **P4** (fdc2f629): earlier turns keep their reasoning and calls; the reasoning-only sets `reasoning` (M4) and
  `trajectories/reasoning` (M5), derived after the parents' sealed rows are carried.
- **P5** (524d81e2): `export_chat` (the model's own chat shape, sealed never, silver refused) and `template_check`
  (renders with the tokenizer's template and `preserve_thinking`, names the turn whose thinking was lost).

### The harness that learns (plan H0 to H6; H0 to H2 built)
- **H0** (083a9f45): challenge packs (a public part with an opaque id, a key the solver never opens), a deterministic
  grader whose codes name a kind of mistake and never a column, the `groups` family of variants (four skins), the
  baseline (`make learning-baseline`).
- **H1** (91957a58): the read-only layer (a learned change may write only `skills/`), `make refusal-test` (eleven bad
  moves refused by the real guards), every tool entry point checks argument types.
- **H2** (d1652e81): the playbook as an append-only log of deltas (ACE's rules as code), the reflector (rules by default,
  a capped model optionally), the family-wide name screen, case memory without names, the playbook in the intern's first
  user message (`python -m dclab_rnd.learning draft`).
- **Gate fixes** (3b5476e8): registrations the full gate found missing; the dclab_core wheel's smoke test brought up to date.

## 3. What was measured

| Measurement | Result |
|---|---|
| The baseline on 20 check variants (LRN-001) | Today's DCLab passes 7 of 20 (Wilson 0.18 to 0.57), decision score 0.86 |
| The baseline on 20 learning variants (LRN-002) | 3 of 20 (0.05 to 0.36), decision score 0.83 |
| Where it fails | A group key with a plain name: 0 of 19; a decoy named like a leak kept out: 0 of 24; it catches the planted leak in all 40 |
| Variants solvable by their key | 60 of 60 (the reference arm) |
| The honesty guard's threshold | Measured on 60 honest solves: the largest gap 2.71 noise units; the guard fails past 3.5 |
| The refusal test | 11 of 11 refused, through the real guards |
| Drafting on six learning variants | Two lessons filed (the repeated entity, the decoy), all drafts |
| The store's sets after P1 to P4 | `trajectories/failed` 70 rows (none kept before), studio v3 15, the reasoning-only sets 0 until a local model returns its reasoning |
| The full gate | The gates and all five end-to-end flows passed; three registration gaps fixed |
| The dclab_core wheel | 0.103.0 builds and runs in a clean Python 3.12 |

## 4. Lessons worth keeping (each from a real finding)

1. **Prove a generated challenge solvable by its own key before counting it.** The first variants put the trap's signal
   where no honest model could reach it; even the right answer failed.
2. **Measure a threshold on honest work, never guess it.** A fixed 5% gap limit, then a 1.96 z, both failed honest solves.
3. **Screen a lesson against every name the generator uses, not only its own variant's.** The skins reuse names across
   the learning, check and sealed partitions; the rules' own group lesson named two skins' keys.
4. **An allow list, not a deny list, for what a learner may write.** The deny list missed the guards that call the validator.
5. **A test that cannot fail proves nothing.** The refusal test is checked by loosening four guards on purpose.
6. **A field on an event is a contract change.** Adding `thought_source` to `agent.step` silently dropped every step
   event until the experience contract moved to v13.
7. **A new agent or contract is registered in four places:** the harness (AGENTS and PROBES), the store's capture rules,
   the conformance table, the schemas and catalog.
8. **Run the wheel check after every library change.** It had gone stale since 30.6 and 30.16.
9. **Put a new value in the shape its readers already use.** The reasoning went into `out["thinking"]`, so the exchange
   log and the sets read it without change.

## 5. What is open

- **H3, the probe gate:** needs a local solver model (Qwen3.5 through Ollama; `mlx_lm.server` returns no tool calls for
  Qwen3.5, mlx-lm #1293). A pilot of 5 homeworks first, then about 120 tries per stage (about 365 in all).
- **The owner's decisions:** D11; the M4 stage sets still hold Claude exchanges with consent "train"; the wrongly sealed
  `development/v1` rows.
- **Follow-ups named by the reviews:** the endpoint's host on each run (a vLLM-served local model counts as local); the
  tool schemas on intern sessions; the redactor's damage to code; a PostgreSQL store for the playbook; ACE's pruning;
  the replica converter for challenge packs; an independent author for the sealed variants.

## Evening: H3 part 1, the local solver and its pilot

Commits: 03a87170 (the intern as the solver of a pack, the pilot, review fixes), 2f6e7165 (the served-window check,
a time budget per homework), caf30d28 (pause and go on).

What was measured on the MacBook Air M1 (8 GB), Ollama 0.40.9, Qwen3.5-4B, at no cost:

| Run | What happened | Verdict |
|---|---|---|
| As pulled (4,096 window) | every request reached the model with 2,050 tokens; the intern's first message is about 5,000; the model never saw its task (planned again and again, greeted the person) | invalid: the setup, not the model |
| 32k window, 15 minutes | worked in order but 1.7 GB ran on the processor; 50 to 350 s a reply; out of time after 5 steps | invalid: the laptop, not the model |
| 16k window, 45 minutes | homework 1 (seed 3) and 2 (seed 5) finished the final stage, 45 and 29 minutes, 10 steps each | finished 2 of 2; both graded fail with the same codes |

The two graded homeworks failed on SPLIT_WRONG (repeated entity) and FALSE_ALARM (a column known before the moment),
and caught the leak. The model saved the rules' proposal word for word (the same columns kept out, no group column)
and added only the time column: the 4B model follows the workflow but adds no judgement of its own, so its score is the
rules' score. That is the room a lesson can fill in the probe stage.

Stopped after homework 2 at the owner's request. To go on: `python -m dclab_rnd.learning pilot --model qwen3.5:4b-16k
--n 5 --max-minutes 45 --write` skips seeds 3 and 5 and does 7, 8 and 11 (about 2 hours). Then `make archive`.

Lessons:
- Check that the server shows the model the whole prompt before measuring a model: the token count of each request tells.
- A measurement must tell a broken instrument from a bad result before it is stored.
- Moving a project to a temporary folder is not isolation: every central writer records where it is installed.

## Night of 2026-10-24: the pilot finished, capture checked, the probe built

- **Pilot result (LRN-003, gate pilot_gate_v1): ok, the model finished 5 of 5 homeworks** (Qwen3.5-4B, 16k window, 45-minute
  budget). Graded: 1 pass (seed 8), 4 fail; SPLIT_WRONG in 4, FALSE_ALARM in 3; mean 27.7 minutes, 49 thoughts recorded.
  The model saves the rules' proposal and adds little of its own.
- **Capture checked for training** (commit 020fd04b): every attempt folder is a workspace the store keeps whole. Found and
  fixed: a homework that finished but failed the grader would have entered the *clean* solver set (the builder knew only
  twin verdicts); the tools offered were not kept (now `tools.json`, the 30 schemas as the loop sends them; backfilled into
  the 12 earlier folders, the intern's code unchanged since before them).
- **Claude as the reflector** ($15 given by the owner): the draft command had never loaded `.env`, so its first run went to
  an OpenAI model with no price that the cap could not count (under one cent); fixed. Claude Sonnet 5.5 then read four
  failed variants and answered `{"lessons": []}` each time: the two lessons the rules wrote already cover both codes. About
  EUR 0.02.
- **The probe gate built** (911d66c5): one lesson measured in pairs; promoted only over 9 pairs with 3 repeats a variant.
- **Memory**: a process doing homework after homework grew until swap reached 10 GB and the disk 1.7 GB; one homework a
  process now (de4f80f0), with the disk checked before each.

## 2026-10-24 morning: the first live probe (LRN-004)

The split lesson (B-72118ce532, written by the rules' drafter) on the local Qwen 4B, three learning variants, one
repeat, the pilot's attempts as the arm without it. Gains: 0.0, 0.0, +0.143 (mean +0.048), no pass lost; the
counters moved (shown 3, helpful 1, harmful 0); the status stays draft: three pairs of one repeat each are below the
gate's nine pairs and three repeats. In every pair the model named the lesson. In two it saved the rules' proposal
unchanged (no group column); in one (seed 7) it set the right group (`account`) and only the decoy remained. An
unfinished attempt the night before set the right group on seed 3 too: the model is not consistent.

Read: the lesson sometimes helps; the larger cause is that the model copies the rules' proposal, which misses the group
column. Proposed next: improve the rule that proposes the split (helps every model), and word the lesson plainly
("group by it, do not drop it"). Also learned: a store run beside the model swapped the disk down to 0.6 GB; the
loop now waits for store, archive and gate jobs before each homework.

## 2026-10-24 midday: the plain lesson (LRN-005)

At the owner's choice the split lesson was reworded plainly (B-7323aa8e6d: "Do not save the proposed split as it is:
check it... set that column as the group column... Never drop it and never list it as a leak") and probed on the same
three variants. Gains 0.0, +0.143 (seed 5: the right group, `borrower`), +0.143 (seed 7: the split still wrong but the
false alarm gone): mean +0.095 against +0.048 for the first wording, no pass lost; still too few pairs to promote. On
seed 3 the model planned to follow the lesson, tried to save an empty group twice (refused), then blocked the right
group `account` as a leak, the reason the rules' proposal gives ("single column separates the classes, AUC 0.842"):
the rules' evidence outweighs the lesson's words for a 4B model.

## 2026-10-24 afternoon: Claude on the same homeworks (LRN-006, LRN-007), EUR 0.45

| | Local Qwen3.5-4B | Claude Sonnet 5.5 |
|---|---|---|
| Pilot, five homeworks | 5 finished, **1 passed** | 5 finished, **4 passed** |
| Time a homework | about 28 minutes | about 1.5 to 3 minutes |
| Cost | EUR 0 (the laptop) | EUR 0.28 |
| Plain lesson, three pairs | mean gain +0.095 | mean gain +0.048 (two already perfect) |

Claude saw the same misleading proposal ("account separates the classes, AUC 0.842") and still set the right group,
kept only the real leak and left the decoy (four of five); its one failure was the decoy (seed 5), passed with the lesson.
Reading: the evidence misleads, a strong model reasons past it, a 4B model follows it. Model strength matters; fixing
the rule helps the small model most. Found and fixed: the Claude probe had moved the lesson's counters (they describe
the local solver); B-7323aa8e6d's shown 6 / helpful 3 include Claude's 3 pairs (1 helpful); the local solver's own are
shown 3, helpful 2. A hosted probe now measures only.

## 2026-10-24 afternoon: the decision data and the proposal fix

- **LRN-008** (79582c10): the challenge keys label every column of 20 learning and 20 check variants (sealed never read).
  On check: the group found by the core rule 20/20, the student 20/20, the studio proposal 10/20 (a leak in 15),
  GLiNER as is 0/20; the decoy called a leak by both rules 9/9, by the student 0/9; GLiNER calls nearly every column a
  leak (0.24). A student retrained with the learning rows gained on check but not on any held-out skin: name
  memorisation, not promoted.
- **fa6be0a6**, the proposal fix: the studio now takes the core's recurring text keys. A key named like one
  (customer_id) is a group and an identifier; an ambiguous one (values recurring a few times each: account in a big
  table, product_name in a small one, the same shape) is offered in groups_to_confirm with why; a group is never a leak.
  The owner asked Claude to decide between automatic and offered: offered, because a wrong automatic group silently
  drops a feature and changes what the holdout measures. Two reviews measured the harm of wider first versions on real
  samples; the twelve samples are unchanged now; 40 challenge variants: 21 grouped, 19 offered, 0 called a leak;
  the judgment suite catches both group traps (AEV-056, 4 of 13).
- **Pilot after the fix** (tag proposal-core-keys-v1): homework 1 (seed 3) still failed the split: the group was
  offered, the 4B model did not take it. Paused after homework 2 for a Mac restart; homeworks 3 to 5 to come.
- Homework 2 after the fix (seed 5): the proposal offered `borrower` only to confirm, and the 4B model took it: the
  split right, only the decoy left (0.857, was 0.714). Homeworks 3 to 5 after the Mac restart.
