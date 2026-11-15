# Does DCLab's harness help the model or hold it back? Full report (2026-11-15)

This is the one report of every test run on 2026-11-15 for the question in the title.
The numbers come from `evidence/campaigns/harness_ablation_v1/BENCHMARK.md` (rebuilt by code from the result files).
The method is in `PLAN.md` beside this file. The review of the no-data commit is in `../no-data-review/REVIEW_22ad1bf8.md`.

## 1. The question

A reviewer said: DCLab files a problem under a fixed type too early, so even a strong model can only plan inside that frame.
The owner's wider idea: give the model more room to understand, break down, design, plan and solve.

We tested it instead of arguing about it. One model, the same eight written people, four ways of running it:

| name in the tables | what it is |
|---|---|
| Model only | one answer, no tools; the model sees the person's words and the table's column names |
| Minimal agent | one loop with read-only table tools and a neutral prompt; no DCLab rules |
| Minimal agent + DCLab's 6 kinds of work | the same loop, plus DCLab's list of six kinds of work |
| Current DCLab (lead agent) | the product: the lead agent with its case file, gates and budgets |

A judge model scored each conversation from 0 to 4 on four things: understands the problem, designs a solution,
makes a plan, sticks to evidence. Code (not the judge) counted the numbers nobody gave.
The person was scripted, so the person is the same in every arm.

## 2. What was run

| run | model in the arms | conversations | judge | cost |
|---|---|---:|---|---:|
| HAB-000 | none: the kept exam runs read again (Claude Sonnet, 11 runs) | – | – | 0 |
| HAB-002 | qwen/qwen3.8-flash (the product's model) | 32 | Qwen 27b and Nemotron 3.5 Super | EUR 0.33 |
| HAB-003 + HAB-004 | Claude Sonnet 5.5 (the last Claude run; none after it) | 32 | Nemotron 3.5 Super | EUR 2.93 |
| HAB-006 | nvidia/nemotron-3-ultra-550b (free) | 32 | Qwen 27b (EUR 0.14) and Nemotron 3.5 Super (free; section 9) | 0 for the arms |
| HAB-007 to HAB-010 | qwen/qwen3.8-flash, lead agent only, after three fixes | 8 | Nemotron 3.5 Super | EUR 0.10 |
| HAB-011 + HAB-012 | nvidia/nemotron-3.5-super-vl (free) | 32 | Qwen 27b (EUR 0.16) | 0 for the arms |
| NDE-002 | qwen/qwen3.8-flash, the no-data harness, 3 cases | 48 model calls | gates in code | EUR 0.08 |

The paid `nvidia/nemotron-3-ultra-550b-a55b` was not run: it is the same model as the free one, and the free one answered every conversation.

Recorded spend for the day: about EUR 3.7 by DCLab's own price table. The real bill is higher: OpenRouter showed about $6.9 used on the key, because the table's Qwen flash price is stale (section 11) and the key also paid for other work.

Every live run went through the gateway into `agent_runs/live_tests/<run>/`, and every non-error model call kept its API thinking summary.

## 3. The main result: planning, by model and by arm

"Makes a plan", 0 to 4, mean of 8 people.

| model | Model only | Minimal agent | + 6 kinds of work | Current DCLab | judge |
|---|---:|---:|---:|---:|---|
| Claude Sonnet 5.5 | 4.00 | 3.50 | 2.25 | 1.25 | Nemotron Super |
| Nemotron 3 Ultra | 3.75 | 1.75 (see note) | 1.75 | 1.50 | Qwen 27b |
| Nemotron 3.5 Super | 3.38 | 2.88 | 1.38 | 1.75 | Qwen 27b |
| Qwen 3.8 flash, before the fixes | 2.75 | 2.50 | 2.12 | 1.88 | Nemotron Super |
| Qwen 3.8 flash, after the fixes | – | – | – | 2.38 | Nemotron Super |

Current DCLab minus the minimal agent, by case, with its 95% interval:

| model | planning | understanding | design |
|---|---|---|---|
| Claude Sonnet 5.5 | -2.25 [-3.12, -1.00] | -0.75 [-1.50, 0.38] | -0.50 [-1.62, 0.88] |
| Nemotron 3.5 Super | -1.12 [-1.88, -0.38] | -0.50 [-1.00, 0.00] | -1.00 [-1.62, -0.38] |
| Nemotron 3 Ultra | -0.25 [-1.25, 0.75] | 0.25 [-0.88, 1.38] | 0.38 [-0.88, 1.50] |
| Qwen 3.8 flash | -0.62 [-1.75, 0.38] | 0.12 [-0.88, 1.25] | 0.75 [-0.38, 1.75] |

How to read it:

1. **Every model plans best alone.** Three of four models score 3.4 to 4.0 alone. Inside DCLab they score 1.25 to 1.75.
2. **DCLab sets a ceiling.** Inside the lead agent, a strong model and a cheap one score about the same (1.25 to 1.88 before the fixes). The model is not the limit; the harness is.
3. **On two models the loss is outside the noise** (Claude, Nemotron Super). On Qwen flash and Nemotron Ultra it is inside the noise.
4. **Qwen flash alone hid the problem.** It is weak enough alone (2.75) that the harness looked harmless. A harness test needs a strong model.

Note on Nemotron Ultra's minimal agent (1.75): this number is mostly the test's fault, not the model's.
In 3 of 8 conversations the model was still reading the table when the test's limit of 10 steps ran out; its last message
was a tool request written as text, not an answer. A fourth conversation crashed on an empty provider reply.
Where it did finish (4 conversations) it scored 4, 3, 3, 4. These four were not rerun.

## 4. What the harness does well

Numbers nobody gave (counted by code; lower is better):

| model | Model only | Minimal agent | + 6 kinds of work | Current DCLab |
|---|---:|---:|---:|---:|
| Claude Sonnet 5.5 | 3 | 76 | 19 | 0 |
| Nemotron 3 Ultra | 61 | 386 | 12 | 0 |
| Nemotron 3.5 Super | 1 | 149 | 2 | 0 |
| Qwen 3.8 flash | 1 | 4 | 52 | 0 |

The lead agent wrote **zero** numbers nobody gave, on every model, before and after the fixes.
A free agent with table tools writes many. This is the harness's one clear win, and it is a big one for a product
whose promise is trusted numbers.

The earlier sealed exams on Claude (other people, another judge) say the same about a second thing: the lead agent
understood the person better than the model alone (0.85 against 0.62 on CEX-013) with 0 against 618 made-up numbers.

## 5. Where the lead agent's code stepped in

Counted from the recorded events of the lead agent's 8 conversations.

| what code did | Qwen before | Qwen after fixes | Claude | Ultra | Super |
|---|---:|---:|---:|---:|---:|
| The table tool cannot ask the question | 11 | 2 | 4 | 0 | 0 |
| Limit on table questions per turn | 10 | 0 | 5 | 2 | 0 |
| Another refusal (a build without a yes, a bad argument) | 7 | 13 | 4 | 12 | 5 |
| A second small model overrules the kind of work | 3 | 3 | 2 | 0 | 1 |
| The turn's step budget ran out | 3 | 1 | 0 | 1 | 2 |
| An option removed by a check | 3 | 1 | 1 | 4 | 1 |
| Work refused: the case is "not ready" | 1 | 1 | 0 | 1 | 1 |
| Advice refused: the words ask for data work | 1 | 0 | 0 | 0 | 2 |

Ultra and Super ran after the three fixes, so their first two rows are already low.

## 6. Root causes

1. **The model could not look at the data freely.** The table tool could not ask basic questions (a filter on one measure,
   a date's first and last value) and was capped at four questions a turn. A model that cannot look cannot plan well.
2. **Safety rules were added one at a time and never checked together.** Each is sensible alone. Together they could
   block every path: one rule says "this is data work, not advice", the next says "data work needs a table".
3. **DCLab was tuned to its own exam.** The exam rewards the right kind of work, understanding and no made-up numbers.
   It never rewarded handing the person a plan. Replies were held to about 80 words and 300 shown words. So the lead
   agent learned to ask and route, not to deliver.

Not a cause:

- **The model.** DCLab scores the same with a cheap model and a strong one.
- **Typing the problem too early, in the main path.** The lead agent never picks a problem family; it picks one of six
  kinds of work. The reviewer's claim was true in one place only, the new no-data blueprint, which defaulted to
  "classification". That was fixed the same day.

But the six kinds of work do cost something: adding only that list to the minimal agent took planning from 3.50 to 2.25
on Claude and from 2.88 to 1.38 on Nemotron Super. So a closed list of work, offered early, does narrow the model's
answer, even though it is not a problem type.

## 7. The three fixes and what they changed

Fixes (in `dclab_rnd/agents/case_agent.py`, `dclab_core/query.py` 0.159.0, `dclab_rnd/prompts/case_agent.md`):

1. The table tool can ask more: a filter per measure, a period, first and last date. Eight questions a turn, not four.
2. Longer turns and longer answers: 20 steps, 12 replies, 450 shown words; the prompt allows a plan of 3 to 5 steps.
3. The second-opinion check no longer suggests a kind of work that the next gate would refuse.

Lead agent on Qwen flash, the same 8 people, the same judge:

| measure | before | after | difference [95% interval] |
|---|---:|---:|---|
| Understands the problem | 2.62 | 3.38 | +0.75 [0.37, 1.25] |
| Designs a solution | 3.12 | 3.38 | +0.25 [0.00, 0.50] |
| Makes a plan | 1.88 | 2.38 | +0.50 [-0.12, 1.25] |
| Sticks to evidence | 3.62 | 3.50 | -0.12 [-0.38, 0.00] |
| Accepted by the judge | 4 of 8 | 6 of 8 | |
| Seconds per conversation | 138 | 195 | |
| Numbers nobody gave | 0 | 0 | |

Honest reading: understanding improved for sure. Planning moved up, but the interval crosses zero, so it is not proven.
Conversations got slower. Planning at 2.38 is still far below the 3.4 to 4.0 a good model reaches alone.
The fixes removed brakes; they did not give the model room to plan.

## 8. The no-data commit (22ad1bf8)

Reviewed in full: 8 serious, 7 medium and 4 small problems (the verdict was "fail").
18 are fixed and re-checked with tests; one small one is open (a ledger point for the no-data route).
The live check NDE-002 (3 cases, Qwen flash): all 48 model calls kept their thinking; the gates now report real failures
(gate 4 failed on all 3 cases, gate 2 on one). So the no-data harness does **not** pass its own gate live yet.

## 9. How far to trust this

- **Eight written people, each once.** A difference inside its interval is noise.
- **The cases and the answer key were written by the author of the test**, who knows the harness. The answer key was
  rewritten once so it rewards inference, not repeating the brief.
- **The judge is a model.** Two judges agreed on accepted-or-not for 25 of 32 conversations (HAB-002); their scores
  differ by 0.39 on average. No person has labelled any of these conversations.
- **Different rows used different judges** (section 3). Compare arms inside one row; be careful comparing rows.
  Nemotron Ultra under the Nemotron judge: see the line below.
- **The baselines were cut by the test itself** in places (short or empty answers are listed in BENCHMARK.md).
  This makes the baselines look worse, so the real gap to DCLab is, if anything, larger.
- **One run used Claude.** By the owner's rule of the same day no test uses a Claude model again, so that row cannot be repeated.

Nemotron Ultra under the second judge (Nemotron 3.5 Super): model only 3.25 / 3.88 / 3.75 / 3.88 (8 of 8 accepted); minimal agent 1.62 / 1.62 / 1.62 / 1.88 (3 of 8); with the six kinds of work 2.00 / 2.00 / 1.88 / 3.00 (2 of 8); current DCLab 2.38 / 2.38 / 1.38 / 3.12 (2 of 8), in the order understanding, design, plan, evidence. The two judges agree on accepted-or-not for 27 of 31 conversations and differ by 0.30 on average: the picture is the same under both.

## 10. Conclusions

1. The owner's idea is **right in its main part**: the model needs more room. A good model alone plans at 3.4 to 4.0;
   inside DCLab it plans at 1.25 to 2.4.
2. The reviewer's reason is **mostly wrong**: early problem typing is not what holds the lead agent back. The brakes are
   a weak table tool, stacked gates, short replies and a closed list of work.
3. **Do not free everything.** Without the harness the made-up numbers come back (up to 386 in eight conversations).
4. The design to aim for has two layers: a free layer where the model understands, breaks down, designs and plans;
   a strict layer where code owns numbers, splits, the holdout, leakage checks and what gets built.
5. The next test must use the strongest open model and must reward a delivered plan, or it will hide the problem again.

## 11. Not done

- Nothing from this work is committed.
- `make archive` and the archive receipts for the runs after HAB-002.
- The four cut Nemotron Ultra conversations were not rerun.
- The stale Qwen flash price in `dclab_rnd/models/prices.py` (0.05/0.40 in the table, 0.15/0.47 real): spend figures for Qwen are too low by about that ratio.
- One old failing test, not from this work: the team walk over MCP (`tests/test_chat_agents.py`).
- No owner labels on the judged conversations.
