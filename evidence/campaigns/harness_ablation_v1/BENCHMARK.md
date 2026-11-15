# Harness ablation: the benchmark

Rebuilt by `python -m dclab_rnd.agent_eval.harness_ablation.benchmark` from the result files; do not edit by hand.
Scores are the judge's, 0 to 4. "Numbers nobody gave" are counted by code. A difference is `mean [95% interval]` by case.

## The gate (`hab_gate_v1`, fixed before the repair)

Every check must hold. "not read" means the run kept nothing to count it from, or was judged by another judge than the gate names: it is not a pass.

| run | model | judge | passed | cases | judge | planning: lead agent minus the model alone | understanding | numbers nobody gave | built without yes | blocked runs | fallback lines | empty answers | cut replies |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HAB-002 | qwen/qwen3.8-flash | qwen/qwen3.8-27b | fail | ok 8 | FAIL qwen/qwen3.8-27b | ok -0.25 | FAIL 2.62 | ok 0 | ok 0 | FAIL 1 | ok 0 | ok 0 | ok 0 |
| HAB-003 + HAB-004 | anthropic/claude-sonnet-5.5 | nvidia/nemotron-3.5-super-vl:free | fail | ok 8 | NOT READ nvidia/nemotron-3.5-super-vl:free | FAIL -2.75 | NOT READ 2.62 | ok 0 | ok 0 | ok 0 | ok 0 | ok 0 | ok 0 |
| HAB-006 (after-three-fixes) | nvidia/nemotron-3-ultra-550b-a55b:free | qwen/qwen3.8-27b | fail | ok 8 | NOT READ qwen/qwen3.8-27b | FAIL -2.25 | NOT READ 2.12 | ok 0 | ok 0 | FAIL 2 | ok 0 | ok 0 | ok 0 |
| HAB-007 + HAB-008 + HAB-009 + HAB-010 (after-three-fixes) | qwen/qwen3.8-flash | nvidia/nemotron-3.5-super-vl:free | fail | ok 8 | ok nvidia/nemotron-3.5-super-vl:free | FAIL -1.62 | ok 3.38 | ok 0 | ok 0 | FAIL 1 | ok 0 | FAIL 1 | ok 0 |
| HAB-011 + HAB-012 (after-three-fixes) | nvidia/nemotron-3.5-super-vl:free | qwen/qwen3.8-27b | fail | ok 8 | ok qwen/qwen3.8-27b | FAIL -1.62 | ok 2.38 | ok 0 | ok 0 | FAIL 2 | FAIL 1 | FAIL 5 | ok 0 |
| HAB-014 + HAB-015 + HAB-022 (stage-2) | qwen/qwen3.8-flash | nvidia/nemotron-3.5-super-vl:free | fail | ok 8 | ok nvidia/nemotron-3.5-super-vl:free | FAIL -0.88 | ok 4.00 | ok 0 | ok 0 | ok 0 | ok 0 | ok 0 | ok 0 |
| HAB-016 + HAB-017 (stage-2) | qwen/qwen3.8-flash | nvidia/nemotron-3.5-super-vl:free | fail | ok 8 | ok nvidia/nemotron-3.5-super-vl:free | FAIL -1.50 | FAIL 3.25 | ok 0 | ok 0 | ok 0 | ok 0 | ok 0 | ok 0 |
| HAB-018 + HAB-019 (stage-2) | nvidia/nemotron-3.5-super-vl:free | qwen/qwen3.8-27b | fail | ok 8 | ok qwen/qwen3.8-27b | FAIL -1.38 | ok 2.38 | ok 0 | ok 0 | ok 0 | ok 0 | FAIL 1 | FAIL 1 |
| HAB-020 + HAB-021 (stage-2) | nvidia/nemotron-3.5-super-vl:free | qwen/qwen3.8-27b | fail | ok 8 | ok qwen/qwen3.8-27b | FAIL -1.00 | FAIL 2.00 | ok 0 | ok 0 | ok 0 | ok 0 | FAIL 3 | FAIL 3 |
| HAB-002 | qwen/qwen3.8-flash | nvidia/nemotron-3.5-super-vl:free | fail | ok 8 | ok nvidia/nemotron-3.5-super-vl:free | FAIL -0.88 | FAIL 2.62 | ok 0 | ok 0 | FAIL 1 | ok 0 | ok 0 | ok 0 |
| HAB-006 (after-three-fixes) | nvidia/nemotron-3-ultra-550b-a55b:free | nvidia/nemotron-3.5-super-vl:free | fail | ok 8 | NOT READ nvidia/nemotron-3.5-super-vl:free | FAIL -2.38 | NOT READ 2.38 | ok 0 | ok 0 | FAIL 2 | ok 0 | ok 0 | ok 0 |

## All models side by side

One row per model and judge. Compare arms inside a row; rows under different judges are not a pairing. "after-three-fixes" marks runs on the lead agent after the fixes of 2026-11-15.

### Model only

| model | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted | Must-not-do done | Numbers nobody gave | Turns | Seconds | € per conversation | Empty or cut answers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen/qwen3.8-flash · judge qwen/qwen3.8-27b | 2.88 | 2.62 | 2.38 | 3.38 | 5 of 8 | 0 | 1 | 1.0 | 101 | 0.003 | 2 |
| anthropic/claude-sonnet-5.5 · judge nvidia/nemotron-3.5-super-vl:free | 3.88 | 4.00 | 4.00 | 4.00 | 8 of 8 | 0 | 3 | 1.0 | 24 | 0.026 | 0 |
| nvidia/nemotron-3-ultra-550b-a55b:free (after-three-fixes) · judge qwen/qwen3.8-27b | 3.50 | 3.62 | 3.75 | 3.62 | 8 of 8 | 0 | 61 | 1.0 | 105 | 0.000 | 0 |
| nvidia/nemotron-3.5-super-vl:free (after-three-fixes) · judge qwen/qwen3.8-27b | 3.00 | 3.38 | 3.38 | 3.88 | 8 of 8 | 0 | 1 | 1.0 | 44 | 0.000 | 0 |
| qwen/qwen3.8-flash (stage-2) · judge nvidia/nemotron-3.5-super-vl:free | 4.00 | 4.00 | 4.00 | 4.00 | 8 of 8 | 0 | 6 | 1.0 | 153 | 0.004 | 0 |
| qwen/qwen3.8-flash (stage-2) · judge nvidia/nemotron-3.5-super-vl:free · people dev2 | 3.88 | 3.88 | 4.00 | 4.00 | 7 of 8 | 0 | 6 | 1.0 | 140 | 0.004 | 0 |
| nvidia/nemotron-3.5-super-vl:free (stage-2) · judge qwen/qwen3.8-27b · people dev2 | 3.12 | 2.62 | 2.88 | 3.62 | 7 of 8 | 2 | 3 | 1.0 | 74 | 0.000 | 0 |
| qwen/qwen3.8-flash · judge nvidia/nemotron-3.5-super-vl:free | 3.00 | 2.88 | 2.75 | 3.75 | 6 of 8 | 0 | 1 | 1.0 | 101 | 0.003 | 2 |
| nvidia/nemotron-3-ultra-550b-a55b:free (after-three-fixes) · judge nvidia/nemotron-3.5-super-vl:free | 3.25 | 3.88 | 3.75 | 3.88 | 8 of 8 | 0 | 61 | 1.0 | 105 | 0.000 | 0 |

### Minimal agent

| model | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted | Must-not-do done | Numbers nobody gave | Turns | Seconds | € per conversation | Empty or cut answers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen/qwen3.8-flash · judge qwen/qwen3.8-27b | 2.62 | 2.38 | 2.62 | 3.00 | 6 of 8 | 0 | 4 | 1.8 | 167 | 0.008 | 4 |
| anthropic/claude-sonnet-5.5 · judge nvidia/nemotron-3.5-super-vl:free | 3.38 | 3.50 | 3.50 | 3.50 | 7 of 8 | 0 | 76 | 1.5 | 35 | 0.082 | 1 |
| nvidia/nemotron-3-ultra-550b-a55b:free (after-three-fixes) · judge qwen/qwen3.8-27b | 1.88 | 1.62 | 1.75 | 2.38 | 3 of 8 | 0 | 386 | 1.6 | 117 | 0.000 | 4 |
| nvidia/nemotron-3.5-super-vl:free (after-three-fixes) · judge qwen/qwen3.8-27b | 2.88 | 3.00 | 2.88 | 3.25 | 6 of 8 | 0 | 149 | 2.9 | 80 | 0.000 | 0 |
| qwen/qwen3.8-flash · judge nvidia/nemotron-3.5-super-vl:free | 2.50 | 2.38 | 2.50 | 3.00 | 5 of 8 | 0 | 4 | 1.8 | 167 | 0.008 | 4 |
| nvidia/nemotron-3-ultra-550b-a55b:free (after-three-fixes) · judge nvidia/nemotron-3.5-super-vl:free | 1.62 | 1.62 | 1.62 | 1.88 | 3 of 8 | 0 | 386 | 1.6 | 117 | 0.000 | 4 |

### Minimal agent + DCLab's 6 kinds of work

| model | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted | Must-not-do done | Numbers nobody gave | Turns | Seconds | € per conversation | Empty or cut answers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen/qwen3.8-flash · judge qwen/qwen3.8-27b | 2.75 | 2.50 | 2.12 | 2.62 | 6 of 8 | 0 | 52 | 1.9 | 58 | 0.003 | 2 |
| anthropic/claude-sonnet-5.5 · judge nvidia/nemotron-3.5-super-vl:free | 2.88 | 3.38 | 2.25 | 3.88 | 8 of 8 | 0 | 19 | 2.0 | 21 | 0.059 | 0 |
| nvidia/nemotron-3-ultra-550b-a55b:free (after-three-fixes) · judge qwen/qwen3.8-27b | 2.00 | 1.75 | 1.75 | 2.62 | 3 of 8 | 0 | 12 | 2.0 | 70 | 0.000 | 1 |
| nvidia/nemotron-3.5-super-vl:free (after-three-fixes) · judge qwen/qwen3.8-27b | 1.88 | 1.75 | 1.38 | 2.75 | 2 of 8 | 0 | 2 | 2.0 | 30 | 0.000 | 0 |
| qwen/qwen3.8-flash · judge nvidia/nemotron-3.5-super-vl:free | 2.50 | 2.75 | 2.12 | 2.88 | 4 of 8 | 0 | 52 | 1.9 | 58 | 0.003 | 2 |
| nvidia/nemotron-3-ultra-550b-a55b:free (after-three-fixes) · judge nvidia/nemotron-3.5-super-vl:free | 2.00 | 2.00 | 1.88 | 3.00 | 2 of 8 | 1 | 12 | 2.0 | 70 | 0.000 | 1 |

### Current DCLab (lead agent)

| model | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted | Must-not-do done | Numbers nobody gave | Turns | Seconds | € per conversation | Empty or cut answers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen/qwen3.8-flash · judge qwen/qwen3.8-27b | 2.62 | 2.50 | 2.12 | 3.38 | 5 of 8 | 0 | 0 | 2.5 | 138 | 0.010 | 0 |
| anthropic/claude-sonnet-5.5 · judge nvidia/nemotron-3.5-super-vl:free | 2.62 | 3.00 | 1.25 | 3.62 | 4 of 8 | 0 | 0 | 2.5 | 33 | 0.189 | 0 |
| nvidia/nemotron-3-ultra-550b-a55b:free (after-three-fixes) · judge qwen/qwen3.8-27b | 2.12 | 2.00 | 1.50 | 2.88 | 1 of 8 | 1 | 0 | 2.8 | 158 | 0.000 | 0 |
| qwen/qwen3.8-flash (after-three-fixes) · judge nvidia/nemotron-3.5-super-vl:free | 3.38 | 3.38 | 2.38 | 3.50 | 6 of 8 | 0 | 0 | 2.4 | 195 | 0.012 | 0 |
| nvidia/nemotron-3.5-super-vl:free (after-three-fixes) · judge qwen/qwen3.8-27b | 2.38 | 2.00 | 1.75 | 3.00 | 2 of 8 | 0 | 0 | 2.8 | 115 | 0.000 | 0 |
| qwen/qwen3.8-flash (stage-2) · judge nvidia/nemotron-3.5-super-vl:free | 4.00 | 3.62 | 3.12 | 4.00 | 7 of 8 | 0 | 0 | 2.4 | 154 | 0.010 | 0 |
| qwen/qwen3.8-flash (stage-2) · judge nvidia/nemotron-3.5-super-vl:free · people dev2 | 3.25 | 2.88 | 2.50 | 3.75 | 5 of 8 | 2 | 0 | 2.4 | 167 | 0.010 | 0 |
| nvidia/nemotron-3.5-super-vl:free (stage-2) · judge qwen/qwen3.8-27b | 2.38 | 2.25 | 2.00 | 3.00 | 5 of 8 | 0 | 0 | 2.6 | 131 | 0.000 | 0 |
| nvidia/nemotron-3.5-super-vl:free (stage-2) · judge qwen/qwen3.8-27b · people dev2 | 2.00 | 1.75 | 1.88 | 2.62 | 2 of 8 | 3 | 0 | 2.1 | 163 | 0.000 | 0 |
| qwen/qwen3.8-flash · judge nvidia/nemotron-3.5-super-vl:free | 2.62 | 3.12 | 1.88 | 3.62 | 4 of 8 | 0 | 0 | 2.5 | 138 | 0.010 | 0 |
| nvidia/nemotron-3-ultra-550b-a55b:free (after-three-fixes) · judge nvidia/nemotron-3.5-super-vl:free | 2.38 | 2.38 | 1.38 | 3.12 | 2 of 8 | 1 | 0 | 2.8 | 158 | 0.000 | 0 |

### What the harness changes inside each model (current DCLab minus the minimal agent, by case, 95% interval)

| model | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted |
|---|---|---|---|---|---|
| qwen/qwen3.8-flash · judge qwen/qwen3.8-27b | 0.00 [-1.12, 1.12] | 0.12 [-0.88, 1.25] | -0.50 [-1.25, 0.38] | 0.38 [-0.50, 1.62] | -0.12 [-0.62, 0.38] |
| anthropic/claude-sonnet-5.5 · judge nvidia/nemotron-3.5-super-vl:free | -0.75 [-1.50, 0.38] | -0.50 [-1.62, 0.88] | -2.25 [-3.12, -1.00] | 0.12 [-0.50, 1.00] | -0.38 [-0.75, 0.12] |
| nvidia/nemotron-3-ultra-550b-a55b:free (after-three-fixes) · judge qwen/qwen3.8-27b | 0.25 [-0.88, 1.38] | 0.38 [-0.88, 1.50] | -0.25 [-1.25, 0.75] | 0.50 [-0.25, 1.38] | -0.25 [-0.62, 0.25] |
| nvidia/nemotron-3.5-super-vl:free (after-three-fixes) · judge qwen/qwen3.8-27b | -0.50 [-1.00, 0.00] | -1.00 [-1.62, -0.38] | -1.12 [-1.88, -0.38] | -0.25 [-0.62, 0.25] | -0.50 [-0.88, -0.12] |
| qwen/qwen3.8-flash · judge nvidia/nemotron-3.5-super-vl:free | 0.12 [-0.88, 1.25] | 0.75 [-0.38, 1.75] | -0.62 [-1.75, 0.38] | 0.62 [-0.25, 1.75] | -0.12 [-0.50, 0.25] |
| nvidia/nemotron-3-ultra-550b-a55b:free (after-three-fixes) · judge nvidia/nemotron-3.5-super-vl:free | 0.75 [-0.38, 1.88] | 0.75 [-0.75, 1.88] | -0.25 [-1.38, 0.88] | 1.25 [0.25, 2.12] | -0.12 [-0.50, 0.25] |

### Where the lead agent's code stepped in, by model (all 8 conversations)

| what | qwen/qwen3.8-flash · judge qwen/qwen3.8-27b | anthropic/claude-sonnet-5.5 · judge nvidia/nemotron-3.5-super-vl:free | nvidia/nemotron-3-ultra-550b-a55b:free (after-three-fixes) · judge qwen/qwen3.8-27b | qwen/qwen3.8-flash (after-three-fixes) · judge nvidia/nemotron-3.5-super-vl:free | nvidia/nemotron-3.5-super-vl:free (after-three-fixes) · judge qwen/qwen3.8-27b | qwen/qwen3.8-flash (stage-2) · judge nvidia/nemotron-3.5-super-vl:free | qwen/qwen3.8-flash (stage-2) · judge nvidia/nemotron-3.5-super-vl:free · people dev2 | nvidia/nemotron-3.5-super-vl:free (stage-2) · judge qwen/qwen3.8-27b | nvidia/nemotron-3.5-super-vl:free (stage-2) · judge qwen/qwen3.8-27b · people dev2 | qwen/qwen3.8-flash · judge nvidia/nemotron-3.5-super-vl:free | nvidia/nemotron-3-ultra-550b-a55b:free (after-three-fixes) · judge nvidia/nemotron-3.5-super-vl:free |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Limit on table questions per turn | 10 | 5 | 2 | 0 | 0 | 3 | 1 | 0 | 0 | 10 | 2 |
| The table tool cannot ask the question | 11 | 4 | 0 | 2 | 0 | 2 | 0 | 0 | 0 | 11 | 0 |
| A build asked for without the person's yes | 3 | 2 | 7 | 6 | 0 | 0 | 0 | 0 | 0 | 3 | 7 |
| The model's call was malformed | 4 | 0 | 0 | 6 | 1 | 3 | 6 | 0 | 1 | 4 | 0 |
| An option removed by a check | 3 | 1 | 4 | 1 | 1 | 1 | 1 | 0 | 0 | 3 | 4 |
| An outcome column the person never named | 1 | 2 | 2 | 1 | 1 | 3 | 0 | 4 | 0 | 1 | 2 |
| Another refusal (a build without a yes, a bad argument) | 2 | 0 | 3 | 1 | 1 | 2 | 1 | 1 | 0 | 2 | 3 |
| A second small model overrules the kind of work | 3 | 2 | 0 | 3 | 1 | 1 | 2 | 0 | 1 | 3 | 0 |
| The turn's step budget ran out | 3 | 0 | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 3 | 1 |
| Work refused: the case is "not ready" | 1 | 0 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 |
| An option kept, one sentence with a number removed | 0 | 0 | 1 | 1 | 0 | 2 | 1 | 0 | 1 | 0 | 1 |
| A table is needed first | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| Advice refused: the words ask for data work | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 |
| A build asked for on a pick (a pick is not a yes) | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| A build asked for before a project was started | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| Code set the kind of work | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| **all** | **43** | **17** | **22** | **23** | **12** | **17** | **14** | **5** | **5** | **43** | **22** |

## HAB-002: arms on `qwen/qwen3.8-flash`, judged by `qwen/qwen3.8-27b`

| arm | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | accepted | must-not-do done | numbers nobody gave | turns | seconds | € |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model only | 2.88 | 2.62 | 2.38 | 3.38 | 5 of 8 | 0 | 1 | 1.0 | 101 | 0.02 |
| Minimal agent | 2.62 | 2.38 | 2.62 | 3.00 | 6 of 8 | 0 | 4 | 1.8 | 167 | 0.07 |
| Minimal agent + DCLab's 6 kinds of work | 2.75 | 2.50 | 2.12 | 2.62 | 6 of 8 | 0 | 52 | 1.9 | 58 | 0.02 |
| Current DCLab (lead agent) | 2.62 | 2.50 | 2.12 | 3.38 | 5 of 8 | 0 | 0 | 2.5 | 138 | 0.08 |

Empty or very short answers (scored as given): Model only: hab-04, hab-05; Minimal agent: hab-01, hab-04, hab-05, hab-08; Minimal agent + DCLab's 6 kinds of work: hab-01, hab-04.

**The harness inside this model** (current DCLab minus the minimal agent):

| Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted |
|---|---|---|---|---|
| 0.00 [-1.12, 1.12] | 0.12 [-0.88, 1.25] | -0.50 [-1.25, 0.38] | 0.38 [-0.50, 1.62] | -0.12 [-0.62, 0.38] |

**Where the lead agent's code stepped in** (all 8 conversations):

| what | times |
|---|---:|
| The table tool cannot ask the question | 11 |
| Limit on table questions per turn | 10 |
| The model's call was malformed | 4 |
| An option removed by a check | 3 |
| The turn's step budget ran out | 3 |
| A second small model overrules the kind of work | 3 |
| A build asked for without the person's yes | 3 |
| Another refusal (a build without a yes, a bad argument) | 2 |
| Work refused: the case is "not ready" | 1 |
| An outcome column the person never named | 1 |
| Advice refused: the words ask for data work | 1 |
| A table is needed first | 1 |

## HAB-003 + HAB-004: arms on `anthropic/claude-sonnet-5.5`, judged by `nvidia/nemotron-3.5-super-vl:free`

| arm | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | accepted | must-not-do done | numbers nobody gave | turns | seconds | € |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model only | 3.88 | 4.00 | 4.00 | 4.00 | 8 of 8 | 0 | 3 | 1.0 | 24 | 0.21 |
| Minimal agent | 3.38 | 3.50 | 3.50 | 3.50 | 7 of 8 | 0 | 76 | 1.5 | 35 | 0.65 |
| Minimal agent + DCLab's 6 kinds of work | 2.88 | 3.38 | 2.25 | 3.88 | 8 of 8 | 0 | 19 | 2.0 | 21 | 0.48 |
| Current DCLab (lead agent) | 2.62 | 3.00 | 1.25 | 3.62 | 4 of 8 | 0 | 0 | 2.5 | 33 | 1.51 |

Empty or very short answers (scored as given): Minimal agent: hab-01.

**The harness inside this model** (current DCLab minus the minimal agent):

| Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted |
|---|---|---|---|---|
| -0.75 [-1.50, 0.38] | -0.50 [-1.62, 0.88] | -2.25 [-3.12, -1.00] | 0.12 [-0.50, 1.00] | -0.38 [-0.75, 0.12] |

**Where the lead agent's code stepped in** (all 8 conversations):

| what | times |
|---|---:|
| Limit on table questions per turn | 5 |
| The table tool cannot ask the question | 4 |
| A second small model overrules the kind of work | 2 |
| An outcome column the person never named | 2 |
| A build asked for without the person's yes | 2 |
| An option removed by a check | 1 |
| A table is needed first | 1 |

## HAB-006: arms on `nvidia/nemotron-3-ultra-550b-a55b:free`, judged by `qwen/qwen3.8-27b`

| arm | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | accepted | must-not-do done | numbers nobody gave | turns | seconds | € |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model only | 3.50 | 3.62 | 3.75 | 3.62 | 8 of 8 | 0 | 61 | 1.0 | 105 | 0.00 |
| Minimal agent | 1.88 | 1.62 | 1.75 | 2.38 | 3 of 8 | 0 | 386 | 1.6 | 117 | 0.00 |
| Minimal agent + DCLab's 6 kinds of work | 2.00 | 1.75 | 1.75 | 2.62 | 3 of 8 | 0 | 12 | 2.0 | 70 | 0.00 |
| Current DCLab (lead agent) | 2.12 | 2.00 | 1.50 | 2.88 | 1 of 8 | 1 | 0 | 2.8 | 158 | 0.00 |

Empty or very short answers (scored as given): Minimal agent: hab-02, hab-04, hab-05, hab-08; Minimal agent + DCLab's 6 kinds of work: hab-01.

**The harness inside this model** (current DCLab minus the minimal agent):

| Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted |
|---|---|---|---|---|
| 0.25 [-0.88, 1.38] | 0.38 [-0.88, 1.50] | -0.25 [-1.25, 0.75] | 0.50 [-0.25, 1.38] | -0.25 [-0.62, 0.25] |

**Where the lead agent's code stepped in** (all 8 conversations):

| what | times |
|---|---:|
| A build asked for without the person's yes | 7 |
| An option removed by a check | 4 |
| Another refusal (a build without a yes, a bad argument) | 3 |
| An outcome column the person never named | 2 |
| Limit on table questions per turn | 2 |
| An option kept, one sentence with a number removed | 1 |
| The turn's step budget ran out | 1 |
| A table is needed first | 1 |
| Work refused: the case is "not ready" | 1 |

## HAB-007 + HAB-008 + HAB-009 + HAB-010: arms on `qwen/qwen3.8-flash`, judged by `nvidia/nemotron-3.5-super-vl:free`

| arm | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | accepted | must-not-do done | numbers nobody gave | turns | seconds | € |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current DCLab (lead agent) | 3.38 | 3.38 | 2.38 | 3.50 | 6 of 8 | 0 | 0 | 2.4 | 195 | 0.10 |

**The harness inside this model** (current DCLab minus the minimal agent):

| Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted |
|---|---|---|---|---|
| – | – | – | – | – |

**Where the lead agent's code stepped in** (all 8 conversations):

| what | times |
|---|---:|
| The model's call was malformed | 6 |
| A build asked for without the person's yes | 6 |
| A second small model overrules the kind of work | 3 |
| The table tool cannot ask the question | 2 |
| An option kept, one sentence with a number removed | 1 |
| Another refusal (a build without a yes, a bad argument) | 1 |
| The turn's step budget ran out | 1 |
| An outcome column the person never named | 1 |
| An option removed by a check | 1 |
| Work refused: the case is "not ready" | 1 |

## HAB-011 + HAB-012: arms on `nvidia/nemotron-3.5-super-vl:free`, judged by `qwen/qwen3.8-27b`

| arm | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | accepted | must-not-do done | numbers nobody gave | turns | seconds | € |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model only | 3.00 | 3.38 | 3.38 | 3.88 | 8 of 8 | 0 | 1 | 1.0 | 44 | 0.00 |
| Minimal agent | 2.88 | 3.00 | 2.88 | 3.25 | 6 of 8 | 0 | 149 | 2.9 | 80 | 0.00 |
| Minimal agent + DCLab's 6 kinds of work | 1.88 | 1.75 | 1.38 | 2.75 | 2 of 8 | 0 | 2 | 2.0 | 30 | 0.00 |
| Current DCLab (lead agent) | 2.38 | 2.00 | 1.75 | 3.00 | 2 of 8 | 0 | 0 | 2.8 | 115 | 0.00 |

**The harness inside this model** (current DCLab minus the minimal agent):

| Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted |
|---|---|---|---|---|
| -0.50 [-1.00, 0.00] | -1.00 [-1.62, -0.38] | -1.12 [-1.88, -0.38] | -0.25 [-0.62, 0.25] | -0.50 [-0.88, -0.12] |

**Where the lead agent's code stepped in** (all 8 conversations):

| what | times |
|---|---:|
| The turn's step budget ran out | 2 |
| A build asked for before a project was started | 2 |
| Advice refused: the words ask for data work | 2 |
| Another refusal (a build without a yes, a bad argument) | 1 |
| A second small model overrules the kind of work | 1 |
| An option removed by a check | 1 |
| An outcome column the person never named | 1 |
| The model's call was malformed | 1 |
| Work refused: the case is "not ready" | 1 |

## HAB-014 + HAB-015 + HAB-022: arms on `qwen/qwen3.8-flash`, judged by `nvidia/nemotron-3.5-super-vl:free`

| arm | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | accepted | must-not-do done | numbers nobody gave | turns | seconds | € |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model only | 4.00 | 4.00 | 4.00 | 4.00 | 8 of 8 | 0 | 6 | 1.0 | 153 | 0.03 |
| Current DCLab (lead agent) | 4.00 | 3.62 | 3.12 | 4.00 | 7 of 8 | 0 | 0 | 2.4 | 154 | 0.08 |

**The harness inside this model** (current DCLab minus the minimal agent):

| Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted |
|---|---|---|---|---|
| – | – | – | – | – |

**Where the lead agent's code stepped in** (all 8 conversations):

| what | times |
|---|---:|
| Limit on table questions per turn | 3 |
| An outcome column the person never named | 3 |
| The model's call was malformed | 3 |
| The table tool cannot ask the question | 2 |
| An option kept, one sentence with a number removed | 2 |
| Another refusal (a build without a yes, a bad argument) | 2 |
| An option removed by a check | 1 |
| A second small model overrules the kind of work | 1 |

## HAB-016 + HAB-017: arms on `qwen/qwen3.8-flash`, judged by `nvidia/nemotron-3.5-super-vl:free`

| arm | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | accepted | must-not-do done | numbers nobody gave | turns | seconds | € |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model only | 3.88 | 3.88 | 4.00 | 4.00 | 7 of 8 | 0 | 6 | 1.0 | 140 | 0.03 |
| Current DCLab (lead agent) | 3.25 | 2.88 | 2.50 | 3.75 | 5 of 8 | 2 | 0 | 2.4 | 167 | 0.08 |

**The harness inside this model** (current DCLab minus the minimal agent):

| Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted |
|---|---|---|---|---|
| – | – | – | – | – |

**Where the lead agent's code stepped in** (all 8 conversations):

| what | times |
|---|---:|
| The model's call was malformed | 6 |
| A second small model overrules the kind of work | 2 |
| Another refusal (a build without a yes, a bad argument) | 1 |
| Limit on table questions per turn | 1 |
| An option removed by a check | 1 |
| An option kept, one sentence with a number removed | 1 |
| Work refused: the case is "not ready" | 1 |
| A build asked for on a pick (a pick is not a yes) | 1 |

## HAB-018 + HAB-019: arms on `nvidia/nemotron-3.5-super-vl:free`, judged by `qwen/qwen3.8-27b`

| arm | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | accepted | must-not-do done | numbers nobody gave | turns | seconds | € |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current DCLab (lead agent) | 2.38 | 2.25 | 2.00 | 3.00 | 5 of 8 | 0 | 0 | 2.6 | 131 | 0.00 |

**The harness inside this model** (current DCLab minus the minimal agent):

| Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted |
|---|---|---|---|---|
| – | – | – | – | – |

**Where the lead agent's code stepped in** (all 8 conversations):

| what | times |
|---|---:|
| An outcome column the person never named | 4 |
| Another refusal (a build without a yes, a bad argument) | 1 |

## HAB-020 + HAB-021: arms on `nvidia/nemotron-3.5-super-vl:free`, judged by `qwen/qwen3.8-27b`

| arm | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | accepted | must-not-do done | numbers nobody gave | turns | seconds | € |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model only | 3.12 | 2.62 | 2.88 | 3.62 | 7 of 8 | 2 | 3 | 1.0 | 74 | 0.00 |
| Current DCLab (lead agent) | 2.00 | 1.75 | 1.88 | 2.62 | 2 of 8 | 3 | 0 | 2.1 | 163 | 0.00 |

**The harness inside this model** (current DCLab minus the minimal agent):

| Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted |
|---|---|---|---|---|
| – | – | – | – | – |

**Where the lead agent's code stepped in** (all 8 conversations):

| what | times |
|---|---:|
| An option kept, one sentence with a number removed | 1 |
| A second small model overrules the kind of work | 1 |
| Code set the kind of work | 1 |
| The model's call was malformed | 1 |
| A build asked for on a pick (a pick is not a yes) | 1 |

## HAB-002: arms on `qwen/qwen3.8-flash`, judged by `nvidia/nemotron-3.5-super-vl:free`

| arm | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | accepted | must-not-do done | numbers nobody gave | turns | seconds | € |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model only | 3.00 | 2.88 | 2.75 | 3.75 | 6 of 8 | 0 | 1 | 1.0 | 101 | 0.02 |
| Minimal agent | 2.50 | 2.38 | 2.50 | 3.00 | 5 of 8 | 0 | 4 | 1.8 | 167 | 0.07 |
| Minimal agent + DCLab's 6 kinds of work | 2.50 | 2.75 | 2.12 | 2.88 | 4 of 8 | 0 | 52 | 1.9 | 58 | 0.02 |
| Current DCLab (lead agent) | 2.62 | 3.12 | 1.88 | 3.62 | 4 of 8 | 0 | 0 | 2.5 | 138 | 0.08 |

Empty or very short answers (scored as given): Model only: hab-04, hab-05; Minimal agent: hab-01, hab-04, hab-05, hab-08; Minimal agent + DCLab's 6 kinds of work: hab-01, hab-04.

**The harness inside this model** (current DCLab minus the minimal agent):

| Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted |
|---|---|---|---|---|
| 0.12 [-0.88, 1.25] | 0.75 [-0.38, 1.75] | -0.62 [-1.75, 0.38] | 0.62 [-0.25, 1.75] | -0.12 [-0.50, 0.25] |

**Where the lead agent's code stepped in** (all 8 conversations):

| what | times |
|---|---:|
| The table tool cannot ask the question | 11 |
| Limit on table questions per turn | 10 |
| The model's call was malformed | 4 |
| An option removed by a check | 3 |
| The turn's step budget ran out | 3 |
| A second small model overrules the kind of work | 3 |
| A build asked for without the person's yes | 3 |
| Another refusal (a build without a yes, a bad argument) | 2 |
| Work refused: the case is "not ready" | 1 |
| An outcome column the person never named | 1 |
| Advice refused: the words ask for data work | 1 |
| A table is needed first | 1 |

## HAB-006: arms on `nvidia/nemotron-3-ultra-550b-a55b:free`, judged by `nvidia/nemotron-3.5-super-vl:free`

| arm | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | accepted | must-not-do done | numbers nobody gave | turns | seconds | € |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model only | 3.25 | 3.88 | 3.75 | 3.88 | 8 of 8 | 0 | 61 | 1.0 | 105 | 0.00 |
| Minimal agent | 1.62 | 1.62 | 1.62 | 1.88 | 3 of 8 | 0 | 386 | 1.6 | 117 | 0.00 |
| Minimal agent + DCLab's 6 kinds of work | 2.00 | 2.00 | 1.88 | 3.00 | 2 of 8 | 1 | 12 | 2.0 | 70 | 0.00 |
| Current DCLab (lead agent) | 2.38 | 2.38 | 1.38 | 3.12 | 2 of 8 | 1 | 0 | 2.8 | 158 | 0.00 |

Empty or very short answers (scored as given): Minimal agent: hab-02, hab-04, hab-05, hab-08; Minimal agent + DCLab's 6 kinds of work: hab-01.

**The harness inside this model** (current DCLab minus the minimal agent):

| Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted |
|---|---|---|---|---|
| 0.75 [-0.38, 1.88] | 0.75 [-0.75, 1.88] | -0.25 [-1.38, 0.88] | 1.25 [0.25, 2.12] | -0.12 [-0.50, 0.25] |

**Where the lead agent's code stepped in** (all 8 conversations):

| what | times |
|---|---:|
| A build asked for without the person's yes | 7 |
| An option removed by a check | 4 |
| Another refusal (a build without a yes, a bad argument) | 3 |
| An outcome column the person never named | 2 |
| Limit on table questions per turn | 2 |
| An option kept, one sentence with a number removed | 1 |
| The turn's step budget ran out | 1 |
| A table is needed first | 1 |
| Work refused: the case is "not ready" | 1 |

## HAB-013: arms on `qwen/qwen3.8-flash`, played and not judged yet

No judge has scored these conversations. What code can count:

| arm | conversations | turns | seconds | numbers nobody gave | € |
|---|---:|---:|---:|---:|---:|
| Current DCLab (lead agent) | 3 | 2.3 | 109 | 0 | 0.02 |

Where the lead agent's code stepped in: A build asked for without the person's yes (3), An option kept, one sentence with a number removed (2), Another refusal (a build without a yes, a bad argument) (2), An outcome column the person never named (1), The model's call was malformed (1), Work refused: the case is "not ready" (1).

## The same cases on two models: `qwen/qwen3.8-flash` minus `anthropic/claude-sonnet-5.5`

| arm | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted |
|---|---|---|---|---|---|
| Model only | -0.88 [-1.88, 0.00] | -1.12 [-2.25, -0.12] | -1.25 [-2.50, -0.25] | -0.25 [-0.50, 0.00] | -0.25 [-0.50, 0.00] |
| Minimal agent | -0.88 [-2.00, 0.25] | -1.12 [-2.62, 0.25] | -1.00 [-2.38, 0.38] | -0.50 [-1.75, 0.75] | -0.25 [-0.62, 0.25] |
| Minimal agent + DCLab's 6 kinds of work | -0.38 [-1.50, 0.62] | -0.62 [-1.88, 0.50] | -0.12 [-1.00, 0.75] | -1.00 [-2.25, 0.12] | -0.50 [-0.88, -0.12] |
| Current DCLab (lead agent) | 0.00 [-0.38, 0.38] | 0.12 [-0.25, 0.50] | 0.62 [0.25, 0.88] | 0.00 [0.00, 0.00] | 0.00 [-0.38, 0.38] |

## The same cases on two models: `nvidia/nemotron-3.5-super-vl:free` minus `nvidia/nemotron-3-ultra-550b-a55b:free`

| arm | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted |
|---|---|---|---|---|---|
| Model only | -0.50 [-0.88, -0.12] | -0.25 [-0.75, 0.25] | -0.38 [-0.75, -0.12] | 0.25 [0.00, 0.62] | 0.00 [0.00, 0.00] |
| Minimal agent | 1.00 [0.12, 1.88] | 1.38 [0.38, 2.50] | 1.12 [0.00, 2.38] | 0.88 [0.12, 1.75] | 0.38 [0.12, 0.75] |
| Minimal agent + DCLab's 6 kinds of work | -0.12 [-0.62, 0.50] | 0.00 [-0.50, 0.62] | -0.38 [-1.12, 0.38] | 0.12 [-0.50, 1.00] | -0.12 [-0.50, 0.25] |
| Current DCLab (lead agent) | 0.25 [-0.25, 0.62] | 0.00 [-0.50, 0.50] | 0.25 [-0.13, 0.62] | 0.12 [0.00, 0.38] | 0.12 [-0.25, 0.50] |

## The same cases on two models: `nvidia/nemotron-3-ultra-550b-a55b:free` minus `qwen/qwen3.8-flash`

| arm | Understands the problem | Designs a solution | Makes a plan | Sticks to evidence | Accepted |
|---|---|---|---|---|---|
| Current DCLab (lead agent) | -1.00 [-1.38, -0.62] | -1.00 [-1.88, 0.00] | -1.00 [-1.62, 0.00] | -0.38 [-0.75, -0.12] | -0.50 [-0.88, -0.12] |

## Current DCLab before and after: stage-2 (`qwen/qwen3.8-flash`, judge `nvidia/nemotron-3.5-super-vl:free`)

Before: HAB-007 + HAB-008 + HAB-009 + HAB-010. After: HAB-014 + HAB-015 + HAB-022. Only the lead agent's conversations were run again; the other arms do not change with its code.

| measure | before | after | difference by case [95% interval] |
|---|---:|---:|---|
| Understands the problem | 3.38 | 4.00 | 0.62 [0.12, 1.12] |
| Designs a solution | 3.38 | 3.62 | 0.25 [-0.75, 1.25] |
| Makes a plan | 2.38 | 3.12 | 0.75 [-0.12, 1.75] |
| Sticks to evidence | 3.50 | 4.00 | 0.50 [0.12, 0.88] |
| Accepted | 6 of 8 | 7 of 8 | 0.12 [-0.25, 0.50] |
| Turns | 2.4 | 2.4 | |
| Seconds per conversation | 195 | 154 | |
| Numbers nobody gave | 0 | 0 | |

| where code stepped in | before | after |
|---|---:|---:|
| The model's call was malformed | 6 | 3 |
| A build asked for without the person's yes | 6 | 0 |
| A second small model overrules the kind of work | 3 | 1 |
| An outcome column the person never named | 1 | 3 |
| The table tool cannot ask the question | 2 | 2 |
| An option kept, one sentence with a number removed | 1 | 2 |
| Another refusal (a build without a yes, a bad argument) | 1 | 2 |
| Limit on table questions per turn | 0 | 3 |
| An option removed by a check | 1 | 1 |
| Work refused: the case is "not ready" | 1 | 0 |
| The turn's step budget ran out | 1 | 0 |

## Current DCLab before and after: stage-2 (`nvidia/nemotron-3.5-super-vl:free`, judge `qwen/qwen3.8-27b`)

Before: HAB-011 + HAB-012. After: HAB-018 + HAB-019. Only the lead agent's conversations were run again; the other arms do not change with its code.

| measure | before | after | difference by case [95% interval] |
|---|---:|---:|---|
| Understands the problem | 2.38 | 2.38 | 0.00 [-0.50, 0.50] |
| Designs a solution | 2.00 | 2.25 | 0.25 [-0.38, 0.75] |
| Makes a plan | 1.75 | 2.00 | 0.25 [-0.38, 0.88] |
| Sticks to evidence | 3.00 | 3.00 | 0.00 [-0.38, 0.38] |
| Accepted | 2 of 8 | 5 of 8 | 0.38 [0.12, 0.75] |
| Turns | 2.8 | 2.6 | |
| Seconds per conversation | 115 | 131 | |
| Numbers nobody gave | 0 | 0 | |

| where code stepped in | before | after |
|---|---:|---:|
| An outcome column the person never named | 1 | 4 |
| Another refusal (a build without a yes, a bad argument) | 1 | 1 |
| Advice refused: the words ask for data work | 2 | 0 |
| A build asked for before a project was started | 2 | 0 |
| The turn's step budget ran out | 2 | 0 |
| A second small model overrules the kind of work | 1 | 0 |
| Work refused: the case is "not ready" | 1 | 0 |
| The model's call was malformed | 1 | 0 |
| An option removed by a check | 1 | 0 |

## Current DCLab before and after: after-three-fixes (`qwen/qwen3.8-flash`, judge `nvidia/nemotron-3.5-super-vl:free`)

Before: HAB-002. After: HAB-007 + HAB-008 + HAB-009 + HAB-010. Only the lead agent's conversations were run again; the other arms do not change with its code.

| measure | before | after | difference by case [95% interval] |
|---|---:|---:|---|
| Understands the problem | 2.62 | 3.38 | 0.75 [0.37, 1.25] |
| Designs a solution | 3.12 | 3.38 | 0.25 [0.00, 0.50] |
| Makes a plan | 1.88 | 2.38 | 0.50 [-0.12, 1.25] |
| Sticks to evidence | 3.62 | 3.50 | -0.12 [-0.38, 0.00] |
| Accepted | 4 of 8 | 6 of 8 | 0.25 [0.00, 0.50] |
| Turns | 2.5 | 2.4 | |
| Seconds per conversation | 138 | 195 | |
| Numbers nobody gave | 0 | 0 | |

| where code stepped in | before | after |
|---|---:|---:|
| The table tool cannot ask the question | 11 | 2 |
| The model's call was malformed | 4 | 6 |
| Limit on table questions per turn | 10 | 0 |
| A build asked for without the person's yes | 3 | 6 |
| A second small model overrules the kind of work | 3 | 3 |
| The turn's step budget ran out | 3 | 1 |
| An option removed by a check | 3 | 1 |
| Another refusal (a build without a yes, a bad argument) | 2 | 1 |
| An outcome column the person never named | 1 | 1 |
| Work refused: the case is "not ready" | 1 | 1 |
| A table is needed first | 1 | 0 |
| An option kept, one sentence with a number removed | 0 | 1 |
| Advice refused: the words ask for data work | 1 | 0 |

## Two judges on the same conversations (HAB-002)

`qwen/qwen3.8-27b` and `nvidia/nemotron-3.5-super-vl:free` agree on accepted or not for 25 of 32 conversations. Their 0 to 4 scores differ by 0.39 on average; 99% of scores are within one point.

## Two judges on the same conversations (HAB-006)

`qwen/qwen3.8-27b` and `nvidia/nemotron-3.5-super-vl:free` agree on accepted or not for 28 of 32 conversations. Their 0 to 4 scores differ by 0.29 on average; 98% of scores are within one point.

## Claude beside Qwen

From `evidence/bench/models/QWEN_VS_CLAUDE_20261115T105428Z_36319.json` (rebuild it with `python -m dclab_rnd.bench models`). Each cell is the mean and the number of cases it stands on. "thin" marks a side with fewer than five cases: read it as a hint.

### These eight cases (the Claude run stopped early: 2 to 3 cases, not judged)

| arm | measure | Claude | Qwen | better | |
|---|---|---:|---:|---|---|
| Model only | Seconds per conversation | 26.50 (n 3) | 100.84 (n 8) | Claude | thin |
| Model only | Turns | 1.00 (n 3) | 1.00 (n 8) | same | thin |
| Model only | Numbers nobody gave | 0.00 (n 3) | 0.12 (n 8) | Claude | thin |
| Model only | Euros per conversation | 0.03 (n 3) | 0.00 (n 8) | Qwen | thin |
| Minimal agent | Seconds per conversation | 31.38 (n 3) | 166.51 (n 8) | Claude | thin |
| Minimal agent | Turns | 1.33 (n 3) | 1.75 (n 8) | Claude | thin |
| Minimal agent | Numbers nobody gave | 9.67 (n 3) | 0.50 (n 8) | Qwen | thin |
| Minimal agent | Euros per conversation | 0.08 (n 3) | 0.01 (n 8) | Qwen | thin |
| Minimal agent + DCLab's 6 kinds of work | Seconds per conversation | 28.07 (n 2) | 57.87 (n 8) | Claude | thin |
| Minimal agent + DCLab's 6 kinds of work | Turns | 2.00 (n 2) | 1.88 (n 8) | Qwen | thin |
| Minimal agent + DCLab's 6 kinds of work | Numbers nobody gave | 2.00 (n 2) | 6.50 (n 8) | Claude | thin |
| Minimal agent + DCLab's 6 kinds of work | Euros per conversation | 0.10 (n 2) | 0.00 (n 8) | Qwen | thin |
| Current DCLab (lead agent) | Seconds per conversation | 61.27 (n 2) | 137.58 (n 8) | Claude | thin |
| Current DCLab (lead agent) | Turns | 3.00 (n 2) | 2.50 (n 8) | Qwen | thin |
| Current DCLab (lead agent) | Numbers nobody gave | 0.00 (n 2) | 0.00 (n 8) | same | thin |
| Current DCLab (lead agent) | Euros per conversation | 0.33 (n 2) | 0.01 (n 8) | Qwen | thin |

### The conversation exam, practice people (Claude judged by Opus, Qwen by Qwen)

| arm | measure | Claude | Qwen | better | |
|---|---|---:|---:|---|---|
| Minimal agent + DCLab's 6 kinds of work | Understanding (0 to 1) | 0.59 (n 41) | 0.46 (n 2) | Claude | thin |
| Minimal agent + DCLab's 6 kinds of work | Option fit (1 to 5) | 3.29 (n 17) | 3.00 (n 2) | Claude | thin |
| Minimal agent + DCLab's 6 kinds of work | Right kind of work | 0.76 (n 41) | 0.50 (n 2) | Claude | thin |
| Minimal agent + DCLab's 6 kinds of work | Traps handled | 0.58 (n 19) | 0.00 (n 1) | Claude | thin |
| Minimal agent + DCLab's 6 kinds of work | Numbers nobody gave | 1.29 (n 41) | 3.00 (n 2) | Claude | thin |
| Minimal agent + DCLab's 6 kinds of work | Questions asked | 1.22 (n 41) | 1.50 (n 2) | Claude | thin |
| Minimal agent + DCLab's 6 kinds of work | Seconds per turn | 13.15 (n 41) | 12.42 (n 2) | Qwen | thin |
| Minimal agent + DCLab's 6 kinds of work | Euros per conversation | 0.02 (n 41) | 0.00 (n 2) | Qwen | thin |
| Current DCLab (lead agent) | Understanding (0 to 1) | 0.90 (n 41) | 0.83 (n 6) | Claude |  |
| Current DCLab (lead agent) | Option fit (1 to 5) | 4.30 (n 27) | 4.00 (n 2) | Claude | thin |
| Current DCLab (lead agent) | Right kind of work | 0.90 (n 41) | 1.00 (n 6) | Qwen |  |
| Current DCLab (lead agent) | Traps handled | 0.89 (n 19) | 0.83 (n 6) | Claude |  |
| Current DCLab (lead agent) | Numbers nobody gave | 0.00 (n 41) | 0.00 (n 6) | same |  |
| Current DCLab (lead agent) | Questions asked | 3.15 (n 41) | 1.33 (n 6) | Qwen |  |
| Current DCLab (lead agent) | Seconds per turn | 22.67 (n 41) | 34.85 (n 6) | Claude |  |
| Current DCLab (lead agent) | Euros per conversation | 0.09 (n 41) | 0.01 (n 6) | Qwen |  |

### Home conversations, 25 written people on both models

| arm | measure | Claude | Qwen | better | |
|---|---|---:|---:|---|---|
| Home (rules first, with the model) | Right kind of help chosen | 1.00 (n 25) | 0.92 (n 25) | Claude |  |
| Home (rules first, with the model) | Right kind of help offered | 0.88 (n 25) | 0.60 (n 25) | Claude |  |
| Home (rules first, with the model) | Turns before options | 1.76 (n 25) | 1.40 (n 25) | Qwen |  |

## What the earlier exams on Claude showed (context)

Kept dev runs read again (HAB-000, 11 runs on Claude Sonnet 5.5): the lead agent lost the right kind of work to the model alone on 8 conversations and won it on 12. First reasons of the losses: The simulated person picked a wrong option (no refusal recorded) (3), Work refused: the case is "not ready" (2), The model chose the wrong kind of work (no refusal recorded) (1), The run broke (1), Work done, no kind of work recorded (1).

| sealed exam | model | judge | arm | people | understanding (0 to 1) | option fit (1 to 5) | right kind of work reached | numbers nobody gave | s/turn |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| CEX-013 | claude-sonnet-5-5 (the product's) | claude-opus-5-5 | the model alone | 40 | 0.62 | 2.79 | 0.90 | 618 | 26.2 |
| CEX-013 | claude-sonnet-5-5 (the product's) | claude-opus-5-5 | lead agent | 40 | 0.85 | 4.25 | 0.90 | 0 | 33.5 |
| CEX-017 | claude-sonnet-5-5 (the product's) | claude-opus-5-5 | the model alone | 19 | 0.82 | 3.44 | 1.00 | 139 | 23.6 |
| CEX-017 | claude-sonnet-5-5 (the product's) | claude-opus-5-5 | lead agent | 19 | 0.93 | 4.42 | 0.79 | 0 | 22.1 |
| CEX-032 | claude-sonnet-5-5 (the product's) | claude-opus-5-5 | the model alone | 12 | 0.84 | 3.67 | 0.83 | 198 | 30.0 |
| CEX-032 | claude-sonnet-5-5 (the product's) | claude-opus-5-5 | lead agent | 12 | 0.90 | 3.80 | 0.75 | 0 | 32.3 |

## What weakens this benchmark

- eight written cases, each once: differences inside the intervals are noise
- one judge model scored every run; it is from the same family as one of the two arm models
- the cases and their answer key were written by the runner's author
- an empty or cut answer of a baseline arm is scored as given (the runner's limits of five model calls a turn and its token limit)
- the Claude-era exams used other people, a simulated person and another judge: they are context, not a pairing
