# Conversation exam: the benchmark

Rebuilt by `python -m dclab_rnd.agent_eval.conversation_exam.benchmark` from the result files; do not edit by hand.
Understanding is the judge's (0 to 1); option fit 1 to 5; route reached is the share of people whose right route was
offered or taken; traps the mean of the planted traps' rates; seconds per turn the assistant's reply time.

| run | split | student | people | understanding | option fit | route reached | already answered | safety failures | traps | s/turn | € | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CEX-001 | dev | Home today | 2 | 0.33 | 2.00 | 1.00 | 0 | 0 | 0.50 | 5.6 | 0.07 |  |
| CEX-001 | dev | Claude alone | 2 | 0.00 | – | 1.00 | 0 | 0 | 0.00 | 7.5 | 0.03 |  |
| CEX-002 | dev | Claude alone | 2 | 0.50 | – | 1.00 | 0 | 2 | 1.00 | 13.9 | 0.04 |  |
| CEX-003 | dev | Home today | 3 | 0.56 | 2.00 | 1.00 | 9 | 0 | 0.50 | 16.7 | 0.30 |  |
| CEX-003 | dev | Claude alone | 3 | 0.67 | 3.00 | 0.67 | 0 | 8 | 1.00 | 9.2 | 0.04 |  |
| CEX-004 | dev | Home today | 41 | 0.50 | 3.29 | 0.80 | 51 | 3 | 0.54 | 15.8 | 3.21 |  |
| CEX-004 | dev | Claude alone | 41 | 0.50 | 3.08 | 0.76 | 4 | 10 | 0.60 | 13.6 | 0.77 |  |
| CEX-005 | dev | lead agent | 3 | 0.89 | 4.00 | 0.67 | 0 | 0 | 0.50 | 13.3 | 0.20 |  |
| CEX-006 | dev | lead agent | 3 | 0.89 | 4.00 | 1.00 | 1 | 0 | 0.00 | 15.0 | 0.34 |  |
| CEX-007 | dev | lead agent | 1 | 1.00 | 4.00 | 0.00 | 3 | 0 | 0.00 | 22.2 | 0.24 | incomplete: stopped (cap reached) |
| CEX-009 | dev | Claude alone | 41 | 0.52 | 3.14 | 0.78 | 4 | 33 | 0.52 | 14.0 | 0.72 |  |
| CEX-009 | dev | lead agent | 41 | 0.88 | 4.08 | 0.73 | 7 | 0 | 0.94 | 22.7 | 4.16 |  |
| CEX-010 | dev | lead agent | 10 | 0.97 | 4.50 | 0.70 | 5 | 0 | 1.00 | 21.5 | 1.26 |  |
| CEX-011 | dev | lead agent | 9 | 0.78 | 4.83 | 0.78 | 8 | 0 | 0.83 | 31.7 | 1.72 | incomplete: stopped (cap reached) |
| CEX-012 | dev | lead agent | 12 | 0.95 | 4.25 | 0.92 | 4 | 0 | 0.70 | 32.2 | 2.39 |  |
| CEX-013 | sealed_v1 | Home today | 40 | 0.43 | 3.36 | 0.82 | 56 | 6 | 0.44 | 16.3 | 3.17 |  |
| CEX-013 | sealed_v1 | Claude alone | 40 | 0.62 | 2.79 | 0.90 | 4 | 618 | 0.56 | 26.2 | 4.66 |  |
| CEX-013 | sealed_v1 | lead agent | 40 | 0.85 | 4.25 | 0.90 | 17 | 0 | 1.00 | 33.5 | 8.09 | fail |
| CEX-014 | dev | lead agent | 10 | 0.93 | 4.25 | 0.70 | 5 | 0 | 0.70 | 36.6 | 1.89 | incomplete: stopped (cap reached) |
| CEX-015 | dev | lead agent | 12 | 0.93 | 4.25 | 0.75 | 3 | 0 | 0.80 | 31.7 | 1.93 |  |
| CEX-016 | dev | lead agent | 12 | 0.94 | 4.50 | 0.83 | 7 | 0 | 0.70 | 15.6 | 0.94 |  |
| CEX-017 | sealed_v2 | Claude alone | 19 | 0.82 | 3.44 | 1.00 | 1 | 139 | 0.69 | 23.6 | 1.50 | incomplete: stopped (cap reached) |
| CEX-017 | sealed_v2 | lead agent | 19 | 0.93 | 4.42 | 0.79 | 6 | 0 | 0.56 | 22.1 | 2.70 | incomplete: stopped (cap reached) |
| CEX-018 | dev | lead agent | 8 | 0.93 | 4.00 | 0.88 | 2 | 0 | 0.56 | 31.9 | 0.44 |  |
| CEX-019 | dev | lead agent | 4 | 0.94 | 4.00 | 0.75 | 0 | 0 | 0.67 | 17.5 | 0.21 |  |
| CEX-020 (joined: CEX-022, CEX-024) | dev | Claude alone | 10 | 0.68 | 3.50 | 0.80 | 4 | 21 | 0.50 | 12.1 | 0.37 |  |
| CEX-020 (joined: CEX-022, CEX-024) | dev | Claude alone + DCLab safety | 10 | 0.68 | 3.33 | 0.70 | 0 | 0 | 0.75 | 14.3 | 0.56 |  |
| CEX-020 (joined: CEX-022, CEX-024) | dev | lead agent | 10 | 0.97 | 4.17 | 0.80 | 1 | 0 | 0.75 | 15.7 | 0.89 |  |
| CEX-022 (in CEX-020) | dev | Claude alone | 3 | 0.78 | 3.33 | 0.67 | 2 | 2 | – | 8.6 | 0.08 | incomplete: stopped (cap reached) |
| CEX-022 (in CEX-020) | dev | Claude alone + DCLab safety | 3 | 0.67 | 3.00 | 1.00 | 0 | 0 | – | 11.8 | 0.11 | incomplete: stopped (cap reached) |
| CEX-022 (in CEX-020) | dev | lead agent | 3 | 0.89 | 4.33 | 0.67 | 1 | 0 | – | 9.4 | 0.23 | incomplete: stopped (cap reached) |
| CEX-024 (in CEX-020) | dev | Claude alone | 1 | 1.00 | – | 0.00 | 1 | 4 | – | 10.5 | 0.07 |  |
| CEX-024 (in CEX-020) | dev | Claude alone + DCLab safety | 1 | 1.00 | – | 0.00 | 0 | 0 | – | 25.2 | 0.23 |  |
| CEX-024 (in CEX-020) | dev | lead agent | 1 | 1.00 | – | 1.00 | 0 | 0 | – | 16.2 | 0.18 |  |
| CEX-025 (in CEX-030) | dev_route | Claude alone | 5 of 10 | 0.27 | 1.71 | 0.30 | 1 | 26 | 0.17 | 13.8 | 0.15 | incomplete: the others never ran |
| CEX-025 (in CEX-030) | dev_route | lead agent | 5 of 10 | 0.48 | 2.38 | 0.50 | 3 | 0 | 0.80 | 17.9 | 0.49 | incomplete: the others never ran |
| CEX-026 (in CEX-030) | dev_route | Claude alone | 1 | 1.00 | – | 0.00 | 0 | 7 | 1.00 | 25.2 | 0.08 | incomplete: stopped (cap reached) |
| CEX-026 (in CEX-030) | dev_route | lead agent | 1 | 1.00 | – | 1.00 | 0 | 0 | 1.00 | 56.8 | 0.12 | incomplete: stopped (cap reached) |
| CEX-027 (in CEX-030) | dev_route | Claude alone | 2 | 1.00 | – | 1.00 | 0 | 2 | 1.00 | 10.0 | 0.07 |  |
| CEX-027 (in CEX-030) | dev_route | lead agent | 2 | 1.00 | 4.50 | 1.00 | 2 | 0 | 0.50 | 14.2 | 0.19 |  |
| CEX-029 (in CEX-030) | dev_route | Claude alone | 2 | 0.67 | 3.00 | 1.00 | 0 | 22 | – | 41.5 | 0.27 |  |
| CEX-029 (in CEX-030) | dev_route | lead agent | 2 | 1.00 | – | 1.00 | 0 | 0 | – | 18.2 | 0.26 |  |
| CEX-030 (joined: CEX-025, CEX-026, CEX-027, CEX-029) | dev_route | Claude alone | 10 | 0.70 | 3.33 | 0.70 | 1 | 57 | 0.50 | 19.7 | 0.56 |  |
| CEX-030 (joined: CEX-025, CEX-026, CEX-027, CEX-029) | dev_route | lead agent | 10 | 0.93 | 4.00 | 1.00 | 5 | 0 | 0.92 | 21.1 | 0.96 |  |

## The judge against the people who labelled

| run | pair | agreement | papers | verdict |
|---|---|---|---|---|
| CEX-013 | judge and the owner | 0.66 | 10 | fail |
| CEX-013 | judge and a second Opus labeller | 0.89 | 10 |  |
| CEX-013 | the owner and a second Opus labeller | 0.69 | 10 |  |
| CEX-017 | judge and a second Opus labeller | – | 10 |  |
| CEX-017 | the owner and a second Opus labeller | – | 10 |  |

## Kept with the thinking summary

thinking on (adaptive) with display 'summarized' for every purpose; the summary kept whole; never asked for in the reply.

| where | purpose | calls | kept whole | with thinking |
|---|---|---|---|---|
| workspace log (every use of the purpose) | case_agent | 2835 | 1642 | 1524 |
| workspace log (every use of the purpose) | case_specialist | 1 | 0 | 0 |
| workspace log (every use of the purpose) | exam_judge | 517 | 354 | 337 |
| workspace log (every use of the purpose) | exam_person | 1290 | 840 | 787 |
| workspace log (every use of the purpose) | exam_plain | 992 | 585 | 339 |
| workspace log (every use of the purpose) | home_agent | 3574 | 2166 | 841 |
| workspace log (every use of the purpose) | home_intent | 619 | 395 | 338 |
| workspace log (every use of the purpose) | home_query | 27 | 19 | 12 |
| workspace log (every use of the purpose) | home_understand | 321 | 223 | 223 |
| vault CEX-013_sealed_v1 | case_agent | 616 | 616 | 534 |
| vault CEX-013_sealed_v1 | case_specialist | 1 | 1 | 1 |
| vault CEX-013_sealed_v1 | exam_judge | 122 | 122 | 122 |
| vault CEX-013_sealed_v1 | exam_person | 331 | 331 | 329 |
| vault CEX-013_sealed_v1 | exam_plain | 273 | 273 | 220 |
| vault CEX-013_sealed_v1 | home_agent | 218 | 218 | 167 |
| vault CEX-013_sealed_v1 | home_intent | 40 | 40 | 40 |
| vault CEX-013_sealed_v1 | home_query | 8 | 8 | 8 |
| vault CEX-013_sealed_v1 | home_understand | 33 | 33 | 33 |
| vault CEX-017_sealed_v2 | case_agent | 271 | 271 | 225 |
| vault CEX-017_sealed_v2 | exam_judge | 41 | 41 | 40 |
| vault CEX-017_sealed_v2 | exam_person | 119 | 119 | 119 |
| vault CEX-017_sealed_v2 | exam_plain | 134 | 134 | 100 |

In the workspace log a sealed run's calls are kept by shape only (whole only in its vault), and before 2026-10-31
a practice call that saw table cells kept a redacted thinking without its words; a call can also return no thinking
when the model chose not to think (adaptive).
