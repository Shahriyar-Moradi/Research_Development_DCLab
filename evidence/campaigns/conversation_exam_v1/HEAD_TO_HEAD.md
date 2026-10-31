# The lead agent against Claude alone, person by person

Rebuilt by `python -m dclab_rnd.agent_eval.conversation_exam.headtohead`; do not edit by hand. Both arms met the same
people; 'better/same/worse' counts people for the lead agent; the interval is the paired difference's 90% bootstrap
range; 'clear' means the whole range is on one side of zero.

## CEX-013 (sealed, 40 people, run once)

| measure | by | lead better | same | lead worse | lead agent | Claude alone | difference | 90% range | winner |
|---|---|---|---|---|---|---|---|---|---|
| understanding | judge | 18 | 18 | 4 | 0.85 | 0.62 | +0.23 | +0.14 to +0.33 | lead agent |
| option fit | judge | 10 | 0 | 0 | 4.20 | 2.60 | +1.60 | +1.20 to +2.00 | lead agent |
| right route reached | code | 3 | 34 | 3 | 0.90 | 0.90 | +0.00 | -0.10 to +0.10 | no clear difference |
| trap handled | judge | 7 | 12 | 0 | 1.00 | 0.63 | +0.37 | +0.21 to +0.53 | lead agent |
| useful questions (share) | judge | 4 | 15 | 13 | 0.87 | 0.95 | -0.08 | -0.15 to -0.00 | Claude alone |
| questions already answered | judge | 4 | 23 | 13 | 0.42 | 0.10 | +0.33 | +0.12 to +0.55 | Claude alone |
| invented numbers and broken rules | code | 27 | 13 | 0 | 0.00 | 15.45 | -15.45 | -29.93 to -5.58 | lead agent |
| seconds per reply | code | 7 | 0 | 33 | 33.54 | 26.19 | +7.35 | +1.33 to +12.57 | Claude alone |
| cost per person (EUR) | code | 6 | 0 | 34 | 0.20 | 0.12 | +0.09 | +0.00 to +0.15 | Claude alone |

| trap | lead agent | Claude alone |
|---|---|---|
| clear_request | 6 of 6 | 6 of 6 |
| missing_column | 3 of 3 | 3 of 3 |
| no_outcome | 1 of 1 | 1 of 1 |
| real_prediction | 2 of 2 | 0 of 2 |
| report_not_model | 2 of 2 | 1 of 2 |
| rows_request | 1 of 1 | 1 of 1 |
| silent_drop | 1 of 1 | 0 of 1 |
| vague_goal | 3 of 3 | 0 of 3 |

## CEX-009 (practice, 41 people)

| measure | by | lead better | same | lead worse | lead agent | Claude alone | difference | 90% range | winner |
|---|---|---|---|---|---|---|---|---|---|
| understanding | judge | 25 | 16 | 0 | 0.88 | 0.52 | +0.36 | +0.28 to +0.44 | lead agent |
| option fit | judge | 11 | 0 | 2 | 4.08 | 3.08 | +1.00 | +0.54 to +1.38 | lead agent |
| right route reached | code | 3 | 33 | 5 | 0.73 | 0.78 | -0.05 | -0.17 to +0.07 | no clear difference |
| trap handled | judge | 7 | 12 | 0 | 0.95 | 0.58 | +0.37 | +0.21 to +0.53 | lead agent |
| useful questions (share) | judge | 4 | 15 | 4 | 0.91 | 0.91 | +0.00 | -0.08 to +0.08 | no clear difference |
| questions already answered | judge | 3 | 32 | 6 | 0.17 | 0.10 | +0.07 | -0.05 to +0.20 | no clear difference |
| invented numbers and broken rules | code | 8 | 33 | 0 | 0.00 | 0.80 | -0.80 | -1.39 to -0.29 | lead agent |
| seconds per reply | code | 2 | 0 | 39 | 22.73 | 13.98 | +8.76 | +6.06 to +11.10 | Claude alone |
| cost per person (EUR) | code | 0 | 0 | 41 | 0.10 | 0.02 | +0.08 | +0.07 to +0.10 | Claude alone |

| trap | lead agent | Claude alone |
|---|---|---|
| clear_request | 6 of 6 | 6 of 6 |
| missing_column | 3 of 3 | 2 of 3 |
| no_outcome | 1 of 1 | 1 of 1 |
| real_prediction | 1 of 2 | 0 of 2 |
| report_not_model | 2 of 2 | 1 of 2 |
| rows_request | 1 of 1 | 1 of 1 |
| silent_drop | 1 of 1 | 0 of 1 |
| vague_goal | 3 of 3 | 0 of 3 |

## Checked by people

The judge agreed with the owner on 0.66 of 10 sealed papers (the bar was 0.70), so the measures the judge marks (understanding, option fit, traps, useful and repeated
questions) are not yet trusted; the measures counted by code (route reached, invented numbers, seconds, cost) do
not depend on the judge.
