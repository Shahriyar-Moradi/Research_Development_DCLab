# Does the bounded Lead loop help? A live check (2026-11-19)

The owner asked whether another agent's bounded Lead choice loop (A1-A2, 7f7800ec and 504e313f; switch
`DCLAB_LEAD_BOUNDED_CHOICE`, off by default) helps at all. Its own evaluation (A7) claimed 18 of 18 correct next actions
against 3 of 18; a review found those scores were written into the harness, not measured.

Live, on harness_ablation's dev people (8), the lead agent arm only, model `qwen/qwen3.8-flash`, judge
`nvidia/nemotron-3.5-super-vl:free`, one repeat each:

| Run | Loop | Understanding | Planning | Accepted | Spent |
|---|---|---|---|---|---|
| HAB-065 | off (today's Lead) | 3.625 | 3.75 | 8 of 8 | EUR 0.105 |
| HAB-066 | on, as committed | 0.375 | 0.375 | 1 of 8 | EUR 0.067 |
| HAB-067 | on, after the review's fixes (f119d419) | 1.75 | 1.375 | 4 of 8 | EUR 0.036 |

Paired with the loop off, the fixed loop's planning is lower on 6 of 8 people and higher on none (mean -2.4); its
understanding is lower on 5 and higher on none (mean -1.9).

Why: as committed, most replies the person saw were "Could not validate the proposed next action" or "Completed
bounded investigation step." After the fixes, the loop still ends conversations early (34 model calls against 113) and
its tool calls often fail on their arguments.

**Conclusion: the loop does not help; it hurts. Keep the switch off.** Eight people, one repeat: the direction is clear
(no person better), the size is not precise. Every call kept its thinking (113, 55, 34 of each run's calls).
