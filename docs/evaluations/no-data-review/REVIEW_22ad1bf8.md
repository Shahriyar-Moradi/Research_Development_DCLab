# Review of commit `22ad1bf8`: No-Data Solution Design and AI Harness

**Reviewed:** 2026-11-15. **Verdict: not ready.** 8 high, 7 medium and 4 low findings.
**How:** an independent reviewer read the commit, ran its tests on a clean checkout, and ran its own probes with no live
model. I then re-checked the main claims myself (marked "re-checked"). The web UI part of the commit was not reviewed.

## Status after the fixes (2026-11-15, same day)

18 of the 19 findings are fixed and re-checked with the reviewer's own inputs. 1 is left open (see the last table).
The 21 affected test files pass (616 tests). No live model was called for the fixes, so the harness's model roles are
not re-measured live.

| Finding | Status | How it was re-checked |
|---|---|---|
| H1 gates fail on a clean checkout | Fixed by committing the harness-ablation package with fresh `make agent-eval` and `make harness-eval` results | the prompt, harness, bench and devlog gate tests pass |
| H2 route forced to prediction as the person's choice | Fixed: no word pattern sets the route | "no data scientist… weekly report" → route `reporting`, read by the rules; no blueprint |
| H3 classification blueprint for any no-data person | Fixed: a blueprint only when the words name something to predict; otherwise one plain message | routes, complaints, A/B test, explain, dashboard → no blueprint |
| H4 invented numbers as facts | Fixed: only the person's numbers are sources; the rest is marked "assumed by DCLab" | "95% uptime" is no longer a churn rate; numbers the model makes up are ignored; no made-up citation id |
| H5 gates cannot fail | Fixed: all five are computed; the mode says AI only when a model's answer was used | a made-up number or "production-ready" fails gate 2; broken replies → "DCLab default", not "AI inferred" |
| H6 evaluation is not a measurement | Fixed: every arm scored on what it returned; gate v2 can fail; live runs follow rule 9 | the rewritten evaluation found the number reader wrong on 5 of 7 cases; that is fixed too |
| H7 thinking switched off | Fixed: never switched off or lowered; a cut answer gets more room | a test through the real gateway checks the requests and the kept thinking |
| H8 hidden live call | Fixed: with no gateway given the rules' plan is used | a test fails if the process's gateway is taken |
| M1 tool limits | Fixed: refused with a table, once a turn, inside the case's cap | tested |
| M2 Home stops listening | Fixed: open questions stay open; a new card only for a number or a request; the latest number wins | tested |
| M3 detector | Fixed | 8 wrong matches and 8 misses from the review, all tested |
| M4 specialists' checks bypassed | Fixed: the bypass ends when a real table is shared | tested |
| M5 two splits on the card | Fixed: one split; "rules' plan" when no model debated | tested |
| M6 prompts in code | Fixed: five prompt files, inside the prompt gate | tested |
| M7 tests pass with the feature broken | Fixed: failure paths are tested (14 and 9 tests; were 5 and 4) | — |
| L1 API accepts bad numbers | Fixed: 422 | tested |
| L3 invented columns shown as the person's file | Fixed | — |
| L4 every suspect forbidden | Fixed: only the ones marked exclude | — |
| L2 no ledger record for the no-data route | **Open**: needs a new decision point in `dclab_core` (its own package) | — |

Also found and fixed while repairing: the commit had broken 3 older tests in `tests/test_draft_flow.py` and 1 in
`tests/test_case_agent.py`.

Still open, said plainly:
- One harness pass is a dozen model calls inside the person's turn (7 to 17 minutes in the kept log). It should run as a job.
- Each blueprint card stores a full copy of the solution (about 27 KB).
- "Route support tickets to the right team" is refused as not a prediction because of the word "route".
- The fix adds a second word pattern ("is this a prediction?"). It errs toward asking, but it is still a pattern in front of the model.

## The three things that matter most

1. **Almost every no-data person gets the same answer: a yes/no prediction model.** Re-checked with three different
   problems (plan delivery routes; find why complaints rose; design an A/B test). All three came back as family `binary`,
   target `outcome_label`, metric ROC-AUC. None of the three is a prediction problem.
2. **Invented numbers are shown as facts.** When the person gives no number, the code fills in 10,000 customers, an 8%
   rate, a cost of 15, a value of 180, and prints "break-even precision 23.8%" with no word "assumed".
3. **The "5 gates" and the evaluation cannot fail.** Several results are written as constants in the code, so they always
   say "passed".

## High

| # | What is wrong | Example (what was run → what came out) | Smallest fix |
|---|---|---|---|
| H1 | The commit fails 3 tests on a clean checkout | `test_prompt_gate` (the lead agent's prompt and tools changed, no new agent-eval result committed); `test_bench_versions` and `test_devlog` (they point at the harness-ablation files, which are not committed). Part of this is my lines being swept into the commit | Commit the harness-ablation package with `AEV-115` and `HRN-123`, or remove its lines and rerun `make agent-eval` and `make sft-dev` |
| H2 | A word pattern sets the route to "prediction" and records it as **the person's own confirmed choice** (`draft/chat.py:1749`, `:1993`) | "There is no data scientist on our team, so I need a simple weekly report of sales by region" → route `prediction`, source `user`, confidence 1.0, and a model blueprint. Re-checked: the pattern matches this sentence | Do not set the route here. If a default is wanted, record it as the rules' guess, with the matched words |
| H3 | A no-data person with a non-prediction problem gets a classification blueprint (`draft/no_data.py:788-806`, `chat.py:1760-1766`) | Re-checked: routes, complaints and A/B test all → `binary`. On the feature's own 12 test cases, 7 get the wrong family, and nothing checks it | Post a blueprint only for a prediction route with a family found in the person's words. Add `binary` to the tuple at line 791. Check the family in the evaluation |
| H4 | Invented numbers pass as facts, and pass the number check (`no_data.py:164-188`, `:237-255`) | "We operate at 95% uptime" → read as a 95% churn rate (it finds "rate" inside "operate"). The model can also pass its own numbers as arguments and they are marked "stated". Citations always fall back to two fixed ids; one of them does not exist | Keep assumed numbers out of the "said" list; print "assumed" beside each; require the person's words for every number |
| H5 | The 5 gates cannot fail (`draft/no_data_harness.py:1128-1160`) | Re-checked: gates 2 and 4 are `"passed": True` written in code. With every model reply broken, the card still said "Gates passed: 5/5" and labelled code defaults "AI inferred" | Compute each gate from a real check. With no valid model reply, say "DCLab default", not "AI inferred" |
| H6 | The evaluation is not a measurement (`agent_eval/no_data/__init__.py:167-168`) | Re-checked: the plain arm's scores are the constants 0.50 and 0.40. Its fallback text holds invented numbers, so it is built to lose. The live runs on 2026-11-13 (170 requests, about EUR 0.21) went to `data/workspace`, with no run folder, no archive and no stored result | Score every arm on its real replies with one scorer. Follow rule 9 for live runs |
| H7 | The harness turns the model's thinking **off** on a retry, and lowers your thinking effort to "low" (`no_data_harness.py:311`, `:384`, `:396`) | Re-checked: the three lines are there. The kept log shows 13 requests sent with thinking off. 46 exchanges were kept without their words. This breaks your rule that every call keeps its thinking | Remove both overrides; raise the token limit instead. Add a test that the thinking is stored |
| H8 | The "no model" path still calls a live model | With the network blocked, `solve_without_data(problem)` tried to reach OpenRouter once (the task planner takes the process's gateway). Its cost is not counted in the case's cap | Pass an explicit "no model" switch to the planner call |

## Medium

| # | What is wrong | Example | Smallest fix |
|---|---|---|---|
| M1 | The new lead-agent tool has no table check, no per-turn limit, and ignores the case's money cap | It ran three times in a row on a case already over its cap, 12 model calls each. One pass took 7 to 17 minutes in the live log | Refuse when a table is shared; one call a turn; refuse when the cap is used; run it as a job |
| M2 | After a blueprint, Home stops listening | "thanks, what does AUC mean?" → another blueprint card. A correction ("we have 50,000 customers, not 20,000") is ignored: the first number wins | Rebuild only when asked; read the latest number |
| M3 | The no-data detector matches the wrong sentences | Matches "I have no data quality issues". Misses "we lack historical data" (re-checked) and several Persian sentences | Match only a stated lack of data |
| M4 | One blueprint switches off the specialists' checks for the rest of the case, even after a real table arrives | The leakage auditor's trigger is skipped | Apply the bypass only while there is no table |
| M5 | The card shows two validation splits that contradict each other | "Entity-grouped + time-ordered" and, below, "stratified" for the same churn problem. The second is the leak-prone one | Derive both from one rule |
| M6 | Prompts are written as strings in code, outside the prompt gate; the changed lead-agent prompt has no live measurement | Six places in `no_data_harness.py` | Move them to prompt files; add a no-data case to the harness suite |
| M7 | The tests would pass with the feature broken | They assert on the constants of H5 and H6 and use fake clients that accept anything | Test the failure paths: broken reply, cap reached, table shared, non-prediction request |

## Low

- The API route accepts bad numbers (`"abc"` → error 500; a negative cost → a negative break-even).
- No ledger record for the no-data route or the family chosen.
- The invented columns are handed to the problem analyst as if they were the person's file.
- Every leak suspect is forbidden, even one the model marked "keep".

## Does this commit make DCLab bind problems to a type earlier? Yes.

The path: a word pattern → the route is set to prediction as the person's choice → the family defaults to `binary` → a
fixed table, metric and model template. The lead agent's prompt now also says to call this tool "right away" and not to
wait for options. This is the same early binding the harness evaluation is investigating, added in front of the model.

## What is sound

- The tool is registered once, as a write, behind the case guard.
- No synthetic data is made without the person's answer.
- Build still refuses a draft with no table.
- No key, token or prompt text leaks into the solution or the report.
- The economics formulas are consistent.
- No prompt asks the model to write its reasoning in the reply.

## Suggested order of repair

1. H1 (the repository's gates fail today).
2. H7 (thinking switched off: your standing rule).
3. H2 and H3 together (stop forcing the route; stop the default family).
4. H4 (assumed numbers labelled).
5. H5 and H6 (real gates, a real measurement).
6. The rest.
