# Tool format and tool count trial (local Qwen): one-page plan

Status: plan only (2026-10-25). Nothing built or run. The owner decides whether to run it.

## The question

A slide from Hugging Face's multi-harness RL guide showed a small model (LFM2.5-2.6B) solving 62% of data tasks in
Mini-SWE-Agent (plain-text bash actions) and 33% in Claude Code (structured tool calls, a long prompt, many tools).
Does the same effect exist inside DCLab for a small local model?

- **A. Action format:** JSON tool calls (today) vs a plain-text action that the harness parses.
- **B. Tool count:** the intern's 28 tools vs about 8.

## A correction: the dev journeys cannot test this

The dev journeys (`make journeys`) walk the coordinator with rules and a reader model. No model chooses among tools
there, and there are only 15 cases. The right test bed is the **H3 learning pilot** (`python -m dclab_rnd.learning
pilot`): the intern, with its tools, on a local model, solving graded challenge packs (`groups` family), with
`learning`, `check` and `sealed` variants.

## What we already know (evidence/campaigns/learning_v1)

- LRN-003, qwen3.5:4b-16k: 5/5 homeworks finished, 1/5 passed. Fail codes: SPLIT_WRONG 4, FALSE_ALARM 3.
- LRN-006, Claude Sonnet 5.5 as the intern: 4/5 passed.
- No refused or not-offered tool calls are recorded in either result.
- The intern's first message is about 5,000 tokens (its rules and 28 tools); the solver needs a 16k context.

So the local model's failures so far are **judgment** (wrong split, false alarms), not obviously **format**. The
slide's effect may not be DCLab's main problem. Stage 0 checks this before anything is built.

## Stages

0. **Read what we have (free, about an hour, no code).** Go through the whole intern sessions of LRN-003 and the
   post-fix pilot: failed tool calls (unknown tool, bad arguments, unparsed reply), steps wasted on them, context
   used. If format errors are under 10% of steps and no homework was lost to them, **drop arm A** and run only B.
1. **Tool count (B).** JSON calls in both arms; 28 tools vs 8. The 8 are chosen from the tools Claude actually used
   in its passing runs (LRN-006/007), plus anything needed to finish: data, not taste. The validator and guards are
   unchanged. Paired: the same 20 `learning` variants in both arms.
2. **Action format (A), only if stage 0 says format matters.** The model writes one action as plain text; the
   harness parses it into the same registry call, so every guard still runs. This needs new code (the owner's go).
3. **Confirm once.** The winning setup runs once on the 20 `check` variants. The `sealed` variants are not touched.

## What we measure

| Measure | Where it comes from |
|---|---|
| Pass / fail and fail codes | the pack grader |
| Failed tool calls (unknown, bad arguments, unparsed) | the intern's session, `tool_counts` |
| Steps, minutes, prompt tokens, finished within budget | the run record |
| Guards still hold | `make refusal-test` (must stay 11/11) |

## Keep-or-drop rules (fixed before any run)

**Keep** a change only if all of these hold:

- on the 20 paired `learning` variants it wins at least 4 more tasks than it loses;
- no fail code grows by 3 or more;
- the refusal test stays 11/11;
- the `check` confirmation is not lower than today's.

Otherwise **drop** it.

**Limit:** 20 tasks can show only big effects (about 20 points or more). The slide's gap was 29 points, so an
effect that size would show. Smaller effects need more variants or repeats.

## Setup and cost

- The local model is **not installed now** (`ollama list` is empty). Pulling `qwen3.5:4b-16k` again is the owner's
  step; keep at least 2 GB of disk free.
- €0 (local only). The same sampling settings for every arm; each arm tagged.
- Time: up to 45 minutes per homework. Stage 1 is 40 homeworks, about 15 to 30 hours in the background with the
  existing loop (`scripts/learning_loop.sh`, pause and resume). Stage 2 adds 40 more.
- Capture: `learning` attempts are kept whole as training data, tagged by arm; `check` attempts leave only a grade.

## What this does not answer

- Claude: not tested. It was built around this harness, so gains there would be small.
- Other families: only `groups`. An 8-tool set fitted to one family could hurt another; check one more family
  before adopting it.

## Sources

- [The ultimate guide to multi-harness RL](https://fineenvs-multi-harness-rl.hf.space/)
- [FineEnvs RESULTS.md](https://github.com/adithya-s-k/FineEnvs/blob/main/05-multi-harness-rl/RESULTS.md)
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
