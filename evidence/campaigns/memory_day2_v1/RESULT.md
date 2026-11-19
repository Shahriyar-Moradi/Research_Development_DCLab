# Does problem memory help the Lead on its own? A day-2 check (2026-11-19)

**The question.** A person tells DCLab three facts on day 1: when the prediction is made, which column is the outcome,
and the unit or horizon (for example "one row per order, in minutes"). DCLab stores them in the problem's memory. On
day 2 the person opens a new conversation and says something short, like "Let's build the model now." Does the Lead
agent ask for those facts again? We compare two runs. In one, memory is off: the Lead does not see the stored facts.
In the other, memory is on: the Lead sees them as a note.

**How it ran.** There were 8 invented dev stories (`dclab_rnd/agent_eval/memory_day2_stories.json`): food delivery,
mobile churn, bike share, loans, clinic no-shows, store sales, machine breakdowns and insurance claims. Each story has
a small table made by code. For each story, the Lead's first day-2 turn ran once with memory off and once with it on.
The model was `qwen/qwen3.8-flash`. The bounded loop was off in both runs. One repeat. Result:
`results/MD2-001_dev.json`. The whole run is kept in `agent_runs/live_tests/20261119T165904Z-memory-day2/`. The
switch worked: the note reached every "on" request and no "off" request.

## The numbers

There are 3 facts in each of 8 stories, so 24 facts in all.

| Count | Memory off | Memory on |
|---|---|---|
| Facts re-asked, **scored by code** (the rule fixed before the run) | 2 | 4 |
| Facts re-asked, **read by hand** after the run (clear cases only) | 3 | 0 (1 borderline) |
| Question sentences in all replies | 12 | 9 |
| Facts named outside a question (code) | 11 | 10 |

Paired by person, scored by code: 0 better, 2 worse (MD2-02, MD2-05) and 6 equal. Read by hand: 3 better (MD2-03,
MD2-04 and MD2-06; MD2-03 only if its borderline "on" question is not counted), 0 worse and 5 equal.

Cost: EUR 0.0575 of a EUR 0.30 cap. That was 84 model calls, all answered and all kept whole with their thinking (84
of 84 with thinking).

## What it means, in plain words

**The code score points the wrong way, and the hand reading is the one to believe.** The code rule counts any
question that contains a fact's words. When memory is on, the Lead uses the stored facts inside its questions. For
example, it asked: "When the model says a customer will cancel within 90 days, what actually happens?" That question
does not ask for the 90 days. It builds on them. The rule still counted it as asking for the horizon again. 5 of the
6 code matches are of this kind: four in the "on" runs and one in an "off" run. The rule also missed real re-asks
with memory off. One example: "Should the number it predicts be bikes rented, bikes returned, or the net imbalance?"
That asks for the outcome again in plain words, so the keyword `bikes_rented` did not match.

**Read by hand, memory helped a little.** With memory off, the Lead asked for a stated fact again in 3 of 8 stories:
the outcome twice and the prediction moment once. With memory on, it asked for none clearly. The "on" replies also
restated the facts more often ("the moment a customer places the order ... predicts `delivery_minutes` ... one
prediction per order").

**Most questions are about something no one stored.** In both runs, nearly every question asked what the person will
do with the prediction and who reads it. Day 1 never stated that, so memory cannot remove those questions.

**Limits.** 8 stories, one repeat, one model. The hand reading was done after the run by the person who built the
check (no blind second reader). So these are counts with no interval, not proof. The scoring rule needs a fix before
it can be trusted: a question must ask *for* a fact, not just mention it. Then the check should run again on fresh
stories. This says nothing about other conversations or about production readiness.

## Checked by hand by the main session (2026-11-19)

Read reply by reply, without the scorer: without memory, three people were asked again for something they had said on
day 1 (bike rentals: which number to predict; loans: the moment of scoring; store sales: which column to predict);
with memory, none was, and the store-sales reply used "Sunday night" from memory. 3 better, 0 worse, 5 equal. The
code score is not used until its rule is fixed (it counts a question that uses a fact as a re-ask).
