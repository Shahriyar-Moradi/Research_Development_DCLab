# DCLab field notes: what we learned building models the careful way

*For readers with no machine-learning background. Every number here comes from a file in this repository, and the file is named next to it.*

---

## The question behind all of this

A company wants a computer to make a prediction before something happens. Will this delivery order be accepted? Will this customer leave? Will this client say yes to the bank's offer? You give the computer past examples where you already know the answer, it finds patterns, and it guesses on new cases.

The hard part is not getting a good-looking score. The hard part is getting a score that is **true**: one that holds when the model meets tomorrow's cases. Most mistakes in this field produce a beautiful score that is a lie. DCLab's R&D exists to learn, with measurements, which habits produce true scores.

We studied this on 10 public datasets in a 50-experiment campaign, on 4 more datasets covering harder kinds of problems (20 more experiments), on DCLab's own delivery data (83 experiments), on a customer-churn dataset, and in a set of experiments that measure common mistakes directly.

---

## Lesson 1: the biggest danger is cheating by accident

Imagine predicting whether a bank client will subscribe to an offer, before calling them. One column in the data is **call duration**. It is the strongest predictor by far: long calls mean interested clients.

But you only know how long a call lasted *after the call ends*. At the moment you decide whom to call, that number does not exist yet. A model that uses it is reading the answer sheet.

What it did to the score (`campaigns/model_building_50_v1/results/EXP-007_bank_marketing_leakage_audit.json`):

| | Score (ROC-AUC, 0.5 = guessing, 1.0 = perfect) |
|---|---:|
| Without call duration (honest) | 0.713 |
| With call duration | 0.897 |
| **Illusion** | **+0.184** |

This is called **leakage**: information that would not be available at the real decision moment leaks into training. The same pattern appeared in an online shop (page values known only after the visit, +0.197) and in DCLab's own delivery data, where the final fare is only known after the order is settled (0.980 with it, 0.945 honestly).

**The rule we derived:** before looking at any score, write down the exact moment the prediction is made, and allow only information that exists at that moment. A suspiciously large improvement is a reason to look for cheating, not a reason to celebrate.

**A warning we measured:** an automatic scanner cannot fully do this for you. Running our leakage scanner "blind", without telling it the answers, it found 11 of 18 known cheating columns across 7 datasets (`campaigns/agent_verification_v1/VERIFICATION_REPORT.md`). The ones it missed look like ordinary numbers. Only a person who knows *when* each value is written can catch them.

---

## Lesson 2: some common habits lie a lot, and some barely matter

We ran six common notebook habits the wrong way and the right way on identical data and measured how much the score changed (`campaigns/pitfalls_v1/PITFALLS_REPORT.md`):

| Habit | How much the score lied |
|---|---|
| Duplicating rare cases *before* setting aside test data | up to **+0.30** |
| Turning an ID column into a "how often did this ID say yes" feature using all rows | **+0.18** (perfect 1.00 instead of an honest 0.82) |
| Choosing which columns to keep using all rows, on data with many columns and few rows | **+0.35** on pure noise; almost nothing on ordinary data |
| Picking the best of 30 settings by their test score | about **+0.007** on average, up to +0.018 |
| Scaling numbers using all rows before setting aside test data | about **0.0001**, practically nothing |
| Testing on a random sample instead of the most recent period | **none** on our delivery data in that period |

Two lessons hide in this table. First, "data leakage" is not one thing; some forms are catastrophic and some are harmless, and you can only know by measuring. Second, the last row surprised us: we expected the time-based test to look worse, and it did not. Measuring beats assuming.

---

## Lesson 3: there is no single best algorithm

People often ask, "which model is best?" Across our datasets the answer kept changing (`campaigns/model_building_50_v1/CAMPAIGN_REPORT.md`, `research/churn-prediction/churn_exp/CHURN_BENCHMARK.md`):

- In the controlled 10-dataset campaign, **Extra Trees** was the best candidate on 7 datasets, but LightGBM, histogram boosting and logistic regression each won one.
- On customer churn, the simplest model, **logistic regression**, beat every boosted model (0.850).
- On DCLab's delivery data, an **ensemble** of three models was best (0.9455).

**The rule:** try a simple baseline and a few different kinds of model on exactly the same data splits, then keep the simplest one that is not clearly worse.

---

## Lesson 4: more features usually do not help

A common instinct is to invent many extra columns (ratios, combinations, logs). In the campaign, the plain original columns were the best choice on **7 of 10 datasets**. Where engineered features helped, the gain was small: +0.004 to +0.007. On DCLab's delivery data, growing from 25 to 46 features made the best models *worse* (about 0.945 to 0.937).

**The rule:** start with the raw columns, add features one rung at a time, and keep the smallest set that is within noise of the best.

---

## Lesson 5: tuning helps a little; honesty about uncertainty helps more

After choosing a model, adjusting its settings ("tuning") improved cross-validated scores by at most +0.024, and on four datasets not at all. Meanwhile, the uncertainty of a single honest test score was often larger than that gain: on bank marketing the final score was 0.772 with a 95% range of **0.713 to 0.829**.

**The rule:** tune last, accept a change only if it beats a pre-agreed margin on the same splits, and always report the range, not just one number.

---

## Lesson 6: the AI reviewer also needs reviewing

We asked a language model to critique every experiment. Many critiques were sharp. But when we recomputed each experiment's own selection rule from its numbers, 10 of the critic's 157 objections turned out to be **wrong**. Most came from one confusion: a feature recipe was literally *named* "selected", and the critic took that name for the recipe the rule had selected.

**The rule:** language models propose and question; deterministic code computes and checks. Before an AI's opinion becomes memory or training data, check it against the numbers (`dclab_rnd/critic_gate.py`).

---

## Lesson 7: different problems need different scorecards

The first campaign only asked yes/no questions. We then ran the same five-step workflow on four harder kinds of problem (`campaigns/expansion_v1/CAMPAIGN_REPORT.md`):

| Problem | What we predicted | The trap | Honest result |
|---|---|---|---|
| **Rare events** (credit-card fraud, 0.17% fraud) | Is this transaction fraud? | Saying "never fraud" is already **99.87% accurate**, so accuracy is useless | The model found frauds well (average precision 0.81 vs 0.001 for guessing). Tuned to keep false alarms rare, it caught 52 of 75 frauds with 54 alerts |
| **Many classes** (letter recognition, 26 letters) | Which letter is this? | Some test rows were exact copies of training rows and were always right | Macro-F1 0.975; repeated shapes inflate apparent accuracy |
| **Forecasting over time** (daily bike rentals) | How many bikes will be rented? | Two columns, casual + registered riders, add up exactly to the answer | Testing on random days looked **twice as good** (error 430 bikes) as testing on future days (833). The honest forecast beat "same as two days ago" by 476 bikes a day |
| **Text + numbers** (clothing reviews) | Will the reviewer recommend the item? | The star rating is written at the same moment as the recommendation | Without the text, the rating alone pushed the score from 0.53 to 0.97. With the rating removed, the review text did the real work: 0.946, versus 0.548 for the numbers alone |

Two lessons from this round. First, pick the scorecard from the cost of mistakes: average precision for rare events, macro-F1 when every class matters, average error in real units for forecasts. Second, the forecasting result is the mirror image of Lesson 2's delivery result: there, testing on the future changed nothing; here, it halved the apparent quality. You only learn which world you are in by measuring both.

---

## How DCLab turns these lessons into a product

1. **A searchable memory.** Every rule, experiment, cheating case and measured mistake is stored as a self-contained record with its source file, so any answer can show its proof (`knowledge/rag/records.jsonl`).
2. **A notebook reviewer.** It reads a data scientist's notebook and pins notes beside the cells that contain these mistakes, each with "Show proof". On a demo notebook that reports 96.7% accuracy, it raises three high-severity problems. The careful version of the same analysis scores an honest 0.80 (`docs/guides/NOTEBOOK_COPILOT.md`).
3. **Tools for an AI agent.** The agent looks things up and checks them instead of guessing (`dclab_rnd/tools.py`).
4. **Training material for a future small model.** Built so that every example contains the evidence it depends on (`docs/guides/SFT_DATA_GUIDE.md`).

## Twelve rules to take away

1. Write down the prediction moment first. Leakage is defined by time, not by column names.
2. Split your data before any preprocessing, selection, resampling or encoding.
3. Never compute anything from the target using rows you will test on.
4. A suspicious column name or a surprisingly strong column starts a review; it does not prove cheating.
5. A big jump in score is a reason to audit, not to celebrate.
6. Start with raw features; add features one rung at a time.
7. Compare several kinds of model on identical splits; keep the simplest that is not clearly worse.
8. Tune last, with a pre-agreed margin, on the same splits.
9. Use the test set once, at the end.
10. Report a range, and metrics that match the real cost of mistakes, not accuracy alone.
11. If the future matters, test on the future, and measure the gap rather than assume it.
12. Let AI propose and question; let code compute and verify; let people approve.
