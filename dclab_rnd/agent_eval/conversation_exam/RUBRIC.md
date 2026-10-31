# What "understood" means in the conversation exam

The owner chose this rule on 2026-10-31, after the sealed run CEX-013. The judge's prompt
(`dclab_rnd/prompts/exam_judge.md`) already says the same thing. The labelling desk shows it to every person who
labels. All three must stay the same.

## The rule

A point counts as understood **only when the assistant's messages show it**, in one of three ways:

1. it **said the point back**, in its own words or the person's;
2. it **asked a question built on it**: a question that only makes sense once the point is grasped;
3. it **put it into its options**.

These do **not** count:

- getting the gist without showing it;
- a wrong version or half of the point;
- something the person said that the assistant never used.

You should be able to point at the sentence. If you cannot, the answer is **No**.

**Why this rule:** the person can only correct what they can see. An assistant that understood but never showed it
gives the person nothing to check.

## Examples

The point to judge: *"The decision is which members to call every Monday."* The examples are invented for this
rubric, not taken from any exam.

| The assistant wrote | Counts? | Why |
|---|---|---|
| "So every Monday you want a short list of members to call?" | Yes | said back |
| "Should the list be ready before Monday morning's calls?" | Yes | a question built on it |
| Option: "A Monday call list of the members most likely to leave" | Yes | in its options |
| "I'll analyse your members." | No | the gist at most, never shown |
| "A monthly churn report for your board" | No | a wrong version: monthly, a report, not calls |
| The person said "Mondays", and the assistant never mentioned it | No | never used |
