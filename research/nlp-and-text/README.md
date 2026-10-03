# NLP and text

**Status:** planned

## Research question

When does text add signal to tabular models, and how far do simple TF-IDF models get against embeddings and fine-tuned transformers?

## Why it matters for DCLab

Support tickets, reviews and notes sit next to most business tables; DCLab should know when to use them.

## Leakage traps specific to this field

- **Label written into text:** reviews or tickets that state the outcome ("refund approved").
- **Post-outcome fields next to text:** in the clothing-reviews data, the star rating leaked the recommendation (tabular AUC 0.53 → 0.97, EXP-067).
- **Near-duplicate documents** across splits.

## First experiments

| ID | Question |
|---|---|
| NLP-001 | Seed result, done: clothing reviews, word TF-IDF + logistic regression ROC-AUC 0.946 vs 0.548 for tabular only (EXP-066..070) |
| NLP-002 | TF-IDF vs sentence embeddings vs a fine-tuned small transformer on the same reviews, with grouped splits by product |
| NLP-003 | Near-duplicate detection across train and test, and its effect on reported scores |

## Candidate datasets

Women's E-Commerce Clothing Reviews (Kaggle), Amazon reviews, AG News.

## Start the track

When work begins, scaffold the standard layout (src/, notebooks/, results/, reports/, data.md) from the template:

```bash
make new-track NAME=nlp-and-text TITLE="NLP and text" PREFIX=NLP
```

This keeps the notes on this page and adds the experiment log, prediction-contract table and result schema.
