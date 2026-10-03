# Computer vision

**Status:** planned

## Research question

Which vision approaches (pretrained embeddings + linear head, fine-tuning, small CNNs) give the best accuracy per unit of compute on business image tasks?

## Why it matters for DCLab

Documents, product photos and inspection images appear in DCLab use cases.

## Leakage traps specific to this field

- **Near-duplicate images** across splits (burst photos, augmented copies).
- **Background shortcuts:** the model learns the scanner or watermark, not the object.
- **Patient- or product-level leakage:** several images of one entity across splits.

## First experiments

| ID | Question |
|---|---|
| CV-001 | Frozen pretrained embeddings + logistic regression vs full fine-tuning on a small labeled set |
| CV-002 | Duplicate and near-duplicate detection across splits, and its effect on reported accuracy |
| CV-003 | Shortcut test: accuracy on background-swapped images |

## Candidate datasets

CIFAR-10, Oxford-IIIT Pets, document image datasets (RVL-CDIP).

## Start the track

When work begins, scaffold the standard layout (src/, notebooks/, results/, reports/, data.md) from the template:

```bash
make new-track NAME=computer-vision TITLE="Computer vision" PREFIX=CV
```

This keeps the notes on this page and adds the experiment log, prediction-contract table and result schema.
