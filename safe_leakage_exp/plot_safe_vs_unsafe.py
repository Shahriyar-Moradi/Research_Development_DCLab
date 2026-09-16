"""Side-by-side visual comparison: unsafe vs safe HyperAck results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
csv_path = ROOT / "results" / "safe_vs_unsafe_comparison.csv"
df = pd.read_csv(csv_path)
df = df.dropna(subset=["roc_auc_unsafe", "roc_auc_safe"]).copy()
df["label"] = df["exp_id"].astype(str) + " " + df["name"].str.replace("_", " ")
df = df.sort_values("roc_auc_safe")

fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

# ROC-AUC side by side
y = range(len(df))
axes[0].barh([i - 0.2 for i in y], df["roc_auc_unsafe"], height=0.4, color="#B42318", label="Unsafe")
axes[0].barh([i + 0.2 for i in y], df["roc_auc_safe"], height=0.4, color="#16845B", label="Safe")
axes[0].set_yticks(list(y))
axes[0].set_yticklabels(df["label"])
axes[0].set_xlabel("ROC-AUC")
axes[0].set_title("ROC-AUC: unsafe vs safe")
axes[0].set_xlim(0.92, 0.985)
axes[0].legend(loc="lower right")

# Recall side by side
axes[1].barh([i - 0.2 for i in y], df["recall_unsafe"], height=0.4, color="#B42318", label="Unsafe")
axes[1].barh([i + 0.2 for i in y], df["recall_safe"], height=0.4, color="#16845B", label="Safe")
axes[1].set_xlabel("Recall")
axes[1].set_title("Recall: unsafe vs safe")
axes[1].set_xlim(0.74, 0.93)
axes[1].legend(loc="lower right")

fig.suptitle("HyperAck leakage comparison (same test split)", fontsize=14, fontweight="bold")
plt.tight_layout()
out = ROOT / "results" / "safe_vs_unsafe_side_by_side.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("Wrote", out)

# Also print compact table
show = df.sort_values("roc_auc_safe", ascending=False)[
    ["exp_id", "name", "roc_auc_unsafe", "roc_auc_safe", "delta_roc_auc", "recall_unsafe", "recall_safe"]
]
print(show.to_string(index=False))
