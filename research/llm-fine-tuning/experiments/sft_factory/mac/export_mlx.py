"""
export_mlx.py
=============

Joins the training items, their <think> traces and the verified replay answers into the final files that
train_mlx.py and eval_mlx.py read:

    out/mix/items_{split}.jsonl + out/mix/traces.jsonl + out/mix/replay_items.jsonl
        ──►  out/mix/final_{train,val,test}.jsonl + out/mix/final_report.json

* an item with a kept trace trains in thinking mode with that reasoning and the GOLD answer;
* an item without a trace (a no_think item, or one whose traces all failed) trains in non-thinking mode;
* replay answers: 90% train, 10% val (by task), so validation also shows whether general skill drifts.

Runs anywhere (standard library only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from common import MIX, read_jsonl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", type=Path, default=MIX)
    ap.add_argument("--replay-weight", type=int, default=1)
    args = ap.parse_args()
    traces = {r["id"]: r for r in read_jsonl(args.mix / "traces.jsonl")}
    replay = [r for r in read_jsonl(args.mix / "replay_items.jsonl") if not r.get("failed")]
    report = {}
    for split in ("train", "val", "test"):
        rows = []
        for it in read_jsonl(args.mix / f"items_{split}.jsonl"):
            tr = traces.get(it["id"])
            think = tr["think"] if tr and tr.get("think") else None
            mode = "think" if (it["think_mode"] == "think" and think) else "no_think"
            rows.append({"id": it["id"], "context": it["context"], "answer": it["answer"], "think": think if mode == "think" else None,
                         "mode": mode, "weight": it.get("weight", 1), "stage": it["turn_stage"], "step": it.get("step"),
                         "match": it["match"], "source": "workflow",
                         "trace_method": (tr or {}).get("method", "none")})
        for r in replay:
            in_val = int(hashlib.sha256(r["task"].encode()).hexdigest()[:4], 16) % 10 == 0
            if (split == "val") == in_val and split != "test":
                rows.append({"id": r["id"], "context": r["context"], "answer": r["answer"], "think": r["think"], "mode": "think",
                             "weight": args.replay_weight if split == "train" else 1, "stage": f"replay_{r['kind']}",
                             "step": None, "match": {"kind": "replay"}, "source": "replay", "trace_method": "self", "task": r["task"]})
        with open(args.mix / f"final_{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        report[split] = {"rows": len(rows), "weighted": sum(r["weight"] for r in rows),
                         "modes": dict(Counter(r["mode"] for r in rows)),
                         "trace_methods": dict(Counter(r["trace_method"] for r in rows)),
                         "sources": dict(Counter(r["source"] for r in rows))}
    if traces:
        report["trace_summary"] = dict(Counter(t["method"] for t in traces.values()))
    (args.mix / "final_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
