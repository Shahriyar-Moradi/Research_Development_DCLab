"""Writes SAMPLES_PREVIEW.md: one real training sample per (stage, decision), taken from out/all_stages/train.jsonl."""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["problem_understanding", "contract_check", "data_analysis", "solution_design", "connected_problem_to_plan",
         "end_to_end", "data_understanding", "leakage_audit", "feature_engineering", "model_selection", "optimization_reliability"]

rows = [json.loads(l) for l in open(HERE / "out" / "all_stages" / "train.jsonl", encoding="utf-8")]
report = json.loads((HERE / "out" / "all_stages" / "report.json").read_text())
picked = {}
for r in rows:
    k = (r["metadata"]["stage"], r["metadata"]["action"])
    picked.setdefault(k, r)

totals = Counter(r["metadata"]["stage"] for r in rows)
out = ["# Training sample preview", "",
       "One real sample per stage and decision, copied from `out/all_stages/train.jsonl`. "
       "The system prompt is the same in every sample and is shown once.", "",
       f"**Totals:** train {report['totals']['train']:,} · val {report['totals']['val']:,} · test {report['totals']['test']:,}", "",
       "**System prompt:**", "", "> " + rows[0]["messages"][0]["content"], ""]
for stage in ORDER:
    out += [f"## {stage}", ""]
    for (st, action), r in sorted(picked.items()):
        if st != stage:
            continue
        out += [f"### `{action}`  ({r['metadata']['dataset']}, {r['metadata'].get('generator', r['metadata'].get('source', ''))})", ""]
        for m in r["messages"][1:]:
            body = m["content"]
            if m["role"] == "assistant" and body.lstrip().startswith(("{", "[")):
                body = json.dumps(json.loads(body), indent=1, ensure_ascii=False)
            out += [f"**{'Input' if m['role'] == 'user' else 'Answer'}**", "", "```", body, "```", ""]
(HERE / "SAMPLES_PREVIEW.md").write_text("\n".join(out), encoding="utf-8")
print("wrote", len(picked), "examples")
