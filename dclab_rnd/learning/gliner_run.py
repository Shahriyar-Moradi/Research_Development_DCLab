"""GLiNER2.5-Decide on a file of rendered decisions, in the owner's ``ai12`` environment (torch, gliner2): nothing of
DCLab is imported here, so the environment needs only gliner2.

    /opt/miniconda3/envs/ai12/bin/python -I gliner_run.py IN.jsonl OUT.jsonl

Each input line is ``{"record", "text", "schema"}``; each output line keeps it whole with the model's output, its label
and confidence, the model's name and the library's version.
"""

import json
import sys
import time
from datetime import datetime, timezone

MODEL = "fastino/GLiNER2.5-Decide"


def main(src: str, dst: str, batch: int = 32) -> None:
    import gliner2
    from gliner2 import AutoExtractor

    model = AutoExtractor.from_pretrained(MODEL)
    items = [json.loads(x) for x in open(src, encoding="utf-8") if x.strip()]
    groups = {}
    for d in items:
        groups.setdefault(json.dumps(d["schema"], sort_keys=True), []).append(d)
    with open(dst, "w", encoding="utf-8") as fh:
        for chunk_items in groups.values():
            schema = chunk_items[0]["schema"]
            for i in range(0, len(chunk_items), batch):
                chunk = chunk_items[i:i + batch]
                started = time.perf_counter()
                outs = model.batch_classify_text([d["text"] for d in chunk], schema, include_confidence=True)
                seconds = (time.perf_counter() - started) / max(len(chunk), 1)
                for d, out in zip(chunk, outs):
                    ans = (out or {}).get("answer")
                    label, confidence = (ans.get("label"), ans.get("confidence")) if isinstance(ans, dict) else (ans, None)
                    fh.write(json.dumps({**d, "output": out, "label": label, "confidence": confidence, "model": MODEL,
                                         "library": f"gliner2 {getattr(gliner2, '__version__', '?')}", "seconds": round(seconds, 4),
                                         "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
