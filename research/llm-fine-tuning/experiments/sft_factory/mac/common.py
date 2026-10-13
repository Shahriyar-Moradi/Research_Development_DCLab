"""Shared helpers for the Mac pipeline (traces, export, training, evaluation). Only the standard library here, so
every step runs in a small MLX environment without the repository's ML stack."""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FACTORY = HERE.parent
MIX = FACTORY / "out" / "mix"

# ----------------------------------------------------------------------------------------------- files

def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------------------------- chat format

def assistant_content(think: str | None, answer: str) -> str:
    """The assistant message Qwen's template expects: reasoning inside <think>, then the answer. An empty think
    block is Qwen's non-thinking form."""
    return f"<think>\n{(think or '').strip()}\n</think>\n\n{answer.strip()}"


def render(tokenizer, context: list[dict], answer: str, think: str | None, mode: str) -> tuple[str, str]:
    """(prompt, full) as strings. The prompt ends where the model starts writing (Qwen adds '<think>\\n' in thinking
    mode and an empty think block in non-thinking mode), so the loss covers exactly what the model writes."""
    thinking = mode == "think"
    prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, enable_thinking=thinking, tokenize=False)
    full = tokenizer.apply_chat_template(context + [{"role": "assistant", "content": assistant_content(think if thinking else "", answer)}],
                                         tokenize=False, enable_thinking=thinking)
    if not full.startswith(prompt):
        raise ValueError("chat template: the full conversation does not start with the generation prompt; "
                         "check the model's chat template and the think format")
    return prompt, full


def split_generation(text: str) -> tuple[str, str]:
    """A generated continuation of the thinking prompt ('<think>\\n' already given): (think, answer)."""
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    if "</think>" not in text:
        return text.strip(), ""
    think, answer = text.split("</think>", 1)
    return think.replace("<think>", "").strip(), answer.strip()


# ----------------------------------------------------------------------------------------------- matching

def parse_answer(answer: str) -> dict:
    """The parts of an answer that matching uses (mirrors build_training_mix.answer_kind)."""
    answer = answer.strip()
    if answer.startswith("DECISION:"):
        d = re.search(r"DECISION: (\S+)", answer)
        r = re.search(r"\nNEXT: (\[[^\]]+\])", answer)
        return {"kind": "decision", "decision": d.group(1) if d else None, "route": r.group(1) if r else None}
    m = re.search(r"\{.*\}", answer, re.S)
    if not m:
        return {"kind": "none"}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"kind": "invalid_json"}
    route = obj.get("route") or {}
    r = [route.get("action"), (route.get("to") or "").split(" ")[0]] if route else None
    if "decision" in obj and "checks" in obj:
        return {"kind": "json_decision", "decision": obj["decision"], "route": r}
    if "columns" in obj and "feature_set" in obj:
        return {"kind": "columns", "decisions": {c.get("column"): c.get("decision") for c in obj["columns"]}, "route": r}
    if "unknown" in obj and "open_questions" in obj:
        fields = {f: obj[f].get("value") for f in ("task", "target_column", "prediction_moment", "time_column")
                  if isinstance(obj.get(f), dict)}
        return {"kind": "contract", "fields": fields, "unknown": sorted(obj.get("unknown", [])), "route": r}
    return {"kind": "json", "obj": obj}


def matches(match: dict, answer: str) -> bool:
    got = parse_answer(answer)
    kind = match["kind"]
    if kind == "decision":
        return got.get("kind") == "decision" and got["decision"] == match["decision"] and got["route"] == match["route"]
    if kind == "json_decision":
        return got.get("kind") == "json_decision" and got["decision"] == match["decision"] and got["route"] == match["route"]
    if kind == "columns":
        return got.get("kind") == "columns" and got["decisions"] == match["decisions"] and got["route"] == match["route"]
    if kind == "contract":
        return (got.get("kind") == "contract" and got["fields"] == match["fields"] and got["unknown"] == match["unknown"]
                and got["route"] == match["route"])
    return False


# ----------------------------------------------------------------------------------------------- number check

NUM = re.compile(r"(?<![\w.\-])[+-]?\d[\d,]*(?:\.\d+)?%?(?![\w-])")
POLICY_CONSTANTS = {"0.5"}


def _shown(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def _value(tok: str) -> float | None:
    try:
        return float(tok.rstrip("%").replace(",", ""))
    except ValueError:
        return None


def numbers_ok(output: str, inputs: str, arithmetic: bool = False) -> list[str]:
    """Every number in ``output`` is in ``inputs``: as written, rounded as displayed, or as a percentage of an input
    decimal. With ``arithmetic`` (for reasoning traces) a difference, sum or ratio of two input numbers also counts,
    so a trace may compute a gap it then cites."""
    have = set(NUM.findall(inputs))
    vals = sorted({v for v in (_value(t) for t in have) if v is not None})
    if arithmetic and len(vals) <= 400:
        derived = set()
        for i, a in enumerate(vals):
            for b in vals[i:]:
                derived.update((a - b, b - a, a + b))
                if b:
                    derived.add(a / b)
                if a:
                    derived.add(b / a)
        pool = vals + sorted(derived)
    else:
        pool = vals
    rounded: dict[tuple[int, bool], set] = {}

    def known(v: float, digits: int, pct: bool) -> bool:
        key = (digits, pct)
        if key not in rounded:
            rounded[key] = {round(d * 100 if pct else d, digits) for d in pool}
        return round(v, digits) in rounded[key]

    bad = []
    for n in NUM.findall(output):
        if n in have or n.lstrip("+") in have or len(n.strip("+-")) <= 1 or n in POLICY_CONSTANTS:
            continue
        v = _value(n)
        if v is None:
            bad.append(n)
            continue
        digits = _shown(n.rstrip("%"))
        if (n.endswith("%") and known(v, digits, True)) or known(v, digits, False):
            continue
        bad.append(n)
    return bad


META = re.compile(r"(given (the |an )?answer|expected answer|gold (answer|label)|the answer (says|is given)|"
                  r"i (was|am) (told|given) the|reference answer|reviewed answer|provided answer|answer (was |is )?provided|"
                  r"answer below|correct answer (is|was) (given|provided)|the user (says|states) the answer)", re.I)


def trace_problems(think: str, context_text: str, max_words: int) -> list[str]:
    problems = []
    if not think or len(think.split()) < 15:
        problems.append("too_short")
    if len(think.split()) > max_words:
        problems.append("too_long")
    if META.search(think or ""):
        problems.append("mentions_the_answer")
    bad = numbers_ok(think or "", context_text, arithmetic=True)
    if bad:
        problems.append("numbers:" + ",".join(bad[:4]))
    return problems


# ----------------------------------------------------------------------------------------------- replay checks

TEST_PRELUDE = (
    "def _norm(x):\n"
    "    if isinstance(x, (list, tuple)):\n"
    "        return [_norm(v) for v in x]\n"
    "    if isinstance(x, dict):\n"
    "        return {k: _norm(v) for k, v in x.items()}\n"
    "    return x\n"
)


def extract_code(answer: str) -> str | None:
    m = re.search(r"```python\n(.*?)```", answer, re.S)
    return m.group(1) if m else None


def run_code_tests(code: str, tests: list[str], timeout: int = 20) -> bool:
    import subprocess
    import sys
    script = TEST_PRELUDE + "\n" + code + "\n\n" + "\n".join(tests) + "\nprint('ok')\n"
    try:
        p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return p.returncode == 0 and p.stdout.strip().endswith("ok")


def final_number(answer: str) -> float | None:
    m = re.findall(r"ANSWER:\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
    return float(m[-1].replace(",", "")) if m else None
