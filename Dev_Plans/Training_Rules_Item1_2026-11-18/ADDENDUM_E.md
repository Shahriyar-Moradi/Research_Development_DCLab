# Addendum E: the two blockers found while executing PLAN.md (do these before resuming Task C)

Written 2026-11-18 after the executing model stopped, as the plan required, at Task C2 and at the solver-data rebuild.
Tasks A (`82bdb3c7`) and B (`18528e7e`) are done, reviewed and correct; they are committed locally and not pushed yet
(Task D5 pushes them with everything else). Every rule of `PLAN.md` ("Rules for whoever executes this") still applies.

## The two blockers (both causes verified by reading the code and counting, never opening a sealed file's contents)

| # | What failed | Cause | Task |
|---|---|---|---|
| E1 | The exam dry run (C2) crashed: `JSONDecodeError: Unterminated string`. | The store reads its JSON Lines files with `str.splitlines()`. That also splits on the Unicode line separators U+2028, U+2029 and U+0085, which a stored row keeps raw inside its text. Two files hold 12 of them today: `m5_solver_llm/trajectories/development/v2/train.jsonl` (295 rows read as 302) and `dev.jsonl` (28 read as 31). The same reader is in `export_chat.py` (Task B's export would crash on that set too) and in `sets/__init__.py`. | E1 |
| E2 | `make solver-data` stopped at `FAIL solver/understand_problem ... ss_474a1060f3e3c03f72d168ba: a row of a sealed partition outside the sealed split`. | A Claude answer (origin `teacher_claude`, evaluation only) read from a workspace's **sealed** partition. The lake gives it the split `eval_only` (because it is Claude's), but the gold check requires every row from a sealed partition to be in `sealed`, and also requires every Claude answer to be in `eval_only`. Such a row can satisfy neither rule, so one row stops the whole build. | E2 |

**E2's decision (made by the reviewer; changing the core's check is a separate package):** gold leaves such a row out and
counts it. Nothing is deleted: the row stays in the lake's bronze inputs and in the store.

---

## Task E1: the store reads JSON Lines on "\n" only

### E1.1 `dclab_rnd/store/layout.py`

Right before the line `class Layout:`, add:

```python
def jsonl_lines(text: str) -> list[str]:
    """The non-empty lines of a JSON Lines text, split on "\\n" only. ``str.splitlines()`` also splits on U+2028, U+2029 and
    U+0085, which a stored row keeps raw inside its strings: a row was cut in two and its file could not be read (the exam
    draw of 2026-11-18 stopped on the build history's set)."""
    return [line for line in text.split("\n") if line.strip()]
```

### E1.2 Use it in three files (replace exactly these lines; find each by its text)

1. `dclab_rnd/store/exams.py`, function `_rows`:
   - add `jsonl_lines` to the existing import: `from .layout import Layout` becomes `from .layout import Layout, jsonl_lines`;
   - replace
     ```python
         for line in path.read_text(encoding="utf-8").splitlines():
             if line.strip():
                 yield line, json.loads(line)
     ```
     with
     ```python
         for line in jsonl_lines(path.read_text(encoding="utf-8")):
             yield line, json.loads(line)
     ```
2. `dclab_rnd/store/export_chat.py`:
   - `from .layout import Layout` becomes `from .layout import Layout, jsonl_lines`;
   - replace `[json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]` with
     `[json.loads(line) for line in jsonl_lines(source.read_text(encoding="utf-8"))]`.
3. `dclab_rnd/store/sets/__init__.py`:
   - `from ..layout import Layout` becomes `from ..layout import Layout, jsonl_lines`;
   - replace `for line in store.read(entry).decode("utf-8").splitlines():` with
     `for line in jsonl_lines(store.read(entry).decode("utf-8")):`;
   - replace `sealed_lines = sealed.read_text(encoding="utf-8").splitlines() if sealed.exists() else []` with
     `sealed_lines = jsonl_lines(sealed.read_text(encoding="utf-8")) if sealed.exists() else []`;
   - replace `sealed_count = len(sealed.read_text(encoding="utf-8").splitlines()) if sealed.exists() else 0` with
     `sealed_count = len(jsonl_lines(sealed.read_text(encoding="utf-8"))) if sealed.exists() else 0`.

Do not change any other `splitlines()` in the repository (the other readers are a separate, later fix).

### E1.3 Tests

In `tests/test_store_exams.py`, add this class (the file already imports `json`, `tempfile`, `Path`, `exams`, `Layout`):

```python
class LineBreakTest(unittest.TestCase):
    """2026-11-18: the exam draw stopped because a stored row kept a U+2028 inside its text and the reader split on it."""

    def test_a_row_with_a_unicode_line_separator_in_its_text_is_one_row(self):
        from dclab_rnd.store.layout import jsonl_lines

        text = json.dumps({"id": "a", "text": "one two three\x85four"}, ensure_ascii=False) + "\n" + json.dumps({"id": "b"}) + "\n\n"
        self.assertEqual(len(text.splitlines()), 6)  # the old reader's count: the row cut into pieces
        self.assertEqual([json.loads(line)["id"] for line in jsonl_lines(text)], ["a", "b"])
        path = Path(tempfile.mkdtemp()) / "rows.jsonl"
        path.write_text(text, encoding="utf-8")
        self.assertEqual([row["id"] for _, row in exams._rows(path)], ["a", "b"])
```

In `tests/test_store_export_chat.py`, add this method to the class `ExportTests`:

```python
    def test_a_row_with_a_unicode_line_separator_is_exported_whole(self):
        from dclab_rnd.store import export_chat
        from dclab_rnd.store.sets import write_set

        write_set(self.layout, "m4_workflow_llm", "chat/separators", [chat("u", "train").model_copy(update={"task": "a b"})],
                  version="v1", about="a row whose text keeps a U+2028")
        report = export_chat.export(self.layout, "m4_workflow_llm", "chat/separators", "v1", self.out / "separators")
        self.assertEqual(report["counts"], {"train": 1, "dev": 0})
```

### E1.4 Run

```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_store_exams.py test_store_export_chat.py test_store_versions.py test_store_sets_llm.py test_store_sets_decisions.py test_store_m5_trajectories.py test_store.py
```

**Done when:** `OK`. **Commit** (paths: `dclab_rnd/store/layout.py`, `dclab_rnd/store/exams.py`,
`dclab_rnd/store/export_chat.py`, `dclab_rnd/store/sets/__init__.py`, `tests/test_store_exams.py`,
`tests/test_store_export_chat.py`). First line: `The store reads JSON Lines on newlines only, so a row keeping a Unicode line separator stays whole`.

---

## Task E2: a Claude answer from a sealed partition is left out of gold, counted, and kept in bronze

### E2.1 `dclab_rnd/lake/__init__.py`

1. Right before the line `def reason(` (a module-level function; find it with `grep -n "^def reason(" dclab_rnd/lake/__init__.py`), add:

```python
def gold_split(origin: str, from_sealed: bool, said: Optional[str], group: str) -> Optional[str]:
    """The gold split of a workspace example, or None when gold must leave it out. An evaluation-only answer (Claude's)
    read from a sealed partition may be neither a training target nor outside the sealed split, and the gold check
    allows it in neither place: one such row stopped the whole lake build (2026-11-18). It stays in bronze; gold
    counts it as dropped."""
    if origin in L.EVAL_ONLY:
        return None if from_sealed else "eval_only"
    if from_sealed:
        return "sealed"
    return said or L.split_of(group)
```

   `Optional` is not imported at the top of that file today: change the line `from typing import Any, Callable, Iterable, Iterator`
   to `from typing import Any, Callable, Iterable, Iterator, Optional`.

2. Inside the inner function `def add(` (the one with `split = "eval_only" if origin in L.EVAL_ONLY else (...)`), find:

```python
        split = "eval_only" if origin in L.EVAL_ONLY else ("sealed" if item.get("from_sealed") else item["row"].get("split") or L.split_of(group))
```

   and replace it with:

```python
        split = gold_split(origin, bool(item.get("from_sealed")), item["row"].get("split"), group)
        if split is None:
            run.drop(item["source"], "an evaluation-only answer from a sealed partition (kept in bronze, left out of gold)")
            return
```

### E2.2 Test: add to `tests/test_training_lake.py`

Add this class at the end of the file, before `if __name__ == "__main__":` (the file already imports `json`):

```python
class GoldSplitTest(unittest.TestCase):
    """2026-11-18: a Claude answer read from a sealed partition stopped the whole lake build."""

    def test_every_origin_and_partition_gets_a_split_the_gold_check_accepts_or_is_left_out(self):
        from dclab_core import lake as L
        from dclab_rnd.lake import gold_split

        self.assertIsNone(gold_split("teacher_claude", True, "dev", "g"))  # evaluation only AND sealed: left out of gold
        self.assertEqual(gold_split("teacher_claude", False, "train", "g"), "eval_only")
        self.assertEqual(gold_split("rule_prod", True, "dev", "g"), "sealed")
        self.assertEqual(gold_split("rule_prod", False, "dev", "g"), "dev")
        self.assertIn(gold_split("rule_prod", False, None, "g"), ("train", "dev", "sealed"))  # drawn from its group
        for origin, from_sealed in (("teacher_claude", False), ("rule_prod", True), ("rule_prod", False)):
            split = gold_split(origin, from_sealed, "dev", "g")
            messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
            row = L.TrainingExample(example_id=f"ss_{origin}_{from_sealed}", family="understand_problem", messages=messages,
                                    verification=L.Verification(kind="construction", passed=True),
                                    provenance=L.Provenance(source="t", origin=origin, consent="train", group=f"g{origin}{from_sealed}",
                                                            split=split, from_sealed=from_sealed))
            self.assertEqual(L.gold_problems([row]), [], (origin, from_sealed, split))
```

If `L.TrainingExample` refuses the family `understand_problem` or the verification kind, use the values the existing
tests in that file use for a `TrainingExample` (search the file for `TrainingExample(`) and say so in the report.

### E2.3 Run, then rebuild the prepared data

```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_training_lake.py test_reasoning.py test_own_llm.py test_no_think_in_training.py
```

**Done when:** `OK`. Then, if `df -h .` shows at least 3 GB free:

```bash
DCLAB_NO_LIVE_MODELS=1 make solver-data > /tmp/solver-data.log 2>&1; echo "rc=$?"; tail -5 /tmp/solver-data.log
```

It must end with `rc=0`. Then:
- `grep -c "<think>" lake/export/solver/train.chat.jsonl` prints `0`;
- `grep -c "from_record" lake/export/solver/train.chat.jsonl` prints a number above 0;
- `grep -c "left out of gold" /tmp/solver-data.log` may print 0 or more (the count is in the build's summary; report it if shown).

If it fails on another set, do not fix it: report the last 30 lines of `/tmp/solver-data.log` and go on.

**Commit** (paths: `dclab_rnd/lake/__init__.py`, `tests/test_training_lake.py`, and
`research/llm-fine-tuning/experiments/lake/MANIFEST.json` only if the rebuild ended with `rc=0`; if it failed, restore that
file with `git checkout -- research/llm-fine-tuning/experiments/lake/MANIFEST.json` and do not commit it).
First line: `A Claude answer from a sealed partition is left out of gold and counted, so it no longer stops the lake build`.

---

## Then resume PLAN.md at Task C1

Do Task C (C1 to C6) and Task D exactly as written in `PLAN.md`. In Task D, the record
`training-rules-item1.json` gets two more findings: E1 (`source: "test"`, `severity: "behaviour"`, the exam dry run
crashed) and E2 (`source: "test"`, `severity: "behaviour"`, one row stopped the lake build); and D4's test run adds
`test_store_export_chat.py` (already listed) and nothing else.
