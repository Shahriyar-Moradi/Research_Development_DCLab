# Answer to the C5 stop, and the exact fix (paste everything below the line into the executing session)

---

You were right to stop. Option 1 is approved: patch `exams.PINS` in `FlowTest.setUp()`. This is the owner's approval
to change that one existing test class, and only in the way written here.

Why it is right: `exams.sit`, `exams.gate` and `exams.accept` call `load(layout, ...)` without a `pins` argument, so they
read the repository's real `dclab_rnd/store/exam_pins.json` even for a temporary store. Before the draw that file did not
exist, so the tests passed by luck. A test must never depend on the repository's real pins. Do NOT change
`dclab_rnd/store/exams.py`: its pin checks stay strict. (I checked: no other test file calls these functions.)

Do exactly this:

1. In `tests/test_store_exams.py`, add `from unittest import mock` right after the line `import unittest`.

2. In `class FlowTest`, method `setUp`, find:
   ```python
           self.pins = Path(tempfile.mkdtemp()) / "pins.json"
   ```
   and add these three lines right after it (same indentation):
   ```python
           # every function of the flow reads this test's own pins, never the repository's real exam_pins.json (drawn 2026-11-18)
           patcher = mock.patch.object(exams, "PINS", self.pins)
           patcher.start()
           self.addCleanup(patcher.stop)
   ```
   Change nothing else in `FlowTest`. Keep your `DrawnExamsTest` as it is.

3. Run:
   ```bash
   DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_store_exams.py test_store_versions.py
   ```
   Done when it ends with `OK` and `DrawnExamsTest` ran (not skipped).

4. Do Task C6 of `PLAN.md` at once, before anything else: the exam pins and the exam files are not in git yet, and they
   cannot be drawn again.
   - This repository: commit `dclab_rnd/store/exam_pins.json` and `tests/test_store_exams.py` with the first line
     `The frozen exams are drawn and pinned, so no exam row can enter training`, and in the body the counts per type
     (m1_students 1,153; m2_workflow_decisions 324; m3_ml_decisions 2,415; m4_workflow_llm 807; m5_solver_llm 675;
     m6_stepwise_decisions 483; m7_judges 190; m7_routing and m8_retrieval not drawn: no sealed row with an answer to expect),
     and one sentence that FlowTest now uses its own pins.
   - The store's repository: the three commands of C6 step 2 (`git -C training_records/store ...`). If the push fails,
     do not force it; report the message and go on.

5. Then do Task D of `PLAN.md` (record, `make sft-dev`, freshness check, the final test run, commit, push, report).
   The record gets these findings in addition to A, B and C: E1 and E2 (as `ADDENDUM_E.md` says), and one more:
   `C5` (`source: "test"`, `severity: "latent"`): three exam functions read the repository's real pins file whatever
   store they are given, so two tests passed only because no exam was drawn yet; fixed by isolating the test's pins;
   the functions themselves still take no pins argument (left for a later fix).

6. In your final report, add: the C5 fix, the test counts, and the commit ids in both repositories.
