# Prompt for the model that executes the plan

Paste everything below the line into a new session opened in this repository.

---

You are closing two gaps in how the DCLab R&D repository prepares training data: case files do not say which model
wrote them, and an old set of Home conversations names no writer. Until they do, the training export cannot hold back
Claude's answers. The repository is at `/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D`.

The whole job is written, step by step, in this file:
`Dev_Plans/Home_Answerer_Followups_2026-11-18/PLAN.md`

Do exactly this:

1. Read `PLAN.md` from top to bottom before you change anything.
2. Read the section "Rules for whoever executes this". Follow every rule in it for the whole job.
3. Do the tasks in this order: H1, H2, H3, H4, H5, H6. Never start a task before the one before it is finished.
   There is one commit, at the end (H6).
4. For each task:
   a. Find each place in the code by the quoted text (use `grep -n`), not by a line number.
   b. Make the change exactly as the plan writes it. Copy the code blocks as they are.
   c. Add the tests exactly as the plan writes them, in the place the plan names.
   d. Run the command the plan gives for that task.
   e. The task is finished only when its "Done when" is true (for tests: the output ends with `OK`).
5. The only existing test you may change is the spy in H2.3. Change nothing else in any existing test.
6. H5 rebuilds the store's sets. It can take about 10 minutes and a lot of memory. Check the disk and memory first.
   Run the rebuild in the background and wait for it. Run nothing else heavy beside it. If it fails, report its last
   30 lines; do not try to fix the builder.
7. H5 compares real numbers with the plan's expected numbers. If a number is off by more than 10%, do not change code
   to make it fit. Report both numbers.
8. If a test fails, read the failure and fix the code. Never delete, skip or weaken an assertion. If the plan has an
   "If ... fails" note for that step, follow it.
9. If an existing test fails and you cannot fix it by correcting your own code, stop. Report the test's name and its
   failure. Do not change that test.
10. Never run a full gate (`make check-commit`, `make check-quick`, `make check-all`, `make rd-check`) and never run the
    whole test suite. Run only the files the plan names.
11. Never change a file under `dclab_rnd/prompts/` or `dclab_core/`, any tool description, the exam files or their pins,
    or any stored set file by hand. Never open any `sealed.jsonl` or the held-out case files the plan names. Never commit
    inside `training_records/store`.
12. When H6 is finished, write the report the plan asks for in step H6.6.

Answer the owner in simple words and short sentences. Say plainly what failed or what you are not sure about.
