# Prompt for the model that executes the plan

Paste everything below the line into a new session opened in this repository.

---

You are fixing one gap in how the DCLab R&D repository prepares training data: Home conversations do not say which
model wrote them, so the training export cannot hold back a provider's (Claude's or another hosted model's) answers.
The repository is at `/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D`.

The whole job is written, step by step, in this file:
`Dev_Plans/Home_Conversation_Answerer_2026-11-18/PLAN.md`

Do exactly this:

1. Read `PLAN.md` from top to bottom before you change anything. Skip its "Delivery" line: that part is done.
2. Read the section "Rules for whoever executes this". Follow every rule in it for the whole job.
3. Do the tasks in this order: G1, G2, G3, G4, G5, G6. Never start a task before the one before it is finished.
   There is one commit, at the end (G6).
4. For each task:
   a. Find each place in the code by the quoted text (use `grep -n`), not by a line number.
   b. Make the change exactly as the plan writes it. Copy the code blocks as they are.
   c. Add the tests exactly as the plan writes them.
   d. Run the command the plan gives for that task.
   e. The task is finished only when its "Done when" is true (for tests: the output ends with `OK`).
5. Never write the word "rules" into `check["model"]`. Who wrote a row goes into `check["answerer"]` only.
6. G5 rebuilds the store's sets. It can take a long time and use a lot of memory. Check the disk and memory first, as
   the plan says. Run the rebuild in the background and wait for it. Run nothing else heavy beside it. If it fails,
   report its last 30 lines; do not try to fix the builder.
7. G5 compares real numbers with the plan's expected numbers. If a number is off by more than 10%, do not change code
   to make it fit. Report both numbers.
8. If a test fails, read the failure and fix the code. Never delete, skip or weaken an assertion. If the plan has an
   "If ... fails" note for that step, follow it.
9. If an existing test fails and you cannot fix it by correcting your own code, stop. Report the test's name and its
   failure. Do not change that test.
10. Never run a full gate (`make check-commit`, `make check-quick`, `make check-all`, `make rd-check`) and never run the
    whole test suite. Run only the files the plan names.
11. Never change a file under `dclab_rnd/prompts/` or `dclab_core/`, any tool description, the exam files or their
    pins. Never open any `sealed.jsonl` or the held-out case files the plan names. Never commit inside
    `training_records/store`.
12. When G6 is finished, write the report the plan asks for in step G6.6.

Answer the owner in simple words and short sentences. Say plainly what failed or what you are not sure about.
