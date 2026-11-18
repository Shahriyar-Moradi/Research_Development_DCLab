# Prompt for the model that executes the plan

Paste everything below the line into a new session opened in this repository.

---

You are fixing three problems in how the DCLab R&D repository prepares training data. The repository is at
`/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D`.

The whole job is written, step by step, in this file:
`Dev_Plans/Training_Rules_Item1_2026-11-18/PLAN.md`

Do exactly this:

1. Read `PLAN.md` from top to bottom before you change anything.
2. Read the section "Rules for whoever executes this". Follow every rule in it for the whole job.
3. Do the tasks in this order: A, B, C, D. Never start a task before the one before it is finished and committed.
4. For each task:
   a. Find each place in the code by the quoted text (use `grep -n`), not by a line number.
   b. Make the change exactly as the plan writes it.
   c. Add or replace the tests exactly as the plan writes them.
   d. Run the test command the plan gives for that task.
   e. The task is finished only when the output ends with `OK` and the plan's "Done when" list is true.
   f. Commit only the files the plan names for that task, with the first line of the message the plan gives.
5. Task C draws the frozen exams. A drawn exam is final. Do the dry run (C2) first and keep its output. Do the real
   draw (C3) only if the plan's checks before it pass. Never delete or redraw an exam.
6. If a test fails, read the failure and fix the code. Never delete, skip or weaken an assertion. If the plan has an
   "If ... fails" note for that step, follow it.
7. If an existing test (one the plan does not tell you to replace) fails and you cannot fix it by correcting your own
   code, stop. Report the test's name and its failure. Do not change that test.
8. Never run a full gate (`make check-commit`, `make check-quick`, `make check-all`, `make rd-check`) and never run the
   whole test suite. Run only the files the plan names.
9. Never change a file under `dclab_rnd/prompts/` or `dclab_core/`, any tool description, or any `sealed.jsonl`.
   Never open the held-out case files the plan names.
10. When task D is finished, write the report the plan asks for in step D6.

Answer the owner in simple words and short sentences. Say plainly what failed or what you are not sure about.
