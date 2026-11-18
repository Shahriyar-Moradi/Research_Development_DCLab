# Prompt for the model that executes the plan

Paste everything below the line into a new session opened in this repository.

---

You are fixing five small problems in the DCLab R&D repository at
`/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D`.

The whole job is written, step by step, in this file:
`Dev_Plans/Home_Live_Build_Fixes_2026-11-18/PLAN.md`

Do exactly this:

1. Read `PLAN.md` from top to bottom before you change anything.
2. Read the section "Rules for whoever executes this". Follow every rule in it for the whole job.
3. Do the tasks in this order: A, B, C, D, E, F. Never start a task before the one before it is finished.
4. For each task:
   a. Find each place in the code by the quoted text (use `grep -n`), not by a line number.
   b. Make the code change exactly as the plan writes it.
   c. Add the tests exactly as the plan writes them.
   d. Run the test command the plan gives for that task.
   e. The task is finished only when the output ends with `OK` and the plan's "Done when" list is true.
   f. Commit only the files the plan names for that task, with the first line of the message the plan gives.
5. If a test fails, read the failure and fix the code. Never delete, skip or weaken an assertion. If the plan has an
   "If ... fails" note for that task, follow it.
6. If an existing test (one you did not add) fails and you cannot fix it by correcting your own code, stop. Report the
   test's name and its failure. Do not change that test.
7. Never run a full gate (`make check-commit`, `make check-quick`, `make check-all`, `make rd-check`) and never run the
   whole test suite. Run only the files the plan names.
8. Never change a file under `dclab_rnd/prompts/`, a tool description, or the Build project route.
9. When task F is finished, write a short report: for each task, what you changed, the names of the new tests, the
   test counts of each run, anything that did not go as the plan says, and the commit ids.

Answer the owner in simple words and short sentences. Say plainly what failed or what you are not sure about.
