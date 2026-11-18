# Prompt for the model that executes the plan

Paste everything below the line into a new session opened in this repository.

---

You are making the DCLab R&D repository obey one rule of its owner: everything Claude (or any other commercial model)
wrote is kept and prepared, but no training run and no training export may use it until the owner records a rights
review. "Everything" includes the words of written test people and requests, not only labels and answers. The repository
is at `/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D`.

The whole job is written, step by step, in this file:
`Dev_Plans/Provider_Hold_All_Training_2026-11-18/PLAN.md`

It is a long plan with twelve tasks (G1 to G12). Each task says exactly which file to change, which text to find, what to
put in its place, which test to add, which command to run and what "done" means.

Do exactly this:

1. Read `PLAN.md` from top to bottom before you change anything. Read the Context: it says why, and what the owner
   gives up until the review.
2. Read "Rules for whoever executes this" and "The one allowed change to an existing test". Follow them for the whole job.
3. Do the tasks in order, G1 to G12. Never start a task before the one before it is finished. Commit only in G4, G9 and
   G12, with the files each names. Push only in G12.
4. For each task:
   a. Find each place in the code by the quoted text (use `grep -n`), not by a line number.
   b. Make the change exactly as the plan writes it. Copy the code blocks as they are.
   c. Add the tests exactly as the plan writes them, in the place the plan names.
   d. Run the command the plan gives for that task.
   e. The task is finished only when its "Done when" or its test run is true (for tests: the output ends with `OK`).
5. G1 creates one module. Copy its code exactly; every later task uses it under the name `who_wrote`, imported inside
   the function that uses it.
6. When an existing test fails after a guard, decide with "The one allowed change to an existing test": if it fails only
   because its rows are now held, add the three lines the plan gives, and nothing else, and write its name down. If it
   fails for any other reason, or still fails with those lines, stop and report it.
7. Never run a trainer for real, never run `make capture`, never run a full gate (`make check-commit`, `make check-quick`,
   `make check-all`, `make rd-check`) and never run the whole test suite. Run only the commands the plan names.
8. Never delete data, a fitted model or a stored set. Never change a file under `dclab_rnd/prompts/`, `dclab_core/` or
   `training_records/`, any tool description, the exam files or their pins. Never open a file with `sealed` in its name
   or the held-out case files the plan names.
9. G3 and G11 measure real numbers. If a number differs a lot from the plan's expectation, do not change code to make it
   fit. Report both numbers.
10. If a step says "stop and report", stop there and write what happened.
11. When G12 is finished, write the report the plan asks for in G12.7.

Answer the owner in simple words and short sentences. Say plainly what failed or what you are not sure about.
