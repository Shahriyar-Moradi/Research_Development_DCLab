# Prompt for the model that executes the plan

Paste everything below the line into a new session opened in this repository.

---

You are building and running the first end-to-end comparison in the DCLab R&D repository: DCLab with no model, DCLab with
the product's model (Qwen 3.8 flash) as its agent, and that same model alone with Python in a locked box. Each gets one
table made by code and one request, and code scores them against a future set none of them sees. The repository is at
`/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D`.

The whole job is written, step by step, in this file:
`Dev_Plans/End_to_End_Comparison_2026-11-18/PLAN.md`

It has thirteen tasks (E1 to E13). Most give the exact code; a few say "read X to find Y" where you must check a name in
the repository first and use the real one.

Do exactly this:

1. Read `PLAN.md` from top to bottom before you change anything. Read the Context: it says what is measured and its limits.
2. Read "Rules for whoever executes this". Follow every rule for the whole job.
3. Do the tasks in order, E1 to E13. Never start a task before the one before it is finished. Commit only in E10 and E13.
4. For each task: find each place by its quoted text (`grep -n`); copy the code blocks as they are; add the tests as the
   plan writes them; run the command the task gives; the task is finished only when its tests end with `OK`.
5. When the plan says to check a name (a replica field, a role name, a function's parameters), check it, use the real
   name, and write down what you changed. Never change an assertion to make a test pass.
6. The locked box (E3) must never allow network or writes outside its folder. If Python cannot start inside it, add only
   the narrowest path its error names, and say so.
7. Live model calls happen only in E11 step 2, with exactly that command: two cases, `qwen/qwen3.8-flash`, EUR 0.50 for the
   whole run. Never a Claude model. Never the held-out seeds. Every other command runs with `DCLAB_NO_LIVE_MODELS=1`.
8. Write and pin the gate (E10) before the live run, and commit it then.
9. Never run a full gate (`make check-commit`, `make check-quick`, `make check-all`, `make rd-check`) or the whole suite.
   `make agent-eval` and `make harness-eval` in E4 are needed (they are free).
10. Never change `dclab_core/`, files with `sealed` in their name, the held-out case files, or `training_records/` by hand.
    Never commit anything under `agent_runs/`.
11. If a step says "stop and report", stop and write what happened.
12. When E13 is finished, write the report the plan asks for in E13.7.

Answer the owner in simple words and short sentences. Say plainly what failed or what you are not sure about.
