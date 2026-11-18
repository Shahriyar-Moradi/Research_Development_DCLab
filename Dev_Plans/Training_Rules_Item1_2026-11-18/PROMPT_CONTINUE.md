# Prompt to continue the job (paste everything below the line into the executing session)

---

Good work so far: Tasks A and B were reviewed and are correct. You were right to stop at Task C2 and not to fix the lake.

Both blockers now have exact fixes, written in:
`Dev_Plans/Training_Rules_Item1_2026-11-18/ADDENDUM_E.md`

Do exactly this:

1. Read `ADDENDUM_E.md` from top to bottom.
2. Do Task E1, then Task E2, exactly as written: the code changes, the tests, the test runs, the commits.
3. Then resume `PLAN.md` at Task C1 and do Task C and Task D exactly as written there, with the two extra findings
   the addendum names for the record.
4. All the rules of `PLAN.md` still apply: no full gate, no change to `dclab_core/`, `dclab_rnd/prompts/`, tool
   descriptions or any sealed file; never weaken a test; stop and report if an existing test fails and you cannot fix
   it in your own code. The exam draw is final: dry run first, real draw only after the checks.
5. Do not push before Task D5. Task D5 pushes everything, including the commits of Tasks A and B.
6. End with the report of `PLAN.md` step D6, plus: what E1 and E2 changed and the test counts.
