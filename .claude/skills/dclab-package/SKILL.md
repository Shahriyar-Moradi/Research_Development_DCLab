---
name: dclab-package
description: Build one package of the DCLab plan (docs/guides/PRODUCT_BUILD_PLAYBOOK.md or AGENTIC_FOUNDATION_PLAN.md) end to end - failing test, implementation, make check-all, adversarial review, commit. Use when asked to build, continue or finish a numbered package such as A1.2, A2.1 or 9.3.
---
# Build one DCLab package

Package: $ARGUMENTS

1. **Read** the package (its "Delivers", "Done when" and prompt), the principles of its plan, and CLAUDE.md.
   Investigate unfamiliar code with an Explore subagent; keep only conclusions in the main context.
2. **Check first.** Run the package's "Done when" checks. If they already pass, report that and stop.
3. **Write the test first** for the main behaviour and one failure path; run it and see it fail.
4. **Implement** the smallest change that meets the package. Reuse what exists. No UI change unless the package says so.
5. **Verify** with `make check-all` (rd-check on files, the suite on PostgreSQL, the end-to-end flows). Read the
   output; fix the cause of any failure, never weaken a test.
6. **Review**: run the `dclab-reviewer` subagent on the diff with the package id. Fix correctness findings,
   re-run `make check-all`. Ignore style-only findings.
7. **Record**: mark the package done in its plan's status table, with what was left open.
8. **Commit** with a message that says why. Do not push unless the owner said so in this session.

## Gotchas
- Never `git stash` in this repository: the owner keeps uncommitted work in the tree (docs/recap and others).
- `dclab_rnd/agentic/static/app/**` is generated: edit `dclab_rnd/agentic/web/src` and run `make web`.
- Tests that need PostgreSQL use the `dclab_test` database and empty it; never point them at `dclab_dev`.
- With `DCLAB_DATABASE_URL` set the whole product uses PostgreSQL; schema changes need an Alembic migration in
  `dclab_rnd/storage/migrations/versions` and a matching change to `dclab_rnd/storage/models.py`.
- Model requests only through `dclab_rnd.models` (the gateway); tests use scripted transports.
- The agent environment (`.venv-agent`) lacks pandas; its test files skip there. Use `.venv` for the suite.
