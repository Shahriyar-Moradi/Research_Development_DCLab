# Contributing to dclabai

Thank you for helping. A few rules keep the studio trustworthy:

1. **Leakage first.** Never use information unavailable at the prediction moment; fit every preprocessing, selection and
   encoding step inside the training folds; read the holdout once. A change that weakens this is not merged.
2. **Rules decide, models advise.** Splits, metrics, thresholds and selection are deterministic code. A model may propose,
   never decide; every model request goes through the gateway (`dclabai/models`).
3. **Tests with every change.** Run the tests that cover your change: `python -m pytest tests/test_<area>.py`. CI runs the
   whole suite, the end-to-end flows with no model and with a mock OpenAI-compatible server, and a secret scan.
4. **Generated files are rebuilt, not edited.** The UI in `dclabai/agentic/static/app` is built from
   `dclabai/agentic/web/src` with `python -m dclabai.agentic.web.build`.

Sign your commits off (`git commit -s`, the Developer Certificate of Origin). Open an issue before a large change.
